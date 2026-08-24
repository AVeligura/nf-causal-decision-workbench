from __future__ import annotations

import json
import uuid

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from domain import AnalysisResult, EvidenceBundle, EvidenceItem, GeneratorConfig
from engine.evidence import reference_evidence
from engine.graphs import reference_graphs
from runtime import DataImporter
from visualization import (
    plot_alpha_cascade,
    plot_cate_profiles,
    plot_graph_specific_forest,
    plot_graphs,
    plot_overlap,
    plot_scores,
    plot_stability_map,
    plot_value_regret,
)

from .state import AppState


class PlotPanel(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("plotPanel")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.canvas: FigureCanvasQTAgg | None = None

    def set_figure(self, figure) -> None:
        if self.canvas is not None:
            self.layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
        self.canvas = FigureCanvasQTAgg(figure)
        self.layout.addWidget(self.canvas)
        self.canvas.draw_idle()


class NumericControl(QWidget):
    valueChanged = Signal(float)

    def __init__(
        self, minimum: float, maximum: float, value: float, decimals: int = 2, integer: bool = False
    ):
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.integer = integer
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        if integer:
            self.spin = QSpinBox()
            self.spin.setRange(int(minimum), int(maximum))
            self.spin.setValue(int(value))
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(decimals)
            self.spin.setRange(minimum, maximum)
            self.spin.setSingleStep(10 ** (-decimals))
            self.spin.setValue(value)
        self.spin.setMinimumWidth(92)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self._sync_slider(float(value))
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)

    def _sync_slider(self, value: float) -> None:
        position = round(1000 * (value - self.minimum) / (self.maximum - self.minimum))
        self.slider.blockSignals(True)
        self.slider.setValue(position)
        self.slider.blockSignals(False)

    def _slider_changed(self, position: int) -> None:
        value = self.minimum + (self.maximum - self.minimum) * position / 1000
        if self.integer:
            value = round(value)
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.valueChanged.emit(float(value))

    def _spin_changed(self, value: float) -> None:
        self._sync_slider(float(value))
        self.valueChanged.emit(float(value))

    def set_value(self, value: float) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self._sync_slider(value)

    def set_locked(self, locked: bool) -> None:
        self.slider.setEnabled(not locked)
        self.spin.setEnabled(not locked)


class ProjectDataWorkspace(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.controls: dict[str, NumericControl] = {}
        root = QVBoxLayout(self)
        title = QLabel("Проект и данные")
        title.setObjectName("workspaceTitle")
        root.addWidget(title)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self.tabs.addTab(self._causal_query_tab(), "Причинный вопрос")
        self.tabs.addTab(self._source_parameters_tab(), "Источник и параметры")
        self.tabs.addTab(self._diagnostics_tab(), "Данные и диагностика")
        state.config_changed.connect(self.refresh_config)
        state.data_changed.connect(self.refresh_data)

    def _causal_query_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        self.outcome = QComboBox()
        self.outcome.addItems(
            ["Коэффициент покрытия (Y_CR)", "Операционный денежный поток (Y_CFO)"]
        )
        self.estimand = QComboBox()
        self.estimand.addItems(["ATE", "CATE", "ATT"])
        self.horizon = QSpinBox()
        self.horizon.setRange(1, 12)
        self.horizon.setValue(4)
        self.version = QComboBox()
        self.version.addItems(["Полный пакет", "Частичный пакет (отдельно)", "Базовый режим"])
        population = QLineEdit("Средние промышленные предприятия с повышенной долговой нагрузкой")
        population.setReadOnly(True)
        layout.addRow("Исход", self.outcome)
        layout.addRow("Estimand", self.estimand)
        layout.addRow("Горизонт, кварталы", self.horizon)
        layout.addRow("Версия воздействия", self.version)
        layout.addRow("Целевая совокупность", population)
        note = QLabel(
            "Y_CR и Y_CFO рассматриваются как два отдельных причинных вопроса. "
            "Частичное исполнение не объединяется с полным пакетом."
        )
        note.setWordWrap(True)
        note.setObjectName("noteBox")
        layout.addRow(note)
        return tab

    def _source_parameters_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        header = QHBoxLayout()
        self.mode_label = QLabel("Эталонный профиль · защищён")
        self.mode_label.setObjectName("statusChip")
        self.clone_button = QPushButton("Создать лабораторную копию")
        self.clone_button.clicked.connect(self.state.clone_reference)
        header.addWidget(self.mode_label)
        header.addStretch(1)
        header.addWidget(self.clone_button)
        outer.addLayout(header)
        scenario_row = QHBoxLayout()
        scenario_row.addWidget(QLabel("Сценарий"))
        self.scenario = QComboBox()
        self.scenario.addItem("S1 · Опорный", "reference")
        self.scenario.addItem("S2 · Конфликт свидетельств", "evidence_conflict")
        self.scenario.addItem("S3 · Слабое overlap", "weak_overlap")
        self.scenario.addItem("S4 · Смешение версий", "version_mixing")
        self.scenario.addItem("S5 · Информативные потери", "informative_loss")
        self.scenario.addItem("S6 · Истинная структура вне Γ", "outside_gamma")
        self.scenario.currentIndexChanged.connect(
            lambda _: self.state.set_config_value("scenario", self.scenario.currentData())
        )
        scenario_row.addWidget(self.scenario, 1)
        self.reset_preset = QPushButton("Вернуть параметры пресета")
        self.reset_preset.clicked.connect(self.state.reset_scenario_preset)
        scenario_row.addWidget(self.reset_preset)
        outer.addLayout(scenario_row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        form = QFormLayout(body)
        specifications = [
            ("sample_size", "Размер выборки", 300, 5000, 1500, 0, True),
            ("effect_scale", "Масштаб эффекта", 0.5, 1.5, 1.0, 2, False),
            ("heterogeneity", "Гетерогенность", 0.0, 1.5, 0.8, 2, False),
            ("nonlinearity", "Нелинейность", 0.0, 1.5, 0.8, 2, False),
            ("noise_scale", "Уровень шума", 0.5, 2.0, 1.0, 2, False),
            ("assignment_strength", "Сила назначения", 0.5, 2.5, 1.2, 2, False),
            ("propensity_lower", "Propensity: нижняя граница", 0.03, 0.30, 0.15, 2, False),
            ("propensity_upper", "Propensity: верхняя граница", 0.70, 0.97, 0.85, 2, False),
            ("partial_share", "Доля частичного исполнения", 0.0, 0.40, 0.10, 2, False),
            ("refusal_share", "Доля отказов", 0.0, 0.30, 0.0, 2, False),
            ("missing_share", "Доля пропусков", 0.0, 0.35, 0.05, 2, False),
            ("hidden_confounding", "Сила скрытого смешения", 0.0, 1.5, 0.0, 2, False),
            ("evidence_reliability", "Надёжность свидетельств", 0.50, 0.99, 0.90, 2, False),
            ("evidence_conflict", "Степень конфликта", 0.0, 1.0, 0.0, 2, False),
            ("pilot_share", "Доля пилотной выборки", 0.10, 0.40, 0.20, 2, False),
            ("value_multiplier", "Множитель ценностной модели", 0.5, 1.5, 1.0, 2, False),
            ("sales_loss_scale", "Потери продаж при полном охвате", 0.0, 0.03, 0.006, 3, False),
            ("zombie_risk_scale", "Добавочный zombie-risk", 0.0, 0.10, 0.020, 3, False),
            ("program_cost_a1", "Стоимость пилота a1", 0.0, 0.03, 0.004, 3, False),
            ("program_cost_a2", "Стоимость внедрения a2", 0.0, 0.06, 0.008, 3, False),
            ("cr_weight", "Вес ΔCR", 0.0, 0.50, 0.05, 3, False),
            ("financing_weight", "Вес снижения стоимости финансирования", 0.0, 3.0, 1.0, 2, False),
            ("arrears_weight", "Вес сокращения просроченной задолженности", 0.0, 2.0, 0.60, 2, False),
            ("sales_loss_weight", "Штраф потерь продаж", 0.0, 2.0, 0.25, 2, False),
            ("zombie_weight", "Штраф zombie-risk", 0.0, 1.0, 0.03, 3, False),
            ("pilot_information_cost", "Стоимость пилотной информации", 0.0, 0.02, 0.001, 3, False),
            ("conditional_regret_threshold", "Допуск максимального regret", 0.0, 0.05, 0.005, 3, False),
        ]
        for name, label, minimum, maximum, value, decimals, integer in specifications:
            control = NumericControl(minimum, maximum, value, decimals, integer)
            control.valueChanged.connect(
                lambda changed, key=name: self.state.set_config_value(
                    key, int(changed) if key == "sample_size" else changed
                )
            )
            control.set_locked(True)
            self.controls[name] = control
            form.addRow(label, control)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        source = QLabel(
            "Основа: UCI Polish Companies Bankruptcy · DOI 10.24432/C5F600 · CC BY 4.0 · "
            "локальная копия проверена по SHA-256."
        )
        source.setWordWrap(True)
        source.setObjectName("sourceBox")
        outer.addWidget(source)
        return tab

    def _diagnostics_tab(self) -> QWidget:
        tab = QWidget()
        layout = QSplitter(Qt.Orientation.Vertical, tab)
        outer = QVBoxLayout(tab)
        import_row = QHBoxLayout()
        self.import_status = QLabel("Можно загрузить CSV, XLSX или Parquet и сопоставить схему")
        import_button = QPushButton("Импортировать данные…")
        import_button.clicked.connect(self._import_data)
        import_row.addWidget(self.import_status, 1)
        import_row.addWidget(import_button)
        outer.addLayout(import_row)
        outer.addWidget(layout)
        self.preview = QTableWidget(0, 0)
        self.preview.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preview.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.overlap_plot = PlotPanel()
        layout.addWidget(self.preview)
        layout.addWidget(self.overlap_plot)
        layout.setSizes([330, 350])
        return tab

    def _import_data(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Импорт данных",
            "",
            "Табличные данные (*.csv *.xlsx *.xls *.parquet *.pq)",
        )
        if not path:
            return
        try:
            frame = DataImporter.read(path)
            mapping = self._mapping_dialog(frame)
            if mapping is None:
                return
            mapped = DataImporter.map_columns(frame, mapping)
            valid, warnings = DataImporter.validate(mapped)
            if not valid:
                QMessageBox.warning(self, "Проверка схемы", "\n".join(warnings))
                return
            dataset_spec = DataImporter.build_dataset_spec(path, mapped, mapping)
            self.state.load_imported_data(mapped, dataset_spec)
            self.import_status.setText(f"Импортировано: {len(mapped)} строк · {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка импорта", str(exc))

    def _mapping_dialog(self, frame: pd.DataFrame) -> dict[str, str] | None:
        suggestions = DataImporter.suggest_mapping(frame)
        dialog = QDialog(self)
        dialog.setWindowTitle("Сопоставление столбцов")
        dialog.resize(540, 620)
        layout = QVBoxLayout(dialog)
        note = QLabel(
            "Каждой переменной модели сопоставьте исходный столбец. "
            "Совпадающие имена выбраны автоматически."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        selectors: dict[str, QComboBox] = {}
        for target in DataImporter.required_columns():
            selector = QComboBox()
            selector.addItem("— не сопоставлено —", "")
            for column in frame.columns:
                selector.addItem(str(column), str(column))
            suggested = suggestions.get(target, "")
            selector.setCurrentIndex(max(0, selector.findData(suggested)))
            selectors[target] = selector
            form.addRow(target, selector)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return {target: selector.currentData() for target, selector in selectors.items()}

    def refresh_config(self, config: GeneratorConfig) -> None:
        if self.state.reference_locked:
            mode_text = "Эталонный профиль · защищён"
        elif config.customized:
            mode_text = f"Пользовательский вариант {config.preset_scenario or config.scenario}"
        else:
            mode_text = f"Лабораторный пресет {config.scenario}"
        self.mode_label.setText(mode_text)
        for name, control in self.controls.items():
            control.set_value(float(getattr(config, name)))
            control.set_locked(self.state.reference_locked)
        index = self.scenario.findData(config.scenario)
        self.scenario.blockSignals(True)
        self.scenario.setCurrentIndex(max(0, index))
        self.scenario.setEnabled(not self.state.reference_locked)
        self.reset_preset.setEnabled(not self.state.reference_locked)
        self.scenario.blockSignals(False)

    def refresh_data(self, data: pd.DataFrame) -> None:
        columns = ["row_id", "T", "V", "L", "D", "S", "Y_CR", "Y_CFO", "propensity_true"]
        shown = data.loc[:, [column for column in columns if column in data]].head(80)
        self.preview.setRowCount(len(shown))
        self.preview.setColumnCount(len(shown.columns))
        self.preview.setHorizontalHeaderLabels(list(shown.columns))
        for row, values in enumerate(shown.itertuples(index=False, name=None)):
            for column, value in enumerate(values):
                self.preview.setItem(
                    row,
                    column,
                    QTableWidgetItem(f"{value:.4g}" if isinstance(value, float) else str(value)),
                )
        self.overlap_plot.set_figure(plot_overlap(data))


class EvidenceStructuresWorkspace(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        root = QVBoxLayout(self)
        title_row = QHBoxLayout()
        title = QLabel("Свидетельства и структуры")
        title.setObjectName("workspaceTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(QLabel("α-срез"))
        self.alpha = NumericControl(0.40, 0.92, 0.60, 2)
        self.alpha.valueChanged.connect(self.state.set_alpha)
        self.alpha.setMaximumWidth(350)
        title_row.addWidget(self.alpha)
        root.addLayout(title_row)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        evidence_actions = QHBoxLayout()
        self.add_evidence = QPushButton("Добавить")
        self.remove_evidence = QPushButton("Удалить")
        self.import_evidence = QPushButton("Загрузить CSV/JSON")
        self.reset_evidence = QPushButton("Вернуть пресет")
        self.add_evidence.clicked.connect(self._add_evidence_item)
        self.remove_evidence.clicked.connect(self._remove_evidence_item)
        self.import_evidence.clicked.connect(self._import_evidence_bundle)
        self.reset_evidence.clicked.connect(self.state.reset_evidence)
        for button in (
            self.add_evidence,
            self.remove_evidence,
            self.import_evidence,
            self.reset_evidence,
        ):
            evidence_actions.addWidget(button)
        evidence_actions.addStretch(1)
        left_layout.addLayout(evidence_actions)
        self.evidence_table = QTableWidget()
        self.evidence_table.setColumnCount(9)
        self.evidence_table.setHorizontalHeaderLabels(
            [
                "ID",
                "Утверждение",
                "Вид",
                "Тип источника",
                "Поддержка",
                "Надёжность",
                "Применимость",
                "Группа",
                "Provenance",
            ]
        )
        self.evidence_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.evidence_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.evidence_table.itemChanged.connect(self._evidence_item_changed)
        left_layout.addWidget(self.evidence_table)
        self.core_label = QLabel("Структурный каркас будет сформирован после запуска")
        self.core_label.setWordWrap(True)
        self.core_label.setObjectName("noteBox")
        left_layout.addWidget(self.core_label)
        splitter.addWidget(left)
        self.plot_tabs = QTabWidget()
        self.score_plot = PlotPanel()
        self.graph_plot = PlotPanel()
        self.cascade_plot = PlotPanel()
        self.plot_tabs.addTab(self.score_plot, "μΓ(G)")
        self.plot_tabs.addTab(self.graph_plot, "G₁–G₄")
        self.plot_tabs.addTab(self.cascade_plot, "Каскад Γα")
        splitter.addWidget(self.plot_tabs)
        splitter.setSizes([560, 760])
        self.graph_plot.set_figure(plot_graphs(reference_graphs()))
        self._load_evidence(reference_evidence())
        self._set_evidence_editor_enabled(not state.reference_locked)
        state.result_changed.connect(self.refresh_result)
        state.alpha_changed.connect(self.refresh_alpha)
        state.evidence_changed.connect(self._load_evidence)
        state.state_changed.connect(
            lambda _state: self._set_evidence_editor_enabled(not self.state.reference_locked)
        )

    def _load_evidence(self, bundle) -> None:
        self.evidence_table.blockSignals(True)
        self.evidence_table.setRowCount(len(bundle.items))
        for row, item in enumerate(bundle.items):
            values = [
                item.evidence_id,
                item.assertion,
                item.assertion_kind,
                item.evidence_type,
                f"{item.support:.3f}",
                "—" if item.reliability is None else f"{item.reliability:.2f}",
                f"{item.applicability:.3f}",
                item.dependent_group,
                item.provenance,
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                if column not in {1, 4, 5, 6}:
                    table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.evidence_table.setItem(row, column, table_item)
        self.evidence_table.blockSignals(False)

    def _set_evidence_editor_enabled(self, enabled: bool) -> None:
        for button in (
            self.add_evidence,
            self.remove_evidence,
            self.import_evidence,
            self.reset_evidence,
        ):
            button.setEnabled(enabled)
        self.evidence_table.setEnabled(True)

    def _evidence_item_changed(self, item: QTableWidgetItem) -> None:
        if self.state.reference_locked or item.column() not in {1, 4, 5, 6}:
            return
        evidence_id_item = self.evidence_table.item(item.row(), 0)
        if evidence_id_item is None:
            return
        field = {1: "assertion", 4: "support", 5: "reliability", 6: "applicability"}[
            item.column()
        ]
        try:
            value = item.text() if field == "assertion" else float(item.text().replace(",", "."))
            self.state.update_evidence_item(evidence_id_item.text(), **{field: value})
        except Exception as exc:
            QMessageBox.warning(self, "Некорректное свидетельство", str(exc))
            self._load_evidence(self.state.evidence_bundle)

    def _add_evidence_item(self) -> None:
        item = EvidenceItem(
            evidence_id=f"user_{uuid.uuid4().hex[:8]}",
            assertion_id=f"user_assertion_{uuid.uuid4().hex[:8]}",
            assertion="Пользовательское утверждение: прямой путь T→Y",
            assertion_kind="path",
            path=("T", "Y"),
            expected_present=True,
            support=0.50,
            reliability=0.70,
            applicability=1.0,
            provenance="user_entry",
            evidence_type="expert",
            dependent_group="user_independent_group",
            version="custom",
            context="user_defined",
        )
        self.state.add_evidence_item(item)

    def _remove_evidence_item(self) -> None:
        row = self.evidence_table.currentRow()
        if row < 0:
            return
        evidence_id = self.evidence_table.item(row, 0).text()
        self.state.remove_evidence_item(evidence_id)

    @staticmethod
    def _parse_edge_or_path(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return tuple(str(part) for part in value)
        text = str(value).replace("→", "->")
        parts = [part.strip() for part in text.split("->")]
        return tuple(parts) if len(parts) == 2 else None

    def _import_evidence_bundle(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Импорт корпуса свидетельств",
            "",
            "Корпус свидетельств (*.json *.csv)",
        )
        if not path:
            return
        try:
            if path.lower().endswith(".json"):
                payload = json.loads(open(path, encoding="utf-8").read())
                if isinstance(payload, list):
                    payload = {"version": "custom", "context": "user_defined", "items": payload}
                bundle = EvidenceBundle.model_validate(payload)
            else:
                records = pd.read_csv(path).to_dict(orient="records")
                items = []
                for record in records:
                    cleaned = {
                        key: value
                        for key, value in record.items()
                        if not (isinstance(value, float) and pd.isna(value))
                    }
                    cleaned["edge"] = self._parse_edge_or_path(cleaned.get("edge"))
                    cleaned["path"] = self._parse_edge_or_path(cleaned.get("path"))
                    items.append(EvidenceItem.model_validate(cleaned))
                bundle = EvidenceBundle(
                    version="custom-csv",
                    context="user_defined",
                    items=tuple(items),
                )
            self.state.replace_evidence_bundle(bundle)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка импорта свидетельств", str(exc))

    def refresh_result(self, result: AnalysisResult) -> None:
        self.score_plot.set_figure(plot_scores(result.graph_scores, self.state.config.alpha))
        self.cascade_plot.set_figure(plot_alpha_cascade(result.alpha_cuts))
        self.refresh_alpha(self.state.config.alpha)

    def refresh_alpha(self, alpha: float) -> None:
        self.alpha.set_value(alpha)
        if self.state.result is None:
            return
        self.score_plot.set_figure(plot_scores(self.state.result.graph_scores, alpha))
        cut = min(self.state.result.alpha_cuts, key=lambda item: abs(item.alpha - alpha))
        endogenous_core = [
            f"{a}→{b}"
            for a, b in cut.core_edges
            if a in {"T", "L", "D", "Y"} and b in {"T", "L", "D", "Y"}
        ]
        endogenous_alt = [
            f"{a}→{b}"
            for a, b in cut.alternative_edges
            if a in {"T", "L", "D", "Y"} and b in {"T", "L", "D", "Y"}
        ]
        self.core_label.setText(
            f"Γα = {{{', '.join(cut.graph_ids)}}}\n"
            f"Строгий каркас: {', '.join(endogenous_core) or '—'}\n"
            f"Альтернативы: {', '.join(endogenous_alt) or '—'}"
        )


class CausalInferenceWorkspace(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel("Причинный вывод")
        title.setObjectName("workspaceTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.outcome = QComboBox()
        self.outcome.addItems(["Y_CR", "Y_CFO"])
        self.outcome.currentTextChanged.connect(self.refresh)
        self.graph = QComboBox()
        self.graph.addItems(["G1", "G2", "G3", "G4"])
        self.graph.currentTextChanged.connect(self.refresh)
        header.addWidget(QLabel("Исход"))
        header.addWidget(self.outcome)
        header.addWidget(QLabel("Граф"))
        header.addWidget(self.graph)
        root.addLayout(header)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "Граф",
                "Статус",
                "Функционал",
                "Adjustment",
                "Оценка",
                "Интервал/границы",
                "Диагностика",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        left_layout.addWidget(self.table)
        neural_box = QGroupBox("Neural SCM · одиночный запуск")
        neural_layout = QHBoxLayout(neural_box)
        neural_layout.addWidget(QLabel("2×32 · early stopping · ансамбль 10 seed"))
        self.neural_button = QPushButton("Обучить для выбранного графа")
        self.neural_button.setToolTip(
            "Доступно после идентификации; не меняет identification status"
        )
        self.neural_button.clicked.connect(
            lambda: self.state.train_neural_scm(
                self.graph.currentText(), self.outcome.currentText()
            )
        )
        neural_layout.addWidget(self.neural_button)
        self.neural_status = QLabel("Ожидает запуска")
        self.neural_status.setWordWrap(True)
        neural_layout.addWidget(self.neural_status, 1)
        left_layout.addWidget(neural_box)
        splitter.addWidget(left)
        tabs = QTabWidget()
        self.forest = PlotPanel()
        self.cate = PlotPanel()
        self.overlap = PlotPanel()
        tabs.addTab(self.forest, "Forest plot")
        tabs.addTab(self.cate, "CATE-профили")
        tabs.addTab(self.overlap, "Overlap")
        splitter.addWidget(tabs)
        splitter.setSizes([700, 650])
        state.result_changed.connect(self.refresh)
        state.data_changed.connect(lambda data: self.overlap.set_figure(plot_overlap(data)))
        state.neural_result_changed.connect(self._neural_finished)

    def _neural_finished(self, result) -> None:
        self.neural_status.setText(
            f"{result.backend}: ATE={result.ate:.5f}; "
            f"ансамблевый интервал [{result.interval[0]:.5f}; {result.interval[1]:.5f}]"
        )

    def refresh(self, *_args) -> None:
        result = self.state.result
        if result is None:
            return
        outcome = self.outcome.currentText()
        selected = [effect for effect in result.effects if effect.outcome == outcome]
        self.table.setRowCount(len(selected))
        status_colors = {
            "identified": QColor("#DDEFE5"),
            "partially_identified": QColor("#FFF1C9"),
            "not_identified": QColor("#F7D7D7"),
            "structural_zero": QColor("#DDEAF7"),
        }
        for row, effect in enumerate(selected):
            values = [
                effect.graph_id,
                effect.status.value,
                effect.functional or "—",
                ", ".join(effect.adjustment_set) or "—",
                "—" if effect.estimate is None else f"{effect.estimate:.5f}",
                str(effect.interval or effect.identified_bounds or "—"),
                "; ".join(effect.warnings) or "OK",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1:
                    item.setBackground(status_colors[effect.status.value])
                self.table.setItem(row, column, item)
        self.forest.set_figure(plot_graph_specific_forest(result.effects, outcome))
        self.cate.set_figure(plot_cate_profiles(result.effects, outcome, self.graph.currentText()))


class StabilityDecisionWorkspace(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        root = QVBoxLayout(self)
        title = QLabel("Устойчивость и решение")
        title.setObjectName("workspaceTitle")
        root.addWidget(title)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)
        left = QTabWidget()
        self.stability_plot = PlotPanel()
        self.value_plot = PlotPanel()
        left.addTab(self.stability_plot, "α-траектория")
        left.addTab(self.value_plot, "Value / regret")
        splitter.addWidget(left)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.decision_label = QLabel("Выполните анализ для получения рекомендации")
        self.decision_label.setWordWrap(True)
        self.decision_label.setObjectName("decisionCard")
        right_layout.addWidget(self.decision_label)
        self.trajectory = QTableWidget(0, 5)
        self.trajectory.setHorizontalHeaderLabels(["α", "Γα", "Статус", "Действие", "Макс. regret"])
        self.trajectory.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.trajectory.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.trajectory, 1)
        coefficients = QGroupBox("Коэффициенты ценностной модели")
        form = QFormLayout(coefficients)
        self.coefficient_edits: dict[str, QLineEdit] = {}
        for label, field in (
            ("Вес ΔCR", "cr_weight"),
            ("Вес ΔAR", "arrears_weight"),
            ("Штраф LossSales", "sales_loss_weight"),
            ("Штраф Pzombie", "zombie_weight"),
            ("Cprogram(a1)", "program_cost_a1"),
            ("Cprogram(a2)", "program_cost_a2"),
            ("Стоимость информации", "pilot_information_cost"),
        ):
            edit = QLineEdit()
            edit.setReadOnly(True)
            form.addRow(label, edit)
            self.coefficient_edits[field] = edit
        right_layout.addWidget(coefficients)
        splitter.addWidget(right)
        splitter.setSizes([760, 560])
        state.result_changed.connect(self.refresh)
        state.alpha_changed.connect(lambda _: self.refresh(state.result) if state.result else None)
        state.config_changed.connect(self.refresh_config)
        self.refresh_config(state.config)

    def refresh_config(self, config: GeneratorConfig) -> None:
        for field, edit in self.coefficient_edits.items():
            edit.setText(f"{getattr(config, field):.4g}")

    def refresh(self, result: AnalysisResult) -> None:
        self.stability_plot.set_figure(plot_stability_map(result.stability))
        if not result.decisions:
            return
        target = min(
            result.decisions, key=lambda decision: abs(decision.alpha - self.state.config.alpha)
        )
        self.value_plot.set_figure(plot_value_regret(target))
        summary = result.trajectory_summary
        alpha_range = (
            "—"
            if summary.stable_alpha_range is None
            else f"[{summary.stable_alpha_range[0]:.2f}; {summary.stable_alpha_range[1]:.2f}]"
        )
        evi_se = (
            "—"
            if target.pilot_information_value_se is None
            else f"{target.pilot_information_value_se:.5f}"
        )
        self.decision_label.setText(
            f"<b>{summary.status}</b> · условное действие: "
            f"{summary.selected_action or 'не выбрано'} · диапазон α: {alpha_range}<br>"
            f"Безусловное действие: "
            f"{summary.operational_decision.selected_action or 'мотивированный отказ'}<br>"
            f"{summary.reason}<br>Чистая EVI пилота на текущем срезе: "
            f"{target.pilot_information_value:.5f}; MCSE: {evi_se}; seed: {target.pilot_seed}"
        )
        self.trajectory.setRowCount(len(result.decisions))
        cuts = {cut.alpha: cut for cut in result.alpha_cuts}
        for row, decision in enumerate(result.decisions):
            maximum = [value for value in decision.maximum_regret.values() if value is not None]
            values = [
                f"{decision.alpha:.2f}",
                ", ".join(cuts[decision.alpha].graph_ids),
                decision.status,
                decision.selected_action or "—",
                "—" if not maximum else f"{min(maximum):.5f}",
            ]
            for column, value in enumerate(values):
                self.trajectory.setItem(row, column, QTableWidgetItem(value))


class ExperimentPassportWorkspace(QWidget):
    smoke_requested = Signal(int)
    replay_requested = Signal(str)

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        root = QVBoxLayout(self)
        title = QLabel("Эксперимент и паспорт")
        title.setObjectName("workspaceTitle")
        root.addWidget(title)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self.tabs.addTab(self._queue_tab(), "Очередь эксперимента")
        self.tabs.addTab(self._comparison_tab(), "Сравнение методов")
        self.tabs.addTab(self._passport_tab(), "Паспорт и воспроизведение")
        state.result_changed.connect(self.refresh_result)

    def _queue_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        design = QTableWidget(10, 4)
        design.setHorizontalHeaderLabels(["Сценарий", "n", "Повторы", "Статус"])
        cells = [
            ("S1 Опорный", 600),
            ("S1 Опорный", 1500),
            ("S1 Опорный", 3000),
            ("S2 Конфликт", 1500),
            ("S3 Слабое overlap", 600),
            ("S3 Слабое overlap", 1500),
            ("S3 Слабое overlap", 3000),
            ("S4 Версии", 1500),
            ("S5 Потери", 1500),
            ("S6 Вне Γ", 1500),
        ]
        for row, (scenario, n) in enumerate(cells):
            for column, value in enumerate((scenario, n, 500, "ожидает")):
                design.setItem(row, column, QTableWidgetItem(str(value)))
        design.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(design)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Повторов на ячейку (лабораторный режим)"))
        self.repetitions = QSpinBox()
        self.repetitions.setRange(1, 50)
        self.repetitions.setValue(10)
        controls.addWidget(self.repetitions)
        smoke = QPushButton("Запустить smoke Monte Carlo")
        smoke.clicked.connect(lambda: self.smoke_requested.emit(self.repetitions.value()))
        controls.addWidget(smoke)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.queue_log = QTextEdit()
        self.queue_log.setReadOnly(True)
        self.queue_log.setPlaceholderText("Checkpoint, ошибки и частичные агрегаты появятся здесь")
        layout.addWidget(self.queue_log)
        return tab

    def _comparison_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.comparison_table = QTableWidget(0, 7)
        self.comparison_table.setHorizontalHeaderLabels(
            ["Метод", "Bias", "RMSE", "Coverage", "Regret", "Ошибочный a2", "MCSE"]
        )
        self.comparison_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.comparison_table)
        note = QLabel(
            "Сравниваются только утверждённые конфигурации: полная процедура, максимальный граф, "
            "жёсткое множество и Structure Oracle. Oracle-информация недоступна обычным методам."
        )
        note.setWordWrap(True)
        note.setObjectName("noteBox")
        layout.addWidget(note)
        return tab

    def _passport_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.passport_tree = QTreeWidget()
        self.passport_tree.setHeaderLabels(["Блок", "Значение / статус"])
        self.passport_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.passport_tree, 1)
        saved_row = QHBoxLayout()
        saved_row.addWidget(QLabel("Сохранённые запуски"))
        self.saved_runs = QComboBox()
        self.saved_runs.setObjectName("savedRunsCombo")
        self.saved_runs.currentTextChanged.connect(self._select_saved_run)
        saved_row.addWidget(self.saved_runs, 1)
        self.refresh_saved_runs()
        layout.addLayout(saved_row)
        replay_row = QHBoxLayout()
        replay_row.addWidget(QLabel("Run ID"))
        self.run_id = QLineEdit()
        replay_row.addWidget(self.run_id, 1)
        self.replay_run_button = QPushButton("Повторить и сравнить")
        self.replay_run_button.clicked.connect(
            lambda: self.replay_requested.emit(self.run_id.text().strip())
        )
        replay_row.addWidget(self.replay_run_button)
        layout.addLayout(replay_row)
        self.replay_status = QLabel("Replay не выполнялся")
        self.replay_status.setObjectName("statusChip")
        layout.addWidget(self.replay_status)
        return tab

    def _select_saved_run(self, run_id: str) -> None:
        if run_id:
            self.run_id.setText(run_id)

    def refresh_saved_runs(self, preferred_run_id: str | None = None) -> None:
        current = preferred_run_id or self.saved_runs.currentText()
        run_ids = [row["run_id"] for row in self.state.repository.list_runs()]
        self.saved_runs.blockSignals(True)
        self.saved_runs.clear()
        self.saved_runs.addItems(run_ids)
        if current in run_ids:
            self.saved_runs.setCurrentText(current)
        self.saved_runs.blockSignals(False)
        self._select_saved_run(self.saved_runs.currentText())

    def refresh_result(self, result: AnalysisResult) -> None:
        self.run_id.setText(result.manifest.run_id)
        self.refresh_saved_runs(result.manifest.run_id)
        self.passport_tree.clear()
        passport = result.passport
        blocks = {
            "Причинная задача": [query.as_dict() for query in passport.causal_queries],
            "Данные и свидетельства": {
                "dataset": passport.dataset.as_dict(),
                "evidence": passport.evidence,
            },
            "Структурное пространство": passport.structural_space,
            "Идентификация и оценивание": passport.identification_and_estimation,
            "Профиль неопределённости": passport.uncertainty_profile,
            "α-зависимая устойчивость": passport.alpha_stability,
            "Управленческое решение": passport.decision,
            "Предпосылки и ограничения": passport.assumptions_and_limitations,
            "Воспроизводимость и аудит": passport.validation,
        }
        for name, value in blocks.items():
            root = QTreeWidgetItem([name, ""])
            text = json.dumps(value, ensure_ascii=False, default=str)
            root.addChild(
                QTreeWidgetItem(["Содержимое", text[:1500] + ("…" if len(text) > 1500 else "")])
            )
            self.passport_tree.addTopLevelItem(root)
        self.passport_tree.expandToDepth(0)

    def show_monte_carlo(self, aggregate_path) -> None:
        frame = pd.read_parquet(aggregate_path)
        self.comparison_table.setRowCount(len(frame))
        for row, record in enumerate(frame.itertuples(index=False)):
            bias = getattr(record, "y_cr_ate_error", float("nan"))
            rmse = getattr(record, "y_cr_ate_rmse", float("nan"))
            coverage = getattr(record, "y_cr_coverage", float("nan"))
            regret = getattr(record, "regret", float("nan"))
            error_a2 = getattr(record, "erroneous_a2", float("nan"))
            mcse = getattr(record, "regret_mcse", float("nan"))
            values = (
                f"{record.method} · {record.scenario} · n={record.sample_size}",
                f"{bias:.5f}",
                f"{rmse:.5f}",
                f"{coverage:.3f}",
                f"{regret:.5f}",
                f"{error_a2:.3f}",
                f"{mcse:.5f}",
            )
            for column, value in enumerate(values):
                self.comparison_table.setItem(row, column, QTableWidgetItem(value))
