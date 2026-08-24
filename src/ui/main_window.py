from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from engine.graphs import reference_graphs
from runtime import ExportManager
from visualization import (
    plot_alpha_cascade,
    plot_graph_specific_forest,
    plot_stability_map,
    plot_value_regret,
    save_figure,
)

from .state import AppState
from .workspaces import (
    CausalInferenceWorkspace,
    EvidenceStructuresWorkspace,
    ExperimentPassportWorkspace,
    ProjectDataWorkspace,
    StabilityDecisionWorkspace,
)

LIGHT_STYLE = """
QMainWindow, QWidget { background: #F5F7F9; color: #263442; font-family: 'Segoe UI'; font-size: 10pt; }
QToolBar { background: #FFFFFF; border-bottom: 1px solid #D8E0E7; spacing: 6px; padding: 5px; }
QToolButton, QPushButton { background: #FFFFFF; border: 1px solid #C8D3DD; border-radius: 6px; padding: 6px 11px; }
QPushButton:hover { border-color: #2F6FB0; background: #F0F6FC; }
QPushButton:disabled { color: #9BA5AE; background: #EEF1F4; }
QListWidget { background: #172536; border: none; color: #DDE6EE; outline: none; padding: 8px; }
QListWidget::item { padding: 12px 10px; margin: 3px 0; border-radius: 7px; }
QListWidget::item:selected { background: #2F6FB0; color: white; }
QTabWidget::pane { border: 1px solid #D8E0E7; background: white; border-radius: 7px; }
QTabBar::tab { background: #E9EEF3; padding: 8px 14px; margin-right: 2px; }
QTabBar::tab:selected { background: white; color: #245D94; font-weight: 600; }
QTableWidget, QTreeWidget, QTextEdit { background: white; border: 1px solid #D8E0E7; border-radius: 6px; gridline-color: #E5EAF0; }
QHeaderView::section { background: #E8EFF6; color: #334155; padding: 6px; border: none; border-right: 1px solid #D5DEE7; font-weight: 600; }
QGroupBox { background: white; border: 1px solid #D8E0E7; border-radius: 7px; margin-top: 12px; padding-top: 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: white; border: 1px solid #C8D3DD; border-radius: 5px; padding: 5px; }
QSlider::groove:horizontal { height: 5px; background: #DCE4EB; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #2F6FB0; border-radius: 7px; }
#workspaceTitle { font-size: 18pt; font-weight: 650; color: #19324A; padding: 5px 0; }
#plotPanel { background: white; border: 1px solid #D8E0E7; border-radius: 7px; }
#noteBox, #sourceBox { background: #EEF5FB; border-left: 4px solid #2F6FB0; border-radius: 4px; padding: 9px; }
#decisionCard { background: #EAF5EF; border-left: 5px solid #2E7D5B; border-radius: 6px; padding: 12px; font-size: 11pt; }
#statusChip { background: #E8EFF6; color: #245D94; border-radius: 10px; padding: 4px 9px; font-weight: 600; }
QProgressBar { border: 1px solid #C8D3DD; border-radius: 5px; background: white; text-align: center; }
QProgressBar::chunk { background: #2F6FB0; border-radius: 4px; }
"""


class MainWindow(QMainWindow):
    def __init__(self, repository_root: str | Path):
        super().__init__()
        self.setWindowTitle("NF-Causal Decision Workbench")
        self.resize(1440, 900)
        self.setMinimumSize(1280, 720)
        self.setStyleSheet(LIGHT_STYLE)
        self.state_model = AppState(repository_root)
        self._build_toolbar()
        self._build_central()
        self._build_warning_dock()
        self._build_statusbar()
        self._connect_state()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Главная панель")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        title = QLabel("  NF-Causal Lab  ")
        title.setStyleSheet("font-weight:700; font-size:12pt; color:#19324A")
        toolbar.addWidget(title)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Проект: "))
        self.project_label = QLabel("Эталонный финансовый кейс")
        self.project_label.setObjectName("statusChip")
        toolbar.addWidget(self.project_label)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Run ID: "))
        self.run_label = QLabel("—")
        toolbar.addWidget(self.run_label)
        toolbar.addSeparator()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Один запуск", "Монте-Карло"])
        toolbar.addWidget(self.mode_combo)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self.state_chip = QLabel("draft")
        self.state_chip.setObjectName("statusChip")
        toolbar.addWidget(self.state_chip)
        self.validate_button = QPushButton("Проверить")
        self.run_button = QPushButton("Выполнить")
        self.stop_button = QPushButton("Остановить")
        self.replay_button = QPushButton("Повторить")
        self.export_button = QPushButton("Экспортировать")
        self.stop_button.setEnabled(False)
        self.replay_button.setEnabled(False)
        self.export_button.setEnabled(False)
        for button in (
            self.validate_button,
            self.run_button,
            self.stop_button,
            self.replay_button,
            self.export_button,
        ):
            toolbar.addWidget(button)
        self.validate_button.clicked.connect(self._validate)
        self.run_button.clicked.connect(self._execute)
        self.stop_button.clicked.connect(self.state_model.cancel)
        self.replay_button.clicked.connect(self._replay_current)
        self.export_button.clicked.connect(self._export)

    def _build_central(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        self.navigation = QListWidget()
        self.navigation.setFixedWidth(235)
        labels = [
            "1  Проект и данные",
            "2  Свидетельства и структуры",
            "3  Причинный вывод",
            "4  Устойчивость и решение",
            "5  Эксперимент и паспорт",
        ]
        for label in labels:
            self.navigation.addItem(QListWidgetItem(label))
        layout.addWidget(self.navigation)
        self.pages = QStackedWidget()
        self.project_workspace = ProjectDataWorkspace(self.state_model)
        self.evidence_workspace = EvidenceStructuresWorkspace(self.state_model)
        self.inference_workspace = CausalInferenceWorkspace(self.state_model)
        self.stability_workspace = StabilityDecisionWorkspace(self.state_model)
        self.experiment_workspace = ExperimentPassportWorkspace(self.state_model)
        for workspace in (
            self.project_workspace,
            self.evidence_workspace,
            self.inference_workspace,
            self.stability_workspace,
            self.experiment_workspace,
        ):
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(18, 12, 18, 12)
            container_layout.addWidget(workspace)
            self.pages.addWidget(container)
        layout.addWidget(self.pages, 1)
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.navigation.setCurrentRow(0)
        self.setCentralWidget(central)
        self.experiment_workspace.replay_requested.connect(self._replay_run)
        self.experiment_workspace.smoke_requested.connect(self._smoke_requested)
        self.state_model.monte_carlo_changed.connect(self._monte_carlo_finished)

    def _build_warning_dock(self) -> None:
        self.warning_dock = QDockWidget("Предупреждения и происхождение", self)
        self.warning_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.warning_list = QListWidget()
        self.warning_list.addItem("Анализ ещё не выполнен")
        self.warning_dock.setWidget(self.warning_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.warning_dock)
        self.warning_dock.setMinimumWidth(290)

    def _build_statusbar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(260)
        self.progress_text = QLabel("Готово")
        self.memory_label = QLabel("Память: —")
        bar.addWidget(self.progress)
        bar.addWidget(self.progress_text, 1)
        bar.addPermanentWidget(self.memory_label)

    def _connect_state(self) -> None:
        self.state_model.state_changed.connect(self._state_changed)
        self.state_model.progress_changed.connect(self._progress_changed)
        self.state_model.warnings_changed.connect(self._warnings_changed)
        self.state_model.result_changed.connect(self._result_changed)

    @Slot()
    def _validate(self) -> None:
        valid, warnings = self.state_model.validate()
        if valid:
            self.progress_text.setText("Конфигурация валидна")
        elif warnings:
            self.progress_text.setText("Требуется проверка предупреждений")

    @Slot()
    def _execute(self) -> None:
        self.state_model.execute(compute_cate=True)

    @Slot(str)
    def _state_changed(self, state: str) -> None:
        self.state_chip.setText(state)
        running = state == "running"
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.export_button.setEnabled(self.state_model.result is not None and not running)
        self.replay_button.setEnabled(self.state_model.result is not None and not running)

    @Slot(int, str)
    def _progress_changed(self, value: int, text: str) -> None:
        self.progress.setValue(value)
        self.progress_text.setText(text)

    @Slot(object)
    def _warnings_changed(self, warnings) -> None:
        self.warning_list.clear()
        if not warnings:
            self.warning_list.addItem("Критических предупреждений нет")
            return
        for warning in warnings:
            item = QListWidgetItem(str(warning))
            item.setToolTip(str(warning))
            self.warning_list.addItem(item)

    @Slot(object)
    def _result_changed(self, result) -> None:
        self.run_label.setText(result.manifest.run_id)
        self.export_button.setEnabled(True)
        self.replay_button.setEnabled(True)

    def _replay_current(self) -> None:
        if self.state_model.result:
            self._replay_run(self.state_model.result.manifest.run_id)

    @Slot(str)
    def _replay_run(self, run_id: str) -> None:
        if not run_id:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            replay = self.state_model.repository.replay(run_id)
            if replay.matched:
                text = f"Replay совпал в допуске {replay.tolerance:g}; новый run_id: {replay.replay_run_id}"
            else:
                text = "Replay не совпал: " + "; ".join(replay.differences)
            self.experiment_workspace.replay_status.setText(text)
            self.progress_text.setText(text)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка replay", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    @Slot(int)
    def _smoke_requested(self, repetitions: int) -> None:
        self.experiment_workspace.queue_log.append(
            f"Запущен smoke run: {repetitions} повторов × 10 ячеек; checkpoint включён."
        )
        self.state_model.start_monte_carlo(repetitions)

    @Slot(object, object)
    def _monte_carlo_finished(self, counts, paths) -> None:
        self.experiment_workspace.queue_log.append(
            f"Smoke run завершён: {counts}. Агрегаты: {paths[2]}"
        )
        self.experiment_workspace.show_monte_carlo(paths[1])
        self.progress_text.setText("Monte Carlo завершён")

    def _export(self) -> None:
        result = self.state_model.result
        data = self.state_model.data
        if result is None or data is None:
            return
        destination = QFileDialog.getExistingDirectory(self, "Папка экспорта")
        if not destination:
            return
        target = Path(destination) / result.manifest.run_id
        target.mkdir(parents=True, exist_ok=True)
        ExportManager.export_table(data, target / "dataset.csv")
        ExportManager.export_table(data, target / "dataset.xlsx")
        ExportManager.export_table(data, target / "dataset.parquet")
        ExportManager.export_json(result.passport.as_dict(), target / "CausalDecisionPassport.json")
        ExportManager.passport_pdf(result.passport, target / "CausalDecisionPassport.pdf")
        for graph in reference_graphs():
            ExportManager.export_graphml(graph, target / f"{graph.graph_id}.graphml")
        figures = {
            "graph_specific_forest_Y_CR": plot_graph_specific_forest(result.effects, "Y_CR"),
            "alpha_cascade": plot_alpha_cascade(result.alpha_cuts),
            "stability_map": plot_stability_map(result.stability),
            "value_regret": plot_value_regret(result.trajectory_summary.operational_decision),
        }
        for name, figure in figures.items():
            for suffix in ("png", "svg", "pdf"):
                save_figure(figure, target / f"{name}.{suffix}")
        self.progress_text.setText(f"Экспорт завершён: {target}")
