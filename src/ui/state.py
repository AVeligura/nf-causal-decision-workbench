from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from domain import AnalysisResult, DatasetSpec, EvidenceBundle, EvidenceItem, GeneratorConfig
from engine import NeuralSCMConfig, fit_neural_scm, run_analysis
from engine.evidence import reference_evidence
from engine.graphs import reference_graphs
from runtime import RunRepository
from study import generate_dataset
from study.monte_carlo import MonteCarloRunner
from study.scenarios import apply_scenario


class WorkerSignals(QObject):
    started = Signal(str)
    progress = Signal(int, str)
    finished = Signal(object, object)
    failed = Signal(str)


class AnalysisWorker(QRunnable):
    def __init__(
        self,
        config: GeneratorConfig,
        repository: RunRepository,
        compute_cate: bool = True,
        imported_data: pd.DataFrame | None = None,
        dataset_spec: DatasetSpec | None = None,
        evidence_bundle: EvidenceBundle | None = None,
    ):
        super().__init__()
        self.config = config
        self.repository = repository
        self.compute_cate = compute_cate
        self.imported_data = imported_data
        self.dataset_spec = dataset_spec
        self.evidence_bundle = evidence_bundle
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.started.emit("Подготовка данных")
            if self.imported_data is None:
                self.signals.progress.emit(10, "Формируется полусинтетическая выборка")
                generated = generate_dataset(self.config)
                data = generated.data
                config = generated.config
            else:
                self.signals.progress.emit(10, "Проверенная импортированная выборка принята")
                data = self.imported_data.copy()
                config = self.config
            self.signals.progress.emit(30, "Свидетельства, графы и α-срезы")
            result = run_analysis(
                config,
                data=data,
                dataset_spec=self.dataset_spec,
                evidence_bundle=self.evidence_bundle,
                compute_cate=self.compute_cate,
            )
            self.signals.progress.emit(90, "Сохранение паспорта и пакета запуска")
            self.repository.save(result, data)
            self.signals.progress.emit(100, "Завершено")
            self.signals.finished.emit(result, data)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class NeuralWorker(QRunnable):
    def __init__(
        self,
        data: pd.DataFrame,
        graph_id: str,
        outcome: str,
        repository: RunRepository,
        run_id: str,
        config: NeuralSCMConfig,
    ):
        super().__init__()
        self.data = data
        self.graph_id = graph_id
        self.outcome = outcome
        self.repository = repository
        self.run_id = run_id
        self.config = config
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.progress.emit(5, f"Neural SCM: {self.graph_id}, {self.outcome}")
            graph = next(item for item in reference_graphs() if item.graph_id == self.graph_id)
            result = fit_neural_scm(self.data, graph, self.outcome, self.config)
            self.signals.progress.emit(95, "Сохранение конфигурации и истории Neural SCM")
            relative = Path("results") / f"neural_scm_{self.graph_id}_{self.outcome}.json"
            self.repository.attach_json(self.run_id, relative, result.as_dict())
            self.signals.progress.emit(100, "Neural SCM завершён")
            self.signals.finished.emit(result, relative)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class MonteCarloWorker(QRunnable):
    def __init__(self, runner: MonteCarloRunner, repetitions: int, workers: int = 2):
        super().__init__()
        self.runner = runner
        self.repetitions = repetitions
        self.workers = workers
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            total = self.runner.prepare(self.repetitions)

            def report(counts: dict[str, int], eta: float) -> None:
                completed = counts.get("completed", 0) + counts.get("failed", 0)
                percent = round(100 * completed / max(total, 1))
                message = f"MC {completed}/{total}; ETA {eta / 60:.1f} мин; {counts}"
                self.signals.progress.emit(percent, message)

            counts = self.runner.run(
                workers=self.workers,
                compute_cate=False,
                progress=report,
            )
            paths = self.runner.export()
            self.signals.finished.emit(counts, paths)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class AppState(QObject):
    config_changed = Signal(object)
    data_changed = Signal(object)
    result_changed = Signal(object)
    alpha_changed = Signal(float)
    warnings_changed = Signal(object)
    state_changed = Signal(str)
    progress_changed = Signal(int, str)
    neural_result_changed = Signal(object)
    monte_carlo_changed = Signal(object, object)
    evidence_changed = Signal(object)

    def __init__(self, repository_root: str | Path):
        super().__init__()
        self.config = GeneratorConfig()
        self.data: pd.DataFrame | None = None
        self.result: AnalysisResult | None = None
        self.dataset_spec = DatasetSpec()
        self.evidence_bundle = reference_evidence()
        self.repository = RunRepository(repository_root)
        self.state = "draft"
        self.reference_locked = True
        self.thread_pool = QThreadPool.globalInstance()
        self._worker: AnalysisWorker | None = None
        self._neural_worker: NeuralWorker | None = None
        self._mc_worker: MonteCarloWorker | None = None
        self._mc_runner: MonteCarloRunner | None = None
        self._cancel_requested = False

    def clone_reference(self) -> None:
        values = self.config.model_dump()
        values.update(mode="laboratory", profile_name="laboratory-copy")
        self.config = GeneratorConfig.model_validate(values)
        self.reference_locked = False
        self.state = "draft"
        self.config_changed.emit(self.config)
        self.state_changed.emit(self.state)

    def set_config_value(self, name: str, value: Any) -> None:
        if self.reference_locked:
            return
        values = self.config.model_dump()
        values[name] = value
        if name == "scenario":
            self.config = apply_scenario(GeneratorConfig.model_validate(values))
        else:
            values["customized"] = True
            self.config = GeneratorConfig.model_validate(values)
        if name in {"scenario", "evidence_reliability", "evidence_conflict"} and self.evidence_bundle.version != "custom":
            self.evidence_bundle = reference_evidence(
                reliability_multiplier=self.config.evidence_reliability / 0.90,
                conflict_strength=self.config.evidence_conflict,
            )
            self.evidence_changed.emit(self.evidence_bundle)
        self.state = "stale" if self.result is not None else "draft"
        self.config_changed.emit(self.config)
        self.state_changed.emit(self.state)

    def set_alpha(self, alpha: float) -> None:
        values = self.config.model_dump()
        values["alpha"] = alpha
        self.config = GeneratorConfig.model_validate(values)
        self.alpha_changed.emit(alpha)
        self.config_changed.emit(self.config)

    def reset_scenario_preset(self) -> None:
        if self.reference_locked:
            return
        self.config = apply_scenario(self.config)
        self.evidence_bundle = reference_evidence(
            reliability_multiplier=self.config.evidence_reliability / 0.90,
            conflict_strength=self.config.evidence_conflict,
        )
        self.state = "stale" if self.result is not None else "draft"
        self.config_changed.emit(self.config)
        self.evidence_changed.emit(self.evidence_bundle)
        self.state_changed.emit(self.state)

    def replace_evidence_bundle(self, bundle: EvidenceBundle) -> None:
        if self.reference_locked:
            return
        self.evidence_bundle = bundle
        self.state = "stale" if self.result is not None else "draft"
        self.evidence_changed.emit(bundle)
        self.state_changed.emit(self.state)

    def update_evidence_item(self, evidence_id: str, **changes: Any) -> None:
        items = []
        found = False
        for item in self.evidence_bundle.items:
            if item.evidence_id == evidence_id:
                items.append(item.model_copy(update=changes))
                found = True
            else:
                items.append(item)
        if not found:
            raise KeyError(evidence_id)
        self.replace_evidence_bundle(
            self.evidence_bundle.model_copy(update={"items": tuple(items), "version": "custom"})
        )

    def add_evidence_item(self, item: EvidenceItem) -> None:
        if any(existing.evidence_id == item.evidence_id for existing in self.evidence_bundle.items):
            raise ValueError(f"Дублирующийся evidence_id: {item.evidence_id}")
        self.replace_evidence_bundle(
            self.evidence_bundle.model_copy(
                update={"items": (*self.evidence_bundle.items, item), "version": "custom"}
            )
        )

    def remove_evidence_item(self, evidence_id: str) -> None:
        remaining = tuple(
            item for item in self.evidence_bundle.items if item.evidence_id != evidence_id
        )
        if len(remaining) == len(self.evidence_bundle.items):
            raise KeyError(evidence_id)
        self.replace_evidence_bundle(
            self.evidence_bundle.model_copy(update={"items": remaining, "version": "custom"})
        )

    def reset_evidence(self) -> None:
        if self.reference_locked:
            return
        self.replace_evidence_bundle(
            reference_evidence(
                reliability_multiplier=self.config.evidence_reliability / 0.90,
                conflict_strength=self.config.evidence_conflict,
            )
        )

    def validate(self) -> tuple[bool, tuple[str, ...]]:
        warnings: list[str] = []
        if self.config.propensity_lower >= self.config.propensity_upper:
            warnings.append("Некорректный диапазон propensity score")
        if self.config.scenario == "weak_overlap":
            warnings.append("Сценарий содержит преднамеренно слабое overlap")
        self.state = "valid" if not warnings else "draft"
        self.state_changed.emit(self.state)
        self.warnings_changed.emit(tuple(warnings))
        return not warnings, tuple(warnings)

    def execute(self, *, compute_cate: bool = True) -> None:
        if self.state == "running":
            return
        self._cancel_requested = False
        self.state = "running"
        self.state_changed.emit(self.state)
        imported = self.data if self.config.mode == "import" else None
        worker = AnalysisWorker(
            self.config,
            self.repository,
            compute_cate=compute_cate,
            imported_data=imported,
            dataset_spec=self.dataset_spec,
            evidence_bundle=self.evidence_bundle,
        )
        worker.signals.progress.connect(self.progress_changed)
        worker.signals.finished.connect(self._analysis_finished)
        worker.signals.failed.connect(self._analysis_failed)
        self._worker = worker
        self.thread_pool.start(worker)

    def load_imported_data(
        self, data: pd.DataFrame, dataset_spec: DatasetSpec | None = None
    ) -> None:
        values = self.config.model_dump()
        values.update(mode="import", profile_name="imported-data")
        self.config = GeneratorConfig.model_validate(values)
        self.reference_locked = False
        self.data = data.copy()
        self.dataset_spec = dataset_spec or DatasetSpec(
            kind="imported",
            source="Пользовательский импорт",
            doi=None,
            license=None,
            selected_features=tuple(),
            checksum_sha256="unknown",
            rows=len(data),
            columns=len(data.columns),
            truth_available=False,
        )
        self.result = None
        self.state = "draft"
        self.config_changed.emit(self.config)
        self.data_changed.emit(self.data)
        self.state_changed.emit(self.state)

    def train_neural_scm(
        self,
        graph_id: str,
        outcome: str,
        *,
        ensemble_size: int = 10,
        max_epochs: int = 250,
    ) -> None:
        if self.data is None or self.result is None or self._neural_worker is not None:
            return
        config = NeuralSCMConfig(
            ensemble_size=ensemble_size,
            max_epochs=max_epochs,
            base_seed=self.config.seed,
        )
        worker = NeuralWorker(
            self.data,
            graph_id,
            outcome,
            self.repository,
            self.result.manifest.run_id,
            config,
        )
        worker.signals.progress.connect(self.progress_changed)
        worker.signals.finished.connect(self._neural_finished)
        worker.signals.failed.connect(self._task_failed)
        self._neural_worker = worker
        self.thread_pool.start(worker)

    def start_monte_carlo(self, repetitions: int, *, workers: int = 2) -> None:
        if self._mc_worker is not None:
            return
        if not 1 <= repetitions <= 50:
            self.warnings_changed.emit(("Smoke Monte Carlo допускает от 1 до 50 повторений",))
            return
        experiment_id = f"gui-smoke-{self.config.seed}-{repetitions}"
        runner = MonteCarloRunner(self.repository.root / "experiments", experiment_id)
        worker = MonteCarloWorker(runner, repetitions, workers=workers)
        worker.signals.progress.connect(self.progress_changed)
        worker.signals.finished.connect(self._monte_carlo_finished)
        worker.signals.failed.connect(self._task_failed)
        self._mc_runner = runner
        self._mc_worker = worker
        self.thread_pool.start(worker)

    @Slot(object, object)
    def _neural_finished(self, result, _relative_path) -> None:
        self._neural_worker = None
        self.neural_result_changed.emit(result)

    @Slot(object, object)
    def _monte_carlo_finished(self, counts, paths) -> None:
        self._mc_worker = None
        self.monte_carlo_changed.emit(counts, paths)

    @Slot(str)
    def _task_failed(self, details: str) -> None:
        self._neural_worker = None
        self._mc_worker = None
        self.warnings_changed.emit((details,))

    def cancel(self) -> None:
        if self._mc_runner is not None:
            self._mc_runner.cancel()
        if self.state != "running":
            return
        self._cancel_requested = True
        self.state = "cancelled"
        self.state_changed.emit(self.state)
        self.progress_changed.emit(0, "Отмена запрошена; текущая атомарная операция завершается")

    @Slot(object, object)
    def _analysis_finished(self, result: AnalysisResult, data: pd.DataFrame) -> None:
        if self._cancel_requested:
            return
        self.result = result
        self.data = data
        self.state = "completed"
        self.data_changed.emit(data)
        self.result_changed.emit(result)
        warnings = sorted({warning for effect in result.effects for warning in effect.warnings})
        self.warnings_changed.emit(tuple(warnings))
        self.state_changed.emit(self.state)

    @Slot(str)
    def _analysis_failed(self, details: str) -> None:
        self.state = "failed"
        self.state_changed.emit(self.state)
        self.warnings_changed.emit((details,))

    def inject_result(self, result: AnalysisResult, data: pd.DataFrame) -> None:
        self.result = result
        self.data = data
        self.config = result.config
        self.dataset_spec = result.dataset_spec
        self.state = result.manifest.state
        self.data_changed.emit(data)
        self.result_changed.emit(result)
        self.config_changed.emit(self.config)
        warnings = sorted({warning for effect in result.effects for warning in effect.warnings})
        self.warnings_changed.emit(tuple(warnings))
        self.state_changed.emit(self.state)
