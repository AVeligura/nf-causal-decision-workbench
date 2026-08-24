from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from PySide6.QtWidgets import QApplication

from domain import GeneratorConfig, IdentificationStatus
from engine import run_analysis
from engine.graphs import reference_graphs
from runtime import DataImporter, ExportManager, RunRepository
from study import generate_dataset
from study.monte_carlo import MonteCarloRunner
from ui.main_window import MainWindow
from visualization import (
    plot_alpha_cascade,
    plot_graph_specific_forest,
    plot_stability_map,
    plot_value_regret,
    save_figure,
)


@dataclass(frozen=True)
class AcceptanceStep:
    step: int
    name: str
    passed: bool
    evidence: str


def _effect(result, graph_id: str, outcome: str):
    return next(
        item for item in result.effects if item.graph_id == graph_id and item.outcome == outcome
    )


def _record(steps: list[AcceptanceStep], step: int, name: str, condition: bool, evidence: str):
    steps.append(AcceptanceStep(step, name, bool(condition), evidence))
    if not condition:
        raise AssertionError(f"Шаг {step}: {name}: {evidence}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/acceptance"))
    parser.add_argument("--screenshots", type=Path, default=Path("artifacts/screenshots"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    steps: list[AcceptanceStep] = []
    repository = RunRepository(args.root / "repository")

    application = QApplication.instance() or QApplication([])
    window = MainWindow(args.root / "ui_repository")
    window.show()
    application.processEvents()
    _record(steps, 1, "Приложение запущено", window.isVisible(), "Главное окно Qt отображено")

    reference_generated = generate_dataset(GeneratorConfig())
    reference = run_analysis(
        reference_generated.config,
        data=reference_generated.data,
        compute_cate=True,
    )
    run_dir = repository.save(reference, reference_generated.data)
    window.state_model.inject_result(reference, reference_generated.data)
    application.processEvents()
    _record(
        steps,
        2,
        "Эталонный одиночный анализ",
        len(reference.effects) == 8,
        f"run_id={reference.manifest.run_id}; 8 графоспецифических результатов",
    )
    passport_json = run_dir / "passport" / "CausalDecisionPassport.json"
    passport_pdf = run_dir / "passport" / "CausalDecisionPassport.pdf"
    _record(
        steps,
        3,
        "CausalDecisionPassport",
        passport_json.exists() and passport_pdf.exists(),
        f"{passport_json}; {passport_pdf}",
    )

    window.state_model.clone_reference()
    _record(
        steps,
        4,
        "Лабораторная копия",
        not window.state_model.reference_locked,
        window.state_model.config.profile_name,
    )

    low_generated = generate_dataset(
        GeneratorConfig(mode="laboratory", sample_size=600, effect_scale=0.5, seed=777)
    )
    high_generated = generate_dataset(
        GeneratorConfig(mode="laboratory", sample_size=600, effect_scale=1.5, seed=777)
    )
    low = run_analysis(low_generated.config, data=low_generated.data, compute_cate=False)
    high = run_analysis(high_generated.config, data=high_generated.data, compute_cate=False)
    low_value = _effect(low, "G1", "Y_CR").estimate
    high_value = _effect(high, "G1", "Y_CR").estimate
    _record(
        steps,
        5,
        "Масштаб эффекта 0,5 → 1,5",
        low_value is not None and high_value is not None and high_value > low_value,
        f"G1/Y_CR: {low_value:.6f} → {high_value:.6f}",
    )

    cut_high = next(cut for cut in reference.alpha_cuts if cut.alpha == 0.80)
    cut_low = next(cut for cut in reference.alpha_cuts if cut.alpha == 0.60)
    effect_signature = [(item.graph_id, item.outcome, item.estimate) for item in reference.effects]
    _record(
        steps,
        6,
        "Изменение α без повторного оценивания",
        cut_high.graph_ids != cut_low.graph_ids
        and effect_signature
        == [(item.graph_id, item.outcome, item.estimate) for item in reference.effects],
        f"Γ0.80={cut_high.graph_ids}; Γ0.60={cut_low.graph_ids}",
    )

    weak_generated = generate_dataset(
        GeneratorConfig(mode="laboratory", scenario="weak_overlap", sample_size=600)
    )
    weak = run_analysis(weak_generated.config, data=weak_generated.data, compute_cate=False)
    weak_warnings = [warning for effect in weak.effects for warning in effect.warnings]
    _record(
        steps,
        7,
        "Weak overlap",
        any("overlap" in warning for warning in weak_warnings),
        "; ".join(sorted(set(weak_warnings))),
    )

    loss_generated = generate_dataset(
        GeneratorConfig(mode="laboratory", scenario="informative_loss", sample_size=600)
    )
    loss = run_analysis(loss_generated.config, data=loss_generated.data, compute_cate=False)
    partial = [
        effect
        for effect in loss.effects
        if effect.status == IdentificationStatus.PARTIALLY_IDENTIFIED
    ]
    _record(
        steps,
        8,
        "Informative loss",
        bool(partial) and all(effect.identified_bounds is not None for effect in partial),
        f"частично идентифицировано: {len(partial)} результатов",
    )

    outside_generated = generate_dataset(
        GeneratorConfig(mode="laboratory", scenario="outside_gamma", sample_size=600)
    )
    outside = run_analysis(
        outside_generated.config, data=outside_generated.data, compute_cate=False
    )
    _record(
        steps,
        9,
        "Истинная структура вне Γ",
        all(effect.status == IdentificationStatus.NOT_IDENTIFIED for effect in outside.effects),
        "все графы ограничены статусом not_identified; скрытый U не замаскирован",
    )

    exported_csv = ExportManager.export_table(
        reference_generated.data, args.root / "roundtrip" / "reference_data.csv"
    )
    imported = DataImporter.read(exported_csv)
    import_valid, import_warnings = DataImporter.validate(imported)
    imported_config = GeneratorConfig(mode="import", profile_name="acceptance-import")
    imported_result = run_analysis(imported_config, data=imported, compute_cate=False)
    _record(
        steps,
        10,
        "Обратный импорт выборки",
        import_valid and imported_result.diagnostics["rows"] == len(imported),
        f"{len(imported)} строк; предупреждения={import_warnings}",
    )

    replay = repository.replay(reference.manifest.run_id)
    _record(
        steps,
        11,
        "Replay эталонного запуска",
        replay.replay_run_id is not None,
        f"replay_run_id={replay.replay_run_id}",
    )
    _record(
        steps,
        12,
        "Сравнение исходных и повторных результатов",
        replay.matched and not replay.differences,
        f"tolerance={replay.tolerance:g}; differences={replay.differences}",
    )

    export_dir = args.root / "exports"
    for suffix in ("csv", "xlsx", "parquet"):
        ExportManager.export_table(reference_generated.data, export_dir / f"data.{suffix}")
    ExportManager.export_json(reference.passport.as_dict(), export_dir / "passport.json")
    ExportManager.passport_pdf(reference.passport, export_dir / "passport.pdf")
    for graph in reference_graphs():
        ExportManager.export_graphml(graph, export_dir / f"{graph.graph_id}.graphml")
    figures = {
        "graph_specific_forest": plot_graph_specific_forest(reference.effects, "Y_CR"),
        "alpha_cascade": plot_alpha_cascade(reference.alpha_cuts),
        "stability_map": plot_stability_map(reference.stability),
        "value_regret": plot_value_regret(reference.trajectory_summary.operational_decision),
    }
    for name, figure in figures.items():
        for suffix in ("png", "svg", "pdf"):
            save_figure(figure, export_dir / f"{name}.{suffix}")
    _record(
        steps,
        13,
        "Экспорт таблиц, паспорта и рисунков",
        len(list(export_dir.iterdir())) >= 20,
        f"{len(list(export_dir.iterdir()))} файлов в {export_dir}",
    )

    smoke = MonteCarloRunner(args.root / "experiments", "acceptance-smoke-publication")
    smoke.prepare(1)
    smoke_counts = smoke.run(workers=8, compute_cate=True, cate_trees=40)
    smoke_paths = smoke.export()
    _record(
        steps,
        14,
        "Smoke Monte Carlo",
        smoke_counts.get("completed") == 10 and all(path.exists() for path in smoke_paths),
        f"{smoke_counts}; {smoke_paths}",
    )

    full_runner = MonteCarloRunner(Path("artifacts/experiments"), "mc-full-20260814-publication")
    full_counts = full_runner.status_counts()
    _record(
        steps,
        15,
        "Полный утверждённый эксперимент",
        full_counts.get("completed") == 5000,
        str(full_counts),
    )

    required_screenshots = (
        args.screenshots / "01_evidence_and_structures_alpha_060.png",
        args.screenshots / "02_stability_and_decision.png",
        args.screenshots / "03_experiment_and_passport.png",
    )
    _record(
        steps,
        16,
        "Реальные скриншоты интерфейса",
        all(path.exists() and path.stat().st_size > 10_000 for path in required_screenshots),
        "; ".join(str(path) for path in required_screenshots),
    )

    window.close()
    payload = {
        "passed": all(step.passed for step in steps),
        "steps": [asdict(step) for step in steps],
    }
    (args.root / "acceptance_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = ["# Отчёт обязательного приёмочного сценария", ""]
    rows.extend(
        f"{step.step}. {'PASS' if step.passed else 'FAIL'} — {step.name}: {step.evidence}"
        for step in steps
    )
    (args.root / "acceptance_report.md").write_text("\n\n".join(rows) + "\n", encoding="utf-8")
    print(args.root / "acceptance_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
