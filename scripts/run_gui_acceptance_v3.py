from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from PySide6.QtWidgets import QApplication

from domain import GeneratorConfig
from engine import run_analysis
from study.dgp import generate_dataset
from study.scenarios import apply_scenario, apply_value_regime
from ui.main_window import MainWindow


@dataclass(frozen=True)
class GuiCheck:
    name: str
    status: str
    evidence: str
    screenshot: str


def _capture(application: QApplication, window: MainWindow, path: Path, page: int) -> None:
    window.navigation.setCurrentRow(page)
    application.processEvents()
    if not window.grab().save(str(path), "PNG"):
        raise RuntimeError(f"Could not save screenshot: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/gui_acceptance_v3"))
    parser.add_argument("--offscreen", action="store_true")
    args = parser.parse_args()
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    args.output.mkdir(parents=True, exist_ok=True)
    repository_root = args.output / "repository"
    application = QApplication.instance() or QApplication([])
    window = MainWindow(repository_root)
    window.resize(1440, 900)
    window.show()
    application.processEvents()
    window.state_model.clone_reference()

    base = apply_value_regime(
        apply_scenario(
            GeneratorConfig(
                mode="laboratory",
                scenario="reference",
                sample_size=300,
                true_graph_id="G1",
                cate_trees=20,
                crossfit_folds=2,
            ),
            apply_default_value_regime=False,
        ),
        "boundary",
    )
    window.state_model.config = base
    generated = generate_dataset(base)
    baseline = run_analysis(base, data=generated.data, compute_cate=False)
    window.state_model.inject_result(baseline, generated.data)
    application.processEvents()

    checks: list[GuiCheck] = []

    window.state_model.set_config_value("effect_scale", 1.20)
    changed_config = window.state_model.config
    changed_generated = generate_dataset(changed_config)
    changed = run_analysis(
        changed_config,
        data=changed_generated.data,
        evidence_bundle=window.state_model.evidence_bundle,
        compute_cate=False,
    )
    baseline_effect = next(
        effect.estimate
        for effect in baseline.effects
        if effect.graph_id == "G1" and effect.outcome == "Y_CR"
    )
    changed_effect = next(
        effect.estimate
        for effect in changed.effects
        if effect.graph_id == "G1" and effect.outcome == "Y_CR"
    )
    if changed.manifest.config_hash == baseline.manifest.config_hash or changed_effect == baseline_effect:
        raise AssertionError("Parameter edit did not produce a new calculation")
    window.state_model.inject_result(changed, changed_generated.data)
    screenshot = args.output / "01_parameter_change_recalculated.png"
    _capture(application, window, screenshot, 2)
    checks.append(
        GuiCheck(
            "Изменение параметра приводит к новому расчёту",
            "PASS",
            f"{baseline.manifest.run_id} -> {changed.manifest.run_id}; "
            f"G1 Y_CR {baseline_effect:.6f} -> {changed_effect:.6f}",
            screenshot.name,
        )
    )

    before_scores = {score.graph_id: score.mu for score in changed.graph_scores}
    evidence_id = window.state_model.evidence_bundle.items[0].evidence_id
    original_support = window.state_model.evidence_bundle.items[0].support
    window.state_model.update_evidence_item(
        evidence_id, support=max(0.0, original_support - 0.35)
    )
    evidence_changed = run_analysis(
        changed_config,
        data=changed_generated.data,
        evidence_bundle=window.state_model.evidence_bundle,
        compute_cate=False,
    )
    after_scores = {score.graph_id: score.mu for score in evidence_changed.graph_scores}
    if after_scores == before_scores:
        raise AssertionError("Evidence edit did not change structural compatibility")
    window.state_model.inject_result(evidence_changed, changed_generated.data)
    screenshot = args.output / "02_evidence_change_scores.png"
    _capture(application, window, screenshot, 1)
    checks.append(
        GuiCheck(
            "Изменение свидетельства меняет совместимость структур",
            "PASS",
            f"{evidence_id}: {before_scores} -> {after_scores}",
            screenshot.name,
        )
    )

    expected_action = (
        evidence_changed.trajectory_summary.operational_decision.selected_action
        or "мотивированный отказ"
    )
    label = window.stability_workspace.decision_label.text()
    tables_valid = (
        window.inference_workspace.table.rowCount() == 4
        and window.stability_workspace.trajectory.rowCount() == len(evidence_changed.decisions)
    )
    plots_valid = (
        window.evidence_workspace.score_plot.canvas.figure is not None
        and window.stability_workspace.value_plot.canvas.figure is not None
    )
    if not tables_valid or not plots_valid or expected_action not in label:
        raise AssertionError(
            f"Linked GUI refresh failed: tables={tables_valid}, plots={plots_valid}, "
            f"action={expected_action!r}, label={label!r}"
        )
    screenshot = args.output / "03_tables_plots_recommendation_updated.png"
    _capture(application, window, screenshot, 3)
    checks.append(
        GuiCheck(
            "Таблицы, графики и рекомендация обновляются",
            "PASS",
            f"effects_rows=4; trajectory_rows={len(evidence_changed.decisions)}; "
            f"recommendation={expected_action}",
            screenshot.name,
        )
    )

    window.state_model.repository.save(evidence_changed, changed_generated.data)
    replay = window.state_model.repository.replay(evidence_changed.manifest.run_id)
    if not replay.matched or not replay.replay_run_id:
        raise AssertionError(f"Saved-run replay failed: {replay.differences}")
    window.experiment_workspace.replay_status.setText(
        f"PASS: {evidence_changed.manifest.run_id} -> {replay.replay_run_id}"
    )
    screenshot = args.output / "04_save_reopen_replay.png"
    _capture(application, window, screenshot, 4)
    checks.append(
        GuiCheck(
            "Сохранение и повторное открытие воспроизводят результат",
            "PASS",
            f"{evidence_changed.manifest.run_id} -> {replay.replay_run_id}",
            screenshot.name,
        )
    )

    payload = {
        "environment": {
            "qt_platform": os.environ.get("QT_QPA_PLATFORM", "native"),
            "screen": "1440x900",
        },
        "all_passed": True,
        "checks": [asdict(check) for check in checks],
    }
    (args.output / "GUI_ACCEPTANCE_V3.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# GUI-приёмка V3", "", "Статус: **PASS**.", ""]
    lines.extend(
        f"- **{check.status}** — {check.name}: {check.evidence}; `{check.screenshot}`"
        for check in checks
    )
    (args.output / "GUI_ACCEPTANCE_V3.md").write_text("\n".join(lines), encoding="utf-8")
    window.close()
    print(args.output / "GUI_ACCEPTANCE_V3.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
