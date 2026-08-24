from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import PySide6
from PySide6.QtCore import QLibraryInfo, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QTableWidget

from ui.main_window import MainWindow


@dataclass(frozen=True)
class GuiCheck:
    name: str
    status: str
    before: dict[str, Any]
    after: dict[str, Any]
    screenshot: str


def _capture(application: QApplication, window: MainWindow, path: Path, page: int) -> None:
    window.navigation.setCurrentRow(page)
    application.processEvents()
    QTest.qWait(200)
    if not window.grab().save(str(path), "PNG"):
        raise RuntimeError(f"Could not save screenshot: {path}")


def _enter_spin_value(spin: QAbstractSpinBox, value: str) -> None:
    spin.setFocus()
    spin.selectAll()
    QTest.keyClicks(spin, value)
    QTest.keyClick(spin, Qt.Key.Key_Return)


def _execute_and_wait(window: MainWindow, timeout_ms: int):
    previous_run_id = (
        window.state_model.result.manifest.run_id if window.state_model.result else None
    )
    spy = QSignalSpy(window.state_model.result_changed)
    QTest.mouseClick(window.run_button, Qt.MouseButton.LeftButton)
    if not spy.wait(timeout_ms):
        raise TimeoutError(
            f"GUI calculation did not finish in {timeout_ms} ms; state={window.state_model.state}"
        )
    result = window.state_model.result
    if result is None:
        raise AssertionError("result_changed fired without an AnalysisResult")
    if result.manifest.run_id == previous_run_id:
        raise AssertionError("GUI calculation did not create a new run ID")
    return result


def _edit_table_cell(table: QTableWidget, row: int, column: int, value: str) -> None:
    item = table.item(row, column)
    if item is None:
        raise AssertionError(f"Missing evidence cell ({row}, {column})")
    table.scrollToItem(item)
    rect = table.visualItemRect(item)
    QTest.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    QTest.keyClick(table, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(table, value)
    QTest.keyClick(table, Qt.Key.Key_Return)


def _effect(result, graph_id: str = "G1", outcome: str = "Y_CR") -> float | None:
    estimate = next(
        effect.estimate
        for effect in result.effects
        if effect.graph_id == graph_id and effect.outcome == outcome
    )
    return None if estimate is None else float(estimate)


def _scores(result) -> dict[str, float]:
    return {score.graph_id: score.mu for score in result.graph_scores}


def _snapshot(result) -> dict[str, Any]:
    return {
        "run_id": result.manifest.run_id,
        "config_hash": result.manifest.config_hash,
        "G1_Y_CR": _effect(result),
        "graph_scores": _scores(result),
        "decision_status": result.trajectory_summary.operational_decision.status,
        "recommended_action": (result.trajectory_summary.operational_decision.selected_action),
        "operational_action": (
            result.trajectory_summary.operational_decision.selected_action or "a0"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/gui_acceptance_v3_1"))
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=300_000)
    args = parser.parse_args()
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    args.output.mkdir(parents=True, exist_ok=True)

    existing_application = QApplication.instance()
    application = (
        existing_application if isinstance(existing_application, QApplication) else QApplication([])
    )
    window = MainWindow(args.output / "repository")
    window.resize(1440, 900)
    window.show()
    application.processEvents()

    checks: list[GuiCheck] = []

    # 1. The protected reference is cloned through the visible button.
    window.project_workspace.tabs.setCurrentIndex(1)
    QTest.mouseClick(window.project_workspace.clone_button, Qt.MouseButton.LeftButton)
    if window.state_model.reference_locked:
        raise AssertionError("Clone button did not create an editable laboratory copy")

    # Keep the acceptance run bounded, using the actual integer widget.
    sample_spin = window.project_workspace.controls["sample_size"].spin
    _enter_spin_value(sample_spin, "300")
    baseline = _execute_and_wait(window, args.timeout_ms)
    baseline_snapshot = _snapshot(baseline)

    # 2-5. Edit a QDoubleSpinBox, click Execute, wait for the worker, and
    # verify both the new run ID and a changed result.
    effect_spin = window.project_workspace.controls["effect_scale"].spin
    _enter_spin_value(effect_spin, "1.20")
    parameter_result = _execute_and_wait(window, args.timeout_ms)
    parameter_snapshot = _snapshot(parameter_result)
    if parameter_snapshot["config_hash"] == baseline_snapshot["config_hash"]:
        raise AssertionError("QDoubleSpinBox edit did not change config_hash")
    if parameter_snapshot["G1_Y_CR"] == baseline_snapshot["G1_Y_CR"]:
        raise AssertionError("QDoubleSpinBox edit did not change G1 Y_CR")
    screenshot = args.output / "01_parameter_widget_recalculated.png"
    _capture(application, window, screenshot, 2)
    checks.append(
        GuiCheck(
            "Parameter widget creates a new calculation",
            "PASS",
            baseline_snapshot,
            parameter_snapshot,
            screenshot.name,
        )
    )

    # 6-9. Edit the evidence table itself, execute again, and verify the
    # structural memberships plus linked tables, plots, and recommendation.
    window.navigation.setCurrentRow(1)
    table = window.evidence_workspace.evidence_table
    evidence_id_item = table.item(0, 0)
    support_item = table.item(0, 4)
    if evidence_id_item is None or support_item is None:
        raise AssertionError("Evidence table did not expose its first editable row")
    evidence_id = evidence_id_item.text()
    old_support = float(support_item.text())
    new_support = max(0.0, old_support - 0.35)
    _edit_table_cell(table, 0, 4, f"{new_support:.3f}")
    if window.state_model.evidence_bundle.version != "custom":
        raise AssertionError("Evidence table edit did not update the evidence bundle")
    evidence_result = _execute_and_wait(window, args.timeout_ms)
    evidence_snapshot = _snapshot(evidence_result)
    if evidence_snapshot["graph_scores"] == parameter_snapshot["graph_scores"]:
        raise AssertionError("Evidence table edit did not change mu for structures")
    expected_action = evidence_snapshot["recommended_action"] or "мотивированный отказ"
    label = window.stability_workspace.decision_label.text()
    tables_valid = (
        window.inference_workspace.table.rowCount() == 4
        and window.stability_workspace.trajectory.rowCount() == len(evidence_result.decisions)
    )
    score_canvas = window.evidence_workspace.score_plot.canvas
    value_canvas = window.stability_workspace.value_plot.canvas
    plots_valid = (
        score_canvas is not None
        and value_canvas is not None
        and score_canvas.figure is not None
        and value_canvas.figure is not None
    )
    if not tables_valid or not plots_valid or expected_action not in label:
        raise AssertionError(
            f"Linked refresh failed: tables={tables_valid}, plots={plots_valid}, "
            f"action={expected_action!r}, label={label!r}"
        )
    screenshot = args.output / "02_evidence_widget_scores.png"
    _capture(application, window, screenshot, 1)
    checks.append(
        GuiCheck(
            "Evidence cell changes memberships and linked views",
            "PASS",
            {
                **parameter_snapshot,
                "evidence_id": evidence_id,
                "support": old_support,
            },
            {
                **evidence_snapshot,
                "evidence_id": evidence_id,
                "support": new_support,
                "tables_valid": tables_valid,
                "plots_valid": plots_valid,
            },
            screenshot.name,
        )
    )
    screenshot = args.output / "03_tables_plots_recommendation.png"
    _capture(application, window, screenshot, 3)

    # 10. AnalysisWorker already saved the run. Close the first application
    # window, create a fresh one on the same repository, select the stored run
    # ID from the visible history control, and replay through the GUI button.
    saved_run_id = evidence_result.manifest.run_id
    repository_root = args.output / "repository"
    window.close()
    window.deleteLater()
    application.processEvents()
    QTest.qWait(200)

    reopened_window = MainWindow(repository_root)
    reopened_window.resize(1440, 900)
    reopened_window.show()
    application.processEvents()
    reopened_window.navigation.setCurrentRow(4)
    workspace = reopened_window.experiment_workspace
    workspace.tabs.setCurrentIndex(2)
    stored_index = workspace.saved_runs.findText(saved_run_id)
    if stored_index < 0:
        raise AssertionError(f"Saved run ID is absent from the reopened GUI: {saved_run_id}")
    workspace.saved_runs.setCurrentIndex(stored_index)
    if workspace.run_id.text() != saved_run_id:
        raise AssertionError("Selecting the saved run did not populate the replay field")
    QTest.mouseClick(workspace.replay_run_button, Qt.MouseButton.LeftButton)
    application.processEvents()
    replay_text = workspace.replay_status.text()
    if "Replay совпал" not in replay_text:
        raise AssertionError(f"Replay through GUI did not match: {replay_text}")
    screenshot = args.output / "04_save_reopen_replay.png"
    _capture(application, reopened_window, screenshot, 4)
    checks.append(
        GuiCheck(
            "Saved run replays through GUI controls",
            "PASS",
            {"run_id": saved_run_id, "first_window_closed": True},
            {
                "run_id_found_in_reopened_gui": True,
                "replay_status": replay_text,
            },
            screenshot.name,
        )
    )

    payload: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "pyside6": PySide6.__version__,
            "qt": QLibraryInfo.version().toString(),
            "platform": platform.platform(),
            "qt_platform": os.environ.get("QT_QPA_PLATFORM", "native"),
            "screen": "1440x900",
        },
        "all_passed": True,
        "checks": [asdict(check) for check in checks],
    }
    json_path = args.output / "GUI_ACCEPTANCE_V3_1_1.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# GUI acceptance V3.1.1",
        "",
        "Status: **PASS**.",
        "",
        f"Python {payload['environment']['python']}; PySide6 "
        f"{payload['environment']['pyside6']}; Qt {payload['environment']['qt']}; "
        f"{payload['environment']['platform']}.",
        "",
    ]
    lines.extend(f"- **{check.status}** — {check.name}; `{check.screenshot}`" for check in checks)
    markdown_path = args.output / "GUI_ACCEPTANCE_V3_1_1.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    reopened_window.close()
    application.processEvents()
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
