from __future__ import annotations

from PySide6.QtCore import Qt

from domain import GeneratorConfig
from engine import run_analysis
from study import generate_dataset
from ui.integration_contract import (
    FIXED_ESTIMAND_TEXT,
    FIXED_HORIZON_QUARTERS,
    FIXED_OUTCOME_TEXT,
    FIXED_VERSION_TEXT,
)
from ui.main_window import MainWindow


def test_window_navigation_and_reference_lock(qtbot, tmp_path):
    window = MainWindow(tmp_path / "repository")
    qtbot.addWidget(window)
    window.show()
    assert window.pages.count() == 5
    assert window.state_model.reference_locked
    sample_control = window.project_workspace.controls["sample_size"]
    assert not sample_control.spin.isEnabled()
    window.state_model.clone_reference()
    assert sample_control.spin.isEnabled()
    for page in range(5):
        window.navigation.setCurrentRow(page)
        assert window.pages.currentIndex() == page


def test_fixed_causal_contract_is_informational(qtbot, tmp_path):
    window = MainWindow(tmp_path / "repository")
    qtbot.addWidget(window)
    workspace = window.project_workspace

    assert not hasattr(window, "mode_combo")
    assert window.run_button.text() == "Выполнить анализ"

    assert workspace.outcome.count() == 1
    assert workspace.outcome.currentText() == FIXED_OUTCOME_TEXT
    assert not workspace.outcome.isEnabled()

    assert workspace.estimand.count() == 1
    assert workspace.estimand.currentText() == FIXED_ESTIMAND_TEXT
    assert not workspace.estimand.isEnabled()

    assert workspace.horizon.value() == FIXED_HORIZON_QUARTERS
    assert not workspace.horizon.isEnabled()

    assert workspace.version.count() == 1
    assert workspace.version.currentText() == FIXED_VERSION_TEXT
    assert not workspace.version.isEnabled()


def test_reference_evidence_is_immutable_and_lab_edits_only_active_fields(qtbot, tmp_path):
    window = MainWindow(tmp_path / "repository")
    qtbot.addWidget(window)
    table = window.evidence_workspace.evidence_table

    assertion = table.item(0, 1)
    support = table.item(0, 4)
    assert assertion is not None
    assert support is not None
    assert not bool(assertion.flags() & Qt.ItemFlag.ItemIsEditable)
    assert not bool(support.flags() & Qt.ItemFlag.ItemIsEditable)

    window.state_model.clone_reference()

    assertion = table.item(0, 1)
    support = table.item(0, 4)
    reliability = table.item(0, 5)
    applicability = table.item(0, 6)
    assert assertion is not None
    assert support is not None
    assert reliability is not None
    assert applicability is not None
    assert not bool(assertion.flags() & Qt.ItemFlag.ItemIsEditable)
    assert bool(support.flags() & Qt.ItemFlag.ItemIsEditable)
    assert bool(reliability.flags() & Qt.ItemFlag.ItemIsEditable)
    assert bool(applicability.flags() & Qt.ItemFlag.ItemIsEditable)


def test_stale_result_is_not_exposed_as_current(qtbot, tmp_path):
    generated = generate_dataset(GeneratorConfig(mode="laboratory", sample_size=300))
    result = run_analysis(generated.config, data=generated.data, compute_cate=False)
    window = MainWindow(tmp_path / "repository")
    qtbot.addWidget(window)
    window.state_model.inject_result(result, generated.data)

    assert window.export_button.isEnabled()
    assert window.replay_button.isEnabled()
    assert window.inference_workspace.isEnabled()
    assert window.stability_workspace.isEnabled()

    window.state_model.clone_reference()

    assert not window.export_button.isEnabled()
    assert not window.replay_button.isEnabled()
    assert not window.inference_workspace.isEnabled()
    assert not window.stability_workspace.isEnabled()
    assert "предыдущий" in window.run_label.text()
    assert "предыдущей конфигурации" in window.progress_text.text()
    assert all(
        "требуется новый запуск" in edit.text()
        for edit in window.stability_workspace.coefficient_edits.values()
    )


def test_execute_validates_before_start(qtbot, tmp_path, monkeypatch):
    window = MainWindow(tmp_path / "repository")
    qtbot.addWidget(window)
    called = {"execute": False}

    monkeypatch.setattr(window.state_model, "validate", lambda: (False, ("invalid",)))

    def fake_execute(*, compute_cate=True):
        del compute_cate
        called["execute"] = True

    monkeypatch.setattr(window.state_model, "execute", fake_execute)
    window._execute()

    assert called["execute"] is False
    assert "Запуск отменён" in window.progress_text.text()


def test_result_updates_linked_workspaces(qtbot, tmp_path):
    generated = generate_dataset(GeneratorConfig(mode="laboratory", sample_size=300))
    result = run_analysis(generated.config, data=generated.data, compute_cate=False)
    window = MainWindow(tmp_path / "repository")
    qtbot.addWidget(window)
    window.state_model.inject_result(result, generated.data)
    assert window.run_label.text() == result.manifest.run_id
    assert window.evidence_workspace.core_label.text().startswith("Γα")
    assert window.inference_workspace.table.rowCount() == 4
    assert window.stability_workspace.trajectory.rowCount() == len(result.decisions)
    assert window.experiment_workspace.run_id.text() == result.manifest.run_id


def test_cancel_changes_state(qtbot, tmp_path):
    window = MainWindow(tmp_path / "repository")
    qtbot.addWidget(window)
    window.state_model.state = "running"
    window.state_model.cancel()
    assert window.state_model.state == "cancelled"


def test_imported_data_switches_state_to_import(qtbot, tmp_path):
    generated = generate_dataset(GeneratorConfig(mode="laboratory", sample_size=300))
    window = MainWindow(tmp_path / "repository")
    qtbot.addWidget(window)
    window.state_model.load_imported_data(generated.data)
    assert window.state_model.config.mode == "import"
    assert not window.state_model.reference_locked
    assert window.project_workspace.preview.rowCount() == 80
