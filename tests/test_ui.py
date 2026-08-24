from __future__ import annotations

from domain import GeneratorConfig
from engine import run_analysis
from study import generate_dataset
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
