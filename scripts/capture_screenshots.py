from __future__ import annotations

import argparse
import os
from pathlib import Path

from PySide6.QtWidgets import QApplication

from domain import GeneratorConfig
from engine import run_analysis
from study import generate_dataset
from ui.main_window import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/screenshots"))
    parser.add_argument("--sample-size", type=int, default=1500)
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    args.output.mkdir(parents=True, exist_ok=True)
    config = GeneratorConfig(sample_size=args.sample_size)
    generated = generate_dataset(config)
    result = run_analysis(config, data=generated.data, compute_cate=True)
    application = QApplication.instance() or QApplication([])
    window = MainWindow(args.output / "repository")
    window.state_model.inject_result(result, generated.data)
    window.resize(1440, 900)
    window.show()
    application.processEvents()
    names = {
        1: "01_evidence_and_structures_alpha_060.png",
        3: "02_stability_and_decision.png",
        4: "03_experiment_and_passport.png",
    }
    for page, name in names.items():
        window.navigation.setCurrentRow(page)
        if page == 4:
            window.experiment_workspace.tabs.setCurrentIndex(2)
        application.processEvents()
        window.grab().save(str(args.output / name), "PNG")
    window.close()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
