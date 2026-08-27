from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path


APPLICATION_NAME = "NF-Causal Decision Workbench"
APPLICATION_VERSION = "3.1.1"


def application_root() -> Path:
    if "__compiled__" in globals() or getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def configure_portable_paths(root: Path) -> None:
    os.chdir(root)
    cache = root / "artifacts" / "matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))

    from study import data_pool

    data_path = root / "data" / "raw" / "uci_polish" / "data.csv"
    data_pool._default_data_path = lambda: data_path
    data_pool.load_uci_pool.cache_clear()


def smoke_check(root: Path) -> int:
    # The smoke check must exercise the real Qt window without requiring a
    # desktop session. Set the platform before importing Qt widgets/UI code.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import networkx
    import numpy
    import pandas
    import pyarrow
    import pydantic
    import PySide6
    import scipy
    import sklearn
    import statsmodels
    import torch
    from PySide6.QtWidgets import QApplication

    from engine.graphs import reference_graphs
    from study.data_pool import load_uci_pool
    from ui.main_window import MainWindow

    pool = load_uci_pool()

    application = QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QApplication([APPLICATION_NAME, "--smoke-check"])
    application.setApplicationName(APPLICATION_NAME)

    smoke_repository = root / "artifacts" / "smoke_repository"
    smoke_repository.mkdir(parents=True, exist_ok=True)

    window = MainWindow(smoke_repository)
    window.show()
    application.processEvents()

    if window.windowTitle() != APPLICATION_NAME:
        raise RuntimeError(f"Unexpected main window title: {window.windowTitle()!r}")
    if window.pages.count() != 5:
        raise RuntimeError(f"Unexpected workspace count: {window.pages.count()}")
    if not hasattr(window.experiment_workspace, "run_id"):
        raise RuntimeError("Experiment/passport workspace did not initialize run_id")

    gui_workspace_count = int(window.pages.count())
    window.close()
    application.processEvents()
    if owns_application:
        application.quit()

    payload = {
        "application": APPLICATION_NAME,
        "version": APPLICATION_VERSION,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "data_rows": int(len(pool.raw)),
        "graphs": [graph.graph_id for graph in reference_graphs()],
        "gui_main_window": "PASS",
        "gui_workspace_count": gui_workspace_count,
        "packages": {
            "PySide6": PySide6.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "networkx": networkx.__version__,
            "pyarrow": pyarrow.__version__,
            "pydantic": pydantic.__version__,
            "torch": torch.__version__,
        },
        "status": "PASS",
    }
    (root / "smoke_check.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def run() -> int:
    root = application_root()
    try:
        configure_portable_paths(root)
        if "--smoke-check" in sys.argv:
            return smoke_check(root)
        from ui.app import main

        return main()
    except Exception:
        logs = root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "startup_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
