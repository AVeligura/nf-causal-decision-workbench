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

    from engine.graphs import reference_graphs
    from study.data_pool import load_uci_pool

    pool = load_uci_pool()
    payload = {
        "application": APPLICATION_NAME,
        "version": APPLICATION_VERSION,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "data_rows": int(len(pool.raw)),
        "graphs": [graph.graph_id for graph in reference_graphs()],
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
