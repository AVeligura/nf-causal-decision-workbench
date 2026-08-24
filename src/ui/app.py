from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# The work container lacks a system EGL package.  A project-local copy is used
# for QA only; normal Windows installations use Qt's platform libraries.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_NATIVE = _PROJECT_ROOT / ".native" / "lib" / "usr" / "lib" / "x86_64-linux-gnu"
if _LOCAL_NATIVE.exists():
    os.environ["LD_LIBRARY_PATH"] = (
        str(_LOCAL_NATIVE) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    )

from PySide6.QtWidgets import QApplication  # noqa: E402

from .main_window import MainWindow  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NF-Causal Decision Workbench")
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd() / "artifacts" / "repository",
        help="Каталог локального RunRepository",
    )
    parser.add_argument("--offscreen", action="store_true", help="Запуск Qt с offscreen backend")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    application = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    application.setApplicationName("NF-Causal Decision Workbench")
    window = MainWindow(args.repository)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
