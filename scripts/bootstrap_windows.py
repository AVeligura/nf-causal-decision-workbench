from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import venv
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

INSTALLER_VERSION = "1.0.0"
REQUIRED_PYTHON = (3, 12)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"
STAMP_FILE = VENV_DIR / "nf_causal_install.json"
LOCK_FILE = PROJECT_ROOT / ".nf_causal_installing"
LOG_FILE = PROJECT_ROOT / "installation.log"
STALE_LOCK_SECONDS = 2 * 60 * 60

REQUIRED_IMPORTS = (
    "PySide6",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statsmodels",
    "networkx",
    "matplotlib",
    "seaborn",
    "pyarrow",
    "openpyxl",
    "pydantic",
    "pydantic_settings",
    "yaml",
    "reportlab",
    "psutil",
    "joblib",
    "torch",
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def log(message: str) -> None:
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def installation_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(INSTALLER_VERSION.encode("utf-8"))
    for path in (
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "uv.lock",
        PROJECT_ROOT / "requirements-windows.lock",
        Path(__file__),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run_command(command: Sequence[str], *, cwd: Path = PROJECT_ROOT) -> None:
    rendered = command_text(command)
    log(f"Команда: {rendered}")
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def verify_python_version() -> None:
    if sys.version_info[:2] != REQUIRED_PYTHON:
        raise RuntimeError(
            "Для приложения требуется Python 3.12 x64. "
            f"Сейчас используется Python {sys.version_info.major}.{sys.version_info.minor}."
        )
    if sys.maxsize <= 2**32:
        raise RuntimeError("Требуется 64-разрядная версия Python 3.12.")


def existing_venv_is_compatible() -> bool:
    python = venv_python()
    if not python.exists():
        return False
    check = (
        "import sys; "
        "raise SystemExit(0 if sys.version_info[:2] == (3, 12) "
        "and sys.maxsize > 2**32 else 1)"
    )
    result = subprocess.run(
        [str(python), "-c", check],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def backup_incompatible_venv() -> None:
    if not VENV_DIR.exists() or existing_venv_is_compatible():
        return
    suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = PROJECT_ROOT / f".venv_incompatible_{suffix}"
    log(f"Неполное окружение будет сохранено как {backup.name}")
    VENV_DIR.replace(backup)


def create_venv_if_needed() -> None:
    backup_incompatible_venv()
    if venv_python().exists():
        return
    log("Создаю изолированное окружение .venv")
    venv.EnvBuilder(with_pip=True, clear=False).create(VENV_DIR)


def verify_imports() -> None:
    imports = "; ".join(f"import {name}" for name in REQUIRED_IMPORTS)
    run_command([str(venv_python()), "-c", imports])


def read_stamp() -> dict[str, object] | None:
    if not STAMP_FILE.exists():
        return None
    try:
        return json.loads(STAMP_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def stamp_is_current(fingerprint: str) -> bool:
    stamp = read_stamp()
    return bool(
        venv_python().exists()
        and stamp
        and stamp.get("fingerprint") == fingerprint
        and stamp.get("python") == "3.12"
    )


def write_stamp(fingerprint: str) -> None:
    payload = {
        "installer_version": INSTALLER_VERSION,
        "fingerprint": fingerprint,
        "python": "3.12",
        "installed_at_utc": now_iso(),
    }
    temporary = STAMP_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(STAMP_FILE)


@contextmanager
def installation_lock() -> Iterator[None]:
    if LOCK_FILE.exists():
        age = time.time() - LOCK_FILE.stat().st_mtime
        if age <= STALE_LOCK_SECONDS:
            raise RuntimeError("Установка уже выполняется в другом окне. Дождитесь её завершения.")
        log("Удаляю устаревшую блокировку предыдущей установки")
        LOCK_FILE.unlink()
    try:
        descriptor = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\nstarted={now_iso()}\n")
        yield
    finally:
        LOCK_FILE.unlink(missing_ok=True)


def install(*, repair: bool) -> None:
    fingerprint = installation_fingerprint()
    if not repair and stamp_is_current(fingerprint):
        log("Зависимости уже установлены; повторная загрузка не требуется")
        return

    with installation_lock():
        create_venv_if_needed()
        python = str(venv_python())
        log("Обновляю инструменты установки")
        run_command(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ]
        )
        log("Устанавливаю необходимые пакеты; первый запуск может занять продолжительное время")
        dependency_command = [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--prefer-binary",
        ]
        if repair:
            dependency_command.append("--upgrade")
        dependency_command.extend(["-r", str(PROJECT_ROOT / "requirements-windows.lock")])
        run_command(dependency_command)
        run_command(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "-e",
                str(PROJECT_ROOT),
            ]
        )
        log("Проверяю импорт обязательных компонентов")
        verify_imports()
        write_stamp(fingerprint)
        log("Установка успешно завершена")


def run_application(app_args: Sequence[str]) -> int:
    environment = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = src_path + os.pathsep + environment.get("PYTHONPATH", "")
    command = [str(venv_python()), "-m", "ui.app", *app_args]
    log(f"Запускаю приложение: {command_text(command)}")
    return subprocess.call(command, cwd=PROJECT_ROOT, env=environment)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Автоматическая локальная установка NF-Causal Decision Workbench"
    )
    parser.add_argument("--run", action="store_true", help="запустить приложение после установки")
    parser.add_argument(
        "--repair", action="store_true", help="повторно установить и обновить зависимости"
    )
    parser.add_argument("--dry-run", action="store_true", help="показать план без изменения файлов")
    parser.add_argument("app_args", nargs=argparse.REMAINDER, help="аргументы приложения после --")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app_args = list(args.app_args)
    if app_args[:1] == ["--"]:
        app_args = app_args[1:]
    try:
        verify_python_version()
        if args.dry_run:
            log(f"Проект: {PROJECT_ROOT}")
            log(f"Окружение: {VENV_DIR}")
            log(
                "План: создать .venv, установить зафиксированные зависимости, "
                "подключить приложение, проверить импорты"
            )
            if args.run:
                log(f"Затем запустить ui.app с аргументами: {app_args}")
            return 0
        install(repair=args.repair)
        if args.run:
            return run_application(app_args)
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        log(f"ОШИБКА: {error}")
        log(f"Подробности сохранены в {LOG_FILE}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
