from __future__ import annotations

import argparse
import hashlib
import json
import struct
from datetime import UTC, datetime
from pathlib import Path


FORBIDDEN_SUFFIXES = {".py", ".pyc", ".pyo", ".ipynb"}
FORBIDDEN_PARTS = {"src", "tests", "scripts", "docs", ".git", ".idea", ".pytest_cache"}
EXPECTED_DATA_SHA256 = "afbfbed015d20f8421c32c62db37367c018eb6e92b00ea62a23354af8f84c44e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pe_x64(path: Path) -> None:
    with path.open("rb") as handle:
        if handle.read(2) != b"MZ":
            raise ValueError(f"{path.name}: отсутствует сигнатура MZ")
        handle.seek(0x3C)
        pe_offset = struct.unpack("<I", handle.read(4))[0]
        handle.seek(pe_offset)
        if handle.read(4) != b"PE\x00\x00":
            raise ValueError(f"{path.name}: отсутствует сигнатура PE")
        machine = struct.unpack("<H", handle.read(2))[0]
    if machine != 0x8664:
        raise ValueError(f"{path.name}: ожидалась архитектура AMD64, получено 0x{machine:04x}")


def verify(root: Path) -> dict[str, object]:
    root = root.resolve()
    executable = root / "NF_Causal_Workbench.exe"
    data = root / "data" / "raw" / "uci_polish" / "data.csv"
    config = root / "configs" / "reference.yaml"
    readme = root / "README_TESTERS_RU.md"
    for required in (executable, data, config, readme):
        if not required.is_file():
            raise FileNotFoundError(f"Отсутствует обязательный файл: {required}")

    verify_pe_x64(executable)
    if sha256(data) != EXPECTED_DATA_SHA256:
        raise ValueError("Контрольная сумма UCI data.csv не совпадает с принятой")

    files = [path for path in root.rglob("*") if path.is_file()]
    forbidden = []
    for path in files:
        relative = path.relative_to(root)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or FORBIDDEN_PARTS & set(relative.parts):
            forbidden.append(str(relative))
    if forbidden:
        raise ValueError("В поставку попали исходники или служебные файлы: " + ", ".join(forbidden))

    return {
        "status": "PASS",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "root": str(root),
        "executable_sha256": sha256(executable),
        "data_sha256": sha256(data),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "source_scan": "no .py/.pyc/.pyo/.ipynb and no source/test/script directories",
        "pe_architecture": "AMD64",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Проверка Windows-поставки для тестирования")
    parser.add_argument("distribution", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = verify(args.distribution)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
