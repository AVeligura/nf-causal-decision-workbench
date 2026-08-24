from __future__ import annotations

import json
from pathlib import Path

from study.control_examples import run_control_examples


def main() -> None:
    rows = run_control_examples()
    output = Path("artifacts/control_examples/control_examples.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [
        row
        for row in rows
        if any(row["actual"].get(key) != value for key, value in row["expected"].items())
    ]
    if failed:
        raise SystemExit(f"Control examples failed: {failed}")
    print(output)


if __name__ == "__main__":
    main()
