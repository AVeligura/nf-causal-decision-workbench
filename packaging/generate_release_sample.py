from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain import GeneratorConfig  # noqa: E402
from study import generate_dataset  # noqa: E402

EXPECTED_SHA256 = "d17be3e87145ab69bd514e15b6a2e268ccaeee9a4ee8cede3e1a75d04dad7976"
COLUMNS = [
    "T",
    "V",
    "L",
    "D",
    "S",
    "Y_CR",
    "Y_CFO",
    "X1",
    "X2",
    "X4",
    "X5",
    "X21",
    "X27",
    "X29",
    "X44",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / "reference.yaml"
    config = GeneratorConfig.model_validate(yaml.safe_load(config_path.read_text("utf-8")))
    frame = generate_dataset(config).data[COLUMNS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Unexpected sample checksum: {digest}")
    print(f"Generated {len(frame)} rows; SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

