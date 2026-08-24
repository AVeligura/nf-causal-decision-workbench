from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/365/data.csv"
EXPECTED_SHA256 = "afbfbed015d20f8421c32c62db37367c018eb6e92b00ea62a23354af8f84c44e"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw" / "uci_polish" / "data.csv",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Unexpected UCI checksum: {digest}")
    args.output.write_bytes(payload)
    print(f"Saved {len(payload)} bytes to {args.output}; SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
