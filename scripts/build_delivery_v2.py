from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

SOURCE_TOP_LEVEL = (
    "CHANGELOG.md",
    "KNOWN_LIMITATIONS.md",
    "README.md",
    "configs",
    "data",
    "docs",
    "launcher.py",
    "pyproject.toml",
    "pysidedeploy.spec",
    "scripts",
    "src",
    "tests",
    "uv.lock",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _include_source(path: Path) -> bool:
    return not any(
        part in {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
        or part.endswith(".pyc")
        for part in path.parts
    )


def _write_source_zip(project: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in SOURCE_TOP_LEVEL:
            source = project / name
            paths = sorted(source.rglob("*")) if source.is_dir() else [source]
            for path in paths:
                if path.is_file() and _include_source(path.relative_to(project)):
                    archive.write(path, Path("nf_causal_workbench_v2") / path.relative_to(project))


def _write_directory_zip(source: Path, target: Path, root_name: str | None = None) -> None:
    root_name = root_name or source.name
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, Path(root_name) / path.relative_to(source))


def _copy_acceptance(project: Path, target: Path) -> None:
    target.mkdir(parents=True)
    copies = {
        project / "artifacts" / "acceptance_v2" / "acceptance_report_v2.md": target
        / "acceptance_report_v2.md",
        project / "artifacts" / "acceptance_v2" / "acceptance_report_v2.json": target
        / "acceptance_report_v2.json",
        project / "artifacts" / "control_examples" / "control_examples.json": target
        / "control_examples.json",
        project / "artifacts" / "non_ui_pytest_v2.xml": target / "non_ui_pytest_v2.xml",
        project / "artifacts" / "non_ui_pytest_v2.log": target / "non_ui_pytest_v2.log",
        project / "docs" / "TEST_REPORT_V2.md": target / "TEST_REPORT_V2.md",
        project / "docs" / "MISMATCH_REGISTER_V2.md": target / "MISMATCH_REGISTER_V2.md",
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/delivery_v2_20260817"),
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)

    source_zip = output / "nf_causal_workbench_v2_source.zip"
    smoke_zip = output / "mc_smoke_v2_20260817_r2.zip"
    reference_zip = output / "reference_and_replay_v2.zip"
    acceptance_zip = output / "acceptance_control_v2.zip"
    _write_source_zip(project, source_zip)
    _write_directory_zip(
        project / "artifacts" / "experiments" / "mc-smoke-v2-20260817-r2",
        smoke_zip,
    )
    _write_directory_zip(
        project / "artifacts" / "reference_repository_v2",
        reference_zip,
    )
    acceptance_staging = output / "acceptance_control_v2"
    _copy_acceptance(project, acceptance_staging)
    _write_directory_zip(acceptance_staging, acceptance_zip)

    shutil.copy2(project / "docs" / "HANDOFF_REPORT_V2.md", output / "HANDOFF_REPORT_V2.md")
    shutil.copy2(project / "CHANGELOG.md", output / "CHANGELOG.md")
    shutil.copy2(project / "KNOWN_LIMITATIONS.md", output / "KNOWN_LIMITATIONS.md")
    manifest = {
        "version": "0.2.0-control-stage",
        "date": "2026-08-17",
        "source_run_id": "run-8adee0cc5217",
        "replay_run_id": "run-6c3eb54f49e9",
        "smoke_experiment_id": "mc-smoke-v2-20260817-r2",
        "smoke_replications": 100,
        "full_monte_carlo_run": False,
        "non_ui_tests": {"passed": 56, "failed": 0},
        "ui_tests": {"status": "blocked", "reason": "missing libEGL.so.1"},
        "coordinator_decision_required": "true-graph mechanism before full Monte Carlo",
    }
    (output / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    shutil.rmtree(acceptance_staging)
    checksum_targets = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    internal_sums = "\n".join(
        f"{_sha256(path)}  {path.name}" for path in checksum_targets
    ) + "\n"
    (output / "SHA256SUMS.txt").write_text(internal_sums, encoding="utf-8")

    delivery_zip = output / "nf_causal_workbench_v2_delivery.zip"
    with zipfile.ZipFile(delivery_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path != delivery_zip:
                archive.write(path, Path("nf_causal_workbench_v2_delivery") / path.name)

    final_targets = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    final_sums = "\n".join(f"{_sha256(path)}  {path.name}" for path in final_targets) + "\n"
    (output / "SHA256SUMS.txt").write_text(final_sums, encoding="utf-8")
    print(output)
    for path in sorted(output.iterdir()):
        if path.is_file():
            print(f"{path.name}\t{path.stat().st_size}\t{_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
