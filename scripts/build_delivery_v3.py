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
                    archive.write(path, Path("nf_causal_workbench_v3") / path.relative_to(project))


def _write_directory_zip(source: Path, target: Path, root_name: str | None = None) -> None:
    root_name = root_name or source.name
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, Path(root_name) / path.relative_to(source))


def _copy_reference(project: Path, target: Path) -> None:
    target.mkdir(parents=True)
    repository = project / "artifacts" / "reference_repository_v3"
    for run_id in ("run-2e364b750292", "run-8a64cc06bac7"):
        shutil.copytree(repository / "runs" / run_id, target / "runs" / run_id)
    shutil.copy2(repository / "REFERENCE_REPLAY_V3.md", target / "REFERENCE_REPLAY_V3.md")
    shutil.copy2(repository / "REFERENCE_REPLAY_V3.json", target / "REFERENCE_REPLAY_V3.json")


def _copy_acceptance(project: Path, target: Path) -> None:
    target.mkdir(parents=True)
    for source in (
        project / "artifacts" / "acceptance_v3",
        project / "artifacts" / "test_results_v3",
        project / "artifacts" / "gui_acceptance_v3",
        project / "artifacts" / "control_examples",
    ):
        shutil.copytree(source, target / source.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/delivery_v3_20260820"),
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)

    source_zip = output / "nf_causal_workbench_v3_source.zip"
    r3_zip = output / "smoke_run_R3_444.zip"
    reference_zip = output / "reference_and_replay_v3.zip"
    acceptance_zip = output / "acceptance_v3.zip"
    _write_source_zip(project, source_zip)
    _write_directory_zip(project / "artifacts" / "experiments" / "r3-v3", r3_zip)

    reference_staging = output / "reference_and_replay_v3"
    _copy_reference(project, reference_staging)
    _write_directory_zip(reference_staging, reference_zip)
    shutil.rmtree(reference_staging)

    acceptance_staging = output / "acceptance_v3"
    _copy_acceptance(project, acceptance_staging)
    _write_directory_zip(acceptance_staging, acceptance_zip)
    shutil.rmtree(acceptance_staging)

    documents = (
        "HANDOFF_REPORT_V3.md",
        "PREREGISTRATION_V3.md",
        "DGP_V3.md",
        "VALUE_MODEL_V3.md",
        "R3_RESULTS_V3.md",
        "KNOWN_LIMITATIONS_V3.md",
    )
    for name in documents:
        shutil.copy2(project / "docs" / name, output / name)
    shutil.copy2(project / "CHANGELOG.md", output / "CHANGELOG.md")
    shutil.copy2(project / "KNOWN_LIMITATIONS.md", output / "KNOWN_LIMITATIONS.md")

    manifest = {
        "version": "0.3.0",
        "date": "2026-08-20",
        "v2_overwritten": False,
        "source_run_id": "run-2e364b750292",
        "replay_run_id": "run-8a64cc06bac7",
        "r3_experiment_id": "r3-v3",
        "r3_completed": 444,
        "r3_failed": 0,
        "r3_cate_enabled": False,
        "full_experiment_5000_run": False,
        "non_gui_tests": {"passed": 67, "failed": 0},
        "mypy": "PASS",
        "ruff": "PASS",
        "gui": {
            "status": "BLOCKED",
            "counted_as_pass": False,
            "reason": "missing libEGL.so.1",
        },
    }
    (output / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    component_targets = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    (output / "SHA256SUMS.txt").write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in component_targets) + "\n",
        encoding="utf-8",
    )

    delivery_zip = output / "nf_causal_workbench_v3_delivery.zip"
    with zipfile.ZipFile(delivery_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path != delivery_zip:
                archive.write(path, Path("nf_causal_workbench_v3_delivery") / path.name)
    delivery_sha = _sha256(delivery_zip)
    (output / "DELIVERY_SHA256.txt").write_text(
        f"{delivery_sha}  {delivery_zip.name}\n", encoding="utf-8"
    )

    print(output)
    for path in sorted(output.iterdir()):
        if path.is_file():
            print(f"{path.name}\t{path.stat().st_size}\t{_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
