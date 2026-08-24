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
        part
        in {
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            ".venv",
        }
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
                    archive.write(
                        path,
                        Path("nf_causal_workbench_v3_1") / path.relative_to(project),
                    )


def _write_directory_zip(source: Path, target: Path, root_name: str) -> None:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, Path(root_name) / path.relative_to(source))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts_v3_1/delivery_v3_1_20260820"),
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)

    source_zip = output / "nf_causal_workbench_v3_1_source.zip"
    r3_zip = output / "smoke_run_R3_1_444.zip"
    reference_zip = output / "reference_and_replay_v3_1.zip"
    acceptance_zip = output / "acceptance_v3_1.zip"
    _write_source_zip(project, source_zip)
    _write_directory_zip(
        project / "artifacts_v3_1" / "experiments" / "r3-1-20260820",
        r3_zip,
        "smoke_run_R3_1_444",
    )
    _write_directory_zip(
        project / "artifacts_v3_1" / "reference_repository_v3_1",
        reference_zip,
        "reference_and_replay_v3_1",
    )

    acceptance_staging = output / "acceptance_v3_1"
    acceptance_staging.mkdir()
    shutil.copytree(
        project / "artifacts_v3_1" / "final",
        acceptance_staging / "test_and_tool_logs",
    )
    shutil.copytree(
        project / "artifacts_v3_1" / "gui_acceptance_v3_1",
        acceptance_staging / "gui_acceptance_v3_1",
    )
    _write_directory_zip(acceptance_staging, acceptance_zip, "acceptance_v3_1")
    shutil.rmtree(acceptance_staging)

    documents = (
        "HANDOFF_REPORT_V3_1.md",
        "PREREGISTRATION_V3_1.md",
        "DGP_V3.md",
        "VALUE_MODEL_V3_1.md",
        "FORMAL_ROBUST_EVI_V3_1.md",
        "ABSTAIN_SEMANTICS_V3_1.md",
        "R3_1_RESULTS_V3_1.md",
        "KNOWN_LIMITATIONS_V3_1.md",
    )
    for name in documents:
        shutil.copy2(project / "docs" / name, output / name)
    shutil.copy2(project / "CHANGELOG.md", output / "CHANGELOG.md")
    shutil.copy2(project / "KNOWN_LIMITATIONS.md", output / "KNOWN_LIMITATIONS.md")

    manifest = {
        "version": "0.3.1",
        "date": "2026-08-20",
        "v2_v3_r3_overwritten": False,
        "source_run_id": "run-f247d0fcaf86",
        "replay_run_id": "run-4368fbb23b7a",
        "reference_replay_matched": True,
        "r3_1_experiment_id": "r3-1-20260820",
        "r3_1_completed": 444,
        "r3_1_failed": 0,
        "r3_1_result_rows": 2664,
        "r3_1_cate_enabled": False,
        "full_experiment_5000_run": False,
        "non_gui_tests": {"passed": 78, "failed": 0},
        "mypy": {"status": "PASS", "files": 31},
        "ruff": "PASS",
        "gui": {
            "status": "BLOCKED",
            "counted_as_pass": False,
            "reason": "missing libEGL.so.1",
            "runner_uses_qttest_widgets": True,
            "runner_uses_inject_result": False,
        },
    }
    (output / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    components = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    (output / "SHA256SUMS.txt").write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in components) + "\n",
        encoding="utf-8",
    )

    delivery_zip = output / "nf_causal_workbench_v3_1_delivery.zip"
    with zipfile.ZipFile(delivery_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path != delivery_zip:
                archive.write(
                    path, Path("nf_causal_workbench_v3_1_delivery") / path.name
                )
    (output / "DELIVERY_SHA256.txt").write_text(
        f"{_sha256(delivery_zip)}  {delivery_zip.name}\n", encoding="utf-8"
    )

    print(output)
    for path in sorted(output.iterdir()):
        if path.is_file():
            print(f"{path.name}\t{path.stat().st_size}\t{_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
