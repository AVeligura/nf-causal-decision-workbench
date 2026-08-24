from __future__ import annotations

import hashlib
import json
import platform
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from domain import AnalysisResult, EvidenceBundle, GeneratorConfig
from engine.pipeline import run_analysis

from .import_export import ExportManager


@dataclass(frozen=True)
class ReplayResult:
    source_run_id: str
    replay_run_id: str | None
    matched: bool
    differences: tuple[str, ...]
    tolerance: float


class RunRepository:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        self.db_path = self.root / "runs.sqlite3"
        self._initialize_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    run_path TEXT NOT NULL,
                    replay_of TEXT
                )
                """
            )

    def run_path(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def refresh_checksums(self, run_id: str) -> Path:
        run_dir = self.run_path(run_id)
        checksum_rows = []
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.name != "checksums.sha256":
                checksum_rows.append(
                    f"{self._sha256(path)}  {path.relative_to(run_dir).as_posix()}"
                )
        checksum_file = run_dir / "checksums.sha256"
        checksum_file.write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")
        return checksum_file

    def attach_json(self, run_id: str, relative_path: str | Path, payload: Any) -> Path:
        target = self.run_path(run_id) / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        self.refresh_checksums(run_id)
        return target

    def save(
        self, result: AnalysisResult, data: pd.DataFrame, *, seeds: dict[str, int] | None = None
    ) -> Path:
        run_dir = self.run_path(result.manifest.run_id)
        for name in (
            "inputs",
            "evidence",
            "results",
            "metrics",
            "passport",
            "figures",
            "logs",
        ):
            (run_dir / name).mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(
            result.manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(
                result.config.model_dump(mode="json"), allow_unicode=True, sort_keys=False
            ),
            encoding="utf-8",
        )
        environment = dict(result.manifest.environment)
        environment.update({"machine": platform.machine(), "processor": platform.processor()})
        (run_dir / "environment.json").write_text(
            json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "seeds.json").write_text(
            json.dumps(seeds or {"master": result.config.seed}, indent=2), encoding="utf-8"
        )
        stored_data = data.copy(deep=False)
        stored_data.attrs = {}
        stored_data.to_parquet(run_dir / "inputs" / "data.parquet", index=False)
        (run_dir / "evidence" / "evidence.json").write_text(
            json.dumps(result.passport.evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "results" / "analysis.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        (run_dir / "passport" / "CausalDecisionPassport.json").write_text(
            result.passport.model_dump_json(indent=2), encoding="utf-8"
        )
        ExportManager.passport_pdf(
            result.passport, run_dir / "passport" / "CausalDecisionPassport.pdf"
        )
        (run_dir / "logs" / "run.log").write_text(
            f"{datetime.now(UTC).isoformat()} completed {result.manifest.run_id}\n",
            encoding="utf-8",
        )
        self.refresh_checksums(result.manifest.run_id)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    result.manifest.run_id,
                    result.manifest.project_id,
                    result.manifest.state,
                    result.config.scenario,
                    result.manifest.created_at.isoformat(),
                    str(run_dir),
                    result.manifest.replay_of,
                ),
            )
        return run_dir

    def verify_checksums(self, run_id: str) -> tuple[bool, tuple[str, ...]]:
        run_dir = self.run_path(run_id)
        checksum_file = run_dir / "checksums.sha256"
        differences: list[str] = []
        if not checksum_file.exists():
            return False, ("Отсутствует checksums.sha256",)
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            path = run_dir / relative
            if not path.exists():
                differences.append(f"Отсутствует {relative}")
            elif self._sha256(path) != expected:
                differences.append(f"Контрольная сумма не совпадает: {relative}")
        return not differences, tuple(differences)

    @staticmethod
    def _effect_signature(result: AnalysisResult) -> dict[str, tuple[float | None, Any]]:
        return {
            f"{effect.graph_id}/{effect.outcome}": (effect.estimate, effect.identified_bounds)
            for effect in result.effects
        }

    def replay(
        self, run_id: str, replicate_id: int | None = None, tolerance: float = 1e-8
    ) -> ReplayResult:
        del replicate_id
        valid, checksum_differences = self.verify_checksums(run_id)
        if not valid:
            return ReplayResult(run_id, None, False, checksum_differences, tolerance)
        run_dir = self.run_path(run_id)
        config = GeneratorConfig.model_validate(
            yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        )
        data = pd.read_parquet(run_dir / "inputs" / "data.parquet")
        source = AnalysisResult.model_validate_json(
            (run_dir / "results" / "analysis.json").read_text(encoding="utf-8")
        )
        evidence_bundle = EvidenceBundle.model_validate(source.passport.evidence)
        replayed = run_analysis(
            config,
            data=data,
            dataset_spec=source.dataset_spec,
            evidence_bundle=evidence_bundle,
            compute_cate=any(bool(effect.cate_profiles) for effect in source.effects),
            project_id=source.manifest.project_id,
            replay_of=run_id,
        )
        differences: list[str] = []
        source_signature = self._effect_signature(source)
        replay_signature = self._effect_signature(replayed)
        for key, (source_estimate, source_bounds) in source_signature.items():
            replay_estimate, replay_bounds = replay_signature.get(key, (None, None))
            if source_estimate is not None and replay_estimate is not None:
                if abs(source_estimate - replay_estimate) > tolerance:
                    differences.append(
                        f"{key}: оценка {source_estimate:.12g} != {replay_estimate:.12g}"
                    )
            elif source_estimate != replay_estimate:
                differences.append(f"{key}: различается наличие точечной оценки")
            if source_bounds != replay_bounds:
                differences.append(f"{key}: различаются идентификационные границы")
        source_decisions = [
            (
                decision.alpha,
                decision.status,
                decision.selected_action,
                decision.pilot_seed,
                decision.pilot_virtual_samples,
                decision.pilot_information_value,
            )
            for decision in source.decisions
        ]
        replay_decisions = [
            (
                decision.alpha,
                decision.status,
                decision.selected_action,
                decision.pilot_seed,
                decision.pilot_virtual_samples,
                decision.pilot_information_value,
            )
            for decision in replayed.decisions
        ]
        if len(source_decisions) != len(replay_decisions):
            differences.append("Различается длина α-траектории")
        else:
            for source_row, replay_row in zip(source_decisions, replay_decisions, strict=True):
                if source_row[:5] != replay_row[:5] or abs(source_row[5] - replay_row[5]) > tolerance:
                    differences.append(
                        f"α={source_row[0]}: различаются решение или пилотная симуляция"
                    )
        if (
            source.trajectory_summary.status != replayed.trajectory_summary.status
            or source.trajectory_summary.selected_action
            != replayed.trajectory_summary.selected_action
            or source.trajectory_summary.action_sequence
            != replayed.trajectory_summary.action_sequence
        ):
            differences.append("Различается сводка α-траектории")
        self.save(replayed, data)
        return ReplayResult(
            run_id, replayed.manifest.run_id, not differences, tuple(differences), tolerance
        )

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def export_zip(self, run_id: str, target: str | Path) -> Path:
        return ExportManager.zip_directory(self.run_path(run_id), target)
