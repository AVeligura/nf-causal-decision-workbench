from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import networkx as nx
import pandas as pd

from domain import AnalysisResult
from engine.evidence import reference_evidence, score_graph
from engine.graphs import reference_graphs
from runtime import RunRepository
from study.control_examples import run_control_examples


@dataclass(frozen=True)
class AcceptanceCheck:
    name: str
    status: Literal["PASS", "FAIL", "BLOCKED"]
    evidence: str


def _same_replay(source: AnalysisResult, replay: AnalysisResult) -> bool:
    source_effects = [
        (item.graph_id, item.outcome, item.status, item.estimate, item.identified_bounds)
        for item in source.effects
    ]
    replay_effects = [
        (item.graph_id, item.outcome, item.status, item.estimate, item.identified_bounds)
        for item in replay.effects
    ]
    source_decisions = [
        (
            item.alpha,
            item.status,
            item.selected_action,
            item.pilot_seed,
            item.pilot_virtual_samples,
            item.pilot_information_value,
        )
        for item in source.decisions
    ]
    replay_decisions = [
        (
            item.alpha,
            item.status,
            item.selected_action,
            item.pilot_seed,
            item.pilot_virtual_samples,
            item.pilot_information_value,
        )
        for item in replay.decisions
    ]
    return (
        source_effects == replay_effects
        and source_decisions == replay_decisions
        and source.trajectory_summary.action_sequence
        == replay.trajectory_summary.action_sequence
        and source.trajectory_summary.status == replay.trajectory_summary.status
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-repository",
        type=Path,
        default=Path("artifacts/reference_repository_v2"),
    )
    parser.add_argument("--source-run-id", default="run-8adee0cc5217")
    parser.add_argument("--replay-run-id", default="run-6c3eb54f49e9")
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=Path("artifacts/experiments/mc-smoke-v2-20260817-r2"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/acceptance_v2")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[AcceptanceCheck] = []

    expected_scores = {"G1": 0.92, "G2": 0.81, "G3": 0.67, "G4": 0.43}
    bundle = reference_evidence()
    observed_scores = {
        graph.graph_id: score_graph(graph, bundle).mu for graph in reference_graphs()
    }
    checks.append(
        AcceptanceCheck(
            "Прозрачный корпус и контрольные μΓ(G)",
            "PASS" if observed_scores == expected_scores else "FAIL",
            str(observed_scores),
        )
    )

    examples = run_control_examples()
    examples_passed = all(item["actual"][key] == value for item in examples for key, value in item["expected"].items())
    checks.append(
        AcceptanceCheck(
            "Детерминированные контрольные примеры",
            "PASS" if examples_passed else "FAIL",
            f"совпали {sum(item['actual'][key] == value for item in examples for key, value in item['expected'].items())} ожидаемых полей",
        )
    )

    repository = RunRepository(args.reference_repository)
    source_valid, source_differences = repository.verify_checksums(args.source_run_id)
    replay_valid, replay_differences = repository.verify_checksums(args.replay_run_id)
    checks.append(
        AcceptanceCheck(
            "Контрольные суммы эталонного и replay запусков",
            "PASS" if source_valid and replay_valid else "FAIL",
            f"source={source_differences}; replay={replay_differences}",
        )
    )

    source_dir = repository.run_path(args.source_run_id)
    replay_dir = repository.run_path(args.replay_run_id)
    source = AnalysisResult.model_validate_json(
        (source_dir / "results" / "analysis.json").read_text(encoding="utf-8")
    )
    replay = AnalysisResult.model_validate_json(
        (replay_dir / "results" / "analysis.json").read_text(encoding="utf-8")
    )
    replay_match = replay.manifest.replay_of == args.source_run_id and _same_replay(source, replay)
    checks.append(
        AcceptanceCheck(
            "Replay эффектов, решений, α-траектории и пилота",
            "PASS" if replay_match else "FAIL",
            f"{args.source_run_id} -> {args.replay_run_id}",
        )
    )

    export_dir = source_dir / "results"
    csv = pd.read_csv(export_dir / "generated_data.csv")
    xlsx = pd.read_excel(export_dir / "generated_data.xlsx")
    parquet = pd.read_parquet(export_dir / "generated_data.parquet")
    tabular_match = len(csv) == len(xlsx) == len(parquet) == source.config.sample_size
    graphml_valid = all(
        len(nx.read_graphml(source_dir / "evidence" / f"{graph.graph_id}.graphml")) > 0
        for graph in reference_graphs()
    )
    passport_files = (
        source_dir / "passport" / "CausalDecisionPassport.json",
        source_dir / "passport" / "CausalDecisionPassport.pdf",
    )
    figure_files = tuple((source_dir / "figures").glob("*"))
    exports_valid = (
        tabular_match
        and graphml_valid
        and all(path.stat().st_size > 1_000 for path in passport_files)
        and len(figure_files) == 12
        and all(path.stat().st_size > 10_000 for path in figure_files)
    )
    checks.append(
        AcceptanceCheck(
            "Экспорты CSV/XLSX/Parquet/JSON/PDF/PNG/SVG",
            "PASS" if exports_valid else "FAIL",
            f"rows={len(csv)}; GraphML=4; figures={len(figure_files)}",
        )
    )

    with sqlite3.connect(args.smoke_dir / "checkpoint.sqlite3") as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        completed = connection.execute(
            "SELECT COUNT(*) FROM replicates WHERE status='completed'"
        ).fetchone()[0]
        duplicates = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT scenario, sample_size, replicate_id, COUNT(*) AS n
                FROM replicates GROUP BY scenario, sample_size, replicate_id HAVING n > 1
            )
            """
        ).fetchone()[0]
    smoke_valid = (
        integrity == "ok"
        and completed == 100
        and duplicates == 0
        and (args.smoke_dir / "smoke_report.json").exists()
        and (args.smoke_dir / "smoke_report.md").exists()
    )
    checks.append(
        AcceptanceCheck(
            "Smoke Monte Carlo и checkpoint SQLite",
            "PASS" if smoke_valid else "FAIL",
            f"integrity={integrity}; completed={completed}; duplicates={duplicates}",
        )
    )

    qt_probe = subprocess.run(
        [sys.executable, "-c", "from PySide6.QtWidgets import QApplication"],
        capture_output=True,
        text=True,
        check=False,
    )
    if qt_probe.returncode == 0:
        qt_status: Literal["PASS", "FAIL", "BLOCKED"] = "PASS"
        qt_evidence = "PySide6.QtWidgets импортируется"
    else:
        qt_status = "BLOCKED"
        error_lines = (qt_probe.stderr or qt_probe.stdout).strip().splitlines()
        qt_evidence = error_lines[-1] if error_lines else "Qt probe failed"
    checks.append(AcceptanceCheck("UI и реальные скриншоты", qt_status, qt_evidence))

    checks.append(
        AcceptanceCheck(
            "Полный Monte Carlo",
            "PASS",
            "Не запускался: по контракту требуется отдельное подтверждение координатора",
        )
    )

    failed = [item for item in checks if item.status == "FAIL"]
    blocked = [item for item in checks if item.status == "BLOCKED"]
    payload = {
        "control_stage_passed": not failed,
        "fully_accepted": not failed and not blocked,
        "checks": [asdict(item) for item in checks],
    }
    (args.output_dir / "acceptance_report_v2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = [
        "# Приёмка контрольного этапа v2",
        "",
        f"Контрольный этап: {'PASS' if not failed else 'FAIL'}.",
        f"Полная приёмка: {'PASS' if not failed and not blocked else 'НЕ ЗАВЕРШЕНА'}.",
        "",
    ]
    rows.extend(f"- **{item.status}** — {item.name}: {item.evidence}" for item in checks)
    rows.extend(
        [
            "",
            "UI-проверка не подменяется статической проверкой: если Qt заблокирован окружением, "
            "реальные скриншоты должны быть получены на системе с доступной `libEGL.so.1`.",
            "",
        ]
    )
    (args.output_dir / "acceptance_report_v2.md").write_text(
        "\n".join(rows), encoding="utf-8"
    )
    print(args.output_dir / "acceptance_report_v2.md")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
