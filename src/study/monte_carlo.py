from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
# import resource
import sqlite3
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Thread

import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from domain import GeneratorConfig
from engine import run_analysis

from .dgp import generate_dataset
from .methods import MethodOutput, run_ablations, run_comparison_methods
from .metrics import aggregate_metrics, evaluate_method
from .oracle import run_structure_oracle
from .scenarios import apply_scenario, apply_value_regime

MASTER_SEED = 20260820
GRAPH_IDS = ("G1", "G2", "G3", "G4")
VALUE_REGIMES = ("favorable", "boundary", "unfavorable")


def _current_rss_bytes(process: psutil.Process | None) -> int:
    if process is not None:
        try:
            return int(process.memory_info().rss)
        except psutil.Error:
            pass
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        rss_line = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
        return int(rss_line.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
            if os.name == "nt":
                return 0
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(usage * (1024 if os.uname().sysname == "Linux" else 1))


def _sample_rss(stop: Event, samples: list[int]) -> None:
    try:
        process: psutil.Process | None = psutil.Process(os.getpid())
    except psutil.Error:
        process = None
    while not stop.is_set():
        samples.append(_current_rss_bytes(process))
        stop.wait(0.02)
    samples.append(_current_rss_bytes(process))


@dataclass(frozen=True)
class DesignCell:
    scenario: str
    sample_size: int


@dataclass(frozen=True)
class FactorialDesignRow:
    design_id: str
    scenario: str
    sample_size: int
    replicate_id: int
    subreplicate_id: int
    true_graph_id: str | None
    value_regime: str
    seed: int
    pilot_seed: int
    truth_pilot_seed: int


FULL_DESIGN = (
    DesignCell("reference", 600),
    DesignCell("reference", 1500),
    DesignCell("reference", 3000),
    DesignCell("evidence_conflict", 1500),
    DesignCell("weak_overlap", 600),
    DesignCell("weak_overlap", 1500),
    DesignCell("weak_overlap", 3000),
    DesignCell("version_mixing", 1500),
    DesignCell("informative_loss", 1500),
    DesignCell("outside_gamma", 1500),
)


def deterministic_stream_seed(
    cell: DesignCell,
    true_graph_id: str | None,
    value_regime: str,
    subreplicate_id: int,
    stream: str,
) -> int:
    graph_token = true_graph_id or "OUTSIDE"
    payload = (
        f"{MASTER_SEED}|{cell.scenario}|{cell.sample_size}|{graph_token}|"
        f"{value_regime}|{subreplicate_id}|{stream}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def deterministic_replicate_seed(cell: DesignCell, replicate_id: int) -> int:
    """Compatibility helper; factors remain explicit and are not seed-derived."""

    return deterministic_stream_seed(cell, "G1", "favorable", replicate_id, "data")


def _make_design_row(
    cell: DesignCell,
    *,
    replicate_id: int,
    subreplicate_id: int,
    true_graph_id: str | None,
    value_regime: str,
) -> FactorialDesignRow:
    graph_token = true_graph_id or "OUTSIDE"
    design_id = (
        f"{cell.scenario}-n{cell.sample_size}-{graph_token}-{value_regime}-"
        f"r{subreplicate_id:03d}"
    )
    return FactorialDesignRow(
        design_id=design_id,
        scenario=cell.scenario,
        sample_size=cell.sample_size,
        replicate_id=replicate_id,
        subreplicate_id=subreplicate_id,
        true_graph_id=true_graph_id,
        value_regime=value_regime,
        seed=deterministic_stream_seed(
            cell, true_graph_id, value_regime, subreplicate_id, "data"
        ),
        pilot_seed=deterministic_stream_seed(
            cell, true_graph_id, value_regime, subreplicate_id, "pilot"
        ),
        truth_pilot_seed=deterministic_stream_seed(
            cell, true_graph_id, value_regime, subreplicate_id, "truth_pilot"
        ),
    )


def r3_design_rows(
    design: tuple[DesignCell, ...] = FULL_DESIGN,
    *,
    repetitions_per_subcell: int = 4,
) -> tuple[FactorialDesignRow, ...]:
    rows: list[FactorialDesignRow] = []
    for cell in design:
        graphs: tuple[str | None, ...] = (
            (None,) if cell.scenario == "outside_gamma" else GRAPH_IDS
        )
        replicate_id = 0
        for graph_id in graphs:
            for value_regime in VALUE_REGIMES:
                for subreplicate_id in range(repetitions_per_subcell):
                    rows.append(
                        _make_design_row(
                            cell,
                            replicate_id=replicate_id,
                            subreplicate_id=subreplicate_id,
                            true_graph_id=graph_id,
                            value_regime=value_regime,
                        )
                    )
                    replicate_id += 1
    return tuple(rows)


def balanced_full_design_rows(
    repetitions: int,
    design: tuple[DesignCell, ...] = FULL_DESIGN,
) -> tuple[FactorialDesignRow, ...]:
    """Future 500/cell schedule; provided but never executed without approval."""

    rows: list[FactorialDesignRow] = []
    for cell in design:
        graphs: tuple[str | None, ...] = (
            (None,) if cell.scenario == "outside_gamma" else GRAPH_IDS
        )
        combinations = tuple((graph, regime) for graph in graphs for regime in VALUE_REGIMES)
        for replicate_id in range(repetitions):
            graph_id, regime = combinations[replicate_id % len(combinations)]
            subreplicate_id = replicate_id // len(combinations)
            row = _make_design_row(
                cell,
                replicate_id=replicate_id,
                subreplicate_id=subreplicate_id,
                true_graph_id=graph_id,
                value_regime=regime,
            )
            # Make a design ID unique when the same subreplicate number occurs
            # in different factor positions in the future 500/cell schedule.
            rows.append(
                FactorialDesignRow(
                    **{
                        **asdict(row),
                        "design_id": f"{row.design_id}-i{replicate_id:03d}",
                    }
                )
            )
    return tuple(rows)


def _run_replicate_inner(
    payload: tuple[dict, bool, int],
) -> dict:
    row_data, compute_cate, cate_trees = payload
    row = FactorialDesignRow(**row_data)
    config = apply_scenario(
        GeneratorConfig(
            mode="laboratory",
            profile_name="monte-carlo-r3",
            scenario=row.scenario,
            sample_size=row.sample_size,
            seed=row.seed,
            pilot_seed=row.pilot_seed,
            truth_pilot_seed=row.truth_pilot_seed,
            true_graph_id=row.true_graph_id,
            value_regime=row.value_regime,
            cate_trees=cate_trees,
        ),
        apply_default_value_regime=False,
    )
    config = apply_value_regime(config, row.value_regime)
    memory_stop = Event()
    memory_samples: list[int] = []
    memory_thread = Thread(
        target=_sample_rss,
        args=(memory_stop, memory_samples),
        name="replication-rss-sampler",
        daemon=True,
    )
    memory_thread.start()
    started = time.perf_counter()
    try:
        generated = generate_dataset(config)
        if generated.truth is None:
            raise RuntimeError("Monte Carlo requires a generated truth bundle")
        truth = generated.truth
        shared_started = time.perf_counter()
        analysis = run_analysis(
            generated.config,
            data=generated.data,
            compute_cate=compute_cate,
            project_id="monte-carlo-r3",
        )
        shared_estimation_seconds = time.perf_counter() - shared_started
        methods = run_comparison_methods(analysis)
        oracle_started = time.perf_counter()
        oracle_output = run_structure_oracle(analysis, truth)
        methods["structure_oracle"] = MethodOutput(
            method=oracle_output.method,
            graph_ids=oracle_output.graph_ids,
            effects=oracle_output.effects,
            decision=oracle_output.decision,
            uses_truth=oracle_output.uses_truth,
            decision_runtime_seconds=time.perf_counter() - oracle_started,
        )
        methods.update(run_ablations(analysis))
        elapsed = time.perf_counter() - started
    finally:
        memory_stop.set()
        memory_thread.join(timeout=1.0)
    peak_memory = max(memory_samples, default=0) / 1024.0**2
    metric_rows = [
        evaluate_method(
            analysis,
            output,
            truth,
            runtime_seconds=elapsed,
            peak_memory_mb=peak_memory,
            shared_estimation_seconds=shared_estimation_seconds,
        )
        for output in methods.values()
    ]
    full_row = next(row for row in metric_rows if row["method"] == "full_procedure")
    for metric_row in metric_rows:
        metric_row["status_match"] = (
            metric_row["decision_status"] == full_row["decision_status"]
        )
        metric_row["operational_action_match"] = (
            metric_row["operational_action"] == full_row["operational_action"]
        )
    return {
        **asdict(row),
        "rows": metric_rows,
        "runtime_seconds": elapsed,
    }


def _run_replicate(payload: tuple[dict, bool, int]) -> dict:
    with threadpool_limits(limits=1):
        return _run_replicate_inner(payload)


class MonteCarloRunner:
    def __init__(self, root: str | Path, experiment_id: str = "r3-20260820"):
        self.root = Path(root) / experiment_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "checkpoint.sqlite3"
        self.pause_event = Event()
        self.cancel_event = Event()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS replicates (
                    design_id TEXT PRIMARY KEY,
                    scenario TEXT NOT NULL,
                    sample_size INTEGER NOT NULL,
                    replicate_id INTEGER NOT NULL,
                    subreplicate_id INTEGER NOT NULL,
                    true_graph_id TEXT,
                    value_regime TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    pilot_seed INTEGER NOT NULL,
                    truth_pilot_seed INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    runtime_seconds REAL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _insert_design(self, rows: tuple[FactorialDesignRow, ...]) -> int:
        values = [
            (
                row.design_id,
                row.scenario,
                row.sample_size,
                row.replicate_id,
                row.subreplicate_id,
                row.true_graph_id,
                row.value_regime,
                row.seed,
                row.pilot_seed,
                row.truth_pilot_seed,
                "pending",
            )
            for row in rows
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO replicates (
                    design_id,scenario,sample_size,replicate_id,subreplicate_id,
                    true_graph_id,value_regime,seed,pilot_seed,truth_pilot_seed,status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                values,
            )
        self.export_design()
        return len(rows)

    def prepare_r3(
        self,
        design: tuple[DesignCell, ...] = FULL_DESIGN,
        *,
        repetitions_per_subcell: int = 4,
    ) -> int:
        return self._insert_design(
            r3_design_rows(design, repetitions_per_subcell=repetitions_per_subcell)
        )

    def prepare(self, repetitions: int, design: tuple[DesignCell, ...] = FULL_DESIGN) -> int:
        """Prepare a balanced future design; this method does not execute it."""

        return self._insert_design(balanced_full_design_rows(repetitions, design))

    def export_design(self) -> tuple[Path, Path]:
        with self._connect() as connection:
            frame = pd.read_sql_query(
                """
                SELECT design_id,scenario,sample_size,replicate_id,subreplicate_id,
                       true_graph_id,value_regime,seed,pilot_seed,truth_pilot_seed
                FROM replicates ORDER BY scenario,sample_size,replicate_id
                """,
                connection,
            )
        csv_path = self.root / "r3_design.csv"
        json_path = self.root / "r3_design.json"
        frame.to_csv(csv_path, index=False)
        json_path.write_text(
            json.dumps(frame.where(pd.notna(frame), None).to_dict(orient="records"),
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return csv_path, json_path

    def pause(self) -> None:
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()

    def cancel(self) -> None:
        self.cancel_event.set()

    def retry_failed(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE replicates SET status='pending', error=NULL WHERE status='failed'"
            )
        return cursor.rowcount

    def status_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS n FROM replicates GROUP BY status"
            ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    def _pending(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM replicates WHERE status IN ('pending','running','cancelled') "
                "ORDER BY scenario,sample_size,replicate_id"
            ).fetchall()

    def run(
        self,
        *,
        workers: int | None = None,
        compute_cate: bool = False,
        cate_trees: int = 300,
        progress: Callable[[dict[str, int], float], None] | None = None,
    ) -> dict[str, int]:
        pending = self._pending()
        if not pending:
            return self.status_counts()
        workers = workers or max(1, min(4, (os.cpu_count() or 2) - 1))
        with self._connect() as connection:
            connection.executemany(
                "UPDATE replicates SET status='running' WHERE design_id=?",
                [(row["design_id"],) for row in pending],
            )
        started = time.perf_counter()
        completed_now = 0
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {
                executor.submit(
                    _run_replicate,
                    (
                        {
                            key: row[key]
                            for key in FactorialDesignRow.__dataclass_fields__
                        },
                        compute_cate,
                        cate_trees,
                    ),
                ): row
                for row in pending
            }
            for future in as_completed(futures):
                row = futures[future]
                if self.cancel_event.is_set():
                    for remaining in futures:
                        remaining.cancel()
                    break
                while self.pause_event.is_set() and not self.cancel_event.is_set():
                    time.sleep(0.2)
                try:
                    result = future.result()
                    status = "completed"
                    result_json = json.dumps(result, ensure_ascii=False, default=str)
                    error = None
                    runtime_seconds = result["runtime_seconds"]
                except Exception:
                    status = "failed"
                    result_json = None
                    error = traceback.format_exc()
                    runtime_seconds = None
                with self._connect() as connection:
                    connection.execute(
                        """
                        UPDATE replicates
                        SET status=?, result_json=?, error=?, runtime_seconds=?,
                            updated_at=CURRENT_TIMESTAMP
                        WHERE design_id=?
                        """,
                        (status, result_json, error, runtime_seconds, row["design_id"]),
                    )
                completed_now += 1
                if completed_now % 100 == 0:
                    self.export()
                if progress:
                    counts = self.status_counts()
                    elapsed = time.perf_counter() - started
                    rate = completed_now / max(elapsed, 1e-9)
                    remaining_count = counts.get("running", 0) + counts.get("pending", 0)
                    progress(counts, remaining_count / max(rate, 1e-9))
        if self.cancel_event.is_set():
            with self._connect() as connection:
                connection.execute(
                    "UPDATE replicates SET status='cancelled' WHERE status='running'"
                )
        return self.status_counts()

    def results(self) -> pd.DataFrame:
        rows: list[dict] = []
        with self._connect() as connection:
            records = connection.execute(
                """
                SELECT result_json FROM replicates WHERE status='completed'
                ORDER BY scenario,sample_size,replicate_id
                """
            ).fetchall()
        for record in records:
            payload = json.loads(record["result_json"])
            for metric_row in payload["rows"]:
                for key in (
                    "design_id",
                    "replicate_id",
                    "subreplicate_id",
                    "pilot_seed",
                    "truth_pilot_seed",
                ):
                    metric_row[key] = payload[key]
                rows.append(metric_row)
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["report_status"] = frame["decision_status"]
            full = frame["method"] == "full_procedure"
            frame.loc[full, "report_status"] = frame.loc[full, "trajectory_status"]
            for status in (
                "robust",
                "conditionally_robust",
                "switching",
                "pilot",
                "abstain",
            ):
                frame[status] = frame["report_status"] == status
        return frame

    def export(self) -> tuple[Path, Path, Path, Path, Path]:
        raw = self.results()
        aggregate = aggregate_metrics(raw) if not raw.empty else pd.DataFrame()
        raw_path = self.root / "replicate_metrics.parquet"
        aggregate_path = self.root / "aggregate_metrics.parquet"
        csv_path = self.root / "aggregate_metrics.csv"
        json_path = self.root / "aggregate_metrics.json"
        raw_csv_path = self.root / "replicate_metrics.csv"
        raw.to_parquet(raw_path, index=False)
        raw.to_csv(raw_csv_path, index=False)
        aggregate.to_parquet(aggregate_path, index=False)
        aggregate.to_csv(csv_path, index=False)
        json_path.write_text(
            json.dumps(
                aggregate.astype(object).where(pd.notna(aggregate), None).to_dict(orient="records"),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return raw_path, aggregate_path, csv_path, json_path, raw_csv_path
