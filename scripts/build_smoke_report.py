from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")


PRIMARY_METHODS = (
    "full_procedure",
    "maximum_graph",
    "hard_set",
    "structure_oracle",
)
METHOD_LABELS = {
    "full_procedure": "Полная процедура",
    "maximum_graph": "Максимальный граф",
    "hard_set": "Жёсткое множество",
    "structure_oracle": "Структурный oracle",
}
REGIMES = {
    "reference": "favorable",
    "evidence_conflict": "favorable",
    "weak_overlap": "boundary",
    "version_mixing": "boundary",
    "informative_loss": "unfavorable",
    "outside_gamma": "unfavorable",
}


def _mcse(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.std(ddof=1) / np.sqrt(len(numeric))) if len(numeric) > 1 else 0.0


def _mean_mcse(
    frame: pd.DataFrame,
    groups: list[str],
    columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(groups, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, keys, strict=True))
        row["replications"] = len(group)
        for column in columns:
            numeric = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(numeric.mean())
            row[f"{column}_mcse"] = _mcse(numeric)
        rows.append(row)
    return pd.DataFrame(rows)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return cast(list[dict[str, Any]], clean.to_dict(orient="records"))


def _markdown_table(frame: pd.DataFrame, digits: int = 5) -> str:
    if frame.empty:
        return "_Нет наблюдений._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "N/A" if pd.isna(value) else f"{value:.{digits}f}"
            )
        else:
            display[column] = display[column].fillna("N/A").astype(str)
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    return "\n".join(
        [line(headers), line(["-" * width for width in widths]), *(line(row) for row in rows)]
    )


def _action_frequencies(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    actions = ("a0", "a1", "a2", "abstain")
    for (scenario, sample_size, method), group in frame.groupby(
        ["scenario", "sample_size", "method"], sort=True
    ):
        selected = group["selected_action"].fillna("abstain")
        for action in actions:
            indicator = (selected == action).astype(float)
            rows.append(
                {
                    "scenario": scenario,
                    "value_regime": REGIMES[scenario],
                    "sample_size": sample_size,
                    "method": method,
                    "action": action,
                    "frequency": float(indicator.mean()),
                    "frequency_mcse": _mcse(indicator),
                    "replications": len(group),
                }
            )
    return pd.DataFrame(rows)


def _estimation_quality(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario, sample_size, method), group in frame.groupby(
        ["scenario", "sample_size", "method"], sort=True
    ):
        for outcome in ("y_cr", "y_cfo"):
            errors = pd.to_numeric(group[f"{outcome}_ate_error"], errors="coerce").dropna()
            squared = errors**2
            rmse = float(np.sqrt(squared.mean())) if len(squared) else float("nan")
            mean_squared_mcse = _mcse(squared)
            rmse_mcse = (
                mean_squared_mcse / (2.0 * rmse)
                if np.isfinite(rmse) and rmse > 0
                else float("nan")
            )
            coverage = pd.to_numeric(group[f"{outcome}_coverage"], errors="coerce")
            width = pd.to_numeric(group[f"{outcome}_interval_width"], errors="coerce")
            rows.append(
                {
                    "scenario": scenario,
                    "sample_size": sample_size,
                    "method": method,
                    "outcome": outcome.upper(),
                    "identified_replications": len(errors),
                    "ate_rmse": rmse,
                    "ate_rmse_mcse_delta": rmse_mcse,
                    "coverage": float(coverage.mean()),
                    "coverage_mcse": _mcse(coverage),
                    "interval_width": float(width.mean()),
                    "interval_width_mcse": _mcse(width),
                }
            )
    return pd.DataFrame(rows)


def _method_differences(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "sample_size", "replicate_id"]
    actions = frame.assign(
        operational_action=frame["selected_action"].fillna("abstain")
    ).pivot(index=keys, columns="method", values="operational_action")
    statuses = frame.pivot(index=keys, columns="method", values="decision_status")
    rows: list[dict[str, Any]] = []
    joined = actions.join(statuses, lsuffix="_action", rsuffix="_status")
    for (scenario, sample_size), group in joined.groupby(level=[0, 1], sort=True):
        full_hard_action = (
            group["full_procedure_action"] != group["hard_set_action"]
        ).astype(float)
        full_hard_status = (
            group["full_procedure_status"] != group["hard_set_status"]
        ).astype(float)
        max_oracle_action = (
            group["maximum_graph_action"] != group["structure_oracle_action"]
        ).astype(float)
        rows.append(
            {
                "scenario": scenario,
                "sample_size": sample_size,
                "replications": len(group),
                "full_vs_hard_action_difference": float(full_hard_action.mean()),
                "full_vs_hard_action_difference_mcse": _mcse(full_hard_action),
                "full_vs_hard_status_difference": float(full_hard_status.mean()),
                "full_vs_hard_status_difference_mcse": _mcse(full_hard_status),
                "maximum_vs_oracle_action_difference": float(max_oracle_action.mean()),
                "maximum_vs_oracle_action_difference_mcse": _mcse(max_oracle_action),
            }
        )
    return pd.DataFrame(rows)


def _database_checks(db_path: Path) -> dict[str, Any]:
    with sqlite3.connect(db_path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        statuses = dict(
            connection.execute(
                "SELECT status, COUNT(*) FROM replicates GROUP BY status"
            ).fetchall()
        )
        duplicates = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT scenario, sample_size, replicate_id, COUNT(*) AS n
                FROM replicates
                GROUP BY scenario, sample_size, replicate_id
                HAVING n > 1
            )
            """
        ).fetchone()[0]
    return {
        "integrity_check": integrity,
        "status_counts": statuses,
        "duplicate_design_rows": int(duplicates),
    }


def _save_figures(
    output_dir: Path,
    action_frequencies: pd.DataFrame,
    performance: pd.DataFrame,
    trajectory_statuses: pd.DataFrame,
) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    overall_actions = (
        action_frequencies.groupby(["method", "action"], sort=False)["frequency"]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(index=PRIMARY_METHODS, columns=["a0", "a1", "a2", "abstain"], fill_value=0.0)
    )
    ax = overall_actions.rename(index=METHOD_LABELS).plot(
        kind="bar",
        stacked=True,
        figsize=(10, 5.6),
        color=["#5B8FF9", "#61DDAA", "#F6BD16", "#7262FD"],
    )
    ax.set_ylabel("Доля решений")
    ax.set_xlabel("")
    ax.set_title("Распределение операционных решений")
    ax.legend(title="Действие", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    ax.tick_params(axis="x", rotation=15)
    ax.figure.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        path = figure_dir / f"smoke_action_distribution.{suffix}"
        ax.figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        created.append(str(path.relative_to(output_dir)))
    plt.close(ax.figure)

    overall_performance = _mean_mcse(
        performance,
        ["method"],
        ["regret"],
    ).set_index("method").reindex(PRIMARY_METHODS)
    figure, ax = plt.subplots(figsize=(9, 5.2))
    positions = np.arange(len(overall_performance))
    ax.bar(
        positions,
        overall_performance["regret"],
        yerr=overall_performance["regret_mcse"],
        capsize=4,
        color="#5B8FF9",
    )
    ax.set_xticks(positions, [METHOD_LABELS[item] for item in overall_performance.index], rotation=15)
    ax.set_ylabel("Среднее regret")
    ax.set_title("Сожаление политики, среднее ± MCSE")
    figure.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        path = figure_dir / f"smoke_regret.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        created.append(str(path.relative_to(output_dir)))
    plt.close(figure)

    status_plot = trajectory_statuses.set_index("trajectory_status")["frequency"].sort_index()
    figure, ax = plt.subplots(figsize=(8, 5.2))
    status_plot.plot(kind="bar", ax=ax, color="#61DDAA")
    ax.set_ylabel("Доля репликаций")
    ax.set_xlabel("Статус α-траектории")
    ax.set_title("Статусы полной процедуры")
    ax.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        path = figure_dir / f"smoke_trajectory_status.{suffix}"
        figure.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        created.append(str(path.relative_to(output_dir)))
    plt.close(figure)
    return created


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    root = args.experiment_dir
    raw_path = root / "replicate_metrics.parquet"
    db_path = root / "checkpoint.sqlite3"
    raw = pd.read_parquet(raw_path)
    primary = raw[raw["method"].isin(PRIMARY_METHODS)].copy()
    primary["value_regime"] = primary["scenario"].map(REGIMES)

    replicate_keys = ["scenario", "sample_size", "replicate_id"]
    truth = primary.drop_duplicates(replicate_keys)
    true_action_distribution = (
        truth.groupby(["scenario", "sample_size", "value_regime", "true_action"])
        .size()
        .reset_index(name="count")
    )
    totals = true_action_distribution.groupby(["scenario", "sample_size"])["count"].transform(
        "sum"
    )
    true_action_distribution["frequency"] = true_action_distribution["count"] / totals
    true_action_distribution["frequency_mcse"] = np.sqrt(
        true_action_distribution["frequency"]
        * (1.0 - true_action_distribution["frequency"])
        / totals
    )

    action_frequencies = _action_frequencies(primary)
    performance = _mean_mcse(
        primary,
        ["scenario", "sample_size", "method"],
        [
            "erroneous_a2",
            "optimal_action_selected",
            "regret",
            "policy_value",
            "maximum_regret",
        ],
    )
    estimation_quality = _estimation_quality(primary)
    identification = _mean_mcse(
        primary,
        ["scenario", "sample_size", "method"],
        [
            "share_identified",
            "share_partially_identified",
            "share_not_identified",
            "share_structural_zero",
        ],
    )
    method_differences = _method_differences(primary)

    full = primary[primary["method"] == "full_procedure"].copy()
    trajectory_statuses = (
        full.groupby("trajectory_status").size().reset_index(name="count")
    )
    trajectory_statuses["frequency"] = trajectory_statuses["count"] / len(full)
    trajectory_statuses["frequency_mcse"] = np.sqrt(
        trajectory_statuses["frequency"]
        * (1.0 - trajectory_statuses["frequency"])
        / len(full)
    )
    trajectory_changes = (
        full.groupby(
            ["first_action_change_alpha", "first_action_change_graph"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    trajectory_changes["frequency"] = trajectory_changes["count"] / len(full)

    unique_replicates = primary.drop_duplicates(replicate_keys)
    replication_resources = _mean_mcse(
        unique_replicates,
        ["scenario", "sample_size"],
        [
            "replication_runtime_seconds",
            "replication_peak_memory_mb",
            "shared_estimation_seconds",
        ],
    )
    method_timing = _mean_mcse(
        primary,
        ["method"],
        ["method_decision_seconds"],
    )
    pilot_diagnostics = _mean_mcse(
        full,
        ["scenario", "sample_size"],
        [
            "pilot_information_value",
            "pilot_expected_regret_reduction",
            "pilot_evi_inner_mcse",
            "pilot_virtual_samples",
        ],
    )
    database_checks = _database_checks(db_path)

    tables = {
        "true_action_distribution": true_action_distribution,
        "action_frequencies": action_frequencies,
        "performance": performance,
        "estimation_quality": estimation_quality,
        "identification_statuses": identification,
        "method_differences": method_differences,
        "trajectory_statuses": trajectory_statuses,
        "trajectory_changes": trajectory_changes,
        "replication_resources": replication_resources,
        "method_timing": method_timing,
        "pilot_diagnostics": pilot_diagnostics,
    }
    for name, table in tables.items():
        table.to_csv(root / f"{name}.csv", index=False)

    figure_paths = _save_figures(root, action_frequencies, primary, trajectory_statuses)
    payload = {
        "experiment_id": root.name,
        "database_checks": database_checks,
        "primary_methods": list(PRIMARY_METHODS),
        "replications": len(unique_replicates),
        "replications_per_cell": int(
            unique_replicates.groupby(["scenario", "sample_size"]).size().min()
        ),
        "compute_cate": bool(primary["y_cr_cate_rmse"].notna().any()),
        "tables": {name: _records(table) for name, table in tables.items()},
        "figures": figure_paths,
    }
    (root / "smoke_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overall_actions = (
        action_frequencies.groupby(["method", "action"])["frequency"]
        .mean()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    overall_performance = _mean_mcse(
        primary,
        ["method"],
        ["erroneous_a2", "optimal_action_selected", "regret", "policy_value"],
    )
    overall_differences = (
        method_differences[
            [
                "full_vs_hard_action_difference",
                "maximum_vs_oracle_action_difference",
            ]
        ]
        .mean()
        .to_frame("frequency")
        .reset_index(names="comparison")
    )

    report = [
        "# Smoke Monte Carlo v2",
        "",
        f"Эксперимент: `{root.name}`.",
        f"Завершено репликаций: **{len(unique_replicates)}**; "
        f"минимум на ячейку: **{payload['replications_per_cell']}**.",
        "Полный эксперимент на 5000 репликаций не запускался.",
        "",
        "## Проверка checkpoint SQLite",
        "",
        f"- `PRAGMA integrity_check`: `{database_checks['integrity_check']}`",
        f"- статусы: `{database_checks['status_counts']}`",
        f"- дубли ключей дизайна: `{database_checks['duplicate_design_rows']}`",
        "",
        "## Истинно оптимальные действия",
        "",
        _markdown_table(true_action_distribution),
        "",
        "Распределение задаётся предрегистрированными экономическими режимами, а не "
        "послеэкспериментальным изменением знака эффекта.",
        "",
        "## Решения методов (среднее по ячейкам)",
        "",
        _markdown_table(overall_actions),
        "",
        "## Policy value, regret и ошибочное a2",
        "",
        _markdown_table(overall_performance),
        "",
        "## Различия утверждённых конфигураций",
        "",
        _markdown_table(overall_differences),
        "",
        "Полная процедура использует суммарный статус всей α-траектории; hard-set "
        "оценивает итоговое множество однократно и не использует μΓ или порядок графов.",
        "",
        "## Статусы и точки изменения α-траектории",
        "",
        _markdown_table(trajectory_statuses),
        "",
        _markdown_table(trajectory_changes),
        "",
        "## Качество оценивания",
        "",
        "Полные показатели по ячейкам находятся в `estimation_quality.csv`. "
        "RMSE относится к ATE; CATE в этом быстром smoke-запуске не вычислялся.",
        "",
        "## Идентификация",
        "",
        "Доли статусов и их MCSE находятся в `identification_statuses.csv`.",
        "",
        "## Время и память",
        "",
        "`replication_runtime_seconds` — всё время репликации; "
        "`shared_estimation_seconds` — общий графоспецифический анализ; "
        "`method_decision_seconds` — только дополнительное построение решения метода; "
        "`replication_peak_memory_mb` — максимум RSS процесса, измеренный внутри репликации.",
        "",
        _markdown_table(method_timing),
        "",
        "## Информационная ценность пилота",
        "",
        _markdown_table(pilot_diagnostics),
        "",
        "## Ограничения smoke-этапа",
        "",
        "- Малое число повторений даёт широкую Monte Carlo неопределённость.",
        "- CATE намеренно отключён для быстрого контрольного запуска.",
        "- Результаты не являются основанием утверждать преимущество полной процедуры.",
        "- Следующий, полный запуск требует отдельного подтверждения координатора.",
        "",
    ]
    (root / "smoke_report.md").write_text("\n".join(report), encoding="utf-8")
    print(root / "smoke_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
