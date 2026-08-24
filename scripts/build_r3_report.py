from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PRIMARY_METHODS = (
    "full_procedure",
    "maximum_graph",
    "hard_set",
    "structure_oracle",
)


def _mcse(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def summarize(frame: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    groups = [*dimensions, "method"]
    for keys, group in frame.groupby(groups, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, keys, strict=True))
        recommended = group["recommended_action"].fillna("abstain")
        operational = group["operational_action"]
        opportunities = int(group["erroneous_a2_opportunity"].sum())
        erroneous = int(group["erroneous_a2"].sum())
        row.update(
            replications=len(group),
            recommended_a0=float((recommended == "a0").mean()),
            recommended_a1=float((recommended == "a1").mean()),
            recommended_a2=float((recommended == "a2").mean()),
            recommended_abstain=float((recommended == "abstain").mean()),
            operational_a0=float((operational == "a0").mean()),
            operational_a1=float((operational == "a1").mean()),
            operational_a2=float((operational == "a2").mean()),
            operational_policy_accuracy=float(group["operational_policy_accuracy"].mean()),
            recommendation_exact_match=float(group["recommendation_exact_match"].mean()),
            operational_regret_mean=float(group["regret"].mean()),
            operational_regret_median=float(group["regret"].median()),
            operational_regret_outer_mcse=_mcse(group["regret"]),
            decision_robust=float((group["decision_status"] == "robust").mean()),
            decision_conditionally_robust=float(
                (group["decision_status"] == "conditionally_robust").mean()
            ),
            decision_pilot=float((group["decision_status"] == "pilot").mean()),
            decision_abstain=float((group["decision_status"] == "abstain").mean()),
            trajectory_switching=float(group["switching"].mean()),
            alpha_action_changed=float(group["alpha_action_changed"].mean()),
            gross_evi_mean=float(group["pilot_gross_evi"].mean()),
            net_evi_mean=float(group["pilot_net_evi"].mean()),
            evi_inner_mcse_mean=float(group["pilot_evi_inner_mcse"].mean()),
            r0_mean=float(group["pilot_r0"].mean()),
            r1_mean=float(group["pilot_r1"].mean()),
            information_cost_mean=float(group["pilot_information_cost"].mean()),
            positive_net_evi_count=int((group["pilot_net_evi"] > 0).sum()),
            selected_a1_count=int((recommended == "a1").sum()),
            abstain_count=int((group["decision_status"] == "abstain").sum()),
            false_confidence_rate=float(group["false_confidence"].mean()),
            erroneous_a2_opportunities=opportunities,
            erroneous_a2_count=erroneous,
            erroneous_a2_conditional_rate=(
                float(erroneous / opportunities) if opportunities else float("nan")
            ),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_decision_differences(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for left, right in (
        ("full_procedure", "hard_set"),
        ("full_procedure", "maximum_graph"),
        ("full_procedure", "structure_oracle"),
    ):
        subset = frame[frame["method"].isin((left, right))]
        status = subset.pivot(
            index="design_id", columns="method", values="decision_status"
        ).dropna()
        action = subset.pivot(
            index="design_id", columns="method", values="operational_action"
        ).dropna()
        common = status.index.intersection(action.index)
        rows.append(
            {
                "left_method": left,
                "right_method": right,
                "paired_runs": len(common),
                "status_differences": int(
                    (status.loc[common, left] != status.loc[common, right]).sum()
                ),
                "operational_action_differences": int(
                    (action.loc[common, left] != action.loc[common, right]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _database_checks(db_path: Path) -> dict:
    with sqlite3.connect(db_path) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        statuses = dict(
            connection.execute(
                "SELECT status, COUNT(*) FROM replicates GROUP BY status"
            ).fetchall()
        )
        duplicate_ids = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT design_id, COUNT(*) AS n FROM replicates
                GROUP BY design_id HAVING n > 1
            )
            """
        ).fetchone()[0]
        factor_counts = connection.execute(
            """
            SELECT scenario,sample_size,COALESCE(true_graph_id,'OUTSIDE'),value_regime,COUNT(*)
            FROM replicates GROUP BY scenario,sample_size,true_graph_id,value_regime
            """
        ).fetchall()
        seed_rows = connection.execute(
            "SELECT seed,pilot_seed,truth_pilot_seed FROM replicates"
        ).fetchall()
    all_seeds = [int(value) for row in seed_rows for value in row]
    return {
        "integrity_check": integrity,
        "status_counts": statuses,
        "duplicate_design_ids": int(duplicate_ids),
        "factor_count_min": min((row[-1] for row in factor_counts), default=0),
        "factor_count_max": max((row[-1] for row in factor_counts), default=0),
        "seed_count": len(all_seeds),
        "unique_seed_count": len(set(all_seeds)),
        "within_row_seed_collisions": sum(len(set(row)) != 3 for row in seed_rows),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "N/A" if pd.isna(value) else f"{value:.6f}"
            )
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _build_pdf(
    path: Path,
    checks: dict,
    overall: pd.DataFrame,
    pairwise: pd.DataFrame,
    total_runs: int,
) -> None:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = "Helvetica"
    bold = "Helvetica-Bold"
    if font_path.exists() and bold_path.exists():
        pdfmetrics.registerFont(TTFont("DejaVu", str(font_path)))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(bold_path)))
        font, bold = "DejaVu", "DejaVu-Bold"
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleV3", parent=styles["Title"], fontName=bold, alignment=TA_CENTER
    )
    body = ParagraphStyle("BodyV3", parent=styles["BodyText"], fontName=font, leading=14)
    doc = SimpleDocTemplate(
        str(path), pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm
    )
    story = [
        Paragraph("Контрольный smoke-run R3.1 — отчёт V3.1", title),
        Spacer(1, 5 * mm),
        Paragraph(
            f"Завершено уникальных запусков: {total_runs}. CATE отключён. "
            f"SQLite integrity_check: {checks['integrity_check']}; "
            f"статусы: {checks['status_counts']}; дубликаты: {checks['duplicate_design_ids']}.",
            body,
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            "Равное распределение G1–G4 — дизайн стресс-теста, а не априорные "
            "вероятности структур. Параметры не подбирались по результатам R3.1.",
            body,
        ),
        Spacer(1, 6 * mm),
    ]
    columns = [
        "method",
        "replications",
        "operational_a0",
        "operational_a1",
        "operational_a2",
        "decision_abstain",
        "operational_policy_accuracy",
        "recommendation_exact_match",
        "operational_regret_mean",
        "operational_regret_median",
        "erroneous_a2_opportunities",
        "erroneous_a2_conditional_rate",
    ]
    display = overall.loc[:, columns].copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.5f}"
        )
    short_headers = [
        "Метод",
        "n",
        "op a0",
        "op a1",
        "op a2",
        "abstain",
        "op accuracy",
        "exact rec.",
        "mean regret",
        "median",
        "err. a2 opp.",
        "err. a2 rate",
    ]
    table = Table(
        [short_headers, *display.values.tolist()],
        repeatRows=1,
        colWidths=[
            36 * mm,
            18 * mm,
            16 * mm,
            16 * mm,
            16 * mm,
            19 * mm,
            22 * mm,
            22 * mm,
            25 * mm,
            22 * mm,
            26 * mm,
            20 * mm,
        ],
    )
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), bold),
                ("FONTNAME", (0, 1), (-1, -1), font),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEAF7")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    pairwise_lines = "; ".join(
        f"{row.left_method} / {row.right_method}: статусы {row.status_differences}, "
        f"операционные действия {row.operational_action_differences}"
        for row in pairwise.itertuples(index=False)
    )
    story.extend(
        [
            table,
            Spacer(1, 5 * mm),
            Paragraph(f"Парные различия: {pairwise_lines}.", body),
            Spacer(1, 5 * mm),
        ]
    )
    story.append(
        Paragraph(
            "Полные разрезы по сценарию, размеру выборки, value_regime, "
            "true_graph_id и согласованности с max-membership находятся в CSV/JSON "
            "рядом с отчётом. GUI-приёмка вынесена в отдельный журнал.",
            body,
        )
    )
    doc.build(story)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    root = args.experiment_dir
    raw = pd.read_parquet(root / "replicate_metrics.parquet")
    primary = raw[raw["method"].isin(PRIMARY_METHODS)].copy()
    primary["true_graph_id"] = primary["true_graph_id"].fillna("OUTSIDE")
    unique_runs = int(primary["design_id"].nunique())
    checks = _database_checks(root / "checkpoint.sqlite3")
    if unique_runs != 444 or checks["status_counts"] != {"completed": 444}:
        raise RuntimeError(
            f"R3.1 incomplete: unique={unique_runs}, statuses={checks['status_counts']}"
        )
    if checks["integrity_check"] != "ok" or checks["duplicate_design_ids"]:
        raise RuntimeError(f"R3.1 database check failed: {checks}")
    if checks["factor_count_min"] != 4 or checks["factor_count_max"] != 4:
        raise RuntimeError(f"R3.1 factor allocation differs from preregistration: {checks}")
    if checks["seed_count"] != 1332 or checks["unique_seed_count"] != 1332:
        raise RuntimeError(f"R3.1 seed allocation is not unique: {checks}")
    if checks["within_row_seed_collisions"]:
        raise RuntimeError(f"R3.1 has within-row seed collisions: {checks}")

    breakdowns = {
        "overall_by_method": summarize(primary, []),
        "by_scenario": summarize(primary, ["scenario"]),
        "by_sample_size": summarize(primary, ["sample_size"]),
        "by_value_regime": summarize(primary, ["value_regime"]),
        "by_true_graph_id": summarize(primary, ["true_graph_id"]),
        "by_graph_match": summarize(primary, ["true_graph_matches_maximum"]),
    }
    for name, frame in breakdowns.items():
        frame.to_csv(root / f"r3_1_{name}.csv", index=False)
        (root / f"r3_1_{name}.json").write_text(
            json.dumps(
                frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records"),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    overall = breakdowns["overall_by_method"]
    pairwise = pairwise_decision_differences(primary)
    pairwise.to_csv(root / "r3_1_pairwise_decision_differences.csv", index=False)
    (root / "r3_1_pairwise_decision_differences.json").write_text(
        json.dumps(pairwise.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = f"""# Отчёт V3.1: контрольный smoke-run R3.1

## Валидация дизайна

- Завершено уникальных запусков: **{unique_runs}**.
- CATE: **отключён**.
- SQLite integrity_check: **{checks['integrity_check']}**.
- Статусы: `{checks['status_counts']}`.
- Дубликаты design_id: **{checks['duplicate_design_ids']}**.
- Число повторов на подстрату: min={checks['factor_count_min']}, max={checks['factor_count_max']}.
- Уникальные seed: **{checks['unique_seed_count']} из {checks['seed_count']}**.
- Полный эксперимент 5000/5000 не выполнялся.

Граф и экономический режим назначались явным декартовым расписанием до запуска.
Равная представленность G1–G4 — дизайн стресс-теста, не априорные вероятности.

## Агрегат по методу

{_markdown_table(overall)}

## Различия статусов и операционных действий

{_markdown_table(pairwise)}

## Разрезы

Рядом сохранены полные CSV/JSON по сценарию, размеру выборки, `value_regime`,
`true_graph_id`, согласованности истинного графа с max-membership и по методу.
Каждая таблица раздельно содержит recommended action, decision status и
operational action; operational policy accuracy и recommendation exact match;
mean/median operational regret и MCSE; abstain rate; R0, R1, gross/net EVI,
стоимость информации, внутреннюю MCSE, false confidence и условный erroneous_a2.

Нулевой erroneous_a2 или совпадение full NF-SCM с hard-set не трактуются как
ошибка дизайна. Параметры после просмотра результатов не менялись.
"""
    (root / "R3_1_REPORT_V3_1.md").write_text(report, encoding="utf-8")
    summary = {
        "design_checks": checks,
        "unique_completed_runs": unique_runs,
        "cate_enabled": False,
        "full_experiment_executed": False,
        "overall_by_method": overall.astype(object)
        .where(pd.notna(overall), None)
        .to_dict(orient="records"),
        "pairwise_decision_differences": pairwise.to_dict(orient="records"),
    }
    (root / "R3_1_REPORT_V3_1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _build_pdf(
        root / "R3_1_REPORT_V3_1.pdf", checks, overall, pairwise, unique_runs
    )
    print(root / "R3_1_REPORT_V3_1.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
