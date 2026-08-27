from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from domain import CausalDecisionPassport, DatasetSpec, GraphSpec

REQUIRED_COLUMNS = {"T", "V", "L", "D", "S", "Y_CR", "Y_CFO"}
X_COLUMNS = {"X1", "X2", "X4", "X5", "X21", "X27", "X29", "X44"}


class DataImporter:
    @staticmethod
    def required_columns() -> tuple[str, ...]:
        return tuple(sorted(REQUIRED_COLUMNS | X_COLUMNS))

    @staticmethod
    def suggest_mapping(frame: pd.DataFrame) -> dict[str, str]:
        aliases = {
            "T": ("t", "treatment", "exposure"),
            "V": ("v", "version", "treatment_version"),
            "L": ("l", "liquidity"),
            "D": ("d", "debt", "leverage"),
            "S": ("s", "observed", "selection"),
            "Y_CR": ("y_cr", "cr", "coverage_ratio"),
            "Y_CFO": ("y_cfo", "cfo", "cash_flow"),
        }
        normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
        mapping: dict[str, str] = {}
        for target in sorted(REQUIRED_COLUMNS | X_COLUMNS):
            candidates = (target.lower(), *aliases.get(target, ()))
            source = next((normalized[name] for name in candidates if name in normalized), "")
            mapping[target] = source
        return mapping

    @staticmethod
    def read(path: str | Path) -> pd.DataFrame:
        source = Path(path)
        suffix = source.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(source)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(source)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(source)
        raise ValueError(f"Неподдерживаемый формат: {suffix}")

    @staticmethod
    def map_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
        reverse = {source: target for target, source in mapping.items() if source}
        mapped = frame.rename(columns=reverse).copy()
        return mapped

    @staticmethod
    def validate(frame: pd.DataFrame) -> tuple[bool, tuple[str, ...]]:
        warnings: list[str] = []
        missing = (REQUIRED_COLUMNS | X_COLUMNS) - set(frame.columns)
        if missing:
            warnings.append(f"Отсутствуют обязательные столбцы: {', '.join(sorted(missing))}")
        if "T" in frame and not set(frame["T"].dropna().unique()) <= {0, 1}:
            warnings.append("T должен принимать только значения 0 и 1")
        if "S" in frame and not set(frame["S"].dropna().unique()) <= {0, 1}:
            warnings.append("S должен принимать только значения 0 и 1")
        if "V" in frame and not set(frame["V"].dropna().astype(str).unique()) <= {
            "base",
            "full",
            "partial",
            "refused",
        }:
            warnings.append("V содержит неизвестные версии исполнения")
        if any(
            frame[column].isna().mean() > 0.35 for column in frame.columns if column in X_COLUMNS
        ):
            warnings.append("Доля пропусков в одной из исходных характеристик превышает 35%")
        return not warnings, tuple(warnings)

    @staticmethod
    def preview(frame: pd.DataFrame, rows: int = 100) -> pd.DataFrame:
        return frame.head(rows).copy()

    @staticmethod
    def build_dataset_spec(
        path: str | Path,
        frame: pd.DataFrame,
        mapping: dict[str, str],
        *,
        user_description: str | None = None,
    ) -> DatasetSpec:
        source = Path(path)
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        missingness = {
            str(column): float(frame[column].isna().mean()) for column in frame.columns
        }
        return DatasetSpec(
            kind="imported",
            source=user_description or "Пользовательский импорт",
            doi=None,
            license=None,
            unit="user-defined",
            selected_features=tuple(sorted(column for column in X_COLUMNS if column in frame)),
            checksum_sha256=digest.hexdigest(),
            source_file_name=source.name,
            file_format=source.suffix.lower().lstrip("."),
            rows=len(frame),
            columns=len(frame.columns),
            variable_mapping={target: source_name for target, source_name in mapping.items() if source_name},
            missingness=missingness,
            imported_at=datetime.now(UTC),
            user_description=user_description,
            truth_available=False,
        )


def _register_pdf_font() -> str:
    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates = (
        windows_fonts / "segoeui.ttf",
        windows_fonts / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            pdfmetrics.registerFont(TTFont("PassportSans", str(candidate)))
            return "PassportSans"
    raise RuntimeError(
        "Не найден TrueType-шрифт с поддержкой кириллицы для PDF-паспорта"
    )


def _format_pdf_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (list, tuple)):
        return "[" + "; ".join(_format_pdf_number(item) for item in value) + "]"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


_PDF_STATUS_LABELS = {
    "identified": "идентифицирован",
    "partially_identified": "частично идентифицирован",
    "not_identified": "не идентифицирован",
    "structural_zero": "структурный нуль",
    "robust": "устойчиво",
    "conditionally_robust": "условно устойчиво",
    "switching": "переключение",
    "pilot": "пилот",
    "abstain": "мотивированный отказ",
}


class ExportManager:
    @staticmethod
    def export_graphml(graph: GraphSpec, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        directed = nx.DiGraph()
        directed.graph.update(
            graph_id=graph.graph_id,
            version=graph.version,
            description=graph.description,
        )
        directed.add_nodes_from(graph.nodes)
        directed.add_edges_from(graph.edges)
        nx.write_graphml(directed, target)
        return target

    @staticmethod
    def export_table(frame: pd.DataFrame, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()
        if suffix == ".csv":
            frame.to_csv(target, index=False)
        elif suffix == ".xlsx":
            frame.to_excel(target, index=False)
        elif suffix in {".parquet", ".pq"}:
            stored = frame.copy(deep=False)
            stored.attrs = {}
            stored.to_parquet(target, index=False)
        else:
            raise ValueError(f"Неподдерживаемый формат экспорта: {suffix}")
        return target

    @staticmethod
    def export_json(payload: Any, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        return target

    @staticmethod
    def passport_pdf(passport: CausalDecisionPassport, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        font = _register_pdf_font()
        styles = getSampleStyleSheet()
        for style in styles.byName.values():
            style.fontName = font
        table_cell = styles["BodyText"].clone("PassportTableCell")
        table_cell.fontName = font
        table_cell.fontSize = 7.2
        table_cell.leading = 8.5
        table_cell.wordWrap = "CJK"
        document = SimpleDocTemplate(
            str(target),
            pagesize=A4,
            leftMargin=42,
            rightMargin=42,
            topMargin=42,
            bottomMargin=42,
            title="CausalDecisionPassport",
        )
        story = [
            Paragraph("Каузальный паспорт управленческого решения", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"Run ID: {passport.manifest.run_id}", styles["BodyText"]),
            Paragraph(f"Статус проверки: {passport.review_status}", styles["BodyText"]),
            Spacer(1, 10),
            Paragraph("Причинные вопросы", styles["Heading2"]),
        ]
        for query in passport.causal_queries:
            story.append(
                Paragraph(
                    f"{query.outcome}: {query.estimand}, горизонт {query.horizon_quarters} квартала; "
                    f"версия воздействия {query.treatment_version}",
                    styles["BodyText"],
                )
            )
        story.extend([Spacer(1, 8), Paragraph("Структурное пространство", styles["Heading2"])])
        score_rows = [["Граф", "μΓ(G)", "Покрытие", "Предупреждения"]]
        for score in passport.structural_space["scores"]:
            score_rows.append(
                [
                    score["graph_id"],
                    f"{score['mu']:.3f}",
                    f"{score['coverage']:.2f}",
                    "; ".join(score.get("warnings", [])) or "—",
                ]
            )
        table = Table(score_rows, colWidths=[45, 55, 55, 310], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEAF7")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#7A8793")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.extend(
            [table, PageBreak(), Paragraph("Графоспецифические результаты", styles["Heading2"])]
        )
        effect_rows = [
            [
                Paragraph("Граф", table_cell),
                Paragraph("Исход", table_cell),
                Paragraph("Статус", table_cell),
                Paragraph("Оценка / границы", table_cell),
                Paragraph("Интервал", table_cell),
            ]
        ]
        for effect in passport.identification_and_estimation["graph_specific_results"]:
            value = effect.get("estimate")
            if value is None:
                value = effect.get("identified_bounds") or "—"
            effect_rows.append(
                [
                    Paragraph(str(effect["graph_id"]), table_cell),
                    Paragraph(str(effect["outcome"]), table_cell),
                    Paragraph(
                        _PDF_STATUS_LABELS.get(effect["status"], effect["status"]),
                        table_cell,
                    ),
                    Paragraph(_format_pdf_number(value), table_cell),
                    Paragraph(_format_pdf_number(effect.get("interval")), table_cell),
                ]
            )
        effect_table = Table(effect_rows, colWidths=[38, 55, 105, 125, 145], repeatRows=1)
        effect_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEAF7")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#7A8793")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.extend(
            [effect_table, Spacer(1, 12), Paragraph("Решение и ограничения", styles["Heading2"])]
        )
        summary = passport.decision.get("trajectory_summary")
        if summary:
            story.append(
                Paragraph(
                    f"Сводка α-траектории: "
                    f"{_PDF_STATUS_LABELS.get(summary['status'], summary['status'])}; "
                    f"условное действие: "
                    f"{summary.get('selected_action') or 'не назначено'}. {summary['reason']}",
                    styles["BodyText"],
                )
            )
        for limitation in passport.assumptions_and_limitations:
            story.append(Paragraph(f"• {limitation}", styles["BodyText"]))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Валидация: {passport.validation}", styles["BodyText"]))
        document.build(story)
        return target

    @staticmethod
    def zip_directory(directory: str | Path, target: str | Path) -> Path:
        source = Path(directory)
        output = Path(target)
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(source))
        return output
