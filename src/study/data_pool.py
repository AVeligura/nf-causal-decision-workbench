from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from domain import DatasetSpec

FEATURE_LABELS = {
    "A1": "Чистая прибыль / совокупные активы",
    "A2": "Обязательства / совокупные активы",
    "A4": "Оборотные активы / краткосрочные обязательства",
    "A5": "Прокси запаса денежных средств, дни",
    "A21": "Динамика продаж",
    "A27": "Операционная прибыль / финансовые расходы",
    "A29": "Логарифм совокупных активов",
    "A44": "Продолжительность оборота дебиторской задолженности, дни",
}


@dataclass(frozen=True)
class UCIDataPool:
    standardized: pd.DataFrame
    raw: pd.DataFrame
    medians: dict[str, float]
    centers: dict[str, float]
    scales: dict[str, float]
    spec: DatasetSpec


def _default_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "uci_polish" / "data.csv"


@lru_cache(maxsize=2)
def load_uci_pool(path: str | Path | None = None) -> UCIDataPool:
    source = Path(path) if path is not None else _default_data_path()
    if not source.exists():
        raise FileNotFoundError(
            f"Локальная копия UCI Polish Companies Bankruptcy не найдена: {source}. "
            "Запустите scripts/fetch_uci_data.py."
        )
    spec = DatasetSpec()
    frame = pd.read_csv(source, usecols=[*spec.selected_features, "year", "class"])
    selected = frame.loc[:, spec.selected_features].replace([np.inf, -np.inf], np.nan)
    medians = selected.median(numeric_only=True).to_dict()
    filled = selected.fillna(medians)
    lower = filled.quantile(0.01)
    upper = filled.quantile(0.99)
    winsorized = filled.clip(lower=lower, upper=upper, axis=1)
    centers = winsorized.median().to_dict()
    q75 = winsorized.quantile(0.75)
    q25 = winsorized.quantile(0.25)
    scales_series = (q75 - q25).replace(0, 1.0)
    standardized = (winsorized - pd.Series(centers)) / scales_series
    standardized.columns = [name.replace("A", "X") for name in standardized.columns]
    raw = frame.copy()
    return UCIDataPool(
        standardized=standardized.astype(float),
        raw=raw,
        medians={str(k): float(v) for k, v in medians.items()},
        centers={str(k): float(v) for k, v in centers.items()},
        scales={str(k): float(v) for k, v in scales_series.to_dict().items()},
        spec=spec,
    )
