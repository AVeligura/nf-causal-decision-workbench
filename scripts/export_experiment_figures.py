from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from visualization import (
    plot_method_comparison,
    plot_monte_carlo_distributions,
    plot_runtime_scaling,
    save_figure,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    raw = pd.read_parquet(args.experiment / "replicate_metrics.parquet")
    aggregate = pd.read_parquet(args.experiment / "aggregate_metrics.parquet")
    figures = {
        "method_comparison_regret": plot_method_comparison(aggregate, "regret"),
        "monte_carlo_regret": plot_monte_carlo_distributions(raw, "regret"),
        "runtime_scaling": plot_runtime_scaling(raw),
    }
    target = args.experiment / "figures"
    for name, figure in figures.items():
        for suffix in ("png", "svg", "pdf"):
            save_figure(figure, target / f"{name}.{suffix}")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
