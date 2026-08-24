from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from domain import GeneratorConfig
from engine import NeuralSCMConfig, fit_neural_scm, run_analysis
from engine.graphs import reference_graphs
from runtime import ExportManager, RunRepository
from study import generate_dataset
from visualization import (
    plot_alpha_cascade,
    plot_graph_specific_forest,
    plot_stability_map,
    plot_value_regret,
    save_figure,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path("artifacts/reference_repository"))
    parser.add_argument("--no-cate", action="store_true")
    parser.add_argument("--no-neural", action="store_true")
    args = parser.parse_args()
    config_path = Path(__file__).resolve().parents[1] / "configs" / "reference.yaml"
    config = GeneratorConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    generated = generate_dataset(config)
    result = run_analysis(config, data=generated.data, compute_cate=not args.no_cate)
    repository = RunRepository(args.repository)
    run_dir = repository.save(result, generated.data)
    if not args.no_neural:
        neural = fit_neural_scm(
            generated.data,
            reference_graphs()[0],
            "Y_CR",
            NeuralSCMConfig(base_seed=config.seed),
        )
        repository.attach_json(
            result.manifest.run_id,
            "results/neural_scm_G1_Y_CR.json",
            neural.as_dict(),
        )
    ExportManager.export_table(generated.data, run_dir / "results" / "generated_data.csv")
    ExportManager.export_table(generated.data, run_dir / "results" / "generated_data.xlsx")
    ExportManager.export_table(generated.data, run_dir / "results" / "generated_data.parquet")
    for graph in reference_graphs():
        ExportManager.export_graphml(graph, run_dir / "evidence" / f"{graph.graph_id}.graphml")
    figures = {
        "graph_specific_forest_Y_CR": plot_graph_specific_forest(result.effects, "Y_CR"),
        "alpha_cascade": plot_alpha_cascade(result.alpha_cuts),
        "stability_map": plot_stability_map(result.stability),
        "value_regret": plot_value_regret(result.trajectory_summary.operational_decision),
    }
    for name, figure in figures.items():
        for suffix in ("png", "svg", "pdf"):
            save_figure(figure, run_dir / "figures" / f"{name}.{suffix}")
    # Figures changed the package; refresh checksums by saving the same immutable result.
    repository.save(result, generated.data)
    print(run_dir)
    print(result.passport.validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
