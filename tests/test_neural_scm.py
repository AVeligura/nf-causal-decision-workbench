from __future__ import annotations

import math

from domain import GeneratorConfig
from engine import NeuralSCMConfig, fit_neural_scm
from engine.graphs import reference_graphs
from study import generate_dataset


def test_neural_scm_respects_structural_zero():
    generated = generate_dataset(GeneratorConfig(mode="laboratory", sample_size=300))
    graph = next(item for item in reference_graphs() if item.graph_id == "G2")
    result = fit_neural_scm(
        generated.data,
        graph,
        "Y_CR",
        NeuralSCMConfig(ensemble_size=2, max_epochs=20, patience=4),
    )
    assert result.backend.startswith("PyTorch")
    assert math.isfinite(result.ate)
    assert abs(result.ate) < 1e-10
    assert len(result.histories) == 2
    assert set(result.histories[0]["mechanisms"]) == {"L", "D", "Y"}
