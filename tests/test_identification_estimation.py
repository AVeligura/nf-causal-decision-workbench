from __future__ import annotations

from domain import CausalQuery, GeneratorConfig, IdentificationStatus
from engine import estimate_effect, identify_effect
from engine.graphs import reference_graphs
from study import generate_dataset


def test_g2_is_structural_zero_not_estimated_zero():
    graph = next(graph for graph in reference_graphs() if graph.graph_id == "G2")
    query = CausalQuery()
    identification = identify_effect(graph, query)
    assert identification.status == IdentificationStatus.STRUCTURAL_ZERO
    generated = generate_dataset(GeneratorConfig(mode="laboratory", sample_size=300))
    result = estimate_effect(generated.data, identification, outcome="Y_CR", compute_cate=False)
    assert result.estimate == 0.0
    assert result.interval is None
    assert result.diagnostics["estimated_from_sample"] is False


def test_informative_loss_is_partially_identified():
    graph = reference_graphs()[0]
    query = CausalQuery()
    identification = identify_effect(graph, query, scenario="informative_loss")
    assert identification.status == IdentificationStatus.PARTIALLY_IDENTIFIED
    generated = generate_dataset(
        GeneratorConfig(mode="laboratory", scenario="informative_loss", sample_size=300)
    )
    result = estimate_effect(generated.data, identification, outcome="Y_CR", compute_cate=False)
    assert result.estimate is None
    assert result.identified_bounds is not None
    assert result.bound_intervals is not None


def test_outside_gamma_is_not_identified_without_fake_value():
    identification = identify_effect(reference_graphs()[0], CausalQuery(), scenario="outside_gamma")
    assert identification.status == IdentificationStatus.NOT_IDENTIFIED
    generated = generate_dataset(
        GeneratorConfig(mode="laboratory", scenario="outside_gamma", sample_size=300)
    )
    result = estimate_effect(generated.data, identification, outcome="Y_CR", compute_cate=False)
    assert result.estimate is None
    assert result.interval is None
