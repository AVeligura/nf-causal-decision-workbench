from __future__ import annotations

import numpy as np
import pytest

from domain import GeneratorConfig, ValueModelConfig
from domain.value_model import (
    PilotGraphState,
    estimated_value_inputs,
    evaluate_two_stage_pilot,
)
from engine.graphs import has_directed_path, reference_graphs
from study.dgp import generate_dataset, value_config_from_generator
from study.monte_carlo import FULL_DESIGN, r3_design_rows
from study.scenarios import apply_scenario, apply_value_regime

EXPECTED_ENDOGENOUS_EDGES = {
    "G1": {("T", "L"), ("L", "D"), ("D", "Y"), ("T", "Y")},
    "G2": {("T", "L"), ("D", "L"), ("D", "Y")},
    "G3": {("T", "L"), ("L", "D"), ("D", "Y"), ("L", "Y"), ("T", "Y")},
    "G4": {("T", "L"), ("L", "D"), ("D", "Y"), ("L", "Y")},
}


def _config(graph_id: str, regime: str = "favorable", seed: int = 991):
    config = apply_scenario(
        GeneratorConfig(
            mode="laboratory",
            scenario="reference",
            sample_size=600,
            seed=seed,
            true_graph_id=graph_id,
        ),
        apply_default_value_regime=False,
    )
    return apply_value_regime(config, regime)


def test_r3_design_is_fixed_balanced_and_has_444_rows():
    rows = r3_design_rows(FULL_DESIGN, repetitions_per_subcell=4)
    assert len(rows) == 444
    assert len({row.design_id for row in rows}) == 444
    inside = [row for row in rows if row.scenario != "outside_gamma"]
    outside = [row for row in rows if row.scenario == "outside_gamma"]
    assert len(inside) == 432 and len(outside) == 12
    assert {row.true_graph_id for row in inside} == {"G1", "G2", "G3", "G4"}
    assert {row.true_graph_id for row in outside} == {None}
    for cell in FULL_DESIGN[:-1]:
        subset = [
            row
            for row in rows
            if row.scenario == cell.scenario and row.sample_size == cell.sample_size
        ]
        counts = {}
        for row in subset:
            counts[(row.true_graph_id, row.value_regime)] = (
                counts.get((row.true_graph_id, row.value_regime), 0) + 1
            )
        assert set(counts.values()) == {4}
    # Factors are explicit; changing a stream seed cannot change their labels.
    assert all(row.seed != row.pilot_seed != row.truth_pilot_seed for row in rows)


def test_graph_edges_and_paths_match_the_registered_structures():
    for graph in reference_graphs():
        endogenous = {edge for edge in graph.edges if edge[0] in {"T", "L", "D"}}
        assert endogenous == EXPECTED_ENDOGENOUS_EDGES[graph.graph_id]
        assert has_directed_path(graph, "T", "Y") is (graph.graph_id != "G2")


@pytest.mark.parametrize("graph_id", ("G1", "G2", "G3", "G4"))
def test_dgp_structural_interventions_follow_the_selected_graph(graph_id):
    truth = generate_dataset(_config(graph_id)).truth
    assert truth is not None and truth.true_graph_id == graph_id
    l0, l1 = truth.potential_outcomes["L"]
    d0, d1 = truth.potential_outcomes["D"]
    y0, y1 = truth.potential_outcomes["Y_CR"]
    assert np.mean(np.abs(l1 - l0)) > 0
    if graph_id == "G2":
        np.testing.assert_array_equal(d1, d0)
        np.testing.assert_array_equal(y1, y0)
        assert truth.true_ate["Y_CR"] == 0.0
        assert truth.true_ate["Y_CFO"] == 0.0
    else:
        assert np.mean(np.abs(d1 - d0)) > 0
        assert np.mean(np.abs(y1 - y0)) > 0


def test_side_effect_semantics_and_units_are_identical_in_truth_and_estimation():
    generated = generate_dataset(_config("G3", "boundary"))
    truth = generated.truth
    assert truth is not None
    config = value_config_from_generator(generated.config)
    cr = truth.true_ate["Y_CR"]
    cfo = truth.true_ate["Y_CFO"]
    values = truth.metadata["value_inputs"]
    assert values["financing_cost_reduction"] == pytest.approx(
        config.financing_reduction_ratio * cfo
    )
    assert values["arrears_reduction"] == pytest.approx(config.arrears_reduction_ratio * cr)
    assert values["sales_loss"] == pytest.approx(config.sales_loss_per_full_coverage)
    assert values["zombie_risk"] == pytest.approx(config.zombie_risk_per_full_coverage)
    estimated = estimated_value_inputs((cr, cr), (cfo, cfo), config)
    assert estimated.financing_cost_reduction[0] == pytest.approx(
        values["financing_cost_reduction"]
    )
    assert estimated.arrears_reduction[0] == pytest.approx(values["arrears_reduction"])
    assert estimated.sales_loss[0] == pytest.approx(values["sales_loss"])
    assert estimated.zombie_risk[0] == pytest.approx(values["zombie_risk"])


def test_true_a1_is_inadmissible_when_true_net_evi_is_nonpositive():
    for graph_id in ("G1", "G2", "G3", "G4"):
        truth = generate_dataset(_config(graph_id, "unfavorable")).truth
        assert truth is not None
        if truth.metadata["pilot"]["net_evi"] <= 0:
            assert "a1" not in truth.admissible_actions
            assert truth.optimal_action != "a1"


def test_true_optimum_and_pilot_components_reproduce_at_fixed_seed():
    first = generate_dataset(_config("G4", "boundary", seed=771)).truth
    second = generate_dataset(_config("G4", "boundary", seed=771)).truth
    assert first is not None and second is not None
    assert first.optimal_action == second.optimal_action
    assert first.action_values == second.action_values
    assert first.metadata["pilot"] == second.metadata["pilot"]


def test_two_stage_pilot_reports_registered_components():
    config = ValueModelConfig(
        pilot_share=0.25,
        program_cost_a1=0.002,
        program_cost_a2=0.020,
        pilot_information_cost=0.0001,
        pilot_virtual_samples=400,
    )
    result = evaluate_two_stage_pilot(
        (
            PilotGraphState(
                graph_id="G1",
                cr_mean=0.08,
                cfo_mean=0.01,
                cr_standard_error=0.08,
                cfo_standard_error=0.02,
            ),
        ),
        config,
        seed=17,
    )
    assert set(result.immediate_by_graph) == {"G1"}
    assert set(result.continuation_with_information_by_graph) == {"G1"}
    assert result.net_evi == pytest.approx(result.gross_evi - result.information_cost)
    assert result.inner_mcse is not None
    assert result.samples == 400


def test_scenario_and_value_regime_can_be_crossed_independently():
    weak = apply_scenario(
        GeneratorConfig(scenario="weak_overlap", sample_size=300),
        apply_default_value_regime=False,
    )
    favorable = apply_value_regime(weak, "favorable")
    unfavorable = apply_value_regime(weak, "unfavorable")
    assert favorable.scenario == unfavorable.scenario == "weak_overlap"
    assert favorable.propensity_lower == unfavorable.propensity_lower == 0.03
    assert favorable.program_cost_a2 != unfavorable.program_cost_a2
