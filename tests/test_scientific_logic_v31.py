from __future__ import annotations

from pathlib import Path

import pytest

from domain import EffectResult, GeneratorConfig, IdentificationStatus, ValueModelConfig
from domain.value_model import PilotGraphState, evaluate_two_stage_pilot
from engine.decision import evaluate_decisions
from engine.pipeline import run_analysis
from study.dgp import generate_dataset
from study.methods import MethodOutput
from study.metrics import evaluate_method
from study.scenarios import apply_scenario

REGRESSION_STATES = (
    PilotGraphState(
        graph_id="G1",
        cr_mean=0.12224324115827505,
        cfo_mean=0.003315385427027504,
        cr_standard_error=0.10575500477732054,
        cfo_standard_error=0.01479082497235015,
    ),
    PilotGraphState(
        graph_id="G2",
        cr_mean=-0.023343829296982015,
        cfo_mean=0.03384451569452905,
        cr_standard_error=0.05742858428241121,
        cfo_standard_error=0.02307809560067197,
    ),
    PilotGraphState(
        graph_id="G3",
        cr_mean=-0.03200176266469308,
        cfo_mean=0.02544202218772225,
        cr_standard_error=0.022480037245951205,
        cfo_standard_error=0.005498775981166368,
    ),
)


def _regression_config(**updates) -> ValueModelConfig:
    config = ValueModelConfig(
        pilot_share=0.20,
        program_cost_a1=0.004,
        program_cost_a2=0.020,
        pilot_information_cost=0.0002,
        pilot_virtual_samples=10_000,
    )
    return config.model_copy(update=updates)


def _effect(state: PilotGraphState, outcome: str) -> EffectResult:
    estimate = state.cr_mean if outcome == "Y_CR" else state.cfo_mean
    standard_error = state.cr_standard_error if outcome == "Y_CR" else state.cfo_standard_error
    return EffectResult(
        graph_id=state.graph_id,
        query_id=f"{state.graph_id}-{outcome}",
        outcome=outcome,
        estimand="ATE",
        status=IdentificationStatus.IDENTIFIED,
        estimate=estimate,
        interval=(estimate - 1.96 * standard_error, estimate + 1.96 * standard_error),
        standard_error=standard_error,
        diagnostics={"n_analysis": 200},
    )


def test_multigraph_preposterior_regression_uses_minimum_after_expectation():
    result = evaluate_two_stage_pilot(REGRESSION_STATES, _regression_config(), seed=1022)

    assert result.common_evidence_method == "shared_outcome_pilot_summary"
    assert result.robust_value_after_information >= result.robust_value_before_information
    assert result.gross_evi == pytest.approx(
        max(
            0.0,
            result.robust_value_after_information - result.robust_value_before_information,
        )
    )
    assert result.net_evi == pytest.approx(result.gross_evi - result.information_cost)
    # Frozen V3.1 regression values. They replace the inconsistent V3 pair
    # R0=0.0037043130 and R1=0.0032897443.
    assert result.robust_value_before_information == pytest.approx(0.003699232173834551)
    assert result.robust_value_after_information == pytest.approx(0.005161043675636608)
    assert result.gross_evi == pytest.approx(0.0014618115018020569)
    assert result.net_evi == pytest.approx(0.0012618115018020568)


@pytest.mark.parametrize("multiple_graphs", (False, True))
def test_pilot_is_reproducible_and_graph_order_invariant(multiple_graphs):
    states = REGRESSION_STATES if multiple_graphs else REGRESSION_STATES[:1]
    config = _regression_config(pilot_virtual_samples=500)
    first = evaluate_two_stage_pilot(states, config, seed=1022)
    second = evaluate_two_stage_pilot(tuple(reversed(states)), config, seed=1022)
    assert first == second
    assert first.robust_value_after_information >= first.robust_value_before_information


@pytest.mark.parametrize(
    ("cost_offset", "expected_sign"),
    ((-1e-6, 1), (0.0, 0), (1e-6, -1)),
)
def test_positive_boundary_and_negative_net_evi(cost_offset, expected_sign):
    base = evaluate_two_stage_pilot(
        REGRESSION_STATES, _regression_config(pilot_information_cost=0.0), seed=1022
    )
    result = evaluate_two_stage_pilot(
        REGRESSION_STATES,
        _regression_config(pilot_information_cost=base.gross_evi + cost_offset),
        seed=1022,
    )
    if expected_sign > 0:
        assert result.net_evi > 0
    elif expected_sign < 0:
        assert result.net_evi < 0
    else:
        assert result.net_evi == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("cost_offset", (-1e-6, 0.0, 1e-6))
def test_pilot_admissibility_has_exactly_the_sign_of_net_evi(cost_offset):
    base = evaluate_two_stage_pilot(
        REGRESSION_STATES, _regression_config(pilot_information_cost=0.0), seed=1022
    )
    config = _regression_config(pilot_information_cost=base.gross_evi + cost_offset)
    effects = tuple(
        _effect(state, outcome) for state in REGRESSION_STATES for outcome in ("Y_CR", "Y_CFO")
    )
    decision = evaluate_decisions(
        effects,
        tuple(state.graph_id for state in REGRESSION_STATES),
        alpha=0.0,
        value_config=config,
        seed=1022,
    )
    assert decision.pilot_admissible is (decision.pilot_net_evi > 0)
    if not decision.pilot_admissible:
        assert decision.selected_action != "a1"


def test_abstain_maps_to_a0_only_for_operational_metrics():
    config = apply_scenario(
        GeneratorConfig(mode="laboratory", scenario="informative_loss", sample_size=300)
    )
    generated = generate_dataset(config)
    analysis = run_analysis(config, data=generated.data, compute_cate=False)
    assert generated.truth is not None and generated.truth.optimal_action == "a0"
    forced = MethodOutput(
        method="forced_abstain_test",
        graph_ids=("G1",),
        effects=tuple(effect for effect in analysis.effects if effect.graph_id == "G1"),
        decision=analysis.decisions[0].model_copy(
            update={"selected_action": None, "status": "abstain"}
        ),
    )
    row = evaluate_method(
        analysis,
        forced,
        generated.truth,
        runtime_seconds=1.0,
        peak_memory_mb=100.0,
    )
    assert row["recommended_action"] is None
    assert row["decision_status"] == "abstain"
    assert row["operational_action"] == "a0"
    assert row["recommendation_exact_match"] is False
    assert row["operational_policy_accuracy"] is True
    assert row["optimal_action_selected"] is True
    assert row["erroneous_a2"] is False
    assert row["regret"] == pytest.approx(0.0)


def test_gui_end_to_end_runner_uses_widgets_and_never_injects_results():
    source = (
        Path(__file__).parents[1] / "scripts" / "run_gui_acceptance_v3_1_1.py"
    ).read_text(encoding="utf-8")
    assert "inject_result" not in source
    assert "run_analysis" not in source
    assert "generate_dataset" not in source
    assert "QTest.mouseClick" in source
    assert "QSignalSpy" in source
    assert "evidence_table" in source
    assert "replay_run_button" in source
