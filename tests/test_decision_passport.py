from __future__ import annotations

from domain import GeneratorConfig
from engine import run_analysis
from study import generate_dataset


def _analysis(scenario="reference"):
    config = GeneratorConfig(mode="laboratory", scenario=scenario, sample_size=300)
    generated = generate_dataset(config)
    return run_analysis(generated.config, data=generated.data, compute_cate=False)


def test_decisions_store_value_regret_and_worst_graph():
    result = _analysis()
    decision = result.decisions[-1]
    assert set(decision.values) == {"a0", "a1", "a2"}
    assert set(decision.regrets) == {"a0", "a1", "a2"}
    assert set(decision.maximum_regret) == {"a0", "a1", "a2"}
    assert decision.status in {"robust", "conditionally_robust", "pilot", "abstain"}


def test_passport_serializes_and_validates():
    result = _analysis()
    payload = result.passport.model_dump_json()
    assert (
        "CausalDecisionPassport" not in payload
    )  # object name is structural, not duplicated as data
    assert result.passport.validation["valid"] is True
    assert (
        result.passport.structural_space["compatibility_semantics"]
        == "fuzzy_membership_not_probability"
    )


def test_not_identified_branches_survive_in_passport():
    result = _analysis("outside_gamma")
    statuses = {
        item["status"]
        for item in result.passport.identification_and_estimation["graph_specific_results"]
    }
    assert statuses == {"not_identified"}
    assert result.passport.validation["valid"] is True
