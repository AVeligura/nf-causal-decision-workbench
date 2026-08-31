from __future__ import annotations

from domain import GeneratorConfig
from engine import run_analysis
from engine.evidence import reference_evidence
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
    assert result.passport.passport_version == "1.1"
    assert result.passport.validation["valid"] is True
    assert result.passport.validation["canonical_config"] == result.config.model_dump(mode="json")
    assert len(result.passport.validation["evidence_hash"]) == 64
    assert len(result.passport.validation["analysis_input_hash"]) == 64
    assert (
        result.passport.structural_space["compatibility_semantics"]
        == "fuzzy_membership_not_probability"
    )


def test_evidence_changes_have_separate_input_identity():
    config = GeneratorConfig(mode="laboratory", sample_size=300, seed=77)
    generated = generate_dataset(config)
    reference = reference_evidence()
    first = reference.items[0].model_copy(update={"support": 0.75})
    modified = reference.model_copy(update={"items": (first, *reference.items[1:]), "version": "custom"})

    baseline = run_analysis(
        generated.config,
        data=generated.data,
        evidence_bundle=reference,
        compute_cate=False,
    )
    changed = run_analysis(
        generated.config,
        data=generated.data,
        evidence_bundle=modified,
        compute_cate=False,
    )

    assert baseline.manifest.config_hash == changed.manifest.config_hash
    assert baseline.manifest.data_hash == changed.manifest.data_hash
    assert baseline.passport.validation["evidence_hash"] != changed.passport.validation["evidence_hash"]
    assert (
        baseline.passport.validation["analysis_input_hash"]
        != changed.passport.validation["analysis_input_hash"]
    )


def test_not_identified_branches_survive_in_passport():
    result = _analysis("outside_gamma")
    statuses = {
        item["status"]
        for item in result.passport.identification_and_estimation["graph_specific_results"]
    }
    assert statuses == {"not_identified"}
    assert result.passport.validation["valid"] is True
