from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domain import (
    AlphaCut,
    DecisionResult,
    EffectResult,
    GeneratorConfig,
    IdentificationStatus,
    ValueModelConfig,
)
from domain.value_model import (
    action_value_range,
    estimated_value_inputs,
    point_action_values,
    point_value_inputs,
)
from engine.decision import actions_from_value_config, evaluate_decisions
from engine.evidence import reference_evidence, score_graph
from engine.graphs import reference_graphs
from engine.pipeline import run_analysis
from engine.stability import summarize_alpha_trajectory
from runtime import DataImporter, RunRepository
from study.dgp import generate_dataset
from study.methods import MethodOutput, run_comparison_methods
from study.metrics import evaluate_method
from study.scenarios import LABORATORY_PARAMETER_ROLES, apply_scenario


def _effect(graph: str, outcome: str, estimate: float, se: float | None) -> EffectResult:
    interval = None if se is None else (estimate - 1.96 * se, estimate + 1.96 * se)
    return EffectResult(
        graph_id=graph,
        query_id=f"{graph}-{outcome}",
        outcome=outcome,
        estimand="ATE",
        status=IdentificationStatus.IDENTIFIED,
        estimate=estimate,
        interval=interval,
        standard_error=se,
        diagnostics={"n_analysis": 200},
    )


def _decision(alpha: float, action: str | None, status: str = "robust") -> DecisionResult:
    return DecisionResult(
        alpha=alpha,
        status=status,
        selected_action=action,
        values={"a0": {}, "a1": {}, "a2": {}},
        regrets={"a0": {}, "a1": {}, "a2": {}},
        maximum_regret={"a0": 0.02, "a1": 0.01, "a2": 0.0},
        worst_graph={"a0": "G1", "a1": "G1", "a2": "G1"},
    )


@pytest.fixture(scope="module")
def reference_result():
    config = apply_scenario(
        GeneratorConfig(
            mode="laboratory",
            scenario="reference",
            sample_size=300,
            cate_trees=20,
            crossfit_folds=2,
        )
    )
    return run_analysis(config, compute_cate=False)


def test_full_procedure_is_trajectory_summary_not_last_cut(reference_result):
    outputs = run_comparison_methods(reference_result)
    full = outputs["full_procedure"]
    assert full.decision is reference_result.trajectory_summary.operational_decision
    assert full.decision is not reference_result.decisions[-1]


def test_hard_set_is_invariant_to_mu_values_and_score_order(reference_result):
    original = run_comparison_methods(reference_result)["hard_set"].decision
    altered_scores = tuple(
        score.model_copy(update={"mu": 0.01 * (index + 1)})
        for index, score in enumerate(reversed(reference_result.graph_scores))
    )
    altered = reference_result.model_copy(update={"graph_scores": altered_scores})
    changed = run_comparison_methods(altered)["hard_set"].decision
    assert changed.selected_action == original.selected_action
    assert changed.status == original.status
    assert changed.maximum_regret == original.maximum_regret


def test_alpha_trajectory_detects_switching_and_trigger_graph():
    cuts = (
        AlphaCut(alpha=0.9, graph_ids=("G1",), core_edges=(), alternative_edges=()),
        AlphaCut(alpha=0.6, graph_ids=("G1", "G2"), core_edges=(), alternative_edges=()),
    )
    summary = summarize_alpha_trajectory(
        cuts,
        (_decision(0.9, "a2"), _decision(0.6, "a0")),
        (),
        value_config=ValueModelConfig(),
    )
    assert summary.status == "switching"
    assert summary.first_action_change_alpha == 0.6
    assert summary.first_action_change_graph == "G2"
    assert summary.operational_decision.selected_action is None


def test_true_and_estimated_values_use_one_equation():
    config = ValueModelConfig()
    actions = actions_from_value_config(config)
    estimated = estimated_value_inputs((0.1, 0.1), (0.02, 0.02), config)
    point = point_value_inputs(
        cr=0.1,
        cfo=0.02,
        financing_cost_reduction=0.02 * config.financing_reduction_ratio,
        arrears_reduction=0.1 * config.arrears_reduction_ratio,
        sales_loss=config.sales_loss_per_full_coverage,
        zombie_risk=config.zombie_risk_per_full_coverage,
    )
    point_values = point_action_values(actions, point, config)
    for action in actions:
        if action.action_id == "a0":
            continue
        low, high = action_value_range(action, estimated, config)
        assert low == pytest.approx(point_values[action.action_id])
        assert high == pytest.approx(point_values[action.action_id])


def test_oracle_value_uses_generated_side_outcomes():
    generated = generate_dataset(GeneratorConfig(sample_size=300))
    truth = generated.truth
    assert truth is not None
    for name in (
        "financing_cost_reduction",
        "arrears_reduction",
        "sales_loss",
        "zombie_risk",
    ):
        assert name in truth.potential_outcomes
        expected = float(np.mean(truth.potential_outcomes[name][1]))
        assert truth.metadata["value_inputs"][name] == pytest.approx(expected)


def test_virtual_pilot_is_selected_when_net_information_value_is_positive():
    effects = tuple(
        _effect("G1", outcome, 0.0, se)
        for outcome, se in (("Y_CR", 0.05), ("Y_CFO", 0.015))
    )
    config = ValueModelConfig(
        pilot_share=0.20,
        program_cost_a1=0.0,
        program_cost_a2=0.005,
        pilot_information_cost=0.0,
        pilot_virtual_samples=2000,
        conditional_regret_threshold=0.1,
        sales_loss_per_full_coverage=0.0,
        zombie_risk_per_full_coverage=0.0,
    )
    decision = evaluate_decisions(
        effects, ("G1",), alpha=0.8, value_config=config, seed=7
    )
    assert decision.status == "pilot"
    assert decision.selected_action == "a1"
    assert decision.pilot_information_value > 0
    assert decision.pilot_information_value_se is not None
    assert decision.pilot_seed == 7
    assert decision.pilot_virtual_samples == 2000
    assert decision.pilot_net_evi == decision.pilot_information_value


def test_virtual_pilot_not_selected_when_net_value_is_nonpositive_or_unavailable():
    effects = tuple(
        _effect(graph, outcome, estimate, 0.005)
        for graph in ("G1", "G2")
        for outcome, estimate in (("Y_CR", 0.10), ("Y_CFO", 0.02))
    )
    costly = ValueModelConfig(
        pilot_share=0.30,
        program_cost_a1=0.001,
        program_cost_a2=0.04,
        pilot_information_cost=0.05,
        conditional_regret_threshold=0.1,
    )
    decision = evaluate_decisions(effects, ("G1", "G2"), alpha=0.8, value_config=costly)
    assert decision.selected_action != "a1"
    unavailable = tuple(effect.model_copy(update={"standard_error": None}) for effect in effects)
    decision = evaluate_decisions(
        unavailable, ("G1", "G2"), alpha=0.8, value_config=costly
    )
    assert decision.pilot_expected_regret_reduction == 0.0
    assert decision.pilot_information_value <= 0.0
    assert decision.pilot_information_value_se is None
    assert decision.selected_action != "a1"


def test_erroneous_a2_metric_detects_harmful_rollout():
    config = apply_scenario(
        GeneratorConfig(mode="laboratory", scenario="informative_loss", sample_size=300)
    )
    generated = generate_dataset(config)
    analysis = run_analysis(config, data=generated.data, compute_cate=False)
    assert generated.truth.optimal_action == "a0"
    base = analysis.decisions[0]
    forced = MethodOutput(
        method="forced_a2_test",
        graph_ids=("G1",),
        effects=tuple(effect for effect in analysis.effects if effect.graph_id == "G1"),
        decision=base.model_copy(
            update={"selected_action": "a2", "status": "conditionally_robust"}
        ),
    )
    row = evaluate_method(
        analysis,
        forced,
        generated.truth,
        runtime_seconds=1.0,
        peak_memory_mb=100.0,
        shared_estimation_seconds=0.8,
    )
    assert row["erroneous_a2"] is True
    assert row["regret"] > 0


def test_reference_evidence_is_transparent_and_reproduces_targets():
    bundle = reference_evidence()
    assert {item.evidence_type for item in bundle.items} == {
        "panel",
        "algorithmic",
        "expert",
        "regulatory",
        "interventional",
        "external",
    }
    assert all(item.provenance and item.dependent_group and item.context for item in bundle.items)
    assert all(item.target_graph is None for item in bundle.items)
    assert any(item.assertion_kind == "edge" for item in bundle.items)
    assert any(item.assertion_kind == "path" for item in bundle.items)
    assert any(item.assertion_kind == "rule" for item in bundle.items)
    values = [score_graph(graph, bundle).mu for graph in reference_graphs()]
    assert values == pytest.approx([0.92, 0.81, 0.67, 0.43], abs=1e-8)


def test_manual_parameter_change_is_not_reset_by_generator():
    preset = apply_scenario(
        GeneratorConfig(mode="laboratory", scenario="weak_overlap", sample_size=300)
    )
    custom = preset.model_copy(
        update={"propensity_lower": 0.12, "propensity_upper": 0.88, "customized": True}
    )
    generated = generate_dataset(custom)
    assert generated.config.propensity_lower == 0.12
    assert generated.config.propensity_upper == 0.88
    assert generated.diagnostics["propensity_min"] >= 0.12 - 1e-12
    assert generated.diagnostics["propensity_max"] <= 0.88 + 1e-12


def test_every_editable_parameter_has_declared_computational_or_diagnostic_role():
    expected = {
        "scenario",
        "value_regime",
        "true_graph_id",
        "sample_size",
        "effect_scale",
        "heterogeneity",
        "nonlinearity",
        "noise_scale",
        "assignment_strength",
        "propensity_lower",
        "propensity_upper",
        "partial_share",
        "refusal_share",
        "missing_share",
        "hidden_confounding",
        "evidence_reliability",
        "evidence_conflict",
        "pilot_share",
        "value_multiplier",
        "sales_loss_scale",
        "zombie_risk_scale",
        "program_cost_a1",
        "program_cost_a2",
        "cr_weight",
        "financing_weight",
        "arrears_weight",
        "sales_loss_weight",
        "zombie_weight",
        "pilot_information_cost",
        "conditional_regret_threshold",
        "alpha",
    }
    assert set(LABORATORY_PARAMETER_ROLES) == expected
    assert all(LABORATORY_PARAMETER_ROLES[field] for field in expected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pilot_share", 0.35),
        ("value_multiplier", 1.3),
        ("sales_loss_scale", 0.020),
        ("zombie_risk_scale", 0.080),
        ("program_cost_a1", 0.006),
        ("program_cost_a2", 0.030),
        ("cr_weight", 0.20),
        ("financing_weight", 1.8),
        ("arrears_weight", 1.2),
        ("sales_loss_weight", 0.8),
        ("zombie_weight", 0.2),
    ],
)
def test_editable_value_parameters_change_oracle_action_values(field, value):
    baseline_config = GeneratorConfig(sample_size=300, seed=99)
    changed_config = baseline_config.model_copy(update={field: value, "customized": True})
    baseline = generate_dataset(baseline_config).truth
    changed = generate_dataset(changed_config).truth
    assert (
        changed.action_values != baseline.action_values
        or changed.metadata["pilot"] != baseline.metadata["pilot"]
    )


def test_imported_dataset_gets_real_metadata_not_uci_passport(tmp_path):
    path = tmp_path / "user_sample.csv"
    frame = pd.DataFrame({"T": [0, 1], "Y_CR": [1.0, np.nan], "X1": [0.1, 0.2]})
    frame.to_csv(path, index=False)
    mapping = {"T": "T", "Y_CR": "Y_CR", "X1": "X1"}
    spec = DataImporter.build_dataset_spec(path, frame, mapping, user_description="Тест")
    assert spec.kind == "imported"
    assert spec.source_file_name == "user_sample.csv"
    assert spec.file_format == "csv"
    assert spec.rows == 2 and spec.columns == 3
    assert spec.truth_available is False
    assert spec.doi is None
    assert spec.missingness["Y_CR"] == 0.5
    assert len(spec.checksum_sha256) == 64


def test_cate_name_matches_implementation(reference_result):
    config = reference_result.config.model_copy(update={"sample_size": 300, "cate_trees": 20})
    result = run_analysis(config, compute_cate=True)
    names = {
        effect.diagnostics.get("cate_model")
        for effect in result.effects
        if effect.diagnostics.get("cate_model")
    }
    assert names == {"DR-learner with ExtraTrees regression on pseudo-outcomes"}
    assert all("honest" not in name.lower() for name in names)


def test_reference_overlap_separates_truth_nuisance_and_weight_clipping(reference_result):
    effects = [
        effect
        for effect in reference_result.effects
        if effect.graph_id == "G1" and effect.status == IdentificationStatus.IDENTIFIED
    ]
    assert effects
    for effect in effects:
        diagnostics = effect.diagnostics
        assert diagnostics["propensity_true_min"] >= 0.15 - 1e-12
        assert diagnostics["propensity_true_max"] <= 0.85 + 1e-12
        assert "propensity_estimated_extreme_share" in diagnostics
        assert "propensity_weight_clipped_share" in diagnostics
        assert diagnostics["nuisance_treatment"] == "HistGradientBoostingClassifier"
        assert not any("Слабое overlap" in warning for warning in effect.warnings)


def test_timing_metrics_have_nonmisleading_names(reference_result):
    generated = generate_dataset(reference_result.config)
    output = run_comparison_methods(reference_result)["hard_set"]
    row = evaluate_method(
        reference_result,
        output,
        generated.truth,
        runtime_seconds=2.0,
        peak_memory_mb=128.0,
        shared_estimation_seconds=1.5,
    )
    assert row["replication_runtime_seconds"] == 2.0
    assert row["replication_peak_memory_mb"] == 128.0
    assert row["shared_estimation_seconds"] == 1.5
    assert "method_decision_seconds" in row
    assert "runtime_seconds" not in row


def test_replay_preserves_trajectory_and_pilot(tmp_path):
    config = apply_scenario(
        GeneratorConfig(
            mode="laboratory",
            scenario="reference",
            sample_size=300,
            cate_trees=20,
            crossfit_folds=2,
        )
    )
    generated = generate_dataset(config)
    result = run_analysis(config, data=generated.data, compute_cate=False)
    repository = RunRepository(tmp_path)
    repository.save(result, generated.data)
    replay = repository.replay(result.manifest.run_id)
    assert replay.matched, replay.differences


def test_imported_passport_marks_oracle_truth_na():
    generated = generate_dataset(GeneratorConfig(sample_size=300))
    config = generated.config.model_copy(update={"mode": "import", "profile_name": "imported"})
    # Pipeline fallback must never silently substitute the UCI reference passport.
    result = run_analysis(config, data=generated.data, compute_cate=False)
    assert result.dataset_spec.kind == "imported"
    assert result.dataset_spec.truth_available is False
    assert result.diagnostics["oracle_truth_available"] is False
    assert all(
        "Oracle truth недоступна" in warning
        for effect in result.effects
        if effect.status == IdentificationStatus.IDENTIFIED
        for warning in effect.warnings
    )
