from __future__ import annotations

from domain import GeneratorConfig

LABORATORY_PARAMETER_ROLES = {
    "scenario": "selects_statistical_generation_and_identification_preset",
    "value_regime": "selects_an_independent_economic_value_preset",
    "true_graph_id": "selects_the_structural_equations_inside_gamma",
    "sample_size": "changes_number_of_generated_units",
    "effect_scale": "changes_potential_outcomes",
    "heterogeneity": "changes_unit_level_effect_dispersion",
    "nonlinearity": "changes_outcome_response_surface",
    "noise_scale": "changes_observation_noise",
    "assignment_strength": "changes_treatment_propensity",
    "propensity_lower": "changes_true_propensity_support",
    "propensity_upper": "changes_true_propensity_support",
    "partial_share": "changes_treatment_version_distribution",
    "refusal_share": "changes_treatment_version_distribution",
    "missing_share": "changes_observation_indicator",
    "hidden_confounding": "changes_outcomes_in_outside_gamma_scenario",
    "evidence_reliability": "changes_graph_compatibility_scores",
    "evidence_conflict": "changes_conflict_penalty_and_graph_scores",
    "pilot_share": "changes_pilot_size_coverage_and_information_value",
    "value_multiplier": "changes_action_values",
    "sales_loss_scale": "changes_sales_loss_side_outcome",
    "zombie_risk_scale": "changes_zombie_risk_side_outcome",
    "program_cost_a1": "changes_pilot_value",
    "program_cost_a2": "changes_full_rollout_value",
    "cr_weight": "changes_action_values",
    "financing_weight": "changes_action_values",
    "arrears_weight": "changes_action_values",
    "sales_loss_weight": "changes_action_values",
    "zombie_weight": "changes_action_values",
    "pilot_information_cost": "changes_net_information_value",
    "conditional_regret_threshold": "changes_decision_classification",
    "alpha": "selects_diagnostic_slice_without_changing_generated_data",
}


SCENARIO_DEFAULT_VALUE_REGIME = {
    "reference": "favorable",
    "evidence_conflict": "favorable",
    "weak_overlap": "boundary",
    "version_mixing": "boundary",
    "informative_loss": "unfavorable",
    "outside_gamma": "unfavorable",
}

VALUE_REGIME_PARAMETERS = {
    "favorable": {
        "sales_loss_scale": 0.006,
        "zombie_risk_scale": 0.020,
        "program_cost_a1": 0.004,
        "program_cost_a2": 0.008,
    },
    "boundary": {
        "sales_loss_scale": 0.012,
        "zombie_risk_scale": 0.040,
        "program_cost_a1": 0.005,
        "program_cost_a2": 0.045,
    },
    "unfavorable": {
        "sales_loss_scale": 0.025,
        "zombie_risk_scale": 0.090,
        "program_cost_a1": 0.012,
        "program_cost_a2": 0.050,
    },
}


def apply_value_regime(config: GeneratorConfig, value_regime: str) -> GeneratorConfig:
    """Apply only economic parameters; never change the statistical scenario."""

    if value_regime not in VALUE_REGIME_PARAMETERS:
        raise ValueError(f"Unknown value regime: {value_regime}")
    values = config.model_dump()
    values.update(VALUE_REGIME_PARAMETERS[value_regime])
    values.update(value_regime=value_regime)
    return GeneratorConfig.model_validate(values)


def apply_scenario(
    config: GeneratorConfig, *, apply_default_value_regime: bool = True
) -> GeneratorConfig:
    """Apply a preset once when selected or when a fresh configuration is used.

    The generator calls this only when `preset_scenario != scenario`; subsequent
    manual edits keep matching values and therefore remain effective.
    """

    values = config.model_dump()
    scenario = config.scenario
    values.update(preset_scenario=scenario, customized=False)
    if scenario == "reference":
        values.update(
            propensity_lower=0.15,
            propensity_upper=0.85,
            partial_share=0.10,
            refusal_share=0.0,
            missing_share=0.05,
            hidden_confounding=0.0,
            evidence_conflict=0.0,
        )
    elif scenario == "evidence_conflict":
        values.update(
            evidence_conflict=0.85,
        )
    elif scenario == "weak_overlap":
        values.update(
            propensity_lower=0.03,
            propensity_upper=0.97,
            assignment_strength=2.2,
        )
    elif scenario == "version_mixing":
        values.update(
            partial_share=0.20,
            refusal_share=0.10,
        )
    elif scenario == "informative_loss":
        values.update(
            missing_share=0.25,
        )
    elif scenario == "outside_gamma":
        values.update(
            hidden_confounding=1.0,
        )
    output = GeneratorConfig.model_validate(values)
    if apply_default_value_regime:
        return apply_value_regime(output, SCENARIO_DEFAULT_VALUE_REGIME[scenario])
    return output
