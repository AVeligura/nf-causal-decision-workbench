from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from domain import ActionSpec, GeneratorConfig, ValueModelConfig
from domain.truth import TruthBundle
from domain.value_model import (
    PilotGraphState,
    evaluate_two_stage_pilot,
    point_action_values,
    point_value_inputs,
)

from .data_pool import UCIDataPool, load_uci_pool


@dataclass(frozen=True)
class GeneratedDataset:
    data: pd.DataFrame
    truth: TruthBundle | None
    config: GeneratorConfig
    diagnostics: dict[str, float | str]


def _clip_to_requested_range(raw: np.ndarray, lower: float, upper: float) -> np.ndarray:
    ranks = pd.Series(raw).rank(method="average", pct=True).to_numpy()
    return np.asarray(lower + (upper - lower) * ranks, dtype=float)


def value_config_from_generator(config: GeneratorConfig) -> ValueModelConfig:
    return ValueModelConfig(
        cr_weight=config.cr_weight,
        financing_weight=config.financing_weight,
        arrears_weight=config.arrears_weight,
        sales_loss_weight=config.sales_loss_weight,
        zombie_weight=config.zombie_weight,
        pilot_information_cost=config.pilot_information_cost,
        conditional_regret_threshold=config.conditional_regret_threshold,
        multiplier=config.value_multiplier,
        pilot_share=config.pilot_share,
        program_cost_a1=config.program_cost_a1,
        program_cost_a2=config.program_cost_a2,
        sales_loss_per_full_coverage=config.sales_loss_scale,
        zombie_risk_per_full_coverage=config.zombie_risk_scale,
    )


def action_specs_from_generator(config: GeneratorConfig) -> tuple[ActionSpec, ...]:
    return (
        ActionSpec(action_id="a0", name="Отказ от программы", coverage=0.0, program_cost=0.0),
        ActionSpec(
            action_id="a1",
            name="Ограниченный пилот",
            coverage=config.pilot_share,
            program_cost=config.program_cost_a1,
        ),
        ActionSpec(
            action_id="a2",
            name="Полномасштабное внедрение",
            coverage=1.0,
            program_cost=config.program_cost_a2,
        ),
    )


def _fixed_cate_control(pool: UCIDataPool) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the common 10k covariate population and its heterogeneity driver."""
    rng = np.random.default_rng(20260814 + 10_000)
    indices = rng.integers(0, len(pool.standardized), size=10_000)
    control = pool.standardized.iloc[indices].reset_index(drop=True).copy()
    shock = rng.normal(0.0, 1.0, len(control))
    cash_gap = (
        0.45 * control["X2"].to_numpy()
        - 0.35 * control["X4"].to_numpy()
        - 0.18 * control["X1"].to_numpy()
        + 0.12 * control["X44"].to_numpy()
        + 0.10 * shock
    )
    driver = np.tanh(
        0.8 * control["X4"].to_numpy() - 0.7 * control["X2"].to_numpy() - 0.25 * cash_gap
    )
    return control, driver


def generate_dataset(config: GeneratorConfig, pool: UCIDataPool | None = None) -> GeneratedDataset:
    if config.preset_scenario != config.scenario:
        from .scenarios import apply_scenario

        config = apply_scenario(config)
    pool = pool or load_uci_pool()
    seed_sequence = np.random.SeedSequence(config.seed)
    data_rng, assignment_rng, version_rng, loss_rng, evidence_rng, estimator_rng, pilot_rng = [
        np.random.default_rng(child) for child in seed_sequence.spawn(7)
    ]
    del evidence_rng, estimator_rng, pilot_rng

    indices = data_rng.integers(0, len(pool.standardized), size=config.sample_size)
    x = pool.standardized.iloc[indices].reset_index(drop=True).copy()
    n = len(x)
    x1, x2, x4, _x5 = (x[name].to_numpy() for name in ("X1", "X2", "X4", "X5"))
    x21, _x27, x29, x44 = (
        x[name].to_numpy() for name in ("X21", "X27", "X29", "X44")
    )
    u = data_rng.normal(0.0, 1.0, n)
    common_shock = data_rng.normal(0.0, config.noise_scale, n)
    cash_gap = 0.45 * x2 - 0.35 * x4 - 0.18 * x1 + 0.12 * x44 + 0.10 * common_shock

    assignment_index = config.assignment_strength * (
        0.55 * x2 - 0.45 * x4 - 0.20 * x1 + 0.15 * cash_gap + 0.08 * x29
    )
    if config.scenario == "outside_gamma":
        assignment_index += config.hidden_confounding * 0.55 * u
    propensity_raw = expit(assignment_index)
    propensity = np.clip(propensity_raw, config.propensity_lower, config.propensity_upper)
    assigned = assignment_rng.binomial(1, propensity, n)

    version = np.full(n, "base", dtype=object)
    version_draw = version_rng.random(n)
    assigned_mask = assigned == 1
    partial_boundary = config.partial_share
    refusal_boundary = partial_boundary + config.refusal_share
    version[assigned_mask & (version_draw < partial_boundary)] = "partial"
    version[
        assigned_mask & (version_draw >= partial_boundary) & (version_draw < refusal_boundary)
    ] = "refused"
    version[assigned_mask & (version_draw >= refusal_boundary)] = "full"
    dose = np.where(version == "full", 1.0, np.where(version == "partial", 0.45, 0.0))
    treatment = (version == "full").astype(int)

    nonlinear_term = config.nonlinearity * (
        0.12 * np.tanh(x4 - x2) + 0.04 * np.sin(np.clip(x21, -3, 3))
    )
    hetero_driver = np.tanh(0.8 * x4 - 0.7 * x2 - 0.25 * cash_gap)
    direct_cr = config.effect_scale * (
        0.10 + 0.055 * config.heterogeneity * hetero_driver
    )
    direct_cfo = config.effect_scale * (
        0.015 + 0.010 * config.heterogeneity * hetero_driver
    )
    t_to_l = config.effect_scale * (
        0.25 + 0.05 * config.heterogeneity * hetero_driver
    )

    l_noise = data_rng.normal(0, 0.18 * config.noise_scale, n)
    d_noise = data_rng.normal(0, 0.15 * config.noise_scale, n)
    ycr_noise = data_rng.normal(0, 0.055 * config.noise_scale, n)
    ycfo_noise = data_rng.normal(0, 0.012 * config.noise_scale, n)

    l_base = 1.25 + 0.30 * x4 - 0.18 * x2 + nonlinear_term + l_noise
    d_base = 0.65 + 0.32 * x2 - 0.12 * x1 + d_noise
    hidden_shift = (
        config.hidden_confounding * 0.035 * u if config.scenario == "outside_gamma" else 0.0
    )
    mechanism_graph = config.true_graph_id or "G1"
    ycr_base = 1.10 + 0.05 * x1 + nonlinear_term + ycr_noise + hidden_shift
    ycfo_base = (
        0.04
        + 0.010 * x1
        + 0.20 * nonlinear_term
        + ycfo_noise
        + 0.4 * hidden_shift
    )
    if mechanism_graph == "G2":
        # D precedes L. T changes L, but neither D nor Y is a descendant of L;
        # identical exogenous disturbances therefore give Y(1)=Y(0) exactly.
        d0 = d_base
        d1 = d_base
        l0 = l_base - 0.20 * d0
        l1 = l_base - 0.20 * d1 + t_to_l
        ycr0 = ycr_base - 0.22 * d0
        ycr1 = ycr_base - 0.22 * d1
        ycfo0 = ycfo_base - 0.025 * d0
        ycfo1 = ycfo_base - 0.025 * d1
    else:
        l0 = l_base
        l1 = l_base + t_to_l
        d0 = d_base - 0.20 * l0
        d1 = d_base - 0.20 * l1
        cr_l_coefficient = 0.28 if mechanism_graph in {"G3", "G4"} else 0.0
        cfo_l_coefficient = 0.035 if mechanism_graph in {"G3", "G4"} else 0.0
        cr_direct = direct_cr if mechanism_graph in {"G1", "G3"} else 0.0
        cfo_direct = direct_cfo if mechanism_graph in {"G1", "G3"} else 0.0
        ycr0 = ycr_base - 0.22 * d0 + cr_l_coefficient * l0
        ycr1 = ycr_base - 0.22 * d1 + cr_l_coefficient * l1 + cr_direct
        ycfo0 = ycfo_base - 0.025 * d0 + cfo_l_coefficient * l0
        ycfo1 = ycfo_base - 0.025 * d1 + cfo_l_coefficient * l1 + cfo_direct

    l_observed = l0 + dose * (l1 - l0)
    d_observed = d0 + dose * (d1 - d0)
    ycr_observed = ycr0 + dose * (ycr1 - ycr0)
    ycfo_observed = ycfo0 + dose * (ycfo1 - ycfo0)

    loss_signal = np.zeros(n)
    if config.scenario == "informative_loss":
        loss_signal = (
            0.8 * (d_observed - np.median(d_observed))
            - 0.6 * (l_observed - np.median(l_observed))
            - 0.8 * (ycfo_observed - np.median(ycfo_observed))
        )
    loss_prob = _clip_to_requested_range(
        expit(loss_signal), 0.005, min(0.70, 2 * config.missing_share)
    )
    scale = config.missing_share / max(float(loss_prob.mean()), 1e-9)
    loss_prob = np.clip(loss_prob * scale, 0.0, 0.95)
    observed = 1 - loss_rng.binomial(1, loss_prob, n)

    result = x.copy()
    result.insert(0, "row_id", np.arange(n, dtype=int))
    result["A"] = assigned
    result["T"] = treatment
    result["V"] = version
    result["dose"] = dose
    result["propensity_true"] = propensity
    result["cash_gap"] = cash_gap
    result["L"] = l_observed
    result["D"] = d_observed
    result["S"] = observed
    result["Y_CR"] = np.where(observed == 1, ycr_observed, np.nan)
    result["Y_CFO"] = np.where(observed == 1, ycfo_observed, np.nan)
    value_config = value_config_from_generator(config)
    financing_reduction_full = value_config.financing_reduction_ratio * (ycfo1 - ycfo0)
    arrears_reduction_full = value_config.arrears_reduction_ratio * (ycr1 - ycr0)
    # V3 declares these two quantities external calibrated parameters. The DGP
    # and estimated value branch use exactly the same values and coverage units.
    sales_loss_full = np.full(n, value_config.sales_loss_per_full_coverage)
    zombie_risk_full = np.full(n, value_config.zombie_risk_per_full_coverage)
    result["financing_cost_reduction"] = np.where(
        observed == 1, dose * financing_reduction_full, np.nan
    )
    result["arrears_reduction"] = np.where(
        observed == 1, dose * arrears_reduction_full, np.nan
    )
    result["sales_loss"] = np.where(observed == 1, dose * sales_loss_full, np.nan)
    result["zombie_risk"] = np.where(observed == 1, dose * zombie_risk_full, np.nan)

    control_x, control_driver = _fixed_cate_control(pool)
    result.attrs["cate_control_x"] = control_x
    control_t_to_l = config.effect_scale * (
        0.25 + 0.05 * config.heterogeneity * control_driver
    )
    control_direct_cr = config.effect_scale * (
        0.10 + 0.055 * config.heterogeneity * control_driver
    )
    control_direct_cfo = config.effect_scale * (
        0.015 + 0.010 * config.heterogeneity * control_driver
    )
    if mechanism_graph == "G2":
        control_true_cate = {
            "Y_CR": np.zeros_like(control_driver),
            "Y_CFO": np.zeros_like(control_driver),
        }
    else:
        cr_l = 0.28 if mechanism_graph in {"G3", "G4"} else 0.0
        cfo_l = 0.035 if mechanism_graph in {"G3", "G4"} else 0.0
        cr_direct = control_direct_cr if mechanism_graph in {"G1", "G3"} else 0.0
        cfo_direct = control_direct_cfo if mechanism_graph in {"G1", "G3"} else 0.0
        control_true_cate = {
            "Y_CR": (-0.22 * -0.20 + cr_l) * control_t_to_l + cr_direct,
            "Y_CFO": (-0.025 * -0.20 + cfo_l) * control_t_to_l + cfo_direct,
        }

    true_ate = {"Y_CR": float(np.mean(ycr1 - ycr0)), "Y_CFO": float(np.mean(ycfo1 - ycfo0))}
    treated = treatment == 1
    true_att = {
        "Y_CR": float(np.mean((ycr1 - ycr0)[treated])) if treated.any() else float("nan"),
        "Y_CFO": float(np.mean((ycfo1 - ycfo0)[treated])) if treated.any() else float("nan"),
    }
    true_value_inputs = point_value_inputs(
        cr=true_ate["Y_CR"],
        cfo=true_ate["Y_CFO"],
        financing_cost_reduction=float(np.mean(financing_reduction_full)),
        arrears_reduction=float(np.mean(arrears_reduction_full)),
        sales_loss=float(np.mean(sales_loss_full)),
        zombie_risk=float(np.mean(zombie_risk_full)),
    )
    action_values = point_action_values(
        action_specs_from_generator(config),
        true_value_inputs,
        value_config,
    )
    truth_design_se = {
        "Y_CR": float(
            np.sqrt(2.0 * (np.var(ycr0, ddof=1) + np.var(ycr1, ddof=1)) / n)
        ),
        "Y_CFO": float(
            np.sqrt(2.0 * (np.var(ycfo0, ddof=1) + np.var(ycfo1, ddof=1)) / n)
        ),
    }
    truth_pilot_seed = int(
        config.truth_pilot_seed
        if config.truth_pilot_seed is not None
        else (config.seed ^ 0xA17E5EED) & 0xFFFFFFFF
    )
    truth_pilot = evaluate_two_stage_pilot(
        (
            PilotGraphState(
                graph_id=mechanism_graph if config.scenario != "outside_gamma" else "OUTSIDE",
                cr_mean=true_ate["Y_CR"],
                cfo_mean=true_ate["Y_CFO"],
                cr_standard_error=truth_design_se["Y_CR"],
                cfo_standard_error=truth_design_se["Y_CFO"],
            ),
        ),
        value_config,
        seed=truth_pilot_seed,
    )
    truth_pilot_graph = next(iter(truth_pilot.total_by_graph))
    action_values["a1"] = truth_pilot.total_by_graph[truth_pilot_graph]
    admissible_actions = ("a0", "a2") + (("a1",) if truth_pilot.net_evi > 0 else ())
    optimal_action = max(
        admissible_actions, key=lambda action_id: action_values[action_id]
    )
    high_liquidity = x4 >= np.median(x4)
    high_debt = x2 >= np.median(x2)
    profile_masks = {
        "высокая L / низкий D": high_liquidity & ~high_debt,
        "высокая L / высокий D": high_liquidity & high_debt,
        "низкая L / низкий D": ~high_liquidity & ~high_debt,
        "низкая L / высокий D": ~high_liquidity & high_debt,
    }
    true_profiles = {
        outcome: {
            label: float(np.mean(cate[mask])) if mask.any() else float("nan")
            for label, mask in profile_masks.items()
        }
        for outcome, cate in {"Y_CR": ycr1 - ycr0, "Y_CFO": ycfo1 - ycfo0}.items()
    }
    truth = TruthBundle(
        true_graph_id=None if config.scenario == "outside_gamma" else mechanism_graph,
        potential_outcomes={
            "L": (l0, l1),
            "D": (d0, d1),
            "Y_CR": (ycr0, ycr1),
            "Y_CFO": (ycfo0, ycfo1),
            "financing_cost_reduction": (np.zeros(n), financing_reduction_full),
            "arrears_reduction": (np.zeros(n), arrears_reduction_full),
            "sales_loss": (np.zeros(n), sales_loss_full),
            "zombie_risk": (np.zeros(n), zombie_risk_full),
        },
        true_ate=true_ate,
        true_att=true_att,
        true_cate={"Y_CR": ycr1 - ycr0, "Y_CFO": ycfo1 - ycfo0},
        action_values=action_values,
        optimal_action=optimal_action,
        admissible_actions=admissible_actions,
        hidden_factor_present=config.scenario == "outside_gamma",
        metadata={
            "seed_streams": 7,
            "generator": "semi_synthetic_uci_v3_structural",
            "true_cate_profiles": true_profiles,
            "fixed_control_size": 10_000,
            "fixed_control_true_cate": control_true_cate,
            "value_model": value_config.as_dict(),
            "value_inputs": {
                "financing_cost_reduction": float(np.mean(financing_reduction_full)),
                "arrears_reduction": float(np.mean(arrears_reduction_full)),
                "sales_loss": float(np.mean(sales_loss_full)),
                "zombie_risk": float(np.mean(zombie_risk_full)),
            },
            "side_effect_semantics": "external_calibrated_same_formula_truth_and_estimate",
            "truth_design_standard_error": truth_design_se,
            "pilot": {
                "immediate_value": truth_pilot.immediate_by_graph[truth_pilot_graph],
                "continuation_without_information": (
                    truth_pilot.continuation_without_information_by_graph[truth_pilot_graph]
                ),
                "continuation_with_information": (
                    truth_pilot.continuation_with_information_by_graph[truth_pilot_graph]
                ),
                "gross_evi": truth_pilot.gross_evi,
                "information_cost": truth_pilot.information_cost,
                "net_evi": truth_pilot.net_evi,
                "r0": truth_pilot.robust_value_before_information,
                "r1": truth_pilot.robust_value_after_information,
                "adaptive_r1": truth_pilot.adaptive_robust_value,
                "common_evidence_method": truth_pilot.common_evidence_method,
                "inner_mcse": truth_pilot.inner_mcse,
                "seed": truth_pilot.seed,
                "samples": truth_pilot.samples,
            },
            "structural_edges": {
                "G1": (("T", "L"), ("L", "D"), ("D", "Y"), ("T", "Y")),
                "G2": (("T", "L"), ("D", "L"), ("D", "Y")),
                "G3": (
                    ("T", "L"),
                    ("L", "D"),
                    ("D", "Y"),
                    ("L", "Y"),
                    ("T", "Y"),
                ),
                "G4": (("T", "L"), ("L", "D"), ("D", "Y"), ("L", "Y")),
            }[mechanism_graph],
        },
    )
    diagnostics: dict[str, float | str] = {
        "n": n,
        "treatment_share": float(treatment.mean()),
        "partial_share_assigned": float(np.mean(version[assigned_mask] == "partial"))
        if assigned_mask.any()
        else 0.0,
        "refusal_share_assigned": float(np.mean(version[assigned_mask] == "refused"))
        if assigned_mask.any()
        else 0.0,
        "loss_share": float(1.0 - observed.mean()),
        "propensity_min": float(propensity.min()),
        "propensity_max": float(propensity.max()),
        "scenario": config.scenario,
        "value_regime": config.value_regime,
        "true_graph_id": "OUTSIDE" if config.scenario == "outside_gamma" else mechanism_graph,
    }
    return GeneratedDataset(data=result, truth=truth, config=config, diagnostics=diagnostics)
