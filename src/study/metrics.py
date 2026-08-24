from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from domain import AnalysisResult, EffectResult, IdentificationStatus
from domain.truth import TruthBundle

from .methods import MethodOutput


def _effect_for_evaluation(
    output: MethodOutput, outcome: str, truth: TruthBundle
) -> EffectResult | None:
    evaluation_graph = truth.true_graph_id
    if evaluation_graph is not None:
        exact = next(
            (
                effect
                for effect in output.effects
                if effect.graph_id == evaluation_graph and effect.outcome == outcome
            ),
            None,
        )
        if exact is not None:
            return exact
    return next((effect for effect in output.effects if effect.outcome == outcome), None)


def _profile_truth(analysis: AnalysisResult, truth: TruthBundle, outcome: str) -> dict[str, float]:
    # The DGP stores CATE per generated unit. Profile metrics use the same four
    # pre-registered strata; the fixed 10k population is handled by the full
    # experiment preparation script.
    cate = truth.true_cate[outcome]
    return {
        "mean": float(np.mean(cate)),
        "std": float(np.std(cate)),
    }


def evaluate_method(
    analysis: AnalysisResult,
    output: MethodOutput,
    truth: TruthBundle,
    *,
    runtime_seconds: float,
    peak_memory_mb: float,
    shared_estimation_seconds: float | None = None,
) -> dict[str, Any]:
    report_status = (
        analysis.trajectory_summary.status
        if output.method == "full_procedure"
        else output.decision.status
    )
    recommended_action = output.decision.selected_action
    operational_action = recommended_action or "a0"
    recommendation_exact_match = recommended_action == truth.optimal_action
    operational_policy_accuracy = operational_action == truth.optimal_action
    row: dict[str, Any] = {
        "method": output.method,
        "scenario": analysis.config.scenario,
        "sample_size": analysis.config.sample_size,
        "value_regime": analysis.config.value_regime,
        "true_graph_id": truth.true_graph_id,
        "seed": analysis.config.seed,
        "replication_runtime_seconds": runtime_seconds,
        "replication_peak_memory_mb": peak_memory_mb,
        "shared_estimation_seconds": shared_estimation_seconds,
        "method_decision_seconds": output.decision_runtime_seconds,
        "decision_status": output.decision.status,
        "recommended_action": recommended_action,
        # Retained as a replay-compatible alias for V2/V3 result readers.
        "selected_action": recommended_action,
        "operational_action": operational_action,
        "true_action": truth.optimal_action,
        "recommendation_exact_match": recommendation_exact_match,
        "operational_policy_accuracy": operational_policy_accuracy,
        # Headline accuracy now follows the implementable policy. The explicit
        # recommendation metric above preserves abstain versus robust a0.
        "optimal_action_selected": operational_policy_accuracy,
        "erroneous_a2": operational_action == "a2" and truth.optimal_action != "a2",
        "erroneous_a2_opportunity": truth.optimal_action != "a2",
        "report_status": report_status,
        "pilot": report_status == "pilot",
        "abstain": report_status == "abstain",
        "abstain_rate": output.decision.status == "abstain",
        "robust": report_status == "robust",
        "conditionally_robust": report_status == "conditionally_robust",
        "truth_applicable": True,
        "trajectory_status": (
            analysis.trajectory_summary.status
            if output.method == "full_procedure"
            else "not_applicable"
        ),
        "pilot_information_value": output.decision.pilot_information_value,
        "pilot_expected_regret_reduction": (
            output.decision.pilot_expected_regret_reduction
        ),
        "pilot_evi_inner_mcse": (
            output.decision.pilot_information_value_se
        ),
        "pilot_virtual_samples": output.decision.pilot_virtual_samples,
        "pilot_immediate_value": output.decision.pilot_immediate_value,
        "pilot_expected_continuation_value": (
            output.decision.pilot_expected_continuation_value
        ),
        "pilot_gross_evi": output.decision.pilot_gross_evi,
        "pilot_information_cost": output.decision.pilot_information_cost,
        "pilot_net_evi": output.decision.pilot_net_evi,
        "pilot_r0": output.decision.pilot_r0,
        "pilot_r1": output.decision.pilot_r1,
        "pilot_adaptive_r1": output.decision.pilot_adaptive_r1,
        "pilot_common_evidence_method": (
            output.decision.pilot_common_evidence_method
        ),
        "pilot_admissible": output.decision.pilot_admissible,
        "true_pilot_immediate_value": truth.metadata.get("pilot", {}).get(
            "immediate_value"
        ),
        "true_pilot_gross_evi": truth.metadata.get("pilot", {}).get("gross_evi"),
        "true_pilot_information_cost": truth.metadata.get("pilot", {}).get(
            "information_cost"
        ),
        "true_pilot_net_evi": truth.metadata.get("pilot", {}).get("net_evi"),
        "true_pilot_r0": truth.metadata.get("pilot", {}).get("r0"),
        "true_pilot_r1": truth.metadata.get("pilot", {}).get("r1"),
    }
    if output.method == "full_procedure":
        trajectory = analysis.trajectory_summary
        row.update(
            {
                "first_action_change_alpha": trajectory.first_action_change_alpha,
                "first_action_change_graph": trajectory.first_action_change_graph,
                "trajectory_first_identification_loss_alpha": (
                    trajectory.first_identification_loss_alpha
                ),
                "trajectory_first_sign_loss_alpha": trajectory.first_sign_loss_alpha,
                "trajectory_first_threshold_loss_alpha": (
                    trajectory.first_threshold_loss_alpha
                ),
                "trajectory_stable_alpha_low": (
                    trajectory.stable_alpha_range[0]
                    if trajectory.stable_alpha_range is not None
                    else None
                ),
                "trajectory_stable_alpha_high": (
                    trajectory.stable_alpha_range[1]
                    if trajectory.stable_alpha_range is not None
                    else None
                ),
            }
        )
    else:
        row.update(
            {
                "first_action_change_alpha": None,
                "first_action_change_graph": None,
                "trajectory_first_identification_loss_alpha": None,
                "trajectory_first_sign_loss_alpha": None,
                "trajectory_first_threshold_loss_alpha": None,
                "trajectory_stable_alpha_low": None,
                "trajectory_stable_alpha_high": None,
            }
        )
    row["switching"] = report_status == "switching"
    row["alpha_action_changed"] = row["first_action_change_alpha"] is not None
    row["policy_value"] = truth.action_values.get(operational_action, float("nan"))
    row["regret"] = max(
        0.0,
        truth.action_values[truth.optimal_action]
        - truth.action_values.get(operational_action, truth.action_values["a0"]),
    )
    max_regrets = [value for value in output.decision.maximum_regret.values() if value is not None]
    row["maximum_regret"] = min(max_regrets) if max_regrets else float("nan")
    for outcome in ("Y_CR", "Y_CFO"):
        effect = _effect_for_evaluation(output, outcome, truth)
        prefix = outcome.lower()
        true_effect = truth.true_ate[outcome]
        if effect is None or effect.status in {
            IdentificationStatus.NOT_IDENTIFIED,
            IdentificationStatus.PARTIALLY_IDENTIFIED,
        }:
            row[f"{prefix}_ate_estimate"] = float("nan")
            row[f"{prefix}_ate_error"] = float("nan")
            row[f"{prefix}_coverage"] = float("nan")
            row[f"{prefix}_interval_width"] = float("nan")
        else:
            estimate = effect.estimate
            row[f"{prefix}_ate_estimate"] = estimate
            row[f"{prefix}_ate_error"] = (
                float("nan") if estimate is None else float(estimate - true_effect)
            )
            if effect.interval is None:
                row[f"{prefix}_coverage"] = float(
                    effect.status == IdentificationStatus.STRUCTURAL_ZERO
                    and abs(true_effect) < 1e-12
                )
                row[f"{prefix}_interval_width"] = 0.0
            else:
                row[f"{prefix}_coverage"] = float(
                    effect.interval[0] <= true_effect <= effect.interval[1]
                )
                row[f"{prefix}_interval_width"] = effect.interval[1] - effect.interval[0]
        row[f"{prefix}_true_ate"] = true_effect
        true_att = truth.true_att[outcome]
        att_estimate = effect.diagnostics.get("att_estimate") if effect is not None else None
        att_interval = effect.diagnostics.get("att_interval") if effect is not None else None
        row[f"{prefix}_true_att"] = true_att
        row[f"{prefix}_att_estimate"] = (
            float(att_estimate) if att_estimate is not None else float("nan")
        )
        row[f"{prefix}_att_error"] = (
            float(att_estimate - true_att) if att_estimate is not None else float("nan")
        )
        row[f"{prefix}_att_coverage"] = (
            float(att_interval[0] <= true_att <= att_interval[1])
            if att_interval is not None
            else float("nan")
        )
        fixed_predictions = (
            effect.diagnostics.get("fixed_control_predictions", []) if effect is not None else []
        )
        fixed_truth = truth.metadata.get("fixed_control_true_cate", {}).get(outcome)
        if fixed_predictions and fixed_truth is not None:
            row[f"{prefix}_cate_rmse"] = float(
                np.sqrt(
                    np.mean(
                        (
                            np.asarray(fixed_predictions, dtype=float)
                            - np.asarray(fixed_truth, dtype=float)
                        )
                        ** 2
                    )
                )
            )
            row[f"{prefix}_cate_control_size"] = len(fixed_predictions)
        else:
            cate_predictions = (
                effect.diagnostics.get("cate_predictions", []) if effect is not None else []
            )
            cate_row_ids = effect.diagnostics.get("cate_row_ids", []) if effect is not None else []
            if not cate_predictions or not cate_row_ids:
                row[f"{prefix}_cate_rmse"] = float("nan")
                row[f"{prefix}_cate_control_size"] = float("nan")
                continue
            true_unit_cate = truth.true_cate[outcome][np.asarray(cate_row_ids, dtype=int)]
            row[f"{prefix}_cate_rmse"] = float(
                np.sqrt(
                    np.mean(
                        (
                            np.asarray(cate_predictions, dtype=float)
                            - np.asarray(true_unit_cate, dtype=float)
                        )
                        ** 2
                    )
                )
            )
            row[f"{prefix}_cate_control_size"] = len(cate_predictions)
        true_profiles = truth.metadata.get("true_cate_profiles", {}).get(outcome, {})
        if effect is not None and effect.cate_profiles and true_profiles:
            profile_errors = []
            for label, true_value in true_profiles.items():
                estimated = effect.cate_profiles.get(label, float("nan"))
                error = float(estimated - true_value)
                safe_label = (
                    label.replace(" ", "_")
                    .replace("/", "_")
                    .replace("высокая", "high")
                    .replace("низкий", "low")
                    .replace("низкая", "low")
                    .replace("высокий", "high")
                )
                row[f"{prefix}_cate_profile_error_{safe_label}"] = error
                profile_errors.append(error)
            row[f"{prefix}_cate_profile_rmse"] = float(
                np.sqrt(np.nanmean(np.square(profile_errors)))
            )
        else:
            row[f"{prefix}_cate_profile_rmse"] = float("nan")
    statuses = [effect.status.value for effect in output.effects]
    for status in ("identified", "partially_identified", "not_identified", "structural_zero"):
        row[f"share_{status}"] = (
            statuses.count(status) / len(statuses) if statuses else float("nan")
        )
    row["true_structure_in_set"] = (
        False if truth.true_graph_id is None else truth.true_graph_id in output.graph_ids
    )
    maximum_graph = max(
        analysis.graph_scores, key=lambda item: item.mu
    ).graph_id
    row["maximum_membership_graph"] = maximum_graph
    row["true_graph_matches_maximum"] = (
        False if truth.true_graph_id is None else truth.true_graph_id == maximum_graph
    )
    if truth.true_graph_id is None:
        row["first_true_structure_alpha"] = float("nan")
    else:
        inclusion_levels = [
            cut.alpha for cut in analysis.alpha_cuts if truth.true_graph_id in cut.graph_ids
        ]
        row["first_true_structure_alpha"] = (
            max(inclusion_levels) if inclusion_levels else float("nan")
        )
    target_stability = next(
        (profile for profile in analysis.stability if profile.outcome == "Y_CR"), None
    )
    if target_stability is not None:
        point = min(
            target_stability.points,
            key=lambda item: abs(item.alpha - output.decision.alpha),
        )
        row["sign_stable"] = point.sign_stable
        row["threshold_stable"] = point.threshold_stable
        row["first_identification_loss_alpha"] = (
            target_stability.first_identification_loss_alpha
            if target_stability.first_identification_loss_alpha is not None
            else float("nan")
        )
        row["first_sign_loss_alpha"] = (
            target_stability.first_sign_loss_alpha
            if target_stability.first_sign_loss_alpha is not None
            else float("nan")
        )
        row["first_threshold_loss_alpha"] = (
            target_stability.first_threshold_loss_alpha
            if target_stability.first_threshold_loss_alpha is not None
            else float("nan")
        )
    row["false_confidence"] = bool(
        output.decision.status in {"robust", "conditionally_robust"}
        and (
            not row["true_structure_in_set"]
            or not operational_policy_accuracy
        )
    )
    return row


def aggregate_metrics(frame):

    group_columns = [
        "scenario",
        "sample_size",
        "value_regime",
        "true_graph_id",
        "true_graph_matches_maximum",
        "method",
    ]
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in group_columns
    ]
    means = frame.groupby(group_columns, dropna=False)[numeric].mean().reset_index()
    std = frame.groupby(group_columns, dropna=False)[numeric].std().reset_index()
    counts = frame.groupby(group_columns, dropna=False).size().reset_index(name="replications")
    std = std.rename(columns={column: f"{column}_std" for column in numeric})
    output = means.merge(counts, on=group_columns).merge(std, on=group_columns)
    mcse = pd.DataFrame(
        {
            f"{column}_mcse": output[f"{column}_std"] / np.sqrt(output["replications"])
            for column in numeric
        },
        index=output.index,
    )
    output = pd.concat([output, mcse], axis=1)
    median_regret = (
        frame.groupby(group_columns, dropna=False)["regret"]
        .median()
        .reset_index(name="regret_median")
    )
    output = output.merge(median_regret, on=group_columns)
    opportunities = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            erroneous_a2_opportunities=("erroneous_a2_opportunity", "sum"),
            erroneous_a2_count=("erroneous_a2", "sum"),
        )
        .reset_index()
    )
    opportunities["erroneous_a2_conditional_rate"] = np.where(
        opportunities["erroneous_a2_opportunities"] > 0,
        opportunities["erroneous_a2_count"]
        / opportunities["erroneous_a2_opportunities"],
        np.nan,
    )
    output = output.merge(opportunities, on=group_columns)
    for outcome in ("y_cr", "y_cfo"):
        for estimand in ("ate", "att"):
            error = f"{outcome}_{estimand}_error"
            if error not in frame:
                continue
            rmse = (
                frame.assign(_sq=frame[error] ** 2)
                .groupby(group_columns, dropna=False)["_sq"]
                .mean()
                .pow(0.5)
                .reset_index(name=f"{outcome}_{estimand}_rmse")
            )
            output = output.merge(rmse, on=group_columns)
    return output
