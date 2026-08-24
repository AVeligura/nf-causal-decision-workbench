from __future__ import annotations

from domain import (
    AlphaCut,
    AlphaTrajectorySummary,
    CausalQuery,
    DecisionResult,
    EffectResult,
    GraphSpec,
    IdentificationStatus,
    StabilityPoint,
    StabilityProfile,
    ValueModelConfig,
)

from .estimation import estimate_effect
from .identification import identify_effect


def propagate_graph_uncertainty(
    graph_set: tuple[GraphSpec, ...],
    causal_query: CausalQuery,
    data,
    alpha: float,
    estimator_config: dict,
    *,
    scenario: str = "reference",
) -> tuple[EffectResult, ...]:
    del alpha
    results: list[EffectResult] = []
    for graph in graph_set:
        identification = identify_effect(graph, causal_query, scenario=scenario)
        try:
            result = estimate_effect(
                data,
                identification,
                outcome=causal_query.outcome,
                estimand=causal_query.estimand,
                folds=int(estimator_config.get("folds", 5)),
                cate_trees=int(estimator_config.get("cate_trees", 300)),
                seed=int(estimator_config.get("seed", 20260814)),
                compute_cate=bool(estimator_config.get("compute_cate", True)),
            )
        except Exception as exc:  # branch failure is preserved, never silently dropped
            result = EffectResult(
                graph_id=graph.graph_id,
                query_id=causal_query.query_id,
                outcome=causal_query.outcome,
                estimand=causal_query.estimand,
                status=IdentificationStatus.NOT_IDENTIFIED,
                warnings=(f"Вычислительный отказ ветви: {type(exc).__name__}: {exc}",),
                diagnostics={"computational_failure": True},
            )
        results.append(result)
    return tuple(results)


def _interval_for_sign(effect: EffectResult) -> tuple[float, float] | None:
    if effect.status == IdentificationStatus.STRUCTURAL_ZERO:
        return (0.0, 0.0)
    if effect.status == IdentificationStatus.PARTIALLY_IDENTIFIED:
        return effect.identified_bounds
    if effect.status == IdentificationStatus.IDENTIFIED:
        return effect.interval
    return None


def assess_effect_stability(
    alpha_cuts: tuple[AlphaCut, ...],
    effects: tuple[EffectResult, ...],
    *,
    outcome: str,
    practical_threshold: float,
) -> StabilityProfile:
    by_graph = {effect.graph_id: effect for effect in effects if effect.outcome == outcome}
    points: list[StabilityPoint] = []
    previous_graphs: set[str] = set()
    first_id_loss: float | None = None
    first_sign_loss: float | None = None
    first_threshold_loss: float | None = None
    for cut in sorted(alpha_cuts, key=lambda item: item.alpha, reverse=True):
        selected = [by_graph[graph_id] for graph_id in cut.graph_ids if graph_id in by_graph]
        statuses = {effect.status for effect in selected}
        uniform = bool(selected) and statuses <= {
            IdentificationStatus.IDENTIFIED,
            IdentificationStatus.STRUCTURAL_ZERO,
        }
        functionals = {effect.functional for effect in selected if effect.functional is not None}
        functional_stable = uniform and len(functionals) <= 1
        intervals = [_interval_for_sign(effect) for effect in selected]
        fully_bounded = bool(intervals) and all(interval is not None for interval in intervals)
        sign_positive = fully_bounded and all(interval[0] > 0 for interval in intervals if interval)
        sign_negative = fully_bounded and all(interval[1] < 0 for interval in intervals if interval)
        sign_stable = bool(uniform and (sign_positive or sign_negative))
        threshold_stable = bool(
            uniform
            and fully_bounded
            and all(interval[0] >= practical_threshold for interval in intervals if interval)
        )
        point_estimates = [
            effect.estimate
            for effect in selected
            if effect.status
            in {IdentificationStatus.IDENTIFIED, IdentificationStatus.STRUCTURAL_ZERO}
            and effect.estimate is not None
        ]
        spread = max(point_estimates) - min(point_estimates) if point_estimates else None
        added = set(cut.graph_ids) - previous_graphs
        changed_by = sorted(added)[0] if added else None
        warnings: list[str] = []
        if IdentificationStatus.PARTIALLY_IDENTIFIED in statuses:
            warnings.append("В α-срезе присутствует частично идентифицированная ветвь")
        if IdentificationStatus.NOT_IDENTIFIED in statuses:
            warnings.append("В α-срезе присутствует неидентифицируемая ветвь")
        if IdentificationStatus.STRUCTURAL_ZERO in statuses:
            warnings.append("Структурный нуль исключает строгую устойчивость положительного знака")
        points.append(
            StabilityPoint(
                alpha=cut.alpha,
                graph_ids=cut.graph_ids,
                uniformly_identified=uniform,
                functional_stable=functional_stable,
                sign_stable=sign_stable,
                threshold_stable=threshold_stable,
                structural_spread=spread,
                first_changed_by=changed_by,
                warnings=tuple(warnings),
            )
        )
        if not uniform and first_id_loss is None:
            first_id_loss = cut.alpha
        if not sign_stable and first_sign_loss is None:
            first_sign_loss = cut.alpha
        if not threshold_stable and first_threshold_loss is None:
            first_threshold_loss = cut.alpha
        previous_graphs = set(cut.graph_ids)
    return StabilityProfile(
        outcome=outcome,
        points=tuple(points),
        first_identification_loss_alpha=first_id_loss,
        first_sign_loss_alpha=first_sign_loss,
        first_threshold_loss_alpha=first_threshold_loss,
    )


def _first_profile_event(
    profiles: tuple[StabilityProfile, ...],
    field: str,
) -> tuple[float | None, str | None]:
    candidates: list[tuple[float, str | None]] = []
    for profile in profiles:
        alpha = getattr(profile, field)
        if alpha is None:
            continue
        point = next((item for item in profile.points if abs(item.alpha - alpha) < 1e-12), None)
        candidates.append((float(alpha), point.first_changed_by if point else None))
    return max(candidates, key=lambda item: item[0]) if candidates else (None, None)


def summarize_alpha_trajectory(
    alpha_cuts: tuple[AlphaCut, ...],
    decisions: tuple[DecisionResult, ...],
    stability_profiles: tuple[StabilityProfile, ...],
    *,
    value_config: ValueModelConfig,
) -> AlphaTrajectorySummary:
    """Summarise the ordered alpha path without aliasing its widest cut."""

    ordered = tuple(sorted(decisions, key=lambda item: item.alpha, reverse=True))
    cuts = {round(cut.alpha, 12): cut for cut in alpha_cuts}
    action_sequence = tuple((decision.alpha, decision.selected_action) for decision in ordered)
    non_null_actions = {action for _alpha, action in action_sequence if action is not None}
    first_change_alpha: float | None = None
    first_change_graph: str | None = None
    previous_action: str | None = None
    previous_graphs: set[str] = set()
    has_previous = False
    for decision in ordered:
        cut = cuts.get(round(decision.alpha, 12))
        current_graphs = set(cut.graph_ids) if cut else set()
        if has_previous and decision.selected_action != previous_action:
            first_change_alpha = decision.alpha
            added = sorted(current_graphs - previous_graphs)
            first_change_graph = added[0] if added else None
            break
        previous_action = decision.selected_action
        previous_graphs = current_graphs
        has_previous = True

    identification_alpha, identification_graph = _first_profile_event(
        stability_profiles, "first_identification_loss_alpha"
    )
    sign_alpha, sign_graph = _first_profile_event(stability_profiles, "first_sign_loss_alpha")
    threshold_alpha, threshold_graph = _first_profile_event(
        stability_profiles, "first_threshold_loss_alpha"
    )

    selected_action: str | None = None
    stable_range: tuple[float, float] | None = None
    structural_condition: tuple[str, ...] = ()
    if len(non_null_actions) > 1:
        status = "switching"
        reason = "При расширении Γα предпочтительное действие изменяется"
    elif len(non_null_actions) == 1:
        selected_action = next(iter(non_null_actions))
        matching = [decision.alpha for decision in ordered if decision.selected_action == selected_action]
        stable_range = (min(matching), max(matching))
        broadest_matching = min(matching)
        cut = cuts.get(round(broadest_matching, 12))
        structural_condition = cut.graph_ids if cut else ()
        has_abstain = any(decision.selected_action is None for decision in ordered)
        has_pilot = any(decision.status == "pilot" for decision in ordered)
        if selected_action == "a1" and has_pilot and not has_abstain:
            status = "pilot"
            reason = "Пилот устойчив вдоль α-траектории и имеет положительную чистую EVI"
        elif not has_abstain and all(
            decision.status in {"robust", "pilot"} for decision in ordered
        ):
            status = "robust"
            reason = "Одно действие сохраняется на всём анализируемом диапазоне α"
        else:
            status = "conditionally_robust"
            reason = "Действие сохраняется только на указанном диапазоне α и наборе структур"
    else:
        status = "abstain"
        reason = "На α-траектории отсутствует безусловно допустимая рекомендация"

    trajectory_regret: dict[str, float | None] = {}
    action_ids = sorted({key for decision in ordered for key in decision.maximum_regret})
    for action_id in action_ids:
        values = [decision.maximum_regret.get(action_id) for decision in ordered]
        numeric_values = [float(value) for value in values if value is not None]
        trajectory_regret[action_id] = (
            max(numeric_values) if len(numeric_values) == len(values) else None
        )

    representative = ordered[-1]
    unconditional = selected_action if status in {"robust", "pilot"} else None
    operational_status = status if status in {"robust", "pilot"} else "abstain"
    operational_reason = reason
    if status == "conditionally_robust":
        operational_reason += "; без структурного условия выполняется мотивированный отказ"
    operational = representative.model_copy(
        update={
            "status": operational_status,
            "selected_action": unconditional,
            "maximum_regret": trajectory_regret,
            "reason": operational_reason,
        }
    )
    return AlphaTrajectorySummary(
        status=status,
        selected_action=selected_action,
        stable_alpha_range=stable_range,
        structural_condition=structural_condition,
        first_identification_loss_alpha=identification_alpha,
        first_identification_loss_graph=identification_graph,
        first_sign_loss_alpha=sign_alpha,
        first_sign_loss_graph=sign_graph,
        first_threshold_loss_alpha=threshold_alpha,
        first_threshold_loss_graph=threshold_graph,
        first_action_change_alpha=first_change_alpha,
        first_action_change_graph=first_change_graph,
        action_sequence=action_sequence,
        trajectory_maximum_regret=trajectory_regret,
        operational_decision=operational,
        reason=reason,
    )
