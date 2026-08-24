from __future__ import annotations

from dataclasses import dataclass

from domain import ActionSpec, DecisionResult, EffectResult, IdentificationStatus, ValueModelConfig
from domain.value_model import (
    PilotGraphState,
    TwoStagePilotValue,
    action_value_range,
    estimated_value_inputs,
    evaluate_two_stage_pilot,
)

DEFAULT_ACTIONS = (
    ActionSpec(action_id="a0", name="Отказ от программы", coverage=0.0, program_cost=0.0),
    ActionSpec(action_id="a1", name="Двухэтапный пилот", coverage=0.20, program_cost=0.004),
    ActionSpec(action_id="a2", name="Полномасштабное внедрение", coverage=1.0, program_cost=0.008),
)


@dataclass(frozen=True)
class _DecisionStatistics:
    values: dict[str, dict[str, float | tuple[float, float] | None]]
    regrets: dict[str, dict[str, float | None]]
    maximum_regret: dict[str, float | None]
    worst_graph: dict[str, str | None]
    common_optima: set[str]
    missing_value: bool


def actions_from_value_config(config: ValueModelConfig) -> tuple[ActionSpec, ...]:
    return (
        ActionSpec(action_id="a0", name="Отказ от программы", coverage=0.0, program_cost=0.0),
        ActionSpec(
            action_id="a1",
            name="Двухэтапный пилот",
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


def _effect_range(effect: EffectResult | None) -> tuple[float, float] | None:
    if effect is None or effect.status == IdentificationStatus.NOT_IDENTIFIED:
        return None
    if effect.status == IdentificationStatus.PARTIALLY_IDENTIFIED:
        return effect.identified_bounds
    if effect.status == IdentificationStatus.STRUCTURAL_ZERO:
        return (0.0, 0.0)
    if effect.interval is not None:
        return effect.interval
    if effect.estimate is not None:
        return (effect.estimate, effect.estimate)
    return None


def _pilot_states(
    effects: tuple[EffectResult, ...], graph_ids: tuple[str, ...]
) -> tuple[PilotGraphState, ...]:
    by_key = {(effect.graph_id, effect.outcome): effect for effect in effects}
    states: list[PilotGraphState] = []
    for graph_id in graph_ids:
        values: dict[str, tuple[float, float]] = {}
        for outcome in ("Y_CR", "Y_CFO"):
            effect = by_key.get((graph_id, outcome))
            if effect is None:
                return ()
            if effect.status == IdentificationStatus.STRUCTURAL_ZERO:
                values[outcome] = (0.0, 0.0)
            elif (
                effect.status == IdentificationStatus.IDENTIFIED
                and effect.estimate is not None
                and effect.standard_error is not None
                and effect.standard_error >= 0
            ):
                values[outcome] = (float(effect.estimate), float(effect.standard_error))
            else:
                return ()
        states.append(
            PilotGraphState(
                graph_id=graph_id,
                cr_mean=values["Y_CR"][0],
                cfo_mean=values["Y_CFO"][0],
                cr_standard_error=values["Y_CR"][1],
                cfo_standard_error=values["Y_CFO"][1],
            )
        )
    return tuple(states)


def _decision_statistics(
    effects: tuple[EffectResult, ...],
    graph_ids: tuple[str, ...],
    actions: tuple[ActionSpec, ...],
    config: ValueModelConfig,
    *,
    pilot: TwoStagePilotValue,
) -> _DecisionStatistics:
    by_key = {(effect.graph_id, effect.outcome): effect for effect in effects}
    values: dict[str, dict[str, float | tuple[float, float] | None]] = {
        action.action_id: {} for action in actions
    }
    for graph_id in graph_ids:
        cr = _effect_range(by_key.get((graph_id, "Y_CR")))
        cfo = _effect_range(by_key.get((graph_id, "Y_CFO")))
        for action in actions:
            if action.action_id == "a0":
                values[action.action_id][graph_id] = 0.0
            elif action.action_id == "a1":
                values[action.action_id][graph_id] = pilot.total_by_graph.get(graph_id)
            elif cr is None or cfo is None:
                values[action.action_id][graph_id] = None
            else:
                inputs = estimated_value_inputs(cr, cfo, config)
                low, high = action_value_range(action, inputs, config)
                values[action.action_id][graph_id] = (
                    low if abs(high - low) < 1e-12 else (low, high)
                )

    regrets: dict[str, dict[str, float | None]] = {action.action_id: {} for action in actions}
    maximum_regret: dict[str, float | None] = {}
    worst_graph: dict[str, str | None] = {}
    graph_optima: dict[str, set[str]] = {}
    missing_value = False
    for graph_id in graph_ids:
        lows: dict[str, float] = {}
        highs: dict[str, float] = {}
        for action in actions:
            value = values[action.action_id][graph_id]
            if value is None:
                missing_value = True
                continue
            if isinstance(value, tuple):
                lows[action.action_id], highs[action.action_id] = value
            else:
                lows[action.action_id] = highs[action.action_id] = float(value)
        if not lows:
            continue
        best_low = max(lows.values())
        graph_optima[graph_id] = {
            action_id for action_id, low in lows.items() if abs(low - best_low) < 1e-12
        }
        best_possible = max(highs.values())
        for action in actions:
            action_id = action.action_id
            regrets[action_id][graph_id] = (
                None if action_id not in lows else max(0.0, best_possible - lows[action_id])
            )

    for action in actions:
        action_regrets = {
            graph_id: regret
            for graph_id, regret in regrets[action.action_id].items()
            if regret is not None
        }
        if len(action_regrets) != len(graph_ids):
            maximum_regret[action.action_id] = None
            worst_graph[action.action_id] = None
        else:
            maximum_regret[action.action_id] = max(action_regrets.values(), default=0.0)
            worst_graph[action.action_id] = max(
                action_regrets, key=lambda item: action_regrets[item], default=None
            )

    common = (
        set.intersection(*graph_optima.values())
        if graph_optima and len(graph_optima) == len(graph_ids)
        else set()
    )
    return _DecisionStatistics(
        values=values,
        regrets=regrets,
        maximum_regret=maximum_regret,
        worst_graph=worst_graph,
        common_optima=common,
        missing_value=missing_value,
    )


def evaluate_decisions(
    effects: tuple[EffectResult, ...],
    graph_ids: tuple[str, ...],
    *,
    alpha: float,
    actions: tuple[ActionSpec, ...] | None = None,
    value_config: ValueModelConfig | None = None,
    seed: int = 20260814,
) -> DecisionResult:
    value_config = value_config or ValueModelConfig()
    actions = actions or actions_from_value_config(value_config)
    pilot = evaluate_two_stage_pilot(
        _pilot_states(effects, graph_ids), value_config, seed=seed
    )
    statistics = _decision_statistics(
        effects,
        graph_ids,
        actions,
        value_config,
        pilot=pilot,
    )

    admissible_common = set(statistics.common_optima)
    if pilot.net_evi <= 0:
        admissible_common.discard("a1")
    if admissible_common and not statistics.missing_value:
        selected = sorted(admissible_common)[0]
        if selected == "a1":
            status = "pilot"
            reason = "Двухэтапный пилот оптимален при каждой структуре и имеет net EVI > 0"
        else:
            status = "robust"
            reason = "Действие оптимально при каждой структуре текущего α-среза"
    else:
        available = {
            key: value
            for key, value in statistics.maximum_regret.items()
            if value is not None and (key != "a1" or pilot.net_evi > 0)
        }
        if not available or statistics.missing_value:
            selected = None
            status = "abstain"
            reason = "Для существенной структурной ветви ценность не идентифицирована"
        else:
            selected = min(available, key=lambda action_id: available[action_id])
            selected_regret = float(available[selected])
            if selected == "a1":
                status = "pilot"
                reason = "Двухэтапный пилот минимизирует max regret и имеет net EVI > 0"
            elif selected_regret <= value_config.conditional_regret_threshold:
                status = "conditionally_robust"
                reason = "Максимальное сожаление не превышает практический допуск"
            else:
                status = "abstain"
                selected = None
                reason = "Ни одно действие не удовлетворяет критерию допустимого сожаления"

    immediate = min(pilot.immediate_by_graph.values(), default=0.0)
    continuation = min(pilot.continuation_with_information_by_graph.values(), default=0.0)
    return DecisionResult(
        alpha=alpha,
        status=status,
        selected_action=selected,
        values=statistics.values,
        regrets=statistics.regrets,
        maximum_regret=statistics.maximum_regret,
        worst_graph=statistics.worst_graph,
        pilot_information_value=pilot.net_evi,
        pilot_expected_regret_reduction=pilot.gross_evi,
        pilot_immediate_value=immediate,
        pilot_expected_continuation_value=continuation,
        pilot_gross_evi=pilot.gross_evi,
        pilot_information_cost=pilot.information_cost,
        pilot_net_evi=pilot.net_evi,
        pilot_r0=pilot.robust_value_before_information,
        pilot_r1=pilot.robust_value_after_information,
        pilot_adaptive_r1=pilot.adaptive_robust_value,
        pilot_common_evidence_method=pilot.common_evidence_method,
        pilot_admissible=pilot.net_evi > 0,
        pilot_information_value_se=pilot.inner_mcse,
        pilot_seed=pilot.seed,
        pilot_virtual_samples=pilot.samples,
        reason=reason,
    )
