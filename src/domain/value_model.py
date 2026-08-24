from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from domain import ActionSpec, ValueModelConfig


@dataclass(frozen=True)
class ValueInputs:
    """Per-full-coverage outcome and side-effect ranges for one structure."""

    cr: tuple[float, float]
    cfo: tuple[float, float]
    financing_cost_reduction: tuple[float, float]
    arrears_reduction: tuple[float, float]
    sales_loss: tuple[float, float]
    zombie_risk: tuple[float, float]


@dataclass(frozen=True)
class PilotGraphState:
    """Current pre-pilot belief for one structural branch, in value-model units."""

    graph_id: str
    cr_mean: float
    cfo_mean: float
    cr_standard_error: float
    cfo_standard_error: float


@dataclass(frozen=True)
class TwoStagePilotValue:
    """Registered two-stage value of a1 on the common decision horizon."""

    immediate_by_graph: dict[str, float]
    continuation_without_information_by_graph: dict[str, float]
    continuation_with_information_by_graph: dict[str, float]
    total_by_graph: dict[str, float]
    robust_value_before_information: float
    robust_value_after_information: float
    adaptive_robust_value: float
    gross_evi: float
    information_cost: float
    net_evi: float
    inner_mcse: float | None
    seed: int
    samples: int
    information_used: bool
    common_evidence_method: str


def _scaled_range(values: tuple[float, float], factor: float) -> tuple[float, float]:
    candidates = (factor * values[0], factor * values[1])
    return min(candidates), max(candidates)


def estimated_value_inputs(
    cr: tuple[float, float],
    cfo: tuple[float, float],
    config: ValueModelConfig,
) -> ValueInputs:
    """Build all components using the registered external-parameter semantics."""

    return ValueInputs(
        cr=cr,
        cfo=cfo,
        financing_cost_reduction=_scaled_range(cfo, config.financing_reduction_ratio),
        arrears_reduction=_scaled_range(cr, config.arrears_reduction_ratio),
        sales_loss=(
            config.sales_loss_per_full_coverage,
            config.sales_loss_per_full_coverage,
        ),
        zombie_risk=(
            config.zombie_risk_per_full_coverage,
            config.zombie_risk_per_full_coverage,
        ),
    )


def point_value_inputs(
    *,
    cr: float,
    cfo: float,
    financing_cost_reduction: float,
    arrears_reduction: float,
    sales_loss: float,
    zombie_risk: float,
) -> ValueInputs:
    return ValueInputs(
        cr=(cr, cr),
        cfo=(cfo, cfo),
        financing_cost_reduction=(financing_cost_reduction, financing_cost_reduction),
        arrears_reduction=(arrears_reduction, arrears_reduction),
        sales_loss=(sales_loss, sales_loss),
        zombie_risk=(zombie_risk, zombie_risk),
    )


def _unit_value(cr: np.ndarray | float, cfo: np.ndarray | float, config: ValueModelConfig):
    """Utility at full coverage before programme cost, in one registered unit."""

    financing = config.financing_reduction_ratio * cfo
    arrears = config.arrears_reduction_ratio * cr
    return config.multiplier * (
        cfo
        + config.cr_weight * cr
        + config.financing_weight * financing
        + config.arrears_weight * arrears
        - config.sales_loss_weight * config.sales_loss_per_full_coverage
        - config.zombie_weight * config.zombie_risk_per_full_coverage
    )


def action_value_range(
    action: ActionSpec,
    inputs: ValueInputs,
    config: ValueModelConfig,
    *,
    information_value: float = 0.0,
) -> tuple[float, float]:
    """Evaluate the registered immediate/full-rollout equation at range vertices.

    `information_value` is retained for backwards-compatible file replay only.
    V3 decisions obtain a1 from :func:`evaluate_two_stage_pilot` and never use
    this argument to append a sample EVI to an immediate value.
    """

    components = (
        inputs.cr,
        inputs.cfo,
        inputs.financing_cost_reduction,
        inputs.arrears_reduction,
        inputs.sales_loss,
        inputs.zombie_risk,
    )
    candidates: list[float] = []
    for cr in components[0]:
        for cfo in components[1]:
            for financing in components[2]:
                for arrears in components[3]:
                    for sales_loss in components[4]:
                        for zombie_risk in components[5]:
                            net = (
                                cfo
                                + config.cr_weight * cr
                                + config.financing_weight * financing
                                + config.arrears_weight * arrears
                                - config.sales_loss_weight * sales_loss
                                - config.zombie_weight * zombie_risk
                            )
                            legacy_information = (
                                information_value if action.action_id == "a1" else 0.0
                            )
                            candidates.append(
                                config.multiplier * action.coverage * net
                                - action.program_cost
                                + legacy_information
                            )
    return float(min(candidates)), float(max(candidates))


def point_action_values(
    actions: tuple[ActionSpec, ...],
    inputs: ValueInputs,
    config: ValueModelConfig,
) -> dict[str, float]:
    """Point values for a0/a2 and the stage-one immediate component of a1."""

    values: dict[str, float] = {}
    for action in actions:
        if action.action_id == "a0":
            values[action.action_id] = 0.0
            continue
        low, high = action_value_range(action, inputs, config)
        if abs(high - low) > 1e-12:
            raise ValueError("Point truth inputs unexpectedly produced a value interval")
        values[action.action_id] = low
    return values


def evaluate_two_stage_pilot(
    states: tuple[PilotGraphState, ...],
    config: ValueModelConfig,
    *,
    seed: int,
) -> TwoStagePilotValue:
    """Preposterior value of a real two-stage pilot for truth and estimation.

    The pilot treats the current effect distribution as the pre-pilot state of
    knowledge. Conditional on each possible true graph ``g``, it generates one
    common pilot result ``z`` for each outcome. The same ``z`` updates every
    alternative structural branch by normal-normal precision weighting. The
    continuation decision is then made once for that shared evidence, rather
    than from independently simulated graph-specific pseudo-observations.

    The robust preposterior order is ``min_g E_{z|g}``, not ``E_z min_g``.
    The baseline stop/rollout rule remains available as an information-ignoring
    strategy. Consequently ``R1 >= R0`` by construction.
    """

    samples = max(50, int(config.pilot_virtual_samples))
    if not states:
        return TwoStagePilotValue(
            immediate_by_graph={},
            continuation_without_information_by_graph={},
            continuation_with_information_by_graph={},
            total_by_graph={},
            robust_value_before_information=0.0,
            robust_value_after_information=0.0,
            adaptive_robust_value=0.0,
            gross_evi=0.0,
            information_cost=config.pilot_information_cost,
            net_evi=-config.pilot_information_cost,
            inner_mcse=None,
            seed=seed,
            samples=samples,
            information_used=False,
            common_evidence_method="shared_outcome_pilot_summary",
        )

    ordered_states = tuple(sorted(states, key=lambda state: state.graph_id))
    graph_ids = [state.graph_id for state in ordered_states]
    pilot_share = max(float(config.pilot_share), 1e-12)
    prior_cr = np.asarray([state.cr_mean for state in ordered_states], dtype=float)
    prior_cfo = np.asarray([state.cfo_mean for state in ordered_states], dtype=float)
    prior_cr_se = np.asarray(
        [max(float(state.cr_standard_error), 0.0) for state in ordered_states],
        dtype=float,
    )
    prior_cfo_se = np.asarray(
        [max(float(state.cfo_standard_error), 0.0) for state in ordered_states],
        dtype=float,
    )
    immediate_values = (
        config.pilot_share * _unit_value(prior_cr, prior_cfo, config)
        - config.program_cost_a1
    )
    incremental_cost = config.program_cost_a2 - config.program_cost_a1
    remaining_share = 1.0 - config.pilot_share
    prior_continuation = (
        remaining_share * _unit_value(prior_cr, prior_cfo, config) - incremental_cost
    )
    baseline_continue = bool(np.min(prior_continuation) > 0.0)
    continuation_without = (
        prior_continuation.copy()
        if baseline_continue
        else np.zeros_like(prior_continuation)
    )
    r0 = float(np.min(continuation_without))

    adaptive_samples: dict[str, np.ndarray] = {}
    adaptive_continuation = np.empty(len(ordered_states), dtype=float)
    for true_index, true_state in enumerate(ordered_states):
        # A graph-keyed stream makes the result invariant to input ordering.
        stream_key = f"{seed}|{true_state.graph_id}|shared-pilot-v3.1".encode()
        stream_seed = int.from_bytes(hashlib.sha256(stream_key).digest()[:8], "big")
        rng = np.random.default_rng(stream_seed)

        if prior_cr_se[true_index] <= 1e-15:
            latent_cr = np.full(samples, prior_cr[true_index], dtype=float)
            observed_cr = latent_cr
        else:
            latent_cr = rng.normal(
                prior_cr[true_index], prior_cr_se[true_index], samples
            )
            observed_cr = rng.normal(
                latent_cr, prior_cr_se[true_index] / np.sqrt(pilot_share), samples
            )
        if prior_cfo_se[true_index] <= 1e-15:
            latent_cfo = np.full(samples, prior_cfo[true_index], dtype=float)
            observed_cfo = latent_cfo
        else:
            latent_cfo = rng.normal(
                prior_cfo[true_index], prior_cfo_se[true_index], samples
            )
            observed_cfo = rng.normal(
                latent_cfo, prior_cfo_se[true_index] / np.sqrt(pilot_share), samples
            )

        cr_posterior = np.broadcast_to(prior_cr, (samples, len(ordered_states))).copy()
        cfo_posterior = np.broadcast_to(
            prior_cfo, (samples, len(ordered_states))
        ).copy()
        positive_cr_se = prior_cr_se > 1e-15
        positive_cfo_se = prior_cfo_se > 1e-15
        # With pilot_se = prior_se/sqrt(pilot_share), the conjugate gain is
        # pilot_share/(1+pilot_share). Structural-zero branches remain fixed.
        gain = pilot_share / (1.0 + pilot_share)
        cr_posterior[:, positive_cr_se] += gain * (
            observed_cr[:, None] - prior_cr[positive_cr_se]
        )
        cfo_posterior[:, positive_cfo_se] += gain * (
            observed_cfo[:, None] - prior_cfo[positive_cfo_se]
        )
        posterior_continuation = (
            remaining_share * _unit_value(cr_posterior, cfo_posterior, config)
            - incremental_cost
        )
        informed_continue = np.min(posterior_continuation, axis=1) > 0.0
        latent_continuation = (
            remaining_share * _unit_value(latent_cr, latent_cfo, config)
            - incremental_cost
        )
        payoffs = latent_continuation * informed_continue
        adaptive_samples[true_state.graph_id] = payoffs
        adaptive_continuation[true_index] = float(np.mean(payoffs))

    adaptive_r1 = float(np.min(adaptive_continuation))
    information_used = adaptive_r1 > r0
    if information_used:
        continuation_with = adaptive_continuation
        r1 = adaptive_r1
    else:
        continuation_with = continuation_without.copy()
        r1 = r0

    gross_evi = max(0.0, r1 - r0)
    worst_adaptive_index = int(np.argmin(adaptive_continuation))
    worst_adaptive_graph = graph_ids[worst_adaptive_index]
    inner_mcse = (
        float(
            np.std(adaptive_samples[worst_adaptive_graph], ddof=1)
            / np.sqrt(samples)
        )
        if samples > 1
        else 0.0
    )
    net_evi = gross_evi - config.pilot_information_cost
    total = immediate_values + continuation_with - config.pilot_information_cost

    return TwoStagePilotValue(
        immediate_by_graph={
            graph_id: float(immediate_values[index])
            for index, graph_id in enumerate(graph_ids)
        },
        continuation_without_information_by_graph={
            graph_id: float(continuation_without[index])
            for index, graph_id in enumerate(graph_ids)
        },
        continuation_with_information_by_graph={
            graph_id: float(continuation_with[index])
            for index, graph_id in enumerate(graph_ids)
        },
        total_by_graph={
            graph_id: float(total[index]) for index, graph_id in enumerate(graph_ids)
        },
        robust_value_before_information=r0,
        robust_value_after_information=r1,
        adaptive_robust_value=adaptive_r1,
        gross_evi=gross_evi,
        information_cost=config.pilot_information_cost,
        net_evi=net_evi,
        inner_mcse=inner_mcse,
        seed=seed,
        samples=samples,
        information_used=information_used,
        common_evidence_method="shared_outcome_pilot_summary",
    )
