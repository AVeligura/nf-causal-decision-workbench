from __future__ import annotations

from typing import Any

from domain import AlphaCut, EffectResult, GeneratorConfig, IdentificationStatus, ValueModelConfig
from engine.decision import evaluate_decisions
from engine.stability import summarize_alpha_trajectory

from .dgp import generate_dataset
from .scenarios import apply_scenario


def _identified(graph: str, outcome: str, estimate: float, se: float = 0.001) -> EffectResult:
    return EffectResult(
        graph_id=graph,
        query_id=f"control-{graph}-{outcome}",
        outcome=outcome,
        estimand="ATE",
        status=IdentificationStatus.IDENTIFIED,
        estimate=estimate,
        standard_error=se,
        interval=(estimate - 1.96 * se, estimate + 1.96 * se),
        diagnostics={"n_analysis": 1000},
    )


def _not_identified(graph: str, outcome: str) -> EffectResult:
    return EffectResult(
        graph_id=graph,
        query_id=f"control-{graph}-{outcome}",
        outcome=outcome,
        estimand="ATE",
        status=IdentificationStatus.NOT_IDENTIFIED,
    )


def run_control_examples() -> list[dict[str, Any]]:
    costly_information = ValueModelConfig(pilot_information_cost=0.10)
    positive = tuple(
        _identified("G1", outcome, estimate)
        for outcome, estimate in (("Y_CR", 0.10), ("Y_CFO", 0.02))
    )
    negative = tuple(
        _identified("G1", outcome, estimate)
        for outcome, estimate in (("Y_CR", -0.10), ("Y_CFO", -0.02))
    )
    stable_a2 = evaluate_decisions(
        positive, ("G1",), alpha=0.9, value_config=costly_information, seed=11
    )
    stable_a0 = evaluate_decisions(
        negative, ("G1",), alpha=0.9, value_config=costly_information, seed=12
    )

    pilot_effects = tuple(
        _identified("G1", outcome, 0.0, se)
        for outcome, se in (("Y_CR", 0.05), ("Y_CFO", 0.015))
    )
    pilot = evaluate_decisions(
        pilot_effects,
        ("G1",),
        alpha=0.8,
        value_config=ValueModelConfig(
            pilot_share=0.20,
            program_cost_a1=0.0,
            program_cost_a2=0.005,
            pilot_information_cost=0.0,
            pilot_virtual_samples=2000,
            conditional_regret_threshold=0.1,
            sales_loss_per_full_coverage=0.0,
            zombie_risk_per_full_coverage=0.0,
        ),
        seed=7,
    )

    switching_effects = positive + tuple(
        _identified("G2", outcome, estimate)
        for outcome, estimate in (("Y_CR", -0.20), ("Y_CFO", -0.05))
    )
    switch_config = ValueModelConfig(
        pilot_information_cost=0.10,
        conditional_regret_threshold=0.20,
    )
    narrow = evaluate_decisions(
        switching_effects, ("G1",), alpha=0.9, value_config=switch_config, seed=13
    )
    broad = evaluate_decisions(
        switching_effects,
        ("G1", "G2"),
        alpha=0.6,
        value_config=switch_config,
        seed=13,
    )
    switching = summarize_alpha_trajectory(
        (
            AlphaCut(alpha=0.9, graph_ids=("G1",), core_edges=(), alternative_edges=()),
            AlphaCut(
                alpha=0.6,
                graph_ids=("G1", "G2"),
                core_edges=(),
                alternative_edges=(),
            ),
        ),
        (narrow, broad),
        (),
        value_config=switch_config,
    )

    loss_effects = positive + tuple(
        _not_identified("G2", outcome) for outcome in ("Y_CR", "Y_CFO")
    )
    identification_loss = evaluate_decisions(
        loss_effects,
        ("G1", "G2"),
        alpha=0.6,
        value_config=costly_information,
        seed=14,
    )

    outside = generate_dataset(
        apply_scenario(
            GeneratorConfig(
                mode="laboratory",
                scenario="outside_gamma",
                sample_size=300,
                seed=15,
            )
        )
    )
    assert outside.truth is not None
    return [
        {
            "example": "stable_a2",
            "expected": {"status": "robust", "action": "a2"},
            "actual": {"status": stable_a2.status, "action": stable_a2.selected_action},
        },
        {
            "example": "stable_a0",
            "expected": {"status": "robust", "action": "a0"},
            "actual": {"status": stable_a0.status, "action": stable_a0.selected_action},
        },
        {
            "example": "justified_pilot",
            "expected": {"status": "pilot", "action": "a1"},
            "actual": {
                "status": pilot.status,
                "action": pilot.selected_action,
                "net_evi": pilot.pilot_information_value,
                "mcse": pilot.pilot_information_value_se,
                "seed": pilot.pilot_seed,
            },
        },
        {
            "example": "alpha_switching",
            "expected": {"status": "switching", "action": None},
            "actual": {
                "status": switching.status,
                "action": switching.operational_decision.selected_action,
                "trigger_alpha": switching.first_action_change_alpha,
                "trigger_graph": switching.first_action_change_graph,
            },
        },
        {
            "example": "identification_loss",
            "expected": {"status": "abstain", "action": None},
            "actual": {
                "status": identification_loss.status,
                "action": identification_loss.selected_action,
            },
        },
        {
            "example": "true_structure_outside_gamma",
            "expected": {"true_graph_id": None},
            "actual": {"true_graph_id": outside.truth.true_graph_id},
        },
    ]
