from __future__ import annotations

from domain import (
    AnalysisResult,
    EffectResult,
    IdentificationStatus,
)
from domain.truth import TruthBundle
from engine.decision import evaluate_decisions

from .dgp import value_config_from_generator
from .methods import MethodOutput


def run_structure_oracle(result: AnalysisResult, truth: TruthBundle) -> MethodOutput:
    """Diagnostic structure oracle; potential outcomes are never used for estimation."""
    if truth.true_graph_id is None:
        effects = tuple(
            EffectResult(
                graph_id="ORACLE",
                query_id=f"oracle_{outcome}",
                outcome=outcome,
                estimand="ATE",
                status=IdentificationStatus.NOT_IDENTIFIED,
                warnings=("Истинная структура содержит ненаблюдаемый U",),
            )
            for outcome in ("Y_CR", "Y_CFO")
        )
        decision = evaluate_decisions(
            effects,
            ("ORACLE",),
            alpha=1.0,
            value_config=value_config_from_generator(result.config),
            seed=(result.config.pilot_seed if result.config.pilot_seed is not None else result.config.seed),
        )
        return MethodOutput("structure_oracle", ("ORACLE",), effects, decision, uses_truth=True)
    effects = tuple(effect for effect in result.effects if effect.graph_id == truth.true_graph_id)
    decision = evaluate_decisions(
        effects,
        (truth.true_graph_id,),
        alpha=1.0,
        value_config=value_config_from_generator(result.config),
        seed=(result.config.pilot_seed if result.config.pilot_seed is not None else result.config.seed),
    )
    return MethodOutput(
        "structure_oracle", (truth.true_graph_id,), effects, decision, uses_truth=True
    )
