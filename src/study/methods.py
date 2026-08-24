from __future__ import annotations

import time
from dataclasses import dataclass

from domain import AnalysisResult, DecisionResult, EffectResult, EvidenceBundle
from engine.decision import evaluate_decisions
from engine.evidence import alpha_cut, reference_evidence, score_graph
from engine.graphs import reference_graphs

from .dgp import value_config_from_generator


@dataclass(frozen=True)
class MethodOutput:
    method: str
    graph_ids: tuple[str, ...]
    effects: tuple[EffectResult, ...]
    decision: DecisionResult
    uses_truth: bool = False
    decision_runtime_seconds: float = 0.0


def _effects_for_graphs(
    result: AnalysisResult, graph_ids: tuple[str, ...]
) -> tuple[EffectResult, ...]:
    return tuple(effect for effect in result.effects if effect.graph_id in graph_ids)


def run_comparison_methods(result: AnalysisResult) -> dict[str, MethodOutput]:
    scores = {score.graph_id: score.mu for score in result.graph_scores}
    maximum_graph = max(scores, key=lambda graph_id: scores[graph_id])
    all_graphs = tuple(score.graph_id for score in result.graph_scores if score.valid)
    full_decision = result.trajectory_summary.operational_decision
    value_config = value_config_from_generator(result.config)
    max_effects = _effects_for_graphs(result, (maximum_graph,))
    started = time.perf_counter()
    max_decision = evaluate_decisions(
        max_effects,
        (maximum_graph,),
        alpha=max(scores.values()),
        value_config=value_config,
        seed=(result.config.pilot_seed if result.config.pilot_seed is not None else result.config.seed),
    )
    max_runtime = time.perf_counter() - started
    hard_effects = _effects_for_graphs(result, all_graphs)
    started = time.perf_counter()
    hard_decision = evaluate_decisions(
        hard_effects,
        all_graphs,
        alpha=0.0,
        value_config=value_config,
        seed=(result.config.pilot_seed if result.config.pilot_seed is not None else result.config.seed),
    )
    hard_runtime = time.perf_counter() - started
    return {
        "full_procedure": MethodOutput(
            method="full_procedure",
            graph_ids=all_graphs,
            effects=_effects_for_graphs(result, all_graphs),
            decision=full_decision,
            decision_runtime_seconds=float(
                result.diagnostics.get("full_procedure_decision_seconds", 0.0)
            ),
        ),
        "maximum_graph": MethodOutput(
            method="maximum_graph",
            graph_ids=(maximum_graph,),
            effects=max_effects,
            decision=max_decision,
            decision_runtime_seconds=max_runtime,
        ),
        "hard_set": MethodOutput(
            method="hard_set",
            graph_ids=all_graphs,
            effects=hard_effects,
            decision=hard_decision,
            decision_runtime_seconds=hard_runtime,
        ),
    }


def run_ablations(result: AnalysisResult) -> dict[str, MethodOutput]:
    value_config = value_config_from_generator(result.config)
    fixed_cut = min(result.alpha_cuts, key=lambda cut: abs(cut.alpha - 0.80))
    fixed_effects = _effects_for_graphs(result, fixed_cut.graph_ids)
    fixed_decision = evaluate_decisions(
        fixed_effects,
        fixed_cut.graph_ids,
        alpha=0.80,
        value_config=value_config,
        seed=(result.config.pilot_seed if result.config.pilot_seed is not None else result.config.seed),
    )
    graphs = reference_graphs()
    evidence = reference_evidence(
        reliability_multiplier=result.config.evidence_reliability / 0.90,
        conflict_strength=result.config.evidence_conflict,
    )
    no_expert = EvidenceBundle(
        version=f"{evidence.version}-without-expert",
        context=evidence.context,
        items=tuple(item for item in evidence.items if item.evidence_type != "expert"),
    )
    no_expert_scores = tuple(score_graph(graph, no_expert) for graph in graphs)
    no_expert_cuts = tuple(
        alpha_cut(graphs, no_expert_scores, alpha)
        for alpha in sorted(result.config.alpha_grid, reverse=True)
    )
    widest = next(cut for cut in reversed(no_expert_cuts) if not cut.empty)
    no_expert_effects = _effects_for_graphs(result, widest.graph_ids)
    no_expert_decision = evaluate_decisions(
        no_expert_effects,
        widest.graph_ids,
        alpha=widest.alpha,
        value_config=value_config,
        seed=(result.config.pilot_seed if result.config.pilot_seed is not None else result.config.seed),
    )
    return {
        "ablation_fixed_alpha_080": MethodOutput(
            method="ablation_fixed_alpha_080",
            graph_ids=fixed_cut.graph_ids,
            effects=fixed_effects,
            decision=fixed_decision,
        ),
        "ablation_without_expert_evidence": MethodOutput(
            method="ablation_without_expert_evidence",
            graph_ids=widest.graph_ids,
            effects=no_expert_effects,
            decision=no_expert_decision,
        ),
    }
