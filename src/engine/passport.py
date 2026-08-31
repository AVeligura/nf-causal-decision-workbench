from __future__ import annotations

from typing import Any

from domain import (
    AlphaCut,
    AlphaTrajectorySummary,
    CausalDecisionPassport,
    CausalQuery,
    DatasetSpec,
    DecisionResult,
    EffectResult,
    EvidenceBundle,
    GeneratorConfig,
    GraphScore,
    GraphSpec,
    IdentificationStatus,
    RunManifest,
    StabilityProfile,
)


def _validate_passport_inputs(
    effects: tuple[EffectResult, ...], decisions: tuple[DecisionResult, ...]
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for effect in effects:
        if effect.status == IdentificationStatus.NOT_IDENTIFIED and effect.estimate is not None:
            errors.append(
                f"{effect.graph_id}/{effect.outcome}: not_identified содержит точечную оценку"
            )
        if (
            effect.status == IdentificationStatus.PARTIALLY_IDENTIFIED
            and effect.identified_bounds is None
        ):
            errors.append(
                f"{effect.graph_id}/{effect.outcome}: отсутствуют границы partial identification"
            )
        if effect.status == IdentificationStatus.STRUCTURAL_ZERO and effect.interval is not None:
            errors.append(
                f"{effect.graph_id}/{effect.outcome}: структурный нуль не должен иметь интервал"
            )
    allowed = {"a0", "a1", "a2", None}
    for decision in decisions:
        if decision.selected_action not in allowed:
            errors.append(f"Недопустимое действие {decision.selected_action}")
        if decision.status == "robust" and decision.selected_action is None:
            errors.append("Робастный статус не содержит действия")
    if any(effect.warnings for effect in effects):
        warnings.append("Паспорт содержит диагностические предупреждения отдельных ветвей")
    return {"valid": not errors, "errors": errors, "warnings": warnings}


def build_causal_structure_passport(
    *,
    manifest: RunManifest,
    config: GeneratorConfig,
    queries: tuple[CausalQuery, ...],
    dataset_spec: DatasetSpec,
    evidence: EvidenceBundle,
    evidence_hash: str,
    analysis_input_hash: str,
    graphs: tuple[GraphSpec, ...],
    graph_scores: tuple[GraphScore, ...],
    alpha_cuts: tuple[AlphaCut, ...],
    effects: tuple[EffectResult, ...],
    stability: tuple[StabilityProfile, ...],
    decisions: tuple[DecisionResult, ...],
    trajectory_summary: AlphaTrajectorySummary,
    assumptions_and_limitations: tuple[str, ...],
) -> CausalDecisionPassport:
    validation = _validate_passport_inputs(effects, decisions)
    validation.update(
        {
            "input_identity_schema": "1.0",
            "config_hash": manifest.config_hash,
            "data_hash": manifest.data_hash,
            "evidence_hash": evidence_hash,
            "analysis_input_hash": analysis_input_hash,
            "canonical_config": config.model_dump(mode="json"),
        }
    )
    return CausalDecisionPassport(
        passport_version="1.1",
        manifest=manifest,
        causal_queries=queries,
        dataset=dataset_spec,
        evidence={
            "version": evidence.version,
            "context": evidence.context,
            "items": [item.as_dict() for item in evidence.items],
        },
        structural_space={
            "graphs": [graph.as_dict() for graph in graphs],
            "compatibility_semantics": "fuzzy_membership_not_probability",
            "scores": [score.as_dict() for score in graph_scores],
            "alpha_cuts": [cut.as_dict() for cut in alpha_cuts],
        },
        identification_and_estimation={
            "graph_specific_results": [effect.as_dict() for effect in effects]
        },
        uncertainty_profile={
            "components_are_not_normalized": True,
            "not_applicable_is_not_zero": True,
            "profiles": [profile.as_dict() for profile in stability],
        },
        alpha_stability={"trajectories": [profile.as_dict() for profile in stability]},
        decision={
            "trajectory": [decision.as_dict() for decision in decisions],
            "trajectory_summary": trajectory_summary.as_dict(),
        },
        assumptions_and_limitations=assumptions_and_limitations,
        validation=validation,
        audit_trail=(
            {
                "event": "passport_built",
                "run_id": manifest.run_id,
                "rule": "build_causal_structure_passport",
                "analysis_input_hash": analysis_input_hash,
            },
        ),
    )
