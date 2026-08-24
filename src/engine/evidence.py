from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from domain import AlphaCut, EvidenceBundle, EvidenceItem, GraphScore, GraphSpec

from .graphs import has_directed_path, validate_graph


@dataclass(frozen=True)
class _GroupContribution:
    score: float
    weight: float
    ids: tuple[str, ...]


def _item_compatibility(item: EvidenceItem, graph: GraphSpec) -> float | None:
    if item.target_graph is not None and item.target_graph != graph.graph_id:
        return None
    if item.assertion_kind == "soft_rule":
        return item.support
    if item.assertion_kind == "path":
        if item.path is None or item.expected_present is None:
            return None
        present = has_directed_path(graph, item.path[0], item.path[1])
        return item.support if present == item.expected_present else 1.0 - item.support
    if item.assertion_kind == "rule":
        if item.rule_id == "direct_or_no_liquidity_outcome":
            matches = ("T", "Y") in graph.edges or ("L", "Y") not in graph.edges
        elif item.rule_id == "mediated_direct_without_liquidity_outcome":
            matches = (
                ("L", "D") in graph.edges
                and ("T", "Y") in graph.edges
                and ("L", "Y") not in graph.edges
                and ("D", "L") not in graph.edges
            )
        else:
            return None
        expected = True if item.expected_present is None else item.expected_present
        return item.support if matches == expected else 1.0 - item.support
    if item.edge is None or item.expected_present is None:
        return None
    state_matches = (item.edge in graph.edges) == item.expected_present
    return item.support if state_matches else 1.0 - item.support


def score_graph(graph: GraphSpec, evidence: EvidenceBundle) -> GraphScore:
    """Compute compatibility exactly as formalised in section 7.6.2.

    Reliability and applicability form group weights; dependent records are
    collapsed before independent provenance groups are aggregated.  Conflict
    is penalised for directional edge assertions only.  Missing information is
    excluded, never recoded as 0.5.
    """

    valid, hard_warnings = validate_graph(graph, evidence)
    if not valid:
        return GraphScore(graph_id=graph.graph_id, mu=0.0, valid=False, warnings=hard_warnings)

    by_assertion_group: dict[tuple[str, str], list[tuple[EvidenceItem, float]]] = defaultdict(list)
    considered = 0
    potentially_relevant = 0
    for item in evidence.items:
        if item.constraint == "hard":
            continue
        if item.target_graph is None or item.target_graph == graph.graph_id:
            potentially_relevant += 1
        compatibility = _item_compatibility(item, graph)
        if compatibility is None or item.reliability is None or item.applicability <= 0:
            continue
        considered += 1
        by_assertion_group[(item.assertion_id, item.dependent_group)].append((item, compatibility))

    groups_by_assertion: dict[str, list[_GroupContribution]] = defaultdict(list)
    provenance: dict[str, list[str]] = defaultdict(list)
    kind_by_assertion: dict[str, str] = {}
    for (assertion_id, _group_id), entries in by_assertion_group.items():
        weighted = [
            (item.reliability * item.applicability, compatibility)
            for item, compatibility in entries
            if item.reliability is not None
        ]
        denominator = sum(weight for weight, _ in weighted)
        if denominator <= 0:
            continue
        content = sum(weight * value for weight, value in weighted) / denominator
        group_weight = max(weight for weight, _ in weighted)
        ids = tuple(item.evidence_id for item, _ in entries)
        groups_by_assertion[assertion_id].append(_GroupContribution(content, group_weight, ids))
        provenance[assertion_id].extend(ids)
        kind_by_assertion[assertion_id] = entries[0][0].assertion_kind

    local: dict[str, float] = {}
    corrected: dict[str, float] = {}
    conflicts: dict[str, float] = {}
    omega: dict[str, float] = {}
    for assertion_id, groups in groups_by_assertion.items():
        denominator = sum(group.weight for group in groups)
        if denominator <= 0:
            continue
        cq = sum(group.weight * group.score for group in groups) / denominator
        kappa = 0.0
        if kind_by_assertion.get(assertion_id) == "edge":
            w_plus = sum(group.weight for group in groups if group.score > 0.5)
            w_minus = sum(group.weight for group in groups if group.score < 0.5)
            if w_plus + w_minus > 0 and w_plus > 0 and w_minus > 0:
                kappa = 2.0 * min(w_plus, w_minus) / (w_plus + w_minus)
        local[assertion_id] = cq
        conflicts[assertion_id] = kappa
        corrected[assertion_id] = cq / (1.0 + kappa)
        omega[assertion_id] = sum(group.weight for group in groups) / len(groups)

    denominator = sum(omega.values())
    mu = sum(omega[q] * corrected[q] for q in corrected) / denominator if denominator else 0.0
    coverage = considered / potentially_relevant if potentially_relevant else 0.0
    warnings = list(hard_warnings)
    if coverage < 0.60:
        warnings.append("Недостаточное информационное покрытие структуры")
    if any(value >= 0.70 for value in conflicts.values()):
        warnings.append("Сильный конфликт независимых групп свидетельств")
    return GraphScore(
        graph_id=graph.graph_id,
        mu=round(float(mu), 8),
        valid=True,
        local_scores=local,
        corrected_scores=corrected,
        conflicts=conflicts,
        effective_weights=omega,
        coverage=coverage,
        provenance=dict(provenance),
        warnings=tuple(warnings),
    )


def extract_structural_core(
    graphs: tuple[GraphSpec, ...],
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    if not graphs:
        return (), ()
    sets = [set(graph.edges) for graph in graphs]
    core = set.intersection(*sets)
    union = set.union(*sets)
    alternatives = union - core
    return tuple(sorted(core)), tuple(sorted(alternatives))


def alpha_cut(
    graphs: tuple[GraphSpec, ...], scores: tuple[GraphScore, ...], alpha: float
) -> AlphaCut:
    score_map = {score.graph_id: score for score in scores}
    selected = tuple(
        graph
        for graph in graphs
        if score_map[graph.graph_id].valid and score_map[graph.graph_id].mu >= alpha - 1e-12
    )
    core, alternatives = extract_structural_core(selected)
    return AlphaCut(
        alpha=float(alpha),
        graph_ids=tuple(graph.graph_id for graph in selected),
        core_edges=core,
        alternative_edges=alternatives,
        empty=not selected,
    )


def reference_evidence(
    reliability_multiplier: float = 1.0,
    conflict_strength: float = 0.0,
) -> EvidenceBundle:
    """Pre-registered reference corpus yielding 0.92, 0.81, 0.67, 0.43.

    The values arise from edge, path and rule assertions with explicit source
    types and dependency groups.  No record is addressed to a graph id.
    """

    items: list[EvidenceItem] = []
    scaled_reliability = min(0.99, 0.90 * reliability_multiplier)

    def add_pair(
        *,
        assertion_id: str,
        assertion: str,
        kind: str,
        weight: float,
        sources: tuple[tuple[str, str], tuple[str, str]],
        edge: tuple[str, str] | None = None,
        path: tuple[str, str] | None = None,
        rule_id: str | None = None,
        expected_present: bool | None = True,
    ) -> None:
        applicability = min(1.0, weight / 0.90)
        for index, (evidence_type, provenance) in enumerate(sources, start=1):
            items.append(
                EvidenceItem(
                    evidence_id=f"{assertion_id}_{index}",
                    assertion_id=assertion_id,
                    assertion=assertion,
                    assertion_kind=kind,
                    edge=edge,
                    path=path,
                    rule_id=rule_id,
                    expected_present=expected_present,
                    support=1.0,
                    reliability=scaled_reliability,
                    applicability=applicability,
                    provenance=provenance,
                    evidence_type=evidence_type,
                    dependent_group=f"{assertion_id}_dependent",
                    context="financial_recovery_program",
                    comment="Зависимые записи объединяются до межгрупповой агрегации",
                )
            )

    add_pair(
        assertion_id="core_T_to_L",
        assertion="Полный пакет изменяет ликвидность предприятия (T→L)",
        kind="edge",
        edge=("T", "L"),
        weight=0.35,
        sources=(("panel", "panel_enterprise_quarters_v3"), ("regulatory", "program_design_2026")),
    )
    add_pair(
        assertion_id="admissible_outcome_channel",
        assertion="Допустим прямой канал T→Y либо отсутствие самостоятельного L→Y",
        kind="rule",
        rule_id="direct_or_no_liquidity_outcome",
        weight=0.24,
        sources=(("algorithmic", "causal_discovery_bootstrap_v2"), ("external", "external_review_2025")),
    )
    add_pair(
        assertion_id="no_independent_L_to_Y",
        assertion="После учёта долга самостоятельное ребро L→Y отсутствует",
        kind="edge",
        edge=("L", "Y"),
        expected_present=False,
        weight=0.14,
        sources=(("expert", "expert_panel_round_2"), ("regulatory", "program_logic_appendix")),
    )
    add_pair(
        assertion_id="direct_and_mediated_profile",
        assertion="Совместимы цепочка T→L→D→Y и прямой T→Y без отдельного L→Y",
        kind="rule",
        rule_id="mediated_direct_without_liquidity_outcome",
        weight=0.11,
        sources=(("interventional", "regional_pilot_wave_1"), ("panel", "matched_panel_sensitivity")),
    )
    items.append(
        EvidenceItem(
            evidence_id="external_path_neutral_1",
            assertion_id="external_path_neutral",
            assertion="Внешние исследования неоднозначны относительно наличия пути T→Y",
            assertion_kind="path",
            path=("T", "Y"),
            expected_present=True,
            support=0.5,
            reliability=0.80,
            applicability=0.20,
            provenance="systematic_external_review_mixed_findings",
            evidence_type="external",
            dependent_group="external_path_review",
            context="adjacent_policy_programs",
            comment="Нейтральная внешняя опора; не кодирует конкретный граф",
        )
    )

    if conflict_strength > 0:
        # Two independent, pre-outcome groups support competing direct-path claims.
        items.extend(
            [
                EvidenceItem(
                    evidence_id="conflict_direct_A",
                    assertion_id="direct_T_Y_conflict",
                    assertion="Панельные данные поддерживают прямое ребро T→Y",
                    assertion_kind="edge",
                    edge=("T", "Y"),
                    expected_present=True,
                    support=0.5 + 0.5 * conflict_strength,
                    reliability=0.85,
                    applicability=0.25,
                    provenance="independent_panel_group",
                    evidence_type="panel",
                    dependent_group="conflict_group_A",
                    conflict=True,
                ),
                EvidenceItem(
                    evidence_id="conflict_direct_B",
                    assertion_id="direct_T_Y_conflict",
                    assertion="Экспертная группа отвергает прямое ребро T→Y",
                    assertion_kind="edge",
                    edge=("T", "Y"),
                    expected_present=False,
                    support=0.5 + 0.5 * conflict_strength,
                    reliability=0.80,
                    applicability=0.25,
                    provenance="independent_expert_group",
                    evidence_type="expert",
                    dependent_group="conflict_group_B",
                    conflict=True,
                ),
            ]
        )
    return EvidenceBundle(
        version="reference-1.0" if conflict_strength == 0 else "conflict-1.0",
        context="enterprise_financial_recovery",
        items=tuple(items),
    )
