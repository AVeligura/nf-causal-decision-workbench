from __future__ import annotations

from domain import CausalQuery, GraphSpec, IdentificationResult, IdentificationStatus

from .graphs import X_NODES, has_directed_path


def identify_effect(
    graph: GraphSpec,
    causal_query: CausalQuery,
    *,
    scenario: str = "reference",
    imported: bool = False,
) -> IdentificationResult:
    if scenario == "outside_gamma":
        return IdentificationResult(
            graph_id=graph.graph_id,
            query_id=causal_query.query_id,
            status=IdentificationStatus.NOT_IDENTIFIED,
            functional=None,
            assumptions=("Истинный механизм содержит ненаблюдаемый U вне Γ",),
            warnings=("Скрытое смешение T–Y не устраняется структурами G1–G4",),
        )
    if not has_directed_path(graph, causal_query.treatment, "Y"):
        return IdentificationResult(
            graph_id=graph.graph_id,
            query_id=causal_query.query_id,
            status=IdentificationStatus.STRUCTURAL_ZERO,
            functional="τ(G)=0 из отсутствия направленного пути T→Y",
            assumptions=("Полнота представленного графа относительно путей T→Y",),
            structural_origin=True,
        )
    if scenario == "informative_loss":
        return IdentificationResult(
            graph_id=graph.graph_id,
            query_id=causal_query.query_id,
            status=IdentificationStatus.PARTIALLY_IDENTIFIED,
            functional="Manski-type bounds under outcome-dependent selection",
            adjustment_set=X_NODES,
            assumptions=("Границы исхода заданы наблюдаемым устойчивым диапазоном",),
            warnings=("S зависит от промежуточных показателей и результата",),
        )
    if graph.graph_id == "G4":
        functional = "Последовательная g-формула: T→L→(D,Y)"
    else:
        functional = "E[m1(X)-m0(X)] с AIPW/DML и 5-fold cross-fitting"
    assumptions = [
        "Условная обменимость относительно предэкспозиционных X",
        "Согласованность версии полного пакета",
        "Положительность в целевой совокупности",
        "Отсутствие существенной интерференции",
    ]
    warnings: tuple[str, ...] = ()
    if imported:
        assumptions.append("Структура и потенциальные исходы импортированной выборки неизвестны")
        warnings = ("Oracle truth недоступна для импортированных данных",)
    return IdentificationResult(
        graph_id=graph.graph_id,
        query_id=causal_query.query_id,
        status=IdentificationStatus.IDENTIFIED,
        functional=functional,
        adjustment_set=X_NODES,
        assumptions=tuple(assumptions),
        warnings=warnings,
    )
