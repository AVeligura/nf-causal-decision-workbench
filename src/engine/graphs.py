from __future__ import annotations

import networkx as nx

from domain import EvidenceBundle, GraphSpec

X_NODES = ("X1", "X2", "X4", "X5", "X21", "X27", "X29", "X44")
ENDOGENOUS = ("T", "L", "D", "Y")


def reference_graphs() -> tuple[GraphSpec, ...]:
    nodes = (*X_NODES, *ENDOGENOUS)
    baseline_edges = tuple((x, v) for x in X_NODES for v in ENDOGENOUS)
    endogenous: dict[str, tuple[tuple[str, str], ...]] = {
        "G1": (("T", "L"), ("L", "D"), ("D", "Y"), ("T", "Y")),
        "G2": (("T", "L"), ("D", "L"), ("D", "Y")),
        "G3": (("T", "L"), ("L", "D"), ("D", "Y"), ("L", "Y"), ("T", "Y")),
        "G4": (("T", "L"), ("L", "D"), ("D", "Y"), ("L", "Y")),
    }
    descriptions = {
        "G1": "Прямой и опосредованный путь через ликвидность и долг",
        "G2": "Направленного пути T→Y нет; структурный нуль",
        "G3": "Прямой и два опосредованных пути",
        "G4": "Только опосредованные пути через L и D",
    }
    return tuple(
        GraphSpec(
            graph_id=graph_id,
            nodes=nodes,
            edges=baseline_edges + edges,
            description=descriptions[graph_id],
        )
        for graph_id, edges in endogenous.items()
    )


def validate_graph(
    graph: GraphSpec, evidence: EvidenceBundle | None = None
) -> tuple[bool, tuple[str, ...]]:
    warnings: list[str] = []
    directed = nx.DiGraph()
    directed.add_nodes_from(graph.nodes)
    directed.add_edges_from(graph.edges)
    if not nx.is_directed_acyclic_graph(directed):
        warnings.append("Ориентированный цикл нарушает класс DAG")
    unknown = [(a, b) for a, b in graph.edges if a not in graph.nodes or b not in graph.nodes]
    if unknown:
        warnings.append(f"Рёбра с неизвестными вершинами: {unknown}")
    if evidence is not None:
        for item in evidence.items:
            if item.constraint != "hard" or item.edge is None or item.expected_present is None:
                continue
            present = item.edge in graph.edges
            if present != item.expected_present:
                warnings.append(f"Нарушено hard-ограничение {item.evidence_id}: {item.assertion}")
    return not warnings, tuple(warnings)


def has_directed_path(graph: GraphSpec, source: str, target: str) -> bool:
    directed = nx.DiGraph(graph.edges)
    return source in directed and target in directed and nx.has_path(directed, source, target)
