from __future__ import annotations

from domain import GraphSpec
from engine.evidence import alpha_cut, extract_structural_core, reference_evidence, score_graph
from engine.graphs import reference_graphs, validate_graph


def test_reference_graphs_are_dags():
    assert all(validate_graph(graph)[0] for graph in reference_graphs())


def test_cycle_is_rejected():
    graph = GraphSpec(graph_id="bad", nodes=("A", "B"), edges=(("A", "B"), ("B", "A")))
    valid, warnings = validate_graph(graph)
    assert not valid
    assert "цикл" in warnings[0].lower()


def test_reference_scores_are_reproduced_from_evidence():
    expected = {"G1": 0.92, "G2": 0.81, "G3": 0.67, "G4": 0.43}
    bundle = reference_evidence()
    actual = {graph.graph_id: score_graph(graph, bundle).mu for graph in reference_graphs()}
    assert actual == expected
    assert sum(actual.values()) != 1.0


def test_reliability_changes_score_without_direct_mu_edit():
    graphs = reference_graphs()
    baseline = score_graph(graphs[0], reference_evidence()).mu
    changed = score_graph(graphs[0], reference_evidence(reliability_multiplier=0.60)).mu
    assert baseline != changed


def test_alpha_cut_boundary_is_inclusive():
    graphs = reference_graphs()
    scores = tuple(score_graph(graph, reference_evidence()) for graph in graphs)
    cut = alpha_cut(graphs, scores, 0.81)
    assert cut.graph_ids == ("G1", "G2")
    assert alpha_cut(graphs, scores, 0.8100001).graph_ids == ("G1",)


def test_structural_core_and_alternatives():
    graphs = reference_graphs()
    core, alternatives = extract_structural_core(graphs)
    assert ("T", "L") in core
    assert ("D", "Y") in core
    assert ("T", "Y") in alternatives
    assert ("L", "D") in alternatives
