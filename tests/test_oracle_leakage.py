from __future__ import annotations

import inspect

import engine
from engine import pipeline
from study import methods


def test_ordinary_interfaces_do_not_accept_truth_bundle():
    ordinary = [
        engine.validate_graph,
        engine.score_graph,
        engine.alpha_cut,
        engine.extract_structural_core,
        engine.identify_effect,
        engine.estimate_effect,
        engine.propagate_graph_uncertainty,
        engine.assess_effect_stability,
        engine.evaluate_decisions,
        engine.build_causal_structure_passport,
        pipeline.run_analysis,
        methods.run_comparison_methods,
        methods.run_ablations,
    ]
    for function in ordinary:
        signature = str(inspect.signature(function))
        assert "TruthBundle" not in signature
        assert "truth" not in signature.lower()


def test_truth_bundle_not_reexported_by_domain():
    import domain

    assert not hasattr(domain, "TruthBundle")
