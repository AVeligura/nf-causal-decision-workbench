from __future__ import annotations

from study.monte_carlo import DesignCell, MonteCarloRunner, deterministic_replicate_seed


def test_seed_stream_is_deterministic_and_cell_specific():
    a = deterministic_replicate_seed(DesignCell("reference", 600), 3)
    b = deterministic_replicate_seed(DesignCell("reference", 600), 3)
    c = deterministic_replicate_seed(DesignCell("reference", 1500), 3)
    assert a == b
    assert a != c


def test_checkpoint_resume_and_no_duplicates(tmp_path):
    runner = MonteCarloRunner(tmp_path, "checkpoint-test")
    cell = (DesignCell("reference", 300),)
    assert runner.prepare(1, cell) == 1
    assert runner.prepare(1, cell) == 1
    counts = runner.run(workers=1, compute_cate=False)
    assert counts == {"completed": 1}
    results = runner.results()
    assert len(results["replicate_id"].unique()) == 1
    assert {
        "recommended_action",
        "decision_status",
        "operational_action",
        "operational_policy_accuracy",
        "recommendation_exact_match",
        "abstain_rate",
        "status_match",
        "operational_action_match",
    }.issubset(results.columns)
