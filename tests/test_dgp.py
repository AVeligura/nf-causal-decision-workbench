from __future__ import annotations

import numpy as np
import pandas as pd

from domain import GeneratorConfig
from study import generate_dataset


def test_dgp_is_reproducible():
    config = GeneratorConfig(mode="laboratory", sample_size=300, seed=4567)
    first = generate_dataset(config)
    second = generate_dataset(config)
    pd.testing.assert_frame_equal(first.data, second.data)
    assert first.truth.true_ate == second.truth.true_ate


def test_reference_overlap_and_versions():
    generated = generate_dataset(GeneratorConfig(mode="laboratory", sample_size=1000))
    assert generated.diagnostics["propensity_min"] >= 0.15 - 1e-9
    assert generated.diagnostics["propensity_max"] <= 0.85 + 1e-9
    assert 0.04 <= generated.diagnostics["partial_share_assigned"] <= 0.17


def test_weak_overlap_reaches_requested_range():
    generated = generate_dataset(
        GeneratorConfig(mode="laboratory", scenario="weak_overlap", sample_size=1000)
    )
    assert generated.diagnostics["propensity_min"] <= 0.04
    assert generated.diagnostics["propensity_max"] >= 0.96


def test_version_mixing_targets_are_respected():
    generated = generate_dataset(
        GeneratorConfig(mode="laboratory", scenario="version_mixing", sample_size=2500)
    )
    assert abs(generated.diagnostics["partial_share_assigned"] - 0.20) < 0.04
    assert abs(generated.diagnostics["refusal_share_assigned"] - 0.10) < 0.03


def test_informative_loss_and_potential_outcomes_exist():
    generated = generate_dataset(
        GeneratorConfig(mode="laboratory", scenario="informative_loss", sample_size=1200)
    )
    assert 0.18 <= generated.diagnostics["loss_share"] <= 0.32
    y0, y1 = generated.truth.potential_outcomes["Y_CR"]
    assert len(y0) == len(y1) == 1200
    assert np.isfinite(y0).all() and np.isfinite(y1).all()


def test_outside_gamma_truth_is_explicit():
    generated = generate_dataset(
        GeneratorConfig(mode="laboratory", scenario="outside_gamma", sample_size=300)
    )
    assert generated.truth.true_graph_id is None
    assert generated.truth.hidden_factor_present
