"""Grouped cross-validation and the per-group statistics."""

import numpy as np
import pytest

from edarecover.calibration import RunSet
from edarecover.config import Settings
from edarecover.validation import auc_by_group, cross_validate


@pytest.fixture
def runs():
    """Six seeds crossed with four blur levels, cleanly separable."""
    blur_levels = np.array([0.0, 0.2, 2.0, 6.0])
    seeds = np.arange(6)
    blur = np.tile(blur_levels, seeds.size)
    seed = np.repeat(seeds, blur_levels.size)
    rng = np.random.default_rng(0)
    coherence = 0.05 + 0.13 * blur + 0.004 * rng.standard_normal(blur.size)
    rmse_before = 0.01 + 0.14 * blur
    return RunSet(
        coherence=coherence,
        rmse_before=rmse_before,
        rmse_after=np.full(blur.size, 0.07),
        blur=blur,
        seed=seed,
    )


def test_every_group_is_held_out_exactly_once(runs):
    result = cross_validate(runs, Settings())
    assert len(result.folds) == 6
    assert sorted(f.group for f in result.folds) == list(range(6))
    assert all(f.n_test == 4 for f in result.folds)


def test_held_out_accuracy_does_not_exceed_the_in_sample_claim(runs):
    result = cross_validate(runs, Settings())
    in_sample = (result.in_sample.sensitivity + result.in_sample.specificity) / 2
    pooled = (result.pooled_sensitivity + result.pooled_specificity) / 2
    assert pooled <= in_sample + 1e-9


def test_threshold_spread_is_reported(runs):
    result = cross_validate(runs, Settings())
    assert result.threshold_std >= 0.0
    assert min(f.threshold for f in result.folds) <= result.threshold_mean
    assert result.threshold_mean <= max(f.threshold for f in result.folds)


def test_group_auc_skips_groups_holding_one_class(runs):
    single = RunSet(
        coherence=runs.coherence,
        rmse_before=np.full(runs.rmse_before.size, 0.5),
        rmse_after=runs.rmse_after,
        blur=runs.blur,
        seed=runs.seed,
    )
    assert auc_by_group(single, Settings()) == {}
    assert len(auc_by_group(runs, Settings())) == 6
