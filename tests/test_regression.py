"""The figures that appear in the paper and on the poster.

These pin the published numbers to the digits that are printed. A failure here
is not a refactor: either the change is wrong, or the paper has to change with
it. The whole sweep is recomputed, which takes several minutes, so the module
is marked slow and left out of the default run.
"""

from __future__ import annotations

import numpy as np
import pytest

from edarecover.calibration import sweep, youden_threshold
from edarecover.config import Settings
from edarecover.validation import cross_validate

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def runs():
    return sweep(Settings())


def test_the_sweep_has_the_published_shape(runs, settings: Settings):
    assert len(runs) == 66
    assert runs.labels(settings.rmse_tolerance).sum() == 41
    assert sorted(set(runs.seed.tolist())) == list(range(6))
    assert len(set(runs.blur.tolist())) == 11


def test_threshold_and_operating_point(runs, settings: Settings):
    fitted = youden_threshold(runs.coherence, runs.labels(settings.rmse_tolerance))
    assert fitted.value == pytest.approx(0.156, abs=5e-4)
    assert fitted.auc == pytest.approx(0.979, abs=5e-4)
    assert fitted.sensitivity == pytest.approx(1.00, abs=5e-3)
    assert fitted.specificity == pytest.approx(0.88, abs=5e-3)


def test_residual_error_does_not_track_the_observable(runs):
    correlation = np.corrcoef(runs.coherence, runs.rmse_after)[0, 1]
    assert correlation == pytest.approx(0.053, abs=5e-4)


def test_zone_medians_match_the_poster(runs, settings: Settings):
    below = runs.coherence < settings.threshold
    above = ~below
    assert below.sum() == 22
    assert above.sum() == 44

    assert np.median(runs.rmse_before[below]) == pytest.approx(0.0189, abs=5e-5)
    assert np.median(runs.rmse_after[below]) == pytest.approx(0.0672, abs=5e-5)
    assert np.median(runs.rmse_before[above]) == pytest.approx(0.2658, abs=5e-5)
    assert np.median(runs.rmse_after[above]) == pytest.approx(0.0713, abs=5e-5)


def test_correction_helps_above_the_threshold_and_hurts_below(runs, settings: Settings):
    below = runs.coherence < settings.threshold
    above = ~below
    harm = np.median(runs.rmse_after[below]) / np.median(runs.rmse_before[below])
    gain = np.median(runs.rmse_before[above]) / np.median(runs.rmse_after[above])
    assert harm == pytest.approx(3.6, abs=0.1)
    assert gain == pytest.approx(3.7, abs=0.1)


def test_held_out_numbers(runs, settings: Settings):
    result = cross_validate(runs, settings)
    assert result.threshold_mean == pytest.approx(0.160, abs=5e-4)
    assert result.threshold_std == pytest.approx(0.009, abs=5e-4)
    assert result.pooled_sensitivity == pytest.approx(0.98, abs=5e-3)
    assert result.pooled_specificity == pytest.approx(0.88, abs=5e-3)
    assert result.pooled_accuracy == pytest.approx(0.94, abs=5e-3)
