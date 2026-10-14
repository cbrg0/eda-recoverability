"""Threshold fitting and the run set container."""

import numpy as np
import pytest

from edarecover.calibration import RunSet, needs_correction, youden_threshold
from edarecover.config import Settings


@pytest.fixture
def separable():
    coherence = np.array([0.02, 0.05, 0.08, 0.40, 0.60, 0.80])
    labels = np.array([0, 0, 0, 1, 1, 1])
    return coherence, labels


def test_threshold_separates_a_clean_two_class_problem(separable):
    coherence, labels = separable
    fitted = youden_threshold(coherence, labels)
    assert fitted.auc == pytest.approx(1.0)
    assert fitted.sensitivity == pytest.approx(1.0)
    assert fitted.specificity == pytest.approx(1.0)
    assert 0.08 < fitted.value <= 0.40


def test_a_single_class_leaves_the_threshold_undefined():
    with pytest.raises(ValueError, match="need both classes"):
        youden_threshold(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1]))


def test_decision_is_inclusive_at_the_threshold():
    assert needs_correction(0.156, 0.156)
    assert not needs_correction(0.1559, 0.156)


def test_labels_follow_the_tolerance():
    runs = RunSet(
        coherence=np.zeros(3),
        rmse_before=np.array([0.05, 0.08, 0.12]),
        rmse_after=np.zeros(3),
        blur=np.zeros(3),
        seed=np.zeros(3, dtype=int),
    )
    assert runs.labels(0.08).tolist() == [0, 0, 1]


def test_run_set_survives_a_save_and_load(tmp_path):
    runs = RunSet(
        coherence=np.array([0.1, 0.2]),
        rmse_before=np.array([0.01, 0.5]),
        rmse_after=np.array([0.07, 0.07]),
        blur=np.array([0.0, 6.0]),
        seed=np.array([0, 1]),
    )
    path = tmp_path / "nested" / "runs.npz"
    runs.save(path)
    restored = RunSet.load(path)

    assert len(restored) == 2
    assert np.array_equal(restored.coherence, runs.coherence)
    assert np.array_equal(restored.seed, runs.seed)


def test_default_threshold_matches_the_published_value():
    assert Settings().threshold == pytest.approx(0.156)
