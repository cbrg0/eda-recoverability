"""The blur operator and the forward model."""

import numpy as np
import pytest

from edarecover import forward, synthetic


@pytest.fixture
def acc():
    return synthetic.motion(duration_s=60.0, fs=16.0, seed=0, n_bursts=2)


def test_rows_sum_to_one_so_the_mean_level_is_preserved(acc):
    operator = forward.blur_operator(acc, fs=16.0, max_blur_s=1.0)
    assert np.allclose(np.asarray(operator.sum(axis=1)).ravel(), 1.0)


def test_constant_signal_passes_through_unchanged(acc):
    operator = forward.blur_operator(acc, fs=16.0, max_blur_s=2.0)
    constant = np.full(len(acc), 3.0)
    assert np.allclose(operator @ constant, constant)


def test_stronger_blur_removes_more_degrees_of_freedom():
    acc = synthetic.motion(duration_s=20.0, fs=16.0, seed=0, n_bursts=2)
    ranks = [
        forward.effective_rank(forward.blur_operator(acc, 16.0, b), 0.01)[0]
        for b in (0.0, 1.0, 4.0)
    ]
    assert ranks[0] > ranks[1] > ranks[2]


def test_zero_blur_leaves_the_operator_at_full_rank():
    acc = synthetic.motion(duration_s=20.0, fs=16.0, seed=0, n_bursts=2)
    rank, fraction = forward.effective_rank(forward.blur_operator(acc, 16.0, 0.0), 0.01)
    assert fraction == pytest.approx(1.0)
    assert rank == len(acc)


def test_severity_of_a_flat_trace_is_zero_not_a_division_by_zero():
    assert np.all(forward.movement_severity(np.zeros(100), fs=16.0) == 0.0)


def test_severity_stays_within_the_unit_interval(acc):
    severity = forward.movement_severity(acc, fs=16.0)
    assert severity.min() == pytest.approx(0.0)
    assert severity.max() == pytest.approx(1.0)


def test_observation_adds_the_trace_and_the_noise(acc):
    operator = forward.blur_operator(acc, 16.0, 1.0)
    signal = np.full(len(acc), 3.0)
    envelope = np.ones(len(acc))
    noise = np.zeros(len(acc))
    observed = forward.observe(signal, envelope, operator, 1.0, 0.3, noise)
    assert np.allclose(observed - signal, 0.3)
