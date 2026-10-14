"""Both reconstruction stages and the L-curve."""

import numpy as np
import pytest
from scipy import sparse

from edarecover import forward, reconstruct, synthetic
from edarecover.config import Settings
from edarecover.features import motion_envelope


def test_second_difference_annihilates_a_straight_line():
    n = 200
    operator = reconstruct.second_difference_operator(n)
    assert np.allclose(operator @ np.linspace(0.0, 1.0, n), 0.0, atol=1e-12)


def test_second_difference_has_the_documented_shape():
    assert reconstruct.second_difference_operator(50).shape == (48, 50)


def test_second_difference_rejects_a_signal_too_short_for_it():
    with pytest.raises(ValueError, match="at least 3 samples"):
        reconstruct.second_difference_operator(2)


def test_tikhonov_without_regularisation_matches_a_direct_solve():
    rng = np.random.default_rng(0)
    n = 60
    operator = sparse.csr_matrix(np.eye(n) + 0.1 * rng.standard_normal((n, n)))
    penalty = reconstruct.second_difference_operator(n)
    target = rng.standard_normal(n)

    regularised = reconstruct.tikhonov(target, operator, penalty, lambda_=0.0)
    direct = np.linalg.solve(operator.toarray(), target)
    assert np.allclose(regularised, direct, atol=1e-8)


def test_stronger_regularisation_gives_a_smoother_estimate():
    rng = np.random.default_rng(1)
    n = 300
    operator = sparse.csr_matrix(sparse.eye(n))
    penalty = reconstruct.second_difference_operator(n)
    noisy = np.cumsum(rng.standard_normal(n)) / 10

    roughness = [
        np.linalg.norm(penalty @ reconstruct.tikhonov(noisy, operator, penalty, lam))
        for lam in (0.01, 1.0, 100.0)
    ]
    assert roughness[0] > roughness[1] > roughness[2]


def test_removing_the_additive_trace_keeps_the_tonic_level():
    rng = np.random.default_rng(2)
    n = 500
    envelope = np.abs(rng.standard_normal(n))
    signal = np.full(n, 3.0)
    cleaned = reconstruct.remove_additive_trace(signal + 0.4 * envelope, envelope)
    assert cleaned.mean() == pytest.approx(3.0, abs=1e-6)
    assert np.std(cleaned) < 1e-6


def test_two_stage_beats_the_raw_recording_at_strong_distortion():
    settings = Settings(window_s=120.0)
    _, clean = synthetic.clean_signal(settings.window_s, settings.fs, seed=0)
    acc = synthetic.motion(settings.window_s, settings.fs, seed=0)
    envelope = motion_envelope(acc, settings.fs, settings.envelope_smooth_s)
    operator = forward.blur_operator(acc, settings.fs, max_blur_s=2.0)
    noise = settings.noise_std * np.random.default_rng(0).standard_normal(len(clean))
    recording = forward.observe(
        clean, envelope, operator, 2.0, settings.beta_per_blur, noise
    )

    penalty = reconstruct.second_difference_operator(len(clean))
    estimate = reconstruct.two_stage(recording, envelope, operator, penalty, settings)

    assert reconstruct.rmse(estimate.signal, clean) < reconstruct.rmse(recording, clean)
    assert estimate.lambda_ > 0


def test_two_stage_equals_regression_followed_by_inversion():
    settings = Settings(window_s=120.0)
    _, clean = synthetic.clean_signal(settings.window_s, settings.fs, seed=1)
    acc = synthetic.motion(settings.window_s, settings.fs, seed=1)
    envelope = motion_envelope(acc, settings.fs, settings.envelope_smooth_s)
    operator = forward.blur_operator(acc, settings.fs, max_blur_s=1.0)
    noise = settings.noise_std * np.random.default_rng(1).standard_normal(len(clean))
    recording = forward.observe(
        clean, envelope, operator, 1.0, settings.beta_per_blur, noise
    )
    penalty = reconstruct.second_difference_operator(len(clean))

    combined = reconstruct.two_stage(recording, envelope, operator, penalty, settings)
    piecewise = reconstruct.invert_blur(
        reconstruct.remove_additive_trace(recording, envelope),
        operator,
        penalty,
        settings,
    )

    assert np.allclose(combined.signal, piecewise.signal)
    assert combined.lambda_ == piecewise.lambda_
