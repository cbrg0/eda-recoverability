"""Movement envelope, band coherence and its noise floor."""

import numpy as np
import pytest

from edarecover import synthetic
from edarecover.config import Settings
from edarecover.features import (
    band_coherence,
    coherence_noise_floor,
    coherence_per_window,
    motion_envelope,
)


@pytest.fixture
def settings():
    return Settings()


def test_a_signal_is_perfectly_coherent_with_itself(settings):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(settings.samples_per_window)
    assert band_coherence(x, x, settings) == pytest.approx(1.0)


def test_scaling_and_shifting_do_not_change_coherence(settings):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(settings.samples_per_window)
    y = rng.standard_normal(settings.samples_per_window)
    assert band_coherence(x, y, settings) == pytest.approx(
        band_coherence(x, 5.0 * y + 2.0, settings)
    )


def test_unrelated_signals_sit_near_the_measured_floor(settings):
    mean, std = coherence_noise_floor(settings, n_trials=40, seed=7)
    assert 0.0 < mean < 0.12
    assert std < 0.1


def test_mismatched_lengths_are_rejected(settings):
    with pytest.raises(ValueError, match="differ in shape"):
        band_coherence(np.zeros(100), np.zeros(99), settings)


def test_windows_do_not_overlap_and_the_remainder_is_dropped(settings):
    n = int(2.5 * settings.samples_per_window)
    rng = np.random.default_rng(2)
    windows = coherence_per_window(
        rng.standard_normal(n), rng.standard_normal(n), settings
    )
    assert [w.number for w in windows] == [0, 1]
    assert windows[0].centre_s == pytest.approx(settings.window_s / 2)
    assert windows[1].centre_s == pytest.approx(1.5 * settings.window_s)


def test_envelope_is_non_negative_and_keeps_its_length():
    acc = synthetic.motion(60.0, 16.0, seed=0)
    envelope = motion_envelope(acc, 16.0, smooth_s=2.0)
    assert envelope.shape == acc.shape
    assert envelope.min() >= 0.0


def test_envelope_follows_motion_intensity_rather_than_its_oscillation():
    fs = 16.0
    t = np.arange(int(60 * fs)) / fs
    carrier = np.sin(2 * np.pi * 2.0 * t)
    gate = np.where(t < 30, 0.2, 2.0)
    envelope = motion_envelope(carrier * gate, fs, smooth_s=2.0)
    assert envelope[: len(t) // 2].mean() < envelope[len(t) // 2 :].mean()
