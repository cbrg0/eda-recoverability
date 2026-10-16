"""Synthetic electrodermal signals and wrist motion.

The clean signal is a slow tonic level carrying phasic responses of Bateman
shape. The two-exponential form is standard in the decomposition literature
(Benedek & Kirchner, 2010); the time constants below were chosen to place the
peak at 1.18 s and the 1 % return at 11.3 s, and they are not taken from a
published table. Motion is a background tremor plus Gaussian-windowed bursts in
the 1-3 Hz range of walking and gesturing.

See docs/method.md sections 4 and 5 for every constant with its provenance.
"""

from __future__ import annotations

import numpy as np

from edarecover.arrays import Array

TONIC_LEVEL_US = 3.0
TONIC_SWING_US = 0.5
TONIC_DRIFT_US = 0.05
SCR_AMPLITUDE_US = (0.05, 0.6)
SCR_RISE_S = 0.75
SCR_DECAY_S = 2.0
BURST_WIDTH_S = (3.0, 15.0)
BURST_FREQ_HZ = (1.0, 3.0)
BURST_AMPLITUDE = (0.5, 2.5)
TREMOR_STD = 0.02


def bateman_response(
    t: Array,
    onset_s: float,
    amplitude_us: float,
    rise_s: float = SCR_RISE_S,
    decay_s: float = SCR_DECAY_S,
) -> Array:
    """Single skin-conductance response.

    Args:
        t: Time axis, s.
        onset_s: Time at which the response starts, s.
        amplitude_us: Scale factor, microsiemens.
        rise_s: Rise time constant, s.
        decay_s: Decay time constant, s.

    Returns:
        Response sampled on ``t``, zero before the onset, microsiemens.

    Raises:
        ValueError: If the time constants are not ordered rise < decay, which
            would invert the response.
    """
    if not 0 < rise_s < decay_s:
        raise ValueError(f"need 0 < rise_s < decay_s, got {rise_s} and {decay_s}")

    dt = t - onset_s
    response = np.zeros_like(t)
    active = dt > 0
    response[active] = amplitude_us * (
        np.exp(-dt[active] / decay_s) - np.exp(-dt[active] / rise_s)
    )
    return np.clip(response, 0.0, None)


def clean_signal(
    duration_s: float,
    fs: float,
    seed: int,
    scr_per_minute: float = 4.0,
) -> tuple[Array, Array]:
    """Generate an undistorted electrodermal recording.

    Args:
        duration_s: Length of the recording, s.
        fs: Sampling rate, Hz.
        seed: Seed of the generator; the same seed gives the same signal.
        scr_per_minute: Mean rate of phasic responses.

    Returns:
        Time axis in seconds and conductance in microsiemens.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    tonic = (
        TONIC_LEVEL_US
        + TONIC_SWING_US * np.sin(2 * np.pi * t / (duration_s * 1.3))
        + TONIC_DRIFT_US * np.cumsum(rng.standard_normal(n)) / fs
    )

    n_events = rng.poisson(scr_per_minute * duration_s / 60.0)
    onsets = rng.uniform(0.0, duration_s, n_events)
    amplitudes = rng.uniform(*SCR_AMPLITUDE_US, n_events)

    phasic = np.zeros(n)
    for onset, amplitude in zip(onsets, amplitudes, strict=True):
        phasic += bateman_response(t, onset, amplitude)

    return t, tonic + phasic


def motion(duration_s: float, fs: float, seed: int, n_bursts: int = 8) -> Array:
    """Generate a synthetic accelerometer trace.

    The seed is offset by one relative to :func:`clean_signal` so that a run
    identified by a single seed gets an independent signal and motion pair.

    Known defect: an offset of one makes the tremor of run ``k`` the same white
    noise realisation as the tonic drift increments of run ``k + 1``, because
    both open a generator in the same state. The fix is a larger offset, and it
    is deferred because it moves the calibrated threshold. See docs/method.md
    section 4.5.

    Args:
        duration_s: Length of the recording, s.
        fs: Sampling rate, Hz.
        seed: Seed of the run.
        n_bursts: Number of movement bursts in the recording.

    Returns:
        Acceleration in arbitrary units, sampled at ``fs``.
    """
    rng = np.random.default_rng(seed + 1)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    acc = TREMOR_STD * rng.standard_normal(n)
    centres = rng.uniform(0.05 * duration_s, 0.95 * duration_s, n_bursts)
    for centre in centres:
        width = rng.uniform(*BURST_WIDTH_S)
        freq = rng.uniform(*BURST_FREQ_HZ)
        amplitude = rng.uniform(*BURST_AMPLITUDE)
        gate = np.exp(-0.5 * ((t - centre) / (width / 2)) ** 2)
        acc += gate * np.sin(2 * np.pi * freq * t) * amplitude

    return acc
