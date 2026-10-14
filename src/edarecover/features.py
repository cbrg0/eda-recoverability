"""Observable quantities computed from a recording and its accelerometer.

The criterion needs a number that can be computed in the field, where the blur
operator is unknown. Band-limited magnitude-squared coherence between the
recording and the movement envelope is that number.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import signal as sps

from edarecover.arrays import Array
from edarecover.config import Settings


class WindowCoherence(NamedTuple):
    """Coherence of one analysis window.

    Attributes:
        number: Position of the window in the recording, counting from zero.
        centre_s: Time at the middle of the window, s.
        value: Mean magnitude-squared coherence over the band.
    """

    number: int
    centre_s: float
    value: float


def motion_envelope(acc: Array, fs: float, smooth_s: float) -> Array:
    """Smooth the absolute accelerometer trace into an intensity envelope.

    Raw acceleration oscillates between roughly 1 and 3 Hz during walking,
    far above the band in which electrodermal activity carries sympathetic
    information. The skin responds to the envelope of that oscillation, and the
    envelope does fall inside the band.

    Args:
        acc: Accelerometer trace.
        fs: Sampling rate, Hz.
        smooth_s: Moving-average length, s.

    Returns:
        Envelope of the same length as ``acc``.
    """
    width = max(1, int(smooth_s * fs))
    return np.convolve(np.abs(acc), np.ones(width) / width, mode="same")


def band_coherence(recording: Array, envelope: Array, settings: Settings) -> float:
    """Mean coherence between a recording and an envelope over the band.

    Args:
        recording: Conductance over exactly one analysis window.
        envelope: Movement envelope over the same window.
        settings: Sampling rate, segment length and band.

    Returns:
        Mean magnitude-squared coherence over the band, in [0, 1].

    Raises:
        ValueError: If the inputs differ in length, or if the band contains no
            frequency bin at this resolution.
    """
    if recording.shape != envelope.shape:
        raise ValueError(
            f"recording and envelope differ in shape: "
            f"{recording.shape} and {envelope.shape}"
        )

    freqs, coherence = sps.coherence(
        recording, envelope, fs=settings.fs, nperseg=settings.samples_per_segment
    )
    low, high = settings.band_hz
    in_band = (freqs >= low) & (freqs <= high)
    if not in_band.any():
        raise ValueError(
            f"band {settings.band_hz} Hz holds no bin at resolution "
            f"{settings.frequency_resolution_hz:.4f} Hz; lengthen segment_s"
        )
    return float(coherence[in_band].mean())


def coherence_per_window(
    recording: Array, envelope: Array, settings: Settings
) -> list[WindowCoherence]:
    """Split a recording into windows and compute coherence in each.

    Windows do not overlap, so each value rests on its own data and the values
    can be treated as independent observations.

    Args:
        recording: Full conductance trace.
        envelope: Movement envelope of the same length.
        settings: Window length, segment length, sampling rate and band.

    Returns:
        One entry per complete window; a trailing partial window is dropped.
    """
    width = settings.samples_per_window
    out: list[WindowCoherence] = []
    for number, start in enumerate(range(0, len(recording) - width + 1, width)):
        stop = start + width
        out.append(
            WindowCoherence(
                number=number,
                centre_s=(start + width / 2) / settings.fs,
                value=band_coherence(
                    recording[start:stop], envelope[start:stop], settings
                ),
            )
        )
    return out


def coherence_noise_floor(
    settings: Settings, n_trials: int, seed: int
) -> tuple[float, float]:
    """Measure the upward bias of the coherence estimate on unrelated signals.

    A coherence estimate from a finite number of averages does not reach zero
    for independent inputs. Any threshold used as a decision rule has to sit
    above this floor, so the floor is measured rather than assumed.

    Args:
        settings: Parameters used for the real estimates.
        n_trials: Number of independent white-noise pairs.
        seed: Seed of the generator.

    Returns:
        Mean and sample standard deviation of the band coherence.
    """
    rng = np.random.default_rng(seed)
    n = settings.samples_per_window
    values = [
        band_coherence(rng.standard_normal(n), rng.standard_normal(n), settings)
        for _ in range(n_trials)
    ]
    return float(np.mean(values)), float(np.std(values, ddof=1))
