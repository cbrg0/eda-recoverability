"""Forward model of a motion-corrupted recording.

The recording is modelled as ``Y = A @ S + beta * env + noise``: a blur operator
whose local width grows with movement intensity, plus an additive trace
proportional to the envelope of that intensity.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from edarecover.arrays import Array

MIN_SIGMA_SAMPLES = 0.5
SEVERITY_SMOOTH_S = 0.5
KERNEL_HALF_WIDTH_SIGMAS = 3


def movement_severity(acc: Array, fs: float) -> Array:
    """Rescale smoothed absolute acceleration to the unit interval.

    Args:
        acc: Accelerometer trace.
        fs: Sampling rate, Hz.

    Returns:
        Severity in [0, 1], one value per sample. A trace with no variation
        gives all zeros rather than a division by zero.
    """
    window = max(1, int(SEVERITY_SMOOTH_S * fs))
    smoothed: Array = np.convolve(np.abs(acc), np.ones(window) / window, mode="same")
    spread = float(smoothed.max() - smoothed.min())
    if spread < 1e-12:
        return np.zeros_like(smoothed)
    severity: Array = (smoothed - smoothed.min()) / spread
    return severity


def blur_operator(acc: Array, fs: float, max_blur_s: float) -> sparse.csr_matrix:
    """Build the position-dependent blur operator.

    Row ``i`` holds a Gaussian kernel centred on sample ``i`` and normalised to
    unit sum, so the operator preserves the mean level and acts on temporal
    detail only. The kernel width grows linearly with local movement severity.

    Args:
        acc: Accelerometer trace of the same length as the signal.
        fs: Sampling rate, Hz.
        max_blur_s: Kernel width where movement is strongest, s.

    Returns:
        Sparse operator of shape ``(len(acc), len(acc))``. The dense form does
        not fit in memory for recordings of a few thousand samples upward.
    """
    n = len(acc)
    max_sigma = max(MIN_SIGMA_SAMPLES, max_blur_s * fs)
    sigma = MIN_SIGMA_SAMPLES + movement_severity(acc, fs) * (
        max_sigma - MIN_SIGMA_SAMPLES
    )

    half_width = int(np.ceil(KERNEL_HALF_WIDTH_SIGMAS * max_sigma))
    offsets = np.arange(-half_width, half_width + 1)
    columns = np.arange(n)[:, None] + offsets[None, :]
    inside = (columns >= 0) & (columns < n)

    weights = np.exp(-0.5 * (offsets[None, :] / sigma[:, None]) ** 2)
    weights = np.where(inside, weights, 0.0)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    weights /= row_sums

    flat = inside.ravel()
    rows = np.repeat(np.arange(n), offsets.size)[flat]
    return sparse.csr_matrix(
        (weights.ravel()[flat], (rows, columns.ravel()[flat])), shape=(n, n)
    )


def observe(
    signal: Array,
    envelope: Array,
    operator: sparse.csr_matrix,
    blur_s: float,
    beta_per_blur: float,
    noise: Array,
) -> Array:
    """Apply the forward model to a clean signal.

    Noise is passed in rather than drawn here so that the caller controls the
    generator and the order in which it is consumed; that order is what makes
    a calibration run reproducible.

    Args:
        signal: Clean conductance, microsiemens.
        envelope: Envelope of movement intensity.
        operator: Blur operator built from the accelerometer trace.
        blur_s: Blur width of this run, s.
        beta_per_blur: Additive-trace amplitude per second of blur.
        noise: Measurement noise of the same length as ``signal``.

    Returns:
        Recorded conductance, microsiemens.
    """
    observed: Array = operator @ signal + blur_s * beta_per_blur * envelope + noise
    return observed


def effective_rank(
    operator: sparse.csr_matrix, noise_level: float
) -> tuple[int, float]:
    """Count singular values that stay above the noise floor.

    This is the discrete Picard condition of Hansen (1990). The condition
    number is not used instead: for a blur operator the smallest singular
    values fall below machine precision, so the ratio of largest to smallest is
    set by rounding rather than by how much of the signal survives.

    Args:
        operator: Blur operator; densified internally, so keep it small.
        noise_level: Noise as a fraction of the largest singular value.

    Returns:
        Count of singular values above the floor, and that count as a fraction
        of the full rank.
    """
    singular = np.linalg.svd(operator.toarray(), compute_uv=False)
    above = int(np.sum(singular > noise_level * singular[0]))
    return above, above / singular.size
