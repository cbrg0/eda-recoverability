"""Two-stage reconstruction of a corrupted recording.

Stage one removes the additive trace by regressing the recording on the
movement envelope. Stage two inverts the blur by Tikhonov regularisation with a
second-difference penalty, the regularisation parameter chosen at the point of
maximum curvature of the L-curve (Hansen & O'Leary, 1993).

The order matters. With the additive trace still present, a single regularised
inversion assigns it to the signal, because the trace is smooth and so costs
little under the smoothness penalty.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from edarecover.arrays import Array
from edarecover.config import Settings


class Reconstruction(NamedTuple):
    """Result of the two-stage procedure.

    Attributes:
        signal: Estimated clean conductance, microsiemens.
        lambda_: Regularisation parameter selected on the L-curve.
    """

    signal: Array
    lambda_: float


def second_difference_operator(n: int) -> sparse.csr_matrix:
    """Second-difference operator used as the smoothness penalty.

    Args:
        n: Length of the signal the operator acts on.

    Returns:
        Sparse operator of shape ``(n - 2, n)`` with rows (1, -2, 1).

    Raises:
        ValueError: If ``n`` is too short for a second difference.
    """
    if n < 3:
        raise ValueError(f"need at least 3 samples for a second difference, got {n}")
    stencil = np.array([1.0, -2.0, 1.0])
    rows = np.repeat(np.arange(n - 2), stencil.size)
    cols = (np.arange(n - 2)[:, None] + np.arange(stencil.size)).ravel()
    return sparse.csr_matrix((np.tile(stencil, n - 2), (rows, cols)), shape=(n - 2, n))


def remove_additive_trace(recording: Array, envelope: Array) -> Array:
    """Regress out the component locked to movement intensity.

    The fitted envelope term is subtracted while the intercept is kept, so the
    tonic level stays and only the motion-locked part is removed.

    Args:
        recording: Conductance, microsiemens.
        envelope: Movement envelope of the same length.

    Returns:
        Recording with the envelope term removed, microsiemens.
    """
    design = np.column_stack([envelope, np.ones(len(recording))])
    coefficients, *_ = np.linalg.lstsq(design, recording, rcond=None)
    return recording - design @ coefficients + coefficients[1]


def tikhonov(
    recording: Array,
    operator: sparse.csr_matrix,
    penalty: sparse.csr_matrix,
    lambda_: float,
) -> Array:
    """Solve the regularised normal equations.

    Args:
        recording: Right-hand side, microsiemens.
        operator: Forward blur operator.
        penalty: Smoothness operator.
        lambda_: Regularisation parameter.

    Returns:
        Estimated signal, microsiemens.
    """
    left = (operator.T @ operator + lambda_**2 * penalty.T @ penalty).tocsc()
    return np.asarray(spsolve(left, operator.T @ recording), dtype=np.float64)


def _max_curvature_index(residual: Array, roughness: Array) -> int:
    """Index of the L-curve corner, found from finite differences of the logs."""
    log_r = np.log(residual + 1e-12)
    log_s = np.log(roughness + 1e-12)
    dr, ds = np.gradient(log_r), np.gradient(log_s)
    ddr, dds = np.gradient(dr), np.gradient(ds)
    curvature = np.abs(dr * dds - ds * ddr) / (dr**2 + ds**2 + 1e-12) ** 1.5
    return int(np.argmax(curvature))


def invert_blur(
    recording: Array,
    operator: sparse.csr_matrix,
    penalty: sparse.csr_matrix,
    settings: Settings,
) -> Reconstruction:
    """Stage two on its own: regularised inversion with lambda from the L-curve.

    Exposed separately so that the contribution of each stage can be measured
    against the other.

    Args:
        recording: Conductance over one window, microsiemens.
        operator: Blur operator used for the inversion.
        penalty: Smoothness operator.
        settings: Supplies the lambda grid.

    Returns:
        The estimate and the lambda chosen for it.
    """
    grid = np.logspace(*settings.lambda_log_bounds, settings.lambda_grid_size)
    estimates = [tikhonov(recording, operator, penalty, lam) for lam in grid]
    residual = np.array([np.linalg.norm(operator @ e - recording) for e in estimates])
    roughness = np.array([np.linalg.norm(penalty @ e) for e in estimates])

    corner = _max_curvature_index(residual, roughness)
    return Reconstruction(signal=estimates[corner], lambda_=float(grid[corner]))


def two_stage(
    recording: Array,
    envelope: Array,
    operator: sparse.csr_matrix,
    penalty: sparse.csr_matrix,
    settings: Settings,
) -> Reconstruction:
    """Remove the additive trace, then invert the blur.

    Args:
        recording: Conductance over one window, microsiemens.
        envelope: Movement envelope of the same length.
        operator: Blur operator used for the inversion.
        penalty: Smoothness operator.
        settings: Supplies the lambda grid.

    Returns:
        The estimate and the lambda chosen for it.
    """
    cleaned = remove_additive_trace(recording, envelope)
    return invert_blur(cleaned, operator, penalty, settings)


def rmse(estimate: Array, truth: Array) -> float:
    """Root mean square error between an estimate and the known signal."""
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))
