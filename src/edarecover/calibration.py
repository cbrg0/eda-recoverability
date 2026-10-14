"""Building the calibration set and choosing the threshold.

A run is one synthetic recording at one blur level and one seed. Each run
yields an observable coherence, the error before correction, the error after
correction, and a label saying whether the run needed correction at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from edarecover import forward, reconstruct, synthetic
from edarecover.arrays import Array, IntArray
from edarecover.config import Settings
from edarecover.features import band_coherence, motion_envelope

log = logging.getLogger(__name__)

CACHE_KEYS = ("coherence", "rmse_before", "rmse_after", "blur", "seed")


@dataclass(frozen=True, slots=True)
class RunSet:
    """Outcome of every run in a calibration sweep.

    Attributes:
        coherence: Band coherence of each run.
        rmse_before: Error of the uncorrected recording, microsiemens.
        rmse_after: Error after two-stage correction, microsiemens.
        blur: Blur width of each run, s.
        seed: Seed of each run, used as the grouping variable.
    """

    coherence: Array
    rmse_before: Array
    rmse_after: Array
    blur: Array
    seed: IntArray

    def __len__(self) -> int:
        return int(self.coherence.size)

    def labels(self, tolerance: float) -> IntArray:
        """Mark runs whose uncorrected error exceeds the tolerance."""
        return (self.rmse_before > tolerance).astype(int)

    def save(self, path: Path) -> None:
        """Write the run set to a compressed archive."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{key: getattr(self, key) for key in CACHE_KEYS})
        log.info("wrote %d runs to %s", len(self), path)

    @classmethod
    def load(cls, path: Path) -> RunSet:
        """Read a run set written by :meth:`save`."""
        with np.load(path) as archive:
            return cls(**{key: archive[key] for key in CACHE_KEYS})


@dataclass(frozen=True, slots=True)
class Threshold:
    """A decision threshold and how it performs where it was fitted.

    Attributes:
        value: Coherence above which correction is applied.
        auc: Area under the ROC curve on the fitting data.
        sensitivity: True positive rate at the threshold.
        specificity: True negative rate at the threshold.
    """

    value: float
    auc: float
    sensitivity: float
    specificity: float


def youden_threshold(coherence: Array, labels: IntArray) -> Threshold:
    """Pick the operating point maximising sensitivity plus specificity.

    Args:
        coherence: Observable values.
        labels: One for runs needing correction, zero otherwise.

    Returns:
        The threshold and its in-sample operating characteristics.

    Raises:
        ValueError: If only one class is present, which leaves the threshold
            undefined.
    """
    present = np.unique(labels)
    if present.size < 2:
        raise ValueError(
            f"need both classes to fit a threshold, got only class {present[0]}"
        )

    false_positive, true_positive, cuts = roc_curve(labels, coherence)
    best = int(np.argmax(true_positive - false_positive))
    return Threshold(
        value=float(cuts[best]),
        auc=float(roc_auc_score(labels, coherence)),
        sensitivity=float(true_positive[best]),
        specificity=float(1 - false_positive[best]),
    )


def needs_correction(coherence: float, threshold: float) -> bool:
    """Decide whether a window should be corrected."""
    return coherence >= threshold


def sweep(settings: Settings, noise_seed: int = 0) -> RunSet:
    """Run every blur level against every seed.

    The noise generator is created once and advanced across all runs, with
    blur as the outer loop and seed as the inner one. That order is part of the
    specification: changing it changes which noise sequence each run receives
    and moves the results in the fourth decimal place.

    Args:
        settings: Parameters of the sweep.
        noise_seed: Seed of the measurement-noise generator.

    Returns:
        One entry per run.
    """
    rng = np.random.default_rng(noise_seed)
    coherence, before, after, blur_of, seed_of = [], [], [], [], []

    for blur_s in settings.blur_levels:
        for seed in range(settings.n_seeds):
            _, clean = synthetic.clean_signal(settings.window_s, settings.fs, seed)
            acc = synthetic.motion(settings.window_s, settings.fs, seed)
            envelope = motion_envelope(acc, settings.fs, settings.envelope_smooth_s)
            operator = forward.blur_operator(acc, settings.fs, blur_s)
            noise = settings.noise_std * rng.standard_normal(len(clean))

            recording = forward.observe(
                clean, envelope, operator, blur_s, settings.beta_per_blur, noise
            )
            penalty = reconstruct.second_difference_operator(len(clean))
            estimate = reconstruct.two_stage(
                recording, envelope, operator, penalty, settings
            )

            coherence.append(band_coherence(recording, envelope, settings))
            before.append(reconstruct.rmse(recording, clean))
            after.append(reconstruct.rmse(estimate.signal, clean))
            blur_of.append(blur_s)
            seed_of.append(seed)

        log.info("blur %.1f s done", blur_s)

    return RunSet(
        coherence=np.asarray(coherence),
        rmse_before=np.asarray(before),
        rmse_after=np.asarray(after),
        blur=np.asarray(blur_of),
        seed=np.asarray(seed_of, dtype=np.int_),
    )


def load_or_sweep(settings: Settings, cache: Path, refresh: bool = False) -> RunSet:
    """Return a cached run set, computing and caching it when absent.

    Args:
        settings: Parameters of the sweep.
        cache: Archive path.
        refresh: Recompute even when the archive exists.

    Returns:
        The run set.
    """
    if cache.exists() and not refresh:
        log.info("reading cached runs from %s", cache)
        return RunSet.load(cache)

    log.info("computing %d runs", len(settings.blur_levels) * settings.n_seeds)
    runs = sweep(settings)
    runs.save(cache)
    return runs
