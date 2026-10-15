"""Checks that the threshold survives outside the data it was fitted on.

Four questions are asked here. Does the threshold hold on runs excluded from
its calibration? Does the result depend on the assumed coupling between
electrode pressure and blur? Does it depend on knowing the blur operator
exactly? Does it depend on the sampling rate?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal as sps
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

from edarecover import forward, reconstruct, synthetic
from edarecover.arrays import Array, IntArray
from edarecover.calibration import RunSet, Threshold, youden_threshold
from edarecover.config import Settings
from edarecover.features import band_coherence, motion_envelope

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FoldResult:
    """One fold of the grouped cross-validation.

    Attributes:
        group: Seed held out in this fold.
        n_test: Number of runs held out.
        threshold: Value fitted on the remaining folds.
        sensitivity: True positive rate on the held-out runs.
        specificity: True negative rate on the held-out runs.
        accuracy: Fraction of held-out runs classified correctly.
    """

    group: int
    n_test: int
    threshold: float
    sensitivity: float
    specificity: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class CrossValidation:
    """Outcome of leaving out one group at a time.

    Attributes:
        folds: Per-fold results, in group order.
        pooled_sensitivity: Sensitivity over all runs, each judged by the
            threshold of the fold that excluded it.
        pooled_specificity: Specificity under the same pooling.
        pooled_accuracy: Accuracy under the same pooling.
        in_sample: Threshold fitted and measured on the whole set.
    """

    folds: tuple[FoldResult, ...]
    pooled_sensitivity: float
    pooled_specificity: float
    pooled_accuracy: float
    in_sample: Threshold

    @property
    def threshold_mean(self) -> float:
        """Mean of the per-fold thresholds."""
        return float(np.mean([f.threshold for f in self.folds]))

    @property
    def threshold_std(self) -> float:
        """Sample standard deviation of the per-fold thresholds."""
        return float(np.std([f.threshold for f in self.folds], ddof=1))


def _rates(predicted: IntArray, actual: IntArray) -> tuple[float, float, float]:
    """Sensitivity, specificity and accuracy, with NaN where a class is absent."""
    positive, negative = actual == 1, actual == 0
    sensitivity = float(np.mean(predicted[positive])) if np.any(positive) else np.nan
    specificity = (
        float(1 - np.mean(predicted[negative])) if np.any(negative) else np.nan
    )
    return sensitivity, specificity, float(np.mean(predicted == actual))


def cross_validate(runs: RunSet, settings: Settings) -> CrossValidation:
    """Refit the threshold with one seed held out at a time.

    Splitting run by run would be wrong: a seed fixes both the clean signal and
    the movement trajectory, so the runs that share a seed differ only in the
    blur parameter. Grouping by seed keeps correlated runs on the same side of
    every split, which is what :class:`LeaveOneGroupOut` enforces.

    Args:
        runs: Calibration set.
        settings: Supplies the tolerance that defines the labels.

    Returns:
        Per-fold and pooled results alongside the in-sample threshold.
    """
    labels = runs.labels(settings.rmse_tolerance)
    splitter = LeaveOneGroupOut()
    pooled = np.empty(len(runs), dtype=int)
    folds: list[FoldResult] = []

    for train_idx, test_idx in splitter.split(runs.coherence, labels, runs.seed):
        fitted = youden_threshold(runs.coherence[train_idx], labels[train_idx])
        predicted = (runs.coherence[test_idx] >= fitted.value).astype(int)
        pooled[test_idx] = predicted
        sensitivity, specificity, accuracy = _rates(predicted, labels[test_idx])
        folds.append(
            FoldResult(
                group=int(runs.seed[test_idx][0]),
                n_test=test_idx.size,
                threshold=fitted.value,
                sensitivity=sensitivity,
                specificity=specificity,
                accuracy=accuracy,
            )
        )

    sensitivity, specificity, accuracy = _rates(pooled, labels)
    return CrossValidation(
        folds=tuple(folds),
        pooled_sensitivity=sensitivity,
        pooled_specificity=specificity,
        pooled_accuracy=accuracy,
        in_sample=youden_threshold(runs.coherence, labels),
    )


@dataclass(frozen=True, slots=True)
class CouplingPoint:
    """The criterion refitted at one value of the pressure coupling.

    Attributes:
        beta_per_blur: Coupling used.
        n_positive: Runs whose error exceeded the tolerance.
        threshold: Refitted threshold, or NaN when one class was empty.
        auc: Area under the ROC curve, or NaN for the same reason.
        coherence_at_zero_blur: Mean coherence with no blur applied.
        coherence_at_max_blur: Mean coherence at the strongest blur.
    """

    beta_per_blur: float
    n_positive: int
    threshold: float
    auc: float
    coherence_at_zero_blur: float
    coherence_at_max_blur: float


def coupling_sweep(
    settings: Settings, couplings: tuple[float, ...], noise_seed: int = 0
) -> tuple[CouplingPoint, ...]:
    """Refit the threshold across a range of pressure couplings.

    The additive trace is what makes the distortion observable, and its
    amplitude is assumed rather than measured. This sweep says how much the
    conclusion depends on that assumption. Reconstruction is not needed: both
    the coherence and the label follow from the recording alone.

    Args:
        settings: Parameters held fixed across the sweep.
        couplings: Values of ``beta_per_blur`` to try.
        noise_seed: Seed of the measurement-noise generator, reset per value so
            that the runs differ only in the coupling.

    Returns:
        One entry per coupling, in the order given.
    """
    points: list[CouplingPoint] = []

    for beta in couplings:
        rng = np.random.default_rng(noise_seed)
        coherence, before, blur_of = [], [], []

        for blur_s in settings.blur_levels:
            for seed in range(settings.n_seeds):
                _, clean = synthetic.clean_signal(settings.window_s, settings.fs, seed)
                acc = synthetic.motion(settings.window_s, settings.fs, seed)
                envelope = motion_envelope(acc, settings.fs, settings.envelope_smooth_s)
                operator = forward.blur_operator(acc, settings.fs, blur_s)
                noise = settings.noise_std * rng.standard_normal(len(clean))
                recording = forward.observe(
                    clean, envelope, operator, blur_s, beta, noise
                )
                coherence.append(band_coherence(recording, envelope, settings))
                before.append(reconstruct.rmse(recording, clean))
                blur_of.append(blur_s)

        values = np.asarray(coherence, dtype=np.float64)
        blur = np.asarray(blur_of, dtype=np.float64)
        labels = (np.asarray(before) > settings.rmse_tolerance).astype(int)

        try:
            fitted = youden_threshold(values, labels)
            threshold, auc = fitted.value, fitted.auc
        except ValueError:
            log.info("coupling %.3f leaves one class empty", beta)
            threshold, auc = np.nan, np.nan

        points.append(
            CouplingPoint(
                beta_per_blur=beta,
                n_positive=int(np.sum(labels)),
                threshold=threshold,
                auc=auc,
                coherence_at_zero_blur=float(np.mean(values[blur == 0.0])),
                coherence_at_max_blur=float(
                    np.mean(values[blur == max(settings.blur_levels)])
                ),
            )
        )

    return tuple(points)


@dataclass(frozen=True, slots=True)
class MismatchPoint:
    """Reconstruction error when the inverting operator is wrong.

    Attributes:
        blur_s: True blur of the recording, s.
        factor: Ratio of assumed blur to true blur.
        rmse_after: Median error after correction, microsiemens.
    """

    blur_s: float
    factor: float
    rmse_after: float


def operator_mismatch(
    settings: Settings,
    blur_levels: tuple[float, ...],
    factors: tuple[float, ...],
) -> tuple[MismatchPoint, ...]:
    """Invert with an operator that differs from the generating one.

    Generating data with an operator and inverting with the same operator is an
    inverse crime: the reported error is a lower bound on what the method would
    achieve when the operator is unknown. Scaling the assumed blur shows how
    quickly the result degrades.

    Args:
        settings: Parameters of the runs.
        blur_levels: True blur widths to test, s.
        factors: Ratios of assumed to true blur.

    Returns:
        One entry per blur level and factor.
    """
    points: list[MismatchPoint] = []

    for blur_s in blur_levels:
        for factor in factors:
            rng = np.random.default_rng(100 + int(blur_s * 10))
            errors = []
            for seed in range(settings.n_seeds):
                _, clean = synthetic.clean_signal(settings.window_s, settings.fs, seed)
                acc = synthetic.motion(settings.window_s, settings.fs, seed)
                envelope = motion_envelope(acc, settings.fs, settings.envelope_smooth_s)
                true_operator = forward.blur_operator(acc, settings.fs, blur_s)
                noise = settings.noise_std * rng.standard_normal(len(clean))
                recording = forward.observe(
                    clean,
                    envelope,
                    true_operator,
                    blur_s,
                    settings.beta_per_blur,
                    noise,
                )
                assumed = forward.blur_operator(acc, settings.fs, blur_s * factor)
                penalty = reconstruct.second_difference_operator(len(clean))
                estimate = reconstruct.two_stage(
                    recording, envelope, assumed, penalty, settings
                )
                errors.append(reconstruct.rmse(estimate.signal, clean))

            points.append(
                MismatchPoint(
                    blur_s=blur_s,
                    factor=factor,
                    rmse_after=float(np.median(errors)),
                )
            )

    return tuple(points)


@dataclass(frozen=True, slots=True)
class DecimationCheck:
    """Coherence of one realisation measured at two sampling rates.

    Attributes:
        high: Values at the original rate.
        low: Values after anti-aliased decimation.
        threshold_high: Threshold fitted on ``high``.
        threshold_low: Threshold fitted on ``low``.
        max_abs_difference: Largest absolute difference between the two.
        correlation: Pearson correlation between the two.
    """

    high: Array
    low: Array
    threshold_high: Threshold
    threshold_low: Threshold
    max_abs_difference: float
    correlation: float


def decimation_invariance(settings: Settings, factor: int = 4) -> DecimationCheck:
    """Compare coherence before and after decimating the same realisation.

    Regenerating the signals at a lower rate would compare two different
    processes, because the motion generator puts energy between 1 and 3 Hz and
    that energy aliases below a Nyquist frequency of 2 Hz. Decimating one
    realisation isolates the effect of sampling alone, and it mirrors what the
    wrist-worn recordings require, where the accelerometer is anti-alias
    filtered before being brought down to the conductance rate.

    Args:
        settings: Parameters at the original rate.
        factor: Integer decimation factor.

    Returns:
        The paired values and the thresholds fitted on each.
    """
    low_settings = settings.at_sampling_rate(settings.fs / factor)
    rng = np.random.default_rng(0)
    high, low, before = [], [], []

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

            high.append(band_coherence(recording, envelope, settings))
            low.append(
                band_coherence(
                    sps.decimate(recording, factor, ftype="iir", zero_phase=True),
                    sps.decimate(envelope, factor, ftype="iir", zero_phase=True),
                    low_settings,
                )
            )
            before.append(reconstruct.rmse(recording, clean))

    high_arr, low_arr = np.asarray(high), np.asarray(low)
    labels = (np.asarray(before) > settings.rmse_tolerance).astype(int)

    return DecimationCheck(
        high=high_arr,
        low=low_arr,
        threshold_high=youden_threshold(high_arr, labels),
        threshold_low=youden_threshold(low_arr, labels),
        max_abs_difference=float(np.max(np.abs(low_arr - high_arr))),
        correlation=float(np.corrcoef(high_arr, low_arr)[0, 1]),
    )


@dataclass(frozen=True, slots=True)
class AblationPoint:
    """Error at one blur level under each variant of the procedure.

    Attributes:
        blur_s: Blur width, s.
        raw: Median error of the uncorrected recording, microsiemens.
        regression_only: Median error after removing the additive trace alone.
        inversion_only: Median error after inverting the blur alone.
        both_stages: Median error after the full procedure.
    """

    blur_s: float
    raw: float
    regression_only: float
    inversion_only: float
    both_stages: float


def stage_ablation(settings: Settings) -> tuple[AblationPoint, ...]:
    """Measure what each stage of the reconstruction contributes.

    The paper argues that the stages have to run in this order. The sharper
    question is whether the second stage earns its place at all, which only a
    comparison against each stage alone can answer.

    Args:
        settings: Parameters of the runs; the sweep covers every blur level.

    Returns:
        One entry per blur level, medians over the seeds.
    """
    rng = np.random.default_rng(0)
    errors: dict[float, dict[str, list[float]]] = {}

    for blur_s in settings.blur_levels:
        bucket = errors.setdefault(
            blur_s, {"raw": [], "regression": [], "inversion": [], "both": []}
        )
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

            bucket["raw"].append(reconstruct.rmse(recording, clean))
            bucket["regression"].append(
                reconstruct.rmse(
                    reconstruct.remove_additive_trace(recording, envelope), clean
                )
            )
            bucket["inversion"].append(
                reconstruct.rmse(
                    reconstruct.invert_blur(
                        recording, operator, penalty, settings
                    ).signal,
                    clean,
                )
            )
            bucket["both"].append(
                reconstruct.rmse(
                    reconstruct.two_stage(
                        recording, envelope, operator, penalty, settings
                    ).signal,
                    clean,
                )
            )

        log.info("blur %.1f s done", blur_s)

    return tuple(
        AblationPoint(
            blur_s=blur_s,
            raw=float(np.median(bucket["raw"])),
            regression_only=float(np.median(bucket["regression"])),
            inversion_only=float(np.median(bucket["inversion"])),
            both_stages=float(np.median(bucket["both"])),
        )
        for blur_s, bucket in errors.items()
    )


def auc_by_group(runs: RunSet, settings: Settings) -> dict[int, float]:
    """Area under the curve within each group.

    Reported for completeness and not as a performance estimate: each group
    holds only as many runs as there are blur levels, which is too few for the
    statistic to mean much.

    Args:
        runs: Calibration set.
        settings: Supplies the tolerance.

    Returns:
        Mapping from seed to area under the curve.
    """
    labels = runs.labels(settings.rmse_tolerance)
    out: dict[int, float] = {}
    for group in np.unique(runs.seed):
        mask = runs.seed == group
        if np.unique(labels[mask]).size < 2:
            continue
        out[int(group)] = float(roc_auc_score(labels[mask], runs.coherence[mask]))
    return out
