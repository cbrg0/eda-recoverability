"""Command line for the recoverability experiments.

Each subcommand computes a result and returns it as a plain dictionary. The
dictionary is then either rendered as a table or dumped as JSON, so the same
numbers can be read by a person or piped into another tool.

Run ``edarecover --help``, or ``python -m edarecover`` from a checkout.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from edarecover import validation
from edarecover.calibration import RunSet, load_or_sweep, sweep, youden_threshold
from edarecover.config import DEFAULT, Settings
from edarecover.dalia import (
    iter_subject_files,
    load_subject,
    report_subject,
    settings_for_dalia,
)
from edarecover.features import coherence_noise_floor

DEFAULT_CACHE = Path("data/runs.npz")
COUPLINGS = (0.0, 0.075, 0.15, 0.3, 0.6)
MISMATCH_BLURS = (0.4, 1.0, 2.0)
MISMATCH_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)

QUICK_BLUR_LEVELS = (0.0, 0.4, 1.0, 3.0)
QUICK_SEEDS = 2

Payload = dict[str, Any]
Renderer = Callable[[Payload], None]


class UsageError(Exception):
    """Raised when the arguments are valid but the request cannot be served."""


def quick(settings: Settings) -> Settings:
    """Shrink a sweep to something that returns in seconds.

    Useful for a first look and for smoke tests. The threshold it produces is
    not the published one and should not be quoted.
    """
    return dataclasses.replace(
        settings, blur_levels=QUICK_BLUR_LEVELS, n_seeds=QUICK_SEEDS
    )


def _runs_for(args: argparse.Namespace, settings: Settings) -> RunSet:
    """Load the cached sweep, or compute it when allowed to."""
    cache = args.cache
    if args.quick:
        return sweep(settings)
    if not cache.exists() and not getattr(args, "refresh", False):
        raise UsageError(
            f"no calibration cache at {cache}. "
            f"Run 'edarecover calibrate' first, or pass --quick for a short sweep."
        )
    return load_or_sweep(settings, cache, refresh=getattr(args, "refresh", False))


def calibrate(args: argparse.Namespace, settings: Settings) -> Payload:
    """Fit the threshold on the sweep and measure the coherence noise floor."""
    runs = (
        sweep(settings)
        if args.quick
        else load_or_sweep(settings, args.cache, refresh=args.refresh)
    )
    labels = runs.labels(settings.rmse_tolerance)
    fitted = youden_threshold(runs.coherence, labels)
    floor, floor_sd = coherence_noise_floor(settings, args.noise_trials, seed=7)

    return {
        "runs": len(runs),
        "needing_correction": int(np.sum(labels)),
        "threshold": fitted.value,
        "auc": fitted.auc,
        "sensitivity": fitted.sensitivity,
        "specificity": fitted.specificity,
        "correlation_with_residual": float(
            np.corrcoef(runs.coherence, runs.rmse_after)[0, 1]
        ),
        "coherence_floor": floor,
        "coherence_floor_sd": floor_sd,
    }


def render_calibrate(payload: Payload) -> None:
    """Print the calibration summary."""
    print(f"runs {payload['runs']}, needing correction {payload['needing_correction']}")
    print(
        f"threshold {payload['threshold']:.3f}  AUC {payload['auc']:.3f}  "
        f"sensitivity {payload['sensitivity']:.2f}  "
        f"specificity {payload['specificity']:.2f}"
    )
    print(
        "correlation of coherence with error after correction "
        f"r = {payload['correlation_with_residual']:.3f}"
    )
    print(
        f"coherence floor on unrelated signals {payload['coherence_floor']:.3f} "
        f"(SD {payload['coherence_floor_sd']:.3f})"
    )


def validate(args: argparse.Namespace, settings: Settings) -> Payload:
    """Refit the threshold with one seed held out at a time."""
    result = validation.cross_validate(_runs_for(args, settings), settings)
    return {
        "folds": [dataclasses.asdict(fold) for fold in result.folds],
        "threshold_mean": result.threshold_mean,
        "threshold_sd": result.threshold_std,
        "held_out": {
            "sensitivity": result.pooled_sensitivity,
            "specificity": result.pooled_specificity,
            "accuracy": result.pooled_accuracy,
        },
        "in_sample": {
            "threshold": result.in_sample.value,
            "auc": result.in_sample.auc,
            "sensitivity": result.in_sample.sensitivity,
            "specificity": result.in_sample.specificity,
        },
    }


def render_validate(payload: Payload) -> None:
    """Print the per-fold table and the pooled comparison."""
    print(f"{'seed':>5} {'n':>4} {'threshold':>10} {'sens':>7} {'spec':>7} {'acc':>7}")
    for fold in payload["folds"]:
        print(
            f"{fold['group']:>5} {fold['n_test']:>4} {fold['threshold']:>10.3f} "
            f"{fold['sensitivity']:>7.2f} {fold['specificity']:>7.2f} "
            f"{fold['accuracy']:>7.2f}"
        )
    held, sample = payload["held_out"], payload["in_sample"]
    print(
        f"\nthreshold across folds {payload['threshold_mean']:.3f} "
        f"(SD {payload['threshold_sd']:.3f})"
    )
    print(
        f"held out:  sensitivity {held['sensitivity']:.2f}  "
        f"specificity {held['specificity']:.2f}  accuracy {held['accuracy']:.2f}"
    )
    print(
        f"in sample: sensitivity {sample['sensitivity']:.2f}  "
        f"specificity {sample['specificity']:.2f}  "
        f"threshold {sample['threshold']:.3f}  AUC {sample['auc']:.3f}"
    )


def coupling(args: argparse.Namespace, settings: Settings) -> Payload:
    """Refit the threshold across a range of assumed pressure couplings."""
    del args
    points = validation.coupling_sweep(settings, COUPLINGS)
    return {"points": [dataclasses.asdict(point) for point in points]}


def render_coupling(payload: Payload) -> None:
    """Print one row per coupling."""
    print(
        f"{'beta/b':>7} {'n+':>4} {'threshold':>10} {'AUC':>7} "
        f"{'C_B at b=0':>11} {'C_B at b=max':>13}"
    )
    for point in payload["points"]:
        threshold = (
            "n/a" if np.isnan(point["threshold"]) else f"{point['threshold']:.3f}"
        )
        auc = "n/a" if np.isnan(point["auc"]) else f"{point['auc']:.3f}"
        print(
            f"{point['beta_per_blur']:>7.3f} {point['n_positive']:>4} "
            f"{threshold:>10} {auc:>7} "
            f"{point['coherence_at_zero_blur']:>11.3f} "
            f"{point['coherence_at_max_blur']:>13.3f}"
        )


def mismatch(args: argparse.Namespace, settings: Settings) -> Payload:
    """Invert with an operator that differs from the generating one."""
    del args
    points = validation.operator_mismatch(settings, MISMATCH_BLURS, MISMATCH_FACTORS)
    return {"points": [dataclasses.asdict(point) for point in points]}


def render_mismatch(payload: Payload) -> None:
    """Print error against the ratio of assumed to true blur."""
    table: dict[float, dict[float, float]] = {}
    for point in payload["points"]:
        table.setdefault(point["blur_s"], {})[point["factor"]] = point["rmse_after"]

    factors = sorted(next(iter(table.values())))
    print(f"{'blur (s)':>9} " + " ".join(f"{'x' + str(f):>9}" for f in factors))
    for blur_s in sorted(table):
        cells = " ".join(f"{table[blur_s][f]:>9.4f}" for f in factors)
        print(f"{blur_s:>9.1f} {cells}")


def ablation(args: argparse.Namespace, settings: Settings) -> Payload:
    """Measure each reconstruction stage on its own."""
    del args
    points = validation.stage_ablation(settings)
    return {"points": [dataclasses.asdict(point) for point in points]}


def render_ablation(payload: Payload) -> None:
    """Print median error under each variant of the procedure."""
    print(f"{'blur (s)':>9} {'raw':>10} {'stage 1':>10} {'stage 2':>10} {'both':>10}")
    for point in payload["points"]:
        print(
            f"{point['blur_s']:>9.1f} {point['raw']:>10.4f} "
            f"{point['regression_only']:>10.4f} {point['inversion_only']:>10.4f} "
            f"{point['both_stages']:>10.4f}"
        )


def sampling(args: argparse.Namespace, settings: Settings) -> Payload:
    """Compare one realisation before and after anti-aliased decimation."""
    check = validation.decimation_invariance(settings, factor=args.factor)
    return {
        "factor": args.factor,
        "fs_high": settings.fs,
        "fs_low": settings.fs / args.factor,
        "threshold_high": check.threshold_high.value,
        "threshold_low": check.threshold_low.value,
        "auc_high": check.threshold_high.auc,
        "auc_low": check.threshold_low.auc,
        "max_abs_difference": check.max_abs_difference,
        "correlation": check.correlation,
    }


def render_sampling(payload: Payload) -> None:
    """Print the thresholds at both rates and how far the values moved."""
    print(
        f"threshold at {payload['fs_high']:.0f} Hz "
        f"{payload['threshold_high']:.3f}  AUC {payload['auc_high']:.3f}"
    )
    print(
        f"threshold at {payload['fs_low']:.0f} Hz "
        f"{payload['threshold_low']:.3f}  AUC {payload['auc_low']:.3f}"
    )
    print(
        f"largest absolute difference {payload['max_abs_difference']:.4f}, "
        f"correlation r = {payload['correlation']:.4f}"
    )


def dalia(args: argparse.Namespace, settings: Settings) -> Payload:
    """Classify every window of every subject in a corpus directory."""
    corpus = settings_for_dalia(settings)
    files = iter_subject_files(args.root)
    if not files:
        raise UsageError(f"no subject files under {args.root}")

    subjects = []
    for path in files:
        reports = report_subject(load_subject(path, corpus), corpus)
        values = [report.coherence for report in reports]
        subjects.append(
            {
                "subject": reports[0].subject if reports else path.stem,
                "windows": len(values),
                "first": values[0] if values else None,
                "median_middle": (
                    float(np.median(values[1:-1])) if len(values) > 2 else None
                ),
                "last": values[-1] if values else None,
                "needing_correction": sum(r.correct for r in reports),
            }
        )
    return {"subjects": subjects}


def render_dalia(payload: Payload) -> None:
    """Print one row per subject."""
    print(
        f"{'subject':>8} {'windows':>8} {'first':>8} {'median':>8} "
        f"{'last':>8} {'correct':>8}"
    )
    for row in payload["subjects"]:
        middle = (
            "n/a" if row["median_middle"] is None else f"{row['median_middle']:.4f}"
        )
        print(
            f"{row['subject']:>8} {row['windows']:>8} {row['first']:>8.4f} "
            f"{middle:>8} {row['last']:>8.4f} {row['needing_correction']:>8}"
        )


COMMANDS: dict[str, tuple[Callable[..., Payload], Renderer, str]] = {
    "calibrate": (calibrate, render_calibrate, "fit the threshold"),
    "validate": (validate, render_validate, "leave one seed out"),
    "coupling": (coupling, render_coupling, "sweep the pressure coupling"),
    "mismatch": (mismatch, render_mismatch, "invert with the wrong operator"),
    "ablation": (ablation, render_ablation, "measure each stage separately"),
    "sampling": (sampling, render_sampling, "decimate and compare"),
    "dalia": (dalia, render_dalia, "run on PPG-DaLiA recordings"),
}


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser."""
    parser = argparse.ArgumentParser(
        prog="edarecover",
        description="Recoverability of electrodermal signals under motion",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log progress to stderr"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the result as JSON instead"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="shrink the sweep so it returns in seconds; figures are not the "
        "published ones",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"calibration cache (default: {DEFAULT_CACHE})",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    for name, (handler, renderer, help_text) in COMMANDS.items():
        command = sub.add_parser(name, help=help_text)
        command.set_defaults(handler=handler, renderer=renderer)
        if name == "calibrate":
            command.add_argument(
                "--refresh", action="store_true", help="ignore the cache"
            )
            command.add_argument(
                "--noise-trials",
                type=int,
                default=200,
                help="pairs used to measure the coherence floor",
            )
        if name == "sampling":
            command.add_argument(
                "--factor", type=int, default=4, help="decimation factor"
            )
        if name == "dalia":
            command.add_argument(
                "root", type=Path, help="directory holding S1/S1.pkl and so on"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the subcommand and render its result.

    Args:
        argv: Arguments to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit status: zero on success, two on a usage error.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = quick(DEFAULT) if args.quick else DEFAULT
    try:
        payload = args.handler(args, settings)
    except UsageError as error:
        print(f"edarecover: {error}", file=sys.stderr)
        return 2

    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        args.renderer(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
