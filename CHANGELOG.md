# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-10-14

### Added

- `ablation` subcommand measuring each reconstruction stage separately.
- `--quick` flag reducing the sweep to a few seconds for a first look.
- `--json` output on every subcommand, so results can be piped.
- `reconstruct.invert_blur`, the second stage on its own.
- Regression test pinning the published threshold, area under the curve,
  sensitivity and specificity.
- Tests for the corpus reader and the command line.
- Continuous integration, pre-commit hooks, strict type checking, coverage.

### Changed

- `two_stage` is now the composition of `remove_additive_trace` and
  `invert_blur`; a test asserts the two paths agree exactly.

### Notes

The ablation shows the second stage contributes nothing to the error and
degrades it at strong blur: across all blur levels the first stage alone gives
0.0665 to 0.0685 microsiemens, while both stages give 0.0661 to 0.0769. The
reported threshold is unaffected, since it is calibrated on the uncorrected
error.

## [0.3.0] - 2026-10-13

### Added

- `coupling` subcommand sweeping the assumed pressure coupling. The criterion
  holds for couplings from 0.075 to 0.6 and has nothing to detect at zero.
- `mismatch` subcommand inverting with a deliberately wrong operator. Scaling
  the assumed blur by two changes the error by about five per cent.
- `sampling` subcommand comparing one realisation before and after anti-aliased
  decimation. The threshold transfers to 4 Hz unchanged.

### Changed

- Rewritten from three scripts into a package with a single parameter object,
  type annotations, logging and a command line. The published figures are
  reproduced to the printed digits.

### Removed

- Deprecated kernel builders, the superseded distortion model and the
  three-zone classifier. The three-zone hypothesis was already withdrawn: after
  correction the residual error does not track the observable, r = .05.

## [0.2.0] - 2026-10-08

### Added

- Leave-one-seed-out cross-validation. Held-out accuracy 0.94 against 0.95 in
  sample; the threshold moves from 0.156 to 0.160 with a standard deviation of
  0.009.

### Fixed

- The module self-check ran a withdrawn classifier with placeholder thresholds
  and printed figures that contradicted the paper.

## [0.1.0] - 2026-10-05

### Added

- Forward model, band-limited coherence, two-stage reconstruction and threshold
  calibration by Youden's index on 66 synthetic runs.
