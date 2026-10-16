# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-10-16

### Added

- `docs/method.md`, the full method written in the order the code runs, with
  every equation, a table of every variable with its unit and value, and a
  column stating for each value whether it came from published work or was
  chosen for this study.

### Fixed

- Effective rank is now computed on the 240-second analysis window with the
  default eight movement bursts, which is the configuration the rest of the
  work uses. The earlier table came from a 60-second window with two bursts and
  is superseded: at six seconds of blur the surviving fraction is 0.389 rather
  than 0.294. Two bursts leave 47.6 per cent of the samples below a severity of
  0.05 against 3.1 per cent with eight, so nearly half the rows of the operator
  stayed close to the identity.
- The singular-value cutoff is documented as one per cent of the largest
  singular value, a relative quantity, rather than as an absolute noise level
  in microsiemens. The count itself is unchanged.
- The response time constants are no longer attributed to Alexander et al.
  (2005); the attribution could not be confirmed against that paper and has
  been withdrawn. The reference is removed from the README.
- The spectral estimate is documented as 23 overlapping Welch segments under a
  Hann window with 50 per cent overlap, with 12 given as the conservative count
  of independent averages. No printed value changes.
- Notation in `docs/method.md` no longer uses Unicode subscript glyphs or LaTeX
  brace syntax inside plain code fences, where neither renders. Summations are
  written `sum[...]`, ceilings and floors as `ceil(...)` and `floor(...)`,
  transposes as `^T`, and estimates as `S_hat` and `c_hat`.
- Two symbol collisions are resolved: the tonic constants are now `g_0`, `g_1`
  and `g_2` rather than `L`, which is the smoothness operator, the burst
  amplitude is `q_j` rather than `A`, which is the forward operator, and the
  error tolerance is `δ` rather than `τ`, which carries the time constants.

### Documented

- The random streams of the two generators overlap across seeds:
  `motion(seed=k)` and `clean_signal(seed=k+1)` open their bit generators in
  the same state. No reported statistic has been shown to move because of it.
  The fix is a larger offset and is deferred, because changing the streams
  moves the calibrated threshold in the third decimal place and 0.156 appears
  in submitted materials.

### Notes

Two figures quoted in the 0.3.0 entry below were imprecise and are restated
here from the current measurements. The coupling sweep covers 0.05 to 0.8 and
the area under the curve stays between 0.961 and 0.990 across it, while the
fitted threshold moves from 0.164 to 0.323. Operator mismatch costs 1.2 per
cent at one second of blur, 5.2 per cent at two and 20.4 per cent at four, so
"about five per cent" held only at the middle level.

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
