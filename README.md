# edarecover

When is a motion-corrupted electrodermal recording worth correcting, and when
does correcting it make things worse?

A recording is modelled as a discrete linear inverse problem,

```
Y = A(acc) @ S + beta * env(acc) + noise
```

where `A` blurs the signal with a width that grows with movement intensity and
`beta * env` is an additive trace left by pressure on the electrode. What
governs recoverability is the singular spectrum of `A`, which cannot be
observed in the field, so band-limited magnitude-squared coherence between the
recording and the movement envelope over 0.045 to 0.25 Hz is used in its place.

The headline result is a decision rule rather than a filter. Above a coherence
of 0.156 the correction reduces the median error by a factor of 3.7; below it
the same correction increases the error by a factor of 3.6.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Python 3.11 or newer. NumPy, SciPy and scikit-learn are the only runtime
dependencies.

## Use

See something in a few seconds:

```bash
edarecover --quick calibrate
```

The published figures need the full sweep, which takes a few minutes and is
cached afterwards:

```bash
edarecover calibrate -v         # fit the threshold, write data/runs.npz
edarecover validate             # leave one seed out
edarecover coupling             # sweep the assumed pressure coupling
edarecover mismatch             # invert with a deliberately wrong operator
edarecover ablation             # measure each reconstruction stage alone
edarecover sampling             # decimate and compare
edarecover dalia path/to/corpus # run on PPG-DaLiA recordings
```

Every subcommand takes `--json`, so results can be piped:

```bash
edarecover --json validate | jq '.held_out.accuracy'
```

## What the experiments found

| Command | Question | Answer |
| --- | --- | --- |
| `calibrate` | Where is the threshold? | 0.156, AUC 0.979, on 66 runs of which 41 need correction |
| `validate` | Does it hold off its own data? | 0.160 (SD 0.009), held-out accuracy 0.94 against 0.95 in sample |
| `coupling` | How much does the assumed pressure coupling matter? | AUC stays 0.979 to 0.992 across an eightfold range; at zero coupling there is nothing to detect |
| `mismatch` | Must the blur operator be known exactly? | Scaling it by two moves the error by about five per cent |
| `ablation` | Does the second stage earn its place? | No. The regression alone gives 0.0665 to 0.0685 microsiemens across all blur levels; both stages give 0.0661 to 0.0769 |
| `sampling` | Does the threshold transfer to 4 Hz wrist data? | Yes. Largest difference 0.0005, correlation 1.0000, same threshold |

The ablation is the uncomfortable one and it is reported deliberately. The
regularised inversion contributes nothing to the error and degrades it at
strong blur, so the method that should ship is one regression plus the
threshold. The inverse-problem analysis keeps its place as the explanation:
at six seconds of blur the operator suppresses 71 per cent of the singular
directions while the error grows by less than 0.01 microsiemens, because the
energy of an electrodermal signal sits in the directions that survive.

## Layout

| Module | Contents |
| --- | --- |
| `arrays.py` | the two array aliases every other module annotates with |
| `config.py` | every parameter, in one frozen dataclass that validates itself |
| `synthetic.py` | clean conductance and wrist motion |
| `forward.py` | blur operator, forward model, effective rank |
| `features.py` | movement envelope, band coherence, its noise floor |
| `reconstruct.py` | the two stages, separately and composed |
| `calibration.py` | the run sweep and the Youden threshold |
| `validation.py` | grouped cross-validation and four robustness checks |
| `dalia.py` | reading the PPG-DaLiA corpus |
| `cli.py` | argument parsing, result payloads and rendering |

Experiments return frozen dataclasses and never print. Formatting lives in the
command layer, which is why the same numbers can be tested, rendered or dumped
as JSON.

## Reproducibility

The noise generator is created once with seed 0 and advanced across all runs,
blur as the outer loop and seed as the inner one. That order is part of the
specification: changing it changes which noise sequence each run receives and
moves the results in the fourth decimal place. Signal and motion are seeded per
run, so any run can be rebuilt from its `(blur, seed)` pair alone.

`tests/test_regression.py` pins the figures that appear in the paper and on the
poster, down to the printed digits, and runs in its own CI job.

## Checks

```bash
ruff check . && ruff format --check .
mypy
pytest -m "not slow"     # 52 tests, under a minute
pytest -m slow           # recomputes the sweep and pins the published numbers
```

The tests assert mathematical invariants rather than stored outputs: rows of
the blur operator sum to one, a constant passes through unchanged, a signal is
perfectly coherent with itself, coherence is invariant to scaling and offset,
Tikhonov at zero regularisation matches a direct solve, a second difference
annihilates a straight line, and held-out accuracy never exceeds the in-sample
claim.

## What this does not establish

Data are generated with `A` and inverted with the same `A`. In the inverse
problems literature that is an inverse crime, and it makes the reported errors
a lower bound on what the method would achieve when the operator is unknown.
`edarecover mismatch` measures the gap; validation on real recordings is still
outstanding.

The coupling `beta = 0.3 * b` ties the observable additive trace to the
unobservable blur by assumption, and no measurement in this work fixes it.
`edarecover coupling` reports how far the conclusion survives that uncertainty.

No comparison against a published artifact-detection baseline has been run.

## References

- Alexander, D. M., et al. (2005). Separating individual skin conductance
  responses in a short interstimulus-interval paradigm. *Journal of
  Neuroscience Methods*, 146(1), 116-123.
- Hansen, P. C. (1990). The discrete Picard condition for discrete ill-posed
  problems. *BIT*, 30, 658-672.
- Hansen, P. C., & O'Leary, D. P. (1993). The use of the L-curve in the
  regularization of discrete ill-posed problems. *SIAM Journal on Scientific
  Computing*, 14(6), 1487-1503.
- Posada-Quintero, H. F., et al. (2016). Power spectral density analysis of
  electrodermal activity for sympathetic function assessment. *Annals of
  Biomedical Engineering*, 44(10), 3124-3135.
- Reiss, A., et al. (2019). Deep PPG: large-scale heart rate estimation with
  convolutional neural networks. *Sensors*, 19(14), 3079.
- Welch, P. D. (1967). The use of fast Fourier transform for the estimation of
  power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15(2),
  70-73.
- Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer*, 3(1),
  32-35.

## Licence

MIT. See [LICENSE](LICENSE).
