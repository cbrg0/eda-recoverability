# edarecover

When is a motion-corrupted electrodermal recording worth correcting, and when
does correcting it make things worse?

Wrist-worn sensors record skin conductance to track sympathetic arousal, and
every time the wrist moves, the electrode contact changes while the
physiological signal of interest is still arriving. The usual response is to
detect the corrupted stretches and either repair them or discard them. What that
approach does not supply is a limit: past some amount of distortion the
information needed to tell the true signal from a plausible imitation of it is no
longer in the data, and a repair applied past that point returns an estimate that
looks clean, carries no warning, and is wrong.

This package locates that limit and turns it into a decision rule that a device
can evaluate while it is recording. A recording is modelled as a discrete linear
inverse problem,

```
Y = A(acc) * S + beta * env(acc) + noise
```

where `A` blurs the signal with a width that grows with movement intensity and
`beta * env` is an additive trace left by pressure on the electrode. What governs
recoverability is the singular spectrum of `A`, which cannot be computed in the
field because it would require knowing the operator in advance, so band-limited
magnitude-squared coherence between the recording and the movement envelope over
0.045 to 0.25 Hz is used in its place. That band is where electrodermal activity
carries sympathetic information, and coherence against the accelerometer asks
whether movement has contaminated the part of the spectrum the measurement is
actually for.

The headline result is a decision rule rather than a filter. Above a coherence of
0.156 the correction reduces the median error by a factor of 3.7, from 0.2658 to
0.0713 microsiemens; below it the same correction increases the median error by a
factor of 3.6, from 0.0189 to 0.0672. One coherence evaluation costs 2.6 ms
against 4,045 ms for a full reconstruction on the same window, so gating the
correction is roughly three orders of magnitude cheaper than always running it.

Everything here is calibrated on simulated recordings, because a threshold on
reconstruction error can only be calibrated where the clean signal is known, and
on a human recording it is not. The criterion has not yet been run against human
data. [`docs/method.md`](docs/method.md) is the full method, written step by step
in the order the code runs: every equation, a table of every variable with its
unit and its value, and a column saying for each value whether it came from
published work or was chosen for this study.

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
cached afterwards. Global flags go before the subcommand, as in
`git -C dir status`:

```bash
edarecover -v calibrate         # fit the threshold, write data/runs.npz
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
| `coupling` | How much does the assumed pressure coupling matter? | For the ranking, very little: AUC stays between 0.961 and 0.990 across a sixteenfold range. For the number, a great deal: the threshold moves from 0.164 to 0.323 |
| `mismatch` | Must the blur operator be known exactly? | No, and that is a symptom rather than a strength. Doubling the assumed width costs 1.2 per cent at one second of blur and 20.4 per cent at four |
| `ablation` | Does the second stage earn its place? | No. The regression alone gives 0.0665 to 0.0685 microsiemens across all blur levels; both stages give 0.0661 to 0.0769 |
| `sampling` | Does the threshold transfer to 4 Hz wrist data? | Yes. Largest difference 0.00053, correlation 0.9999999, threshold 0.15613 against 0.15586 |

Two of these results work against the method as specified, and they are reported
because a reader who reproduces the package will find them anyway.

The ablation is the blunt one. The regularised inversion contributes nothing to
the error at low blur and degrades it at high blur, where applying it alone
leaves 1.0409 microsiemens against the 0.8944 of touching nothing, so the method
that should ship is one regression plus the threshold. The two-stage ordering
argument turns out to be right about the order and wrong about the value of the
second step.

The coupling sweep is the one that limits how far the number travels. The
observable separates recoverable from unrecoverable windows about equally well
whatever the coupling is, so the procedure transfers; the fitted threshold varies
by a factor of two across the same range, so 0.156 does not. A device would have
to calibrate its own threshold against its own hardware. At zero coupling the
criterion has nothing to detect at all, and the coherence moves the wrong way
with blur, falling to 0.031 at six seconds against 0.050 at none.

The inverse-problem analysis keeps its place as the explanation rather than as
the mechanism. At six seconds of blur the operator suppresses 61 per cent of the
singular directions while the uncorrected error grows only from 0.0100 to 0.0190
microsiemens, never reaching a quarter of the 0.08 tolerance that defines a
label, because the energy of an electrodermal signal sits in the slow directions
that a row-normalised Gaussian average leaves nearly untouched. The error the
criterion detects comes from the additive trace.

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
| `validation.py` | grouped cross-validation and four checks on the assumptions |
| `dalia.py` | reading the PPG-DaLiA corpus |
| `cli.py` | argument parsing, result payloads and rendering |

Experiments return frozen dataclasses and never print. Formatting lives in the
command layer, which is why the same numbers can be tested, rendered or dumped as
JSON.

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
pytest -m "not slow"     # 54 tests, under a minute
pytest -m slow           # 6 tests, recomputes the sweep and pins the published numbers
```

The tests assert mathematical invariants rather than stored outputs: rows of the
blur operator sum to one, a constant passes through unchanged, a signal is
perfectly coherent with itself, coherence is invariant to scaling and offset,
Tikhonov at zero regularisation matches a direct solve, a second difference
annihilates a straight line, and held-out accuracy never exceeds the in-sample
claim.

## What this does not establish

The criterion has never been applied to a human recording. Every number in this
repository comes from simulation, `src/edarecover/dalia.py` reads the PPG-DaLiA
corpus and that run has not been made, and until it is, 0.156 is a property of
this model rather than of wrist electrodermal recording.

The coupling `beta = 0.3 * b` ties the observable additive trace to the
unobservable blur by assumption, and no measurement in this work fixes it.
`edarecover coupling` reports how far the conclusion survives that uncertainty,
and the short version is that the ranking survives and the number does not.

Data are generated with `A` and inverted with the same `A`. In the inverse
problems literature that is an inverse crime, and it makes the reported errors a
lower bound on what the method would achieve when the operator is unknown.
`edarecover mismatch` measures the gap for an operator wrong by a factor of two,
which is not the same as estimating the operator from the data.

The random streams of the two generators overlap across seeds. `motion(seed=k)`
and `clean_signal(seed=k+1)` open their bit generators in the same state, so the
tremor of one run is the same white noise as the baseline increments of the next.
No reported statistic has been shown to move because of it. The fix is a larger
offset and it is deferred, because changing the streams moves the calibrated
threshold in the third decimal place and 0.156 appears in submitted materials.

No comparison against a published artifact-detection baseline has been run.

## Corrections

Rewriting the analysis code as a tested package surfaced five defects in the
earlier draft of the write-up, three of which changed printed numbers. The
effective-rank table had been computed on a 60-second window with two movement
bursts while everything else used 240-second windows with eight, and it has been
recomputed on the window the rest of the work uses; the threshold in that count
is one per cent of the largest singular value, a relative quantity, and had been
described as an absolute noise level; the response time constants had been
attributed to a paper that does not report them, and the attribution has been
withdrawn; the spectral estimate had been described as twelve non-overlapping
Welch segments when the routine averages twenty-three overlapping ones; and the
seed overlap above had not been documented. The full list, with what each one
changed, is in the Corrections section of the accompanying report.

## References

- Benedek, M., & Kirchner, C. (2010). A continuous measure of phasic
  electrodermal activity. *Journal of Neuroscience Methods*, 190(1), 80-91.
- Hansen, P. C. (1990). The discrete Picard condition for discrete ill-posed
  problems. *BIT Numerical Mathematics*, 30(4), 658-672.
- Hansen, P. C., & O'Leary, D. P. (1993). The use of the L-curve in the
  regularization of discrete ill-posed problems. *SIAM Journal on Scientific
  Computing*, 14(6), 1487-1503.
- Kaipio, J., & Somersalo, E. (2007). Statistical inverse problems:
  discretization, model reduction and inverse crimes. *Journal of Computational
  and Applied Mathematics*, 198(2), 493-504.
- Kleckner, I. R., et al. (2018). Simple, transparent, and flexible automated
  quality assessment procedures for ambulatory electrodermal activity data.
  *IEEE Transactions on Biomedical Engineering*, 65(7), 1460-1467.
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
