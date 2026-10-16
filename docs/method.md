# Method

This document describes, in the order the code executes them, every operation
that `edarecover` performs on a signal, together with the equation behind each
operation and a table naming every variable, its unit, its value in the default
configuration, and where that value came from. The last column of each table is
the one that matters most for anybody auditing this work, because it separates
the quantities taken from published measurements from the quantities that were
chosen by the author of this study. Where a value was chosen rather than
measured, the table says so, and the surrounding text says what the choice was
based on and what was done to find out how much the result depends on it.

The document is written to be read from the top down. Nothing is referred to
before it has been defined, no symbol appears in an equation before the table
that introduces it, and no result is quoted before the procedure that produced
it has been described in full.

---

## 1. The question the code answers

A wrist-worn sensor that records skin conductance also records the consequences
of the wrist moving. Movement changes the mechanical contact between the
electrode and the skin, and it changes it while the physiological signal of
interest is still arriving, so the recorded trace carries both at once. The
established response to this problem is to detect the corrupted stretches and
either repair them or discard them, and a substantial literature does exactly
that, with Kleckner et al. (2018) giving the most widely used rule-based
implementation and Taylor et al. (2015) an automated classifier trained on
labelled artifacts.

What that literature does not state is a limit. If a stretch of recording has
been distorted past some point, no amount of processing recovers the original,
because the information that would be needed to distinguish the true signal
from a plausible imitation of it is no longer present in the data. Repair
applied past that point produces an estimate that looks clean, carries no
warning, and is wrong. This project asks where that point sits, and whether it
can be located using quantities a wearable device is able to measure while it is
recording, which excludes anything that requires knowing the undistorted signal.

The answer this repository produces is a single number, a decision threshold on
one observable quantity, together with the measurements that show how the
threshold behaves when the assumptions behind it are changed. Everything below
describes how that number is produced.

---

## 2. Notation

Every symbol used in this document appears in the table below. Each one is
introduced again, with its value, at the step where it first does something.

| Symbol | Meaning | Unit |
|---|---|---|
| `t` | continuous time from the start of a window | s |
| `n` | sample index, running from 0 | dimensionless |
| `N` | number of samples in a window | dimensionless |
| `f_s` | sampling rate | Hz |
| `T` | length of one analysis window | s |
| `s` | clean skin conductance, the quantity to be recovered | µS |
| `a` | accelerometer trace of wrist movement | arbitrary |
| `e` | smoothed envelope of movement intensity | arbitrary |
| `m` | movement severity, a rescaling of the envelope onto [0, 1] | dimensionless |
| `b` | blur width, the distortion parameter of a run | s |
| `A` | blur operator, an `N × N` matrix | dimensionless |
| `β` | pressure coupling, additive drift per second of blur | µS |
| `ε` | measurement noise | µS |
| `y` | the recorded trace, what the device actually stores | µS |
| `σ_n` | standard deviation of the measurement noise | µS |
| `C_B` | band-limited magnitude-squared coherence, the observable | dimensionless |
| `θ` | decision threshold on `C_B` | dimensionless |
| `L` | second-difference operator, an `(N-2) × N` matrix | dimensionless |
| `λ` | Tikhonov regularisation parameter | dimensionless |
| `τ_1`, `τ_2` | rise and decay time constants of a phasic response | s |
| `S_hat` | estimate of `s` produced by the reconstruction | µS |
| `δ` | error tolerance separating recoverable from unrecoverable | µS |
| `k` | number of singular values above the noise floor | dimensionless |

---

## 3. Why the data are synthetic, and what that costs

The criterion being developed here is a statement about reconstruction error,
and reconstruction error is the distance between an estimate and the true clean
signal. On a human recording the true clean signal is not available, which is
the whole reason motion artifacts are a problem in the first place. There is no
way to calibrate a threshold on error without a dataset in which the error can
be computed, and the only datasets with that property are ones where the clean
signal was constructed before being distorted.

So the calibration runs on simulated recordings, and the price of that decision
belongs here rather than in a limitations paragraph at the end. A threshold
calibrated on simulation is a threshold calibrated on the author's model of how
movement corrupts conductance, and if that model is wrong in a way that matters,
the threshold inherits the error. Section 13 lists the five measurements that
were run specifically to find out how much the result depends on each
contestable part of the model, and section 14 states plainly which claims the
simulation supports and which it does not.

The signal generator lives in `src/edarecover/synthetic.py`. It is ordinary
NumPy arithmetic, roughly 130 lines including the documentation, and it uses no
recorded data of any kind. Sections 4 and 5 describe what it does, line for
line.

---

## 4. Step one: building the clean conductance signal

### 4.1 The time axis

Every run begins by constructing a time axis, because every subsequent quantity
is a function evaluated on it. The axis holds `N` evenly spaced samples,

```
N   = round(T * f_s)
t_n = n / f_s          for n = 0, 1, ..., N - 1
```

and with the default window length of 240 s at 16 Hz this gives `N` = 3840
samples running from 0 s to 239.9375 s. Nothing physiological happens at a
particular sample index, and the index exists only because the operators built
later in the pipeline are matrices, and matrices need positions.

| Symbol | Meaning | Unit | Default | Source |
|---|---|---|---|---|
| `T` | window length | s | 240 | chosen for this study, see below |
| `f_s` | sampling rate | Hz | 16 | chosen to sit between the Empatica E4 conductance and accelerometer rates |
| `N` | samples per window | dimensionless | 3840 | derived, `round(T * f_s)` |

The window length was chosen to be long enough that a Welch spectral estimate
over the band of interest averages over more than twenty segments, which section
8 works through arithmetically, and short enough that the tonic conductance level
does not wander far inside one window. Four minutes satisfies both conditions.
The sampling rate relates to the Empatica E4, which samples conductance at 4 Hz
and acceleration at 32 Hz; 16 Hz sits between the two, so the simulation can be
decimated to either rate without being regenerated, and section 13.4 reports
what happens to the result under exactly that decimation.

### 4.2 The tonic level

Skin conductance consists of a slowly varying baseline with faster responses
riding on top of it, a decomposition that is standard in the field and underlies
every deconvolution method in use, including the continuous decomposition
analysis of Benedek and Kirchner (2010) and the convex optimisation approach of
Greco et al. (2016). The generator builds the two parts separately and adds
them.

The baseline, written `s_tonic`, is the sum of a constant level, one slow
sinusoid, and a random walk:

```
s_tonic(t_n) = g_0 + g_1 * sin(2π t_n / (1.3 T)) + g_2 * (1/f_s) * sum[k = 0..n] z_k
```

where the `z_k` are independent standard normal draws. The period of the
sinusoid is 1.3 times the window length rather than some round number of
seconds, and that is deliberate. A period longer than the window means the
sinusoid never completes a cycle inside a run, so the baseline drifts in one
direction over the four minutes instead of returning to where it started, which
is what real tonic conductance does over comparable spans. The random walk adds
the irregularity that a pure sinusoid lacks, and it is scaled by `1/f_s` so that
its variance accumulates per second rather than per sample, which keeps the
baseline's behaviour unchanged if the sampling rate is changed.

| Symbol | Meaning | Unit | Default | Source |
|---|---|---|---|---|
| `g_0` | mean tonic level | µS | 3.0 | chosen; inside the range commonly reported for resting recordings |
| `g_1` | amplitude of the slow sinusoid | µS | 0.5 | chosen for this study |
| `g_2` | random-walk scale | µS | 0.05 | chosen for this study |
| `z_k` | standard normal increments | dimensionless | drawn | `rng.standard_normal(N)` |

These three constants are the author's choices and are not taken from any
published table. What they control is how much slow variation the reconstruction
has to preserve, and every measurement reported below was run at these values
only, so a reader who wants to know how the result behaves under a quieter or a
noisier baseline would have to rerun the sweep with them changed. That is a
one-line edit in `synthetic.py`, and it is listed in section 14 among the work
that has not been done.

### 4.3 The phasic responses

Each individual skin conductance response is generated from the two-exponential
Bateman form, which rises on one time constant and decays on a slower one:

```
s_scr(t; t_0, α) = α * [ exp(-(t - t_0)/τ_2) - exp(-(t - t_0)/τ_1) ]    for t > t_0
s_scr(t; t_0, α) = 0                                                otherwise
```

and the result is clipped at zero from below, so no response ever contributes a
negative conductance. The functional form is standard and is the impulse
response assumed by the decomposition methods cited above. The specific time
constants are the author's choice, and the reasoning runs as follows. With `τ_1` =
0.75 s and `τ_2` = 2.0 s the response reaches its maximum at

```
t_peak = (τ_1 τ_2 / (τ_2 - τ_1)) * ln(τ_2 / τ_1) = 1.18 s
```

and it falls back below one per cent of that maximum 11.3 s after onset, which
places the simulated response inside the 1–3 s rise and 2–10 s recovery that the
applied literature treats as typical. The values were selected to land in that
window and were not copied from a measurement in any specific paper. An earlier
draft of this work attributed them to Alexander et al. (2005), and that
attribution was wrong; it has been removed rather than replaced, because the
author has not found a published source that reports these two numbers together.

One consequence of the Bateman form deserves stating, because it is easy to
misread the code. The parameter `α` is a scale factor and not the peak height.
Substituting `t_peak` back into the equation gives a peak of `0.347 α`, so the
amplitude range the generator draws from corresponds to peaks between 0.017 µS
and 0.208 µS.

| Symbol | Meaning | Unit | Default | Source |
|---|---|---|---|---|
| `τ_1` | rise time constant | s | 0.75 | chosen; places the peak at 1.18 s, inside the reported 1–3 s |
| `τ_2` | decay time constant | s | 2.0 | chosen; places the 1 % return at 11.3 s |
| `t_0` | onset time of one response | s | drawn `U(0, T)` | chosen distribution |
| `α` | amplitude scale factor | µS | drawn `U(0.05, 0.6)` | chosen; gives peaks of 0.017–0.208 µS |
| `λ_scr` | mean response rate | per minute | 4.0 | chosen; inside the 1–10 per minute reported for resting adults |
| `N_scr` | number of responses in a window | dimensionless | drawn `Poisson(λ_scr T / 60)` | chosen distribution |

The number of responses in a window is a Poisson draw with mean `λ_scr T / 60`,
which is 16 for the default settings, and the realised counts for the six seeds
used in calibration are 12, 15, 25, 7, 23 and 16. The spread is wide because a
Poisson variable with mean 16 has a standard deviation of 4, and it is worth
knowing about, because the seed that drew 7 responses is a considerably easier
reconstruction problem than the seed that drew 25. Section 12 reports the result
separately for each seed, partly for that reason.

The clean signal is then the sum of the baseline and all the responses,

```
s(t_n) = s_tonic(t_n) + sum[j = 1..N_scr] s_scr(t_n; t_0j, α_j)
```

and this array, in microsiemens, is the ground truth against which every error
in this study is measured.

### 4.4 What a seed is, concretely

The word "seed" appears throughout the results, and it means one specific thing.
`clean_signal` opens a NumPy `default_rng(seed)`, which is a PCG64 bit
generator, and draws from it in a fixed order: first `N` standard normal values
for the random walk, then one Poisson value for the response count, then `N_scr`
uniform values for the onsets, then `N_scr` uniform values for the amplitudes.
Because the order is fixed and the generator is deterministic, a given integer
seed always produces the identical array of 3840 floating-point numbers, on any
machine and under any supported NumPy version.

A run of this study is therefore identified by two integers and one real number:
the seed, which fixes both the clean signal and the movement trace, and the blur
width `b`, which fixes how badly that pair is corrupted. The calibration sweep
uses six seeds and eleven blur widths, giving 66 runs in total, and the six seeds
are simply 0 through 5.

### 4.5 A known defect in the seed arrangement

The movement generator described in the next section opens
`default_rng(seed + 1)` rather than `default_rng(seed)`, the intention being that
the movement of a run should be statistically independent of its conductance.
The arrangement does achieve that within a run. Across runs it produces an
overlap that was not intended: the first `N` normal draws taken by
`motion(seed = k)` are the same `N` numbers as the first `N` normal draws taken
by `clean_signal(seed = k + 1)`, because both generators are initialised to the
same state. This has been verified directly, and the two sequences agree to
every digit.

The practical effect is that the tremor component of run `k` and the random-walk
increments of run `k + 1` are the same realisation of white noise, for five of
the six seeds. Both quantities are small (`σ_tr` = 0.02 in arbitrary
accelerometer units, `g_2` = 0.05 µS per unit time), they enter the pipeline
through different channels, and no reported statistic has been shown to move
because of the overlap. It remains a defect, and it appears here in the main text
rather than in a footnote.

The correction is to offset the movement generator by a large constant instead of
by one, and it has not been applied, for a reason worth being open about.
Changing the random streams moves the calibrated threshold in the third decimal
place, and the threshold value of 0.156 appears in materials that have already
been submitted for assessment. The fix is scheduled for the version that follows
that assessment, and when it is applied the threshold will be recalibrated and
both values reported side by side.

---

## 5. Step two: building the wrist movement

Movement is generated as a one-dimensional acceleration trace, with a continuous
low-amplitude tremor present throughout and a small number of louder bursts
standing in for episodes of walking or gesturing:

```
a(t_n) = σ_tr * w_n + sum[j = 1..J] q_j * exp( -0.5 ((t_n - c_j) / (w_j/2))^2 ) * sin(2π f_j t_n)
```

where the `w_n` are independent standard normal draws. Each burst is a sinusoid
at its own frequency multiplied by a Gaussian window that switches it on and off
smoothly, so the trace has stretches of quiet and stretches of activity rather
than uniform agitation. The burst frequency range of 1–3 Hz is the one movement
of this kind actually occupies, with normal walking cadence sitting near 2 Hz,
and that range is the single part of the movement model with external support.
The number of bursts, their widths and their amplitudes are all the author's
choices.

| Symbol | Meaning | Unit | Default | Source |
|---|---|---|---|---|
| `σ_tr` | tremor standard deviation | arbitrary | 0.02 | chosen for this study |
| `J` | number of bursts per window | dimensionless | 8 | chosen for this study |
| `c_j` | burst centre time | s | drawn `U(0.05 T, 0.95 T)` | chosen; keeps bursts away from the window edges |
| `w_j` | burst duration | s | drawn `U(3, 15)` | chosen for this study |
| `f_j` | burst frequency | Hz | drawn `U(1, 3)` | walking and gesture cadence |
| `q_j` | burst amplitude | arbitrary | drawn `U(0.5, 2.5)` | chosen for this study |
| `w_n` | tremor increments | dimensionless | drawn | `rng.standard_normal(N)` |

The units of this trace are arbitrary, and that is a real limitation rather than
a notational convenience. Nothing in the generator is calibrated against g or
against any physical accelerometer output, so the amplitude scale of `a` carries
no meaning on its own. It acquires meaning only in combination with the coupling
constant `β` introduced in section 7, which converts movement intensity into
microsiemens of spurious conductance, and the two quantities cannot be separated
from inside this simulation. A measurement that fixed the scale, for instance
recording conductance while loading the wrist with known weights, would
determine `β` and make the combination identifiable. That measurement has not
been performed, and section 13.1 reports a sweep across `β` in its place.

---

## 6. Step three: two smoothed versions of the movement

The raw acceleration trace is not used directly anywhere in the pipeline.
Movement oscillates between 1 and 3 Hz, far above the band in which
electrodermal activity carries sympathetic information, and skin does not follow
an oscillation at that rate. What the skin responds to is the intensity of the
movement, which is to say the envelope of the oscillation, and the envelope
varies slowly enough to fall inside the band where conductance lives. Two
different smoothings of `|a|` are therefore computed, for two different purposes,
and the distinction between them is a frequent source of confusion when reading
the code.

The first is the movement envelope, a moving average of the absolute
acceleration over a window of `W_e` samples:

```
e(t_n) = (1/W_e) * sum[i] |a(t_(n-i))|          W_e = round(T_e * f_s)
```

implemented as `np.convolve(np.abs(acc), np.ones(W_e)/W_e, mode="same")`. This
envelope is the quantity the coherence of section 8 is computed against, and the
quantity the regression of section 10 removes. Its smoothing length of 2.0 s
corresponds to a cutoff low enough to leave the envelope inside the 0.045–0.25 Hz
band that section 8 defines.

The second is the movement severity, a shorter moving average rescaled onto the
unit interval:

```
m_raw(t_n) = (1/W_m) * sum[i] |a(t_(n-i))|      W_m = max(1, floor(T_m * f_s))
m(t_n)     = (m_raw(t_n) - min m_raw) / (max m_raw - min m_raw)
```

and if the denominator falls below 1e-12 the function returns zeros instead of
dividing, which is what happens for a trace with no variation at all. Severity
exists only to set the local width of the blur operator in the next section,
where what is needed is a normalised measure of where the movement is strongest,
and the shorter smoothing of 0.5 s is used so that the operator's width tracks
individual bursts rather than being flattened across them.

| Symbol | Meaning | Unit | Default | Source |
|---|---|---|---|---|
| `T_e` | envelope smoothing length | s | 2.0 | chosen; keeps the envelope inside the analysis band |
| `W_e` | envelope smoothing length in samples | dimensionless | 32 | derived, `round(T_e f_s)` |
| `T_m` | severity smoothing length | s | 0.5 | chosen; tracks individual bursts |
| `W_m` | severity smoothing length in samples | dimensionless | 8 | derived, `floor(T_m f_s)` |
| `e` | movement envelope | arbitrary | computed | used by sections 8 and 10 |
| `m` | movement severity | dimensionless, [0, 1] | computed | used by section 7 |

---

## 7. Step four: damaging the recording

This is the step that turns a clean signal into a recording, and it is the step
where the model of what movement does to conductance is stated. Three things
happen to the signal, and the recording is their sum:

```
y = A s + b β e + ε
```

Each of the three terms is described below, in the order it appears.

### 7.1 The blur operator

Pressing an electrode against the skin spreads the measured conductance in time,
because the contact area and the moisture distribution under the electrode take
time to respond, and because the value recorded at any instant reflects a
neighbourhood of the true one. The model represents this as a local Gaussian
average whose width grows where the movement is strongest. Row `i` of the
operator holds a Gaussian kernel centred on sample `i`:

```
A_raw_ij = exp( -0.5 ((j - i) / σ_i)^2 )         for |j - i| <= H, and 0 otherwise
A_ij = A_raw_ij / sum[j'] A_raw_ij'
```

with the local width set by the severity computed in the previous section,

```
σ_i   = σ_min + m(t_i) * (σ_max - σ_min)
σ_max = max(σ_min, b * f_s)
H     = ceil(3 σ_max)
```

The row normalisation in the second line is what makes this operator a blur
rather than an attenuation. Each row sums to one, so a constant signal passes
through unchanged, which means the operator leaves the tonic conductance level
alone and acts on temporal detail only. Checking that property is one of the
unit tests in `tests/test_forward.py`.

| Symbol | Meaning | Unit | Default | Source |
|---|---|---|---|---|
| `b` | blur width, the distortion parameter | s | swept, see below | chosen sweep range |
| `σ_min` | kernel width where movement is absent | samples | 0.5 | chosen; the smallest width that is not a delta |
| `σ_max` | kernel width where movement is strongest | samples | `max(0.5, b f_s)` | derived |
| `H` | kernel half-width | samples | `ceil(3 σ_max)` | chosen; a 3σ truncation retains 99.7 % of the kernel |
| `A` | blur operator | dimensionless | `N × N` sparse | computed |

The operator is held as a `scipy.sparse.csr_matrix` and not as a dense array,
and the reason is arithmetic rather than taste. At `N` = 3840 a dense
double-precision matrix occupies 0.118 GB, while the sparse form occupies
between 0.25 MB at `b` = 0 and 25.6 MB at `b` = 6 s, because the number of
non-zero entries per row is bounded by `2H + 1`. The measured occupancies are 5.0
non-zeros per row at `b` = 0, 68.7 at `b` = 0.7 s, 190.6 at `b` = 2 s, and 555.3
at `b` = 6 s, each count falling a little below `2H + 1` because rows near the
start and the end of the window have their kernels truncated by the boundary.

The sweep runs eleven blur widths, `b` ∈ {0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0,
3.0, 4.0, 6.0} s. The spacing is deliberately uneven and the lower half is
deliberately dense, because section 12 will show that the transition between
recoverable and unrecoverable happens between 0.4 s and 0.7 s, and a uniform grid
would have placed at most one point inside it.

### 7.2 The additive pressure term

Alongside spreading the signal in time, pressure on the electrode adds
conductance of its own, and the model represents that as a term proportional to
both the movement envelope and the blur width:

```
y_pressure(t_n) = b * β * e(t_n)
```

This term is the reason the distortion is detectable at all. The blur operator
on its own leaves a trace that is smoother than the original, and smoothness is
not something a detector can measure without knowing what the original looked
like. The additive term, by contrast, writes a copy of the movement envelope into
the conductance channel, and a copy of a known signal is exactly what the
coherence of section 8 is built to find.

| Symbol | Meaning | Unit | Default | Source |
|---|---|---|---|---|
| `β` | pressure coupling per second of blur | µS per unit of envelope | 0.3 | **assumed, not measured** |

The value of `β` is the weakest assumption in this study and it is marked as
such. It was not measured, it was not taken from a publication, and the author is
not aware of a published measurement of this coupling for a wrist-worn electrode.
It was chosen so that the additive term at moderate blur is comparable in size to
a phasic response, which makes the detection problem neither trivial nor
impossible, and choosing a constant to make a problem interesting is not evidence
about the world. Section 13.1 therefore refits the entire criterion at seven
values of `β` spanning zero to 0.8 and reports what happens to the threshold,
and that sweep is the only honest defence available until the coupling is
measured on a bench.

### 7.3 Measurement noise

The third term is independent Gaussian noise added to every sample,

```
ε_n ~ N(0, σ_n^2)
```

with `σ_n` = 0.01 µS. The value stands for quantisation and amplifier noise in a
consumer-grade sensor, and it is the author's estimate rather than a
specification figure. It matters in two places beyond adding scatter. It sets the
floor below which the singular values of the blur operator carry no usable
information, which is what section 9 counts, and it is one eighth of the error
tolerance defined in section 11, so a run whose only error source is noise is
classified as recoverable with room to spare.

The noise array is drawn by the caller and passed into the forward model rather
than being drawn inside it, which is a reproducibility decision. The calibration
sweep creates one generator at the start and advances it across all 66 runs, with
blur width as the outer loop and seed as the inner one, so the sequence of noise
realisations is a property of the whole sweep rather than of any single run.
Changing that loop order changes which realisation each run receives and moves
the results in the fourth decimal place, which is why the order is documented
here and pinned by a regression test.

| Symbol | Meaning | Unit | Default | Source |
|---|---|---|---|---|
| `σ_n` | measurement noise standard deviation | µS | 0.01 | chosen; the author's estimate of consumer sensor noise |
| `ε` | noise realisation | µS | drawn | `σ_n * rng.standard_normal(N)` |

---

## 8. Step five: the quantity a device can actually measure

Everything up to this point has been construction. This section introduces the
one quantity the criterion is allowed to use, and the constraint it has to
satisfy is that it must be computable inside a wearable device during recording,
which rules out anything requiring the clean signal or the blur operator.

What a device does have is the conductance channel and the accelerometer channel.
The additive term of section 7.2 put a copy of the movement envelope into the
conductance channel, so the question becomes how much of the recorded conductance
is explained, frequency by frequency, by the movement envelope. That is
magnitude-squared coherence:

```
C(f) = |P_ye(f)|^2 / ( P_yy(f) * P_ee(f) )
```

where `P_yy` and `P_ee` are the power spectral densities of the recording and the
envelope and `P_ye` is their cross-spectral density, all three estimated by
Welch's method. Coherence runs from zero to one and is dimensionless, so it does
not care about the arbitrary units of the accelerometer trace, which is a useful
property given section 5.

The estimate is then averaged over a band rather than used frequency by
frequency:

```
C_B = (1/|F_B|) * sum[f in F_B] C(f),        F_B = { f : 0.045 Hz <= f <= 0.25 Hz }
```

The band is not an arbitrary choice. Posada-Quintero et al. (2016) identified
0.045–0.25 Hz as the frequency range in which electrodermal activity carries
sympathetic information, and that band has been used in subsequent work on
spectral indices of electrodermal activity, including the systematic review by
Posada-Quintero and Chon (2020). Restricting the average to that band means the
criterion is asking whether movement has contaminated the part of the spectrum
the measurement is actually for.

The arithmetic of the estimate is worth setting out explicitly, because the
numbers are small enough to matter. With a Welch segment length of 20 s at 16 Hz
the segment holds 320 samples, and scipy's defaults apply a Hann window with 50
per cent overlap and remove the mean from each segment. The frequency resolution
is therefore

```
Δf = f_s / N_seg = 16 / 320 = 0.05 Hz
```

and a 3840-sample window yields 23 overlapping segments to average. The band runs
from 0.045 Hz to 0.25 Hz, the available bins sit at integer multiples of 0.05 Hz,
and so the average is taken over exactly five bins, at 0.05, 0.10, 0.15, 0.20 and
0.25 Hz. The lower band edge of 0.045 Hz falls between bins, and the first bin
inside the band is the one at 0.05 Hz. Five bins is coarse, and it appears in
section 14 among the things a longer window would improve.

| Symbol | Meaning | Unit | Default | Source |
|---|---|---|---|---|
| `T_seg` | Welch segment length | s | 20 | chosen; gives 23 segments per window |
| `N_seg` | Welch segment length in samples | dimensionless | 320 | derived, `round(T_seg f_s)` |
| `Δf` | frequency resolution | Hz | 0.05 | derived, `f_s / N_seg` |
| `F_B` | analysis band | Hz | 0.045–0.25 | Posada-Quintero et al. (2016) |
| `\|F_B\|` | number of bins averaged | dimensionless | 5 | derived |
| `C_B` | band coherence, the observable | dimensionless, [0, 1] | computed | the quantity the threshold is set on |

### 8.1 The noise floor of the estimate

A coherence estimate computed from a finite number of averaged segments does not
return zero for unrelated inputs, and any threshold placed on coherence has to
sit above whatever value unrelated inputs produce. The floor was measured rather
than assumed, by computing `C_B` for 200 independent pairs of white noise under
the settings above. The result is a mean of 0.0486 with a sample standard
deviation of 0.0239.

The threshold that section 11 arrives at is 0.156, which sits 4.49 standard
deviations above that floor. That margin is the reason the criterion is not
simply detecting the bias of its own estimator.

---

## 9. Step six: how much information the blur removes

Before asking whether a reconstruction succeeds, it is worth asking how much of
the signal survives the blur at all, because that question has an answer
independent of any particular reconstruction method. The answer comes from the
singular values of the operator. A blur operator has singular values that decay,
and once a singular value falls below the noise level, the component of the
signal along the corresponding direction has been overwritten by noise and cannot
be recovered by any method whatsoever. Counting the survivors is the discrete
Picard condition of Hansen (1990):

```
k(b) = count{ i : σ_i(A) > 0.01 * σ_1(A) }
```

where the `σ_i(A)` are the singular values in descending order. The threshold is
one per cent of the largest singular value, which is a relative quantity; it
happens to coincide numerically with `σ_n` = 0.01 µS, and the coincidence is
arithmetic rather than physical, so the two should not be read as the same
thing. The count is reported as a fraction of the full rank, which is the
proportion of signal directions that remain informative.

| `b` (s) | `k` | `k / N` |
|---|---|---|
| 0.0 | 3840 | 1.000 |
| 0.1 | 3717 | 0.968 |
| 0.2 | 3298 | 0.859 |
| 0.4 | 2874 | 0.748 |
| 0.7 | 2563 | 0.667 |
| 1.0 | 2392 | 0.623 |
| 1.5 | 2211 | 0.576 |
| 2.0 | 2100 | 0.547 |
| 3.0 | 1933 | 0.503 |
| 4.0 | 1784 | 0.465 |
| 6.0 | 1492 | 0.389 |

These counts are computed on the full 3840-sample analysis window with the
default eight movement bursts, which is the configuration every other number in
this document uses. The window length matters more than it looks: the same sweep
on a shortened 1200-sample trace gives 0.136 at `b` = 6 s rather than 0.389,
because a shorter window holds fewer small singular values to begin with. A
table of this kind is therefore only comparable against others computed on the
same window and the same movement model, and an earlier version of this
measurement was not. That version used a 60-second window with two bursts
instead of eight, which leaves 47.6 per cent of the samples at a severity below
0.05 so that nearly half the rows of `A` stay close to the identity, against 3.1
per cent with eight bursts; the mean severity is almost the same either way,
0.265 against 0.272, so what changes is the distribution of movement in time
rather than its amount.

Something important follows from this table when it is read against the next
one, and it belongs here rather than being left for the reader to notice. The
collapse of the effective rank is not what drives the reconstruction error in
these runs. Switching the additive pressure term off entirely, by setting `β` =
0 and leaving everything else alone, gives the following median uncorrected
error:

| `b` (s) | 0.0 | 0.1 | 0.2 | 0.4 | 0.7 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 6.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RMSE at `β` = 0 (µS) | 0.0100 | 0.0101 | 0.0101 | 0.0102 | 0.0105 | 0.0109 | 0.0117 | 0.0126 | 0.0144 | 0.0162 | 0.0190 |

The error roughly doubles across a range over which the effective rank falls by
a factor of 2.6, and at its worst it reaches 0.0190 µS, which is less than a
quarter of the 0.08 µS tolerance that defines the label. Under this model the
blur destroys a great deal of information, and almost none of the measurable
error comes from that destruction; the error is produced by the additive term
instead. The reason is that the blur operator removes detail at frequencies well
above the band the error is dominated by, and the slow components of the signal,
which carry most of its variance, pass through a row-normalised Gaussian average
almost untouched. That finding is uncomfortable for the framing of this work as
an inverse problem, and section 14 says so in those words.

---

## 10. Step seven: the two-stage reconstruction

The reconstruction exists so that the error after correction can be measured and
compared against the error before it. It runs in two stages, and the order is not
interchangeable.

### 10.1 Stage one, removing the additive trace

The first stage removes the component of the recording that is locked to the
movement envelope, by ordinary least squares on a two-column design:

```
X     = [ e , 1 ]
c_hat = argmin_c || X c - y ||^2
y_1   = y - X c_hat + c_hat_0 = y - c_hat_1 e
```

The intercept is subtracted and then added back, which is what the third line
records: the operation reduces to removing `c_hat_1 e` and leaving the tonic level in
place. Removing the intercept as well would strip the mean conductance, and the
mean conductance is part of the signal.

| Symbol | Meaning | Unit | Source |
|---|---|---|---|
| `X` | design matrix, `N × 2` | mixed | the envelope and a constant column |
| `c_hat_1` | fitted envelope coefficient | µS per unit of envelope | `np.linalg.lstsq` |
| `c_hat_0` | fitted intercept | µS | `np.linalg.lstsq` |
| `y_1` | recording with the additive trace removed | µS | computed |

### 10.2 Stage two, inverting the blur

The second stage inverts the blur operator by Tikhonov regularisation with a
second-difference penalty,

```
S_hat(λ) = argmin_s ( || A s - y_1 ||^2 + λ^2 || L s ||^2 )
```

solved through the regularised normal equations

```
( A^T A + λ^2 L^T L ) S_hat = A^T y_1
```

with `scipy.sparse.linalg.spsolve`. The penalty operator `L` is the second
difference, an `(N-2) × N` sparse matrix whose rows hold the stencil (1, -2, 1),
so `|| L s ||^2` is the summed squared curvature of the estimate, and the penalty
suppresses the high-frequency oscillation that an unregularised inversion of a
blur operator produces.

The regularisation parameter is selected by the L-curve criterion of Hansen and
O'Leary (1993). The residual norm `|| A S_hat(λ) - y_1 ||` and the solution roughness
`|| L S_hat(λ) ||` are computed at each `λ` on a logarithmic grid, both are taken in
logarithmic coordinates, and the chosen `λ` is the one at the point of maximum
curvature of that trajectory, found from finite differences:

```
κ(λ) = | r' s'' - s' r'' | / ( r'^2 + s'^2 )^(3/2)
```

where `r = ln || A S_hat - y_1 ||` and `s = ln || L S_hat ||`, with the derivatives taken
by `np.gradient` over the grid.

| Symbol | Meaning | Unit | Default | Source |
|---|---|---|---|---|
| `L` | second-difference penalty | dimensionless | `(N-2) × N` | standard smoothness penalty |
| `λ` | regularisation parameter | dimensionless | selected per run | L-curve, Hansen and O'Leary (1993) |
| grid | candidate `λ` values | dimensionless | `logspace(-3, 2, 7)` | chosen for this study |
| `S_hat` | reconstruction | µS | computed | the estimate the error is measured on |

Seven grid points is coarse enough to be a limitation in its own right, because
the corner of the L-curve can only ever be located at one of seven positions, and
a finer grid would both find a better `λ` and cost proportionally more time. The
grid size is a configuration field and can be raised without touching any other
code.

### 10.3 Why this order

Running a single regularised inversion on the raw recording, with the additive
trace still present, assigns that trace to the signal. The additive trace is a
smoothed envelope and is therefore smooth, the penalty term charges almost
nothing for smooth content, and so the inversion has no reason to reject it. The
regression has to come first for the inversion to be inverting the right thing.
Section 13.3 measures what each stage contributes on its own, and the result
there is not the one this ordering argument would lead a reader to expect.

### 10.4 How error is measured

Error is root mean square difference from the known clean signal, computed twice
per run, once on the recording and once on the reconstruction:

```
RMSE_before = sqrt( (1/N) sum[n] ( y(t_n) - s(t_n) )^2 )
RMSE_after  = sqrt( (1/N) sum[n] ( S_hat(t_n) - s(t_n) )^2 )
```

Both are in microsiemens. `RMSE_before` is the quantity the labels in the next
section are defined on, and `RMSE_after` is the quantity that says whether
correction helped.

---

## 11. Step eight: the label and the threshold

### 11.1 Defining the label

A run is labelled as needing correction when its uncorrected error exceeds a
tolerance:

```
label = 1   if RMSE_before > δ
label = 0   otherwise
```

with `δ` = 0.08 µS. The tolerance is the author's choice, and the reasoning is
that it sits eight times above the measurement noise standard deviation, so noise
alone never triggers it, and below the peak amplitude of a mid-sized phasic
response, so an error large enough to distort a response of interest does trigger
it. There is no published convention for this number that the author is aware of,
and a different tolerance would move the threshold. What a different tolerance
would not change is the shape of the relationship between coherence and error,
which the next subsection reports.

| Symbol | Meaning | Unit | Default | Source |
|---|---|---|---|---|
| `δ` | error tolerance | µS | 0.08 | chosen; 8 σ_n, and below a mid-sized response peak |

Under this tolerance the 66 calibration runs split into 41 labelled as needing
correction and 25 labelled as not needing it.

### 11.2 What the sweep produced

The table below is the whole calibration set summarised by blur width. Every
column is a median over the six seeds, apart from the coherence range, which
gives the minimum and the maximum, and the positive count, which is out of six.

| `b` (s) | median `C_B` | `C_B` range | median RMSE before (µS) | median RMSE after (µS) | positives |
|---|---|---|---|---|---|
| 0.0 | 0.0496 | 0.014–0.083 | 0.0100 | 0.0661 | 0 / 6 |
| 0.1 | 0.0626 | 0.022–0.076 | 0.0180 | 0.0676 | 0 / 6 |
| 0.2 | 0.0707 | 0.035–0.132 | 0.0313 | 0.0669 | 0 / 6 |
| 0.4 | 0.0991 | 0.089–0.292 | 0.0603 | 0.0670 | 0 / 6 |
| 0.7 | 0.2186 | 0.156–0.488 | 0.1047 | 0.0672 | 5 / 6 |
| 1.0 | 0.3288 | 0.218–0.630 | 0.1492 | 0.0673 | 6 / 6 |
| 1.5 | 0.4578 | 0.313–0.786 | 0.2238 | 0.0679 | 6 / 6 |
| 2.0 | 0.5426 | 0.398–0.867 | 0.2983 | 0.0686 | 6 / 6 |
| 3.0 | 0.6551 | 0.519–0.936 | 0.4475 | 0.0702 | 6 / 6 |
| 4.0 | 0.7260 | 0.607–0.961 | 0.5965 | 0.0721 | 6 / 6 |
| 6.0 | 0.8177 | 0.719–0.983 | 0.8944 | 0.0769 | 6 / 6 |

Three things in this table should be read before the threshold is fitted to it.
The coherence rises monotonically with blur width in the medians, which is what
makes a single threshold possible at all. The transition from zero positives to
five happens between 0.4 s and 0.7 s, which is the interval the uneven sweep grid
of section 7.1 was designed to resolve. And the median error after correction is
nearly flat across the entire range, at 0.066 to 0.077 µS, which means the
reconstruction returns roughly the same quality of estimate no matter how bad the
input was, and at `b` = 0 it is six times worse than doing nothing. The
reconstruction is useful above about 0.7 s of blur and harmful below it, and that
asymmetry is the practical argument for gating it rather than always running it.

The correlation between coherence and uncorrected error across all 66 runs is
0.853, between coherence and blur width 0.862, and between blur width and
uncorrected error 0.981.

### 11.3 Fitting the threshold

The threshold is the coherence value that best separates the two label classes,
chosen by Youden's index on the receiver operating characteristic curve. The
curve is computed with `sklearn.metrics.roc_curve`, which returns the true and
false positive rates at every candidate cut point, and the selected cut maximises

```
J = sensitivity + specificity - 1 = TPR - FPR
```

A window is then corrected when its coherence reaches the threshold, `C_B >= θ`,
and left alone otherwise.

The fit on all 66 runs gives

| Quantity | Value |
|---|---|
| threshold `θ` | 0.1561 |
| area under the ROC curve | 0.9785 |
| sensitivity at `θ` | 1.000 |
| specificity at `θ` | 0.880 |
| runs labelled positive | 41 |
| runs labelled negative | 25 |

and one feature of this fit should be pointed out here rather than waiting for
somebody else to point it out. The threshold 0.15613 is exactly the smallest
coherence value among the positive runs, the one at `b` = 0.7 s that appears in
the range column of the previous table. Youden's index placed the cut directly on
that single observation, so the threshold is determined by one run out of 66, and
if that run were removed the threshold would move up to the next observation. The
cross-validation in the next section was run partly to find out how much that
fragility matters, and the answer is that the threshold moves by 0.009 across
folds.

---

## 12. Step nine: checking the threshold on data it was not fitted on

A threshold quoted on the data that produced it is not an estimate of anything.
The check used here is leave-one-group-out cross-validation, with the seed as the
grouping variable, implemented with `sklearn.model_selection.LeaveOneGroupOut`.

Grouping by seed rather than splitting run by run is the part of this design that
carries the argument. A seed fixes both the clean signal and the movement
trajectory, so the eleven runs that share a seed differ from each other only in
the blur width, and they are strongly correlated as a result. A random split
would put correlated runs on both sides of the division, and the threshold fitted
on the training side would have already seen most of what makes the test runs
what they are. Grouping by seed keeps all eleven runs of a seed on the same side
of every split, so each fold fits on five seeds and tests on a sixth that
contributed nothing at all to the fit.

With six seeds there are six folds, each holding out eleven runs:

| Held-out seed | Runs | Fitted `θ` | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|
| 0 | 11 | 0.1561 | 1.000 | 1.000 | 1.000 |
| 1 | 11 | 0.1561 | 1.000 | 1.000 | 1.000 |
| 2 | 11 | 0.1561 | 1.000 | 1.000 | 1.000 |
| 3 | 11 | 0.1561 | 1.000 | 0.750 | 0.909 |
| 4 | 11 | 0.1783 | 0.857 | 1.000 | 0.909 |
| 5 | 11 | 0.1561 | 1.000 | 0.600 | 0.818 |

The fitted threshold is stable, with a mean of 0.1598 and a sample standard
deviation of 0.0091 across folds, and five of the six folds landed on the same
value because the run that determines the cut is present in five of the six
training sets. Pooling the predictions, so that every run is judged by the
threshold of the fold that excluded it, gives a sensitivity of 0.976, a
specificity of 0.880 and an accuracy of 0.939.

The honest reading of the per-fold columns is that specificity is where the
variation sits. Seed 5 produced two false positives out of five negatives, and
seed 3 produced one. Sensitivity held at or near one throughout, which means the
criterion rarely misses a window that needs correcting and occasionally flags a
window that did not need it. For a gate whose purpose is to prevent silent
failure that is the preferable direction of error, and it is also the direction a
device paying for unnecessary reconstruction has to budget for.

---

## 13. Step ten: five measurements on the assumptions

Each measurement below exists because a specific part of the model is
contestable, and each one answers the question of how far the conclusion survives
that part being wrong. The first four are implemented in
`src/edarecover/validation.py` and run from the command line through
`edarecover validate`.

### 13.1 Dependence on the assumed pressure coupling

The coupling `β` was assumed rather than measured, as section 7.2 states, so the
whole criterion is refitted at seven values of it. Reconstruction is not needed
for this sweep, because both the coherence and the label follow from the
recording alone, which makes it cheap enough to run across the full grid. The
column headed positives counts how many of the 66 runs exceeded the error
tolerance at that coupling.

| `β` | positives | fitted `θ` | AUC | mean `C_B` at `b` = 0 | mean `C_B` at `b` = 6 s |
|---|---|---|---|---|---|
| 0.00 | 0 / 66 | undefined | undefined | 0.0504 | 0.0310 |
| 0.05 | 12 / 66 | 0.1644 | 0.9614 | 0.0504 | 0.4210 |
| 0.10 | 24 / 66 | 0.1713 | 0.9772 | 0.0504 | 0.6198 |
| 0.20 | 34 / 66 | 0.2032 | 0.9789 | 0.0504 | 0.7708 |
| 0.30 | 41 / 66 | 0.1561 | 0.9785 | 0.0504 | 0.8421 |
| 0.50 | 46 / 66 | 0.2494 | 0.9859 | 0.0504 | 0.9124 |
| 0.80 | 51 / 66 | 0.3231 | 0.9895 | 0.0504 | 0.9552 |

Three separate things have to be read out of this table, and only one of them is
good news.

The discriminative power of the observable is insensitive to the coupling.
Across the whole range from 0.05 to 0.8 the area under the curve stays between
0.961 and 0.990, so coherence separates recoverable from unrecoverable windows
about equally well whatever the coupling turns out to be. That is the claim the
sweep supports.

The numeric value of the threshold is not insensitive at all. It runs 0.164,
0.171, 0.203, 0.156, 0.249, 0.323 as the coupling increases, which is a factor
of two across the range, and it is not even monotone: the default coupling of
0.3 sits at a local dip, because Youden's index landed on a different individual
observation there than it did at the neighbouring values. So the number 0.156 is
specific to `β` = 0.3 and should not be quoted as though it were a constant of
the problem. A device would have to calibrate its own threshold against its own
hardware, which is the honest version of what this criterion offers.

At zero coupling the criterion does not exist. Not one of the 66 runs exceeded
the error tolerance, so there are no positive cases to fit a threshold to, and
the two coherence columns show why the observable would be useless even if there
were: the mean coherence at the strongest blur is 0.031, which is *lower* than
the 0.050 at no blur at all. With no additive term, heavier blur makes the
recording smoother and therefore less like the movement envelope, so coherence
moves in the wrong direction. Everything the criterion detects, it detects
through the additive term.

### 13.2 Dependence on knowing the blur operator

Generating data with an operator and then inverting with that same operator is an
inverse crime in the sense of Kaipio and Somersalo (2007), and an error reported
under those conditions is a lower bound on what the method would achieve in a
setting where the operator has to be estimated. The measurement scales the
assumed blur width by a factor while leaving the generating width alone, so the
inversion is performed with a deliberately wrong operator. A factor of 0.5 means
the inversion assumed half the true blur, and a factor of 2.0 means it assumed
twice as much.

| true `b` (s) | assumed / true | median RMSE after (µS) | change against the correct operator |
|---|---|---|---|
| 1.0 | 0.5 | 0.0671 | -0.4 % |
| 1.0 | 1.0 | 0.0673 | reference |
| 1.0 | 2.0 | 0.0681 | +1.2 % |
| 2.0 | 0.5 | 0.0675 | -1.4 % |
| 2.0 | 1.0 | 0.0685 | reference |
| 2.0 | 2.0 | 0.0721 | +5.2 % |
| 4.0 | 0.5 | 0.0690 | -4.4 % |
| 4.0 | 1.0 | 0.0722 | reference |
| 4.0 | 2.0 | 0.0869 | +20.4 % |

The error is almost unaffected by getting the operator wrong. Halving the
assumed blur width improves the result slightly at every level tested, and
doubling it costs between one and twenty per cent depending on how much blur
there is to begin with. A method that depended on the operator would not behave
like this, and that is the right way to read the table. The insensitivity is a
symptom that the inversion is contributing very little, which the next
subsection measures directly, and reading it as tolerance to operator error
would get the explanation backwards. The
inverse crime was worth checking for, and what the check found is that this
particular inverse problem is not being solved by the inverse step.

### 13.3 What each reconstruction stage contributes

Section 10.3 argued that the stages have to run in a particular order. The
sharper question is whether the second stage earns its place at all, and only a
comparison against each stage alone answers it. Each column is a median over the
six seeds.

| `b` (s) | no correction | stage 1 only | stage 2 only | both stages |
|---|---|---|---|---|
| 0.0 | 0.0100 | 0.0665 | 0.0077 | 0.0661 |
| 0.1 | 0.0180 | 0.0665 | 0.0177 | 0.0676 |
| 0.2 | 0.0313 | 0.0666 | 0.0316 | 0.0669 |
| 0.4 | 0.0603 | 0.0666 | 0.0604 | 0.0670 |
| 0.7 | 0.1047 | 0.0666 | 0.1048 | 0.0672 |
| 1.0 | 0.1492 | 0.0666 | 0.1497 | 0.0673 |
| 1.5 | 0.2238 | 0.0670 | 0.2258 | 0.0679 |
| 2.0 | 0.2983 | 0.0671 | 0.3035 | 0.0686 |
| 3.0 | 0.4475 | 0.0675 | 0.4667 | 0.0702 |
| 4.0 | 0.5965 | 0.0678 | 0.6435 | 0.0721 |
| 6.0 | 0.8944 | 0.0685 | 1.0409 | 0.0769 |

All figures are median root mean square error in microsiemens over the six
seeds. This is the least flattering table in the document and it is the one that
should be read most carefully.

The second stage on its own does essentially nothing up to about 0.4 s of blur,
where its column tracks the uncorrected column to three decimal places, and
above that it makes the recording worse. At `b` = 6 s the inversion alone
returns 1.0409 µS against the 0.8944 µS of doing nothing, so running it has
increased the error by 16 per cent. The reason is visible in the structure of
the problem rather than in any bug: the additive term is still present when the
inversion runs in this variant, the term is smooth, the second-difference
penalty charges almost nothing for smooth content, and so the inversion fits the
contamination into the estimate and amplifies it. The ordering argument of
section 10.3 predicted exactly this, and this column is the measurement that
confirms it.

The first stage on its own is flat at 0.066 to 0.069 µS across the entire blur
range. Comparing it against the full procedure, the two stages together are
better at `b` = 0 by 0.0003 µS and worse at every other blur level, by between
one and twelve per cent, with the gap widening as the blur grows. The second
stage therefore does not earn its place. The reconstruction reported everywhere
else in this document is the full two-stage procedure, because that is what was
specified in advance and changing it after seeing this table would be fitting
the method to the result, and the correct conclusion to draw from the table is
that a future version should either drop the inversion or replace the
second-difference penalty with one that does not treat the additive trace as
free.

One number in this table matters for the decision rule rather than for the
reconstruction. At `b` = 0 the first stage turns an error of 0.0100 µS into one
of 0.0665 µS, which is a factor of 6.7 worse, because regressing a clean
recording on the movement envelope removes real signal that happens to
correlate with movement by chance. Correction applied to an undamaged recording
does measurable harm, and that is the whole case for putting a gate in front of
it.

### 13.4 Dependence on the sampling rate

A wrist device that records conductance at 4 Hz would have to compute the
coherence at that rate, so the question is whether the observable survives
decimation by a factor of four.

The measurement decimates one realisation rather than regenerating the signals at
the lower rate, and that distinction took one wrong attempt to get right. The
first version of this check regenerated the signals at 4 Hz and compared the
coherences, and the comparison was meaningless: the movement generator puts its
energy between 1 and 3 Hz, a 4 Hz sampling rate has a Nyquist frequency of 2 Hz,
and most of that energy aliases. The two sets of numbers were describing
different processes. The corrected version decimates a single realisation with
`scipy.signal.decimate`, using an anti-aliasing IIR filter in zero-phase mode,
which isolates the effect of sampling and also mirrors what a real device does,
since the accelerometer channel is anti-alias filtered before being brought down
to the conductance rate.

| Quantity | At 16 Hz | At 4 Hz after decimation |
|---|---|---|
| fitted threshold `θ` | 0.15613 | 0.15586 |
| area under the ROC curve | 0.9785 | 0.9785 |
| largest absolute difference in `C_B` over 66 runs | | 0.00053 |
| Pearson correlation between the two sets of 66 values | | 0.9999999 |

The observable survives decimation to the rate a real conductance channel runs
at. The threshold moves by 0.00027, which is a twentieth of the 0.009 spread
seen across cross-validation folds, the area under the curve is identical to
four decimal places, and the largest disagreement between any pair of
corresponding coherence values is five parts in ten thousand. This is the
cleanest of the four results, and the explanation is straightforward: the whole
analysis band from 0.045 to 0.25 Hz sits an order of magnitude below the 2 Hz
Nyquist frequency of the decimated series, so nothing the criterion looks at is
near the edge where decimation does damage.

### 13.5 What the gate costs to run

The criterion is proposed as a gate in front of reconstruction, so the ratio of
the two costs decides whether the proposal makes sense at all. Measured on the
development machine, with a 240 s window at `b` = 2 s, a coherence evaluation
takes 2.6 ms and a full two-stage reconstruction takes 4045 ms, a ratio of
roughly 1550. Absolute timings are hardware-dependent and will differ on other
machines; the ratio of three orders of magnitude is the part that transfers.

---

## 14. What this work does not establish

The claims this study supports are narrow, and keeping them narrow is more useful
than inflating them.

**The criterion has never been applied to a human recording.** Every number in
this document comes from simulation. The PPG-DaLiA dataset of Reiss et al. (2019)
contains wrist conductance and wrist acceleration for fifteen participants under
activities including walking and cycling, the repository includes a reader for it
in `src/edarecover/dalia.py`, and the criterion has not yet been run against it.
Until that is done, the threshold of 0.156 is a property of this simulation and
nothing more.

**The pressure coupling is assumed, and the threshold depends on it.** Section
7.2 states the problem and section 13.1 measures the consequence. The area under
the curve holds between 0.961 and 0.990 across couplings from 0.05 to 0.8, so
the observable discriminates reliably whatever the coupling is, and the fitted
threshold moves from 0.164 to 0.323 over the same range, so the number 0.156 is
not transferable. At zero coupling the criterion has nothing to detect and the
coherence moves in the wrong direction with blur. A bench measurement, recording
conductance while the wrist is loaded with known weights, would replace the
assumption with a number, and short of that a device would have to calibrate its
own threshold.

**The reconstruction is evaluated under an inverse crime.** Section 13.2
quantifies how much the error grows when the operator is wrong by a factor of
two, which is not the same thing as evaluating the method with the operator
estimated from the data, and estimating it from the data is what a deployment
would have to do. The measured sensitivity to the operator is low, and section
13.2 explains why that is the wrong thing to be pleased about.

**The second reconstruction stage does not pay for itself.** Section 13.3 reports
the measurement and the measurement is unfavourable. The first stage alone beats
the full two-stage procedure at ten of the eleven blur levels, by up to twelve
per cent, and the inversion applied on its own at `b` = 6 s leaves an error 16
per cent worse than leaving the recording untouched. Reporting it anyway is the
point. The two-stage structure was motivated by an argument about the order of
operations, that argument turned out to be correct about the order and wrong
about the value of the second step, and the next version should either drop the
inversion or change the penalty it uses.

**The error is driven by the additive term rather than by the blur.** Sections 9
and 13.1 together make this plain. At `β` = 0 the uncorrected error moves only
from 0.0100 to 0.0190 µS across the entire blur range, never reaching a quarter
of the tolerance that defines the label, while the effective rank falls by a
factor of 2.6 over the same range. The inverse-problem framing is therefore a
description of the model rather than a description of what limits accuracy here,
and a reader who wants the blur to be the operative difficulty would need a
regime with far more aggressive smoothing or far less additive contamination.

**The calibration set is small and narrow.** Sixty-six runs, six seeds, one
window length, one noise level, one baseline configuration. The threshold is
determined by a single observation, as section 11.3 notes. Six groups is the
minimum at which leave-one-group-out says anything at all.

**The random streams overlap across seeds.** Section 4.5 documents the defect in
full, including the reason it has not yet been fixed.

**The spectral resolution is coarse.** Five frequency bins inside the analysis
band, from a 20 s Welch segment. A longer window would give a finer grid and a
better-conditioned estimate, at the cost of a slower gate and a coarser time
resolution for the decision.

**Nothing has been compared against an existing artifact detector.** The
rule-based procedure of Kleckner et al. (2018) is the obvious baseline and has
not been run.

---

## 15. Reproducing every number in this document

The package has no data dependencies, so a clone and an install reproduce
everything.

```bash
git clone https://github.com/cbrg0/eda-recoverability
cd eda-recoverability
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The calibration of section 11, which is the step that produces the threshold:

```bash
edarecover calibrate
```

The cross-validation of section 12 and the four measurements of sections 13.1
through 13.4:

```bash
edarecover validate
```

Any single quantity, with machine-readable output:

```bash
edarecover --json coherence
edarecover --json rank
edarecover --json cost
```

A faster pass over a reduced grid, for checking that the installation works
before committing to the full sweep:

```bash
edarecover --quick calibrate
```

The test suite, which includes regression tests pinning the published values of
the threshold, the area under the curve and the pooled cross-validation rates:

```bash
pytest                    # the fast tests
pytest -m slow            # the regression tests, several minutes
```

Global flags belong before the subcommand, so `edarecover -v calibrate` works
and `edarecover calibrate -v` is rejected by the parser. That is standard
`argparse` behaviour for a program with a shared global flag, it is documented
here because the error message is not self-explanatory, and two tests pin it so
that it cannot change silently.

The numbers in this document were produced under Python 3.14.6, NumPy 2.5.3,
SciPy 1.18.1 and scikit-learn 1.9.1. The regression tests assert the published
values to four decimal places, so a future version of any dependency that moves
them will fail the suite rather than changing the documentation silently.

---

## 16. Corrections to the earlier write-up

An earlier draft of this work circulated before the analysis code was rewritten
as a tested package. The rewrite surfaced five defects, three of which changed
printed numbers, and they are collected here so that anybody comparing the two
versions can see at once which figures moved and why. Each one is also described
where it belongs in the sections above.

**The effective-rank table was computed under a different movement model from
the rest of the work.** It came from a 60-second window with two movement bursts,
while every other figure uses 240-second windows with eight, and the note under
the table did not record the burst count. Two bursts in a short window leave 47.6
per cent of the samples at a severity below 0.05, so nearly half the rows of `A`
stay close to the identity and their singular values do not fall, against 3.1 per
cent with eight bursts; the mean severity is almost the same either way, 0.265
against 0.272, so what changed was the distribution of movement in time rather
than its amount. Section 9 now reports the table on the 240-second window, where
the surviving fraction at `b` = 6 s is 0.389 rather than the 0.294 previously
printed for the smaller window.

**The threshold in that count was described as the noise level in
microsiemens.** It is one per cent of the largest singular value, which is a
relative quantity that coincides numerically with `σ_n` = 0.01 µS without being
the same kind of thing. Section 9 now states the relative form. No count changes.

**The response time constants were attributed to a published source that does
not report them.** Section 4.3 now states that the Bateman form comes from the
decomposition literature while `τ_1` = 0.75 s and `τ_2` = 2.0 s were chosen here
to place the peak at 1.18 s and the one per cent return at 11.3 s. The
attribution has been withdrawn rather than replaced.

**The spectral estimate was described as twelve non-overlapping segments.** The
routine applies a Hann window with 50 per cent overlap and averages 23
overlapping segments in a 240-second window. Section 8 now gives both figures and
explains that 12 is the conservative count of independent averages, because
overlapping segments share half their samples. No printed value changes.

**The overlap between the random streams of the two generators was not
documented.** Section 4.5 now describes it in full, including the reason the
correction has been deferred.

Three measurements were added in the same rewrite and have no counterpart in the
earlier draft: the coupling sweep of section 13.1, the stage ablation of section
13.3 and the decimation check of section 13.4. The ablation is unfavourable to
the method as specified, and section 14 says so.

---

## References

Benedek, M., & Kirchner, C. (2010). A continuous measure of phasic electrodermal
activity. *Journal of Neuroscience Methods*, *190*(1), 80–91.
https://doi.org/10.1016/j.jneumeth.2010.04.028

Greco, A., Valenza, G., Lanata, A., Scilingo, E. P., & Citi, L. (2016). cvxEDA: A
convex optimization approach to electrodermal activity processing. *IEEE
Transactions on Biomedical Engineering*, *63*(4), 797–804.
https://doi.org/10.1109/TBME.2015.2474131

Hansen, P. C. (1990). The discrete Picard condition for discrete ill-posed
problems. *BIT Numerical Mathematics*, *30*(4), 658–672.
https://doi.org/10.1007/BF01933214

Hansen, P. C., & O'Leary, D. P. (1993). The use of the L-curve in the
regularization of discrete ill-posed problems. *SIAM Journal on Scientific
Computing*, *14*(6), 1487–1503. https://doi.org/10.1137/0914086

Kaipio, J., & Somersalo, E. (2007). Statistical inverse problems:
Discretization, model reduction and inverse crimes. *Journal of Computational and
Applied Mathematics*, *198*(2), 493–504.
https://doi.org/10.1016/j.cam.2005.09.027

Kleckner, I. R., Jones, R. M., Wilder-Smith, O., Wormwood, J. B., Akcakaya, M.,
Quigley, K. S., Lord, C., & Goodwin, M. S. (2018). Simple, transparent, and
flexible automated quality assessment procedures for ambulatory electrodermal
activity data. *IEEE Transactions on Biomedical Engineering*, *65*(7),
1460–1467. https://doi.org/10.1109/TBME.2017.2758643

Posada-Quintero, H. F., & Chon, K. H. (2020). Innovations in electrodermal
activity data collection and signal processing: A systematic review. *Sensors*,
*20*(2), 479. https://doi.org/10.3390/s20020479

Posada-Quintero, H. F., Florian, J. P., Orjuela-Cañón, A. D., Aljama-Corrales,
T., Charleston-Villalobos, S., & Chon, K. H. (2016). Power spectral density
analysis of electrodermal activity for sympathetic function assessment. *Annals
of Biomedical Engineering*, *44*(10), 3124–3135.
https://doi.org/10.1007/s10439-016-1606-6

Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). Deep PPG:
Large-scale heart rate estimation with convolutional neural networks. *Sensors*,
*19*(14), 3079. https://doi.org/10.3390/s19143079

Taylor, S., Jaques, N., Chen, W., Fedor, S., Sano, A., & Picard, R. (2015).
Automatic identification of artifacts in electrodermal activity data. In
*Proceedings of the 37th Annual International Conference of the IEEE Engineering
in Medicine and Biology Society* (pp. 1934–1937).
https://doi.org/10.1109/EMBC.2015.7318762
