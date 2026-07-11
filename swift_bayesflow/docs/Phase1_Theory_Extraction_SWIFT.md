# Phase 1 — Theory Extraction: The Simplified SWIFT Model
### Source: Engbert & Rabe (2024), *J. Math. Psychol.* 119, 102843

This document isolates every mathematical object needed to (a) write a forward simulator and
(b) know what the BayesFlow summary network will actually be summarizing. It is organized as:
parameters → state variables → generative equations (baseline, then coupled/advanced model) →
likelihood decomposition (tells you what "data" means) → priors/true values used by the authors →
simulation algorithm in pseudocode → notes on what to keep vs. simplify for your project.

---

## 1. What kind of object is being generated?

The model generates a **fixation sequence** for one sentence of `N` words:

```
S = { f_1, f_2, ..., f_S },   f_i = (x_i, y_i)
```

- `x_i` ∈ {1, ..., N} — the word index fixated at step *i* (spatial/discrete)
- `y_i` ∈ ℝ⁺ — the duration of fixation *i* in ms (temporal/continuous)

This is exactly the structure of your `fixseqin_PB2expVP10.dat` file (fixation position + duration
per row), and `Rcorpus_PB2.dat` supplies the per-word properties (length, frequency → needed for
`a_max_w` and, if you keep it, for the spatial layout). **This is your target "observed data"
tensor for BayesFlow**: a variable-length sequence of (word_index, duration) pairs.

---

## 2. Parameters to infer (θ)

These are the free parameters of the *simplified* SWIFT model discussed in the tutorial. Symbol,
meaning, domain, and where it enters the model:

| Symbol | Name | Domain | Role |
|---|---|---|---|
| `ν` (nu) | processing-span shape / eccentricity decay | `(0, 1]` | shapes λ₋₁, λ₁, λ₂ relative to λ₀ (Eq. 1–2) |
| `r` | overall processing rate | `r > 0` | scales all processing rates rλ_w(t) (Eq. 3, 6) |
| `μ_T` | mean saccade timer interval | `μ_T > 0` | mean of the gamma fixation-duration distribution (Eq. 12) |
| `β` | word-frequency parameter | `(0, 1)` | controls how word frequency lowers `a_max_w` (Eq. 5) |
| `ι` (iota) | temporal–spatial coupling strength | `(-0.5, 2)` in paper's prior | only in the **advanced/coupled** model; speeds up saccade timer with fixated-word activation (Eq. 22) |

Fixed (not inferred) constants used by the authors:
- `η = 10⁻³` — baseline saliency added to all words (Eq. 8), prevents zero denominators
- `α = 9` — shape parameter of the gamma fixation-duration distribution (fixed, not free)
- `Δt = 1 ms` — Euler integration step (only needed if you integrate continuously rather than
  jumping fixation-to-fixation; see §5, the authors actually use the closed-form per-fixation
  update, Eq. 7, so Δt is not really needed in practice)

**Minimal parameter set for your project (baseline, decoupled model):** `θ = (ν, r, μ_T, β)`
**If you want the coupled version:** `θ = (ν, r, μ_T, β, ι)`

For a *minimal working model* (per the assignment's advice to avoid too many moving parts), you
can even set `β → fixed` or ignore word frequency entirely (as the authors themselves do in their
own simulation examples in Section 3 — "we neglect word frequency effects, so a_max_w = 1 for all
words"), leaving just **`θ = (ν, r, μ_T)`** as the core 3-parameter recoverable set that the paper's
own profile-likelihood experiment (Fig. 5) is built around.

---

## 3. State variables

| Symbol | Meaning | Init | Bound |
|---|---|---|---|
| `a_w(t)` | activation ("processing state") of word *w* at time *t* | `a_w(0) = 0` | `0 ≤ a_w(t) ≤ a_max_w ≤ 1` |
| `k(t)` | index of the word currently fixated (gaze position) at time *t* | current fixation | discrete, 1..N |
| `s_w(t)` | saliency of word *w* (derived from activation) | — | ≥ 0 |
| `p_w(t)` | probability that word *w* is the next saccade target | — | Σ_w p_w(t) = 1 |

---

## 4. Generative equations — Baseline model

### 4.1 Eccentricity-dependent processing rate (Eq. 1–2)

For gaze on word `k(t)` at time `t`, word `w` receives processing rate:

```
λ_w(t) =
    0            if w ≤ k(t) − 2
    σν           if w = k(t) − 1
    σ            if w = k(t)          (fixated word, λ0)
    σν           if w = k(t) + 1
    σν²          if w = k(t) + 2
    0            if w ≥ k(t) + 3
```

with normalization
```
σ = 1 / (1 + 2ν + ν²)
```
so that `σ · Σ_{i=-1}^{2} λ_i(t) = 1`. Note the **asymmetric span**: word k+2 gets rate σν²
but word k−2 gets 0 (models the known rightward asymmetry of the perceptual span).

### 4.2 Word activation dynamics (Eq. 3–7)

Continuous-time ODE:
```
d a_w / dt = r · λ_w(t),   for 0 ≤ a_w(t) ≤ 1,   a_w(0) = 0
```

Discrete Euler update (only if you need within-fixation resolution):
```
a_w(t + Δt) = a_w(t) + r·λ_w(t)·Δt
```

**Practical closed form actually used by the authors** — since λ_w(t) is piecewise constant
during a fixation (it only depends on which word is fixated, not on time within the fixation),
activation can be updated once per fixation rather than per millisecond:
```
a_w(t + T_i) = a_w(t) + r·λ_w(t) · T_i        (Eq. 7)
```
where `T_i` is the duration of fixation *i*. **This is the key simplification that makes the
simplified SWIFT model fast to simulate** — you never need fine time-stepping, only one update per
fixation event.

### 4.3 Word-frequency-modulated maximum activation (Eq. 4–5)

```
q_w = log10(F_w) / max_i{log10(F_i)}         # normalized log word frequency
a_max_w = 1 − β · q_w                        # in (0,1]
```
`F_w` = corpus word frequency (this is exactly the column you'll pull from
`Rcorpus_PB2.dat`). If you drop word-frequency effects for simplicity (as the paper itself does in
its own simulation illustration), just set `a_max_w = 1` for all words and drop `β` from θ.

### 4.4 Saliency transform (Eq. 8, Fig. 3)

```
s_w(t) = a_max_w · sin(π · a_w(t) / a_max_w) + η
```
Unimodal in `a_w`: saliency is 0 at `a_w=0` and `a_w=a_max_w`, peaks at `a_w = a_max_w/2`. This is
what turns "how processed is this word" into "how attractive a saccade target is this word right
now" — a word that's untouched or fully read is a boring target; a half-processed word is the
juiciest target.

### 4.5 Saccade target selection probability (Eq. 9)

```
p_w(t) = s_w(t) / Σ_{v=1}^{N} s_v(t)
```
Next fixated word `x_i` is drawn categorically from this distribution over all N words.

### 4.6 Fixation duration / saccade timer (Eq. 10–12)

Fixation durations are i.i.d. **Gamma** draws (baseline/decoupled model):
```
T_i ~ Gamma(shape = α, rate = ρ),     ρ = α / μ_T
E[T_i] = μ_T,      SD[T_i] = μ_T/√α = μ_T/3   (since α fixed at 9)
CV = 1/3
```
Crucially: in the **baseline model**, `T_i` is drawn **independently of the activation state** —
timing and target selection are decoupled processes that merely happen to share the same
underlying scanpath.

### 4.7 One simulation step, in order

```
1. At current time t, gaze is on word k(t). Sample T_i ~ Gamma(α, ρ=α/μ_T).
2. Update all word activations using Eq. 7: a_w(t+T_i) = a_w(t) + r·λ_w(t)·T_i, clipped to a_max_w.
3. Compute saliencies s_w(t+T_i) via Eq. 8.
4. Compute target probabilities p_w(t+T_i) via Eq. 9.
5. Sample next fixated word x_{i+1} ~ Categorical(p(t+T_i)); set k(t+T_i) = x_{i+1}.
6. Advance t ← t + T_i. Record fixation f_i = (x_i, T_i).
7. Stop when the last word of the sentence has been fixated (paper's own termination rule).
```

---

## 5. Generative equations — Advanced / coupled model (Section 4.2, Eq. 22)

Adds one parameter `ι` that lets the **fixated word's current activation** speed up (reduce) the
mean saccade timer:

```
rate' = rate · (1 + ι · a_k(t))
```
where `a_k(t)` is the activation of the *currently fixated* word at the *start* of the new
fixation. As `a_k(t) → 1`, mean fixation duration shrinks by a factor of ≈ `1/(1+ι)`. This is the
only structural change from §4 — everything else (Eq. 1–9) is identical. This coupling is what
lets `ν` and `r` (which affect activation, hence timing) leak into the **temporal** likelihood
component (see §6), which is not possible in the fully decoupled baseline model.

**Recommendation for a minimal working model:** implement the baseline (decoupled) model first
(§4 only); add `ι`-coupling only if time permits, since it is a strict superset (one extra line,
Eq. 22, replacing `rate` in step 1 of §4.7 with `rate'`).

---

## 6. Likelihood structure (tells you what BayesFlow's summary network should output)

Although BayesFlow will not use this likelihood directly (it's simulation-based/likelihood-free),
this section is essential because it tells you exactly **what quantities are sufficient
statistics** of the data, which is useful when designing/sanity-checking your summary network.

The full-sequence likelihood factorizes sequentially over fixations (Eq. 15):
```
L(θ | f_1,...,f_n) = P(f_1) · Π_{i=2}^{n} P(f_i | f_1,...,f_{i-1}; θ)
```
and, because target selection of `x_i` is independent of its own duration `y_i` given history,
each step further decomposes into **temporal × spatial** factors (Eq. 16–17):
```
P(f_i | history; θ) = P_temp(y_i | history, x_i; θ) · P_spat(x_i | history; θ)
log L = l_temp(θ) + l_spat(θ)
```
- `l_temp`: log-density of `y_i` under Gamma(shape=α, rate=ρ or ρ') — Eq. 18.
  In the **baseline model** this depends *only* on `μ_T` (flat w.r.t. ν, r — confirmed empirically
  in Fig. 5, top row). In the **coupled model** it also depends on `ν, r, ι` through `a_k(t)`.
- `l_spat`: log of the categorical target-selection probability `p_{x_i}(t')` after activations are
  updated using the *actually observed* fixation duration `y_{i-1}` (Eq. 19–21). This depends on
  `ν, r, β` (via activation dynamics) but **never on `μ_T`**, because real fixation durations are
  plugged in directly rather than modeled probabilistically at this stage.

**Practical implication for your BayesFlow summary network:** the sequence of fixation durations
carries information almost exclusively about `μ_T` (and `ι` if coupled), while the sequence of
fixated word positions (skips, refixations, regressions) carries information about `ν, r, (β)`.
A summary network (RNN/LSTM/DeepSet) that separately or jointly encodes `(x_i, y_i)` pairs across
the sequence should be able to recover this near-decoupling — this is a good sanity check/
diagnostic once training starts (does your posterior for μ_T become insensitive to permuting the
spatial sequence, etc.).

---

## 7. Priors / true parameter values used by the authors (for calibrating your own priors)

From the MCMC parameter-recovery experiment (Section 5), the authors use **uniform priors**:

| Parameter | Prior | "True" value used to simulate recovery data |
|---|---|---|
| ν | Uniform(0, 1) | 0.25 |
| r | Uniform(0, 12) | 5.0 |
| μ_T | Uniform(100, 400) [ms] | 220 |
| ι | Uniform(−0.5, 2) | 0.5 |
| β | Uniform(0, 1) | 0.6 |

From the earlier simulation-example section (Section 3, used for Fig. 4/5 profile likelihoods):
`ν = 0.3, r = 10, μ_T = 200 ms` (word-frequency effects neglected, i.e. β irrelevant, a_max_w=1).

Fixed constants: `η = 10⁻³`, gamma shape `α = 9`.

**Suggested priors for your BayesFlow implementation** (directly reusable, since they are already
the authors' own weakly-informative choices validated by their own parameter-recovery study):
```
ν     ~ Uniform(0, 1)
r     ~ Uniform(0, 12)
μ_T   ~ Uniform(100, 400)      # ms
β     ~ Uniform(0, 1)          # only if keeping word-frequency effect
ι     ~ Uniform(-0.5, 2)       # only if using the coupled/advanced model
```

---

## 8. Data requirements recap (linking theory to your two files)

- `Rcorpus_PB2.dat` → gives you `N` (words per sentence) and `F_w` (word frequency) per word,
  needed for `q_w` (Eq. 4) and hence `a_max_w` (Eq. 5) — only needed if you keep `β` in the model.
  It also implicitly gives sentence boundaries / word count, needed to know when to terminate a
  simulated scanpath (rule in §4.7 step 7).
- `fixseqin_PB2expVP10.dat` → real fixation sequences `{(x_i, y_i)}` for one participant (VP10) —
  this is the observed data your trained BayesFlow amortizer will condition on at inference time,
  and its empirical shape/length distribution is what your simulator's synthetic sequences must
  resemble (same N range 6–12 words per the original Risse & Seelig 2019 corpus, similar fixation
  count per trial, similar duration range 25–1000 ms after the authors' own preprocessing cutoffs).

---

## 9. What to simplify further, if needed (explicitly licensed by the paper itself)

The authors already present the "neglect word frequency" version as a legitimate simplification
(Section 3): set `a_max_w = 1` for all words, drop `β`. Combined with dropping the coupling `ι`,
your **absolute minimal generative model** reduces to 3 parameters `(ν, r, μ_T)`, driven entirely
by Eq. 1–2 (processing rates), Eq. 3/7 (activation update), Eq. 8–9 (saliency → target
probability), and Eq. 10–12 (Gamma fixation timer) — this is a self-contained, fully vectorizable
simulator and a very reasonable Phase 3 target.

---

## 10. Quick symbol glossary

| Symbol | Description |
|---|---|
| N | number of words in sentence |
| w | word index, 1..N |
| t | continuous time (ms) |
| k(t) | currently fixated word index |
| a_w(t) | activation of word w |
| a_max_w | max activation word w can reach (word-difficulty ceiling) |
| λ_w(t) | eccentricity-dependent processing rate for word w |
| ν | processing-span shape parameter |
| σ | normalization constant (function of ν) |
| r | overall processing rate |
| q_w | normalized log word frequency |
| β | word-frequency discount parameter |
| s_w(t) | saliency of word w |
| η | baseline saliency constant (fixed, 10⁻³) |
| p_w(t) | probability word w is next saccade target |
| T_i / y_i | duration of fixation i |
| x_i | fixated word index at fixation i |
| f_i | fixation i, = (x_i, y_i) |
| α, ρ | Gamma distribution shape (fixed=9) and rate (=α/μ_T) |
| μ_T | mean fixation duration / saccade timer interval |
| ι | temporal-spatial coupling strength (advanced model only) |
