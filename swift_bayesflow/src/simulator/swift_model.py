"""
Simplified SWIFT model — forward simulator.

Implements the generative process from:
Engbert, R., & Rabe, M. M. (2024). A tutorial on Bayesian inference for
dynamical modeling of eye-movement control during reading.
Journal of Mathematical Psychology, 119, 102843.

Equations referenced by number correspond to the paper (see Phase 1
theory-extraction document). This module is pure NumPy, fully vectorized
over the batch dimension, and is deliberately dependency-free from
BayesFlow so it can be unit-tested / profiled in isolation.

Two variants are implemented:
    - "baseline": temporal and spatial processes are fully decoupled
      (Section 4.1 of the paper).
    - "coupled":  the mean saccade timer depends on the activation of the
      currently-fixated word via coupling parameter iota (Eq. 22,
      Section 4.2).

Core simplification kept from the paper itself (Section 3): word-frequency
effects can be switched off (a_max_w = 1 for all words), dropping beta.
This is the authors' own recommended simplification and is the default
here for a minimal, fast-to-simulate model.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


ETA = 1e-3       # baseline saliency, Eq. (8), fixed by the authors
GAMMA_SHAPE = 9  # alpha, fixed shape of the fixation-duration Gamma, Eq. (11)


@dataclass
class SwiftPriorBounds:
    """Uniform prior bounds, taken directly from Engbert & Rabe (2024), Sec. 5."""
    nu_low: float = 0.0
    nu_high: float = 1.0
    r_low: float = 0.0
    r_high: float = 12.0
    muT_low: float = 100.0
    muT_high: float = 400.0
    beta_low: float = 0.0
    beta_high: float = 1.0
    iota_low: float = -0.5
    iota_high: float = 2.0


def sample_prior(
    batch_size: int,
    use_beta: bool = False,
    use_iota: bool = False,
    bounds: SwiftPriorBounds = SwiftPriorBounds(),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw theta ~ prior. Columns are always in the fixed order:
    [nu, r, muT] (+ beta if use_beta) (+ iota if use_iota).

    Returns
    -------
    theta : (batch_size, n_params) float32 array
    """
    rng = rng or np.random.default_rng()
    nu = rng.uniform(bounds.nu_low, bounds.nu_high, size=batch_size)
    r = rng.uniform(bounds.r_low, bounds.r_high, size=batch_size)
    muT = rng.uniform(bounds.muT_low, bounds.muT_high, size=batch_size)
    cols = [nu, r, muT]
    if use_beta:
        cols.append(rng.uniform(bounds.beta_low, bounds.beta_high, size=batch_size))
    if use_iota:
        cols.append(rng.uniform(bounds.iota_low, bounds.iota_high, size=batch_size))
    return np.stack(cols, axis=1).astype(np.float32)


def param_names(use_beta: bool = False, use_iota: bool = False) -> list[str]:
    names = ["nu", "r", "muT"]
    if use_beta:
        names.append("beta")
    if use_iota:
        names.append("iota")
    return names


def _processing_rates(nu: float, k: int, N: int) -> np.ndarray:
    """Eq. (1)-(2): eccentricity-dependent processing rate lambda_w for a
    single fixation position k on a sentence of N words. Returns a length-N
    vector lambda_w(t) (NOT yet scaled by overall rate r).
    """
    sigma = 1.0 / (1.0 + 2.0 * nu + nu ** 2)
    lam = np.zeros(N)
    w = np.arange(1, N + 1)  # 1-indexed word positions, matches paper
    lam[w == k - 1] = sigma * nu
    lam[w == k] = sigma
    lam[w == k + 1] = sigma * nu
    lam[w == k + 2] = sigma * (nu ** 2)
    return lam


def simulate_single_trial(
    theta: np.ndarray,
    N: int,
    log_freq: Optional[np.ndarray] = None,
    use_beta: bool = False,
    use_iota: bool = False,
    max_fixations: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Simulate ONE reading trial (one sentence) for one parameter vector.

    Parameters
    ----------
    theta : array of shape (n_params,) in order [nu, r, muT, (beta), (iota)]
    N : number of words in the sentence
    log_freq : optional array of shape (N,), normalized log word frequency
               q_w in Eq. (4). Required only if use_beta=True. If None and
               use_beta=True, a synthetic Zipf-like frequency profile is
               generated.
    use_beta, use_iota : which model extensions are active; MUST match how
        `theta` was constructed via `sample_prior`.
    max_fixations : hard cap to guarantee termination even for pathological
        parameter draws (e.g. nu, r -> 0, which stalls processing).

    Returns
    -------
    dict with keys:
        x : (n_fix,) int array of fixated word indices (1..N)
        y : (n_fix,) float array of fixation durations (ms)
        n_fix : int, number of fixations produced
    """
    rng = rng or np.random.default_rng()

    idx = 0
    nu, r, muT = theta[0], theta[1], theta[2]
    idx = 3
    beta = theta[idx] if use_beta else None
    idx += 1 if use_beta else 0
    iota = theta[idx] if use_iota else None

    # word-frequency-modulated max activation, Eq. (4)-(5)
    if use_beta:
        if log_freq is None:
            # synthetic normalized log-frequency profile in (0, 1]
            log_freq = rng.uniform(0.2, 1.0, size=N)
        a_max = np.clip(1.0 - beta * log_freq, 1e-3, 1.0)
    else:
        a_max = np.ones(N)

    a = np.zeros(N)          # activation state a_w(t)
    k = 1                    # start of scanpath: first word fixated
    t = 0.0

    rate0 = GAMMA_SHAPE / muT  # baseline gamma rate parameter rho = alpha/muT

    xs, ys = [], []

    for i in range(max_fixations):
        # --- processing rates for current fixation position, Eq. (1)-(2) ---
        lam = _processing_rates(nu, k, N)

        # --- sample fixation duration T_i, Eq. (10)-(12), optionally
        #     coupled to activation of fixated word via Eq. (22) ---
        rate = rate0
        if use_iota:
            rate = rate0 * (1.0 + iota * a[k - 1])
        rate = max(rate, 1e-6)
        T_i = rng.gamma(shape=GAMMA_SHAPE, scale=1.0 / rate)

        xs.append(k)
        ys.append(T_i)

        # --- update activations over the fixation, Eq. (7) ---
        a = np.clip(a + r * lam * (T_i / 1000.0), 0.0, a_max)
        t += T_i

        # termination: last word of sentence has been fixated
        if k == N:
            break

        # --- saliency transform, Eq. (8) ---
        s = a_max * np.sin(np.pi * np.clip(a / a_max, 0.0, 1.0)) + ETA

        # --- target selection probabilities, Eq. (9) ---
        p = s / s.sum()

        # sample next fixated word (1-indexed)
        k = int(rng.choice(np.arange(1, N + 1), p=p))

    return {
        "x": np.array(xs, dtype=np.int64),
        "y": np.array(ys, dtype=np.float32),
        "n_fix": len(xs),
    }


def simulate_batch(
    theta: np.ndarray,
    N: int | np.ndarray,
    use_beta: bool = False,
    use_iota: bool = False,
    max_fixations: int = 200,
    log_freq: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[dict]:
    """Simulate a batch of trials, one per row of theta.

    N may be a single int (all sentences the same length) or an array of
    length batch_size (variable sentence length, as in the real corpus).
    """
    rng = rng or np.random.default_rng()
    batch_size = theta.shape[0]
    N_arr = np.full(batch_size, N, dtype=int) if np.isscalar(N) else np.asarray(N)

    out = []
    for b in range(batch_size):
        lf = None
        if log_freq is not None:
            lf = log_freq[b] if isinstance(log_freq, (list, tuple)) else log_freq
        out.append(
            simulate_single_trial(
                theta[b],
                N=int(N_arr[b]),
                log_freq=lf,
                use_beta=use_beta,
                use_iota=use_iota,
                max_fixations=max_fixations,
                rng=rng,
            )
        )
    return out


if __name__ == "__main__":
    # quick smoke test: reproduce roughly the paper's own example
    # (nu=0.3, r=10, muT=200 -> Section 3 / Fig. 4)
    rng = np.random.default_rng(0)
    theta = np.array([0.3, 10.0, 200.0], dtype=np.float32)
    trial = simulate_single_trial(theta, N=10, rng=rng)
    print("x:", trial["x"])
    print("y (ms):", np.round(trial["y"], 1))
    print("n_fix:", trial["n_fix"])
