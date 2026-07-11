"""
Phase 3/4 bridge, v2 — MULTI-SENTENCE ("participant-level") amortization.

CHANGED FROM v1: each training example is no longer "theta -> one sentence".
It is now "theta -> K sentences read by the same simulated reader", which
is what actually carries enough signal to recover nu and r (see
src/networks/multi_sentence_summary.py docstring for the full explanation
of why the single-sentence version failed to recover nu, r).

Design choices:

  1. Sentence length N is drawn per-sentence in [N_MIN, N_MAX] (matches the
     real corpus range of 6-12 words/sentence). Instead of a separate
     'inference_conditions' plumbing line (which would also need to be
     nested over K and pooled), N is normalized and appended as a 4th,
     per-timestep-constant channel directly inside the padded sequence
     tensor. This keeps the whole pipeline to a single nested tensor and
     removes the need to handle conditions at two different nesting levels.
  2. Each simulated "participant" (one theta draw) reads K_SENTENCES
     sentences, padded to FIX_MAX fixations each, with a mask channel.
     Final "seq" tensor shape: (batch, K_SENTENCES, FIX_MAX, 4).
  3. Word-frequency effects (beta) and temporal-spatial coupling (iota)
     remain off by default (USE_BETA/USE_IOTA flags), same as v1.
"""
from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import bayesflow as bf

from src.simulator.swift_model import (
    sample_prior,
    simulate_single_trial,
    param_names,
    SwiftPriorBounds,
)

# ---------------------------------------------------------------------
# Fixed configuration for the padded-sequence representation
# ---------------------------------------------------------------------
N_MIN, N_MAX = 6, 12          # sentence length range (words), matches real corpus
FIX_MAX = 40                  # hard cap on fixations per sentence (padding target)
DUR_SCALE = 500.0             # ms, rough normalization scale for durations
K_SENTENCES = 20               # NEW: sentences simulated per "participant" (per theta draw)
USE_BETA = True              # word-frequency effect off by default (see docstring)
USE_IOTA = False              # temporal-spatial coupling off by default
BOUNDS = SwiftPriorBounds()

PARAM_NAMES = param_names(use_beta=USE_BETA, use_iota=USE_IOTA)
N_PARAMS = len(PARAM_NAMES)


def _pad_trial(x: np.ndarray, y: np.ndarray, N: int) -> np.ndarray:
    """Build one (FIX_MAX, 4) padded [x_norm, y_norm, mask, N_norm] array
    from a single simulated sentence's (x, y) fixation sequence. N_norm is
    broadcast identically across every timestep so it survives the later
    per-sentence encoding step as ordinary sequence content.
    """
    seq = np.zeros((FIX_MAX, 4), dtype=np.float32)
    n = min(len(x), FIX_MAX)
    seq[:n, 0] = x[:n].astype(np.float32) / float(N)     # normalized word position
    seq[:n, 1] = y[:n].astype(np.float32) / DUR_SCALE     # normalized duration
    seq[:n, 2] = 1.0                                      # mask: 1 = real fixation
    seq[:, 3] = N / N_MAX                                 # normalized sentence length (all steps)
    return seq


def _simulate_one_participant(theta_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate K_SENTENCES independent sentences for ONE parameter vector
    (one simulated "participant" / "reader"). Returns (K_SENTENCES, FIX_MAX, 4).
    """
    participant_seqs = np.zeros((K_SENTENCES, FIX_MAX, 4), dtype=np.float32)
    for k in range(K_SENTENCES):
        N = int(rng.integers(N_MIN, N_MAX + 1))
        log_freq = rng.uniform(0.2, 1.0, size=N).astype(np.float32) if USE_BETA else None
        trial = simulate_single_trial(
            theta_b, N=N, log_freq=log_freq,
            use_beta=USE_BETA, use_iota=USE_IOTA,
            max_fixations=FIX_MAX, rng=rng,
        )
        participant_seqs[k] = _pad_trial(trial["x"], trial["y"], N)
    return participant_seqs


def batched_sample_fn(batch_shape, rng: np.random.Generator | None = None) -> dict:
    """The core `is_batched=True` sampling function passed to
    `bf.simulators.LambdaSimulator`. `batch_shape` gives the number of
    simulated PARTICIPANTS requested for this call (each with K_SENTENCES
    sentences).
    """
    rng = rng or np.random.default_rng()
    batch_size = int(np.prod(batch_shape)) if not np.isscalar(batch_shape) else int(batch_shape)

    theta = sample_prior(batch_size, use_beta=USE_BETA, use_iota=USE_IOTA, bounds=BOUNDS, rng=rng)

    seqs = np.zeros((batch_size, K_SENTENCES, FIX_MAX, 4), dtype=np.float32)
    for b in range(batch_size):
        seqs[b] = _simulate_one_participant(theta[b], rng=rng)

    out = {"seq": seqs}  # (B, K_SENTENCES, FIX_MAX, 4)
    for j, name in enumerate(PARAM_NAMES):
        out[name] = theta[:, j: j + 1]  # (B, 1) each

    return out


def make_swift_simulator() -> bf.simulators.Simulator:
    """Returns a ready-to-use BayesFlow Simulator wrapping the SWIFT model."""
    return bf.simulators.LambdaSimulator(batched_sample_fn, is_batched=True)


def make_adapter() -> bf.Adapter:
    """Standard BayesFlow key routing:
       - theta params -> 'inference_variables' (what we want the posterior over)
       - padded seq   -> 'summary_variables'   (fed to the two-level summary net)

       NOTE: 'inference_conditions' (N) is gone in v2 -- N now travels inside
       the sequence tensor itself as channel 4 (see _pad_trial), since it
       needs to be present at BOTH nesting levels (per-sentence AND pooled),
       which is simplest to achieve by just keeping it in the raw sequence.
    """
    adapter = (
        bf.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")

        .constrain("nu",   lower=BOUNDS.nu_low,   upper=BOUNDS.nu_high,   method="sigmoid")
        .constrain("r",    lower=BOUNDS.r_low,    upper=BOUNDS.r_high,    method="sigmoid")
        .constrain("muT",  lower=BOUNDS.muT_low,  upper=BOUNDS.muT_high,  method="sigmoid")
        .constrain("beta", lower=BOUNDS.beta_low, upper=BOUNDS.beta_high, method="sigmoid")

        .concatenate(PARAM_NAMES, into="inference_variables")
        .rename("seq", "summary_variables")
    )
    return adapter


def build_condition_batch_multi(participant_trials: list[list[dict]]) -> dict:
    """Helper for inference time: build a (num_participants, K, FIX_MAX, 4)
    'seq' tensor from REAL data, where `participant_trials` is a list (one
    entry per participant) of lists of trial-dicts (each trial-dict has
    keys 'x', 'y', 'N' as produced by src/dataio/data_pipeline.py).

    If a participant has fewer than K_SENTENCES real trials, sentences are
    resampled with replacement to fill K_SENTENCES (documented in README);
    if more, a random subset of K_SENTENCES is used.
    """
    rng = np.random.default_rng(0)
    batch_size = len(participant_trials)
    seqs = np.zeros((batch_size, K_SENTENCES, FIX_MAX, 4), dtype=np.float32)
    for b, trials in enumerate(participant_trials):
        idx = rng.choice(len(trials), size=K_SENTENCES, replace=len(trials) < K_SENTENCES)
        for k, i in enumerate(idx):
            t = trials[i]
            seqs[b, k] = _pad_trial(t["x"], t["y"], t["N"])
    return {"seq": seqs}


if __name__ == "__main__":
    sim = make_swift_simulator()
    sample = sim.sample(4)
    for k, v in sample.items():
        print(k, np.asarray(v).shape)
