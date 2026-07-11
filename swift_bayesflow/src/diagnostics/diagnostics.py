"""
Phase 5 — Diagnostics: parameter recovery and posterior predictive checks
(PPCs), mirroring the validation logic the paper itself uses (Section 5
parameter recovery via DREAM_ZS MCMC; Section 6 posterior predictive checks
against interindividual reading measures) but computed here for our
amortized BayesFlow posterior instead of MCMC.
"""
from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd

from src.networks.bf_simulator_adapter import (
    make_swift_simulator, PARAM_NAMES, build_condition_batch_multi,
)
from src.simulator.swift_model import simulate_single_trial


# ----------------------------------------------------------------------
# Parameter recovery (simulation-based calibration style)
# ----------------------------------------------------------------------

def parameter_recovery(workflow, n_test: int = 200, n_posterior_samples: int = 500,
                        seed: int = 123) -> pd.DataFrame:
    """Draw `n_test` fresh (theta, K-sentence-batch) pairs from the
    simulator, obtain the amortized posterior for each SIMULATED PARTICIPANT
    (conditioned on their full K_SENTENCES-sentence "seq" tensor), and
    compare posterior mean/credible interval against the ground-truth theta
    used to generate that participant's data.

    Returns a tidy DataFrame with one row per (test case, parameter):
    columns = [case, param, true, post_mean, post_sd, q05, q50, q95, in_90ci]
    """
    sim = make_swift_simulator()
    batch = sim.sample(n_test)  # batch["seq"] has shape (n_test, K_SENTENCES, FIX_MAX, 4)

    rows = []
    for i in range(n_test):
        conditions = {"seq": batch["seq"][i: i + 1]}  # (1, K_SENTENCES, FIX_MAX, 4)
        post = workflow.sample(num_samples=n_posterior_samples, conditions=conditions)
        for p in PARAM_NAMES:
            samples = np.asarray(post[p]).reshape(-1)
            true_val = float(batch[p][i, 0])
            q05, q50, q95 = np.quantile(samples, [0.05, 0.5, 0.95])
            rows.append({
                "case": i,
                "param": p,
                "true": true_val,
                "post_mean": float(samples.mean()),
                "post_sd": float(samples.std()),
                "q05": q05, "q50": q50, "q95": q95,
                "in_90ci": bool(q05 <= true_val <= q95),
            })
    return pd.DataFrame(rows)


def recovery_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-parameter summary: correlation(true, post_mean) and empirical
    90% CI coverage (should be close to 0.9 if the posterior is
    well-calibrated)."""
    out = []
    for p, g in df.groupby("param"):
        corr = np.corrcoef(g["true"], g["post_mean"])[0, 1]
        coverage = g["in_90ci"].mean()
        out.append({"param": p, "correlation": corr, "coverage_90ci": coverage})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# Posterior predictive checks against behavioral summary measures
# (mirrors the paper's Fig. 9: SFD, gaze duration, total fixation time,
#  skipping / refixation / regression probabilities)
# ----------------------------------------------------------------------

def _fixation_measures(x: np.ndarray, y: np.ndarray, N: int) -> dict:
    """Compute the 6 behavioral measures used in the paper's Fig. 9 for a
    single fixation sequence (x = fixated word indices, y = durations)."""
    # single-fixation duration: words fixated exactly once
    counts = {w: 0 for w in range(1, N + 1)}
    dur_by_word = {w: [] for w in range(1, N + 1)}
    for xi, yi in zip(x, y):
        if 1 <= xi <= N:
            counts[xi] += 1
            dur_by_word[xi].append(yi)

    sfd_vals = [dur_by_word[w][0] for w in range(1, N + 1) if counts[w] == 1]
    sfd = float(np.mean(sfd_vals)) if sfd_vals else np.nan

    # gaze duration: sum of consecutive same-word fixations on first pass
    gd_vals, tt_vals = [], []
    seen_first_pass_done = set()
    i = 0
    first_pass_gaze = {}
    while i < len(x):
        w = x[i]
        if w not in first_pass_gaze and w not in seen_first_pass_done:
            j = i
            total = 0.0
            while j < len(x) and x[j] == w:
                total += y[j]
                j += 1
            first_pass_gaze[w] = total
            seen_first_pass_done.add(w)
            i = j
        else:
            i += 1
    gd_vals = list(first_pass_gaze.values())
    gd = float(np.mean(gd_vals)) if gd_vals else np.nan

    tt_vals = [sum(dur_by_word[w]) for w in range(1, N + 1) if counts[w] > 0]
    tt = float(np.mean(tt_vals)) if tt_vals else np.nan

    # skipping probability: words never fixated in first pass
    fixated_words = set(x.tolist())
    skip_prob = 1.0 - (len(fixated_words) / N)

    # refixation probability: probability of an immediate re-fixation on the same word
    refix_count, total_landings = 0, 0
    for i in range(len(x) - 1):
        total_landings += 1
        if x[i] == x[i + 1]:
            refix_count += 1
    refix_prob = refix_count / total_landings if total_landings else np.nan

    # regression probability: saccade moves to an earlier word
    reg_count, total_saccades = 0, 0
    for i in range(len(x) - 1):
        total_saccades += 1
        if x[i + 1] < x[i]:
            reg_count += 1
    reg_prob = reg_count / total_saccades if total_saccades else np.nan

    return {
        "SFD": sfd, "GD": gd, "TT": tt,
        "P_skip": skip_prob, "P_refix": refix_prob, "P_reg": reg_prob,
    }


def dataset_measures(trials: list[dict]) -> pd.DataFrame:
    """Compute the 6 measures for every trial in a dataset (real or
    simulated) and return their means (analogous to one "participant" row
    in the paper's Fig. 9 scatter plots)."""
    rows = [_fixation_measures(t["x"], t["y"], t["N"]) for t in trials]
    df = pd.DataFrame(rows)
    return df.mean(numeric_only=True).to_frame(name="value").reset_index().rename(
        columns={"index": "measure"}
    )


def posterior_predictive_check(
    workflow, participant_trial_groups: list[list[dict]],
    n_posterior_draws: int = 30, seed: int = 7,
) -> pd.DataFrame:
    """For each PARTICIPANT (a list of trial-dicts sharing one reader,
    resampled/subset to K_SENTENCES sentences internally): get ONE amortized
    posterior conditioned on that participant's full sentence batch, draw
    parameter samples from it, re-simulate synthetic sentences with those
    parameters, and compare the resulting behavioral measures against the
    participant's own real measures (averaged across their real trials).

    `participant_trial_groups` is a list of participants, each itself a
    list of trial-dicts (as produced by src/dataio/data_pipeline.py). If
    you only have one participant's data (the common case for this
    project's real files), pass `[trials]` -- a single-element list.

    Returns a tidy DataFrame with columns [measure, real_mean, sim_mean,
    correlation_across_participants] -- directly analogous to the paper's
    Fig. 9 (one point per participant per measure there; here we also
    support the n=1-participant case by reporting means only).
    """
    rng = np.random.default_rng(seed)
    real_rows, sim_rows = [], []

    for trials in participant_trial_groups:
        # ---- get ONE posterior for this participant, conditioned on all
        #      their (resampled-to-K_SENTENCES) sentences at once ----
        cond = build_condition_batch_multi([trials])  # (1, K_SENTENCES, FIX_MAX, 4)
        post = workflow.sample(num_samples=n_posterior_draws, conditions=cond)
        thetas = np.stack([np.asarray(post[p]).reshape(-1) for p in PARAM_NAMES], axis=1)

        # ---- real behavioral measures: averaged across this participant's
        #      own real trials ----
        real_meas_list = [_fixation_measures(t["x"], t["y"], t["N"]) for t in trials]
        real_rows.append(pd.DataFrame(real_meas_list).mean(numeric_only=True))

        # ---- simulated behavioral measures: for each posterior draw,
        #      simulate one sentence per real trial's sentence length N and
        #      average the resulting measures ----
        sim_meas_list = []
        for k in range(n_posterior_draws):
            for t in trials:
                N = t["N"]
                log_freq = t.get("log_freq", np.ones(N, dtype=np.float32))
                sim_trial = simulate_single_trial(thetas[k], N=N, log_freq=log_freq, rng=rng)
                sim_meas_list.append(_fixation_measures(sim_trial["x"], sim_trial["y"], N))
        sim_rows.append(pd.DataFrame(sim_meas_list).mean(numeric_only=True))

    real_df = pd.DataFrame(real_rows)
    sim_df = pd.DataFrame(sim_rows)

    out = []
    multi_participant = len(real_df) > 1

    for measure in real_df.columns:
        row = {
            "measure": measure,
            "real_mean": real_df[measure].mean(),
            "sim_mean": sim_df[measure].mean(),
        }

        if multi_participant:
            row["correlation_across_participants"] = np.corrcoef(
                real_df[measure].fillna(0),
                sim_df[measure].fillna(0)
            )[0, 1]

        out.append(row)

    return pd.DataFrame(out)
