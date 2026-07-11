"""
scripts/run_inference.py — Phase 5 (posterior inference + PPC on real data), v2.

CHANGED FROM v1: since the model now amortizes over "theta -> K sentences
from one reader" rather than "theta -> one sentence", inference is now a
SINGLE posterior call conditioned on the whole participant's batch of
sentences (resampled/subset to K_SENTENCES internally), instead of
sampling a separate posterior per trial and averaging afterwards. This
is both simpler and statistically correct for this architecture.

Loads a trained checkpoint, loads the real participant data (or synthesizes
a stand-in dataset if the raw files are not present at the given paths —
see src/dataio/data_pipeline.py), obtains ONE amortized posterior for the
whole participant, and runs posterior predictive checks against the 6
behavioral measures from the paper's Fig. 9 (SFD, GD, TT,
skipping/refixation/regression probabilities).

Usage
-----
    python -m scripts.run_inference \
        --fixation-file data/raw/fixseqin_PB2expVP10.dat \
        --corpus-file   data/raw/Rcorpus_PB2.dat \
        --checkpoint    outputs/checkpoints/swift_bayesflow.keras
"""
from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras

from src.dataio.data_pipeline import load_or_synthesize
from src.networks.bf_simulator_adapter import PARAM_NAMES, build_condition_batch_multi
from src.networks.multi_sentence_summary import MultiSentenceSummaryNetwork
from src.diagnostics.diagnostics import posterior_predictive_check


def parse_args():
    p = argparse.ArgumentParser(description="Amortized SWIFT inference on real/synthetic data")
    p.add_argument("--fixation-file", type=str, default="data/raw/fixseqin_PB2expVP10.dat")
    p.add_argument("--corpus-file", type=str, default="data/raw/Rcorpus_PB2.dat")
    p.add_argument("--checkpoint", type=str, default="outputs/checkpoints/swift_bayesflow.keras")
    p.add_argument("--n-posterior-samples", type=int, default=2000)
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--max-trials", type=int, default=200,
                    help="cap number of real trials used (for speed / memory)")
    return p.parse_args()


class _WF:
    """Thin wrapper so diagnostics.posterior_predictive_check (written
    against the BasicWorkflow.sample interface) also works with a bare
    loaded approximator (keras.saving.load_model returns the raw
    approximator, not a BasicWorkflow)."""
    def __init__(self, approx):
        self.approx = approx

    def sample(self, num_samples, conditions):
        return self.approx.sample(num_samples=num_samples, conditions=conditions)


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)

    print("Loading data...")
    trials, source = load_or_synthesize(args.fixation_file, args.corpus_file)
    print(f"  source={source}, n_trials={len(trials)}")
    if source != "real":
        raise RuntimeError(
            f"Expected real participant data but got source='{source}'. "
            f"Check --fixation-file/--corpus-file paths: "
            f"{args.fixation_file}, {args.corpus_file}"
        )
    trials = trials[: args.max_trials]

    print(f"Loading trained approximator from {args.checkpoint} ...")
    approximator = keras.saving.load_model(args.checkpoint, custom_objects={'MultiSentenceSummaryNetwork': MultiSentenceSummaryNetwork})
    workflow = _WF(approximator)

    # ---- ONE posterior for the whole participant, conditioned on their
    #      full (resampled-to-K_SENTENCES) batch of sentences ----
    conditions = build_condition_batch_multi([trials])  # (1, K_SENTENCES, FIX_MAX, 4)
    print(f"Sampling {args.n_posterior_samples} posterior draws for this participant "
          f"(pooled over {len(trials)} real trials, resampled to K_SENTENCES per call)...")
    post = workflow.sample(num_samples=args.n_posterior_samples, conditions=conditions)

    rows = []
    for p in PARAM_NAMES:
        samples = np.asarray(post[p]).reshape(-1)
        q05, q50, q95 = np.quantile(samples, [0.05, 0.5, 0.95])
        rows.append({
            "param": p,
            "post_mean": float(samples.mean()),
            "post_sd": float(samples.std()),
            "q05": q05, "q50": q50, "q95": q95,
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(args.outdir, "participant_posterior_summary.csv"), index=False)
    print(summary_df.to_string(index=False))

    # Posterior densities plot
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(5 * len(PARAM_NAMES), 4))
    if len(PARAM_NAMES) == 1:
        axes = [axes]
    for ax, p in zip(axes, PARAM_NAMES):
        samples = np.asarray(post[p]).reshape(-1)
        ax.hist(samples, bins=40, density=True, alpha=0.7)
        ax.set_title(f"Posterior: {p}")
        ax.set_xlabel(p)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "figures", "participant_posterior.png"), dpi=150)
    plt.close(fig)

    # ---- Posterior predictive check against the 6 behavioral measures ----
    print("Running posterior predictive checks...")
    ppc_df = posterior_predictive_check(workflow, [trials], n_posterior_draws=20)
    ppc_df.to_csv(os.path.join(args.outdir, "ppc_summary.csv"), index=False)
    print(ppc_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ppc_df["real_mean"], ppc_df["sim_mean"])
    for _, row in ppc_df.iterrows():
        ax.annotate(row["measure"], (row["real_mean"], row["sim_mean"]), fontsize=8)
    lo = min(ppc_df["real_mean"].min(), ppc_df["sim_mean"].min())
    hi = max(ppc_df["real_mean"].max(), ppc_df["sim_mean"].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("real / observed")
    ax.set_ylabel("simulated (posterior predictive)")
    ax.set_title("Posterior predictive check: behavioral measures")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "figures", "ppc_scatter.png"), dpi=150)
    plt.close(fig)

    print("Done. Outputs in:", args.outdir)


if __name__ == "__main__":
    main()
