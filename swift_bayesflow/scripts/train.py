"""
scripts/train.py — Phase 5 training entry point.

Trains the amortized SWIFT posterior via BayesFlow's online-simulation
training loop (fresh simulated batches every step, no fixed training set
needed since simulation is cheap), then runs parameter-recovery diagnostics
and saves a summary report + plots.

Usage
-----
    cd swift_bayesflow
    python -m scripts.train --epochs 30 --batches-per-epoch 100 --batch-size 64

Outputs
-------
    outputs/checkpoints/swift_bayesflow.keras   (trained weights)
    outputs/figures/recovery_*.png              (parameter recovery plots)
    outputs/recovery_summary.csv                (correlation / coverage table)
"""
from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.networks.build_workflow import build_workflow
from src.networks.bf_simulator_adapter import PARAM_NAMES
from src.diagnostics.diagnostics import parameter_recovery, recovery_summary


def parse_args():
    p = argparse.ArgumentParser(description="Train the SWIFT BayesFlow amortizer")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batches-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--summary-dim", type=int, default=16)
    p.add_argument("--flow-depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--n-recovery-cases", type=int, default=150)
    p.add_argument("--n-posterior-samples", type=int, default=500)
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def plot_recovery(df: pd.DataFrame, outdir: str):
    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(5 * len(PARAM_NAMES), 4.5))
    if len(PARAM_NAMES) == 1:
        axes = [axes]
    for ax, p in zip(axes, PARAM_NAMES):
        g = df[df["param"] == p]
        ax.errorbar(
            g["true"], g["post_mean"],
            yerr=[g["post_mean"] - g["q05"], g["q95"] - g["post_mean"]],
            fmt="o", alpha=0.4, ecolor="lightgray", markersize=3,
        )
        lo, hi = g["true"].min(), g["true"].max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="identity")
        ax.set_xlabel(f"true {p}")
        ax.set_ylabel(f"posterior mean {p}")
        ax.set_title(p)
        ax.legend(fontsize=8)
    fig.suptitle("Parameter recovery (amortized SWIFT posterior)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figures", "recovery_scatter.png"), dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "checkpoints"), exist_ok=True)

    print("=" * 70)
    print("Building SWIFT BayesFlow workflow")
    print("=" * 70)
    checkpoint_path = os.path.join(args.outdir, "checkpoints")
    workflow = build_workflow(
        summary_dim=args.summary_dim,
        flow_depth=args.flow_depth,
        initial_learning_rate=args.lr,
        checkpoint_filepath=checkpoint_path,
    )

    print("=" * 70)
    print(f"Training: {args.epochs} epochs x {args.batches_per_epoch} batches "
          f"x batch_size={args.batch_size}")
    print("=" * 70)
    t0 = time.time()
    history = workflow.fit_online(
        epochs=args.epochs,
        num_batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
    )
    print(f"Training completed in {time.time() - t0:.1f}s")

    loss_curve = np.asarray(history.history["loss"])
    np.save(os.path.join(args.outdir, "loss_curve.npy"), loss_curve)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_curve)
    ax.set_xlabel("epoch")
    ax.set_ylabel("negative log-likelihood (loss)")
    ax.set_title("Training loss")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "figures", "loss_curve.png"), dpi=150)
    plt.close(fig)

    print("=" * 70)
    print(f"Parameter recovery on {args.n_recovery_cases} held-out simulated trials")
    print("=" * 70)
    rec_df = parameter_recovery(
        workflow, n_test=args.n_recovery_cases,
        n_posterior_samples=args.n_posterior_samples, seed=args.seed + 1,
    )
    rec_df.to_csv(os.path.join(args.outdir, "recovery_raw.csv"), index=False)
    summary_df = recovery_summary(rec_df)
    summary_df.to_csv(os.path.join(args.outdir, "recovery_summary.csv"), index=False)
    print(summary_df.to_string(index=False))

    plot_recovery(rec_df, args.outdir)

    print("=" * 70)
    print("Saved:")
    print(f"  - {checkpoint_path}/swift_bayesflow.keras (weights)")
    print(f"  - {args.outdir}/recovery_summary.csv")
    print(f"  - {args.outdir}/figures/recovery_scatter.png")
    print(f"  - {args.outdir}/figures/loss_curve.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
