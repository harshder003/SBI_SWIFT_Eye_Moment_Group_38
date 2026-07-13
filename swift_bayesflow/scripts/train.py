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
import torch

from src.networks.build_workflow import build_workflow
from src.networks.bf_simulator_adapter import PARAM_NAMES
from src.diagnostics.diagnostics import (
    parameter_recovery, recovery_summary, parameter_recovery_arrays,
)
import bayesflow as bf  # NEW: needed for bf.diagnostics.calibration_ecdf / coverage


def setup_device():
    if torch.cuda.is_available():
        print("CUDA is available! Setting PyTorch to use GPU.")
        torch.set_default_device("cuda")
    else:
        print("CUDA is NOT available. PyTorch will use CPU.")


def parse_args():
    p = argparse.ArgumentParser(description="Train the SWIFT BayesFlow amortizer")
    p.add_argument("--epochs", type=int, default=80)          # raised from 20 - early stopping cuts it short
    p.add_argument("--batches-per-epoch", type=int, default=25)  # keep at 20-25, not 100 (too slow/epoch)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--summary-dim", type=int, default=16)
    p.add_argument("--flow-depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--n-recovery-cases", type=int, default=250)      # was 150 - less noisy diagnostics
    p.add_argument("--n-posterior-samples", type=int, default=1000)  # was 500
    p.add_argument("--val-participants", type=int, default=200)      # held-out validation batch size
    p.add_argument("--patience", type=int, default=15)               # early stopping patience (epochs)
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

    setup_device()

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
          f"x batch_size={args.batch_size}  (early stopping patience={args.patience})")
    print("=" * 70)
    import keras
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True,
    )

    t0 = time.time()
    val_loss_curve = np.array([])
    try:
        history = workflow.fit_online(
            epochs=args.epochs,
            num_batches_per_epoch=args.batches_per_epoch,
            batch_size=args.batch_size,
            validation_data=args.val_participants,   # int -> auto-generates held-out val batch
            callbacks=[early_stop],
        )
        loss_curve = history.history.get("loss", [])
        val_loss_curve = np.asarray(history.history.get("val_loss", []))
        print(f"Training completed in {time.time() - t0:.1f}s "
              f"({len(loss_curve)} epochs actually run)")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Stopping gracefully...")
        loss_curve = []
        if hasattr(workflow, "approximator") and hasattr(workflow.approximator, "history"):
            if workflow.approximator.history is not None:
                loss_curve = workflow.approximator.history.history.get("loss", [])
                val_loss_curve = np.asarray(
                    workflow.approximator.history.history.get("val_loss", [])
                )
        # Diagnostics below run on the weights as of the interrupted epoch
        # (or the best-so-far weights if EarlyStopping's restore_best_weights
        # already kicked in before the interrupt).
        print(f"Training stopped after {time.time() - t0:.1f}s")

    loss_curve = np.asarray(loss_curve)
    if len(loss_curve) > 0:
        np.save(os.path.join(args.outdir, "loss_curve.npy"), loss_curve)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(loss_curve, label="train")
        if len(val_loss_curve) > 0:
            np.save(os.path.join(args.outdir, "val_loss_curve.npy"), val_loss_curve)
            ax.plot(val_loss_curve, label="val")
            ax.legend()
        ax.set_xlabel("epoch")
        ax.set_ylabel("negative log-likelihood (loss)")
        ax.set_title("Training loss")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "figures", "loss_curve.png"), dpi=150)
        plt.close(fig)
    else:
        print("No completed epochs to plot loss curve.")

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

    # ---- NEW: Simulation-based calibration (SBC) - ECDF + coverage plots ----
    print("=" * 70)
    print("Simulation-based calibration (SBC): ECDF + coverage plots")
    print("=" * 70)
    # Reuses the same simulation-based-calibration setup as parameter_recovery()
    # above, but keeps the RAW posterior draws (rather than just summary
    # stats) in the (n_test, n_draws, n_params) / (n_test, n_params) shape
    # BayesFlow's built-in diagnostics expect.
    estimates, targets = parameter_recovery_arrays(
        workflow, n_test=args.n_recovery_cases,
        n_posterior_samples=args.n_posterior_samples, seed=args.seed + 2,
    )

    fig_ecdf = bf.diagnostics.calibration_ecdf(
        estimates, targets, variable_names=PARAM_NAMES, figsize=(4 * len(PARAM_NAMES), 4),
    )
    fig_ecdf.savefig(os.path.join(args.outdir, "figures", "sbc_ecdf.png"), dpi=150)
    plt.close(fig_ecdf)

    fig_cov = bf.diagnostics.coverage(
        estimates, targets, variable_names=PARAM_NAMES, figsize=(4 * len(PARAM_NAMES), 4),
    )
    fig_cov.savefig(os.path.join(args.outdir, "figures", "sbc_coverage.png"), dpi=150)
    plt.close(fig_cov)
    # ---- END NEW ----

    print("=" * 70)
    print("Saved:")
    print(f"  - {checkpoint_path}/swift_bayesflow.keras (weights)")
    print(f"  - {args.outdir}/recovery_summary.csv")
    print(f"  - {args.outdir}/figures/recovery_scatter.png")
    print(f"  - {args.outdir}/figures/loss_curve.png")
    print(f"  - {args.outdir}/figures/sbc_ecdf.png")
    print(f"  - {args.outdir}/figures/sbc_coverage.png")
    print("=" * 70)

if __name__ == "__main__":
    main()
