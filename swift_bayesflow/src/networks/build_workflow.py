"""
Phase 4 — BayesFlow architecture factory (v2: multi-sentence / participant-level).

Builds the amortized-inference workflow for the simplified SWIFT model:

    raw fixation sequences for K sentences (padded, masked, N-tagged)
            │  shape (batch, K, T, F)
            ▼
    Summary Network  (MultiSentenceSummaryNetwork: per-sentence
                       TimeSeriesNetwork encoder -> DeepSet pooling over K)
            │  -> fixed-length participant-level embedding vector h
            ▼
    Inference Network (CouplingFlow: affine coupling normalizing flow)
            │  conditioned on h
            ▼
    Approximate posterior samples over theta = (nu, r, muT, [beta], [iota])

CHANGED FROM v1: the summary network is now hierarchical
(MultiSentenceSummaryNetwork, see src/networks/multi_sentence_summary.py)
instead of a single flat TimeSeriesNetwork, because a single sentence does
not carry enough signal to recover nu and r (only muT). See that module's
docstring and the project README for the full explanation.
"""
from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import bayesflow as bf

from src.networks.bf_simulator_adapter import make_swift_simulator, make_adapter
from src.networks.multi_sentence_summary import MultiSentenceSummaryNetwork


def build_summary_network(summary_dim: int = 16) -> bf.networks.SummaryNetwork:
    """Hierarchical summary network: encodes each of the K sentences with a
    shared TimeSeriesNetwork, then pools the K per-sentence embeddings with
    a permutation-invariant DeepSet into one participant-level embedding.
    """
    return MultiSentenceSummaryNetwork(
        inner_summary_dim=summary_dim,
        outer_summary_dim=summary_dim,
        recurrent_dim=128,
    )


def build_inference_network(depth: int = 6) -> bf.networks.InferenceNetwork:
    """Affine-coupling normalizing flow amortized posterior approximator."""
    return bf.networks.CouplingFlow(
        subnet="mlp",
        depth=depth,
        transform="affine",
        permutation="random",
        use_actnorm=True,
    )


def build_workflow(
    summary_dim: int = 16,
    flow_depth: int = 6,
    initial_learning_rate: float = 5e-4,
    checkpoint_filepath: str | None = None,
) -> bf.BasicWorkflow:
    """Assembles the full SWIFT amortized-inference workflow."""
    simulator = make_swift_simulator()
    adapter = make_adapter()
    summary_net = build_summary_network(summary_dim=summary_dim)
    inference_net = build_inference_network(depth=flow_depth)

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_net,
        summary_network=summary_net,
        initial_learning_rate=initial_learning_rate,
        checkpoint_filepath=checkpoint_filepath,
        checkpoint_name="swift_bayesflow",
    )
    return workflow


if __name__ == "__main__":
    wf = build_workflow()
    print("Workflow built:", wf)
