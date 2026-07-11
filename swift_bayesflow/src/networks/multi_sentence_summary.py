"""
Multi-sentence summary network — the architectural fix for the "nu, r don't
recover" problem observed in single-sentence amortization.

WHY THIS FILE EXISTS
--------------------
A single sentence (6-12 words, ~10-20 fixations) carries plenty of signal
about mu_T (duration is an i.i.d. Gamma draw repeated ~15 times -> the mean
is estimable from one trial) but almost no signal about nu, r individually
(their effect only shows up statistically across MANY skip/refixation/
regression events, which a single short sentence just doesn't contain
enough of). The real SWIFT tutorial estimates parameters from ~15,000
fixations across ~57 sentences per participant, not one sentence at a time.

THE FIX
-------
Instead of amortizing over (theta -> one sentence), we amortize over
(theta -> K sentences read by the "same reader"), exactly mirroring how a
real participant's data looks. This requires a *hierarchical* summary
network:

    (batch, K, T, F) fixation sequences
            │
            ▼  encode EACH of the K sentences independently
    inner network: bf.networks.TimeSeriesNetwork   (shared weights, applied
                                                     per-sentence)
            │  -> (batch, K, inner_summary_dim) per-sentence embeddings
            ▼  pool across the K sentences (permutation-invariant: order of
               sentences read shouldn't matter)
    outer network: bf.networks.DeepSet
            │  -> (batch, outer_summary_dim) participant-level embedding
            ▼
    fed into the inference (flow) network, same as before
"""
from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import bayesflow as bf


@keras.saving.register_keras_serializable(name="MultiSentenceSummaryNetwork")
class MultiSentenceSummaryNetwork(bf.networks.SummaryNetwork):
    """Hierarchical summary network: per-sentence TimeSeriesNetwork encoder
    + permutation-invariant DeepSet pooling across sentences.

    Expects input of shape (batch, K, T, F):
        K = number of sentences ("trials") simulated per participant/theta
        T = padded fixation-sequence length (FIX_MAX)
        F = per-timestep feature dim (word_pos_norm, dur_norm, mask, N_norm)
    Returns embedding of shape (batch, outer_summary_dim).
    """

    def __init__(
        self,
        inner_summary_dim: int = 16,
        outer_summary_dim: int = 16,
        recurrent_dim: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inner_summary_dim = inner_summary_dim
        self.outer_summary_dim = outer_summary_dim

        # per-sentence encoder (shared across all K sentences and all
        # participants -- this is what lets the network generalize across
        # a variable number of sentences K)
        self.sentence_encoder = bf.networks.TimeSeriesNetwork(
            summary_dim=inner_summary_dim,
            recurrent_dim=recurrent_dim,
            bidirectional=True,
        )

        # permutation-invariant pooling across the K per-sentence embeddings
        self.participant_pooler = bf.networks.DeepSet(
            summary_dim=outer_summary_dim,
        )

    def call(self, x, **kwargs):
        # x: (batch, K, T, F)
        shape = keras.ops.shape(x)
        batch, K, T, F = shape[0], shape[1], shape[2], shape[3]

        # --- flatten (batch, K) into one axis so the shared per-sentence
        #     encoder can be applied to all B*K sentences in one pass ---
        x_flat = keras.ops.reshape(x, (batch * K, T, F))
        sentence_embeddings_flat = self.sentence_encoder(x_flat, **kwargs)  # (B*K, inner_dim)

        # --- unflatten back into (batch, K, inner_dim) so DeepSet can pool
        #     over the K axis (treating it as a "set" of sentence embeddings) ---
        sentence_embeddings = keras.ops.reshape(
            sentence_embeddings_flat, (batch, K, self.inner_summary_dim)
        )

        participant_embedding = self.participant_pooler(sentence_embeddings, **kwargs)
        return participant_embedding
