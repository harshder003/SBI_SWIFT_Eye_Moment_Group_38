"""
Phase 2 — Data wrangling.

Parses the two OSF files referenced in the assignment:
    - Rcorpus_PB2.dat            -> per-word properties (sentence id, word id,
                                     word length, frequency, ...)
    - fixseqin_PB2expVP10.dat    -> per-fixation records for one participant
                                     (trial/sentence id, fixated word index,
                                     fixation duration)

These SWIFT-project corpus/fixation-sequence files do not have a single
universal column layout across releases, so this module is written
defensively: it inspects the file, tries a set of known/likely column
layouts, and falls back to a best-effort whitespace parse. If no file is
found at all, `load_or_synthesize()` produces a synthetic dataset with the
exact same downstream schema, generated using the simulator itself, so the
rest of the pipeline (Phases 3-5) can be built, tested, and demoed without
blocking on file availability.

Downstream schema (what everything else consumes), a list of "trials":
    trial = {
        "sentence_id": int,
        "N": int,                       # number of words in the sentence
        "x": np.ndarray[int64]  (n_fix,) # fixated word index, 1..N
        "y": np.ndarray[float32](n_fix,) # fixation duration in ms
        "log_freq": np.ndarray[float32](N,)  # normalized log word frequency q_w
    }
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Optional


# ----------------------------------------------------------------------
# Corpus parsing (word properties)
# ----------------------------------------------------------------------

def _looks_like_header(path: str) -> bool:
    """Return True when the first line contains alphabetic header names."""
    with open(path, "r", encoding="utf-8") as fh:
        first_line = fh.readline()
    return any(ch.isalpha() for ch in first_line)

def parse_corpus_file(path: str) -> pd.DataFrame:
    """Parse an R-corpus-style .dat file into a DataFrame with (at minimum)
    columns: sentence_id, word_id, word_length, freq.

    Tries several strategies since exact column names/order vary by release:
      1. whitespace-delimited with header row containing recognizable names
      2. whitespace-delimited, no header -> assign canonical names by
         position for the common 4..6 column SWIFT corpus layout
    """
    # Strategy 1: has header
    df = pd.read_csv(path, sep=r"\s+", engine="python", quotechar='"')
    df.columns = [c.strip('"') for c in df.columns]
    lower_cols = {c.lower(): c for c in df.columns}

    def find(*candidates):
        for c in candidates:
            if c in lower_cols:
                return lower_cols[c]
        return None

    sent_col = find("sentence", "sent", "sentence_id", "sentid", "sno", "isnr", "sentnr")
    word_col = find("word", "word_id", "wordid", "wno", "wordnr", "inr")
    len_col = find("length", "wordlength", "nl", "len")
    freq_col = find("freq", "frequency", "wfreq", "lfreq")

    if sent_col and word_col:
        out = pd.DataFrame({
            "sentence_id": df[sent_col].astype(int),
            "word_id": df[word_col].astype(int),
        })
        out["word_length"] = df[len_col].astype(float) if len_col else np.nan
        out["freq"] = df[freq_col].astype(float) if freq_col else np.nan
        return out

    # Strategy 2: no usable header, assume canonical SWIFT layout
    # (sentence_id, word_id, word_length, freq, ...) by column position
    raw = pd.read_csv(path, sep=r"\s+", engine="python", header=None, quotechar='"')
    ncol = raw.shape[1]
    colmap = ["sentence_id", "word_id", "word_length", "freq"][:min(4, ncol)]
    raw = raw.iloc[:, : len(colmap)]
    raw.columns = colmap
    for c in ("sentence_id", "word_id"):
        if c in raw.columns:
            raw[c] = raw[c].astype(int)
    return raw


def compute_log_freq_per_sentence(corpus_df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Implements Eq. (4): q_w = log10(F_w) / max_i log10(F_i), computed
    within each sentence (the paper normalizes by the max over the corpus;
    here we normalize per-sentence which is a safe, self-contained choice
    for a minimal model — see README for discussion of this simplification).

    Returns dict: sentence_id -> array of length N with q_w per word,
    ordered by word_id ascending.
    """
    out = {}
    for sid, g in corpus_df.groupby("sentence_id"):
        g = g.sort_values("word_id")
        freq = g["freq"].to_numpy(dtype=float)
        freq = np.where(np.isnan(freq) | (freq <= 0), 1.0, freq)  # guard log(0)
        logf = np.log10(freq)
        denom = np.max(logf) if np.max(logf) > 0 else 1.0
        out[int(sid)] = (logf / denom).astype(np.float32)
    return out


# ----------------------------------------------------------------------
# Fixation-sequence parsing
# ----------------------------------------------------------------------

def parse_fixation_file(path: str) -> pd.DataFrame:
    """Parse a fixseqin_*.dat file into long-format DataFrame with columns:
    sentence_id, fixation_index, word_id, duration (ms).

    As with the corpus file, layout is not perfectly standardized, so we
    try a header-based read first, then fall back to positional columns
    matching the common SWIFT fixation-sequence layout:
    (sentence_id, fixation_index, word_id, duration, ...)
    """
    if _looks_like_header(path):
        df = pd.read_csv(path, sep=r"\s+", engine="python", quotechar='"')
        df.columns = [c.strip('"') for c in df.columns]
        lower_cols = {c.lower(): c for c in df.columns}

        def find(*candidates):
            for c in candidates:
                if c in lower_cols:
                    return lower_cols[c]
            return None

        sent_col = find("sentence", "sent", "sentence_id", "sentid", "sno", "isnr", "trial")
        fix_col = find("fixation", "fix", "fixnr", "seq", "nfix")
        word_col = find("word", "word_id", "wordid", "wordnr", "inr", "fixword")
        dur_col = find("duration", "dur", "fdur", "time")

        if sent_col and word_col and dur_col:
            out = pd.DataFrame({
                "sentence_id": df[sent_col].astype(int),
                "word_id": df[word_col].astype(int),
                "duration": df[dur_col].astype(float),
            })
            out["fixation_index"] = (
                df[fix_col].astype(int) if fix_col else out.groupby("sentence_id").cumcount() + 1
            )
            return out[["sentence_id", "fixation_index", "word_id", "duration"]]

    raw = pd.read_csv(path, sep=r"\s+", engine="python", header=None, quotechar='"')
    ncol = raw.shape[1]
    if ncol >= 4:
        # Common SWIFT layout: sentence id, word/landing position, a metadata
        # column, then fixation duration in ms.
        raw = raw.iloc[:, [0, 1, 3]].copy()
    elif ncol == 3:
        raw = raw.iloc[:, :3].copy()
    else:
        raise ValueError(f"Expected at least 3 columns in fixation file, found {ncol}")

    raw.columns = ["sentence_id", "word_id", "duration"]
    raw["sentence_id"] = raw["sentence_id"].astype(int)
    raw["word_id"] = raw["word_id"].astype(int)
    raw["duration"] = raw["duration"].astype(float)
    raw["fixation_index"] = raw.groupby("sentence_id").cumcount() + 1
    return raw[["sentence_id", "fixation_index", "word_id", "duration"]]


def fixation_df_to_trials(
    fix_df: pd.DataFrame,
    corpus_df: Optional[pd.DataFrame] = None,
    dur_min: float = 25.0,
    dur_max: float = 1000.0,
    min_fixations: int = 3,
) -> list[dict]:
    """Group a long-format fixation DataFrame into per-sentence trials, and
    attach word-frequency information from the corpus if available.

    Applies the same basic preprocessing filters the authors describe
    (Appendix): drop fixations shorter than 25ms or longer than 1000ms,
    drop trials with fewer than 3 fixations.
    """
    log_freq_lookup = compute_log_freq_per_sentence(corpus_df) if corpus_df is not None else {}
    n_words_lookup = {}
    if corpus_df is not None:
        for sid, g in corpus_df.groupby("sentence_id"):
            n_words_lookup[int(sid)] = int(g["word_id"].max())

    trials = []
    for sid, g in fix_df.groupby("sentence_id"):
        g = g.sort_values("fixation_index")
        g = g[(g["duration"] >= dur_min) & (g["duration"] <= dur_max)]
        if len(g) < min_fixations:
            continue
        x = g["word_id"].to_numpy(dtype=np.int64)
        y = g["duration"].to_numpy(dtype=np.float32)
        N = n_words_lookup.get(int(sid), int(x.max()))
        log_freq = log_freq_lookup.get(int(sid), np.ones(N, dtype=np.float32))
        trials.append({
            "sentence_id": int(sid),
            "N": N,
            "x": x,
            "y": y,
            "log_freq": log_freq,
        })
    return trials


# ----------------------------------------------------------------------
# Top-level loader with synthetic fallback
# ----------------------------------------------------------------------

def load_or_synthesize(
    fixation_path: str,
    corpus_path: str,
    n_synthetic_trials: int = 200,
    synthetic_true_theta: Optional[np.ndarray] = None,
    seed: int = 0,
) -> tuple[list[dict], str]:
    """Load real participant data if both files exist and parse cleanly;
    otherwise synthesize a plausible dataset with the SAME schema using the
    simulator itself (so the rest of the pipeline is always exercised).

    Returns (trials, source) where source is "real" or "synthetic".
    """
    if os.path.exists(fixation_path) and os.path.exists(corpus_path):
        try:
            corpus_df = parse_corpus_file(corpus_path)
            fix_df = parse_fixation_file(fixation_path)
            trials = fixation_df_to_trials(fix_df, corpus_df)
            if len(trials) > 0:
                return trials, "real"
        except Exception as e:  # pragma: no cover
            print(f"[dataio] Failed to parse real data ({e}); falling back to synthetic.")

    # ---- synthetic fallback, built with the simulator itself ----
    from src.simulator.swift_model import simulate_single_trial

    rng = np.random.default_rng(seed)
    theta_true = (
        synthetic_true_theta
        if synthetic_true_theta is not None
        else np.array([0.28, 6.0, 235.0], dtype=np.float32)
    )
    trials = []
    for i in range(n_synthetic_trials):
        N = int(rng.integers(6, 13))  # 6-12 words, matches Risse & Seelig (2019)
        log_freq = rng.uniform(0.2, 1.0, size=N).astype(np.float32)
        trial = simulate_single_trial(theta_true, N=N, log_freq=log_freq, rng=rng)
        trials.append({
            "sentence_id": i,
            "N": N,
            "x": trial["x"],
            "y": trial["y"],
            "log_freq": log_freq,
        })
    return trials, "synthetic"


if __name__ == "__main__":
    trials, source = load_or_synthesize(
        fixation_path="data/raw/fixseqin_PB2expVP10.dat",
        corpus_path="data/raw/Rcorpus_PB2.dat",
        n_synthetic_trials=50,
    )
    print(f"source={source}, n_trials={len(trials)}")
    print("example trial:", {k: (v if np.isscalar(v) else v[:5]) for k, v in trials[0].items()})
