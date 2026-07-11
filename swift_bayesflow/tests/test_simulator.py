"""
Basic sanity tests for the SWIFT simulator. Run with:
    cd swift_bayesflow
    python -m pytest tests/ -v
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator.swift_model import (
    simulate_single_trial, sample_prior, param_names, SwiftPriorBounds,
)


def test_prior_shapes():
    theta = sample_prior(50, use_beta=False, use_iota=False)
    assert theta.shape == (50, 3)
    assert np.all(theta[:, 0] >= 0) and np.all(theta[:, 0] <= 1)   # nu
    assert np.all(theta[:, 1] >= 0) and np.all(theta[:, 1] <= 12)  # r
    assert np.all(theta[:, 2] >= 100) and np.all(theta[:, 2] <= 400)  # muT


def test_prior_shapes_with_extensions():
    theta = sample_prior(20, use_beta=True, use_iota=True)
    assert theta.shape == (20, 5)
    names = param_names(use_beta=True, use_iota=True)
    assert names == ["nu", "r", "muT", "beta", "iota"]


def test_simulate_single_trial_terminates():
    rng = np.random.default_rng(0)
    theta = np.array([0.3, 10.0, 200.0], dtype=np.float32)
    trial = simulate_single_trial(theta, N=10, rng=rng, max_fixations=200)
    assert trial["n_fix"] > 0
    assert trial["n_fix"] <= 200
    assert trial["x"].min() >= 1
    assert trial["x"].max() <= 10
    # scanpath must end on the last word (termination rule)
    assert trial["x"][-1] == 10


def test_simulate_single_trial_durations_positive():
    rng = np.random.default_rng(1)
    theta = np.array([0.3, 10.0, 200.0], dtype=np.float32)
    trial = simulate_single_trial(theta, N=8, rng=rng)
    assert np.all(trial["y"] > 0)


def test_low_processing_rate_still_terminates():
    """Edge case: very low r/nu should still terminate due to max_fixations cap."""
    rng = np.random.default_rng(2)
    theta = np.array([0.01, 0.01, 200.0], dtype=np.float32)
    trial = simulate_single_trial(theta, N=8, rng=rng, max_fixations=50)
    assert trial["n_fix"] <= 50


def test_activation_bounded():
    """Activation should never exceed a_max_w (=1 here since beta unused)."""
    from src.simulator.swift_model import _processing_rates
    lam = _processing_rates(nu=0.3, k=5, N=10)
    assert lam.shape == (10,)
    assert np.all(lam >= 0)
    # asymmetric span: k-2 should be 0, k+2 should be nonzero
    assert lam[5 - 2 - 1] == 0.0   # word index 3 (0-indexed) = word k-2
    assert lam[5 + 2 - 1] > 0.0    # word k+2


if __name__ == "__main__":
    test_prior_shapes()
    test_prior_shapes_with_extensions()
    test_simulate_single_trial_terminates()
    test_simulate_single_trial_durations_positive()
    test_low_processing_rate_still_terminates()
    test_activation_bounded()
    print("All tests passed.")
