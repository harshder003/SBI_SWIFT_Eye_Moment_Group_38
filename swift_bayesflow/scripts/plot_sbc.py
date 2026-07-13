import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import matplotlib
matplotlib.use("Agg")
import keras

import bayesflow.diagnostics as bf_diag
from src.networks.multi_sentence_summary import MultiSentenceSummaryNetwork
from src.networks.bf_simulator_adapter import PARAM_NAMES
from src.diagnostics.diagnostics import parameter_recovery_arrays

class _WF:
    def __init__(self, approx):
        self.approx = approx
    def sample(self, num_samples, conditions):
        return self.approx.sample(num_samples=num_samples, conditions=conditions)

def main():
    checkpoint_path = os.path.join("outputs", "checkpoints", "swift_bayesflow.keras")
    print(f"Loading trained approximator from {checkpoint_path} ...")
    approximator = keras.saving.load_model(checkpoint_path, custom_objects={'MultiSentenceSummaryNetwork': MultiSentenceSummaryNetwork})
    workflow = _WF(approximator)
    
    print("Generating simulation-based calibration data (this may take a minute)...")
    estimates, targets = parameter_recovery_arrays(
        workflow, n_test=100, n_posterior_samples=300
    )
    
    print("Creating SBC ECDF plot...")
    fig_ecdf = bf_diag.calibration_ecdf(estimates, targets, variable_names=PARAM_NAMES)
    ecdf_path = os.path.join("outputs", "figures", "sbc_ecdf.png")
    fig_ecdf.savefig(ecdf_path, dpi=150)
    print(f"Saved {ecdf_path}")
    
    print("Creating SBC Coverage plot...")
    fig_cov = bf_diag.coverage(estimates, targets, variable_names=PARAM_NAMES)
    cov_path = os.path.join("outputs", "figures", "sbc_coverage.png")
    fig_cov.savefig(cov_path, dpi=150)
    print(f"Saved {cov_path}")

if __name__ == "__main__":
    main()
