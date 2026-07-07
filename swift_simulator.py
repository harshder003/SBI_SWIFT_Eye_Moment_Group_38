import numpy as np

# ==============================================================================
# 1. The Simplified SWIFT Simulator
# ==============================================================================
# In SBI, we need a simulator that takes parameters (theta) and generates data (x).
# Here we define a highly simplified mechanistic model of eye movement.

def simplified_swift_simulator(theta, n_fixations=100):
    """
    A minimal simulator inspired by the SWIFT model.
    
    Parameters:
    - theta: A 1D numpy array of cognitive parameters.
             Let's assume a highly simplified 3-parameter model:
             theta[0] = baseline_fixation_duration (e.g., ~200 ms)
             theta[1] = variance_in_duration
             theta[2] = forward_saccade_probability (e.g., ~0.8)
             
    Returns:
    - data: A 2D array of generated fixation sequences [n_fixations, 2].
            Column 0: Fixation duration (ms)
            Column 1: Saccade target direction (1=forward, 0=refixate, -1=regression)
    """
    baseline_duration, duration_var, fwd_saccade_prob = theta
    
    # Simulate Fixation Durations (with biological clipping)
    durations = np.random.normal(loc=baseline_duration, scale=duration_var, size=n_fixations)
    durations = np.clip(durations, 50, 600) 
    
    # Simulate Saccade Decisions
    # Probabilities: forward = fwd_saccade_prob, refixation = (1-fwd)/2, regression = (1-fwd)/2
    other_prob = (1.0 - fwd_saccade_prob) / 2.0
    saccades = np.random.choice([1, 0, -1], size=n_fixations, p=[fwd_saccade_prob, other_prob, other_prob])
    
    # Stack into a dataset of shape (n_fixations, 2)
    simulated_data = np.column_stack((durations, saccades))
    return simulated_data

# ==============================================================================
# 2. Prior Definition
# ==============================================================================
# The prior defines the biologically plausible range for each parameter.

def prior_generator():
    """Generates random parameter sets (theta) for training."""
    # baseline duration uniform between 150 and 250 ms
    p1 = np.random.uniform(150, 250)
    # duration variance uniform between 20 and 50
    p2 = np.random.uniform(20, 50)
    # forward saccade probability uniform between 0.6 and 0.95
    p3 = np.random.uniform(0.6, 0.95)
    
    return np.array([p1, p2, p3])

# ==============================================================================
# Example Usage (Sanity Check)
# ==============================================================================
if __name__ == "__main__":
    print("Testing the Prior Generator...")
    sample_theta = prior_generator()
    print(f"Sample Parameters (theta):\n Baseline Duration: {sample_theta[0]:.2f}ms\n Variance: {sample_theta[1]:.2f}\n Fwd Saccade Prob: {sample_theta[2]:.2f}\n")
    
    print("Testing the Simulator...")
    sample_data = simplified_swift_simulator(sample_theta, n_fixations=5)
    print("Generated Data (Durations | Saccades):")
    print(sample_data)
    
    print("\nBoilerplate complete. Next up: Setting up BayesFlow architecture!")
