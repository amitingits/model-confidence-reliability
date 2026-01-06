"""
metrics/divergence.py

Confidence–Correctness divergence metrics.

Primary metric:
Δ = E[c] - E[z]

where:
- c = model confidence (max predicted probability)
- z = correctness indicator (1 if correct, 0 otherwise)
"""

import numpy as np


def confidence_correctness_gap(confidence, correctness):
    """
    Compute the confidence–correctness gap.

    Args:
        confidence (np.ndarray): shape (n_samples,)
            Model confidence per sample, c_i in [0, 1]

        correctness (np.ndarray): shape (n_samples,)
            Correctness indicator per sample, z_i ∈ {0, 1}

    Returns:
        gap (float):
            Δ = E[c] - E[z]

    Interpretation:
        Δ > 0  → overconfidence
        Δ = 0  → perfect reliability
        Δ < 0  → underconfidence
    """

    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness)

    if confidence.shape != correctness.shape:
        raise ValueError(
            f"Shape mismatch: confidence {confidence.shape}, "
            f"correctness {correctness.shape}"
        )

    if confidence.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")

    mean_confidence = confidence.mean()
    mean_correctness = correctness.mean()

    gap = mean_confidence - mean_correctness

    return gap
