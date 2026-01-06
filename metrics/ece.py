"""
metrics/ece.py

Expected Calibration Error (ECE).

Definition:
ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|

where:
- B_m is the set of samples whose confidence falls into bin m
- acc(B_m) is empirical accuracy in bin m
- conf(B_m) is mean confidence in bin m
"""

import numpy as np


def expected_calibration_error(
    confidence,
    correctness,
    n_bins: int = 10
):
    """
    Compute Expected Calibration Error (ECE).

    Args:
        confidence (np.ndarray): shape (n_samples,)
            Model confidence per sample, c_i in [0, 1]

        correctness (np.ndarray): shape (n_samples,)
            Correctness indicator per sample, z_i ∈ {0, 1}

        n_bins (int): number of confidence bins

    Returns:
        ece (float)
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

    if not np.all((confidence >= 0) & (confidence <= 1)):
        raise ValueError("Confidence values must be in [0, 1]")

    n = len(confidence)

    # --------------------------------------------------
    # Define bin boundaries (fixed-width)
    # --------------------------------------------------
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0

    # --------------------------------------------------
    # Compute ECE
    # --------------------------------------------------
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        # Include right edge only for last bin
        if i == n_bins - 1:
            mask = (confidence >= bin_lower) & (confidence <= bin_upper)
        else:
            mask = (confidence >= bin_lower) & (confidence < bin_upper)

        bin_size = mask.sum()

        if bin_size == 0:
            continue

        bin_confidence = confidence[mask].mean()
        bin_accuracy = correctness[mask].mean()

        ece += (bin_size / n) * abs(bin_accuracy - bin_confidence)

    return ece
