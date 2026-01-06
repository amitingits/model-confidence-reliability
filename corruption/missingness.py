"""
corruption/missingness.py

Missingness corruption operators.

Implemented:
- MCAR: Missing Completely At Random
- MAR:  Missing At Random (conditional on another feature)
- MNAR: Missing Not At Random (conditional on the value itself)
"""

import numpy as np


def mcar_missingness(X, p: float, seed: int = None):
    """
    MCAR: P(x_ij = NaN) = p
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")

    if seed is not None:
        np.random.seed(seed)

    X = np.asarray(X)
    X_corrupt = X.copy().astype(float)

    mask = np.random.rand(*X_corrupt.shape) < p
    X_corrupt[mask] = np.nan

    return X_corrupt


def mar_missingness(X, p: float, conditioning_feature: int, seed: int = None):
    """
    MAR: P(x_ij = NaN | x_ik) ∝ x_ik
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")

    if seed is not None:
        np.random.seed(seed)

    X = np.asarray(X)
    X_corrupt = X.copy().astype(float)

    k = conditioning_feature
    if k < 0 or k >= X.shape[1]:
        raise ValueError("Invalid conditioning feature index")

    x_k = X[:, k]
    x_k_norm = (x_k - x_k.min()) / (x_k.max() - x_k.min() + 1e-8)

    p_conditional = p * x_k_norm[:, None]

    mask = np.random.rand(*X_corrupt.shape) < p_conditional
    X_corrupt[mask] = np.nan

    return X_corrupt


def mnar_missingness(X, p: float, seed: int = None):
    """
    MNAR: Missingness depends on the value itself.

    High-magnitude values are more likely to be missing.

    Implementation:
    - Compute per-feature quantile threshold
    - Values above threshold are candidates for missingness
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")

    if seed is not None:
        np.random.seed(seed)

    X = np.asarray(X)
    X_corrupt = X.copy().astype(float)

    n_features = X.shape[1]

    for j in range(n_features):
        x_j = X[:, j]

        # Quantile threshold: higher p → more aggressive removal
        threshold = np.quantile(x_j, 1.0 - p)

        # MNAR mask: value-dependent
        mask = (x_j >= threshold) & (np.random.rand(len(x_j)) < p)

        X_corrupt[mask, j] = np.nan

    return X_corrupt
