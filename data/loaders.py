"""
data/loaders.py

Dataset loading utilities.

Currently supported:
- Iris (sklearn)

Design rules:
- Dataset is loaded ONCE
- Train/test split happens ONCE
- No corruption happens here
- No randomness except controlled split
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(config: dict):
    """
    Load and preprocess dataset according to config.

    Returns:
        X_train, X_test, y_train, y_test
    """

    dataset_name = config["dataset"]["name"]

    if dataset_name != "iris":
        raise NotImplementedError(
            f"Dataset '{dataset_name}' is not implemented yet."
        )

    # --------------------------------------------------
    # Load raw Iris dataset
    # --------------------------------------------------
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    # --------------------------------------------------
    # Train / Test Split (ONCE)
    # --------------------------------------------------
    split_cfg = config["dataset"]["split"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y if split_cfg.get("stratify", False) else None
    )

    # --------------------------------------------------
    # Preprocessing (fit ONLY on train)
    # --------------------------------------------------
    prep_cfg = config["dataset"]["preprocessing"]

    if prep_cfg.get("normalize", False):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
