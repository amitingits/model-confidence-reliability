"""
corruption/imputation.py

Missing value handling utilities.

Design rules:
- Imputer is fit ONLY on training data
- Test data is transformed using fitted imputer
- No labels are ever accessed
"""

import numpy as np
from sklearn.impute import SimpleImputer


class FeatureImputer:
    """
    Feature-wise imputer for handling missing values.
    """

    def __init__(self, strategy: str = "mean"):
        if strategy not in ("mean", "median", "most_frequent"):
            raise ValueError(
                "Imputation strategy must be one of: "
                "'mean', 'median', 'most_frequent'"
            )

        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)
        self._is_fitted = False

    def fit(self, X_train):
        """
        Fit imputer on clean training data ONLY.
        """
        X_train = np.asarray(X_train)

        if np.isnan(X_train).any():
            raise ValueError(
                "Training data contains NaNs. "
                "Imputer must be fit on clean data only."
            )

        self.imputer.fit(X_train)
        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Apply imputation to corrupted data.
        """
        if not self._is_fitted:
            raise RuntimeError("Imputer must be fit before transform")

        X = np.asarray(X)
        return self.imputer.transform(X)

    def fit_transform(self, X_train):
        """
        Convenience method (fit + transform).
        """
        return self.fit(X_train).transform(X_train)
