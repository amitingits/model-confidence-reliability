"""
models/train.py

Model training utilities.

Currently supported:
- Logistic Regression (sklearn)

Design rules:
- Train ONCE on clean data
- Return a fitted probabilistic model
- No evaluation here
"""

from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train, config: dict):
    """
    Train a model according to config.

    Returns:
        fitted model with predict_proba()
    """

    model_cfg = config["model"]
    model_type = model_cfg["type"]

    if model_type != "logistic_regression":
        raise NotImplementedError(
            f"Model '{model_type}' is not implemented yet."
        )

    params = model_cfg["hyperparameters"]

    model = LogisticRegression(
        penalty=params.get("penalty", "l2"),
        C=params.get("C", 1.0),
        solver=params.get("solver", "lbfgs"),
        max_iter=params.get("max_iter", 1000),
        random_state=0
    )

    model.fit(X_train, y_train)

    return model
