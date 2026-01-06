"""
models/inference.py

Inference utilities.

Design rules:
- NO training here
- NO data modification
- Model is assumed to be already fitted
- Outputs probabilities, predictions, confidence
"""

import numpy as np


def run_inference(model, X):
    """
    Run inference on given data using a trained model.

    Args:
        model: fitted sklearn-like model with predict_proba()
        X: input features (n_samples, n_features)

    Returns:
        probs: predicted class probabilities (n_samples, n_classes)
        preds: predicted class labels (n_samples,)
        confidence: max predicted probability per sample (n_samples,)
    """

    # --------------------------------------------------
    # Probabilistic prediction
    # --------------------------------------------------
    probs = model.predict_proba(X)

    # --------------------------------------------------
    # Hard prediction
    # --------------------------------------------------
    preds = np.argmax(probs, axis=1)

    # --------------------------------------------------
    # Confidence extraction
    # --------------------------------------------------
    confidence = np.max(probs, axis=1)

    return probs, preds, confidence
