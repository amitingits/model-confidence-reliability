"""
run_experiment.py

Reproducible experiment runner for:
Model Confidence vs Data Quality

Supports:
- MCAR missingness
- MAR missingness

Design invariants:
- Model trained ONCE
- Imputer fit ONCE on clean data
- Corruption applied ONLY to test data
- Each experiment has a unique output namespace
"""

import argparse
import yaml
import numpy as np
import random

from data.loaders import load_dataset
from models.train import train_model
from models.inference import run_inference
from corruption.missingness import mcar_missingness, mar_missingness
from corruption.imputation import FeatureImputer
from metrics.divergence import confidence_correctness_gap
from metrics.ece import expected_calibration_error
from results.save_results import save_jsonl


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


# -----------------------------
# Config loading
# -----------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------
# Core experiment
# -----------------------------
def run_experiment(config: dict):
    dataset = config["dataset"]["name"]
    mechanism = config["degradation"]["mechanism"]
    levels = config["degradation"]["levels"]
    seeds = config["experiment"]["seeds"]

    experiment_name = config["output"]["experiment_name"]
    base_output_dir = f"{config['output']['directory']}/{experiment_name}"

    print(f"[INFO] Dataset    : {dataset}")
    print(f"[INFO] Mechanism  : {mechanism}")
    print(f"[INFO] Experiment : {experiment_name}")
    print(f"[INFO] Seeds      : {seeds}")
    print(f"[INFO] Levels     : {levels}")

    # -----------------------------
    # STEP 1: Load clean data
    # -----------------------------
    X_train, X_test, y_train, y_test = load_dataset(config)

    # -----------------------------
    # STEP 2: Train model (ONCE)
    # -----------------------------
    model = train_model(X_train, y_train, config)

    # -----------------------------
    # STEP 3: Fit imputer (ONCE)
    # -----------------------------
    imputer = FeatureImputer(strategy="mean")
    imputer.fit(X_train)

    results = []

    # -----------------------------
    # STEP 4: Degradation sweep
    # -----------------------------
    for seed in seeds:
        set_seed(seed)

        for d in levels:
            print(f"[RUN] {mechanism} | seed={seed} | p={d}")

            # -------------------------
            # Apply missingness
            # -------------------------
            if mechanism == "MCAR":
                X_corrupt = mcar_missingness(
                        X_test,
                        p=d,
                        seed=seed
                    )
            elif mechanism == "MAR":
                k = config["degradation"]["mar"]["conditioning_feature"]
                X_corrupt = mar_missingness(
                X_test,
                p=d,
                conditioning_feature=k,
                seed=seed
            )

            elif mechanism == "MNAR":
                from corruption.missingness import mnar_missingness
                X_corrupt = mnar_missingness(
                    X_test,
                    p=d,
                    seed=seed
                )

            else:
                raise ValueError(f"Unknown mechanism: {mechanism}")

            # -------------------------
            # Imputation
            # -------------------------
            X_imputed = imputer.transform(X_corrupt)

            # -------------------------
            # Inference
            # -------------------------
            probs, preds, confidence = run_inference(
                model,
                X_imputed
            )

            correctness = (preds == y_test).astype(int)

            # -------------------------
            # Metrics
            # -------------------------
            acc = float(correctness.mean())
            ece = float(
                expected_calibration_error(confidence, correctness)
            )
            gap = float(
                confidence_correctness_gap(confidence, correctness)
            )

            results.append({
                "dataset": dataset,
                "experiment": experiment_name,
                "mechanism": mechanism,
                "seed": seed,
                "missing_rate": d,
                "accuracy": acc,
                "ece": ece,
                "confidence_correctness_gap": gap,
            })

    # -----------------------------
    # Save raw results
    # -----------------------------
    save_jsonl(results, base_output_dir)

    return results


# -----------------------------
# Entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run reliability experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    run_experiment(config)


if __name__ == "__main__":
    main()
