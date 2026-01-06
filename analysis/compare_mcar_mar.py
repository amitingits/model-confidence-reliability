"""
analysis/compare_mcar_mar.py

Generate comparison plots between MCAR and MAR experiments.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def aggregate(df):
    return (
        df.groupby("missing_rate")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            gap_mean=("confidence_correctness_gap", "mean"),
            gap_std=("confidence_correctness_gap", "std"),
        )
        .reset_index()
    )


def plot_comparison(
    df_a,
    df_b,
    label_a,
    label_b,
    x,
    y,
    yerr,
    ylabel,
    title,
    out_path
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.errorbar(
        df_a[x],
        df_a[y],
        yerr=df_a[yerr],
        marker="o",
        capsize=4,
        label=label_a
    )
    plt.errorbar(
        df_b[x],
        df_b[y],
        yerr=df_b[yerr],
        marker="s",
        capsize=4,
        label=label_b
    )

    plt.xlabel("Missing Rate (p)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"[PLOT SAVED] {out_path}")


def main():
    base_dir = Path("results/iris")
    out_dir = base_dir / "comparison" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load MCAR
    mcar_path = base_dir / "mcar" / "logs" / "results.jsonl"
    mar_path = base_dir / "mar" / "logs" / "results.jsonl"

    df_mcar = pd.DataFrame(load_results(mcar_path))
    df_mar = pd.DataFrame(load_results(mar_path))

    agg_mcar = aggregate(df_mcar)
    agg_mar = aggregate(df_mar)

    # ------------------
    # Accuracy
    # ------------------
    plot_comparison(
        agg_mcar,
        agg_mar,
        "MCAR",
        "MAR",
        "missing_rate",
        "accuracy_mean",
        "accuracy_std",
        "Accuracy",
        "Accuracy vs Missingness (MCAR vs MAR)",
        out_dir / "accuracy_comparison.png"
    )

    # ------------------
    # ECE
    # ------------------
    plot_comparison(
        agg_mcar,
        agg_mar,
        "MCAR",
        "MAR",
        "missing_rate",
        "ece_mean",
        "ece_std",
        "ECE",
        "ECE vs Missingness (MCAR vs MAR)",
        out_dir / "ece_comparison.png"
    )

    # ------------------
    # Confidence–Correctness Gap
    # ------------------
    plot_comparison(
        agg_mcar,
        agg_mar,
        "MCAR",
        "MAR",
        "missing_rate",
        "gap_mean",
        "gap_std",
        "Confidence–Correctness Gap",
        "Gap vs Missingness (MCAR vs MAR)",
        out_dir / "gap_comparison.png"
    )


if __name__ == "__main__":
    main()