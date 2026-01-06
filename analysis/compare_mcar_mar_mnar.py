"""
analysis/compare_mcar_mar_mnar.py

Generate comparison plots between:
- MCAR
- MAR
- MNAR
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


def plot_three(
    dfs,
    labels,
    x,
    y,
    yerr,
    ylabel,
    title,
    out_path
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    markers = ["o", "s", "^"]

    plt.figure()
    for df, label, marker in zip(dfs, labels, markers):
        plt.errorbar(
            df[x],
            df[y],
            yerr=df[yerr],
            marker=marker,
            capsize=4,
            label=label
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

    paths = {
        "MCAR": base_dir / "mcar" / "logs" / "results.jsonl",
        "MAR":  base_dir / "mar"  / "logs" / "results.jsonl",
        "MNAR": base_dir / "mnar" / "logs" / "results.jsonl",
    }

    aggregates = {}
    for name, path in paths.items():
        df = pd.DataFrame(load_results(path))
        aggregates[name] = aggregate(df)

    dfs = [aggregates["MCAR"], aggregates["MAR"], aggregates["MNAR"]]
    labels = ["MCAR", "MAR", "MNAR"]

    # ------------------
    # Accuracy
    # ------------------
    plot_three(
        dfs,
        labels,
        "missing_rate",
        "accuracy_mean",
        "accuracy_std",
        "Accuracy",
        "Accuracy vs Missingness (MCAR vs MAR vs MNAR)",
        out_dir / "accuracy_comparison_3way.png"
    )

    # ------------------
    # ECE
    # ------------------
    plot_three(
        dfs,
        labels,
        "missing_rate",
        "ece_mean",
        "ece_std",
        "ECE",
        "ECE vs Missingness (MCAR vs MAR vs MNAR)",
        out_dir / "ece_comparison_3way.png"
    )

    # ------------------
    # Gap
    # ------------------
    plot_three(
        dfs,
        labels,
        "missing_rate",
        "gap_mean",
        "gap_std",
        "Confidence–Correctness Gap",
        "Gap vs Missingness (MCAR vs MAR vs MNAR)",
        out_dir / "gap_comparison_3way.png"
    )


if __name__ == "__main__":
    main()
