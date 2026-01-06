"""
analysis/plot_and_table.py

Generate tables and plots from experiment results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


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


def save_table(df, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[TABLE SAVED] {out_path}")


def plot_metric(df, x, y, yerr, ylabel, title, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.errorbar(
        df[x],
        df[y],
        yerr=df[yerr],
        marker="o",
        capsize=4
    )
    plt.xlabel("Missing Rate (p)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"[PLOT SAVED] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (e.g. mcar, mar)"
    )
    args = parser.parse_args()

    base_dir = Path(f"results/iris/{args.experiment}")

    log_path = base_dir / "logs" / "results.jsonl"
    table_dir = base_dir / "tables"
    plot_dir = base_dir / "plots"

    raw = load_results(log_path)
    df = pd.DataFrame(raw)

    agg = aggregate(df)

    # ------------------
    # Save table
    # ------------------
    save_table(
        agg,
        table_dir / "summary_table.csv"
    )

    # ------------------
    # Plots
    # ------------------
    plot_metric(
        agg,
        "missing_rate",
        "accuracy_mean",
        "accuracy_std",
        "Accuracy",
        "Accuracy vs Missingness",
        plot_dir / "accuracy_vs_missingness.png"
    )

    plot_metric(
        agg,
        "missing_rate",
        "ece_mean",
        "ece_std",
        "ECE",
        "ECE vs Missingness",
        plot_dir / "ece_vs_missingness.png"
    )

    plot_metric(
        agg,
        "missing_rate",
        "gap_mean",
        "gap_std",
        "Confidence–Correctness Gap",
        "Gap vs Missingness",
        plot_dir / "gap_vs_missingness.png"
    )


if __name__ == "__main__":
    main()
