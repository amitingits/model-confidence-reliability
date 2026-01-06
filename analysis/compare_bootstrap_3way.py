"""
analysis/compare_bootstrap_3way.py

Plot bootstrap confidence interval comparisons for
MCAR vs MAR vs MNAR.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_bootstrap(path):
    return pd.read_csv(path)


def plot_bootstrap_ci(
    dfs,
    labels,
    metric,
    ylabel,
    title,
    out_path
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()

    for df, label in zip(dfs, labels):
        x = df["missing_rate"]
        y = df[f"{metric}_mean"]
        lo = df[f"{metric}_ci_low"]
        hi = df[f"{metric}_ci_high"]

        plt.plot(x, y, marker="o", label=label)
        plt.fill_between(x, lo, hi, alpha=0.2)

    plt.xlabel("Missing Rate (p)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"[BOOTSTRAP PLOT SAVED] {out_path}")


def main():
    base_dir = Path("results/iris")

    paths = {
        "MCAR": base_dir / "mcar" / "tables" / "bootstrap_summary.csv",
        "MAR":  base_dir / "mar"  / "tables" / "bootstrap_summary.csv",
        "MNAR": base_dir / "mnar" / "tables" / "bootstrap_summary.csv",
    }

    dfs = [load_bootstrap(p) for p in paths.values()]
    labels = list(paths.keys())

    out_dir = base_dir / "comparison" / "bootstrap_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy
    plot_bootstrap_ci(
        dfs,
        labels,
        metric="accuracy",
        ylabel="Accuracy",
        title="Bootstrap Accuracy CI vs Missingness",
        out_path=out_dir / "accuracy_bootstrap_ci.png"
    )

    # ECE
    plot_bootstrap_ci(
        dfs,
        labels,
        metric="ece",
        ylabel="ECE",
        title="Bootstrap ECE CI vs Missingness",
        out_path=out_dir / "ece_bootstrap_ci.png"
    )

    # Gap
    plot_bootstrap_ci(
        dfs,
        labels,
        metric="confidence_correctness_gap",
        ylabel="Confidence–Correctness Gap",
        title="Bootstrap Gap CI vs Missingness",
        out_path=out_dir / "gap_bootstrap_ci.png"
    )


if __name__ == "__main__":
    main()
