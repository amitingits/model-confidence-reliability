"""
analysis/bootstrap_tables.py

Generate bootstrap confidence interval tables for an experiment.
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from bootstrap import bootstrap_grouped


def load_results(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (mcar, mar, mnar)"
    )
    args = parser.parse_args()

    base_dir = Path(f"results/iris/{args.experiment}")
    log_path = base_dir / "logs" / "results.jsonl"
    out_dir = base_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(load_results(log_path))

    metrics = ["accuracy", "ece", "confidence_correctness_gap"]

    boot = bootstrap_grouped(
        df,
        group_col="missing_rate",
        metric_cols=metrics,
        n_bootstrap=1000
    )

    out_path = out_dir / "bootstrap_summary.csv"
    boot.to_csv(out_path, index=False)

    print(f"[BOOTSTRAP TABLE SAVED] {out_path}")


if __name__ == "__main__":
    main()
