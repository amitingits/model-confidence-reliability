"""
results/save_results.py

Save experiment results to disk.
"""

import json
from pathlib import Path


def save_jsonl(results, base_dir, filename="results.jsonl"):
    """
    Save raw experiment results to logs/ as JSONL.
    """
    log_dir = Path(base_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    path = log_dir / filename

    with open(path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print(f"[SAVED] Raw results → {path}")
