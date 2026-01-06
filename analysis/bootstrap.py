"""
analysis/bootstrap.py

Bootstrap confidence intervals for reliability metrics.
"""

import numpy as np
import pandas as pd


def bootstrap_ci(
    values,
    n_bootstrap=1000,
    ci=0.95,
    random_state=0
):
    """
    Compute bootstrap confidence interval.

    Args:
        values (array-like): metric values
        n_bootstrap (int): number of bootstrap resamples
        ci (float): confidence level
        random_state (int): RNG seed

    Returns:
        (mean, lower, upper)
    """
    rng = np.random.default_rng(random_state)
    values = np.asarray(values)

    boot_means = []
    n = len(values)

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)

    alpha = (1.0 - ci) / 2.0
    lower = np.quantile(boot_means, alpha)
    upper = np.quantile(boot_means, 1 - alpha)

    return values.mean(), lower, upper


def bootstrap_grouped(
    df,
    group_col,
    metric_cols,
    n_bootstrap=1000
):
    """
    Apply bootstrap CI per group.

    Args:
        df (DataFrame)
        group_col (str): column to group by (e.g. missing_rate)
        metric_cols (list): metrics to bootstrap

    Returns:
        DataFrame with mean and CI columns
    """
    rows = []

    for g, gdf in df.groupby(group_col):
        row = {group_col: g}

        for m in metric_cols:
            mean, lo, hi = bootstrap_ci(
                gdf[m].values,
                n_bootstrap=n_bootstrap
            )
            row[f"{m}_mean"] = mean
            row[f"{m}_ci_low"] = lo
            row[f"{m}_ci_high"] = hi

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_col)
