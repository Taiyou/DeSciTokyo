"""Result aggregation and pivot table generation."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def aggregate_by_condition(
    df: pd.DataFrame,
    group_cols: List[str] | None = None,
    metric: str = "net_output",
) -> pd.DataFrame:
    """
    Aggregate results by condition, computing mean, std, CI.

    Default grouping: (scenario, strategy, alpha)
    """
    if group_cols is None:
        group_cols = ["scenario", "strategy", "alpha"]

    agg = df.groupby(group_cols)[metric].agg(
        ["mean", "std", "count", "median"]
    ).reset_index()

    # 95% confidence interval
    agg["ci_95"] = 1.96 * agg["std"] / np.sqrt(agg["count"])
    agg["ci_lower"] = agg["mean"] - agg["ci_95"]
    agg["ci_upper"] = agg["mean"] + agg["ci_95"]

    return agg


def pivot_alpha_strategy(
    df: pd.DataFrame,
    scenario: str,
    metric: str = "net_output",
) -> pd.DataFrame:
    """Create a pivot table: alpha (rows) x strategy (columns)."""
    filtered = df[df["scenario"] == scenario]
    agg = aggregate_by_condition(filtered, metric=metric)
    pivot = agg.pivot_table(
        index="alpha",
        columns="strategy",
        values="mean",
    )
    return pivot


def compute_relative_improvement(
    df: pd.DataFrame,
    baseline_strategy: str = "baseline",
    metric: str = "net_output",
) -> pd.DataFrame:
    """Compute relative improvement of each strategy over baseline."""
    agg = aggregate_by_condition(df, metric=metric)

    baseline_means = agg[agg["strategy"] == baseline_strategy][
        ["scenario", "alpha", "mean"]
    ].rename(columns={"mean": "baseline_mean"})

    merged = agg.merge(baseline_means, on=["scenario", "alpha"], how="left")
    merged["relative_improvement"] = (
        (merged["mean"] - merged["baseline_mean"]) / merged["baseline_mean"].abs().clip(lower=0.001)
    )

    return merged


def strategy_ranking_table(
    df: pd.DataFrame,
    scenario: str,
    metric: str = "net_output",
) -> pd.DataFrame:
    """Rank strategies at each alpha level for a given scenario."""
    filtered = df[df["scenario"] == scenario]
    agg = aggregate_by_condition(filtered, metric=metric)

    rankings = []
    for alpha in sorted(agg["alpha"].unique()):
        alpha_data = agg[agg["alpha"] == alpha].sort_values("mean", ascending=False)
        for rank, (_, row) in enumerate(alpha_data.iterrows(), 1):
            rankings.append({
                "alpha": alpha,
                "strategy": row["strategy"],
                "mean": row["mean"],
                "rank": rank,
            })

    return pd.DataFrame(rankings)
