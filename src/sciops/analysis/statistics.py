"""Statistical tests for strategy comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PairwiseResult:
    """Result of pairwise strategy comparison."""

    strategy_a: str
    strategy_b: str
    mean_a: float
    mean_b: float
    t_statistic: float
    p_value: float
    cohens_d: float
    significant: bool


@dataclass
class MultipleComparisonResult:
    """Result of Kruskal-Wallis + post-hoc Dunn's test."""

    h_statistic: float
    p_value: float
    pairwise_p_values: Dict[Tuple[str, str], float]
    strategy_ranks: Dict[str, float]


def compare_two_strategies(
    df: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
    metric: str = "net_output",
    alpha_filter: float | None = None,
    scenario_filter: str | None = None,
    significance_level: float = 0.05,
) -> PairwiseResult:
    """Welch's t-test + Cohen's d for two strategies."""
    mask = pd.Series(True, index=df.index)
    if alpha_filter is not None:
        mask &= df["alpha"] == alpha_filter
    if scenario_filter is not None:
        mask &= df["scenario"] == scenario_filter

    a = df.loc[mask & (df["strategy"] == strategy_a), metric].values
    b = df.loc[mask & (df["strategy"] == strategy_b), metric].values

    if len(a) == 0 or len(b) == 0:
        raise ValueError(f"No data for comparison: {strategy_a} ({len(a)}), {strategy_b} ({len(b)})")

    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)

    # Cohen's d
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

    return PairwiseResult(
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        t_statistic=float(t_stat),
        p_value=float(p_val),
        cohens_d=float(d),
        significant=float(p_val) < significance_level,
    )


def compare_all_strategies(
    df: pd.DataFrame,
    metric: str = "net_output",
    alpha_filter: float | None = None,
    scenario_filter: str | None = None,
) -> MultipleComparisonResult:
    """Kruskal-Wallis test across all strategies + pairwise post-hoc."""
    mask = pd.Series(True, index=df.index)
    if alpha_filter is not None:
        mask &= df["alpha"] == alpha_filter
    if scenario_filter is not None:
        mask &= df["scenario"] == scenario_filter

    filtered = df.loc[mask]
    strategy_names = sorted(filtered["strategy"].unique())
    groups = [filtered.loc[filtered["strategy"] == s, metric].values for s in strategy_names]

    # Kruskal-Wallis
    h_stat, p_val = stats.kruskal(*groups)

    # Pairwise Mann-Whitney U with Bonferroni correction
    n_comparisons = len(strategy_names) * (len(strategy_names) - 1) // 2
    pairwise_p: Dict[Tuple[str, str], float] = {}

    for i in range(len(strategy_names)):
        for j in range(i + 1, len(strategy_names)):
            _, p = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
            corrected_p = min(1.0, p * n_comparisons)  # Bonferroni
            pairwise_p[(strategy_names[i], strategy_names[j])] = corrected_p

    # Mean ranks
    all_values = filtered[metric].values
    ranks = stats.rankdata(all_values)
    filtered = filtered.copy()
    filtered["_rank"] = ranks
    strategy_ranks = {
        s: float(filtered.loc[filtered["strategy"] == s, "_rank"].mean())
        for s in strategy_names
    }

    return MultipleComparisonResult(
        h_statistic=float(h_stat),
        p_value=float(p_val),
        pairwise_p_values=pairwise_p,
        strategy_ranks=strategy_ranks,
    )


def rank_stability(
    df: pd.DataFrame,
    metric: str = "net_output",
    scenario_filter: str | None = None,
) -> float:
    """
    Kendall's W concordance coefficient across seeds.

    Measures how consistently strategies rank across different random seeds.
    W=1 means perfect agreement, W=0 means random.
    """
    mask = pd.Series(True, index=df.index)
    if scenario_filter is not None:
        mask &= df["scenario"] == scenario_filter

    filtered = df.loc[mask]
    strategies = sorted(filtered["strategy"].unique())
    seeds = sorted(filtered["seed"].unique())

    # Build rank matrix: seeds x strategies
    rank_matrix = np.zeros((len(seeds), len(strategies)))
    for i, seed in enumerate(seeds):
        seed_data = filtered[filtered["seed"] == seed]
        means = []
        for s in strategies:
            val = seed_data.loc[seed_data["strategy"] == s, metric].mean()
            means.append(val)
        rank_matrix[i] = stats.rankdata([-m for m in means])  # Higher is better

    # Kendall's W
    n = len(seeds)
    k = len(strategies)
    if n < 2 or k < 2:
        return 1.0

    mean_ranks = rank_matrix.mean(axis=0)
    ss = n * np.sum((mean_ranks - mean_ranks.mean()) ** 2)
    w = 12 * ss / (k**2 * (n**2 - 1) * n)
    return float(np.clip(w, 0.0, 1.0))


def detect_phase_transition(
    df: pd.DataFrame,
    metric: str = "net_output",
    scenario_filter: str | None = None,
) -> Dict[str, object]:
    """
    Detect strategy ranking change-points as a function of alpha.

    Returns the alpha value(s) where the best strategy switches.
    """
    mask = pd.Series(True, index=df.index)
    if scenario_filter is not None:
        mask &= df["scenario"] == scenario_filter

    filtered = df.loc[mask]
    alphas = sorted(filtered["alpha"].unique())
    strategies = sorted(filtered["strategy"].unique())

    best_at_alpha: List[Tuple[float, str]] = []
    for alpha in alphas:
        alpha_data = filtered[filtered["alpha"] == alpha]
        best_strategy = None
        best_mean = -float("inf")
        for s in strategies:
            s_data = alpha_data[alpha_data["strategy"] == s]
            if len(s_data) > 0:
                mean = s_data[metric].mean()
                if mean > best_mean:
                    best_mean = mean
                    best_strategy = s
        if best_strategy:
            best_at_alpha.append((alpha, best_strategy))

    # Find transitions
    transitions: List[Dict] = []
    for i in range(1, len(best_at_alpha)):
        if best_at_alpha[i][1] != best_at_alpha[i - 1][1]:
            transitions.append({
                "alpha": best_at_alpha[i][0],
                "from_strategy": best_at_alpha[i - 1][1],
                "to_strategy": best_at_alpha[i][1],
            })

    return {
        "best_at_alpha": best_at_alpha,
        "transitions": transitions,
        "num_transitions": len(transitions),
    }
