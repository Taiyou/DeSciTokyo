"""Visualization functions for simulation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sciops.analysis.aggregation import (
    aggregate_by_condition,
    pivot_alpha_strategy,
    strategy_ranking_table,
)

# Consistent color palette for strategies
STRATEGY_COLORS = {
    "baseline": "#9E9E9E",
    "toc_pdca": "#2196F3",
    "kanban": "#4CAF50",
    "agile": "#FF9800",
    "ai_assisted": "#9C27B0",
    "ai_sciops": "#E91E63",
    "oracle": "#F44336",
}

STRATEGY_ORDER = [
    "baseline", "toc_pdca", "kanban", "agile",
    "ai_assisted", "ai_sciops", "oracle",
]


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150


def plot_strategy_comparison(
    df: pd.DataFrame,
    scenario: str,
    alpha: float = 0.0,
    metric: str = "net_output",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart comparing strategies at a fixed alpha (RQ1/RQ2)."""
    _setup_style()
    filtered = df[(df["scenario"] == scenario) & (df["alpha"] == alpha)]
    agg = aggregate_by_condition(filtered, group_cols=["strategy"], metric=metric)

    # Sort by STRATEGY_ORDER
    agg["order"] = agg["strategy"].map(
        {s: i for i, s in enumerate(STRATEGY_ORDER)}
    )
    agg = agg.sort_values("order")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [STRATEGY_COLORS.get(s, "#666") for s in agg["strategy"]]

    bars = ax.bar(agg["strategy"], agg["mean"], yerr=agg["ci_95"], capsize=4, color=colors)
    ax.set_xlabel("Strategy")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Strategy Comparison: {scenario} (alpha={alpha})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_alpha_heatmap(
    df: pd.DataFrame,
    scenario: str,
    metric: str = "net_output",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap: alpha (rows) x strategy (columns) for RQ3."""
    _setup_style()
    pivot = pivot_alpha_strategy(df, scenario, metric)

    # Reorder columns
    ordered_cols = [s for s in STRATEGY_ORDER if s in pivot.columns]
    pivot = pivot[ordered_cols]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": metric.replace("_", " ").title()},
    )
    ax.set_title(f"Performance Heatmap: {scenario}")
    ax.set_ylabel("AI Capability (alpha)")
    ax.set_xlabel("Strategy")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_phase_diagram(
    df: pd.DataFrame,
    scenario: str,
    metric: str = "net_output",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Strategy ranking vs alpha — shows phase transitions (RQ3)."""
    _setup_style()
    rankings = strategy_ranking_table(df, scenario, metric)

    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy in STRATEGY_ORDER:
        s_data = rankings[rankings["strategy"] == strategy]
        if not s_data.empty:
            ax.plot(
                s_data["alpha"],
                s_data["mean"],
                marker="o",
                markersize=4,
                label=strategy,
                color=STRATEGY_COLORS.get(strategy, "#666"),
                linewidth=2,
            )

    ax.set_xlabel("AI Capability (alpha)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Strategy Performance vs AI Capability: {scenario}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_challenge_amplification(
    df: pd.DataFrame,
    scenario: str,
    strategy: str = "ai_sciops",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Challenge metrics vs alpha — shows amplification curves (RQ4)."""
    _setup_style()
    filtered = df[(df["scenario"] == scenario) & (df["strategy"] == strategy)]

    metrics = ["quality_drift", "quality_true_mean", "total_management_overhead"]
    metric_labels = ["Quality Drift (Goodhart)", "True Quality", "Management Overhead"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        agg = aggregate_by_condition(
            filtered, group_cols=["alpha"], metric=metric
        )
        ax.plot(agg["alpha"], agg["mean"], "o-", color="#E91E63", linewidth=2)
        ax.fill_between(
            agg["alpha"], agg["ci_lower"], agg["ci_upper"],
            alpha=0.2, color="#E91E63",
        )
        ax.set_xlabel("AI Capability (alpha)")
        ax.set_ylabel(label)
        ax.set_title(label)

    fig.suptitle(f"Challenge Amplification: {scenario} ({strategy})", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_sobol_tornado(
    sobol_path: Path,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Tornado chart of Sobol sensitivity indices (RQ5)."""
    _setup_style()

    with open(sobol_path) as f:
        data = json.load(f)

    names = data["names"]
    s1 = np.array(data["S1"])
    st = np.array(data["ST"])

    # Sort by total-order index
    order = np.argsort(st)[::-1]
    names = [names[i] for i in order]
    s1 = s1[order]
    st = st[order]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(names))
    width = 0.35

    ax.barh(y + width / 2, st, width, label="Total-order (ST)", color="#E91E63", alpha=0.8)
    ax.barh(y - width / 2, s1, width, label="First-order (S1)", color="#2196F3", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Sobol Index")
    ax.set_title("Sensitivity Analysis: Parameter Importance")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_quality_drift_over_time(
    df: pd.DataFrame,
    scenario: str,
    alpha: float = 0.5,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """True vs proxy quality for each strategy at fixed alpha (Challenge F)."""
    _setup_style()
    filtered = df[(df["scenario"] == scenario) & (df["alpha"] == alpha)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (axes[0], "quality_true_mean", "True Quality"),
        (axes[1], "quality_proxy_mean", "Proxy Quality"),
    ]:
        agg = aggregate_by_condition(filtered, group_cols=["strategy"], metric=metric)
        agg["order"] = agg["strategy"].map({s: i for i, s in enumerate(STRATEGY_ORDER)})
        agg = agg.sort_values("order")

        colors = [STRATEGY_COLORS.get(s, "#666") for s in agg["strategy"]]
        ax.bar(agg["strategy"], agg["mean"], yerr=agg["ci_95"], capsize=4, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Quality")
        plt.sca(ax)
        plt.xticks(rotation=30, ha="right")

    fig.suptitle(f"Quality Drift: {scenario} (alpha={alpha})", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: Path,
    sobol_path: Optional[Path] = None,
) -> None:
    """Generate all standard plots and save to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = df["scenario"].unique()
    alphas = sorted(df["alpha"].unique())

    for scenario in scenarios:
        # Strategy comparison at alpha=0.0
        plot_strategy_comparison(
            df, scenario, alpha=0.0,
            output_path=output_dir / f"{scenario}_strategy_comparison_a0.png",
        )
        plt.close()

        # Heatmap
        plot_alpha_heatmap(
            df, scenario,
            output_path=output_dir / f"{scenario}_alpha_heatmap.png",
        )
        plt.close()

        # Phase diagram
        plot_phase_diagram(
            df, scenario,
            output_path=output_dir / f"{scenario}_phase_diagram.png",
        )
        plt.close()

        # Challenge amplification
        plot_challenge_amplification(
            df, scenario,
            output_path=output_dir / f"{scenario}_challenge_amplification.png",
        )
        plt.close()

        # Quality drift at alpha=0.5
        if 0.5 in alphas:
            plot_quality_drift_over_time(
                df, scenario, alpha=0.5,
                output_path=output_dir / f"{scenario}_quality_drift.png",
            )
            plt.close()

    # Sobol tornado
    if sobol_path and sobol_path.exists():
        plot_sobol_tornado(
            sobol_path,
            output_path=output_dir / "sobol_tornado.png",
        )
        plt.close()

    print(f"All plots saved to {output_dir}")
