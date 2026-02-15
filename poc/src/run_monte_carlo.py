"""
Monte Carlo Experiment: Statistical Validation of Simulation Results
=====================================================================
Runs all optimization variants across N random seeds to determine:

1. Are the results statistically robust?
2. Is TrustDecay's superiority over Oracle statistically significant?
3. How stable are the variant rankings across random seeds?
4. What are the confidence intervals for each variant's output?

This addresses the fundamental concern that single-seed results (seed=42)
may not be representative of the true performance distribution.
"""

import os
import sys
import random
import json
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scientific_process import create_default_pipeline, ProcessStep
from optimizers import BaselineOptimizer
from advanced_optimizers import KanbanSciOpsOptimizer
from management_overhead import get_overhead_profile
from meta_overhead_optimizer import (
    MetaAIOracleOptimizer,
    MetaAINoisyOptimizer,
    MetaAIDelayedOptimizer,
    MetaAIRecursiveOptimizer,
    MetaAITrustDecayOptimizer,
)
from run_meta_overhead import MetaOverheadSimulator, FixedOverheadSimulator
from run_ai_superior import (
    AISuperiorSimulator,
    AISuperiorOracleOptimizer,
    AISuperiorTrustDecayOptimizer,
    AISuperiorRecursiveOptimizer,
)


# ============================================================
# Variant definitions: (label, optimizer_factory, simulator_class)
# ============================================================

CURRENT_WORLD_VARIANTS = [
    ("Baseline", BaselineOptimizer, FixedOverheadSimulator),
    ("Kanban", KanbanSciOpsOptimizer, FixedOverheadSimulator),
    ("Oracle", MetaAIOracleOptimizer, MetaOverheadSimulator),
    ("Noisy", MetaAINoisyOptimizer, MetaOverheadSimulator),
    ("Delayed", MetaAIDelayedOptimizer, MetaOverheadSimulator),
    ("Recursive", MetaAIRecursiveOptimizer, MetaOverheadSimulator),
    ("TrustDecay", MetaAITrustDecayOptimizer, MetaOverheadSimulator),
]

AI_SUPERIOR_VARIANTS = [
    ("Baseline-Sup", BaselineOptimizer, AISuperiorSimulator),
    ("Kanban-Sup", KanbanSciOpsOptimizer, AISuperiorSimulator),
    ("Oracle-Sup", AISuperiorOracleOptimizer, AISuperiorSimulator),
    ("Noisy-Sup", MetaAINoisyOptimizer, AISuperiorSimulator),
    ("Delayed-Sup", MetaAIDelayedOptimizer, AISuperiorSimulator),
    ("Recursive-Sup", AISuperiorRecursiveOptimizer, AISuperiorSimulator),
    ("TrustDecay-Sup", AISuperiorTrustDecayOptimizer, AISuperiorSimulator),
]


def run_single_seed(variants, seed, time_steps=100):
    """Run all variants for a single seed. Returns {label: total_output}."""
    results = {}
    for label, opt_factory, sim_class in variants:
        random.seed(seed)
        opt = opt_factory()
        sim = sim_class(optimizer=opt, seed=seed)
        result = sim.run(time_steps=time_steps)
        total_oh = 0.0
        total_meta = 0.0
        if hasattr(sim, "total_overhead"):
            total_oh = sim.total_overhead
        if hasattr(sim, "total_meta_cost"):
            total_meta = sim.total_meta_cost
        results[label] = {
            "output": result.total_output,
            "overhead": total_oh,
            "meta_cost": total_meta,
        }
    return results


def run_monte_carlo(variants, n_seeds=100, time_steps=100, start_seed=0):
    """Run Monte Carlo simulation across n_seeds."""
    # {label: [output_for_each_seed]}
    all_outputs = defaultdict(list)
    all_overheads = defaultdict(list)
    all_meta_costs = defaultdict(list)
    all_ranks = defaultdict(list)

    labels = [v[0] for v in variants]

    for i in range(n_seeds):
        seed = start_seed + i
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Seed {seed} ({i+1}/{n_seeds})...")

        results = run_single_seed(variants, seed, time_steps)

        for label in labels:
            all_outputs[label].append(results[label]["output"])
            all_overheads[label].append(results[label]["overhead"])
            all_meta_costs[label].append(results[label]["meta_cost"])

        # Compute ranks for this seed (1 = best)
        sorted_labels = sorted(labels, key=lambda l: results[l]["output"], reverse=True)
        for rank, label in enumerate(sorted_labels, 1):
            all_ranks[label].append(rank)

    return {
        "labels": labels,
        "outputs": dict(all_outputs),
        "overheads": dict(all_overheads),
        "meta_costs": dict(all_meta_costs),
        "ranks": dict(all_ranks),
        "n_seeds": n_seeds,
    }


# ============================================================
# Statistical analysis
# ============================================================

def welch_t_test(a, b):
    """Welch's t-test for two independent samples (unequal variance)."""
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-12:
        return 0.0, 1.0  # No difference

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    if denom < 1e-12:
        df = n_a + n_b - 2
    else:
        df = num / denom

    # Approximate p-value using normal distribution (good for n >= 30)
    # For proper p-value we'd need scipy, but this is a reasonable approximation
    z = abs(t_stat)
    # Two-tailed p-value approximation
    p_value = 2 * (1 - _normal_cdf(z))

    return t_stat, p_value


def _normal_cdf(x):
    """Approximate standard normal CDF (Abramowitz & Stegun)."""
    if x < 0:
        return 1.0 - _normal_cdf(-x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * math.exp(-x * x / 2.0)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return 1.0 - p * poly


def compute_statistics(mc_results):
    """Compute comprehensive statistics for Monte Carlo results."""
    stats = {}
    for label in mc_results["labels"]:
        outputs = np.array(mc_results["outputs"][label])
        ranks = np.array(mc_results["ranks"][label])
        n = len(outputs)

        mean = np.mean(outputs)
        std = np.std(outputs, ddof=1)
        se = std / math.sqrt(n)
        ci_95 = (mean - 1.96 * se, mean + 1.96 * se)

        stats[label] = {
            "mean": mean,
            "std": std,
            "se": se,
            "ci_95_low": ci_95[0],
            "ci_95_high": ci_95[1],
            "median": np.median(outputs),
            "min": np.min(outputs),
            "max": np.max(outputs),
            "q25": np.percentile(outputs, 25),
            "q75": np.percentile(outputs, 75),
            "cv": std / mean if mean > 0 else 0,  # coefficient of variation
            "mean_rank": np.mean(ranks),
            "rank_1_pct": np.sum(ranks == 1) / n * 100,
            "median_rank": np.median(ranks),
        }

    return stats


def pairwise_tests(mc_results, reference_label="Oracle"):
    """Run Welch's t-test between reference and each other variant."""
    results = {}
    ref_outputs = mc_results["outputs"].get(reference_label, [])
    if not ref_outputs:
        return results

    for label in mc_results["labels"]:
        if label == reference_label:
            continue
        other_outputs = mc_results["outputs"][label]
        t_stat, p_value = welch_t_test(other_outputs, ref_outputs)
        diff_mean = np.mean(other_outputs) - np.mean(ref_outputs)
        results[label] = {
            "diff_mean": diff_mean,
            "t_stat": t_stat,
            "p_value": p_value,
            "significant_005": p_value < 0.05,
            "significant_001": p_value < 0.01,
        }

    return results


# ============================================================
# Visualization
# ============================================================

COLORS = {
    "Baseline": "#bdc3c7",
    "Kanban": "#2ecc71",
    "Oracle": "#3498db",
    "Noisy": "#e67e22",
    "Delayed": "#9b59b6",
    "Recursive": "#e74c3c",
    "TrustDecay": "#1abc9c",
    "Baseline-Sup": "#bdc3c7",
    "Kanban-Sup": "#2ecc71",
    "Oracle-Sup": "#3498db",
    "Noisy-Sup": "#e67e22",
    "Delayed-Sup": "#9b59b6",
    "Recursive-Sup": "#e74c3c",
    "TrustDecay-Sup": "#1abc9c",
}


def generate_figures(mc_current, mc_superior, stats_current, stats_superior,
                     tests_current, tests_superior, output_dir):
    """Generate Monte Carlo analysis figures."""
    os.makedirs(output_dir, exist_ok=True)

    # ===== Figure 1: Box plots (Current World) =====
    fig, ax = plt.subplots(figsize=(14, 7))

    # Sort by mean output
    order = sorted(mc_current["labels"],
                   key=lambda l: stats_current[l]["mean"], reverse=True)
    data = [mc_current["outputs"][l] for l in order]
    colors = [COLORS.get(l, "gray") for l in order]

    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points (jittered)
    for i, (d, label) in enumerate(zip(data, order)):
        jitter = np.random.normal(0, 0.08, len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d,
                   alpha=0.3, s=12, color=COLORS.get(label, "gray"), zorder=3)

    # Add mean and CI annotations
    for i, label in enumerate(order):
        s = stats_current[label]
        ax.text(i + 1, s["max"] + 1.5,
                f"$\\mu$={s['mean']:.1f}\n$\\sigma$={s['std']:.1f}",
                ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("Total Research Output", fontsize=12)
    ax.set_title(f"Current World: Output Distribution (N={mc_current['n_seeds']} seeds)", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "mc_01_boxplot_current.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 2: Box plots (AI-Superior World) =====
    fig, ax = plt.subplots(figsize=(14, 7))

    order_sup = sorted(mc_superior["labels"],
                       key=lambda l: stats_superior[l]["mean"], reverse=True)
    data_sup = [mc_superior["outputs"][l] for l in order_sup]
    colors_sup = [COLORS.get(l, "gray") for l in order_sup]

    bp = ax.boxplot(data_sup, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_sup):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (d, label) in enumerate(zip(data_sup, order_sup)):
        jitter = np.random.normal(0, 0.08, len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d,
                   alpha=0.3, s=12, color=COLORS.get(label, "gray"), zorder=3)

    for i, label in enumerate(order_sup):
        s = stats_superior[label]
        ax.text(i + 1, s["max"] + 5,
                f"$\\mu$={s['mean']:.1f}\n$\\sigma$={s['std']:.1f}",
                ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(range(1, len(order_sup) + 1))
    ax.set_xticklabels([l.replace("-Sup", "") for l in order_sup], fontsize=9)
    ax.set_ylabel("Total Research Output", fontsize=12)
    ax.set_title(f"AI-Superior World: Output Distribution (N={mc_superior['n_seeds']} seeds)", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "mc_02_boxplot_superior.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 3: Confidence intervals comparison (both worlds) =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Current world
    order_c = sorted(mc_current["labels"],
                     key=lambda l: stats_current[l]["mean"], reverse=True)
    y_pos = np.arange(len(order_c))
    means_c = [stats_current[l]["mean"] for l in order_c]
    ci_low_c = [stats_current[l]["ci_95_low"] for l in order_c]
    ci_high_c = [stats_current[l]["ci_95_high"] for l in order_c]
    errors_c = [[m - lo for m, lo in zip(means_c, ci_low_c)],
                [hi - m for m, hi in zip(means_c, ci_high_c)]]

    ax1.barh(y_pos, means_c, xerr=errors_c, height=0.6,
             color=[COLORS.get(l, "gray") for l in order_c],
             edgecolor="black", linewidth=0.5, capsize=4)
    for i, (m, label) in enumerate(zip(means_c, order_c)):
        s = stats_current[label]
        ax1.text(m + 0.3, i, f"{m:.1f} [{s['ci_95_low']:.1f}, {s['ci_95_high']:.1f}]",
                 va="center", fontsize=7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(order_c, fontsize=9)
    ax1.set_xlabel("Total Output (mean + 95% CI)")
    ax1.set_title("Current World", fontsize=12)
    ax1.grid(True, axis="x", alpha=0.3)
    ax1.invert_yaxis()

    # AI-Superior world
    order_s = sorted(mc_superior["labels"],
                     key=lambda l: stats_superior[l]["mean"], reverse=True)
    y_pos_s = np.arange(len(order_s))
    means_s = [stats_superior[l]["mean"] for l in order_s]
    ci_low_s = [stats_superior[l]["ci_95_low"] for l in order_s]
    ci_high_s = [stats_superior[l]["ci_95_high"] for l in order_s]
    errors_s = [[m - lo for m, lo in zip(means_s, ci_low_s)],
                [hi - m for m, hi in zip(means_s, ci_high_s)]]

    ax2.barh(y_pos_s, means_s, xerr=errors_s, height=0.6,
             color=[COLORS.get(l, "gray") for l in order_s],
             edgecolor="black", linewidth=0.5, capsize=4)
    for i, (m, label) in enumerate(zip(means_s, order_s)):
        s = stats_superior[label]
        ax2.text(m + 0.5, i, f"{m:.1f} [{s['ci_95_low']:.1f}, {s['ci_95_high']:.1f}]",
                 va="center", fontsize=7)
    ax2.set_yticks(y_pos_s)
    ax2.set_yticklabels([l.replace("-Sup", "") for l in order_s], fontsize=9)
    ax2.set_xlabel("Total Output (mean + 95% CI)")
    ax2.set_title("AI-Superior World", fontsize=12)
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.invert_yaxis()

    fig.suptitle(f"95% Confidence Intervals (N={mc_current['n_seeds']} seeds)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "mc_03_confidence_intervals.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 4: Rank stability =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Current world rank histogram
    order_rank = sorted(mc_current["labels"],
                        key=lambda l: stats_current[l]["mean_rank"])
    for label in order_rank:
        ranks = mc_current["ranks"][label]
        rank_counts = [ranks.count(r) for r in range(1, len(order_rank) + 1)]
        ax1.barh(label, stats_current[label]["rank_1_pct"],
                 color=COLORS.get(label, "gray"), edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("% of Seeds Where Variant Ranked #1", fontsize=10)
    ax1.set_title("Current World: Win Rate", fontsize=12)
    ax1.grid(True, axis="x", alpha=0.3)
    for i, label in enumerate(order_rank):
        pct = stats_current[label]["rank_1_pct"]
        ax1.text(pct + 0.5, label, f"{pct:.0f}%", va="center", fontsize=9)

    # AI-Superior world rank histogram
    order_rank_s = sorted(mc_superior["labels"],
                          key=lambda l: stats_superior[l]["mean_rank"])
    for label in order_rank_s:
        short = label.replace("-Sup", "")
        ax2.barh(short, stats_superior[label]["rank_1_pct"],
                 color=COLORS.get(label, "gray"), edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("% of Seeds Where Variant Ranked #1", fontsize=10)
    ax2.set_title("AI-Superior World: Win Rate", fontsize=12)
    ax2.grid(True, axis="x", alpha=0.3)
    for label in order_rank_s:
        short = label.replace("-Sup", "")
        pct = stats_superior[label]["rank_1_pct"]
        ax2.text(pct + 0.5, short, f"{pct:.0f}%", va="center", fontsize=9)

    fig.suptitle(f"Rank Stability: How Often Does Each Variant Win? (N={mc_current['n_seeds']} seeds)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "mc_04_rank_stability.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 5: Statistical significance heatmap =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Current world pairwise significance
    labels_c = [l for l in order if l != "Baseline"]
    n_c = len(labels_c)
    p_matrix_c = np.ones((n_c, n_c))
    for i, la in enumerate(labels_c):
        for j, lb in enumerate(labels_c):
            if i != j:
                _, p = welch_t_test(
                    mc_current["outputs"][la],
                    mc_current["outputs"][lb]
                )
                p_matrix_c[i, j] = p

    im1 = ax1.imshow(p_matrix_c, cmap="RdYlGn", vmin=0, vmax=0.1)
    ax1.set_xticks(range(n_c))
    ax1.set_yticks(range(n_c))
    ax1.set_xticklabels(labels_c, fontsize=8, rotation=45, ha="right")
    ax1.set_yticklabels(labels_c, fontsize=8)
    for i in range(n_c):
        for j in range(n_c):
            if i != j:
                ax1.text(j, i, f"{p_matrix_c[i,j]:.3f}",
                         ha="center", va="center", fontsize=7,
                         color="white" if p_matrix_c[i, j] < 0.05 else "black")
    ax1.set_title("Current World: p-values (Welch t-test)")
    fig.colorbar(im1, ax=ax1, shrink=0.7, label="p-value")

    # AI-Superior world pairwise significance
    labels_s = [l for l in order_sup if "Baseline" not in l]
    n_s = len(labels_s)
    p_matrix_s = np.ones((n_s, n_s))
    for i, la in enumerate(labels_s):
        for j, lb in enumerate(labels_s):
            if i != j:
                _, p = welch_t_test(
                    mc_superior["outputs"][la],
                    mc_superior["outputs"][lb]
                )
                p_matrix_s[i, j] = p

    im2 = ax2.imshow(p_matrix_s, cmap="RdYlGn", vmin=0, vmax=0.1)
    ax2.set_xticks(range(n_s))
    ax2.set_yticks(range(n_s))
    short_labels_s = [l.replace("-Sup", "") for l in labels_s]
    ax2.set_xticklabels(short_labels_s, fontsize=8, rotation=45, ha="right")
    ax2.set_yticklabels(short_labels_s, fontsize=8)
    for i in range(n_s):
        for j in range(n_s):
            if i != j:
                ax2.text(j, i, f"{p_matrix_s[i,j]:.3f}",
                         ha="center", va="center", fontsize=7,
                         color="white" if p_matrix_s[i, j] < 0.05 else "black")
    ax2.set_title("AI-Superior World: p-values (Welch t-test)")
    fig.colorbar(im2, ax=ax2, shrink=0.7, label="p-value")

    fig.suptitle(f"Pairwise Statistical Significance (N={mc_current['n_seeds']} seeds)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "mc_05_significance_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 6: Coefficient of variation (robustness) =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Current
    order_cv = sorted(mc_current["labels"],
                      key=lambda l: stats_current[l]["cv"])
    cvs = [stats_current[l]["cv"] * 100 for l in order_cv]
    ax1.barh(order_cv, cvs,
             color=[COLORS.get(l, "gray") for l in order_cv],
             edgecolor="black", linewidth=0.5)
    for i, (label, cv) in enumerate(zip(order_cv, cvs)):
        ax1.text(cv + 0.2, label, f"{cv:.1f}%", va="center", fontsize=9)
    ax1.set_xlabel("Coefficient of Variation (%)")
    ax1.set_title("Current World: Result Stability")
    ax1.grid(True, axis="x", alpha=0.3)

    # Superior
    order_cv_s = sorted(mc_superior["labels"],
                        key=lambda l: stats_superior[l]["cv"])
    cvs_s = [stats_superior[l]["cv"] * 100 for l in order_cv_s]
    ax2.barh([l.replace("-Sup", "") for l in order_cv_s], cvs_s,
             color=[COLORS.get(l, "gray") for l in order_cv_s],
             edgecolor="black", linewidth=0.5)
    for i, (label, cv) in enumerate(zip(order_cv_s, cvs_s)):
        ax2.text(cv + 0.2, label.replace("-Sup", ""), f"{cv:.1f}%", va="center", fontsize=9)
    ax2.set_xlabel("Coefficient of Variation (%)")
    ax2.set_title("AI-Superior World: Result Stability")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Result Robustness: Lower CV = More Stable Results", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "mc_06_robustness.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll figures saved to {output_dir}")


# ============================================================
# Report generation
# ============================================================

def print_report(mc_results, stats, tests, world_name):
    """Print detailed Monte Carlo analysis report."""
    n = mc_results["n_seeds"]

    print(f"\n{'='*100}")
    print(f"MONTE CARLO ANALYSIS: {world_name} (N={n} seeds)")
    print(f"{'='*100}")

    print(f"\n--- Descriptive Statistics ---")
    print(f"{'Variant':<18} {'Mean':>8} {'Std':>8} {'CV%':>6} "
          f"{'95% CI':>20} {'Min':>8} {'Max':>8} {'Rank1%':>7}")
    print(f"{'-'*95}")

    order = sorted(mc_results["labels"], key=lambda l: stats[l]["mean"], reverse=True)
    for label in order:
        s = stats[label]
        ci = f"[{s['ci_95_low']:.1f}, {s['ci_95_high']:.1f}]"
        print(f"{label:<18} {s['mean']:>8.1f} {s['std']:>8.1f} {s['cv']*100:>5.1f}% "
              f"{ci:>20} {s['min']:>8.1f} {s['max']:>8.1f} {s['rank_1_pct']:>6.0f}%")

    # Statistical tests
    ref_label = "Oracle" if "Oracle" in mc_results["labels"] else "Oracle-Sup"
    if tests:
        print(f"\n--- Statistical Tests vs {ref_label} (Welch's t-test) ---")
        print(f"{'Variant':<18} {'Mean Diff':>10} {'t-stat':>10} {'p-value':>10} {'Sig(5%)':>8} {'Sig(1%)':>8}")
        print(f"{'-'*70}")

        for label in order:
            if label == ref_label:
                print(f"{label:<18} {'(reference)':>10}")
                continue
            if label not in tests:
                continue
            t = tests[label]
            sig5 = "YES ***" if t["significant_005"] else "no"
            sig1 = "YES ***" if t["significant_001"] else "no"
            print(f"{label:<18} {t['diff_mean']:>+10.2f} {t['t_stat']:>10.2f} "
                  f"{t['p_value']:>10.4f} {sig5:>8} {sig1:>8}")

    # Key findings
    print(f"\n--- Key Findings ---")
    best = order[0]
    second = order[1]
    print(f"  Best variant: {best} (mean={stats[best]['mean']:.1f})")
    print(f"  Runner-up:    {second} (mean={stats[second]['mean']:.1f})")
    print(f"  Gap: {stats[best]['mean'] - stats[second]['mean']:.2f}")

    if best in tests:
        p = tests[best]["p_value"]
        print(f"  Statistical significance of best vs {ref_label}: p={p:.4f}")
    elif second in tests:
        # best IS Oracle
        # Check if the second best is significantly different
        p = tests[second]["p_value"]
        print(f"  Statistical significance of {second} vs {best}: p={p:.4f}")

    # Win rate analysis
    print(f"\n  Win rates (% of seeds ranked #1):")
    for label in order[:3]:
        print(f"    {label}: {stats[label]['rank_1_pct']:.0f}%")

    # CI overlap analysis
    best_ci = (stats[best]["ci_95_low"], stats[best]["ci_95_high"])
    second_ci = (stats[second]["ci_95_low"], stats[second]["ci_95_high"])
    overlap = best_ci[0] < second_ci[1] and second_ci[0] < best_ci[1]
    print(f"\n  95% CI overlap between top-2: {'YES (not clearly separable)' if overlap else 'NO (clearly distinct)'}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    N_SEEDS = 100
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures_mc")

    print("=" * 60)
    print(f"MONTE CARLO EXPERIMENT (N={N_SEEDS} seeds)")
    print("=" * 60)

    # --- Current World ---
    print(f"\n--- Running Current World ({N_SEEDS} seeds x 7 variants) ---")
    mc_current = run_monte_carlo(CURRENT_WORLD_VARIANTS, n_seeds=N_SEEDS)
    stats_current = compute_statistics(mc_current)
    tests_current = pairwise_tests(mc_current, reference_label="Oracle")
    print_report(mc_current, stats_current, tests_current, "Current World")

    # --- AI-Superior World ---
    print(f"\n--- Running AI-Superior World ({N_SEEDS} seeds x 7 variants) ---")
    mc_superior = run_monte_carlo(AI_SUPERIOR_VARIANTS, n_seeds=N_SEEDS)
    stats_superior = compute_statistics(mc_superior)
    tests_superior = pairwise_tests(mc_superior, reference_label="Oracle-Sup")
    print_report(mc_superior, stats_superior, tests_superior, "AI-Superior World")

    # --- Generate figures ---
    print(f"\n--- Generating figures ---")
    generate_figures(mc_current, mc_superior, stats_current, stats_superior,
                     tests_current, tests_superior, output_dir)

    # --- Save raw data ---
    raw_data = {
        "n_seeds": N_SEEDS,
        "current_world": {
            "outputs": {k: [round(v, 4) for v in vals] for k, vals in mc_current["outputs"].items()},
            "stats": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                      for k, v in stats_current.items()},
        },
        "superior_world": {
            "outputs": {k: [round(v, 4) for v in vals] for k, vals in mc_superior["outputs"].items()},
            "stats": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                      for k, v in stats_superior.items()},
        },
    }
    raw_path = os.path.join(output_dir, "monte_carlo_raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"\nRaw data saved to {raw_path}")

    print(f"\n{'='*60}")
    print("MONTE CARLO EXPERIMENT COMPLETE")
    print(f"{'='*60}")
