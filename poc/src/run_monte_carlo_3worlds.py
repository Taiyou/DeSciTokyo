"""
Monte Carlo 3-World Comparison
================================
Runs all three worlds with N seeds and produces comparison analysis:

1. Current World: Normal AI + human bottleneck exists
2. Bottleneck-Persists: High AI capability + human bottleneck exists
3. AI-Superior: High AI capability + human bottleneck removed

Key question: How much of the AI-superior world's gains come from
AI capability improvement vs. removing the human review bottleneck?
"""

import os
import random
import json
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optimizers import BaselineOptimizer
from advanced_optimizers import KanbanSciOpsOptimizer
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
from run_bottleneck_persists import (
    BottleneckPersistsSimulator,
    BNPersistsOracleOptimizer,
    BNPersistsTrustDecayOptimizer,
    BNPersistsRecursiveOptimizer,
)


# ============================================================
# Variant definitions
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

BOTTLENECK_PERSISTS_VARIANTS = [
    ("Baseline-BNP", BaselineOptimizer, BottleneckPersistsSimulator),
    ("Kanban-BNP", KanbanSciOpsOptimizer, BottleneckPersistsSimulator),
    ("Oracle-BNP", BNPersistsOracleOptimizer, BottleneckPersistsSimulator),
    ("Noisy-BNP", MetaAINoisyOptimizer, BottleneckPersistsSimulator),
    ("Delayed-BNP", MetaAIDelayedOptimizer, BottleneckPersistsSimulator),
    ("Recursive-BNP", BNPersistsRecursiveOptimizer, BottleneckPersistsSimulator),
    ("TrustDecay-BNP", BNPersistsTrustDecayOptimizer, BottleneckPersistsSimulator),
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


# ============================================================
# Monte Carlo runner (reused from run_monte_carlo.py)
# ============================================================

def run_single_seed(variants, seed, time_steps=100):
    results = {}
    for label, opt_factory, sim_class in variants:
        random.seed(seed)
        opt = opt_factory()
        sim = sim_class(optimizer=opt, seed=seed)
        result = sim.run(time_steps=time_steps)
        total_oh = getattr(sim, "total_overhead", 0.0)
        total_meta = getattr(sim, "total_meta_cost", 0.0)
        results[label] = {
            "output": result.total_output,
            "overhead": total_oh,
            "meta_cost": total_meta,
        }
    return results


def run_monte_carlo(variants, n_seeds=100, time_steps=100, start_seed=0):
    all_outputs = defaultdict(list)
    all_overheads = defaultdict(list)
    all_meta_costs = defaultdict(list)
    all_ranks = defaultdict(list)
    labels = [v[0] for v in variants]

    for i in range(n_seeds):
        seed = start_seed + i
        if (i + 1) % 20 == 0 or i == 0:
            print(f"    Seed {seed} ({i+1}/{n_seeds})...")
        results = run_single_seed(variants, seed, time_steps)
        for label in labels:
            all_outputs[label].append(results[label]["output"])
            all_overheads[label].append(results[label]["overhead"])
            all_meta_costs[label].append(results[label]["meta_cost"])
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
# Statistics (reused)
# ============================================================

def _normal_cdf(x):
    if x < 0:
        return 1.0 - _normal_cdf(-x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    d = 0.3989422804014327
    p = d * math.exp(-x * x / 2.0)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return 1.0 - p * poly


def welch_t_test(a, b):
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-12:
        return 0.0, 1.0
    t_stat = (mean_a - mean_b) / se
    z = abs(t_stat)
    p_value = 2 * (1 - _normal_cdf(z))
    return t_stat, p_value


def compute_statistics(mc_results):
    stats = {}
    for label in mc_results["labels"]:
        outputs = np.array(mc_results["outputs"][label])
        ranks = np.array(mc_results["ranks"][label])
        n = len(outputs)
        mean = np.mean(outputs)
        std = np.std(outputs, ddof=1)
        se = std / math.sqrt(n)
        stats[label] = {
            "mean": mean, "std": std, "se": se,
            "ci_95_low": mean - 1.96 * se, "ci_95_high": mean + 1.96 * se,
            "median": np.median(outputs),
            "min": np.min(outputs), "max": np.max(outputs),
            "q25": np.percentile(outputs, 25), "q75": np.percentile(outputs, 75),
            "cv": std / mean if mean > 0 else 0,
            "mean_rank": np.mean(ranks),
            "rank_1_pct": np.sum(ranks == 1) / n * 100,
            "median_rank": np.median(ranks),
        }
    return stats


def pairwise_tests(mc_results, reference_label):
    results = {}
    ref = mc_results["outputs"].get(reference_label, [])
    if not ref:
        return results
    for label in mc_results["labels"]:
        if label == reference_label:
            continue
        other = mc_results["outputs"][label]
        t_stat, p_value = welch_t_test(other, ref)
        results[label] = {
            "diff_mean": np.mean(other) - np.mean(ref),
            "t_stat": t_stat, "p_value": p_value,
            "significant_005": p_value < 0.05,
            "significant_001": p_value < 0.01,
        }
    return results


# ============================================================
# Print report
# ============================================================

def print_report(mc_results, stats, tests, world_name, ref_label):
    n = mc_results["n_seeds"]
    print(f"\n{'='*100}")
    print(f"MONTE CARLO: {world_name} (N={n} seeds)")
    print(f"{'='*100}")

    print(f"\n{'Variant':<20} {'Mean':>8} {'Std':>8} {'CV%':>6} {'95% CI':>20} {'Min':>8} {'Max':>8} {'Rank1%':>7}")
    print(f"{'-'*95}")

    order = sorted(mc_results["labels"], key=lambda l: stats[l]["mean"], reverse=True)
    for label in order:
        s = stats[label]
        ci = f"[{s['ci_95_low']:.1f}, {s['ci_95_high']:.1f}]"
        print(f"{label:<20} {s['mean']:>8.1f} {s['std']:>8.1f} {s['cv']*100:>5.1f}% "
              f"{ci:>20} {s['min']:>8.1f} {s['max']:>8.1f} {s['rank_1_pct']:>6.0f}%")

    if tests:
        print(f"\n  vs {ref_label} (Welch t-test):")
        for label in order:
            if label == ref_label:
                continue
            if label not in tests:
                continue
            t = tests[label]
            sig = "***" if t["significant_001"] else ("*" if t["significant_005"] else "ns")
            print(f"    {label:<18} diff={t['diff_mean']:>+7.2f}  p={t['p_value']:.4f}  {sig}")


# ============================================================
# 3-World comparison visualization
# ============================================================

COLORS = {
    "Baseline": "#bdc3c7", "Kanban": "#2ecc71", "Oracle": "#3498db",
    "Noisy": "#e67e22", "Delayed": "#9b59b6", "Recursive": "#e74c3c",
    "TrustDecay": "#1abc9c",
}

VARIANT_ORDER = ["Baseline", "Kanban", "Oracle", "Noisy", "Delayed", "Recursive", "TrustDecay"]


def _strip_suffix(label):
    """Remove -BNP or -Sup suffix for display."""
    return label.replace("-BNP", "").replace("-Sup", "")


def _get_color(label):
    return COLORS.get(_strip_suffix(label), "gray")


def generate_3world_figures(mc_curr, mc_bnp, mc_sup,
                            stats_curr, stats_bnp, stats_sup,
                            output_dir):
    """Generate comparison figures across all three worlds."""
    os.makedirs(output_dir, exist_ok=True)
    n_seeds = mc_curr["n_seeds"]

    # ===== Figure 1: 3-world box plots for best variant in each =====
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    worlds = [
        ("Current World\n(Normal AI + Bottleneck)", mc_curr, stats_curr),
        ("Bottleneck-Persists\n(High AI + Bottleneck)", mc_bnp, stats_bnp),
        ("AI-Superior\n(High AI + No Bottleneck)", mc_sup, stats_sup),
    ]

    for ax, (title, mc, stats) in zip(axes, worlds):
        order = sorted(mc["labels"], key=lambda l: stats[l]["mean"], reverse=True)
        data = [mc["outputs"][l] for l in order]
        colors = [_get_color(l) for l in order]
        short_labels = [_strip_suffix(l) for l in order]

        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, label in enumerate(order):
            s = stats[label]
            ax.text(i + 1, s["max"] + (s["max"] - s["min"]) * 0.15,
                    f"$\\mu$={s['mean']:.1f}", ha="center", fontsize=7, fontweight="bold")

        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(short_labels, fontsize=8, rotation=30, ha="right")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Total Output")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"3-World Output Distribution Comparison (N={n_seeds} seeds)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "mc3_01_boxplots_3worlds.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 2: Cross-world comparison for each variant =====
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(VARIANT_ORDER))
    width = 0.25

    # Map variant names to each world
    def _find_label(mc, base_name):
        for l in mc["labels"]:
            if _strip_suffix(l) == base_name:
                return l
        return None

    means_curr = []
    means_bnp = []
    means_sup = []
    ci_curr = []
    ci_bnp = []
    ci_sup = []

    for vname in VARIANT_ORDER:
        lc = _find_label(mc_curr, vname)
        lb = _find_label(mc_bnp, vname)
        ls = _find_label(mc_sup, vname)

        means_curr.append(stats_curr[lc]["mean"] if lc else 0)
        means_bnp.append(stats_bnp[lb]["mean"] if lb else 0)
        means_sup.append(stats_sup[ls]["mean"] if ls else 0)

        ci_curr.append(stats_curr[lc]["se"] * 1.96 if lc else 0)
        ci_bnp.append(stats_bnp[lb]["se"] * 1.96 if lb else 0)
        ci_sup.append(stats_sup[ls]["se"] * 1.96 if ls else 0)

    bars1 = ax.bar(x - width, means_curr, width, yerr=ci_curr, capsize=3,
                   label="Current World", color="#e74c3c", alpha=0.7,
                   edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, means_bnp, width, yerr=ci_bnp, capsize=3,
                   label="Bottleneck-Persists", color="#f39c12", alpha=0.7,
                   edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, means_sup, width, yerr=ci_sup, capsize=3,
                   label="AI-Superior", color="#3498db", alpha=0.7,
                   edgecolor="black", linewidth=0.5)

    for bars, means in [(bars1, means_curr), (bars2, means_bnp), (bars3, means_sup)]:
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f"{m:.0f}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(VARIANT_ORDER, fontsize=10)
    ax.set_ylabel("Mean Total Output", fontsize=12)
    ax.set_title(f"Each Variant Across 3 Worlds (N={n_seeds} seeds, error bars = 95% CI)", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "mc3_02_variant_across_worlds.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 3: Bottleneck cost decomposition =====
    # How much output is lost to the bottleneck vs. gained from AI capability?
    fig, ax = plt.subplots(figsize=(14, 7))

    non_baseline = [v for v in VARIANT_ORDER if v != "Baseline"]
    x = np.arange(len(non_baseline))

    # For each non-baseline variant:
    #   AI capability gain = BNP_output - Current_output
    #   Bottleneck removal gain = Superior_output - BNP_output
    #   Total gain = Superior_output - Current_output
    ai_gains = []
    bn_gains = []
    for vname in non_baseline:
        lc = _find_label(mc_curr, vname)
        lb = _find_label(mc_bnp, vname)
        ls = _find_label(mc_sup, vname)

        mc = stats_curr[lc]["mean"] if lc else 0
        mb = stats_bnp[lb]["mean"] if lb else 0
        ms = stats_sup[ls]["mean"] if ls else 0

        ai_gains.append(mb - mc)   # Gain from AI capability alone
        bn_gains.append(ms - mb)   # Additional gain from removing bottleneck

    ax.bar(x, ai_gains, 0.5, label="AI capability improvement\n(BNP - Current)",
           color="#f39c12", edgecolor="black", linewidth=0.5)
    ax.bar(x, bn_gains, 0.5, bottom=ai_gains,
           label="Bottleneck removal gain\n(Superior - BNP)",
           color="#3498db", edgecolor="black", linewidth=0.5)

    for i, (ag, bg) in enumerate(zip(ai_gains, bn_gains)):
        total = ag + bg
        ax.text(i, ag / 2, f"{ag:.0f}", ha="center", fontsize=9, fontweight="bold", color="white")
        ax.text(i, ag + bg / 2, f"{bg:.0f}", ha="center", fontsize=9, fontweight="bold", color="white")
        ax.text(i, total + 2, f"Total: {total:.0f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(non_baseline, fontsize=10)
    ax.set_ylabel("Output Gain vs Current World", fontsize=12)
    ax.set_title("Decomposition: AI Capability Gain vs. Bottleneck Removal Gain", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "mc3_03_gain_decomposition.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 4: Win rates across 3 worlds =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (title, mc, stats) in zip(axes, worlds):
        order = sorted(mc["labels"], key=lambda l: stats[l]["rank_1_pct"], reverse=True)
        pcts = [stats[l]["rank_1_pct"] for l in order]
        short = [_strip_suffix(l) for l in order]
        colors = [_get_color(l) for l in order]

        ax.barh(range(len(order)), pcts, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(short, fontsize=9)
        ax.set_xlabel("Win Rate (%)")
        ax.set_title(title.split("\n")[0], fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        for i, pct in enumerate(pcts):
            if pct > 0:
                ax.text(pct + 0.5, i, f"{pct:.0f}%", va="center", fontsize=9)
        ax.invert_yaxis()

    fig.suptitle(f"Win Rates Across 3 Worlds (N={n_seeds} seeds)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "mc3_04_win_rates_3worlds.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 5: Bottleneck-persists detailed box + significance =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Box plot
    order_bnp = sorted(mc_bnp["labels"], key=lambda l: stats_bnp[l]["mean"], reverse=True)
    data_bnp = [mc_bnp["outputs"][l] for l in order_bnp]
    colors_bnp = [_get_color(l) for l in order_bnp]
    short_bnp = [_strip_suffix(l) for l in order_bnp]

    bp = ax1.boxplot(data_bnp, patch_artist=True, widths=0.6,
                     medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_bnp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (d, label) in enumerate(zip(data_bnp, order_bnp)):
        jitter = np.random.normal(0, 0.08, len(d))
        ax1.scatter(np.full(len(d), i + 1) + jitter, d,
                    alpha=0.3, s=12, color=_get_color(label), zorder=3)
        s = stats_bnp[label]
        ax1.text(i + 1, s["max"] + 2,
                 f"$\\mu$={s['mean']:.1f}\n$\\sigma$={s['std']:.1f}",
                 ha="center", fontsize=7, fontweight="bold")

    ax1.set_xticks(range(1, len(order_bnp) + 1))
    ax1.set_xticklabels(short_bnp, fontsize=9)
    ax1.set_ylabel("Total Research Output")
    ax1.set_title(f"Bottleneck-Persists World (N={n_seeds})")
    ax1.grid(True, axis="y", alpha=0.3)

    # Significance: CI comparison
    order_ci = sorted(mc_bnp["labels"], key=lambda l: stats_bnp[l]["mean"], reverse=True)
    y_pos = np.arange(len(order_ci))
    means = [stats_bnp[l]["mean"] for l in order_ci]
    ci_low = [stats_bnp[l]["ci_95_low"] for l in order_ci]
    ci_high = [stats_bnp[l]["ci_95_high"] for l in order_ci]
    errors = [[m - lo for m, lo in zip(means, ci_low)],
              [hi - m for m, hi in zip(means, ci_high)]]

    ax2.barh(y_pos, means, xerr=errors, height=0.6, capsize=4,
             color=[_get_color(l) for l in order_ci],
             edgecolor="black", linewidth=0.5)
    for i, (m, label) in enumerate(zip(means, order_ci)):
        s = stats_bnp[label]
        ax2.text(m + 0.5, i, f"{m:.1f} [{s['ci_95_low']:.1f}, {s['ci_95_high']:.1f}]",
                 va="center", fontsize=7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([_strip_suffix(l) for l in order_ci], fontsize=9)
    ax2.set_xlabel("Total Output (mean + 95% CI)")
    ax2.set_title("95% Confidence Intervals")
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.invert_yaxis()

    fig.suptitle("Bottleneck-Persists World: Detailed Analysis", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "mc3_05_bnp_detail.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 6: TrustDecay trajectory across worlds =====
    fig, ax = plt.subplots(figsize=(12, 7))

    # For each world, show TrustDecay's rank_1_pct and mean
    world_names = ["Current\nWorld", "Bottleneck\nPersists", "AI-Superior\nWorld"]
    td_means = [
        stats_curr[_find_label(mc_curr, "TrustDecay")]["mean"],
        stats_bnp[_find_label(mc_bnp, "TrustDecay")]["mean"],
        stats_sup[_find_label(mc_sup, "TrustDecay")]["mean"],
    ]
    td_win_rates = [
        stats_curr[_find_label(mc_curr, "TrustDecay")]["rank_1_pct"],
        stats_bnp[_find_label(mc_bnp, "TrustDecay")]["rank_1_pct"],
        stats_sup[_find_label(mc_sup, "TrustDecay")]["rank_1_pct"],
    ]
    or_means = [
        stats_curr[_find_label(mc_curr, "Oracle")]["mean"],
        stats_bnp[_find_label(mc_bnp, "Oracle")]["mean"],
        stats_sup[_find_label(mc_sup, "Oracle")]["mean"],
    ]
    or_win_rates = [
        stats_curr[_find_label(mc_curr, "Oracle")]["rank_1_pct"],
        stats_bnp[_find_label(mc_bnp, "Oracle")]["rank_1_pct"],
        stats_sup[_find_label(mc_sup, "Oracle")]["rank_1_pct"],
    ]

    x = np.arange(3)
    width = 0.3

    bars_td = ax.bar(x - width/2, td_means, width, label="TrustDecay",
                     color=COLORS["TrustDecay"], edgecolor="black", linewidth=0.5)
    bars_or = ax.bar(x + width/2, or_means, width, label="Oracle",
                     color=COLORS["Oracle"], edgecolor="black", linewidth=0.5)

    for i, (m, wr) in enumerate(zip(td_means, td_win_rates)):
        ax.text(x[i] - width/2, m + 3, f"{m:.0f}\n(win {wr:.0f}%)",
                ha="center", fontsize=9, fontweight="bold", color=COLORS["TrustDecay"])
    for i, (m, wr) in enumerate(zip(or_means, or_win_rates)):
        ax.text(x[i] + width/2, m + 3, f"{m:.0f}\n(win {wr:.0f}%)",
                ha="center", fontsize=9, fontweight="bold", color=COLORS["Oracle"])

    ax.set_xticks(x)
    ax.set_xticklabels(world_names, fontsize=11)
    ax.set_ylabel("Mean Total Output", fontsize=12)
    ax.set_title("TrustDecay vs Oracle: How Bottleneck Determines the Winner", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "mc3_06_trustdecay_vs_oracle.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll 3-world figures saved to {output_dir}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    N_SEEDS = 100
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures_mc3")

    print("=" * 70)
    print(f"MONTE CARLO 3-WORLD COMPARISON (N={N_SEEDS} seeds)")
    print("=" * 70)

    # --- World 1: Current ---
    print(f"\n  [1/3] Current World ({N_SEEDS} seeds x 7 variants)")
    mc_curr = run_monte_carlo(CURRENT_WORLD_VARIANTS, n_seeds=N_SEEDS)
    stats_curr = compute_statistics(mc_curr)
    tests_curr = pairwise_tests(mc_curr, "Oracle")
    print_report(mc_curr, stats_curr, tests_curr, "Current World", "Oracle")

    # --- World 2: Bottleneck-Persists ---
    print(f"\n  [2/3] Bottleneck-Persists World ({N_SEEDS} seeds x 7 variants)")
    mc_bnp = run_monte_carlo(BOTTLENECK_PERSISTS_VARIANTS, n_seeds=N_SEEDS)
    stats_bnp = compute_statistics(mc_bnp)
    tests_bnp = pairwise_tests(mc_bnp, "Oracle-BNP")
    print_report(mc_bnp, stats_bnp, tests_bnp, "Bottleneck-Persists World", "Oracle-BNP")

    # --- World 3: AI-Superior ---
    print(f"\n  [3/3] AI-Superior World ({N_SEEDS} seeds x 7 variants)")
    mc_sup = run_monte_carlo(AI_SUPERIOR_VARIANTS, n_seeds=N_SEEDS)
    stats_sup = compute_statistics(mc_sup)
    tests_sup = pairwise_tests(mc_sup, "Oracle-Sup")
    print_report(mc_sup, stats_sup, tests_sup, "AI-Superior World", "Oracle-Sup")

    # --- Cross-world summary ---
    print(f"\n{'='*70}")
    print("CROSS-WORLD SUMMARY: Decomposing AI Capability vs. Bottleneck Removal")
    print(f"{'='*70}")

    for vname in VARIANT_ORDER:
        if vname == "Baseline":
            continue
        lc = next((l for l in mc_curr["labels"] if l.replace("-BNP", "").replace("-Sup", "") == vname), None)
        lb = next((l for l in mc_bnp["labels"] if l.replace("-BNP", "").replace("-Sup", "") == vname), None)
        ls = next((l for l in mc_sup["labels"] if l.replace("-BNP", "").replace("-Sup", "") == vname), None)

        mc = stats_curr[lc]["mean"] if lc else 0
        mb = stats_bnp[lb]["mean"] if lb else 0
        ms = stats_sup[ls]["mean"] if ls else 0

        ai_gain = mb - mc
        bn_gain = ms - mb
        total = ms - mc
        ai_pct = ai_gain / total * 100 if total > 0 else 0
        bn_pct = bn_gain / total * 100 if total > 0 else 0

        print(f"  {vname:<12}  Current={mc:.1f}  BNP={mb:.1f}  Superior={ms:.1f}  "
              f"| AI gain={ai_gain:+.1f} ({ai_pct:.0f}%)  BN removal={bn_gain:+.1f} ({bn_pct:.0f}%)")

    # --- Generate figures ---
    print(f"\n--- Generating 3-world figures ---")
    generate_3world_figures(mc_curr, mc_bnp, mc_sup,
                            stats_curr, stats_bnp, stats_sup,
                            output_dir)

    # --- Save raw data ---
    raw_data = {
        "n_seeds": N_SEEDS,
        "current_world": {
            "outputs": {k: [round(v, 4) for v in vals] for k, vals in mc_curr["outputs"].items()},
            "stats": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                      for k, v in stats_curr.items()},
        },
        "bottleneck_persists_world": {
            "outputs": {k: [round(v, 4) for v in vals] for k, vals in mc_bnp["outputs"].items()},
            "stats": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                      for k, v in stats_bnp.items()},
        },
        "superior_world": {
            "outputs": {k: [round(v, 4) for v in vals] for k, vals in mc_sup["outputs"].items()},
            "stats": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                      for k, v in stats_sup.items()},
        },
    }
    raw_path = os.path.join(output_dir, "monte_carlo_3worlds_raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Raw data saved to {raw_path}")

    print(f"\n{'='*70}")
    print("3-WORLD MONTE CARLO COMPLETE")
    print(f"{'='*70}")
