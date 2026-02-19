"""
Meta-Science Experiment 1: Publication Bias and False Positive Accumulation
============================================================================
Tests the hypothesis: "Publication bias (where positive/novel results are
preferentially published) causes the scientific community to accumulate
false positives, reducing the truth-content of published knowledge.
AI-optimized labs exacerbate this effect."

4 Conditions:
1. No bias: fair review process
2. Publication bias only: positive results favored
3. Publication bias + p-hacking: researchers inflate results
4. Full bias + AI acceleration: AI labs produce more, amplifying bias

Each condition runs N=100 seeds for statistical validation.
"""

import os
import sys
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meta_science_models import PublicationChannel, EcosystemMetrics
from meta_science_ecosystem import MetaScienceEcosystem

# ============================================================
# Experiment conditions
# ============================================================

CONDITIONS = {
    "No Bias": {
        "positive_result_bias": 0.0,
        "novelty_bias": 0.0,
        "p_hacking_intensity": 0.0,
        "ai_fraction": 0.0,
        "acceptance_rate": 0.25,
        "review_quality": 0.7,
    },
    "Pub Bias Only": {
        "positive_result_bias": 0.4,
        "novelty_bias": 0.3,
        "p_hacking_intensity": 0.0,
        "ai_fraction": 0.0,
        "acceptance_rate": 0.25,
        "review_quality": 0.5,
    },
    "Pub Bias + p-hack": {
        "positive_result_bias": 0.4,
        "novelty_bias": 0.3,
        "p_hacking_intensity": 0.5,
        "ai_fraction": 0.0,
        "acceptance_rate": 0.25,
        "review_quality": 0.5,
    },
    "Full Bias + AI": {
        "positive_result_bias": 0.4,
        "novelty_bias": 0.3,
        "p_hacking_intensity": 0.5,
        "ai_fraction": 0.5,
        "acceptance_rate": 0.25,
        "review_quality": 0.5,
    },
}

COLORS = {
    "No Bias": "#4CAF50",
    "Pub Bias Only": "#FF9800",
    "Pub Bias + p-hack": "#F44336",
    "Full Bias + AI": "#9C27B0",
}


def run_single_condition(condition_name: str, params: dict, seed: int,
                         n_labs: int = 50, time_steps: int = 200) -> list[EcosystemMetrics]:
    """Run a single condition with given parameters and seed."""
    channel = PublicationChannel(
        name="Journal",
        acceptance_rate=params["acceptance_rate"],
        novelty_bias=params["novelty_bias"],
        positive_result_bias=params["positive_result_bias"],
        review_quality=params["review_quality"],
    )

    eco = MetaScienceEcosystem(
        n_labs=n_labs,
        publication_channel=channel,
        base_funding=6.0,
        input_rate=2.0,
        base_truth_rate=0.5,
        paper_threshold=0.3,
        funding_cycle=20,
        ai_fraction=params["ai_fraction"],
        p_hacking_intensity=params["p_hacking_intensity"],
        seed=seed,
    )

    return eco.run(time_steps=time_steps)


def run_monte_carlo(n_seeds: int = 100, n_labs: int = 50,
                    time_steps: int = 200) -> dict:
    """Run all conditions across N seeds."""
    all_results = {}

    for cond_name, params in CONDITIONS.items():
        print(f"\n--- {cond_name} ---")
        condition_results = []
        for seed in range(n_seeds):
            metrics = run_single_condition(cond_name, params, seed, n_labs, time_steps)
            condition_results.append(metrics)
            if (seed + 1) % 25 == 0:
                print(f"  Seed {seed + 1}/{n_seeds} done")
        all_results[cond_name] = condition_results

    return all_results


def compute_statistics(all_results: dict, time_steps: int = 200) -> dict:
    """Compute summary statistics for each condition."""
    stats = {}
    for cond_name, runs in all_results.items():
        # Final truth ratios
        final_truth_ratios = [run[-1].truth_ratio for run in runs]
        final_fpr = [run[-1].false_positive_rate for run in runs]
        final_pubs = [run[-1].total_papers_published for run in runs]

        # Cumulative publications count (sum across all steps)
        total_pubs_per_run = []
        for run in runs:
            total = sum(m.total_papers_published for m in run)
            total_pubs_per_run.append(total)

        stats[cond_name] = {
            "truth_ratio_mean": np.mean(final_truth_ratios),
            "truth_ratio_std": np.std(final_truth_ratios),
            "truth_ratio_ci95": 1.96 * np.std(final_truth_ratios) / np.sqrt(len(final_truth_ratios)),
            "fpr_mean": np.mean(final_fpr),
            "fpr_std": np.std(final_fpr),
            "total_pubs_mean": np.mean(total_pubs_per_run),
            "total_pubs_std": np.std(total_pubs_per_run),
        }
    return stats


# ============================================================
# Visualization functions
# ============================================================

def generate_figures(all_results: dict, stats: dict, output_dir: str) -> None:
    """Generate all 6 figures for Experiment 1."""
    os.makedirs(output_dir, exist_ok=True)

    _fig1_truth_ratio_timeseries(all_results, output_dir)
    _fig2_publication_volume_vs_truth(all_results, output_dir)
    _fig3_false_positive_accumulation(all_results, output_dir)
    _fig4_funding_inequality(all_results, output_dir)
    _fig5_ai_share_timeseries(all_results, output_dir)
    _fig6_boxplot_conditions(all_results, output_dir)

    print(f"\nFigures saved to {output_dir}")


def _fig1_truth_ratio_timeseries(all_results: dict, output_dir: str) -> None:
    """Truth ratio over time for all conditions (mean +/- SE)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond_name, runs in all_results.items():
        time_steps = len(runs[0])
        truth_matrix = np.array([[m.truth_ratio for m in run] for run in runs])
        mean_line = truth_matrix.mean(axis=0)
        se_line = truth_matrix.std(axis=0) / np.sqrt(len(runs))

        x = np.arange(time_steps)
        ax.plot(x, mean_line, label=cond_name, color=COLORS[cond_name], linewidth=2)
        ax.fill_between(x, mean_line - se_line, mean_line + se_line,
                        color=COLORS[cond_name], alpha=0.15)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Truth Ratio (Published Papers)", fontsize=12)
    ax.set_title("Publication Bias: Truth Ratio Over Time (N=100 seeds)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms1_01_truth_ratio_timeseries.png"), dpi=150)
    plt.close()


def _fig2_publication_volume_vs_truth(all_results: dict, output_dir: str) -> None:
    """Scatter: total publications vs final truth ratio."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for cond_name, runs in all_results.items():
        total_pubs = [sum(m.total_papers_published for m in run) for run in runs]
        truth_ratios = [run[-1].truth_ratio for run in runs]
        ax.scatter(total_pubs, truth_ratios, label=cond_name,
                   color=COLORS[cond_name], alpha=0.5, s=20)

    ax.set_xlabel("Total Publications (all labs)", fontsize=12)
    ax.set_ylabel("Final Truth Ratio", fontsize=12)
    ax.set_title("Publication Volume vs Knowledge Quality", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms1_02_publication_volume_vs_truth.png"), dpi=150)
    plt.close()


def _fig3_false_positive_accumulation(all_results: dict, output_dir: str) -> None:
    """Cumulative false positive rate over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond_name, runs in all_results.items():
        time_steps = len(runs[0])
        fpr_matrix = np.array([[m.false_positive_rate for m in run] for run in runs])
        mean_line = fpr_matrix.mean(axis=0)
        se_line = fpr_matrix.std(axis=0) / np.sqrt(len(runs))

        x = np.arange(time_steps)
        ax.plot(x, mean_line, label=cond_name, color=COLORS[cond_name], linewidth=2)
        ax.fill_between(x, mean_line - se_line, mean_line + se_line,
                        color=COLORS[cond_name], alpha=0.15)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("False Positive Rate", fontsize=12)
    ax.set_title("False Positive Accumulation Over Time", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms1_03_false_positive_accumulation.png"), dpi=150)
    plt.close()


def _fig4_funding_inequality(all_results: dict, output_dir: str) -> None:
    """Gini coefficient of funding over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond_name, runs in all_results.items():
        time_steps = len(runs[0])
        gini_matrix = np.array([[m.gini_funding for m in run] for run in runs])
        mean_line = gini_matrix.mean(axis=0)

        x = np.arange(time_steps)
        ax.plot(x, mean_line, label=cond_name, color=COLORS[cond_name], linewidth=2)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Gini Coefficient (Funding)", fontsize=12)
    ax.set_title("Funding Inequality Over Time", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms1_04_funding_inequality.png"), dpi=150)
    plt.close()


def _fig5_ai_share_timeseries(all_results: dict, output_dir: str) -> None:
    """AI lab publication share over time (for AI conditions)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond_name, runs in all_results.items():
        time_steps = len(runs[0])
        ai_matrix = np.array([[m.ai_lab_publication_share for m in run] for run in runs])
        mean_line = ai_matrix.mean(axis=0)

        x = np.arange(time_steps)
        ax.plot(x, mean_line, label=cond_name, color=COLORS[cond_name], linewidth=2)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("AI Lab Publication Share", fontsize=12)
    ax.set_title("AI Lab Dominance in Publications", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms1_05_ai_share_timeseries.png"), dpi=150)
    plt.close()


def _fig6_boxplot_conditions(all_results: dict, output_dir: str) -> None:
    """Box plots of final truth ratio across conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Truth ratio box plot
    data_truth = []
    labels = []
    colors = []
    for cond_name, runs in all_results.items():
        final_truth = [run[-1].truth_ratio for run in runs]
        data_truth.append(final_truth)
        labels.append(cond_name)
        colors.append(COLORS[cond_name])

    bp1 = axes[0].boxplot(data_truth, labels=labels, patch_artist=True)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_ylabel("Truth Ratio", fontsize=12)
    axes[0].set_title("Final Truth Ratio (t=200)", fontsize=13)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, alpha=0.3, axis="y")

    # False positive rate box plot
    data_fpr = []
    for cond_name, runs in all_results.items():
        final_fpr = [run[-1].false_positive_rate for run in runs]
        data_fpr.append(final_fpr)

    bp2 = axes[1].boxplot(data_fpr, labels=labels, patch_artist=True)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel("False Positive Rate", fontsize=12)
    axes[1].set_title("Final False Positive Rate (t=200)", fontsize=13)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms1_06_boxplot_conditions.png"), dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    N_SEEDS = 100
    N_LABS = 50
    TIME_STEPS = 200

    print("=" * 60)
    print("Meta-Science Experiment 1: Publication Bias")
    print("=" * 60)
    print(f"  Labs: {N_LABS}, Steps: {TIME_STEPS}, Seeds: {N_SEEDS}")
    print(f"  Conditions: {list(CONDITIONS.keys())}")

    # Run Monte Carlo
    all_results = run_monte_carlo(n_seeds=N_SEEDS, n_labs=N_LABS, time_steps=TIME_STEPS)

    # Compute statistics
    stats = compute_statistics(all_results, TIME_STEPS)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<25} {'Truth Ratio':>15} {'FPR':>12} {'Total Pubs':>12}")
    print("-" * 64)
    for cond_name, s in stats.items():
        tr = f"{s['truth_ratio_mean']:.3f} +/- {s['truth_ratio_ci95']:.3f}"
        fpr = f"{s['fpr_mean']:.3f}"
        pubs = f"{s['total_pubs_mean']:.0f}"
        print(f"{cond_name:<25} {tr:>15} {fpr:>12} {pubs:>12}")

    # Generate figures
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures_ms1")
    generate_figures(all_results, stats, output_dir)

    # Save stats as JSON
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    with open(os.path.join(results_dir, "ms1_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {results_dir}/ms1_stats.json")
