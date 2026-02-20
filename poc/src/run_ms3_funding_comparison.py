"""
Meta-Science Experiment 3: Funding Allocation Mechanisms
=========================================================
Tests the hypothesis: "Peer-review funding maximizes average output but
suppresses breakthroughs. Lottery maximizes exploration. FRO maximizes
long-term breakthroughs."

4 Mechanisms:
1. Peer Review: reputation-based, conservative
2. Lottery: random among eligible
3. SBIR Staged: progressive filtering (small→medium→large)
4. FRO Long-term: stable goal-directed programs

Each mechanism runs N=100 seeds for statistical validation.
"""

import os
import json
import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meta_science_models import PublicationChannel, EcosystemMetrics, compute_gini
from meta_science_ecosystem import MetaScienceEcosystem, LabState
from scientific_process import create_default_pipeline, ProcessConfig
from funding_models import (
    FundingAllocator,
    PeerReviewFunding,
    LotteryFunding,
    SBIRStagedFunding,
    FROLongTermFunding,
)


class FundingEcosystem(MetaScienceEcosystem):
    """Extends MetaScienceEcosystem with diverse lab populations and funding models."""

    def __init__(
        self,
        funding_allocator: FundingAllocator,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.funding_allocator = funding_allocator
        self.breakthroughs: list[dict] = []  # Track breakthrough papers
        self.breakthrough_threshold = 2.0

        # Override labs with diverse risk appetites
        self._diversify_labs()

    def _diversify_labs(self) -> None:
        """Give labs diverse risk appetites affecting their pipeline parameters."""
        for lab in self.labs:
            lab.risk_appetite = random.random()

            # High-risk labs: higher uncertainty/failure but potentially bigger breakthroughs
            for step in lab.pipeline:
                step.config.uncertainty *= (1.0 + lab.risk_appetite * 0.4)
                step.config.failure_rate *= (1.0 + lab.risk_appetite * 0.25)
                step.config.base_throughput *= (1.0 - lab.risk_appetite * 0.15)
                step.throughput = step.config.base_throughput

    def _generate_paper(self, lab, time_step, output):
        """Override to add breakthrough mechanics."""
        paper = super()._generate_paper(lab, time_step, output)

        # Breakthrough value from Pareto distribution
        alpha = 3.0 - lab.risk_appetite * 1.2  # Fatter tail for high-risk labs
        alpha = max(1.5, alpha)
        breakthrough_value = random.paretovariate(alpha)

        if breakthrough_value > self.breakthrough_threshold:
            paper.novelty = min(1.0, paper.novelty + 0.3)
            self.breakthroughs.append({
                "lab_id": lab.lab_id,
                "time_step": time_step,
                "value": breakthrough_value,
                "risk_appetite": lab.risk_appetite,
                "is_true": paper.is_true,
            })

        return paper

    def _redistribute_funding(self) -> None:
        """Use the configured funding allocator."""
        total_budget = sum(lab.funding for lab in self.labs)

        lab_scores = []
        for lab in self.labs:
            recent_pubs = len([
                p for p in lab.papers_published
                if p.time_step > self.time_step - self.funding_cycle
            ])
            lab_scores.append({
                "lab_id": lab.lab_id,
                "reputation": lab.reputation,
                "recent_pubs": recent_pubs,
                "risk_appetite": lab.risk_appetite,
                "current_funding": lab.funding,
            })

        allocations = self.funding_allocator.allocate(lab_scores, total_budget)

        for lab, funding in zip(self.labs, allocations):
            lab.funding = max(1.0, funding)


# ============================================================
# Experiment configuration
# ============================================================

MECHANISMS = {
    "Peer Review": PeerReviewFunding,
    "Lottery": LotteryFunding,
    "SBIR Staged": SBIRStagedFunding,
    "FRO Long-term": FROLongTermFunding,
}

COLORS = {
    "Peer Review": "#2196F3",
    "Lottery": "#FF9800",
    "SBIR Staged": "#4CAF50",
    "FRO Long-term": "#9C27B0",
}


def run_single_mechanism(mech_name: str, allocator_class, seed: int,
                         n_labs: int = 50, time_steps: int = 200):
    """Run a single mechanism and return (metrics, breakthroughs, final_labs)."""
    channel = PublicationChannel(
        name="Journal",
        acceptance_rate=0.2,
        novelty_bias=0.3,
        positive_result_bias=0.2,
        review_quality=0.7,
    )

    allocator = allocator_class()

    eco = FundingEcosystem(
        funding_allocator=allocator,
        n_labs=n_labs,
        publication_channel=channel,
        base_funding=6.0,
        input_rate=2.0,
        base_truth_rate=0.5,
        paper_threshold=0.3,
        funding_cycle=20,
        ai_fraction=0.3,
        p_hacking_intensity=0.0,
        seed=seed,
    )
    metrics = eco.run(time_steps=time_steps)
    return metrics, eco.breakthroughs, eco.labs


def run_monte_carlo(n_seeds: int = 100, n_labs: int = 50,
                    time_steps: int = 200) -> dict:
    """Run all mechanisms across N seeds."""
    all_results = {}
    for mech_name, allocator_class in MECHANISMS.items():
        print(f"\n--- {mech_name} ---")
        results = []
        breakthroughs_all = []
        for seed in range(n_seeds):
            metrics, bts, labs = run_single_mechanism(
                mech_name, allocator_class, seed, n_labs, time_steps
            )
            results.append(metrics)
            breakthroughs_all.append(bts)
            if (seed + 1) % 25 == 0:
                print(f"  Seed {seed + 1}/{n_seeds} done")
        all_results[mech_name] = {
            "metrics": results,
            "breakthroughs": breakthroughs_all,
        }
    return all_results


def compute_statistics(all_results: dict) -> dict:
    """Compute summary statistics."""
    stats = {}
    for name, data in all_results.items():
        runs = data["metrics"]
        bts_all = data["breakthroughs"]

        total_output = [sum(m.total_community_output for m in run) for run in runs]
        final_truth = [run[-1].truth_ratio for run in runs]
        final_gini = [run[-1].gini_funding for run in runs]
        n_breakthroughs = [len(bts) for bts in bts_all]
        bt_values = [sum(b["value"] for b in bts) for bts in bts_all]

        stats[name] = {
            "output_mean": float(np.mean(total_output)),
            "output_std": float(np.std(total_output)),
            "truth_ratio_mean": float(np.mean(final_truth)),
            "gini_mean": float(np.mean(final_gini)),
            "breakthroughs_mean": float(np.mean(n_breakthroughs)),
            "breakthroughs_std": float(np.std(n_breakthroughs)),
            "bt_value_mean": float(np.mean(bt_values)),
        }
    return stats


# ============================================================
# Visualization
# ============================================================

def generate_figures(all_results: dict, output_dir: str) -> None:
    """Generate all figures for Experiment 3."""
    os.makedirs(output_dir, exist_ok=True)

    _fig1_output_vs_breakthroughs(all_results, output_dir)
    _fig2_funding_inequality(all_results, output_dir)
    _fig3_breakthrough_distribution(all_results, output_dir)
    _fig4_risk_funding(all_results, output_dir)
    _fig5_efficiency_frontier(all_results, output_dir)
    _fig6_summary_bars(all_results, output_dir)

    print(f"\nFigures saved to {output_dir}")


def _fig1_output_vs_breakthroughs(all_results, output_dir):
    """Scatter: total output vs number of breakthroughs."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in all_results.items():
        runs = data["metrics"]
        bts_all = data["breakthroughs"]
        outputs = [sum(m.total_community_output for m in run) for run in runs]
        n_bts = [len(bts) for bts in bts_all]
        ax.scatter(outputs, n_bts, label=name, color=COLORS[name], alpha=0.4, s=15)
        ax.scatter([np.mean(outputs)], [np.mean(n_bts)], color=COLORS[name],
                   s=100, edgecolors="black", linewidths=1.5, zorder=5)

    ax.set_xlabel("Total Community Output")
    ax.set_ylabel("Number of Breakthroughs")
    ax.set_title("Output vs Breakthroughs by Funding Mechanism")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms3_01_output_vs_breakthroughs.png"), dpi=150)
    plt.close()


def _fig2_funding_inequality(all_results, output_dir):
    """Gini coefficient over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in all_results.items():
        runs = data["metrics"]
        T = len(runs[0])
        mat = np.array([[m.gini_funding for m in run] for run in runs])
        mean = mat.mean(axis=0)
        x = np.arange(T)
        ax.plot(x, mean, label=name, color=COLORS[name], linewidth=2)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Gini Coefficient (Funding)")
    ax.set_title("Funding Inequality Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms3_02_funding_inequality.png"), dpi=150)
    plt.close()


def _fig3_breakthrough_distribution(all_results, output_dir):
    """Histogram of breakthrough counts across seeds."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, data) in zip(axes.flat, all_results.items()):
        n_bts = [len(bts) for bts in data["breakthroughs"]]
        ax.hist(n_bts, bins=20, color=COLORS[name], alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(n_bts), color="red", linestyle="--",
                    label=f"Mean: {np.mean(n_bts):.1f}")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Breakthroughs (N=100 seeds)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Breakthrough Distribution by Funding Mechanism", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms3_03_breakthrough_distribution.png"), dpi=150)
    plt.close()


def _fig4_risk_funding(all_results, output_dir):
    """How much funding goes to high-risk vs low-risk labs (from breakthroughs)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(all_results))
    width = 0.35

    high_risk_bts = []
    low_risk_bts = []
    names = []
    for name, data in all_results.items():
        names.append(name)
        all_bts = [b for bts in data["breakthroughs"] for b in bts]
        if all_bts:
            hr = sum(1 for b in all_bts if b["risk_appetite"] > 0.5)
            lr = sum(1 for b in all_bts if b["risk_appetite"] <= 0.5)
            total = hr + lr
            high_risk_bts.append(hr / max(1, total))
            low_risk_bts.append(lr / max(1, total))
        else:
            high_risk_bts.append(0)
            low_risk_bts.append(0)

    ax.bar(x_pos - width / 2, high_risk_bts, width, label="High Risk Labs",
           color="#F44336", alpha=0.7)
    ax.bar(x_pos + width / 2, low_risk_bts, width, label="Low Risk Labs",
           color="#2196F3", alpha=0.7)

    ax.set_xlabel("Funding Mechanism")
    ax.set_ylabel("Share of Breakthroughs")
    ax.set_title("Breakthrough Origin: High-Risk vs Low-Risk Labs")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms3_04_risk_funding.png"), dpi=150)
    plt.close()


def _fig5_efficiency_frontier(all_results, output_dir):
    """Average output vs breakthrough rate: the fundamental tradeoff."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in all_results.items():
        runs = data["metrics"]
        bts_all = data["breakthroughs"]
        outputs = [sum(m.total_community_output for m in run) for run in runs]
        bt_rates = [len(bts) / max(1, sum(m.total_papers_published for m in run))
                     for bts, run in zip(bts_all, runs)]

        ax.scatter(np.mean(outputs), np.mean(bt_rates),
                   s=200, color=COLORS[name], edgecolors="black", linewidths=2,
                   label=name, zorder=5)

        # Error bars
        ax.errorbar(np.mean(outputs), np.mean(bt_rates),
                     xerr=np.std(outputs) / np.sqrt(len(outputs)),
                     yerr=np.std(bt_rates) / np.sqrt(len(bt_rates)),
                     color=COLORS[name], fmt="none", capsize=5)

    ax.set_xlabel("Mean Total Output", fontsize=12)
    ax.set_ylabel("Mean Breakthrough Rate", fontsize=12)
    ax.set_title("Efficiency Frontier: Output vs Innovation", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms3_05_efficiency_frontier.png"), dpi=150)
    plt.close()


def _fig6_summary_bars(all_results, output_dir):
    """Summary bar chart comparing mechanisms on multiple metrics."""
    stats = compute_statistics(all_results)
    names = list(stats.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Total output
    vals = [stats[n]["output_mean"] for n in names]
    colors = [COLORS[n] for n in names]
    axes[0].bar(range(len(names)), vals, color=colors, alpha=0.7, edgecolor="black")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=20, fontsize=9)
    axes[0].set_title("Total Community Output")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Breakthroughs
    vals = [stats[n]["breakthroughs_mean"] for n in names]
    axes[1].bar(range(len(names)), vals, color=colors, alpha=0.7, edgecolor="black")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=20, fontsize=9)
    axes[1].set_title("Mean Breakthroughs")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Funding Gini
    vals = [stats[n]["gini_mean"] for n in names]
    axes[2].bar(range(len(names)), vals, color=colors, alpha=0.7, edgecolor="black")
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=20, fontsize=9)
    axes[2].set_title("Funding Inequality (Gini)")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Funding Mechanism Comparison (N=100 seeds)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms3_06_summary_bars.png"), dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    N_SEEDS = 100
    N_LABS = 50
    TIME_STEPS = 200

    print("=" * 60)
    print("Meta-Science Experiment 3: Funding Allocation")
    print("=" * 60)
    print(f"  Labs: {N_LABS}, Steps: {TIME_STEPS}, Seeds: {N_SEEDS}")
    print(f"  Mechanisms: {list(MECHANISMS.keys())}")

    all_results = run_monte_carlo(n_seeds=N_SEEDS, n_labs=N_LABS, time_steps=TIME_STEPS)
    stats = compute_statistics(all_results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Mechanism':<18} {'Output':>10} {'BTs':>8} {'Gini':>8} {'Truth':>8}")
    print("-" * 52)
    for name, s in stats.items():
        print(f"{name:<18} {s['output_mean']:>10.1f} {s['breakthroughs_mean']:>8.1f} "
              f"{s['gini_mean']:>8.3f} {s['truth_ratio_mean']:>8.3f}")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures_ms3")
    generate_figures(all_results, output_dir)

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    with open(os.path.join(results_dir, "ms3_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {results_dir}/ms3_stats.json")
