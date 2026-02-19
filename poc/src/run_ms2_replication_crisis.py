"""
Meta-Science Experiment 2: Replication Crisis Dynamics
=======================================================
Tests the hypothesis: "The replication crisis emerges from publication bias
and the cost asymmetry between publishing and replicating. AI can either
worsen it (more unreplicable papers) or improve it (cheaper replication)."

5 Incentive Regimes:
1. Status Quo: no replication incentives
2. Replication Mandate: 10% of output must be replication
3. Replication Reward: replications count toward reputation
4. AI Replication Bot: automated systematic replication
5. Pre-registration: p-hacking eliminated, overhead added

Each regime runs N=100 seeds for statistical validation.
"""

import os
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meta_science_models import PublicationChannel, EcosystemMetrics
from meta_science_ecosystem import MetaScienceEcosystem, LabState
from replication_engine import ReplicationEngine


class ReplicationEcosystem(MetaScienceEcosystem):
    """Extends MetaScienceEcosystem with replication mechanics."""

    def __init__(
        self,
        regime: str = "status_quo",
        replication_cost_ratio: float = 3.0,
        ai_cost_discount: float = 0.6,
        mandate_fraction: float = 0.1,
        pre_registration: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.regime = regime
        self.replication_engine = ReplicationEngine(
            replication_cost_ratio=replication_cost_ratio,
            ai_cost_discount=ai_cost_discount,
        )
        self.mandate_fraction = mandate_fraction
        self.pre_registration = pre_registration
        self.retraction_count = 0
        self.replication_investment = 0.0

        # If pre-registration, reduce p-hacking
        if pre_registration:
            for lab in self.labs:
                lab.p_hacking_intensity *= 0.1  # 90% reduction

    def _step(self) -> None:
        """Override to add replication mechanics."""
        t = self.time_step
        papers_this_step = []
        total_output = 0.0

        # 1. Run labs and generate papers
        for lab in self.labs:
            output = self._run_lab(lab, t)
            total_output += output

            if output > self.paper_threshold:
                paper = self._generate_paper(lab, t, output)
                lab.papers_submitted.append(paper)
                papers_this_step.append(paper)

        # 2. Publication review
        published_this_step = []
        for paper in papers_this_step:
            if self.publication_channel.evaluate(paper):
                paper.is_published = True
                lab = self.labs[paper.lab_id]
                lab.papers_published.append(paper)
                self.knowledge_base.append(paper)
                published_this_step.append(paper)

        # 3. Replication phase
        self._run_replication(t)

        # 4. Update reputations (with replication rewards if applicable)
        self._update_reputations_with_replication(published_this_step)

        # 5. Funding
        if t > 0 and t % self.funding_cycle == 0:
            self._redistribute_funding()

        # 6. Metrics
        self._collect_replication_metrics(t, papers_this_step, published_this_step, total_output)

    def _run_replication(self, time_step: int) -> None:
        """Run replication attempts based on the regime."""
        if not self.knowledge_base:
            return

        candidates = [p for p in self.knowledge_base
                       if p.replicated is None and not p.retracted
                       and time_step - p.time_step > 10]  # Wait 10 steps before replicating

        if not candidates:
            return

        if self.regime == "status_quo":
            # Almost no replication (1% chance per candidate per step)
            for paper in candidates:
                if random.random() < 0.01:
                    lab = random.choice(self.labs)
                    self._do_replication(paper, lab, time_step)

        elif self.regime == "mandate":
            # Each lab must allocate mandate_fraction of output to replication
            n_attempts = max(1, int(len(self.labs) * self.mandate_fraction))
            selected = random.sample(candidates, min(n_attempts, len(candidates)))
            for paper in selected:
                lab = random.choice(self.labs)
                self._do_replication(paper, lab, time_step)

        elif self.regime == "reward":
            # Labs voluntarily replicate because it boosts reputation
            # ~5% of labs replicate per step
            n_attempts = max(1, int(len(self.labs) * 0.05))
            # Prefer high-citation papers
            sorted_candidates = sorted(candidates, key=lambda p: p.citations, reverse=True)
            selected = sorted_candidates[:n_attempts]
            for paper in selected:
                lab = random.choice(self.labs)
                self._do_replication(paper, lab, time_step)

        elif self.regime == "ai_bot":
            # AI replication bot systematically replicates top papers
            # Much cheaper, more thorough
            n_attempts = max(2, int(len(candidates) * 0.1))
            sorted_candidates = sorted(candidates, key=lambda p: p.citations, reverse=True)
            selected = sorted_candidates[:n_attempts]
            for paper in selected:
                self.replication_engine.attempt_replication(
                    paper=paper,
                    lab_id=-1,  # Bot
                    time_step=time_step,
                    lab_quality=0.9,
                    is_ai_lab=True,
                )
                self.replication_investment += self.replication_engine.replication_cost_ratio * 0.4
                if paper.replicated is False:
                    self._retract_paper(paper)

        elif self.regime == "pre_registration":
            # Moderate replication rate (3% per step)
            n_attempts = max(1, int(len(candidates) * 0.03))
            selected = random.sample(candidates, min(n_attempts, len(candidates)))
            for paper in selected:
                lab = random.choice(self.labs)
                self._do_replication(paper, lab, time_step)

    def _do_replication(self, paper, lab: LabState, time_step: int) -> None:
        """Perform a single replication attempt."""
        attempt = self.replication_engine.attempt_replication(
            paper=paper,
            lab_id=lab.lab_id,
            time_step=time_step,
            lab_quality=lab.pipeline_quality(),
            is_ai_lab=lab.is_ai_lab,
        )
        self.replication_investment += attempt.cost

        if not attempt.replicated:
            self._retract_paper(paper)

    def _retract_paper(self, paper) -> None:
        """Retract a paper that failed replication + cascade."""
        if paper.retracted:
            return
        paper.retracted = True
        self.retraction_count += 1

        # Cascade: citing papers lose credibility
        lab = self.labs[paper.lab_id]
        lab.reputation = max(0.1, lab.reputation - 0.05)

    def _update_reputations_with_replication(self, published) -> None:
        """Update reputations considering replication rewards."""
        # Standard publication rewards
        self._update_reputations(published)

        # Additional reward for successful replications (reward regime)
        if self.regime == "reward":
            recent_attempts = [
                a for a in self.replication_engine.attempts
                if a.time_step == self.time_step and a.replicated
            ]
            for attempt in recent_attempts:
                if attempt.replicating_lab_id >= 0:
                    lab = self.labs[attempt.replicating_lab_id]
                    lab.reputation = min(2.0, lab.reputation + 0.015)

    def _collect_replication_metrics(self, time_step, submitted, published, total_output):
        """Collect metrics including replication data."""
        # First collect standard metrics
        self._collect_metrics(time_step, submitted, published, total_output)

        # Then update replication-specific fields
        rep_stats = self.replication_engine.get_replication_stats()
        metrics = self.metrics_history[-1]
        metrics.replication_rate = rep_stats["replication_rate"]
        tested_papers = [p for p in self.knowledge_base if p.replicated is not None]
        if tested_papers:
            metrics.successful_replication_rate = (
                sum(1 for p in tested_papers if p.replicated) / len(tested_papers)
            )
        metrics.retraction_count = self.retraction_count


# ============================================================
# Experiment configuration
# ============================================================

REGIMES = {
    "Status Quo": {"regime": "status_quo"},
    "Mandate (10%)": {"regime": "mandate", "mandate_fraction": 0.1},
    "Reward": {"regime": "reward"},
    "AI Bot": {"regime": "ai_bot"},
    "Pre-registration": {"regime": "pre_registration", "pre_registration": True},
}

COLORS = {
    "Status Quo": "#9E9E9E",
    "Mandate (10%)": "#2196F3",
    "Reward": "#4CAF50",
    "AI Bot": "#9C27B0",
    "Pre-registration": "#FF9800",
}


def run_single_regime(regime_name: str, params: dict, seed: int,
                      n_labs: int = 50, time_steps: int = 200) -> list[EcosystemMetrics]:
    """Run a single regime."""
    channel = PublicationChannel(
        name="Journal",
        acceptance_rate=0.2,
        novelty_bias=0.3,
        positive_result_bias=0.4,
        review_quality=0.6,
    )

    eco = ReplicationEcosystem(
        n_labs=n_labs,
        publication_channel=channel,
        base_funding=6.0,
        input_rate=2.0,
        base_truth_rate=0.5,
        paper_threshold=0.3,
        funding_cycle=20,
        ai_fraction=0.3,
        p_hacking_intensity=0.3,
        seed=seed,
        **params,
    )
    return eco.run(time_steps=time_steps)


def run_monte_carlo(n_seeds: int = 100, n_labs: int = 50,
                    time_steps: int = 200) -> dict:
    """Run all regimes across N seeds."""
    all_results = {}
    for regime_name, params in REGIMES.items():
        print(f"\n--- {regime_name} ---")
        results = []
        for seed in range(n_seeds):
            metrics = run_single_regime(regime_name, params, seed, n_labs, time_steps)
            results.append(metrics)
            if (seed + 1) % 25 == 0:
                print(f"  Seed {seed + 1}/{n_seeds} done")
        all_results[regime_name] = results
    return all_results


def compute_statistics(all_results: dict) -> dict:
    """Compute summary statistics."""
    stats = {}
    for name, runs in all_results.items():
        final_truth = [run[-1].truth_ratio for run in runs]
        final_rep = [run[-1].successful_replication_rate for run in runs]
        final_retract = [run[-1].retraction_count for run in runs]

        stats[name] = {
            "truth_ratio_mean": float(np.mean(final_truth)),
            "truth_ratio_std": float(np.std(final_truth)),
            "replication_rate_mean": float(np.mean(final_rep)),
            "replication_rate_std": float(np.std(final_rep)),
            "retraction_mean": float(np.mean(final_retract)),
            "retraction_std": float(np.std(final_retract)),
        }
    return stats


# ============================================================
# Visualization
# ============================================================

def generate_figures(all_results: dict, output_dir: str) -> None:
    """Generate all figures for Experiment 2."""
    os.makedirs(output_dir, exist_ok=True)

    _fig1_reliability_timeseries(all_results, output_dir)
    _fig2_replication_rate(all_results, output_dir)
    _fig3_retraction_cascade(all_results, output_dir)
    _fig4_novel_vs_reliable(all_results, output_dir)
    _fig5_regime_comparison_boxplot(all_results, output_dir)
    _fig6_truth_vs_replication(all_results, output_dir)

    print(f"\nFigures saved to {output_dir}")


def _fig1_reliability_timeseries(all_results, output_dir):
    """Knowledge reliability (truth ratio) over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, runs in all_results.items():
        T = len(runs[0])
        mat = np.array([[m.truth_ratio for m in run] for run in runs])
        mean = mat.mean(axis=0)
        se = mat.std(axis=0) / np.sqrt(len(runs))
        x = np.arange(T)
        ax.plot(x, mean, label=name, color=COLORS[name], linewidth=2)
        ax.fill_between(x, mean - se, mean + se, color=COLORS[name], alpha=0.15)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Truth Ratio")
    ax.set_title("Knowledge Reliability Under Different Replication Regimes (N=100)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms2_01_reliability_timeseries.png"), dpi=150)
    plt.close()


def _fig2_replication_rate(all_results, output_dir):
    """Successful replication rate over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, runs in all_results.items():
        T = len(runs[0])
        mat = np.array([[m.successful_replication_rate for m in run] for run in runs])
        mean = mat.mean(axis=0)
        x = np.arange(T)
        ax.plot(x, mean, label=name, color=COLORS[name], linewidth=2)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Successful Replication Rate")
    ax.set_title("Replication Success Rate Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms2_02_replication_rate.png"), dpi=150)
    plt.close()


def _fig3_retraction_cascade(all_results, output_dir):
    """Cumulative retractions over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, runs in all_results.items():
        T = len(runs[0])
        mat = np.array([[m.retraction_count for m in run] for run in runs])
        mean = mat.mean(axis=0)
        se = mat.std(axis=0) / np.sqrt(len(runs))
        x = np.arange(T)
        ax.plot(x, mean, label=name, color=COLORS[name], linewidth=2)
        ax.fill_between(x, mean - se, mean + se, color=COLORS[name], alpha=0.15)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Retractions")
    ax.set_title("Retraction Cascade Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms2_03_retraction_cascade.png"), dpi=150)
    plt.close()


def _fig4_novel_vs_reliable(all_results, output_dir):
    """Novel output vs reliability tradeoff."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, runs in all_results.items():
        outputs = [sum(m.total_community_output for m in run) for run in runs]
        truths = [run[-1].truth_ratio for run in runs]
        ax.scatter(outputs, truths, label=name, color=COLORS[name], alpha=0.4, s=15)
        # Add mean point
        ax.scatter([np.mean(outputs)], [np.mean(truths)], color=COLORS[name],
                   s=100, edgecolors="black", linewidths=1.5, zorder=5)

    ax.set_xlabel("Total Community Output")
    ax.set_ylabel("Final Truth Ratio")
    ax.set_title("Productivity vs Reliability Tradeoff")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms2_04_novel_vs_reliable.png"), dpi=150)
    plt.close()


def _fig5_regime_comparison_boxplot(all_results, output_dir):
    """Box plots comparing regimes on key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = list(all_results.keys())
    colors = [COLORS[n] for n in labels]

    # Truth ratio
    data = [[run[-1].truth_ratio for run in all_results[n]] for n in labels]
    bp = axes[0].boxplot(data, labels=[l[:10] for l in labels], patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axes[0].set_title("Truth Ratio")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Replication rate
    data = [[run[-1].successful_replication_rate for run in all_results[n]] for n in labels]
    bp = axes[1].boxplot(data, labels=[l[:10] for l in labels], patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axes[1].set_title("Replication Success Rate")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Retractions
    data = [[run[-1].retraction_count for run in all_results[n]] for n in labels]
    bp = axes[2].boxplot(data, labels=[l[:10] for l in labels], patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axes[2].set_title("Total Retractions")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Replication Regime Comparison (t=200, N=100 seeds)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms2_05_regime_boxplot.png"), dpi=150)
    plt.close()


def _fig6_truth_vs_replication(all_results, output_dir):
    """Scatter: truth ratio vs replication effort (investment)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, runs in all_results.items():
        rep_rates = [run[-1].replication_rate for run in runs]
        truths = [run[-1].truth_ratio for run in runs]
        ax.scatter(rep_rates, truths, label=name, color=COLORS[name], alpha=0.4, s=15)
        ax.scatter([np.mean(rep_rates)], [np.mean(truths)], color=COLORS[name],
                   s=100, edgecolors="black", linewidths=1.5, zorder=5)

    ax.set_xlabel("Replication Rate (fraction of papers replicated)")
    ax.set_ylabel("Final Truth Ratio")
    ax.set_title("Replication Effort vs Knowledge Quality")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms2_06_truth_vs_replication.png"), dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    N_SEEDS = 100
    N_LABS = 50
    TIME_STEPS = 200

    print("=" * 60)
    print("Meta-Science Experiment 2: Replication Crisis")
    print("=" * 60)
    print(f"  Labs: {N_LABS}, Steps: {TIME_STEPS}, Seeds: {N_SEEDS}")
    print(f"  Regimes: {list(REGIMES.keys())}")

    all_results = run_monte_carlo(n_seeds=N_SEEDS, n_labs=N_LABS, time_steps=TIME_STEPS)
    stats = compute_statistics(all_results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Regime':<20} {'Truth Ratio':>15} {'Rep Rate':>12} {'Retractions':>12}")
    print("-" * 59)
    for name, s in stats.items():
        tr = f"{s['truth_ratio_mean']:.3f} +/- {s['truth_ratio_std']:.3f}"
        rr = f"{s['replication_rate_mean']:.3f}"
        ret = f"{s['retraction_mean']:.1f}"
        print(f"{name:<20} {tr:>15} {rr:>12} {ret:>12}")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures_ms2")
    generate_figures(all_results, output_dir)

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    with open(os.path.join(results_dir, "ms2_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {results_dir}/ms2_stats.json")
