"""
Meta-Science Experiment 4: Open Science vs Closed Science
==========================================================
Tests the hypothesis: "Open science creates knowledge network effects
that boost community output, but creates a free-rider problem.
AI amplifies both the benefits and the free-rider dynamics."

5 Scenarios:
1. All Closed: all labs openness=0.0
2. Mixed: labs randomly assigned openness from uniform(0,1)
3. All Open: all labs openness=1.0
4. Free-riders: 20% closed in otherwise open community
5. AI+Open: open labs use AI-SciOps optimizer

Each scenario runs N=100 seeds for statistical validation.
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
from knowledge_network import KnowledgeNetwork


class OpenScienceEcosystem(MetaScienceEcosystem):
    """Extends MetaScienceEcosystem with knowledge network effects."""

    def __init__(
        self,
        openness_config: str = "mixed",
        scoop_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.knowledge_network = KnowledgeNetwork(decay_rate=0.01, boost_cap=2.0)
        self.scoop_rate = scoop_rate
        self.scooping_events = 0

        # Configure lab openness
        self._configure_openness(openness_config)

        # Track per-group metrics
        self.open_lab_output: list[float] = []
        self.closed_lab_output: list[float] = []
        self.knowledge_items_count: list[int] = []
        self.mean_boost: list[float] = []

    def _configure_openness(self, config: str) -> None:
        """Set openness levels for labs."""
        if config == "all_closed":
            for lab in self.labs:
                lab.openness = 0.0
        elif config == "all_open":
            for lab in self.labs:
                lab.openness = 1.0
        elif config == "mixed":
            for lab in self.labs:
                lab.openness = random.random()
        elif config == "free_riders":
            # 80% open, 20% closed
            n_closed = int(len(self.labs) * 0.2)
            for i, lab in enumerate(self.labs):
                lab.openness = 0.0 if i < n_closed else 0.9
        elif config == "ai_open":
            # AI labs are open, non-AI labs have varying openness
            for lab in self.labs:
                if lab.is_ai_lab:
                    lab.openness = 0.9
                else:
                    lab.openness = random.uniform(0.3, 0.7)

    def _step(self) -> None:
        """Override to add knowledge network effects."""
        t = self.time_step
        papers_this_step = []
        total_output = 0.0
        open_output = 0.0
        closed_output = 0.0

        # 1. Calculate knowledge boosts
        boosts = {}
        for lab in self.labs:
            boost = self.knowledge_network.knowledge_boost(t, lab.openness)
            boosts[lab.lab_id] = boost

        # 2. Run labs with knowledge boost
        for lab in self.labs:
            # Apply knowledge boost to input rate
            boosted_input = self.input_rate * boosts[lab.lab_id]
            output = self._run_lab_with_input(lab, t, boosted_input)
            total_output += output

            # Track open vs closed output
            if lab.openness > 0.5:
                open_output += output
            else:
                closed_output += output

            if output > self.paper_threshold:
                paper = self._generate_paper(lab, t, output)
                paper.openness = lab.openness

                # Scooping risk for open papers
                if lab.openness > 0.5 and random.random() < self.scoop_rate * lab.openness:
                    self.scooping_events += 1
                    continue  # Paper scooped, not submitted

                lab.papers_submitted.append(paper)
                papers_this_step.append(paper)

        # 3. Publication review
        published_this_step = []
        for paper in papers_this_step:
            if self.publication_channel.evaluate(paper):
                paper.is_published = True
                lab = self.labs[paper.lab_id]
                lab.papers_published.append(paper)
                self.knowledge_base.append(paper)
                published_this_step.append(paper)

                # Add to knowledge network
                self.knowledge_network.add_knowledge(paper, t)

        # 4. Update reputations
        self._update_reputations(published_this_step)

        # 5. Funding
        if t > 0 and t % self.funding_cycle == 0:
            self._redistribute_funding()

        # 6. Track extra metrics
        n_open = max(1, sum(1 for l in self.labs if l.openness > 0.5))
        n_closed = max(1, sum(1 for l in self.labs if l.openness <= 0.5))
        self.open_lab_output.append(open_output / n_open)
        self.closed_lab_output.append(closed_output / n_closed)
        self.knowledge_items_count.append(len(self.knowledge_network.items))
        self.mean_boost.append(np.mean(list(boosts.values())))

        # 7. Standard metrics
        self._collect_metrics(t, papers_this_step, published_this_step, total_output)

    def _run_lab_with_input(self, lab: LabState, time_step: int, input_rate: float) -> float:
        """Run a lab with a custom input rate (knowledge-boosted)."""
        # Apply AI assistance
        if lab.is_ai_lab:
            ai_progress = min(1.0, time_step / 100.0)
            for step in lab.pipeline:
                step.ai_assistance_level = min(
                    0.8 * ai_progress, step.config.ai_automatable * 0.9
                )
        else:
            for step in lab.pipeline:
                step.ai_assistance_level = 0.0

        # Optimize
        lab.pipeline = lab.optimizer.optimize(lab.pipeline, time_step, lab.funding)

        # Feed work through pipeline
        incoming = input_rate
        for step in lab.pipeline:
            output = step.step(incoming)
            incoming = output

        lab.cumulative_output += incoming
        return incoming


# ============================================================
# Experiment configuration
# ============================================================

SCENARIOS = {
    "All Closed": {"openness_config": "all_closed"},
    "Mixed": {"openness_config": "mixed"},
    "All Open": {"openness_config": "all_open"},
    "Free-riders": {"openness_config": "free_riders"},
    "AI + Open": {"openness_config": "ai_open", "ai_fraction": 0.5},
}

COLORS = {
    "All Closed": "#9E9E9E",
    "Mixed": "#FF9800",
    "All Open": "#4CAF50",
    "Free-riders": "#F44336",
    "AI + Open": "#9C27B0",
}


def run_single_scenario(scenario_name: str, params: dict, seed: int,
                        n_labs: int = 50, time_steps: int = 200):
    """Run a single scenario."""
    channel = PublicationChannel(
        name="Journal",
        acceptance_rate=0.2,
        novelty_bias=0.2,
        positive_result_bias=0.2,
        review_quality=0.7,
    )

    ai_fraction = params.pop("ai_fraction", 0.3)

    eco = OpenScienceEcosystem(
        n_labs=n_labs,
        publication_channel=channel,
        base_funding=6.0,
        input_rate=2.0,
        base_truth_rate=0.5,
        paper_threshold=0.3,
        funding_cycle=20,
        ai_fraction=ai_fraction,
        p_hacking_intensity=0.0,
        seed=seed,
        **params,
    )
    metrics = eco.run(time_steps=time_steps)
    return metrics, eco


def run_monte_carlo(n_seeds: int = 100, n_labs: int = 50,
                    time_steps: int = 200) -> dict:
    """Run all scenarios across N seeds."""
    all_results = {}
    for name, params in SCENARIOS.items():
        print(f"\n--- {name} ---")
        results = []
        extra_data = []
        for seed in range(n_seeds):
            params_copy = dict(params)
            metrics, eco = run_single_scenario(name, params_copy, seed, n_labs, time_steps)
            results.append(metrics)
            extra_data.append({
                "open_output": list(eco.open_lab_output),
                "closed_output": list(eco.closed_lab_output),
                "knowledge_items": list(eco.knowledge_items_count),
                "mean_boost": list(eco.mean_boost),
                "scooping_events": eco.scooping_events,
            })
            if (seed + 1) % 25 == 0:
                print(f"  Seed {seed + 1}/{n_seeds} done")
        all_results[name] = {"metrics": results, "extra": extra_data}
    return all_results


def compute_statistics(all_results: dict) -> dict:
    """Compute summary statistics."""
    stats = {}
    for name, data in all_results.items():
        runs = data["metrics"]
        extras = data["extra"]

        total_output = [sum(m.total_community_output for m in run) for run in runs]
        final_truth = [run[-1].truth_ratio for run in runs]
        scoops = [e["scooping_events"] for e in extras]
        final_boost = [e["mean_boost"][-1] if e["mean_boost"] else 1.0 for e in extras]
        final_items = [e["knowledge_items"][-1] if e["knowledge_items"] else 0 for e in extras]

        stats[name] = {
            "output_mean": float(np.mean(total_output)),
            "output_std": float(np.std(total_output)),
            "truth_ratio_mean": float(np.mean(final_truth)),
            "scoops_mean": float(np.mean(scoops)),
            "boost_mean": float(np.mean(final_boost)),
            "knowledge_items_mean": float(np.mean(final_items)),
        }
    return stats


# ============================================================
# Visualization
# ============================================================

def generate_figures(all_results: dict, output_dir: str) -> None:
    """Generate all figures for Experiment 4."""
    os.makedirs(output_dir, exist_ok=True)

    _fig1_community_output(all_results, output_dir)
    _fig2_freerider_dynamics(all_results, output_dir)
    _fig3_knowledge_network_growth(all_results, output_dir)
    _fig4_scooping_vs_boost(all_results, output_dir)
    _fig5_nash_equilibrium(all_results, output_dir)
    _fig6_summary_comparison(all_results, output_dir)

    print(f"\nFigures saved to {output_dir}")


def _fig1_community_output(all_results, output_dir):
    """Community total output over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in all_results.items():
        runs = data["metrics"]
        T = len(runs[0])
        mat = np.array([[m.total_community_output for m in run] for run in runs])
        # Cumulative
        cum_mat = np.cumsum(mat, axis=1)
        mean = cum_mat.mean(axis=0)
        se = cum_mat.std(axis=0) / np.sqrt(len(runs))
        x = np.arange(T)
        ax.plot(x, mean, label=name, color=COLORS[name], linewidth=2)
        ax.fill_between(x, mean - se, mean + se, color=COLORS[name], alpha=0.15)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Community Output")
    ax.set_title("Open Science: Community Output Over Time (N=100 seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms4_01_community_output.png"), dpi=150)
    plt.close()


def _fig2_freerider_dynamics(all_results, output_dir):
    """Per-lab output: open vs closed labs in mixed/free-rider scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, scenario in zip(axes, ["Mixed", "Free-riders"]):
        if scenario not in all_results:
            continue
        data = all_results[scenario]
        extras = data["extra"]
        T = len(extras[0]["open_output"])
        x = np.arange(T)

        open_mat = np.array([e["open_output"] for e in extras])
        closed_mat = np.array([e["closed_output"] for e in extras])

        open_cum = np.cumsum(open_mat, axis=1)
        closed_cum = np.cumsum(closed_mat, axis=1)

        ax.plot(x, open_cum.mean(axis=0), label="Open Labs", color="#4CAF50", linewidth=2)
        ax.plot(x, closed_cum.mean(axis=0), label="Closed Labs", color="#F44336", linewidth=2)

        ax.fill_between(x,
                        open_cum.mean(axis=0) - open_cum.std(axis=0) / np.sqrt(len(extras)),
                        open_cum.mean(axis=0) + open_cum.std(axis=0) / np.sqrt(len(extras)),
                        color="#4CAF50", alpha=0.15)
        ax.fill_between(x,
                        closed_cum.mean(axis=0) - closed_cum.std(axis=0) / np.sqrt(len(extras)),
                        closed_cum.mean(axis=0) + closed_cum.std(axis=0) / np.sqrt(len(extras)),
                        color="#F44336", alpha=0.15)

        ax.set_title(f"{scenario}: Open vs Closed Labs")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Cumulative Output per Lab")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Free-Rider Dynamics: Per-Lab Output Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms4_02_freerider_dynamics.png"), dpi=150)
    plt.close()


def _fig3_knowledge_network_growth(all_results, output_dir):
    """Knowledge network size over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in all_results.items():
        extras = data["extra"]
        T = len(extras[0]["knowledge_items"])
        mat = np.array([e["knowledge_items"] for e in extras])
        mean = mat.mean(axis=0)
        x = np.arange(T)
        ax.plot(x, mean, label=name, color=COLORS[name], linewidth=2)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Knowledge Items in Network")
    ax.set_title("Knowledge Network Growth Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms4_03_knowledge_network_growth.png"), dpi=150)
    plt.close()


def _fig4_scooping_vs_boost(all_results, output_dir):
    """Scooping events vs knowledge boost per scenario."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in all_results.items():
        extras = data["extra"]
        scoops = [e["scooping_events"] for e in extras]
        boosts = [e["mean_boost"][-1] if e["mean_boost"] else 1.0 for e in extras]
        ax.scatter(scoops, boosts, label=name, color=COLORS[name], alpha=0.4, s=15)
        ax.scatter([np.mean(scoops)], [np.mean(boosts)], color=COLORS[name],
                   s=100, edgecolors="black", linewidths=1.5, zorder=5)

    ax.set_xlabel("Scooping Events (total)")
    ax.set_ylabel("Mean Knowledge Boost at t=200")
    ax.set_title("Scooping Risk vs Knowledge Benefit Tradeoff")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms4_04_scooping_vs_boost.png"), dpi=150)
    plt.close()


def _fig5_nash_equilibrium(all_results, output_dir):
    """Compare total output across openness levels to find social optimum."""
    stats = compute_statistics(all_results)
    names = list(stats.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(names))
    outputs = [stats[n]["output_mean"] for n in names]
    colors = [COLORS[n] for n in names]
    errs = [stats[n]["output_std"] / np.sqrt(100) for n in names]

    bars = ax.bar(x_pos, outputs, color=colors, alpha=0.7, edgecolor="black",
                  yerr=errs, capsize=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Mean Total Community Output")
    ax.set_title("Open Science: Social Optimum vs Nash Equilibrium")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate
    max_idx = np.argmax(outputs)
    ax.annotate("Social Optimum", xy=(max_idx, outputs[max_idx]),
                xytext=(max_idx + 0.5, outputs[max_idx] * 1.05),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms4_05_nash_equilibrium.png"), dpi=150)
    plt.close()


def _fig6_summary_comparison(all_results, output_dir):
    """Summary comparison across all scenarios."""
    stats = compute_statistics(all_results)
    names = list(stats.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = [COLORS[n] for n in names]

    # Output
    vals = [stats[n]["output_mean"] for n in names]
    axes[0].barh(range(len(names)), vals, color=colors, alpha=0.7, edgecolor="black")
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].set_title("Total Output")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Knowledge Boost
    vals = [stats[n]["boost_mean"] for n in names]
    axes[1].barh(range(len(names)), vals, color=colors, alpha=0.7, edgecolor="black")
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names)
    axes[1].set_title("Mean Knowledge Boost")
    axes[1].grid(True, alpha=0.3, axis="x")

    # Scooping events
    vals = [stats[n]["scoops_mean"] for n in names]
    axes[2].barh(range(len(names)), vals, color=colors, alpha=0.7, edgecolor="black")
    axes[2].set_yticks(range(len(names)))
    axes[2].set_yticklabels(names)
    axes[2].set_title("Mean Scooping Events")
    axes[2].grid(True, alpha=0.3, axis="x")

    plt.suptitle("Open Science Scenarios: Multi-Dimensional Comparison (N=100 seeds)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ms4_06_summary_comparison.png"), dpi=150)
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    N_SEEDS = 100
    N_LABS = 50
    TIME_STEPS = 200

    print("=" * 60)
    print("Meta-Science Experiment 4: Open Science")
    print("=" * 60)
    print(f"  Labs: {N_LABS}, Steps: {TIME_STEPS}, Seeds: {N_SEEDS}")
    print(f"  Scenarios: {list(SCENARIOS.keys())}")

    all_results = run_monte_carlo(n_seeds=N_SEEDS, n_labs=N_LABS, time_steps=TIME_STEPS)
    stats = compute_statistics(all_results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<15} {'Output':>10} {'Truth':>8} {'Boost':>8} {'Scoops':>8} {'Items':>8}")
    print("-" * 57)
    for name, s in stats.items():
        print(f"{name:<15} {s['output_mean']:>10.1f} {s['truth_ratio_mean']:>8.3f} "
              f"{s['boost_mean']:>8.3f} {s['scoops_mean']:>8.1f} {s['knowledge_items_mean']:>8.0f}")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures_ms4")
    generate_figures(all_results, output_dir)

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    with open(os.path.join(results_dir, "ms4_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {results_dir}/ms4_stats.json")
