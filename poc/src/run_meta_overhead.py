"""
Meta-Overhead Experiment: Can AI Optimize Its Own Management Cost?
==================================================================
Runs 5 variants of AI-driven management overhead optimization,
each facing a different fundamental challenge.

Also runs Kanban-SciOps (fixed overhead) and Baseline for comparison.

Outputs:
- Console comparison table
- 5 figures exploring each challenge
"""

import os
import random
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scientific_process import create_default_pipeline, ProcessStep
from optimizers import BaselineOptimizer
from advanced_optimizers import KanbanSciOpsOptimizer
from management_overhead import get_overhead_profile, OverheadProfile
from simulator import Simulator, SimulationResult, TimeStepMetrics
from meta_overhead_optimizer import (
    MetaAIOracleOptimizer,
    MetaAINoisyOptimizer,
    MetaAIDelayedOptimizer,
    MetaAIRecursiveOptimizer,
    MetaAITrustDecayOptimizer,
)


class MetaOverheadSimulator(Simulator):
    """Simulator for meta-AI overhead optimizers.

    Unlike OverheadAwareSimulator (which uses a fixed profile),
    this simulator queries the optimizer for its current overhead profile
    each step, allowing the profile to change dynamically.
    """

    def __init__(self, optimizer, total_resources=6.0, input_rate=2.0, seed=None):
        super().__init__(optimizer, total_resources, input_rate, seed)
        self.overhead_per_step: list[float] = []
        self.meta_cost_per_step: list[float] = []
        self.total_overhead = 0.0
        self.total_meta_cost = 0.0
        self.trust_per_step: list[float] = []
        self.intensity_per_step: list[float] = []
        self.profile_per_step: list[dict] = []

    def run(self, time_steps: int = 100) -> SimulationResult:
        prev_action_count = 0

        for t in range(time_steps):
            # Get dynamic overhead profile from the meta-AI
            if hasattr(self.optimizer, 'get_overhead_profile'):
                profile = self.optimizer.get_overhead_profile()
                meta_cost = self.optimizer.get_meta_cost()
            else:
                profile = get_overhead_profile(self.optimizer.name)
                meta_cost = 0.0

            # Calculate overhead
            current_action_count = len(self.optimizer.actions)
            new_actions = current_action_count - prev_action_count
            prev_action_count = current_action_count

            is_transition = any(
                a.action_type.endswith("-Init") or a.action_type.endswith("-Transition")
                for a in self.optimizer.actions[-new_actions:]
            ) if new_actions > 0 else False

            overhead = profile.compute_overhead(
                time_step=t,
                num_processes=len(self.pipeline),
                num_actions_this_step=new_actions,
                is_stage_transition=is_transition,
            )

            total_cost_this_step = overhead + meta_cost
            self.overhead_per_step.append(overhead)
            self.meta_cost_per_step.append(meta_cost)
            self.total_overhead += overhead
            self.total_meta_cost += meta_cost

            # Record trust and intensity if available
            if hasattr(self.optimizer, 'state') and hasattr(self.optimizer.state, 'human_trust'):
                self.trust_per_step.append(self.optimizer.state.human_trust)
            else:
                self.trust_per_step.append(1.0)

            if hasattr(self.optimizer, 'optimization_intensity'):
                self.intensity_per_step.append(self.optimizer.optimization_intensity)
            else:
                self.intensity_per_step.append(0.0)

            self.profile_per_step.append({
                "base_cost": profile.base_cost,
                "human_coordination_cost": profile.human_coordination_cost,
                "ai_infrastructure_cost": profile.ai_infrastructure_cost,
                "complexity_scaling": profile.complexity_scaling,
            })

            # Effective resources
            effective_resources = max(0.5, self.total_resources - total_cost_this_step)

            # Let optimizer adjust the pipeline
            self.pipeline = self.optimizer.optimize(
                self.pipeline, t, effective_resources
            )

            # Feed work
            incoming = self.input_rate
            for step in self.pipeline:
                output = step.step(incoming)
                incoming = output

            system_output = incoming
            self.cumulative_output += system_output

            bottleneck = min(self.pipeline, key=lambda p: p.effective_throughput())
            metrics = TimeStepMetrics(
                time_step=t,
                process_throughputs={
                    p.config.name: round(p.effective_throughput(), 4)
                    for p in self.pipeline
                },
                process_wip={
                    p.config.name: round(p.work_in_progress, 4)
                    for p in self.pipeline
                },
                process_backlogs={
                    p.config.name: round(p.human_review_backlog, 4)
                    for p in self.pipeline
                },
                system_throughput=round(system_output, 4),
                cumulative_output=round(self.cumulative_output, 4),
                bottleneck_process=bottleneck.config.name,
                bottleneck_throughput=round(bottleneck.effective_throughput(), 4),
                total_rework=round(sum(p.rework_units for p in self.pipeline), 4),
                total_failures=round(sum(p.failed_units for p in self.pipeline), 4),
            )
            self.metrics.append(metrics)

        final_state = {}
        for step in self.pipeline:
            final_state[step.config.name] = {
                "throughput": round(step.effective_throughput(), 4),
                "completed_units": round(step.completed_units, 4),
                "failed_units": round(step.failed_units, 4),
                "rework_units": round(step.rework_units, 4),
                "human_review_backlog": round(step.human_review_backlog, 4),
                "ai_assistance_level": round(step.ai_assistance_level, 4),
                "allocated_resources": round(step.allocated_resources, 4),
            }

        return SimulationResult(
            optimizer_name=self.optimizer.name,
            total_time_steps=len(self.metrics),
            total_output=round(self.cumulative_output, 4),
            metrics=self.metrics,
            optimization_actions=[
                {"time_step": a.time_step, "target": a.target_process,
                 "type": a.action_type, "description": a.description}
                for a in self.optimizer.actions
            ],
            final_state=final_state,
        )


# ============================================================
# Fixed-overhead simulator for comparison (Kanban, Baseline)
# ============================================================
class FixedOverheadSimulator(Simulator):
    """Run with fixed overhead profile (same as run_overhead_comparison.py)."""

    def __init__(self, optimizer, total_resources=6.0, input_rate=2.0, seed=None):
        super().__init__(optimizer, total_resources, input_rate, seed)
        self.overhead_profile = get_overhead_profile(optimizer.name)
        self.overhead_per_step: list[float] = []
        self.meta_cost_per_step: list[float] = []
        self.total_overhead = 0.0
        self.total_meta_cost = 0.0
        self.trust_per_step: list[float] = []
        self.intensity_per_step: list[float] = []
        self.profile_per_step: list[dict] = []

    def run(self, time_steps: int = 100) -> SimulationResult:
        prev_action_count = 0

        for t in range(time_steps):
            current_action_count = len(self.optimizer.actions)
            new_actions = current_action_count - prev_action_count
            prev_action_count = current_action_count

            is_transition = any(
                a.action_type.endswith("-Init") or a.action_type.endswith("-Transition")
                for a in self.optimizer.actions[-new_actions:]
            ) if new_actions > 0 else False

            overhead = self.overhead_profile.compute_overhead(
                time_step=t, num_processes=len(self.pipeline),
                num_actions_this_step=new_actions, is_stage_transition=is_transition,
            )
            self.overhead_per_step.append(overhead)
            self.meta_cost_per_step.append(0.0)
            self.total_overhead += overhead
            self.trust_per_step.append(1.0)
            self.intensity_per_step.append(0.0)
            self.profile_per_step.append({
                "base_cost": self.overhead_profile.base_cost,
                "human_coordination_cost": self.overhead_profile.human_coordination_cost,
                "ai_infrastructure_cost": self.overhead_profile.ai_infrastructure_cost,
                "complexity_scaling": self.overhead_profile.complexity_scaling,
            })

            effective_resources = max(0.5, self.total_resources - overhead)
            self.pipeline = self.optimizer.optimize(self.pipeline, t, effective_resources)

            incoming = self.input_rate
            for step in self.pipeline:
                output = step.step(incoming)
                incoming = output

            system_output = incoming
            self.cumulative_output += system_output

            bottleneck = min(self.pipeline, key=lambda p: p.effective_throughput())
            metrics = TimeStepMetrics(
                time_step=t,
                process_throughputs={p.config.name: round(p.effective_throughput(), 4) for p in self.pipeline},
                process_wip={p.config.name: round(p.work_in_progress, 4) for p in self.pipeline},
                process_backlogs={p.config.name: round(p.human_review_backlog, 4) for p in self.pipeline},
                system_throughput=round(system_output, 4),
                cumulative_output=round(self.cumulative_output, 4),
                bottleneck_process=bottleneck.config.name,
                bottleneck_throughput=round(bottleneck.effective_throughput(), 4),
                total_rework=round(sum(p.rework_units for p in self.pipeline), 4),
                total_failures=round(sum(p.failed_units for p in self.pipeline), 4),
            )
            self.metrics.append(metrics)

        final_state = {}
        for step in self.pipeline:
            final_state[step.config.name] = {
                "throughput": round(step.effective_throughput(), 4),
                "completed_units": round(step.completed_units, 4),
                "failed_units": round(step.failed_units, 4),
                "rework_units": round(step.rework_units, 4),
                "human_review_backlog": round(step.human_review_backlog, 4),
                "ai_assistance_level": round(step.ai_assistance_level, 4),
                "allocated_resources": round(step.allocated_resources, 4),
            }

        return SimulationResult(
            optimizer_name=self.optimizer.name,
            total_time_steps=len(self.metrics),
            total_output=round(self.cumulative_output, 4),
            metrics=self.metrics,
            optimization_actions=[
                {"time_step": a.time_step, "target": a.target_process,
                 "type": a.action_type, "description": a.description}
                for a in self.optimizer.actions
            ],
            final_state=final_state,
        )


def run_all(seed=42):
    """Run all optimizers and collect results."""
    optimizers_and_sims = [
        ("Baseline", BaselineOptimizer(), FixedOverheadSimulator),
        ("Kanban (Fixed OH)", KanbanSciOpsOptimizer(), FixedOverheadSimulator),
        ("MetaAI-Oracle", MetaAIOracleOptimizer(), MetaOverheadSimulator),
        ("MetaAI-Noisy", MetaAINoisyOptimizer(), MetaOverheadSimulator),
        ("MetaAI-Delayed", MetaAIDelayedOptimizer(), MetaOverheadSimulator),
        ("MetaAI-Recursive", MetaAIRecursiveOptimizer(), MetaOverheadSimulator),
        ("MetaAI-TrustDecay", MetaAITrustDecayOptimizer(), MetaOverheadSimulator),
    ]

    all_results = {}
    all_sims = {}

    for label, opt, SimClass in optimizers_and_sims:
        random.seed(seed)
        sim = SimClass(optimizer=opt, seed=seed)
        result = sim.run(time_steps=100)
        all_results[label] = result
        all_sims[label] = sim

    return all_results, all_sims


def print_results(all_results, all_sims):
    """Print comparison table."""
    baseline_output = all_results["Baseline"].total_output

    print(f"\n{'='*105}")
    print("META-OVERHEAD ANALYSIS: Challenges of AI-Driven Management Cost Optimization")
    print(f"{'='*105}")
    print(f"{'Variant':<35} {'Output':>8} {'MgmtOH':>8} {'MetaOH':>8} "
          f"{'TotalOH':>8} {'Net Gain':>10} {'Challenge':>20}")
    print(f"{'-'*105}")

    rows = sorted(all_results.keys(), key=lambda k: all_results[k].total_output, reverse=True)

    challenges = {
        "Baseline": "N/A",
        "Kanban (Fixed OH)": "Fixed cost",
        "MetaAI-Oracle": "Upper bound",
        "MetaAI-Noisy": "Credit assign.",
        "MetaAI-Delayed": "Oscillation",
        "MetaAI-Recursive": "Self-ref. cost",
        "MetaAI-TrustDecay": "Human factor",
    }

    for label in rows:
        r = all_results[label]
        sim = all_sims[label]
        mgmt_oh = sim.total_overhead
        meta_oh = sim.total_meta_cost if hasattr(sim, 'total_meta_cost') else 0.0
        total_oh = mgmt_oh + meta_oh
        gain_pct = (r.total_output - baseline_output) / baseline_output * 100

        marker = " ***" if r.total_output == max(rr.total_output for rr in all_results.values()) else ""
        print(f"{label:<35} {r.total_output:>8.1f} {mgmt_oh:>8.1f} {meta_oh:>8.1f} "
              f"{total_oh:>8.1f} {gain_pct:>+9.1f}% {challenges.get(label, ''):>20}{marker}")

    print(f"\nMgmtOH = management process overhead, MetaOH = AI meta-optimization overhead")
    print(f"TotalOH = MgmtOH + MetaOH, Net Gain = improvement vs Baseline")


def generate_figures(all_results, all_sims, output_dir):
    """Generate 5 analysis figures."""
    os.makedirs(output_dir, exist_ok=True)

    COLORS = {
        "Baseline": "#bdc3c7",
        "Kanban (Fixed OH)": "#2ecc71",
        "MetaAI-Oracle": "#3498db",
        "MetaAI-Noisy": "#e67e22",
        "MetaAI-Delayed": "#9b59b6",
        "MetaAI-Recursive": "#e74c3c",
        "MetaAI-TrustDecay": "#1abc9c",
    }

    ORDER = ["Baseline", "Kanban (Fixed OH)", "MetaAI-Oracle",
             "MetaAI-Noisy", "MetaAI-Delayed",
             "MetaAI-Recursive", "MetaAI-TrustDecay"]
    ordered = [k for k in ORDER if k in all_results]

    # === Figure 1: Output + overhead comparison bars ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    x = np.arange(len(ordered))
    outputs = [all_results[k].total_output for k in ordered]
    mgmt_ohs = [all_sims[k].total_overhead for k in ordered]
    meta_ohs = [all_sims[k].total_meta_cost if hasattr(all_sims[k], 'total_meta_cost') else 0.0 for k in ordered]
    colors = [COLORS[k] for k in ordered]

    ax1.bar(x, outputs, color=colors, edgecolor="black", linewidth=0.5)
    for i, val in enumerate(outputs):
        ax1.text(i, val + 0.5, f"{val:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([k.replace("MetaAI-", "Meta:\n") for k in ordered], fontsize=8)
    ax1.set_ylabel("Total Research Output")
    ax1.set_title("Research Output (With Overhead)")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x, mgmt_ohs, color=colors, alpha=0.6, label="Management OH",
            edgecolor="black", linewidth=0.5)
    ax2.bar(x, meta_ohs, bottom=mgmt_ohs, color=colors, alpha=1.0,
            label="Meta-AI OH", edgecolor="black", linewidth=0.5, hatch="//")
    for i in range(len(ordered)):
        total = mgmt_ohs[i] + meta_ohs[i]
        ax2.text(i, total + 0.5, f"{total:.1f}", ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([k.replace("MetaAI-", "Meta:\n") for k in ordered], fontsize=8)
    ax2.set_ylabel("Total Overhead Cost")
    ax2.set_title("Overhead Breakdown: Management + Meta-AI")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Meta-Overhead Analysis: Can AI Optimize Its Own Management Cost?", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v4_01_meta_output_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Figure 2: Cumulative output over time ===
    fig, ax = plt.subplots(figsize=(12, 7))
    for k in ordered:
        data = all_results[k]
        steps = [m.time_step for m in data.metrics]
        cumul = [m.cumulative_output for m in data.metrics]
        lw = 2.5 if "Oracle" in k or "TrustDecay" in k else 1.5
        ls = "-" if "MetaAI" in k else "--"
        ax.plot(steps, cumul, label=k, color=COLORS[k], linewidth=lw, linestyle=ls)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Cumulative Research Output", fontsize=12)
    ax.set_title("Cumulative Output: Meta-AI Variants vs. Fixed Overhead", fontsize=14)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v4_02_meta_cumulative.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Figure 3: Dynamic overhead profiles over time ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    meta_keys = [k for k in ordered if "MetaAI" in k]
    components = ["base_cost", "human_coordination_cost",
                  "ai_infrastructure_cost", "complexity_scaling"]
    titles = ["Base Cost", "Human Coordination Cost",
              "AI Infrastructure Cost", "Complexity Scaling"]

    for idx, (comp, title) in enumerate(zip(components, titles)):
        ax = axes[idx // 2][idx % 2]
        for k in meta_keys:
            sim = all_sims[k]
            values = [p[comp] for p in sim.profile_per_step]
            ax.plot(range(len(values)), values, label=k.replace("MetaAI-", ""),
                    color=COLORS[k], linewidth=1.5)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Cost")
        ax.set_title(f"Dynamic {title}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("How Each Meta-AI Variant Adjusts Overhead Components", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v4_03_meta_profiles.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Figure 4: Challenge-specific metrics ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Trust decay (MetaAI-TrustDecay)
    ax = axes[0]
    if "MetaAI-TrustDecay" in all_sims:
        sim = all_sims["MetaAI-TrustDecay"]
        ax.plot(range(len(sim.trust_per_step)), sim.trust_per_step,
                color=COLORS["MetaAI-TrustDecay"], linewidth=2, label="Human Trust")
        human_coords = [p["human_coordination_cost"] for p in sim.profile_per_step]
        ax2 = ax.twinx()
        ax2.plot(range(len(human_coords)), human_coords,
                 color="gray", linewidth=1.5, linestyle="--", label="Human Coord Cost")
        ax2.set_ylabel("Coordination Cost", color="gray")
        ax2.legend(loc="center right", fontsize=8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Trust Level", color=COLORS["MetaAI-TrustDecay"])
    ax.set_title("Challenge: Trust Decay")
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(y=0.7, color="red", linestyle=":", alpha=0.5, label="Critical trust threshold")
    ax.grid(True, alpha=0.3)

    # Panel B: Recursive cost (MetaAI-Recursive)
    ax = axes[1]
    if "MetaAI-Recursive" in all_sims:
        sim = all_sims["MetaAI-Recursive"]
        ax.plot(range(len(sim.overhead_per_step)), sim.overhead_per_step,
                color=COLORS["MetaAI-Recursive"], linewidth=1.5, label="Management OH")
        ax.plot(range(len(sim.meta_cost_per_step)), sim.meta_cost_per_step,
                color=COLORS["MetaAI-Recursive"], linewidth=1.5, linestyle="--",
                label="Meta-AI OH")
        total = [a + b for a, b in zip(sim.overhead_per_step, sim.meta_cost_per_step)]
        ax.fill_between(range(len(total)), total, alpha=0.2,
                        color=COLORS["MetaAI-Recursive"])
        ax.plot(range(len(total)), total, color="black", linewidth=1,
                linestyle=":", label="Total OH")
        # Show intensity on twin axis
        ax2 = ax.twinx()
        ax2.plot(range(len(sim.intensity_per_step)), sim.intensity_per_step,
                 color="purple", linewidth=1, alpha=0.5, label="Opt. Intensity")
        ax2.set_ylabel("Optimization Intensity", color="purple")
        ax2.legend(loc="center right", fontsize=8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Overhead Cost / Step")
    ax.set_title("Challenge: Recursive Self-Referential Cost")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Overhead evolution comparison (all variants)
    ax = axes[2]
    for k in meta_keys:
        sim = all_sims[k]
        total_oh = [a + b for a, b in zip(sim.overhead_per_step, sim.meta_cost_per_step)]
        ax.plot(range(len(total_oh)), total_oh,
                label=k.replace("MetaAI-", ""), color=COLORS[k], linewidth=1.5)
    # Also show Kanban fixed
    if "Kanban (Fixed OH)" in all_sims:
        sim = all_sims["Kanban (Fixed OH)"]
        ax.plot(range(len(sim.overhead_per_step)), sim.overhead_per_step,
                label="Kanban (Fixed)", color=COLORS["Kanban (Fixed OH)"],
                linewidth=1.5, linestyle="--")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Total OH / Step")
    ax.set_title("Overhead Trajectory: Meta-AI vs. Fixed")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Challenge-Specific Analysis", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v4_04_meta_challenges.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Figure 5: Efficiency frontier ===
    fig, ax = plt.subplots(figsize=(10, 7))

    baseline_output = all_results["Baseline"].total_output
    non_baseline = [k for k in ordered if k != "Baseline"]

    gains = [all_results[k].total_output - baseline_output for k in non_baseline]
    total_ohs = [
        all_sims[k].total_overhead + (all_sims[k].total_meta_cost if hasattr(all_sims[k], 'total_meta_cost') else 0.0)
        for k in non_baseline
    ]

    for i, k in enumerate(non_baseline):
        ax.scatter(total_ohs[i], gains[i], c=COLORS[k], s=200,
                   edgecolors="black", linewidths=1, zorder=5)
        ax.annotate(k.replace("MetaAI-", "Meta:\n"),
                    (total_ohs[i], gains[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    # Iso-efficiency lines
    max_oh = max(total_ohs) * 1.2
    for eff in [0.2, 0.5, 1.0]:
        xs = np.linspace(0, max_oh, 100)
        ax.plot(xs, xs * eff, "--", color="gray", alpha=0.3, linewidth=0.8)
        ax.text(max_oh, max_oh * eff, f"eff={eff:.1f}", fontsize=7, color="gray")

    ax.set_xlabel("Total Overhead (Management + Meta-AI)", fontsize=12)
    ax.set_ylabel("Output Gain vs. Baseline", fontsize=12)
    ax.set_title("Efficiency Frontier: Is Dynamic Overhead Optimization Worth It?", fontsize=14)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v4_05_meta_efficiency.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures_v4")

    print("Running Meta-Overhead Experiment...")
    print("=" * 60)

    all_results, all_sims = run_all(seed=42)
    print_results(all_results, all_sims)
    generate_figures(all_results, all_sims, output_dir)
