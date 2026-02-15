"""
Overhead-Aware Comparison Experiment
======================================
Runs all strategies both WITH and WITHOUT management overhead,
demonstrating the true cost-benefit of each management approach.
"""

import json
import os
import random
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scientific_process import create_default_pipeline, ProcessStep
from optimizers import BaselineOptimizer, TOCPDCAOptimizer, AISciOpsOptimizer
from advanced_optimizers import KanbanSciOpsOptimizer, HolisticSciOpsOptimizer
from pm_optimizers import AgileScrumOptimizer, LeanSciOpsOptimizer, SixSigmaSciOpsOptimizer
from management_overhead import get_overhead_profile, OverheadProfile
from simulator import Simulator, SimulationResult, TimeStepMetrics


class OverheadAwareSimulator(Simulator):
    """Simulator that deducts management overhead from available resources."""

    def __init__(self, optimizer, total_resources=6.0, input_rate=2.0, seed=None):
        super().__init__(optimizer, total_resources, input_rate, seed)
        self.overhead_profile = get_overhead_profile(optimizer.name)
        self.total_overhead_consumed = 0.0
        self.overhead_per_step: list[float] = []

    def run(self, time_steps: int = 100) -> SimulationResult:
        prev_action_count = 0

        for t in range(time_steps):
            # Calculate management overhead
            current_action_count = len(self.optimizer.actions)
            new_actions = current_action_count - prev_action_count
            prev_action_count = current_action_count

            is_transition = any(
                a.action_type.endswith("-Init") or a.action_type.endswith("-Transition")
                for a in self.optimizer.actions[-new_actions:]
            ) if new_actions > 0 else False

            overhead = self.overhead_profile.compute_overhead(
                time_step=t,
                num_processes=len(self.pipeline),
                num_actions_this_step=new_actions,
                is_stage_transition=is_transition,
            )
            self.overhead_per_step.append(overhead)
            self.total_overhead_consumed += overhead

            # Effective resources = total - overhead
            effective_resources = max(0.5, self.total_resources - overhead)

            # Let optimizer adjust the pipeline (with reduced resources)
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


def create_all_optimizers():
    return [
        BaselineOptimizer(),
        TOCPDCAOptimizer(pdca_cycle_length=10),
        AgileScrumOptimizer(sprint_length=8),
        LeanSciOpsOptimizer(),
        SixSigmaSciOpsOptimizer(dmaic_cycle_length=20),
        AISciOpsOptimizer(),
        KanbanSciOpsOptimizer(),
        HolisticSciOpsOptimizer(),
    ]


def run_experiment(with_overhead: bool, seed: int = 42):
    """Run all strategies with or without overhead."""
    results = {}
    overhead_data = {}

    for opt in create_all_optimizers():
        random.seed(seed)

        if with_overhead:
            sim = OverheadAwareSimulator(optimizer=opt, seed=seed)
        else:
            sim = Simulator(optimizer=opt, seed=seed)

        result = sim.run(time_steps=100)
        results[opt.name] = result

        if with_overhead and isinstance(sim, OverheadAwareSimulator):
            overhead_data[opt.name] = {
                "total_overhead": round(sim.total_overhead_consumed, 2),
                "avg_overhead_per_step": round(sim.total_overhead_consumed / 100, 3),
                "overhead_per_step": [round(o, 4) for o in sim.overhead_per_step],
            }
        else:
            overhead_data[opt.name] = {
                "total_overhead": 0.0,
                "avg_overhead_per_step": 0.0,
                "overhead_per_step": [0.0] * 100,
            }

    return results, overhead_data


def print_comparison(results_no_oh, results_with_oh, overhead_data):
    """Print full comparison."""
    baseline = results_no_oh["Baseline (No Optimization)"].total_output

    print(f"\n{'='*95}")
    print("MANAGEMENT OVERHEAD ANALYSIS: Impact of Management Cost on Research Output")
    print(f"{'='*95}")
    print(f"{'Strategy':<35} {'No OH':>8} {'With OH':>8} {'OH Cost':>8} "
          f"{'OH/step':>8} {'Net Gain':>10} {'Eff.':>6}")
    print(f"{'-'*95}")

    strategies = sorted(
        results_with_oh.keys(),
        key=lambda n: results_with_oh[n].total_output,
        reverse=True,
    )

    for name in strategies:
        no_oh = results_no_oh[name].total_output
        with_oh = results_with_oh[name].total_output
        oh_total = overhead_data[name]["total_overhead"]
        oh_step = overhead_data[name]["avg_overhead_per_step"]
        net_gain_pct = (with_oh - baseline) / baseline * 100
        # Efficiency = output gain per unit of overhead consumed
        raw_gain = with_oh - baseline
        efficiency = raw_gain / oh_total if oh_total > 0 else float('inf')

        marker = " ***" if with_oh == max(r.total_output for r in results_with_oh.values()) else ""
        print(f"{name:<35} {no_oh:>8.1f} {with_oh:>8.1f} {oh_total:>8.1f} "
              f"{oh_step:>8.3f} {net_gain_pct:>+9.1f}% {efficiency:>6.1f}{marker}")

    print(f"\nOH Cost = total resources consumed by management over 100 steps")
    print(f"OH/step = average overhead per step (out of 6.0 total resources)")
    print(f"Net Gain = improvement vs Baseline (WITH overhead)")
    print(f"Eff. = output gain per unit overhead (higher = more efficient management)")


def generate_figures(results_no_oh, results_with_oh, overhead_data, output_dir):
    """Generate comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    COLORS = {
        "Baseline (No Optimization)": "#bdc3c7",
        "TOC + PDCA": "#3498db",
        "Agile-Scrum": "#1abc9c",
        "Lean-SciOps": "#27ae60",
        "SixSigma-SciOps": "#8e44ad",
        "AI-SciOps (Autonomous Optimization)": "#e74c3c",
        "Kanban-SciOps (Pull-based Flow)": "#2ecc71",
        "Holistic-SciOps (Integrated)": "#9b59b6",
    }
    SHORT = {
        "Baseline (No Optimization)": "Baseline",
        "TOC + PDCA": "TOC+\nPDCA",
        "Agile-Scrum": "Agile\nScrum",
        "Lean-SciOps": "Lean",
        "SixSigma-SciOps": "Six\nSigma",
        "AI-SciOps (Autonomous Optimization)": "AI-SciOps\n(Orig)",
        "Kanban-SciOps (Pull-based Flow)": "Kanban\nSciOps",
        "Holistic-SciOps (Integrated)": "Holistic\nSciOps",
    }

    ORDER = [
        "Baseline (No Optimization)", "TOC + PDCA", "Agile-Scrum",
        "Lean-SciOps", "SixSigma-SciOps",
        "AI-SciOps (Autonomous Optimization)",
        "Kanban-SciOps (Pull-based Flow)", "Holistic-SciOps (Integrated)",
    ]
    ordered = [n for n in ORDER if n in results_with_oh]

    # --- Figure 1: Output comparison (with/without overhead) ---
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(ordered))
    width = 0.35

    vals_no = [results_no_oh[n].total_output for n in ordered]
    vals_oh = [results_with_oh[n].total_output for n in ordered]
    colors = [COLORS.get(n, "gray") for n in ordered]
    labels = [SHORT.get(n, n) for n in ordered]

    bars1 = ax.bar(x - width / 2, vals_no, width, label="Without Overhead",
                   color=colors, alpha=0.5, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, vals_oh, width, label="With Overhead",
                   color=colors, alpha=1.0, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars2, vals_oh):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars1, vals_no):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Total Research Output", fontsize=12)
    ax.set_title("Research Output: With vs. Without Management Overhead", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v3_01_overhead_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Overhead cost breakdown ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    oh_totals = [overhead_data[n]["total_overhead"] for n in ordered]
    oh_steps = [overhead_data[n]["avg_overhead_per_step"] for n in ordered]

    axes[0].barh(range(len(ordered)), oh_totals, color=colors, height=0.6)
    axes[0].set_yticks(range(len(ordered)))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_xlabel("Total Overhead Cost (resource-steps)")
    axes[0].set_title("Total Management Overhead")
    for i, val in enumerate(oh_totals):
        axes[0].text(val + 0.3, i, f"{val:.1f}", va="center", fontsize=8)

    axes[1].barh(range(len(ordered)), oh_steps, color=colors, height=0.6)
    axes[1].set_yticks(range(len(ordered)))
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlabel("Average Overhead / Step (out of 6.0)")
    axes[1].set_title("Average Overhead per Time Step")
    for i, val in enumerate(oh_steps):
        pct = val / 6.0 * 100
        axes[1].text(val + 0.01, i, f"{val:.2f} ({pct:.0f}%)", va="center", fontsize=8)

    fig.suptitle("Management Overhead Cost Analysis", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v3_02_overhead_cost.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Efficiency (output gain / overhead cost) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    baseline_output = results_with_oh["Baseline (No Optimization)"].total_output

    non_baseline = [n for n in ordered if n != "Baseline (No Optimization)"]
    gains = [results_with_oh[n].total_output - baseline_output for n in non_baseline]
    costs = [overhead_data[n]["total_overhead"] for n in non_baseline]
    nb_colors = [COLORS.get(n, "gray") for n in non_baseline]
    nb_labels = [SHORT.get(n, n).replace("\n", " ") for n in non_baseline]

    scatter = ax.scatter(costs, gains, c=nb_colors, s=200, edgecolors="black",
                         linewidths=1, zorder=5)
    for i, label in enumerate(nb_labels):
        ax.annotate(label, (costs[i], gains[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    # Efficiency lines (iso-efficiency)
    max_cost = max(costs) * 1.2
    for eff in [0.2, 0.5, 1.0, 2.0]:
        xs = np.linspace(0, max_cost, 100)
        ys = xs * eff
        ax.plot(xs, ys, "--", color="gray", alpha=0.3, linewidth=0.8)
        ax.text(max_cost, max_cost * eff, f"eff={eff:.1f}", fontsize=7, color="gray")

    ax.set_xlabel("Total Management Overhead Cost", fontsize=12)
    ax.set_ylabel("Output Gain vs. Baseline", fontsize=12)
    ax.set_title("Management Efficiency: Output Gain per Unit Overhead", fontsize=14)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v3_03_efficiency.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 4: Cumulative output with overhead ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for name in ordered:
        data = results_with_oh[name]
        steps = [m.time_step for m in data.metrics]
        cumul = [m.cumulative_output for m in data.metrics]
        lw = 2.5 if name in ("Holistic-SciOps (Integrated)", "Lean-SciOps") else 1.5
        ls = "-" if "SciOps" in name or "Baseline" in name or "Lean" in name else "--"
        ax.plot(steps, cumul, label=SHORT.get(name, name).replace("\n", " "),
                color=COLORS.get(name, "gray"), linewidth=lw, linestyle=ls)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Cumulative Research Output", fontsize=12)
    ax.set_title("Cumulative Output (With Management Overhead)", fontsize=14)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v3_04_cumulative_with_oh.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures_v3")

    print("=" * 60)
    print("Running WITHOUT management overhead...")
    print("=" * 60)
    results_no_oh, _ = run_experiment(with_overhead=False, seed=42)
    for name, r in sorted(results_no_oh.items(), key=lambda x: -x[1].total_output):
        print(f"  {name:<40} {r.total_output:>8.2f}")

    print("\n" + "=" * 60)
    print("Running WITH management overhead...")
    print("=" * 60)
    results_with_oh, overhead_data = run_experiment(with_overhead=True, seed=42)
    for name, r in sorted(results_with_oh.items(), key=lambda x: -x[1].total_output):
        oh = overhead_data[name]["total_overhead"]
        print(f"  {name:<40} {r.total_output:>8.2f}  (OH: {oh:.1f})")

    print_comparison(results_no_oh, results_with_oh, overhead_data)
    generate_figures(results_no_oh, results_with_oh, overhead_data, output_dir)
