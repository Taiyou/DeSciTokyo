"""
Individual Methodology Comparison Experiment
=============================================
Compares 6 methodologies each running INDEPENDENTLY:
1. Baseline (no optimization)
2. TOC (Theory of Constraints) — pure, no PDCA, no AI
3. PDCA (Plan-Do-Check-Act) — pure, no TOC, no AI
4. Agile (Sprint-based) — pure, no AI
5. Kanban (Pull-based Flow) — pure, no AI
6. AI-SciOps (Autonomous Optimization) — full AI-driven approach

This isolates the contribution of each methodology framework.
"""

import json
import os
import random
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from scientific_process import create_default_pipeline
from optimizers import BaselineOptimizer, AISciOpsOptimizer
from individual_optimizers import (
    TOCOnlyOptimizer,
    PDCAOnlyOptimizer,
    KanbanOnlyOptimizer,
    AgileOnlyOptimizer,
)
from simulator import Simulator, SimulationResult


# ─── Configuration ───

STRATEGY_ORDER = [
    "Baseline (No Optimization)",
    "TOC (Theory of Constraints)",
    "PDCA (Plan-Do-Check-Act)",
    "Agile (Sprint-based)",
    "Kanban (Pull-based Flow)",
    "AI-SciOps (Autonomous Optimization)",
]

COLORS = {
    "Baseline (No Optimization)": "#95a5a6",
    "TOC (Theory of Constraints)": "#3498db",
    "PDCA (Plan-Do-Check-Act)": "#2ecc71",
    "Agile (Sprint-based)": "#e67e22",
    "Kanban (Pull-based Flow)": "#9b59b6",
    "AI-SciOps (Autonomous Optimization)": "#e74c3c",
}

SHORT_NAMES = {
    "Baseline (No Optimization)": "Baseline",
    "TOC (Theory of Constraints)": "TOC",
    "PDCA (Plan-Do-Check-Act)": "PDCA",
    "Agile (Sprint-based)": "Agile",
    "Kanban (Pull-based Flow)": "Kanban",
    "AI-SciOps (Autonomous Optimization)": "AI-SciOps",
}


# ─── Run Experiments ───

def run_individual_comparison(
    time_steps: int = 100,
    total_resources: float = 6.0,
    input_rate: float = 2.0,
    seed: int = 42,
    output_dir: str = "results",
) -> dict[str, SimulationResult]:
    """Run all 6 individual methodology simulations."""

    optimizers = [
        BaselineOptimizer(),
        TOCOnlyOptimizer(),
        PDCAOnlyOptimizer(cycle_length=10),
        AgileOnlyOptimizer(sprint_length=8),
        KanbanOnlyOptimizer(wip_limit=3.0),
        AISciOpsOptimizer(),
    ]

    results = {}
    for opt in optimizers:
        print(f"\n{'='*60}")
        print(f"Running: {opt.name}")
        print(f"{'='*60}")

        sim = Simulator(
            optimizer=opt,
            total_resources=total_resources,
            input_rate=input_rate,
            seed=seed,
        )
        result = sim.run(time_steps=time_steps)
        results[opt.name] = result

        print(f"  Total output: {result.total_output:.2f}")
        print(f"  Actions: {len(result.optimization_actions)}")
        for name, state in result.final_state.items():
            print(f"    {name}: tp={state['throughput']:.2f}, "
                  f"done={state['completed_units']:.1f}, "
                  f"ai={state['ai_assistance_level']:.2f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    for name, result in results.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
        filepath = os.path.join(output_dir, f"v6_{safe_name}.json")
        with open(filepath, "w") as f:
            json.dump(
                {
                    "optimizer_name": result.optimizer_name,
                    "total_time_steps": result.total_time_steps,
                    "total_output": result.total_output,
                    "metrics": [asdict(m) for m in result.metrics],
                    "optimization_actions": result.optimization_actions,
                    "final_state": result.final_state,
                },
                f,
                indent=2,
            )

    return results


def print_comparison(results: dict[str, SimulationResult]):
    """Print formatted comparison table."""
    baseline_name = "Baseline (No Optimization)"
    baseline_output = results[baseline_name].total_output

    print(f"\n{'='*80}")
    print("INDIVIDUAL METHODOLOGY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Methodology':<42} {'Output':>8} {'vs Base':>10} {'Avg TP':>8} {'Rework':>8} {'Fail':>8}")
    print(f"{'-'*80}")

    sorted_results = sorted(results.items(), key=lambda x: x[1].total_output, reverse=True)
    for name, result in sorted_results:
        improvement = (result.total_output - baseline_output) / baseline_output * 100
        avg_tp = np.mean([m.system_throughput for m in result.metrics])
        total_rework = result.metrics[-1].total_rework
        total_fail = result.metrics[-1].total_failures
        marker = " ***" if result.total_output == max(r.total_output for r in results.values()) else ""
        print(f"{name:<42} {result.total_output:>8.2f} {improvement:>+9.1f}% "
              f"{avg_tp:>8.3f} {total_rework:>8.1f} {total_fail:>8.1f}{marker}")

    # Print detailed per-process comparison
    processes = ["Survey", "Hypothesis", "Experiment", "Analysis", "Writing", "Review"]
    print(f"\n{'='*80}")
    print("FINAL STATE: PER-PROCESS COMPARISON")
    print(f"{'='*80}")

    for proc in processes:
        print(f"\n  {proc}:")
        print(f"  {'Methodology':<40} {'TP':>6} {'Done':>8} {'AI':>5} {'Rework':>8} {'Backlog':>8}")
        for name, result in sorted_results:
            s = result.final_state[proc]
            short = name[:38]
            print(f"  {short:<40} {s['throughput']:>6.2f} {s['completed_units']:>8.1f} "
                  f"{s['ai_assistance_level']:>5.2f} {s['rework_units']:>8.1f} "
                  f"{s['human_review_backlog']:>8.2f}")

    # Key insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")

    # Find which methodology handled which bottleneck best
    for name, result in sorted_results:
        bn_counts = {}
        for m in result.metrics:
            bn = m.bottleneck_process
            bn_counts[bn] = bn_counts.get(bn, 0) + 1
        main_bn = max(bn_counts, key=bn_counts.get)
        print(f"  {SHORT_NAMES.get(name, name):<15}: primary bottleneck = {main_bn} "
              f"({bn_counts[main_bn]}% of time)")


# ─── Visualization ───

def plot_cumulative_output(results: dict[str, SimulationResult]):
    """Cumulative output comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for name in STRATEGY_ORDER:
        if name not in results:
            continue
        result = results[name]
        steps = [m.time_step for m in result.metrics]
        cumulative = [m.cumulative_output for m in result.metrics]
        lw = 3.0 if name == "AI-SciOps (Autonomous Optimization)" else 2.0
        ls = "-" if name in ("AI-SciOps (Autonomous Optimization)", "Baseline (No Optimization)") else "--"
        ax.plot(steps, cumulative, label=SHORT_NAMES.get(name, name),
                color=COLORS.get(name, "gray"), linewidth=lw, linestyle=ls)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Cumulative Research Output", fontsize=12)
    ax.set_title("Cumulative Output: Individual Methodologies", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_throughput_timeseries(results: dict[str, SimulationResult]):
    """System throughput over time (smoothed)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    window = 5

    for name in STRATEGY_ORDER:
        if name not in results:
            continue
        result = results[name]
        steps = [m.time_step for m in result.metrics]
        tp = [m.system_throughput for m in result.metrics]

        if len(tp) >= window:
            smoothed = np.convolve(tp, np.ones(window) / window, mode="valid")
            lw = 3.0 if name == "AI-SciOps (Autonomous Optimization)" else 1.8
            ax.plot(steps[window - 1:], smoothed, label=SHORT_NAMES.get(name, name),
                    color=COLORS.get(name, "gray"), linewidth=lw)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("System Throughput (5-step moving avg)", fontsize=12)
    ax.set_title("System Throughput Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_ranking_bars(results: dict[str, SimulationResult]):
    """Horizontal bar chart ranking all methodologies."""
    baseline_output = results["Baseline (No Optimization)"].total_output

    # Sort by output
    sorted_items = sorted(results.items(), key=lambda x: x[1].total_output)
    names = [SHORT_NAMES.get(n, n) for n, _ in sorted_items]
    outputs = [r.total_output for _, r in sorted_items]
    colors = [COLORS.get(n, "gray") for n, _ in sorted_items]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), outputs, color=colors, height=0.6, edgecolor="white")

    for bar, val, (name, _) in zip(bars, outputs, sorted_items):
        pct = (val - baseline_output) / baseline_output * 100
        label = f"{val:.1f} ({pct:+.1f}%)" if name != "Baseline (No Optimization)" else f"{val:.1f} (base)"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=11, fontweight="bold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("Total Research Output (100 steps)", fontsize=12)
    ax.set_title("Methodology Ranking by Total Output", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_summary_metrics(results: dict[str, SimulationResult]):
    """4-panel summary of key metrics."""
    ordered = [n for n in STRATEGY_ORDER if n in results]
    short_labels = [SHORT_NAMES.get(n, n) for n in ordered]
    bar_colors = [COLORS.get(n, "gray") for n in ordered]

    metrics_data = {
        "Total Output": [results[n].total_output for n in ordered],
        "Avg System Throughput": [
            np.mean([m.system_throughput for m in results[n].metrics]) for n in ordered
        ],
        "Total Rework": [results[n].metrics[-1].total_rework for n in ordered],
        "Total Failures": [results[n].metrics[-1].total_failures for n in ordered],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        bars = axes[idx].bar(range(len(ordered)), values, color=bar_colors, edgecolor="white")
        axes[idx].set_title(metric_name, fontsize=12, fontweight="bold")
        axes[idx].set_xticks(range(len(ordered)))
        axes[idx].set_xticklabels(short_labels, fontsize=10)
        axes[idx].grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                          f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Key Metrics: Individual Methodology Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_bottleneck_distribution(results: dict[str, SimulationResult]):
    """Bottleneck frequency per methodology."""
    process_names = ["Survey", "Hypothesis", "Experiment", "Analysis", "Writing", "Review"]
    process_colors = {
        "Survey": "#1abc9c", "Hypothesis": "#3498db", "Experiment": "#e74c3c",
        "Analysis": "#f39c12", "Writing": "#9b59b6", "Review": "#e67e22",
    }

    ordered = [n for n in STRATEGY_ORDER if n in results]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, name in enumerate(ordered):
        result = results[name]
        counts = {p: 0 for p in process_names}
        for m in result.metrics:
            bn = m.bottleneck_process
            if bn in counts:
                counts[bn] += 1

        values = [counts[p] for p in process_names]
        colors = [process_colors[p] for p in process_names]
        axes[idx].bar(process_names, values, color=colors)
        axes[idx].set_title(SHORT_NAMES.get(name, name), fontsize=11,
                           fontweight="bold", color=COLORS.get(name, "black"))
        axes[idx].set_ylabel("Times as Bottleneck" if idx % 3 == 0 else "")
        axes[idx].tick_params(axis="x", rotation=45, labelsize=8)
        axes[idx].set_ylim(0, 100)

    fig.suptitle("Bottleneck Distribution per Methodology", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_wip_heatmaps(results: dict[str, SimulationResult]):
    """WIP accumulation heatmap per methodology."""
    process_names = ["Survey", "Hypothesis", "Experiment", "Analysis", "Writing", "Review"]

    ordered = [n for n in STRATEGY_ORDER if n in results]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()

    vmax = 0
    for name in ordered:
        result = results[name]
        for m in result.metrics:
            for p in process_names:
                vmax = max(vmax, m.process_wip.get(p, 0))

    for idx, name in enumerate(ordered):
        result = results[name]
        wip_data = np.zeros((len(process_names), len(result.metrics)))
        for t, m in enumerate(result.metrics):
            for p_idx, pname in enumerate(process_names):
                wip_data[p_idx, t] = m.process_wip.get(pname, 0)

        im = axes[idx].imshow(wip_data, aspect="auto", cmap="YlOrRd",
                              vmin=0, vmax=min(vmax, 30))
        axes[idx].set_title(SHORT_NAMES.get(name, name), fontsize=11,
                           fontweight="bold", color=COLORS.get(name, "black"))
        axes[idx].set_yticks(range(len(process_names)))
        axes[idx].set_yticklabels(process_names, fontsize=8)
        axes[idx].set_xlabel("Time Step", fontsize=9)

    fig.colorbar(im, ax=axes.tolist(), label="Work In Progress", shrink=0.6)
    fig.suptitle("WIP Accumulation Heatmap per Methodology", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_radar_comparison(results: dict[str, SimulationResult]):
    """Radar chart comparing methodologies across multiple dimensions."""
    baseline_output = results["Baseline (No Optimization)"].total_output

    categories = [
        "Total Output",
        "Avg Throughput",
        "Low Rework",
        "Low Failures",
        "Bottleneck\nResolution",
        "Flow Efficiency",
    ]
    N = len(categories)

    # Calculate normalized metrics (0-1 scale, higher is better)
    ordered = [n for n in STRATEGY_ORDER if n in results]
    all_values = {}

    for name in ordered:
        r = results[name]
        output = r.total_output
        avg_tp = np.mean([m.system_throughput for m in r.metrics])
        rework = r.metrics[-1].total_rework
        failures = r.metrics[-1].total_failures

        # Bottleneck resolution: how many different processes were bottleneck
        bn_counts = {}
        for m in r.metrics:
            bn_counts[m.bottleneck_process] = bn_counts.get(m.bottleneck_process, 0) + 1
        bn_concentration = max(bn_counts.values()) / len(r.metrics)

        # Flow efficiency: std dev of throughput (lower is smoother)
        tp_std = np.std([m.system_throughput for m in r.metrics])

        all_values[name] = [output, avg_tp, rework, failures, bn_concentration, tp_std]

    # Normalize each dimension
    for dim in range(N):
        values = [all_values[n][dim] for n in ordered]
        vmin, vmax = min(values), max(values)
        rng = vmax - vmin if vmax != vmin else 1.0
        for name in ordered:
            raw = all_values[name][dim]
            if dim in (2, 3, 4, 5):  # Lower is better for rework, failures, concentration, std
                all_values[name][dim] = 1.0 - (raw - vmin) / rng
            else:
                all_values[name][dim] = (raw - vmin) / rng

    # Plot
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for name in ordered:
        values = all_values[name] + [all_values[name][0]]
        ax.plot(angles, values, 'o-', label=SHORT_NAMES.get(name, name),
                color=COLORS.get(name, "gray"), linewidth=2, markersize=4)
        ax.fill(angles, values, alpha=0.08, color=COLORS.get(name, "gray"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Multi-Dimensional Methodology Comparison", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    return fig


# ─── Main ───

def main():
    src_dir = os.path.dirname(__file__)
    output_dir = os.path.join(src_dir, "..", "results")
    figure_dir = os.path.join(output_dir, "figures_v6")
    os.makedirs(figure_dir, exist_ok=True)

    # Run experiments
    results = run_individual_comparison(
        time_steps=100,
        total_resources=6.0,
        input_rate=2.0,
        seed=42,
        output_dir=output_dir,
    )

    # Print comparison
    print_comparison(results)

    # Generate figures
    figures = {
        "v6_01_cumulative_output": plot_cumulative_output,
        "v6_02_throughput_timeseries": plot_throughput_timeseries,
        "v6_03_ranking": plot_ranking_bars,
        "v6_04_summary_metrics": plot_summary_metrics,
        "v6_05_bottleneck_distribution": plot_bottleneck_distribution,
        "v6_06_wip_heatmaps": plot_wip_heatmaps,
        "v6_07_radar_comparison": plot_radar_comparison,
    }

    for name, fn in figures.items():
        print(f"\nGenerating {name}...")
        fig = fn(results)
        path = os.path.join(figure_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    print(f"\n{'='*60}")
    print(f"All results saved to {output_dir}")
    print(f"All figures saved to {figure_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
