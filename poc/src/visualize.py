"""
Visualization of Simulation Results
=====================================
Generates plots comparing the three optimization strategies.
Uses matplotlib for publication-quality figures.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")


def load_results() -> dict:
    """Load simulation results from JSON files."""
    results = {}
    for filename in os.listdir(RESULT_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(RESULT_DIR, filename)
            with open(filepath) as f:
                data = json.load(f)
                results[data["optimizer_name"]] = data
    return results


def get_color_scheme():
    """Color scheme for the three strategies."""
    return {
        "Baseline (No Optimization)": "#95a5a6",
        "TOC + PDCA": "#3498db",
        "AI-SciOps (Autonomous Optimization)": "#e74c3c",
    }


def plot_cumulative_output(results: dict):
    """Plot cumulative research output over time for all strategies."""
    colors = get_color_scheme()
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data in results.items():
        steps = [m["time_step"] for m in data["metrics"]]
        cumulative = [m["cumulative_output"] for m in data["metrics"]]
        ax.plot(steps, cumulative, label=name, color=colors.get(name, "gray"), linewidth=2)

    # Add stage annotations for AI-SciOps
    stage_boundaries = [
        (0, "Stage 1:\nHuman Feedback"),
        (20, "Stage 2:\nAutonomous"),
        (50, "Stage 3:\nRestructuring"),
        (80, "Stage 4:\nMeta-Optimization"),
    ]
    ymax = ax.get_ylim()[1]
    for x, label in stage_boundaries:
        ax.axvline(x=x, color="#e74c3c", linestyle="--", alpha=0.3)
        ax.text(x + 1, ymax * 0.95, label, fontsize=7, color="#e74c3c", alpha=0.7, va="top")

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Cumulative Research Output", fontsize=12)
    ax.set_title("Cumulative Research Output: Comparison of Optimization Strategies", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_system_throughput(results: dict):
    """Plot system throughput (per time step) with moving average."""
    colors = get_color_scheme()
    fig, ax = plt.subplots(figsize=(10, 6))
    window = 5

    for name, data in results.items():
        steps = [m["time_step"] for m in data["metrics"]]
        throughput = [m["system_throughput"] for m in data["metrics"]]

        # Moving average for smoothing
        if len(throughput) >= window:
            smoothed = np.convolve(throughput, np.ones(window) / window, mode="valid")
            ax.plot(
                steps[window - 1 :],
                smoothed,
                label=name,
                color=colors.get(name, "gray"),
                linewidth=2,
            )
        ax.plot(steps, throughput, color=colors.get(name, "gray"), alpha=0.15, linewidth=0.5)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("System Throughput (per step)", fontsize=12)
    ax.set_title("System Throughput Over Time (5-step moving average)", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_bottleneck_analysis(results: dict):
    """Show bottleneck distribution over time for each strategy."""
    process_names = ["Survey", "Hypothesis", "Experiment", "Analysis", "Writing", "Review"]
    process_colors = {
        "Survey": "#1abc9c",
        "Hypothesis": "#3498db",
        "Experiment": "#e74c3c",
        "Analysis": "#f39c12",
        "Writing": "#9b59b6",
        "Review": "#e67e22",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    strategies = list(results.keys())

    for idx, name in enumerate(strategies):
        data = results[name]
        bottleneck_counts = {p: 0 for p in process_names}
        for m in data["metrics"]:
            bn = m["bottleneck_process"]
            if bn in bottleneck_counts:
                bottleneck_counts[bn] += 1

        values = [bottleneck_counts[p] for p in process_names]
        bar_colors = [process_colors[p] for p in process_names]
        axes[idx].bar(process_names, values, color=bar_colors)
        axes[idx].set_title(name, fontsize=10)
        axes[idx].set_ylabel("Times as Bottleneck" if idx == 0 else "")
        axes[idx].tick_params(axis="x", rotation=45)

    fig.suptitle("Bottleneck Distribution Across Strategies", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_process_throughputs_heatmap(results: dict):
    """Heatmap of process throughputs over time for each strategy."""
    process_names = ["Survey", "Hypothesis", "Experiment", "Analysis", "Writing", "Review"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    strategies = list(results.keys())

    for idx, name in enumerate(strategies):
        data = results[name]
        matrix = []
        for pname in process_names:
            row = [m["process_throughputs"].get(pname, 0) for m in data["metrics"]]
            matrix.append(row)

        im = axes[idx].imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        axes[idx].set_yticks(range(len(process_names)))
        axes[idx].set_yticklabels(process_names)
        axes[idx].set_xlabel("Time Step")
        axes[idx].set_title(name, fontsize=11)
        plt.colorbar(im, ax=axes[idx], label="Throughput")

    fig.suptitle("Process Throughput Heatmap Over Time", fontsize=14)
    plt.tight_layout()
    return fig


def plot_wip_accumulation(results: dict):
    """Show Work-In-Progress accumulation patterns."""
    colors = get_color_scheme()
    process_names = ["Survey", "Hypothesis", "Experiment", "Analysis", "Writing", "Review"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for p_idx, pname in enumerate(process_names):
        ax = axes[p_idx]
        for name, data in results.items():
            steps = [m["time_step"] for m in data["metrics"]]
            wip = [m["process_wip"].get(pname, 0) for m in data["metrics"]]
            ax.plot(steps, wip, label=name, color=colors.get(name, "gray"), linewidth=1.5)

        ax.set_title(pname, fontsize=11)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("WIP")
        ax.grid(True, alpha=0.3)
        if p_idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Work-In-Progress Accumulation by Process", fontsize=14)
    plt.tight_layout()
    return fig


def plot_summary_comparison(results: dict):
    """Bar chart summary comparing final metrics."""
    colors = get_color_scheme()
    strategies = list(results.keys())

    metrics = {
        "Total Output": [r["total_output"] for r in results.values()],
        "Final Bottleneck\nThroughput": [
            r["metrics"][-1]["bottleneck_throughput"] for r in results.values()
        ],
        "Total Rework": [r["metrics"][-1]["total_rework"] for r in results.values()],
        "Total Failures": [
            r["metrics"][-1]["total_failures"] for r in results.values()
        ],
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    bar_colors = [colors.get(s, "gray") for s in strategies]
    short_labels = ["Baseline", "TOC+PDCA", "AI-SciOps"]

    for idx, (metric_name, values) in enumerate(metrics.items()):
        axes[idx].bar(short_labels, values, color=bar_colors)
        axes[idx].set_title(metric_name, fontsize=11)
        axes[idx].tick_params(axis="x", rotation=15)

    fig.suptitle("Summary Comparison of Optimization Strategies", fontsize=14)
    plt.tight_layout()
    return fig


def generate_all_figures():
    """Generate all visualization figures."""
    results = load_results()

    if not results:
        print("No results found. Run simulator.py first.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    figures = {
        "01_cumulative_output": plot_cumulative_output,
        "02_system_throughput": plot_system_throughput,
        "03_bottleneck_analysis": plot_bottleneck_analysis,
        "04_throughput_heatmap": plot_process_throughputs_heatmap,
        "05_wip_accumulation": plot_wip_accumulation,
        "06_summary_comparison": plot_summary_comparison,
    }

    for name, plot_fn in figures.items():
        print(f"Generating {name}...")
        fig = plot_fn(results)
        filepath = os.path.join(OUTPUT_DIR, f"{name}.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filepath}")

    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_figures()
