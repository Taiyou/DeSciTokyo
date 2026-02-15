"""
Extended Visualization for 6-Strategy Comparison
==================================================
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_DIR = os.path.join(RESULT_DIR, "figures_v2")


def load_v2_results() -> dict:
    """Load v2 simulation results."""
    results = {}
    for filename in os.listdir(RESULT_DIR):
        if filename.startswith("v2_") and filename.endswith(".json"):
            filepath = os.path.join(RESULT_DIR, filename)
            with open(filepath) as f:
                data = json.load(f)
                results[data["optimizer_name"]] = data
    return results


# Display order (best last for layering)
STRATEGY_ORDER = [
    "Baseline (No Optimization)",
    "TOC + PDCA",
    "AI-SciOps (Autonomous Optimization)",
    "Kanban-SciOps (Pull-based Flow)",
    "Adaptive-SciOps (Metric-driven)",
    "Holistic-SciOps (Integrated)",
]

COLORS = {
    "Baseline (No Optimization)": "#bdc3c7",
    "TOC + PDCA": "#3498db",
    "AI-SciOps (Autonomous Optimization)": "#e74c3c",
    "Kanban-SciOps (Pull-based Flow)": "#2ecc71",
    "Adaptive-SciOps (Metric-driven)": "#f39c12",
    "Holistic-SciOps (Integrated)": "#9b59b6",
}

SHORT_NAMES = {
    "Baseline (No Optimization)": "Baseline",
    "TOC + PDCA": "TOC+PDCA",
    "AI-SciOps (Autonomous Optimization)": "AI-SciOps\n(Original)",
    "Kanban-SciOps (Pull-based Flow)": "Kanban\n-SciOps",
    "Adaptive-SciOps (Metric-driven)": "Adaptive\n-SciOps",
    "Holistic-SciOps (Integrated)": "Holistic\n-SciOps",
}


def plot_cumulative_comparison(results: dict):
    """Cumulative output for all 6 strategies."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for name in STRATEGY_ORDER:
        if name not in results:
            continue
        data = results[name]
        steps = [m["time_step"] for m in data["metrics"]]
        cumulative = [m["cumulative_output"] for m in data["metrics"]]
        lw = 2.5 if name in ("Holistic-SciOps (Integrated)", "AI-SciOps (Autonomous Optimization)") else 1.5
        ls = "-" if "SciOps" in name or "Baseline" in name else "--"
        ax.plot(steps, cumulative, label=SHORT_NAMES.get(name, name),
                color=COLORS.get(name, "gray"), linewidth=lw, linestyle=ls)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Cumulative Research Output", fontsize=12)
    ax.set_title("Cumulative Output: Original vs. Advanced Strategies", fontsize=14)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_throughput_comparison(results: dict):
    """System throughput (moving average) for all strategies."""
    fig, ax = plt.subplots(figsize=(12, 7))
    window = 5

    for name in STRATEGY_ORDER:
        if name not in results:
            continue
        data = results[name]
        steps = [m["time_step"] for m in data["metrics"]]
        tp = [m["system_throughput"] for m in data["metrics"]]

        if len(tp) >= window:
            smoothed = np.convolve(tp, np.ones(window) / window, mode="valid")
            lw = 2.5 if name in ("Holistic-SciOps (Integrated)", "AI-SciOps (Autonomous Optimization)") else 1.5
            ax.plot(steps[window - 1:], smoothed, label=SHORT_NAMES.get(name, name),
                    color=COLORS.get(name, "gray"), linewidth=lw)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("System Throughput (per step)", fontsize=12)
    ax.set_title("System Throughput: 5-step Moving Average", fontsize=14)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_summary_bars(results: dict):
    """Summary bar chart with all 6 strategies."""
    ordered = [n for n in STRATEGY_ORDER if n in results]
    short_labels = [SHORT_NAMES.get(n, n) for n in ordered]
    bar_colors = [COLORS.get(n, "gray") for n in ordered]

    metrics_data = {
        "Total Output": [results[n]["total_output"] for n in ordered],
        "Final Bottleneck\nThroughput": [
            results[n]["metrics"][-1]["bottleneck_throughput"] for n in ordered
        ],
        "Total Rework": [results[n]["metrics"][-1]["total_rework"] for n in ordered],
        "Total Failures": [results[n]["metrics"][-1]["total_failures"] for n in ordered],
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))

    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        bars = axes[idx].bar(range(len(ordered)), values, color=bar_colors)
        axes[idx].set_title(metric_name, fontsize=11)
        axes[idx].set_xticks(range(len(ordered)))
        axes[idx].set_xticklabels(short_labels, fontsize=7, rotation=0)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                          f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Summary: All 6 Optimization Strategies", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_bottleneck_comparison(results: dict):
    """Bottleneck distribution for all 6 strategies."""
    process_names = ["Survey", "Hypothesis", "Experiment", "Analysis", "Writing", "Review"]
    process_colors = {
        "Survey": "#1abc9c", "Hypothesis": "#3498db", "Experiment": "#e74c3c",
        "Analysis": "#f39c12", "Writing": "#9b59b6", "Review": "#e67e22",
    }

    ordered = [n for n in STRATEGY_ORDER if n in results]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, name in enumerate(ordered):
        data = results[name]
        counts = {p: 0 for p in process_names}
        for m in data["metrics"]:
            bn = m["bottleneck_process"]
            if bn in counts:
                counts[bn] += 1

        values = [counts[p] for p in process_names]
        colors = [process_colors[p] for p in process_names]
        axes[idx].bar(process_names, values, color=colors)
        axes[idx].set_title(SHORT_NAMES.get(name, name), fontsize=10, color=COLORS.get(name, "black"))
        axes[idx].set_ylabel("Times as Bottleneck" if idx % 3 == 0 else "")
        axes[idx].tick_params(axis="x", rotation=45, labelsize=8)

    fig.suptitle("Bottleneck Distribution: All Strategies", fontsize=14)
    plt.tight_layout()
    return fig


def plot_wip_comparison(results: dict):
    """WIP patterns for key processes across strategies."""
    process_names = ["Survey", "Hypothesis", "Experiment", "Analysis", "Writing", "Review"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for p_idx, pname in enumerate(process_names):
        ax = axes[p_idx]
        for name in STRATEGY_ORDER:
            if name not in results:
                continue
            data = results[name]
            steps = [m["time_step"] for m in data["metrics"]]
            wip = [m["process_wip"].get(pname, 0) for m in data["metrics"]]
            lw = 2.0 if name in ("Holistic-SciOps (Integrated)", "AI-SciOps (Autonomous Optimization)") else 1.0
            ax.plot(steps, wip, color=COLORS.get(name, "gray"), linewidth=lw,
                    label=SHORT_NAMES.get(name, name) if p_idx == 0 else None)

        ax.set_title(pname, fontsize=11)
        ax.set_xlabel("Time Step", fontsize=9)
        ax.set_ylabel("WIP", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Add legend to first subplot
    axes[0].legend(fontsize=6, loc="upper left")
    fig.suptitle("Work-In-Progress by Process: All Strategies", fontsize=14)
    plt.tight_layout()
    return fig


def plot_improvement_waterfall(results: dict):
    """Waterfall chart showing incremental improvements."""
    ordered = [n for n in STRATEGY_ORDER if n in results]
    outputs = [results[n]["total_output"] for n in ordered]
    short = [SHORT_NAMES.get(n, n).replace("\n", " ") for n in ordered]

    fig, ax = plt.subplots(figsize=(12, 6))

    baseline = outputs[0]
    improvements = [0] + [outputs[i] - outputs[i - 1] for i in range(1, len(outputs))]
    cumulative = [sum(improvements[:i + 1]) for i in range(len(improvements))]

    bar_colors = [COLORS.get(n, "gray") for n in ordered]

    # Sort by total output for cleaner waterfall
    sorted_indices = sorted(range(len(outputs)), key=lambda i: outputs[i])
    sorted_names = [short[i] for i in sorted_indices]
    sorted_outputs = [outputs[i] for i in sorted_indices]
    sorted_colors = [bar_colors[i] for i in sorted_indices]

    bars = ax.barh(range(len(sorted_names)), sorted_outputs, color=sorted_colors, height=0.6)

    for bar, val in zip(bars, sorted_outputs):
        pct = (val - baseline) / baseline * 100
        label = f"{val:.1f} ({pct:+.1f}%)" if val != baseline else f"{val:.1f} (base)"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=10)

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel("Total Research Output", fontsize=12)
    ax.set_title("Strategy Ranking by Total Research Output", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def generate_all():
    results = load_v2_results()
    if not results:
        print("No v2 results found. Run run_comparison.py first.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    figures = {
        "v2_01_cumulative": plot_cumulative_comparison,
        "v2_02_throughput": plot_throughput_comparison,
        "v2_03_summary_bars": plot_summary_bars,
        "v2_04_bottleneck": plot_bottleneck_comparison,
        "v2_05_wip": plot_wip_comparison,
        "v2_06_ranking": plot_improvement_waterfall,
    }

    for name, fn in figures.items():
        print(f"Generating {name}...")
        fig = fn(results)
        path = os.path.join(OUTPUT_DIR, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    print(f"\nAll v2 figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all()
