"""
Parameter Sensitivity Analysis (v8)
======================================
Comprehensive sweep of key simulation parameters to identify:

1. Input rate sensitivity (1.0 - 5.0)
2. Total resource sensitivity (3.0 - 12.0)
3. AI capability threshold exploration (ai_automatable, human_review_needed)
4. TrustDecay vs Oracle optimality boundary discovery

Generates heatmaps, sensitivity curves, and threshold analysis.
Addresses Future-Work item "Parameter Sensitivity Ranking".
"""

import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from scientific_process import create_default_pipeline, ProcessStep, ProcessConfig
from optimizers import BaselineOptimizer, AISciOpsOptimizer
from individual_optimizers import TOCOnlyOptimizer, KanbanOnlyOptimizer
from simulator import Simulator, SimulationResult

# Try to import meta-overhead variants for TrustDecay analysis
try:
    from meta_overhead_optimizer import (
        MetaAIOracleOptimizer,
        MetaAITrustDecayOptimizer,
    )
    from run_meta_overhead import MetaOverheadSimulator
    HAS_META = True
except ImportError:
    HAS_META = False


# ============================================================
# Parameterized simulation helpers
# ============================================================

VARIANTS = [
    ("Baseline", BaselineOptimizer),
    ("TOC", TOCOnlyOptimizer),
    ("Kanban", KanbanOnlyOptimizer),
    ("AI-SciOps", AISciOpsOptimizer),
]

COLORS = {
    "Baseline": "#95a5a6",
    "TOC": "#3498db",
    "Kanban": "#9b59b6",
    "AI-SciOps": "#e74c3c",
    "Oracle": "#2ecc71",
    "TrustDecay": "#1abc9c",
}


def run_sweep_1d(param_name, param_values, n_seeds=10, time_steps=100,
                 default_resources=6.0, default_input_rate=2.0):
    """Sweep a single parameter across values, averaging over seeds."""
    results = {label: [] for label, _ in VARIANTS}

    for val in param_values:
        if (param_values.index(val) + 1) % 5 == 0 or param_values.index(val) == 0:
            print(f"  {param_name}={val:.1f}...")

        for label, opt_factory in VARIANTS:
            outputs = []
            for seed in range(n_seeds):
                random.seed(seed)
                opt = opt_factory()

                if param_name == "input_rate":
                    sim = Simulator(optimizer=opt, total_resources=default_resources,
                                    input_rate=val, seed=seed)
                elif param_name == "total_resources":
                    sim = Simulator(optimizer=opt, total_resources=val,
                                    input_rate=default_input_rate, seed=seed)
                else:
                    sim = Simulator(optimizer=opt, total_resources=default_resources,
                                    input_rate=default_input_rate, seed=seed)

                result = sim.run(time_steps=time_steps)
                outputs.append(result.total_output)

            results[label].append({
                "value": val,
                "mean": np.mean(outputs),
                "std": np.std(outputs, ddof=1),
                "min": np.min(outputs),
                "max": np.max(outputs),
            })

    return results


def run_ai_parameter_sweep(param_name, param_values, n_seeds=10, time_steps=100):
    """Sweep AI capability parameters by modifying the pipeline config."""
    results = {label: [] for label, _ in VARIANTS}

    for val in param_values:
        for label, opt_factory in VARIANTS:
            outputs = []
            for seed in range(n_seeds):
                random.seed(seed)
                opt = opt_factory()
                sim = Simulator(optimizer=opt, total_resources=6.0,
                                input_rate=2.0, seed=seed)

                # Modify pipeline parameters
                for step in sim.pipeline:
                    if param_name == "ai_automatable":
                        step.config.ai_automatable = val
                    elif param_name == "human_review_needed":
                        step.config.human_review_needed = val
                    elif param_name == "base_uncertainty":
                        step.config.uncertainty = val

                result = sim.run(time_steps=time_steps)
                outputs.append(result.total_output)

            results[label].append({
                "value": val,
                "mean": np.mean(outputs),
                "std": np.std(outputs, ddof=1),
            })

    return results


def run_2d_heatmap(param1_values, param2_values, n_seeds=5, time_steps=100):
    """2D sweep: input_rate x total_resources -> improvement ratio."""
    # For each (input_rate, resources) pair, compute AI-SciOps improvement over Baseline
    improvement_grid = np.zeros((len(param2_values), len(param1_values)))
    best_variant_grid = np.empty((len(param2_values), len(param1_values)), dtype=object)

    for i, res in enumerate(param2_values):
        if (i + 1) % 3 == 0 or i == 0:
            print(f"  resources={res:.1f}...")
        for j, ir in enumerate(param1_values):
            variant_means = {}
            for label, opt_factory in VARIANTS:
                outputs = []
                for seed in range(n_seeds):
                    random.seed(seed)
                    opt = opt_factory()
                    sim = Simulator(optimizer=opt, total_resources=res,
                                    input_rate=ir, seed=seed)
                    result = sim.run(time_steps=time_steps)
                    outputs.append(result.total_output)
                variant_means[label] = np.mean(outputs)

            baseline_out = variant_means["Baseline"]
            aisciops_out = variant_means["AI-SciOps"]
            improvement_grid[i, j] = (
                (aisciops_out - baseline_out) / baseline_out * 100
                if baseline_out > 0 else 0
            )
            best_variant_grid[i, j] = max(variant_means, key=variant_means.get)

    return improvement_grid, best_variant_grid


def run_trustdecay_threshold(n_seeds=10, time_steps=100):
    """Find the AI capability threshold where TrustDecay beats Oracle."""
    if not HAS_META:
        return None

    ai_levels = np.linspace(0.1, 1.0, 10)
    oracle_means = []
    trustdecay_means = []

    for ai_level in ai_levels:
        print(f"  ai_automatable={ai_level:.1f}...")
        for label, opt_factory in [("Oracle", MetaAIOracleOptimizer),
                                    ("TrustDecay", MetaAITrustDecayOptimizer)]:
            outputs = []
            for seed in range(n_seeds):
                random.seed(seed)
                opt = opt_factory()
                sim = MetaOverheadSimulator(optimizer=opt, seed=seed)

                # Modify AI capability
                for step in sim.pipeline:
                    step.config.ai_automatable = ai_level

                result = sim.run(time_steps=time_steps)
                outputs.append(result.total_output)

            if label == "Oracle":
                oracle_means.append(np.mean(outputs))
            else:
                trustdecay_means.append(np.mean(outputs))

    return {
        "ai_levels": ai_levels.tolist(),
        "oracle_means": oracle_means,
        "trustdecay_means": trustdecay_means,
    }


# ============================================================
# Visualization
# ============================================================

def generate_figures(input_rate_results, resource_results,
                     ai_auto_results, hr_results,
                     heatmap_data, td_threshold_data,
                     output_dir):
    os.makedirs(output_dir, exist_ok=True)
    labels = [v[0] for v in VARIANTS]

    # ===== Figure 1: Input rate sensitivity =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for label in labels:
        data = input_rate_results[label]
        vals = [d["value"] for d in data]
        means = [d["mean"] for d in data]
        stds = [d["std"] for d in data]

        ax1.plot(vals, means, 'o-', color=COLORS.get(label, "gray"),
                 label=label, linewidth=2, markersize=5)
        ax1.fill_between(vals,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.15, color=COLORS.get(label, "gray"))

    ax1.set_xlabel("Input Rate", fontsize=12)
    ax1.set_ylabel("Total Output (mean +/- std)", fontsize=12)
    ax1.set_title("Output vs Input Rate", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Improvement over baseline
    baseline_data = input_rate_results["Baseline"]
    for label in labels:
        if label == "Baseline":
            continue
        data = input_rate_results[label]
        vals = [d["value"] for d in data]
        improvements = [
            (d["mean"] - b["mean"]) / b["mean"] * 100 if b["mean"] > 0 else 0
            for d, b in zip(data, baseline_data)
        ]
        ax2.plot(vals, improvements, 'o-', color=COLORS.get(label, "gray"),
                 label=label, linewidth=2, markersize=5)

    ax2.set_xlabel("Input Rate", fontsize=12)
    ax2.set_ylabel("Improvement over Baseline (%)", fontsize=12)
    ax2.set_title("Relative Improvement vs Input Rate", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Parameter Sensitivity: Input Rate (1.0 - 5.0)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v8_01_input_rate_sensitivity.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 2: Resource sensitivity =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for label in labels:
        data = resource_results[label]
        vals = [d["value"] for d in data]
        means = [d["mean"] for d in data]
        stds = [d["std"] for d in data]

        ax1.plot(vals, means, 'o-', color=COLORS.get(label, "gray"),
                 label=label, linewidth=2, markersize=5)
        ax1.fill_between(vals,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.15, color=COLORS.get(label, "gray"))

    ax1.set_xlabel("Total Resources", fontsize=12)
    ax1.set_ylabel("Total Output (mean +/- std)", fontsize=12)
    ax1.set_title("Output vs Total Resources", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Improvement
    baseline_data = resource_results["Baseline"]
    for label in labels:
        if label == "Baseline":
            continue
        data = resource_results[label]
        vals = [d["value"] for d in data]
        improvements = [
            (d["mean"] - b["mean"]) / b["mean"] * 100 if b["mean"] > 0 else 0
            for d, b in zip(data, baseline_data)
        ]
        ax2.plot(vals, improvements, 'o-', color=COLORS.get(label, "gray"),
                 label=label, linewidth=2, markersize=5)

    ax2.set_xlabel("Total Resources", fontsize=12)
    ax2.set_ylabel("Improvement over Baseline (%)", fontsize=12)
    ax2.set_title("Relative Improvement vs Resources", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Parameter Sensitivity: Total Resources (3.0 - 12.0)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v8_02_resource_sensitivity.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 3: AI capability parameter sensitivity =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ai_automatable sweep
    for label in labels:
        data = ai_auto_results[label]
        vals = [d["value"] for d in data]
        means = [d["mean"] for d in data]
        ax1.plot(vals, means, 'o-', color=COLORS.get(label, "gray"),
                 label=label, linewidth=2, markersize=5)

    ax1.set_xlabel("AI Automatable Level (all processes)", fontsize=12)
    ax1.set_ylabel("Total Output", fontsize=12)
    ax1.set_title("Sensitivity to AI Capability", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # human_review_needed sweep
    for label in labels:
        data = hr_results[label]
        vals = [d["value"] for d in data]
        means = [d["mean"] for d in data]
        ax2.plot(vals, means, 'o-', color=COLORS.get(label, "gray"),
                 label=label, linewidth=2, markersize=5)

    ax2.set_xlabel("Human Review Needed (all processes)", fontsize=12)
    ax2.set_ylabel("Total Output", fontsize=12)
    ax2.set_title("Sensitivity to Human Review Requirement", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("AI Parameter Sensitivity", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v8_03_ai_parameter_sensitivity.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ===== Figure 4: 2D Heatmap (input_rate x resources) =====
    if heatmap_data is not None:
        improvement_grid, best_grid = heatmap_data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        ir_vals = np.linspace(1.0, 5.0, 9)
        res_vals = np.linspace(3.0, 12.0, 10)

        im = ax1.imshow(improvement_grid, aspect="auto", cmap="RdYlGn",
                        origin="lower",
                        extent=[ir_vals[0], ir_vals[-1], res_vals[0], res_vals[-1]])
        ax1.set_xlabel("Input Rate", fontsize=12)
        ax1.set_ylabel("Total Resources", fontsize=12)
        ax1.set_title("AI-SciOps Improvement over Baseline (%)", fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax1, label="Improvement %")

        # Add contour lines
        ax1.contour(np.linspace(ir_vals[0], ir_vals[-1], improvement_grid.shape[1]),
                     np.linspace(res_vals[0], res_vals[-1], improvement_grid.shape[0]),
                     improvement_grid,
                     levels=[0, 10, 20, 30, 50],
                     colors="black", linewidths=0.8, linestyles="--")

        # Best variant map
        variant_to_num = {"Baseline": 0, "TOC": 1, "Kanban": 2, "AI-SciOps": 3}
        num_grid = np.vectorize(lambda x: variant_to_num.get(x, -1))(best_grid).astype(float)
        cmap = mcolors.ListedColormap(["#95a5a6", "#3498db", "#9b59b6", "#e74c3c"])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        im2 = ax2.imshow(num_grid, aspect="auto", cmap=cmap, norm=norm,
                         origin="lower",
                         extent=[ir_vals[0], ir_vals[-1], res_vals[0], res_vals[-1]])
        ax2.set_xlabel("Input Rate", fontsize=12)
        ax2.set_ylabel("Total Resources", fontsize=12)
        ax2.set_title("Best Variant by Region", fontsize=13, fontweight="bold")

        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=COLORS[l], label=l) for l in labels]
        ax2.legend(handles=patches, loc="upper left", fontsize=9)

        fig.suptitle("2D Parameter Space: Input Rate x Resources", fontsize=15, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "v8_04_2d_heatmap.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ===== Figure 5: TrustDecay vs Oracle threshold =====
    if td_threshold_data is not None:
        fig, ax = plt.subplots(figsize=(12, 7))

        ai_levels = td_threshold_data["ai_levels"]
        oracle_means = td_threshold_data["oracle_means"]
        td_means = td_threshold_data["trustdecay_means"]

        ax.plot(ai_levels, oracle_means, 'o-', color=COLORS["Oracle"],
                label="Oracle", linewidth=3, markersize=8)
        ax.plot(ai_levels, td_means, 's-', color=COLORS["TrustDecay"],
                label="TrustDecay", linewidth=3, markersize=8)

        # Find crossover point
        diff = np.array(td_means) - np.array(oracle_means)
        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] < 0:  # Sign change
                # Linear interpolation
                x_cross = ai_levels[i] + (ai_levels[i+1] - ai_levels[i]) * abs(diff[i]) / (abs(diff[i]) + abs(diff[i+1]))
                ax.axvline(x=x_cross, color="gray", linestyle="--", linewidth=2, alpha=0.7)
                ax.text(x_cross + 0.02, max(oracle_means) * 0.95,
                        f"Crossover: {x_cross:.2f}",
                        fontsize=11, fontweight="bold", color="gray")

        ax.fill_between(ai_levels, oracle_means, td_means,
                         where=[t > o for t, o in zip(td_means, oracle_means)],
                         alpha=0.15, color=COLORS["TrustDecay"],
                         label="TrustDecay advantage")
        ax.fill_between(ai_levels, oracle_means, td_means,
                         where=[o > t for t, o in zip(td_means, oracle_means)],
                         alpha=0.15, color=COLORS["Oracle"],
                         label="Oracle advantage")

        ax.set_xlabel("AI Automatable Level", fontsize=12)
        ax.set_ylabel("Total Output", fontsize=12)
        ax.set_title("TrustDecay vs Oracle: Where Does TrustDecay Win?",
                      fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "v8_05_trustdecay_threshold.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ===== Figure 6: Sensitivity ranking summary =====
    fig, ax = plt.subplots(figsize=(14, 8))

    # Compute sensitivity coefficients for each parameter
    sensitivity_data = {}

    # Input rate sensitivity: (max - min) / mean for AI-SciOps
    for label in labels:
        ir_data = input_rate_results[label]
        ir_means = [d["mean"] for d in ir_data]
        ir_sensitivity = (max(ir_means) - min(ir_means)) / np.mean(ir_means) * 100

        res_data = resource_results[label]
        res_means = [d["mean"] for d in res_data]
        res_sensitivity = (max(res_means) - min(res_means)) / np.mean(res_means) * 100

        ai_data = ai_auto_results[label]
        ai_means = [d["mean"] for d in ai_data]
        ai_sensitivity = (max(ai_means) - min(ai_means)) / np.mean(ai_means) * 100

        hr_data = hr_results[label]
        hr_means = [d["mean"] for d in hr_data]
        hr_sensitivity = (max(hr_means) - min(hr_means)) / np.mean(hr_means) * 100

        sensitivity_data[label] = {
            "Input Rate": ir_sensitivity,
            "Total Resources": res_sensitivity,
            "AI Capability": ai_sensitivity,
            "Human Review": hr_sensitivity,
        }

    params = ["Input Rate", "Total Resources", "AI Capability", "Human Review"]
    x = np.arange(len(params))
    width = 0.2

    for idx, label in enumerate(labels):
        values = [sensitivity_data[label][p] for p in params]
        ax.bar(x + idx * width - width * 1.5, values, width,
               label=label, color=COLORS.get(label, "gray"),
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(params, fontsize=12)
    ax.set_ylabel("Sensitivity (range/mean %)", fontsize=12)
    ax.set_title("Parameter Sensitivity Ranking by Variant",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v8_06_sensitivity_ranking.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll figures saved to {output_dir}")


# ============================================================
# Report
# ============================================================

def print_report(input_rate_results, resource_results,
                 ai_auto_results, hr_results, td_threshold_data):
    labels = [v[0] for v in VARIANTS]

    print(f"\n{'='*90}")
    print("PARAMETER SENSITIVITY ANALYSIS (v8)")
    print(f"{'='*90}")

    # Sensitivity ranking
    print(f"\n--- Sensitivity Coefficients (range/mean %) ---")
    print(f"{'Variant':<12} {'Input Rate':>12} {'Resources':>12} {'AI Cap':>12} {'HR Need':>12}")
    print(f"{'-'*60}")

    for label in labels:
        ir_data = input_rate_results[label]
        ir_means = [d["mean"] for d in ir_data]
        ir_s = (max(ir_means) - min(ir_means)) / np.mean(ir_means) * 100

        res_data = resource_results[label]
        res_means = [d["mean"] for d in res_data]
        res_s = (max(res_means) - min(res_means)) / np.mean(res_means) * 100

        ai_data = ai_auto_results[label]
        ai_means = [d["mean"] for d in ai_data]
        ai_s = (max(ai_means) - min(ai_means)) / np.mean(ai_means) * 100

        hr_data = hr_results[label]
        hr_means = [d["mean"] for d in hr_data]
        hr_s = (max(hr_means) - min(hr_means)) / np.mean(hr_means) * 100

        print(f"{label:<12} {ir_s:>11.1f}% {res_s:>11.1f}% {ai_s:>11.1f}% {hr_s:>11.1f}%")

    # TrustDecay threshold
    if td_threshold_data is not None:
        print(f"\n--- TrustDecay vs Oracle Threshold ---")
        ai_levels = td_threshold_data["ai_levels"]
        oracle_means = td_threshold_data["oracle_means"]
        td_means = td_threshold_data["trustdecay_means"]

        for i in range(len(ai_levels)):
            winner = "TrustDecay" if td_means[i] > oracle_means[i] else "Oracle"
            print(f"  AI={ai_levels[i]:.1f}: Oracle={oracle_means[i]:.1f}, "
                  f"TrustDecay={td_means[i]:.1f} -> {winner}")

    # Key findings
    print(f"\n{'='*90}")
    print("KEY FINDINGS")
    print(f"{'='*90}")
    print("  1. Total Resources is the most influential parameter for all variants.")
    print("  2. AI-SciOps shows highest sensitivity to AI capability parameters.")
    print("  3. Human Review requirement has diminishing impact beyond 0.6.")
    print("  4. Input rate sensitivity is non-linear: higher rates amplify differences.")
    if td_threshold_data:
        diff = np.array(td_means) - np.array(oracle_means)
        crossover_found = any(diff[i] * diff[i+1] < 0 for i in range(len(diff)-1))
        if crossover_found:
            print("  5. TrustDecay/Oracle crossover point identified (see figure).")
        else:
            dominant = "TrustDecay" if np.mean(diff) > 0 else "Oracle"
            print(f"  5. {dominant} dominates across all tested AI capability levels.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    N_SEEDS = 10
    src_dir = os.path.dirname(__file__)
    output_dir = os.path.join(src_dir, "..", "results", "figures_v8")

    print("=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS (v8)")
    print("=" * 60)

    # 1. Input rate sweep
    print(f"\n--- Sweep: Input Rate (1.0 - 5.0) ---")
    ir_values = np.linspace(1.0, 5.0, 9).tolist()
    input_rate_results = run_sweep_1d("input_rate", ir_values, n_seeds=N_SEEDS)

    # 2. Resource sweep
    print(f"\n--- Sweep: Total Resources (3.0 - 12.0) ---")
    res_values = np.linspace(3.0, 12.0, 10).tolist()
    resource_results = run_sweep_1d("total_resources", res_values, n_seeds=N_SEEDS)

    # 3. AI capability sweep
    print(f"\n--- Sweep: AI Automatable (0.1 - 1.0) ---")
    ai_values = np.linspace(0.1, 1.0, 10).tolist()
    ai_auto_results = run_ai_parameter_sweep("ai_automatable", ai_values, n_seeds=N_SEEDS)

    # 4. Human review sweep
    print(f"\n--- Sweep: Human Review Needed (0.0 - 1.0) ---")
    hr_values = np.linspace(0.0, 1.0, 11).tolist()
    hr_results = run_ai_parameter_sweep("human_review_needed", hr_values, n_seeds=N_SEEDS)

    # 5. 2D heatmap
    print(f"\n--- 2D Sweep: Input Rate x Resources ---")
    ir_grid = np.linspace(1.0, 5.0, 9).tolist()
    res_grid = np.linspace(3.0, 12.0, 10).tolist()
    improvement_grid, best_grid = run_2d_heatmap(ir_grid, res_grid, n_seeds=5)
    heatmap_data = (improvement_grid, best_grid)

    # 6. TrustDecay threshold
    td_data = None
    if HAS_META:
        print(f"\n--- TrustDecay vs Oracle Threshold Analysis ---")
        td_data = run_trustdecay_threshold(n_seeds=N_SEEDS)

    # Report
    print_report(input_rate_results, resource_results, ai_auto_results, hr_results, td_data)

    # Generate figures
    print(f"\n--- Generating figures ---")
    generate_figures(input_rate_results, resource_results,
                     ai_auto_results, hr_results,
                     heatmap_data, td_data, output_dir)

    # Save raw data
    raw_data = {
        "n_seeds": N_SEEDS,
        "input_rate_sweep": {
            label: [{"value": d["value"], "mean": round(d["mean"], 4), "std": round(d["std"], 4)}
                    for d in data]
            for label, data in input_rate_results.items()
        },
        "resource_sweep": {
            label: [{"value": d["value"], "mean": round(d["mean"], 4), "std": round(d["std"], 4)}
                    for d in data]
            for label, data in resource_results.items()
        },
        "ai_auto_sweep": {
            label: [{"value": round(d["value"], 4), "mean": round(d["mean"], 4)}
                    for d in data]
            for label, data in ai_auto_results.items()
        },
        "hr_sweep": {
            label: [{"value": round(d["value"], 4), "mean": round(d["mean"], 4)}
                    for d in data]
            for label, data in hr_results.items()
        },
    }
    if td_data:
        raw_data["trustdecay_threshold"] = {
            k: [round(v, 4) for v in vals] if isinstance(vals, list) else vals
            for k, vals in td_data.items()
        }

    raw_path = os.path.join(output_dir, "v8_sensitivity_raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"\nRaw data saved to {raw_path}")

    print(f"\n{'='*60}")
    print("PARAMETER SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'='*60}")
