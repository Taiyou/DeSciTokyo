"""
Extended Comparison Experiment
===============================
Runs all 6 strategies (3 original + 3 new) and generates comparison results.
"""

import json
import os
import random
from dataclasses import asdict

from scientific_process import create_default_pipeline
from optimizers import BaselineOptimizer, TOCPDCAOptimizer, AISciOpsOptimizer
from advanced_optimizers import (
    KanbanSciOpsOptimizer,
    AdaptiveSciOpsOptimizer,
    HolisticSciOpsOptimizer,
)
from simulator import Simulator, SimulationResult


def run_all_strategies(
    time_steps: int = 100,
    total_resources: float = 6.0,
    input_rate: float = 2.0,
    seed: int = 42,
    output_dir: str = "results",
) -> dict[str, SimulationResult]:
    """Run all 6 optimization strategies."""

    optimizers = [
        BaselineOptimizer(),
        TOCPDCAOptimizer(pdca_cycle_length=10),
        AISciOpsOptimizer(),
        KanbanSciOpsOptimizer(),
        AdaptiveSciOpsOptimizer(),
        HolisticSciOpsOptimizer(),
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
        filepath = os.path.join(output_dir, f"v2_{safe_name}.json")
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
    """Print formatted comparison of all strategies."""
    baseline_output = results["Baseline (No Optimization)"].total_output

    print(f"\n{'='*75}")
    print("FULL COMPARISON: 6 STRATEGIES")
    print(f"{'='*75}")
    print(f"{'Strategy':<45} {'Output':>8} {'vs Base':>10} {'BN-TP':>8}")
    print(f"{'-'*75}")

    sorted_results = sorted(results.items(), key=lambda x: x[1].total_output, reverse=True)
    for name, result in sorted_results:
        improvement = (result.total_output - baseline_output) / baseline_output * 100
        bn_tp = result.metrics[-1].bottleneck_throughput
        marker = " ***" if result.total_output == max(r.total_output for r in results.values()) else ""
        print(f"{name:<45} {result.total_output:>8.2f} {improvement:>+9.1f}% {bn_tp:>8.2f}{marker}")

    print(f"\n{'='*75}")
    print("DETAILED FINAL STATE COMPARISON")
    print(f"{'='*75}")

    processes = ["Survey", "Hypothesis", "Experiment", "Analysis", "Writing", "Review"]
    for proc in processes:
        print(f"\n  {proc}:")
        print(f"  {'Strategy':<42} {'TP':>6} {'Done':>8} {'AI':>5} {'Rework':>8} {'Backlog':>8}")
        for name, result in sorted_results:
            s = result.final_state[proc]
            short = name[:40]
            print(f"  {short:<42} {s['throughput']:>6.2f} {s['completed_units']:>8.1f} "
                  f"{s['ai_assistance_level']:>5.2f} {s['rework_units']:>8.1f} "
                  f"{s['human_review_backlog']:>8.2f}")


if __name__ == "__main__":
    results = run_all_strategies(
        time_steps=100,
        total_resources=6.0,
        input_rate=2.0,
        seed=42,
        output_dir=os.path.join(os.path.dirname(__file__), "..", "results"),
    )
    print_comparison(results)
