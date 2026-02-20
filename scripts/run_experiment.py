#!/usr/bin/env python3
"""CLI entry point for running SciOps-Sim experiments."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sciops.experiment.parallel import run_experiment
from sciops.experiment.sobol import run_sobol_analysis
from sciops.io.results import ResultsStore
from sciops.analysis.statistics import compare_all_strategies, detect_phase_transition
from sciops.visualization.plots import generate_all_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SciOps-Sim: Scientific Pipeline Simulation"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=500,
        help="Number of random seeds per condition (default: 500)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of parallel workers (-1 = all CPUs, default: -1)",
    )
    parser.add_argument(
        "--num-steps", type=int, default=200,
        help="Simulation timesteps per run (default: 200)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5000,
        help="Batch size for parallel execution (default: 5000)",
    )
    parser.add_argument(
        "--sobol", action="store_true",
        help="Run Sobol sensitivity analysis instead of main experiment",
    )
    parser.add_argument(
        "--sobol-n", type=int, default=1024,
        help="Sobol sample size parameter N (default: 1024)",
    )
    parser.add_argument(
        "--sobol-strategy", type=str, default="ai_sciops",
        help="Strategy for Sobol analysis (default: ai_sciops)",
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Only run analysis and visualization on existing results",
    )
    parser.add_argument(
        "--timeseries", action="store_true",
        help="Collect per-tick time series data (increases storage)",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpointing",
    )

    args = parser.parse_args()
    output_dir = args.output_dir

    if args.analyze_only:
        _run_analysis(output_dir)
        return

    if args.sobol:
        print("Running Sobol sensitivity analysis...")
        run_sobol_analysis(
            output_dir=output_dir / "sobol",
            strategy=args.sobol_strategy,
            n=args.sobol_n,
            n_jobs=args.n_jobs,
        )
    else:
        print("Running main experiment...")
        run_experiment(
            output_dir=output_dir,
            n_seeds=args.n_seeds,
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            checkpoint=not args.no_checkpoint,
            collect_timeseries=args.timeseries,
            num_steps=args.num_steps,
        )

    _run_analysis(output_dir)


def _run_analysis(output_dir: Path) -> None:
    """Run statistical analysis and generate plots."""
    store = ResultsStore(output_dir)
    df = store.load_scalar_results()

    if df.empty:
        print("No results found. Run the experiment first.")
        return

    print(f"Loaded {len(df)} results.")

    # Statistical tests
    print("\n--- Multiple Strategy Comparison ---")
    for scenario in df["scenario"].unique():
        result = compare_all_strategies(df, scenario_filter=scenario, alpha_filter=0.0)
        print(f"\n{scenario}: H={result.h_statistic:.2f}, p={result.p_value:.4f}")
        for name, rank in sorted(result.strategy_ranks.items(), key=lambda x: x[1]):
            print(f"  {name}: mean rank = {rank:.1f}")

    # Phase transitions
    print("\n--- Phase Transitions ---")
    for scenario in df["scenario"].unique():
        pt = detect_phase_transition(df, scenario_filter=scenario)
        print(f"\n{scenario}: {pt['num_transitions']} transition(s)")
        for t in pt["transitions"]:
            print(f"  alpha={t['alpha']:.2f}: {t['from_strategy']} -> {t['to_strategy']}")

    # Generate plots
    plots_dir = output_dir / "plots"
    sobol_path = output_dir / "sobol" / "sobol_indices_ai_sciops.json"
    generate_all_plots(
        df, plots_dir,
        sobol_path=sobol_path if sobol_path.exists() else None,
    )


if __name__ == "__main__":
    main()
