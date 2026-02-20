"""Parallel experiment runner with checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from joblib import Parallel, delayed
from tqdm import tqdm

from sciops.experiment.run_config import RunConfig
from sciops.experiment.runner import execute_single_run
from sciops.experiment.scenarios import build_scenarios
from sciops.io.checkpoint import CheckpointManager
from sciops.io.results import ResultsStore
from sciops.metrics.challenges import ChallengeParams
from sciops.strategies.factory import list_strategies

ALPHA_VALUES: List[float] = [round(a * 0.05, 2) for a in range(21)]


def generate_run_configs(
    scenarios: dict,
    strategies: List[str],
    alphas: List[float],
    seeds: List[int],
    num_steps: int = 200,
) -> List[RunConfig]:
    """Generate the full cross-product of RunConfigs."""
    configs: List[RunConfig] = []
    for scenario_name, scenario_def in scenarios.items():
        for strategy_name in strategies:
            for alpha in alphas:
                for seed in seeds:
                    configs.append(
                        RunConfig(
                            scenario_name=scenario_name,
                            strategy_name=strategy_name,
                            alpha=alpha,
                            seed=seed,
                            pipeline_config=scenario_def.pipeline_config,
                            num_steps=num_steps,
                        )
                    )
    return configs


def run_experiment(
    output_dir: Path,
    n_seeds: int = 500,
    n_jobs: int = -1,
    batch_size: int = 5000,
    checkpoint: bool = True,
    collect_timeseries: bool = False,
    challenge_params: Optional[ChallengeParams] = None,
    num_steps: int = 200,
    config_dir: Optional[Path] = None,
) -> Path:
    """
    Run the full experiment suite with parallel execution and checkpointing.

    Total runs: 5 scenarios x 7 strategies x 21 alphas x n_seeds
    Default: 5 * 7 * 21 * 500 = 367,500 runs
    """
    output_dir = Path(output_dir)
    scenarios = build_scenarios(config_dir)
    strategies = list_strategies()
    seeds = list(range(n_seeds))

    all_configs = generate_run_configs(scenarios, strategies, ALPHA_VALUES, seeds, num_steps)

    store = ResultsStore(output_dir)
    ckpt: Optional[CheckpointManager] = None

    if checkpoint:
        ckpt = CheckpointManager(output_dir / "checkpoints")
        ckpt.load()
        completed_keys = ckpt.get_all_completed()
        remaining = [c for c in all_configs if c.key not in completed_keys]
    else:
        remaining = all_configs

    total = len(all_configs)
    done = total - len(remaining)
    print(f"Total runs: {total}, Already completed: {done}, Remaining: {len(remaining)}")

    if not remaining:
        print("All runs already completed.")
        return output_dir

    # Execute in batches
    n_batches = (len(remaining) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(remaining))
        batch = remaining[start:end]

        print(f"Batch {batch_idx + 1}/{n_batches}: {len(batch)} runs")

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(execute_single_run)(cfg, challenge_params, collect_timeseries)
            for cfg in tqdm(batch, desc=f"Batch {batch_idx + 1}")
        )

        # Save scalar results
        run_results = [r[0] for r in results]
        store.save_scalar_results(run_results)

        # Save time series if collected
        if collect_timeseries:
            for _, ts in results:
                if ts is not None:
                    store.save_timeseries(ts)

        # Update checkpoint
        if ckpt is not None:
            ckpt.mark_batch_completed([cfg.key for cfg in batch])
            ckpt.save()

        done += len(batch)
        print(f"Progress: {done}/{total} ({100 * done / total:.1f}%)")

    print(f"Experiment complete. Results in {output_dir}")
    return output_dir
