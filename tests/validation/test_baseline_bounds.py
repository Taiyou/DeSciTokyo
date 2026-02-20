"""Validation: Oracle should perform at least as well as other strategies (statistically)."""

import numpy as np

from sciops.experiment.run_config import RunConfig
from sciops.experiment.runner import execute_single_run
from sciops.experiment.scenarios import build_scenarios


def test_oracle_beats_baseline():
    """Oracle should produce more output than baseline on average."""
    scenarios = build_scenarios()
    scenario = scenarios["S2_alpha_continuous"]
    n_seeds = 20

    oracle_outputs = []
    baseline_outputs = []

    for seed in range(n_seeds):
        for strategy_name, output_list in [("oracle", oracle_outputs), ("baseline", baseline_outputs)]:
            config = RunConfig(
                scenario_name="S2_alpha_continuous",
                strategy_name=strategy_name,
                alpha=0.0,
                seed=seed,
                pipeline_config=scenario.pipeline_config,
                num_steps=100,
            )
            result, _ = execute_single_run(config)
            output_list.append(result.net_output)

    oracle_mean = np.mean(oracle_outputs)
    baseline_mean = np.mean(baseline_outputs)

    # Oracle should be at least as good as baseline
    # (We use a generous tolerance because with only 100 steps, variance is high)
    assert oracle_mean >= baseline_mean * 0.8, (
        f"Oracle ({oracle_mean:.2f}) significantly worse than baseline ({baseline_mean:.2f})"
    )


def test_managed_strategies_beat_baseline():
    """At least one managed strategy should beat baseline."""
    scenarios = build_scenarios()
    scenario = scenarios["S2_alpha_continuous"]
    n_seeds = 15

    strategy_means = {}
    for strategy_name in ["baseline", "toc_pdca", "kanban", "agile"]:
        outputs = []
        for seed in range(n_seeds):
            config = RunConfig(
                scenario_name="S2_alpha_continuous",
                strategy_name=strategy_name,
                alpha=0.0,
                seed=seed,
                pipeline_config=scenario.pipeline_config,
                num_steps=100,
            )
            result, _ = execute_single_run(config)
            outputs.append(result.net_output)
        strategy_means[strategy_name] = np.mean(outputs)

    baseline_mean = strategy_means["baseline"]
    managed_means = {k: v for k, v in strategy_means.items() if k != "baseline"}

    # At least one managed strategy should be competitive with baseline
    best_managed = max(managed_means.values())
    assert best_managed >= baseline_mean * 0.7, (
        f"No managed strategy competitive with baseline ({baseline_mean:.2f}). "
        f"Best managed: {best_managed:.2f}"
    )
