"""Validation: same seed produces identical results."""

from sciops.experiment.run_config import RunConfig
from sciops.experiment.runner import execute_single_run
from sciops.experiment.scenarios import build_scenarios
from sciops.strategies.factory import list_strategies


def test_reproducibility_all_strategies():
    """Same seed must produce identical output for every strategy."""
    scenarios = build_scenarios()
    scenario = scenarios["S2_alpha_continuous"]

    for strategy_name in list_strategies():
        results = []
        for _ in range(3):
            config = RunConfig(
                scenario_name="S2_alpha_continuous",
                strategy_name=strategy_name,
                alpha=0.3,
                seed=99999,
                pipeline_config=scenario.pipeline_config,
                num_steps=50,
            )
            result, _ = execute_single_run(config)
            results.append(result)

        # All runs should produce identical output
        assert results[0].cumulative_output == results[1].cumulative_output == results[2].cumulative_output, (
            f"Reproducibility failure for {strategy_name}: "
            f"{results[0].cumulative_output} vs {results[1].cumulative_output} vs {results[2].cumulative_output}"
        )
        assert results[0].net_output == results[1].net_output == results[2].net_output
