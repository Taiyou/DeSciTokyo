"""Integration test: full single run with each strategy."""

import pytest

from sciops.experiment.run_config import RunConfig
from sciops.experiment.runner import execute_single_run
from sciops.experiment.scenarios import build_scenarios
from sciops.strategies.factory import list_strategies


@pytest.fixture
def scenarios():
    return build_scenarios()


@pytest.mark.parametrize("strategy_name", list_strategies())
def test_single_run_each_strategy(scenarios, strategy_name):
    """Each strategy should complete a full simulation run without error."""
    scenario = scenarios["S2_alpha_continuous"]
    config = RunConfig(
        scenario_name="S2_alpha_continuous",
        strategy_name=strategy_name,
        alpha=0.0,
        seed=42,
        pipeline_config=scenario.pipeline_config,
        num_steps=50,
    )

    result, ts = execute_single_run(config, collect_timeseries=False)

    assert result.cumulative_output >= 0
    assert result.net_output >= 0
    assert result.total_management_overhead >= 0
    assert ts is None


@pytest.mark.parametrize("strategy_name", list_strategies())
def test_single_run_with_timeseries(scenarios, strategy_name):
    """Should produce time series data when requested."""
    scenario = scenarios["S2_alpha_continuous"]
    config = RunConfig(
        scenario_name="S2_alpha_continuous",
        strategy_name=strategy_name,
        alpha=0.0,
        seed=42,
        pipeline_config=scenario.pipeline_config,
        num_steps=30,
    )

    result, ts = execute_single_run(config, collect_timeseries=True)

    assert result.cumulative_output >= 0
    assert ts is not None
    assert len(ts.cumulative_output) == 30


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_single_run_alpha_values(scenarios, alpha):
    """AI-SciOps should run at different alpha levels."""
    scenario = scenarios["S2_alpha_continuous"]
    config = RunConfig(
        scenario_name="S2_alpha_continuous",
        strategy_name="ai_sciops",
        alpha=alpha,
        seed=42,
        pipeline_config=scenario.pipeline_config,
        num_steps=50,
    )

    result, _ = execute_single_run(config)
    assert result.cumulative_output >= 0


@pytest.mark.parametrize("scenario_name", ["S1_baseline", "S3_bottleneck", "S4_theory_lab", "S5_high_risk"])
def test_single_run_each_scenario(scenarios, scenario_name):
    """Each scenario should work with the baseline strategy."""
    scenario = scenarios[scenario_name]
    config = RunConfig(
        scenario_name=scenario_name,
        strategy_name="baseline",
        alpha=0.0,
        seed=42,
        pipeline_config=scenario.pipeline_config,
        num_steps=50,
    )

    result, _ = execute_single_run(config)
    assert result.cumulative_output >= 0
