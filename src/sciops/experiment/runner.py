"""Single-run executor: integrates engine + strategy + challenges + metrics."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from sciops.engine.engine import SimulationEngine
from sciops.engine.overhead import compute_overhead
from sciops.experiment.run_config import RunConfig
from sciops.io.results import RunResult, TimeSeriesResult
from sciops.metrics.challenges import ChallengeParams, ChallengeSet
from sciops.metrics.collector import MetricsCollector
from sciops.pipeline.ai_capability import scale_ai_capability
from sciops.strategies.factory import create_strategy
from sciops.strategies.oracle import OracleStrategy


def execute_single_run(
    config: RunConfig,
    challenge_params: Optional[ChallengeParams] = None,
    collect_timeseries: bool = False,
) -> Tuple[RunResult, Optional[TimeSeriesResult]]:
    """
    Execute a single simulation run.

    Designed for parallel execution: no shared mutable state.
    """
    rng = np.random.default_rng(config.seed)

    # Scale pipeline config by alpha
    scaled_config = scale_ai_capability(config.pipeline_config, config.alpha)

    # Create components
    engine = SimulationEngine(scaled_config, alpha=config.alpha)
    state = engine.create_initial_state()
    strategy = create_strategy(config.strategy_name)

    if isinstance(strategy, OracleStrategy):
        strategy.set_alpha(config.alpha)

    collector = MetricsCollector()
    challenges = ChallengeSet(challenge_params or ChallengeParams())

    # Main simulation loop
    for t in range(config.num_steps):
        # Strategy decides
        actions = strategy.decide(state, t)

        # Apply actions
        engine.apply_actions(actions, state)

        # Compute management overhead
        total_ai = state.total_ai_level()
        num_active = sum(1 for s in state.stages.values() if s.wip_count > 0)
        overhead = compute_overhead(strategy.overhead_config, total_ai, num_active, t)
        state.management_overhead = overhead
        state.cumulative_overhead += overhead

        # Step the engine
        engine.step(state, rng)

        # Apply challenge models
        challenge_metrics = challenges.update_all(state, config.alpha)

        # Record metrics
        collector.record(state, challenge_metrics)

    # Build scalar result
    completed = state.completed_units
    if completed:
        mean_ct = float(np.mean([u.total_time for u in completed]))
        q_true = float(np.mean([u.quality for u in completed]))
        q_proxy = float(np.mean([u.proxy_quality for u in completed]))
        q_drift = float(np.mean([abs(u.quality - u.proxy_quality) for u in completed]))
    else:
        mean_ct = 0.0
        q_true = 0.0
        q_proxy = 0.0
        q_drift = 0.0

    max_wip = max((tick.total_wip for tick in collector.history), default=0)
    bottleneck = state.find_bottleneck()

    run_result = RunResult(
        scenario=config.scenario_name,
        strategy=config.strategy_name,
        alpha=config.alpha,
        seed=config.seed,
        cumulative_output=len(completed),
        net_output=state.net_output,
        mean_cycle_time=mean_ct,
        quality_true_mean=q_true,
        quality_proxy_mean=q_proxy,
        quality_drift=q_drift,
        total_management_overhead=state.cumulative_overhead,
        max_wip=max_wip,
        bottleneck_stage=bottleneck.name.value,
    )

    # Build time-series result if requested
    ts_result = None
    if collect_timeseries:
        arrays = collector.to_arrays()
        ts_result = TimeSeriesResult(
            scenario=config.scenario_name,
            strategy=config.strategy_name,
            alpha=config.alpha,
            seed=config.seed,
            wip_per_stage=arrays.get("wip_per_stage", np.array([])),
            throughput_per_stage=np.array([]),
            cumulative_output=arrays.get("cumulative_output", np.array([])),
            quality_true=arrays.get("quality_true", np.array([])),
            quality_proxy=arrays.get("quality_proxy", np.array([])),
            management_overhead=arrays.get("management_overhead", np.array([])),
            resilience=arrays.get("resilience", np.array([])),
            trust=arrays.get("trust", np.array([])),
        )

    return run_result, ts_result
