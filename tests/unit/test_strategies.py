"""Tests for management strategies."""

import numpy as np

from sciops.engine.engine import SimulationEngine
from sciops.pipeline.config import PipelineConfig
from sciops.strategies.factory import create_strategy, list_strategies


def test_list_strategies():
    strategies = list_strategies()
    assert len(strategies) == 7
    assert "baseline" in strategies
    assert "oracle" in strategies
    assert "ai_sciops" in strategies


def test_create_all_strategies():
    for name in list_strategies():
        strategy = create_strategy(name)
        assert strategy.name == name
        assert strategy.overhead_config is not None


def test_baseline_returns_empty_actions():
    strategy = create_strategy("baseline")
    config = PipelineConfig()
    engine = SimulationEngine(config)
    state = engine.create_initial_state()

    actions = strategy.decide(state, 0)
    assert len(actions) == 0


def test_each_strategy_runs_without_error():
    """Smoke test: run each strategy for 50 steps."""
    config = PipelineConfig()

    for name in list_strategies():
        engine = SimulationEngine(config)
        state = engine.create_initial_state()
        strategy = create_strategy(name)
        rng = np.random.default_rng(42)

        for t in range(50):
            actions = strategy.decide(state, t)
            engine.apply_actions(actions, state)
            engine.step(state, rng)

        # Should not crash and timestep should advance
        assert state.timestep == 50


def test_strategy_reset():
    for name in list_strategies():
        strategy = create_strategy(name)
        strategy.reset()  # Should not raise
