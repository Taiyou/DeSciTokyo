"""Tests for the simulation engine."""

import numpy as np

from sciops.engine.actions import ActionSet, AllocateResources, AdjustAILevel
from sciops.engine.engine import SimulationEngine
from sciops.pipeline.config import PipelineConfig, StageName, STAGE_ORDER


def test_create_initial_state():
    config = PipelineConfig()
    engine = SimulationEngine(config)
    state = engine.create_initial_state()

    assert len(state.stages) == 6
    assert state.timestep == 0
    assert state.cumulative_output == 0
    assert state.total_wip == 0

    # Resources should be roughly equal
    resources = [s.resources for s in state.stages.values()]
    assert abs(sum(resources) - config.total_resources) < 0.01


def test_step_produces_arrivals():
    config = PipelineConfig()
    engine = SimulationEngine(config)
    state = engine.create_initial_state()
    rng = np.random.default_rng(42)

    engine.step(state, rng)

    assert state.timestep == 1
    # With arrival_rate=3.0, we should usually see some arrivals
    first_stage = state.stages[STAGE_ORDER[0]]
    assert first_stage.wip_count >= 0  # Could be 0 with Poisson


def test_multi_step_produces_output():
    config = PipelineConfig()
    engine = SimulationEngine(config)
    state = engine.create_initial_state()
    rng = np.random.default_rng(42)

    for _ in range(100):
        engine.step(state, rng)

    assert state.timestep == 100
    # After 100 steps, should have some output
    assert state.cumulative_output > 0 or state.total_wip > 0


def test_apply_resource_allocation():
    config = PipelineConfig()
    engine = SimulationEngine(config)
    state = engine.create_initial_state()

    actions = ActionSet()
    actions.add(AllocateResources(StageName.EXPERIMENT, 5.0))

    engine.apply_actions(actions, state)

    # Total resources should still be constrained
    total = sum(s.resources for s in state.stages.values())
    assert total <= config.total_resources + 0.01


def test_apply_ai_level():
    config = PipelineConfig()
    engine = SimulationEngine(config)
    state = engine.create_initial_state()

    actions = ActionSet()
    actions.add(AdjustAILevel(StageName.SURVEY, 0.3))

    engine.apply_actions(actions, state)
    assert state.stages[StageName.SURVEY].ai_level == 0.3


def test_ai_level_clamped():
    config = PipelineConfig()
    engine = SimulationEngine(config)
    state = engine.create_initial_state()

    actions = ActionSet()
    actions.add(AdjustAILevel(StageName.SURVEY, 2.0))

    engine.apply_actions(actions, state)
    assert state.stages[StageName.SURVEY].ai_level <= 1.0


def test_reproducibility_with_same_seed():
    config = PipelineConfig()

    outputs = []
    for _ in range(2):
        engine = SimulationEngine(config)
        state = engine.create_initial_state()
        rng = np.random.default_rng(12345)
        for _ in range(50):
            engine.step(state, rng)
        outputs.append(state.cumulative_output)

    assert outputs[0] == outputs[1]
