"""Tests for ResearchUnit."""

from sciops.pipeline.config import StageName
from sciops.pipeline.research_unit import ResearchUnit


def test_creation():
    unit = ResearchUnit(id=0, created_at=5, current_stage=StageName.SURVEY)
    assert unit.id == 0
    assert unit.created_at == 5
    assert unit.quality == 1.0
    assert unit.loop_count == 0
    assert unit.total_time == 0


def test_tick():
    unit = ResearchUnit(id=0, created_at=0, current_stage=StageName.SURVEY)
    unit.tick()
    assert unit.total_time == 1
    assert unit.time_in_current_stage == 1


def test_advance_to():
    unit = ResearchUnit(id=0, created_at=0, current_stage=StageName.SURVEY)
    unit.advance_to(StageName.HYPOTHESIS, timestep=3)
    assert unit.current_stage == StageName.HYPOTHESIS
    assert unit.time_in_current_stage == 0
    assert len(unit.history) == 1
    assert unit.history[0] == (3, StageName.SURVEY)


def test_send_back_increments_loop():
    unit = ResearchUnit(id=0, created_at=0, current_stage=StageName.ANALYSIS)
    unit.send_back_to(StageName.EXPERIMENT, timestep=10)
    assert unit.current_stage == StageName.EXPERIMENT
    assert unit.loop_count == 1
    assert unit.time_in_current_stage == 0
