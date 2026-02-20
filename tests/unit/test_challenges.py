"""Tests for challenge models."""

import numpy as np

from sciops.metrics.challenges import (
    BlackBoxingModel,
    ChallengeParams,
    ChallengeSet,
    DelaySensitivityModel,
    ObservabilityGapModel,
    QualityDriftModel,
    RedundancyModel,
    TrustDecayModel,
)
from sciops.pipeline.config import StageName
from sciops.pipeline.research_unit import ResearchUnit


def _make_unit() -> ResearchUnit:
    return ResearchUnit(id=0, created_at=0, current_stage=StageName.ANALYSIS)


def test_quality_drift_degrades_true_quality():
    params = ChallengeParams(goodhart_drift_rate=0.1)
    model = QualityDriftModel(params)
    unit = _make_unit()

    model.apply(unit, ai_level=0.5, alpha=0.5)
    assert unit.quality < 1.0
    assert unit.proxy_quality > 1.0 or unit.proxy_quality == 1.0


def test_redundancy_novelty_decreases():
    params = ChallengeParams(novelty_decay_rate=0.1)
    model = RedundancyModel(params)

    n0 = model.compute_novelty_factor(0)
    n10 = model.compute_novelty_factor(10)
    n100 = model.compute_novelty_factor(100)

    assert n0 == 1.0
    assert n10 < n0
    assert n100 < n10


def test_blackboxing_decays_understanding():
    params = ChallengeParams(understanding_decay_rate=0.1)
    model = BlackBoxingModel(params)
    unit = _make_unit()

    model.update_understanding(unit, ai_level=0.5)
    assert unit.human_understanding < 1.0
    assert unit.human_understanding >= 0.0


def test_trust_decay_on_no_events():
    params = ChallengeParams()
    model = TrustDecayModel(params)

    from sciops.pipeline.state import PipelineState, StageState
    state = PipelineState(
        stages={StageName.SURVEY: StageState(name=StageName.SURVEY)}
    )

    trust = model.update(state)
    assert trust == 1.0


def test_observability_noise_grows_with_alpha():
    params = ChallengeParams()
    model = ObservabilityGapModel(params)

    noise_low = model.compute_observation_noise(alpha=0.1, timestep=50)
    noise_high = model.compute_observation_noise(alpha=0.9, timestep=50)

    assert noise_high > noise_low


def test_delay_sensitivity_amplifies_with_alpha():
    params = ChallengeParams()
    model = DelaySensitivityModel(params)

    cost_low = model.compute_delay_cost(delay_ticks=10, alpha=0.0)
    cost_high = model.compute_delay_cost(delay_ticks=10, alpha=1.0)

    assert cost_high >= cost_low


def test_challenge_set_update_all():
    challenges = ChallengeSet()
    from sciops.pipeline.state import PipelineState, StageState
    state = PipelineState(
        stages={name: StageState(name=name) for name in StageName}
    )

    metrics = challenges.update_all(state, alpha=0.5)
    assert "trust" in metrics
    assert "resilience" in metrics
    assert "novelty_factor" in metrics
    assert "observability_noise" in metrics
