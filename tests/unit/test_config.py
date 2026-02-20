"""Tests for pipeline configuration."""

from sciops.pipeline.config import (
    DEFAULT_STAGES,
    STAGE_ORDER,
    FeedbackConfig,
    PipelineConfig,
    StageConfig,
    StageName,
)


def test_stage_order_length():
    assert len(STAGE_ORDER) == 6


def test_stage_order_starts_with_survey():
    assert STAGE_ORDER[0] == StageName.SURVEY


def test_stage_order_ends_with_review():
    assert STAGE_ORDER[-1] == StageName.REVIEW


def test_default_stages_complete():
    for name in StageName:
        assert name in DEFAULT_STAGES


def test_stage_config_frozen():
    cfg = StageConfig(
        name=StageName.SURVEY,
        base_throughput=2.0,
        uncertainty=0.2,
        failure_rate=0.05,
        ai_automatable=0.8,
        human_review_needed=0.2,
    )
    try:
        cfg.base_throughput = 5.0  # type: ignore[misc]
        assert False, "Should be frozen"
    except AttributeError:
        pass


def test_feedback_config_defaults():
    fb = FeedbackConfig()
    assert fb.enable_feedback is True
    assert fb.p_revision == 0.2
    assert fb.p_minor_revision == 0.15
    assert fb.p_major_rejection == 0.05
    assert fb.max_loops == 5


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert len(cfg.stages) == 6
    assert cfg.arrival_rate == 3.0
    assert cfg.total_resources == 10.0


def test_stage_throughput_values_positive():
    for stage in DEFAULT_STAGES.values():
        assert stage.base_throughput > 0
        assert 0 <= stage.uncertainty <= 1
        assert 0 <= stage.failure_rate <= 1
        assert 0 <= stage.ai_automatable <= 1
        assert 0 <= stage.human_review_needed <= 1
