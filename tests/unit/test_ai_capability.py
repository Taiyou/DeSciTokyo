"""Tests for AI capability scaling."""

from sciops.pipeline.ai_capability import compute_ai_params, scale_ai_capability
from sciops.pipeline.config import PipelineConfig, StageName


def test_alpha_zero_returns_base():
    config = PipelineConfig()
    scaled = scale_ai_capability(config, alpha=0.0)
    # Should return same config
    assert scaled.stages[StageName.SURVEY].ai_automatable == config.stages[StageName.SURVEY].ai_automatable


def test_alpha_one_increases_ai_automatable():
    config = PipelineConfig()
    scaled = scale_ai_capability(config, alpha=1.0)

    for name in StageName:
        assert scaled.stages[name].ai_automatable >= config.stages[name].ai_automatable


def test_alpha_one_decreases_human_review():
    config = PipelineConfig()
    scaled = scale_ai_capability(config, alpha=1.0)

    for name in StageName:
        assert scaled.stages[name].human_review_needed <= config.stages[name].human_review_needed


def test_ai_params_interpolation():
    params_0 = compute_ai_params(0.0)
    params_05 = compute_ai_params(0.5)
    params_1 = compute_ai_params(1.0)

    assert params_0.uncertainty_reduction_rate == 0.5
    assert params_1.uncertainty_reduction_rate == 0.85
    assert params_0.uncertainty_reduction_rate < params_05.uncertainty_reduction_rate < params_1.uncertainty_reduction_rate


def test_intermediate_alpha_interpolates():
    config = PipelineConfig()
    scaled_03 = scale_ai_capability(config, alpha=0.3)
    scaled_07 = scale_ai_capability(config, alpha=0.7)

    # Higher alpha should mean higher ai_automatable
    for name in StageName:
        assert scaled_03.stages[name].ai_automatable <= scaled_07.stages[name].ai_automatable
