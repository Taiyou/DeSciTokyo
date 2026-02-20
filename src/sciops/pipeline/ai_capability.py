"""Continuous AI capability parameterization (alpha ∈ [0.0, 1.0])."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sciops.pipeline.config import PipelineConfig, StageConfig, StageName


# Interpolation targets: (stage, base_value_at_alpha0, value_at_alpha1)
_AI_AUTOMATABLE_TARGETS: Dict[StageName, tuple[float, float]] = {
    StageName.SURVEY:     (0.80, 0.95),
    StageName.HYPOTHESIS: (0.60, 0.90),
    StageName.EXPERIMENT: (0.30, 0.70),
    StageName.ANALYSIS:   (0.90, 0.98),
    StageName.WRITING:    (0.70, 0.95),
    StageName.REVIEW:     (0.40, 0.85),
}

_HUMAN_REVIEW_TARGETS: Dict[StageName, tuple[float, float]] = {
    StageName.SURVEY:     (0.20, 0.02),
    StageName.HYPOTHESIS: (0.50, 0.05),
    StageName.EXPERIMENT: (0.30, 0.10),
    StageName.ANALYSIS:   (0.40, 0.03),
    StageName.WRITING:    (0.60, 0.05),
    StageName.REVIEW:     (0.80, 0.10),
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


@dataclass(frozen=True)
class AICapabilityParams:
    """Derived AI capability parameters for a given alpha."""

    alpha: float
    uncertainty_reduction_rate: float
    failure_reduction_rate: float


def compute_ai_params(alpha: float) -> AICapabilityParams:
    """Compute derived AI parameters for a given alpha value."""
    return AICapabilityParams(
        alpha=alpha,
        uncertainty_reduction_rate=_lerp(0.50, 0.85, alpha),
        failure_reduction_rate=_lerp(0.30, 0.70, alpha),
    )


def scale_ai_capability(base_config: PipelineConfig, alpha: float) -> PipelineConfig:
    """
    Create a new PipelineConfig with AI-scaled stage parameters.

    alpha = 0.0: current AI capability (uses base_config values)
    alpha = 1.0: AI-superior world
    Intermediate values: linear interpolation
    """
    if alpha == 0.0:
        return base_config

    new_stages: Dict[StageName, StageConfig] = {}
    for name, stage in base_config.stages.items():
        ai_auto_base, ai_auto_sup = _AI_AUTOMATABLE_TARGETS[name]
        hr_base, hr_sup = _HUMAN_REVIEW_TARGETS[name]

        new_stages[name] = StageConfig(
            name=name,
            base_throughput=stage.base_throughput,
            uncertainty=stage.uncertainty,
            failure_rate=stage.failure_rate,
            ai_automatable=_lerp(ai_auto_base, ai_auto_sup, alpha),
            human_review_needed=_lerp(hr_base, hr_sup, alpha),
        )

    return PipelineConfig(
        stages=new_stages,
        feedback=base_config.feedback,
        arrival_rate=base_config.arrival_rate,
        total_resources=base_config.total_resources,
    )
