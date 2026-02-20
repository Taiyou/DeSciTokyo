"""Foundation data model: stage definitions, pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class StageName(str, Enum):
    SURVEY = "Survey"
    HYPOTHESIS = "Hypothesis"
    EXPERIMENT = "Experiment"
    ANALYSIS = "Analysis"
    WRITING = "Writing"
    REVIEW = "Review"


STAGE_ORDER: tuple[StageName, ...] = (
    StageName.SURVEY,
    StageName.HYPOTHESIS,
    StageName.EXPERIMENT,
    StageName.ANALYSIS,
    StageName.WRITING,
    StageName.REVIEW,
)


@dataclass(frozen=True)
class StageConfig:
    """Configuration for a single pipeline stage."""

    name: StageName
    base_throughput: float
    uncertainty: float
    failure_rate: float
    ai_automatable: float
    human_review_needed: float


@dataclass(frozen=True)
class FeedbackConfig:
    """Feedback loop configuration."""

    enable_feedback: bool = True
    p_revision: float = 0.2  # Analysis → Experiment
    p_minor_revision: float = 0.15  # Review → Writing
    p_major_rejection: float = 0.05  # Review → Hypothesis
    max_loops: int = 5


DEFAULT_STAGES: Dict[StageName, StageConfig] = {
    StageName.SURVEY: StageConfig(
        name=StageName.SURVEY,
        base_throughput=2.0,
        uncertainty=0.2,
        failure_rate=0.05,
        ai_automatable=0.8,
        human_review_needed=0.2,
    ),
    StageName.HYPOTHESIS: StageConfig(
        name=StageName.HYPOTHESIS,
        base_throughput=1.5,
        uncertainty=0.4,
        failure_rate=0.1,
        ai_automatable=0.6,
        human_review_needed=0.5,
    ),
    StageName.EXPERIMENT: StageConfig(
        name=StageName.EXPERIMENT,
        base_throughput=0.8,
        uncertainty=0.5,
        failure_rate=0.15,
        ai_automatable=0.3,
        human_review_needed=0.3,
    ),
    StageName.ANALYSIS: StageConfig(
        name=StageName.ANALYSIS,
        base_throughput=1.8,
        uncertainty=0.3,
        failure_rate=0.08,
        ai_automatable=0.9,
        human_review_needed=0.4,
    ),
    StageName.WRITING: StageConfig(
        name=StageName.WRITING,
        base_throughput=1.2,
        uncertainty=0.3,
        failure_rate=0.05,
        ai_automatable=0.7,
        human_review_needed=0.6,
    ),
    StageName.REVIEW: StageConfig(
        name=StageName.REVIEW,
        base_throughput=0.6,
        uncertainty=0.2,
        failure_rate=0.1,
        ai_automatable=0.4,
        human_review_needed=0.8,
    ),
}


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline configuration."""

    stages: Dict[StageName, StageConfig] = field(default_factory=lambda: dict(DEFAULT_STAGES))
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    arrival_rate: float = 3.0
    total_resources: float = 10.0
