"""Scenario definitions for the five experimental conditions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from sciops.pipeline.config import (
    DEFAULT_STAGES,
    FeedbackConfig,
    PipelineConfig,
    StageConfig,
    StageName,
)


@dataclass(frozen=True)
class ScenarioDefinition:
    """A named experimental scenario."""

    name: str
    description: str
    pipeline_config: PipelineConfig


def build_scenarios(config_dir: Optional[Path] = None) -> Dict[str, ScenarioDefinition]:
    """Build all 5 scenarios programmatically."""
    scenarios: Dict[str, ScenarioDefinition] = {}

    ds = dict(DEFAULT_STAGES)
    df = FeedbackConfig()

    # S1: Baseline serial (no feedback loops)
    scenarios["S1_baseline"] = ScenarioDefinition(
        name="S1_baseline",
        description="Serial pipeline, no feedback loops",
        pipeline_config=PipelineConfig(
            stages=ds,
            feedback=FeedbackConfig(enable_feedback=False),
        ),
    )

    # S2: Alpha continuous (primary scenario with feedback)
    scenarios["S2_alpha_continuous"] = ScenarioDefinition(
        name="S2_alpha_continuous",
        description="Full pipeline with feedback loops, alpha sweep",
        pipeline_config=PipelineConfig(stages=ds, feedback=df),
    )

    # S3: Persistent bottleneck at Review (human_review_needed fixed at 0.8)
    s3_stages = dict(ds)
    s3_stages[StageName.REVIEW] = StageConfig(
        name=StageName.REVIEW,
        base_throughput=0.6,
        uncertainty=0.2,
        failure_rate=0.1,
        ai_automatable=0.4,
        human_review_needed=0.8,
    )
    scenarios["S3_bottleneck"] = ScenarioDefinition(
        name="S3_bottleneck",
        description="Persistent bottleneck: Review.human_review_needed fixed at 0.8",
        pipeline_config=PipelineConfig(stages=s3_stages, feedback=df),
    )

    # S4: Theory lab (Hypothesis-dominant, Experiment fast)
    s4_stages = dict(ds)
    s4_stages[StageName.HYPOTHESIS] = StageConfig(
        name=StageName.HYPOTHESIS,
        base_throughput=1.0,
        uncertainty=0.5,
        failure_rate=0.12,
        ai_automatable=0.5,
        human_review_needed=0.6,
    )
    s4_stages[StageName.EXPERIMENT] = StageConfig(
        name=StageName.EXPERIMENT,
        base_throughput=1.5,
        uncertainty=0.3,
        failure_rate=0.08,
        ai_automatable=0.5,
        human_review_needed=0.2,
    )
    scenarios["S4_theory_lab"] = ScenarioDefinition(
        name="S4_theory_lab",
        description="Hypothesis-dominant research, fast experiments",
        pipeline_config=PipelineConfig(stages=s4_stages, feedback=df),
    )

    # S5: High-risk exploratory research
    s5_stages = {
        StageName.SURVEY: StageConfig(
            name=StageName.SURVEY,
            base_throughput=2.0,
            uncertainty=0.4,
            failure_rate=0.1,
            ai_automatable=0.8,
            human_review_needed=0.2,
        ),
        StageName.HYPOTHESIS: StageConfig(
            name=StageName.HYPOTHESIS,
            base_throughput=1.5,
            uncertainty=0.6,
            failure_rate=0.2,
            ai_automatable=0.6,
            human_review_needed=0.5,
        ),
        StageName.EXPERIMENT: StageConfig(
            name=StageName.EXPERIMENT,
            base_throughput=0.8,
            uncertainty=0.7,
            failure_rate=0.3,
            ai_automatable=0.3,
            human_review_needed=0.3,
        ),
        StageName.ANALYSIS: StageConfig(
            name=StageName.ANALYSIS,
            base_throughput=1.8,
            uncertainty=0.5,
            failure_rate=0.15,
            ai_automatable=0.9,
            human_review_needed=0.4,
        ),
        StageName.WRITING: StageConfig(
            name=StageName.WRITING,
            base_throughput=1.2,
            uncertainty=0.3,
            failure_rate=0.08,
            ai_automatable=0.7,
            human_review_needed=0.6,
        ),
        StageName.REVIEW: StageConfig(
            name=StageName.REVIEW,
            base_throughput=0.6,
            uncertainty=0.4,
            failure_rate=0.1,
            ai_automatable=0.4,
            human_review_needed=0.8,
        ),
    }
    s5_feedback = FeedbackConfig(
        p_revision=0.4,
        p_minor_revision=0.3,
        p_major_rejection=0.1,
    )
    scenarios["S5_high_risk"] = ScenarioDefinition(
        name="S5_high_risk",
        description="High-risk research: elevated failure and uncertainty",
        pipeline_config=PipelineConfig(
            stages=s5_stages,
            feedback=s5_feedback,
            arrival_rate=2.0,
        ),
    )

    return scenarios
