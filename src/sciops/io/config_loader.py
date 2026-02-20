"""YAML configuration loader (optional, fallback to programmatic defaults)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from sciops.pipeline.config import (
    FeedbackConfig,
    PipelineConfig,
    StageConfig,
    StageName,
)


def load_scenario_config(scenario_name: str, config_dir: Path) -> PipelineConfig:
    """Load a pipeline configuration from a YAML file."""
    path = config_dir / f"{scenario_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    stages: Dict[StageName, StageConfig] = {}
    for stage_data in data.get("stages", []):
        name = StageName(stage_data["name"])
        stages[name] = StageConfig(
            name=name,
            base_throughput=float(stage_data["base_throughput"]),
            uncertainty=float(stage_data["uncertainty"]),
            failure_rate=float(stage_data["failure_rate"]),
            ai_automatable=float(stage_data["ai_automatable"]),
            human_review_needed=float(stage_data["human_review_needed"]),
        )

    fb_data = data.get("feedback", {})
    feedback = FeedbackConfig(
        enable_feedback=fb_data.get("enable_feedback", True),
        p_revision=fb_data.get("p_revision", 0.2),
        p_minor_revision=fb_data.get("p_minor_revision", 0.15),
        p_major_rejection=fb_data.get("p_major_rejection", 0.05),
        max_loops=fb_data.get("max_loops", 5),
    )

    return PipelineConfig(
        stages=stages,
        feedback=feedback,
        arrival_rate=float(data.get("arrival_rate", 3.0)),
        total_resources=float(data.get("total_resources", 10.0)),
    )
