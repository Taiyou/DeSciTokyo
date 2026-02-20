"""RunConfig: fully specified configuration for a single simulation run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from sciops.pipeline.config import PipelineConfig


@dataclass(frozen=True)
class RunConfig:
    """Fully specified configuration for one simulation run."""

    scenario_name: str
    strategy_name: str
    alpha: float
    seed: int
    pipeline_config: PipelineConfig
    num_steps: int = 200

    @property
    def key(self) -> Tuple[str, str, float, int]:
        """Unique identifier for checkpoint/deduplication."""
        return (self.scenario_name, self.strategy_name, self.alpha, self.seed)
