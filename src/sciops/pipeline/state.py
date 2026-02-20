"""Mutable pipeline state used during simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sciops.pipeline.config import STAGE_ORDER, StageName
from sciops.pipeline.research_unit import ResearchUnit


@dataclass
class StageState:
    """Mutable state for a single pipeline stage."""

    name: StageName
    wip: List[ResearchUnit] = field(default_factory=list)
    wip_limit: Optional[int] = None
    ai_level: float = 0.0
    resources: float = 1.0
    uncertainty: float = 0.0
    cumulative_processed: int = 0
    cumulative_failed: int = 0

    @property
    def wip_count(self) -> int:
        return len(self.wip)

    def is_at_wip_limit(self) -> bool:
        if self.wip_limit is None:
            return False
        return self.wip_count >= self.wip_limit


@dataclass
class PipelineState:
    """Complete mutable state of the pipeline at a point in time."""

    stages: Dict[StageName, StageState] = field(default_factory=dict)
    timestep: int = 0
    cumulative_output: int = 0
    net_output: float = 0.0
    completed_units: List[ResearchUnit] = field(default_factory=list)
    abandoned_units: List[ResearchUnit] = field(default_factory=list)
    management_overhead: float = 0.0
    cumulative_overhead: float = 0.0
    next_unit_id: int = 0

    @property
    def total_wip(self) -> int:
        return sum(s.wip_count for s in self.stages.values())

    def total_ai_level(self) -> float:
        return sum(s.ai_level for s in self.stages.values())

    def wip_counts(self) -> Dict[StageName, int]:
        return {name: stage.wip_count for name, stage in self.stages.items()}

    def find_bottleneck(self) -> StageState:
        """Return the stage with the highest WIP (most congested)."""
        return max(self.stages.values(), key=lambda s: s.wip_count)

    def find_lowest_throughput_stage(self) -> StageState:
        """Return the stage with the lowest effective throughput estimate."""
        if not self.stages:
            raise ValueError("No stages in pipeline")
        return min(
            self.stages.values(),
            key=lambda s: s.cumulative_processed / max(1, self.timestep),
        )

    def create_unit(self) -> ResearchUnit:
        """Create a new research unit entering the pipeline."""
        unit = ResearchUnit(
            id=self.next_unit_id,
            created_at=self.timestep,
            current_stage=STAGE_ORDER[0],
        )
        self.next_unit_id += 1
        return unit
