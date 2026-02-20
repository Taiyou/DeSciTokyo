"""ResearchUnit: atomic unit of work flowing through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from sciops.pipeline.config import StageName


@dataclass
class ResearchUnit:
    """A single research item progressing through the pipeline."""

    id: int
    created_at: int
    current_stage: StageName
    quality: float = 1.0
    proxy_quality: float = 1.0
    human_understanding: float = 1.0
    loop_count: int = 0
    total_time: int = 0
    time_in_current_stage: int = 0
    history: List[Tuple[int, StageName]] = field(default_factory=list)

    def advance_to(self, stage: StageName, timestep: int) -> None:
        """Move this unit to a new stage."""
        self.history.append((timestep, self.current_stage))
        self.current_stage = stage
        self.time_in_current_stage = 0

    def send_back_to(self, stage: StageName, timestep: int) -> None:
        """Send this unit back to an earlier stage (feedback loop)."""
        self.history.append((timestep, self.current_stage))
        self.current_stage = stage
        self.loop_count += 1
        self.time_in_current_stage = 0

    def tick(self) -> None:
        """Advance time counters by one timestep."""
        self.total_time += 1
        self.time_in_current_stage += 1
