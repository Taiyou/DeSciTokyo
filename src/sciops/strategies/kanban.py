"""Kanban strategy: WIP limits, pull-based flow control."""

from __future__ import annotations

from sciops.engine.actions import (
    ActionSet,
    AdjustWIPLimit,
    AllocateResources,
)
from sciops.engine.overhead import OverheadConfig
from sciops.pipeline.config import STAGE_ORDER
from sciops.pipeline.state import PipelineState
from sciops.strategies.base import Strategy


class KanbanStrategy(Strategy):
    """
    Kanban: WIP limits per stage, pull-based flow, bottleneck-aware allocation.

    Sets initial WIP limits proportional to base throughput.
    Continuously adjusts resources to feed the bottleneck.
    """

    ADJUST_INTERVAL = 5

    def __init__(self) -> None:
        self._initialized = False

    @property
    def name(self) -> str:
        return "kanban"

    @property
    def overhead_config(self) -> OverheadConfig:
        return OverheadConfig(base_cost=0.10, ai_infra_cost=0.0, human_coord_cost=0.10)

    def decide(self, state: PipelineState, timestep: int) -> ActionSet:
        actions = ActionSet()

        # Initialize WIP limits on first call
        if not self._initialized:
            for stage_name in STAGE_ORDER:
                # WIP limit = ~2x what the stage can process per tick
                actions.add(AdjustWIPLimit(stage_name, 8))
            self._initialized = True

        # Periodic resource reallocation
        if timestep > 0 and timestep % self.ADJUST_INTERVAL == 0:
            wip_counts = state.wip_counts()
            total_wip = max(1, sum(wip_counts.values()))
            total_resources = sum(s.resources for s in state.stages.values())

            for stage_name in STAGE_ORDER:
                # Allocate more resources to stages with higher relative WIP
                wip_ratio = wip_counts.get(stage_name, 0) / total_wip
                # Blend between equal allocation and WIP-proportional
                equal_share = total_resources / len(STAGE_ORDER)
                wip_share = total_resources * wip_ratio
                allocation = 0.4 * equal_share + 0.6 * wip_share
                actions.add(AllocateResources(stage_name, max(0.5, allocation)))

        return actions

    def reset(self) -> None:
        self._initialized = False
