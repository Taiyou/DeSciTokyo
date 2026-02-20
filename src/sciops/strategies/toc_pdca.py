"""TOC + PDCA strategy: bottleneck identification and iterative improvement."""

from __future__ import annotations

from sciops.engine.actions import (
    ActionSet,
    AllocateResources,
    InvestUncertaintyReduction,
)
from sciops.engine.overhead import OverheadConfig
from sciops.pipeline.config import STAGE_ORDER
from sciops.pipeline.state import PipelineState
from sciops.strategies.base import Strategy


class TOCPDCAStrategy(Strategy):
    """
    Theory of Constraints + Plan-Do-Check-Act.

    Every 10 steps:
    1. PLAN: Identify the bottleneck (highest WIP stage)
    2. DO: Reallocate resources toward the bottleneck
    3. CHECK: Compare output to previous cycle
    4. ACT: Invest in uncertainty reduction at bottleneck if improvement was small
    """

    CYCLE_LENGTH = 10

    def __init__(self) -> None:
        self._prev_output: int = 0
        self._bottleneck_stage = STAGE_ORDER[2]  # Default: Experiment

    @property
    def name(self) -> str:
        return "toc_pdca"

    @property
    def overhead_config(self) -> OverheadConfig:
        return OverheadConfig(base_cost=0.15, ai_infra_cost=0.0, human_coord_cost=0.15)

    def decide(self, state: PipelineState, timestep: int) -> ActionSet:
        actions = ActionSet()

        if timestep % self.CYCLE_LENGTH != 0 or timestep == 0:
            return actions

        # CHECK: measure improvement
        current_output = state.cumulative_output
        improvement = current_output - self._prev_output

        # PLAN: identify bottleneck (highest WIP = most congested)
        bottleneck = state.find_bottleneck()
        self._bottleneck_stage = bottleneck.name

        # DO: reallocate resources — give bottleneck 2x average, reduce others
        num_stages = len(STAGE_ORDER)
        total = sum(s.resources for s in state.stages.values())
        bottleneck_share = total * 0.35
        other_share = (total - bottleneck_share) / max(1, num_stages - 1)

        for stage_name in STAGE_ORDER:
            if stage_name == self._bottleneck_stage:
                actions.add(AllocateResources(stage_name, bottleneck_share))
            else:
                actions.add(AllocateResources(stage_name, other_share))

        # ACT: if improvement was modest, invest in uncertainty reduction
        if improvement < self.CYCLE_LENGTH * 0.5:
            actions.add(InvestUncertaintyReduction(self._bottleneck_stage, 0.3))

        self._prev_output = current_output
        return actions

    def reset(self) -> None:
        self._prev_output = 0
        self._bottleneck_stage = STAGE_ORDER[2]
