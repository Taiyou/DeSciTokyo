"""AI-Assisted strategy: AI-enhanced bottleneck detection, rule-based decisions."""

from __future__ import annotations

from sciops.engine.actions import (
    ActionSet,
    AdjustAILevel,
    AllocateResources,
    InvestUncertaintyReduction,
)
from sciops.engine.overhead import OverheadConfig
from sciops.pipeline.config import STAGE_ORDER
from sciops.pipeline.state import PipelineState
from sciops.strategies.base import Strategy


class AIAssistedStrategy(Strategy):
    """
    AI-Assisted management: AI-enhanced bottleneck detection with rule-based
    resource allocation. Gradually increases AI level. No structure changes.
    """

    ASSESS_INTERVAL = 5
    AI_INCREMENT = 0.05

    @property
    def name(self) -> str:
        return "ai_assisted"

    @property
    def overhead_config(self) -> OverheadConfig:
        return OverheadConfig(base_cost=0.10, ai_infra_cost=0.15, human_coord_cost=0.05)

    def decide(self, state: PipelineState, timestep: int) -> ActionSet:
        actions = ActionSet()

        if timestep % self.ASSESS_INTERVAL != 0 or timestep == 0:
            return actions

        # Gradually increase AI level at all stages
        for stage_name in STAGE_ORDER:
            stage = state.stages[stage_name]
            if stage.ai_level < 0.8:
                actions.add(AdjustAILevel(stage_name, self.AI_INCREMENT))

        # Identify bottleneck using WIP analysis
        bottleneck = state.find_bottleneck()
        total_resources = sum(s.resources for s in state.stages.values())

        # Allocate: bottleneck gets 30%, rest shared equally
        bn_share = total_resources * 0.30
        other_share = (total_resources - bn_share) / max(1, len(STAGE_ORDER) - 1)

        for stage_name in STAGE_ORDER:
            if stage_name == bottleneck.name:
                actions.add(AllocateResources(stage_name, bn_share))
            else:
                actions.add(AllocateResources(stage_name, other_share))

        # Invest in uncertainty reduction at bottleneck
        actions.add(InvestUncertaintyReduction(bottleneck.name, 0.25))

        return actions
