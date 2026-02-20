"""Oracle strategy: perfect information, greedy look-ahead optimization."""

from __future__ import annotations

from sciops.engine.actions import (
    ActionSet,
    AdjustAILevel,
    AllocateResources,
    InvestUncertaintyReduction,
)
from sciops.engine.overhead import OverheadConfig
from sciops.pipeline.config import STAGE_ORDER, StageName
from sciops.pipeline.state import PipelineState
from sciops.strategies.base import Strategy


class OracleStrategy(Strategy):
    """
    Oracle: theoretical upper bound with perfect state observation.

    Uses full knowledge of pipeline state to make optimal decisions each step:
    1. Identifies the true bottleneck (stage with lowest throughput rate)
    2. Allocates resources optimally (proportional to inverse throughput)
    3. Maximizes AI levels greedily
    4. Invests in uncertainty reduction at the most uncertain stage
    """

    def __init__(self) -> None:
        self._alpha: float = 0.0

    @property
    def name(self) -> str:
        return "oracle"

    @property
    def overhead_config(self) -> OverheadConfig:
        return OverheadConfig(base_cost=0.05, ai_infra_cost=0.30, human_coord_cost=0.02)

    def set_alpha(self, alpha: float) -> None:
        self._alpha = alpha

    def decide(self, state: PipelineState, timestep: int) -> ActionSet:
        actions = ActionSet()

        # Maximize AI levels at all stages
        for stage_name in STAGE_ORDER:
            stage = state.stages[stage_name]
            if stage.ai_level < 1.0:
                actions.add(AdjustAILevel(stage_name, min(0.15, 1.0 - stage.ai_level)))

        # Optimal resource allocation: inverse throughput rate weighting
        total_resources = sum(s.resources for s in state.stages.values())
        throughput_rates = {}
        for stage_name in STAGE_ORDER:
            stage = state.stages[stage_name]
            rate = stage.cumulative_processed / max(1, timestep) if timestep > 0 else 1.0
            throughput_rates[stage_name] = rate

        # Inverse-throughput weighting (more resources to slower stages)
        inverse_weights = {
            name: 1.0 / max(0.01, rate) for name, rate in throughput_rates.items()
        }
        total_weight = sum(inverse_weights.values())

        if total_weight > 0:
            for stage_name in STAGE_ORDER:
                share = total_resources * inverse_weights[stage_name] / total_weight
                actions.add(AllocateResources(stage_name, max(0.3, share)))

        # Invest in uncertainty reduction at the most uncertain stage
        worst_uncertainty = max(state.stages.values(), key=lambda s: s.uncertainty)
        actions.add(InvestUncertaintyReduction(worst_uncertainty.name, 0.5))

        return actions
