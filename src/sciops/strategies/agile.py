"""Agile Sprint strategy: sprint-based planning and retrospective."""

from __future__ import annotations

from typing import Dict

from sciops.engine.actions import (
    ActionSet,
    AllocateResources,
    InvestUncertaintyReduction,
)
from sciops.engine.overhead import OverheadConfig
from sciops.pipeline.config import STAGE_ORDER, StageName
from sciops.pipeline.state import PipelineState
from sciops.strategies.base import Strategy


class AgileStrategy(Strategy):
    """
    Agile Sprint: 10-step sprints with planning and retrospective.

    Sprint start: plan resource allocation based on previous sprint's throughput.
    Sprint end: retrospective — identify underperforming stages, invest in improvement.
    """

    SPRINT_LENGTH = 10

    def __init__(self) -> None:
        self._sprint_start_processed: Dict[StageName, int] = {}
        self._sprint_start_output: int = 0

    @property
    def name(self) -> str:
        return "agile"

    @property
    def overhead_config(self) -> OverheadConfig:
        return OverheadConfig(base_cost=0.20, ai_infra_cost=0.0, human_coord_cost=0.25)

    def decide(self, state: PipelineState, timestep: int) -> ActionSet:
        actions = ActionSet()

        if timestep % self.SPRINT_LENGTH == 0:
            if timestep == 0:
                # First sprint: snapshot baseline
                self._snapshot(state)
                return actions

            # Retrospective: compute per-stage throughput during last sprint
            sprint_throughputs: Dict[StageName, int] = {}
            total_sprint_throughput = 0
            for stage_name in STAGE_ORDER:
                current = state.stages[stage_name].cumulative_processed
                prev = self._sprint_start_processed.get(stage_name, 0)
                tp = current - prev
                sprint_throughputs[stage_name] = tp
                total_sprint_throughput += tp

            # Plan: allocate resources inversely proportional to throughput
            # (give more to slow stages)
            total_resources = sum(s.resources for s in state.stages.values())
            if total_sprint_throughput > 0:
                inverse_weights = {}
                for stage_name in STAGE_ORDER:
                    tp = sprint_throughputs[stage_name]
                    inverse_weights[stage_name] = 1.0 / max(1, tp)
                total_weight = sum(inverse_weights.values())

                for stage_name in STAGE_ORDER:
                    share = total_resources * inverse_weights[stage_name] / total_weight
                    actions.add(AllocateResources(stage_name, max(0.5, share)))

                # Invest in uncertainty reduction for the slowest stage
                slowest = min(sprint_throughputs, key=sprint_throughputs.get)  # type: ignore[arg-type]
                actions.add(InvestUncertaintyReduction(slowest, 0.2))

            self._snapshot(state)

        return actions

    def _snapshot(self, state: PipelineState) -> None:
        """Take a snapshot of current processed counts."""
        self._sprint_start_processed = {
            name: state.stages[name].cumulative_processed for name in STAGE_ORDER
        }
        self._sprint_start_output = state.cumulative_output

    def reset(self) -> None:
        self._sprint_start_processed = {}
        self._sprint_start_output = 0
