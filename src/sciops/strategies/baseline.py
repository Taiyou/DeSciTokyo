"""Baseline strategy: no management, equal resource allocation."""

from __future__ import annotations

from sciops.engine.actions import ActionSet
from sciops.engine.overhead import OverheadConfig
from sciops.pipeline.state import PipelineState
from sciops.strategies.base import Strategy


class BaselineStrategy(Strategy):
    """No active management. Resources stay equal, no optimization."""

    @property
    def name(self) -> str:
        return "baseline"

    @property
    def overhead_config(self) -> OverheadConfig:
        return OverheadConfig(base_cost=0.0, ai_infra_cost=0.0, human_coord_cost=0.0)

    def decide(self, state: PipelineState, timestep: int) -> ActionSet:
        return ActionSet()
