"""AI-SciOps strategy: autonomous optimization with bandit-based exploration."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from sciops.engine.actions import (
    Action,
    ActionSet,
    AdjustAILevel,
    AdjustWIPLimit,
    AllocateResources,
    InvestUncertaintyReduction,
    Restructure,
)
from sciops.engine.overhead import OverheadConfig
from sciops.pipeline.config import STAGE_ORDER, StageName
from sciops.pipeline.state import PipelineState
from sciops.strategies.base import Strategy


class AISciOpsStrategy(Strategy):
    """
    AI-SciOps: Full autonomous optimization using all 5 action types.

    Uses UCB1 bandit algorithm to select among action templates.
    Tracks action history and rewards. Includes restructuring.
    """

    DECIDE_INTERVAL = 3

    def __init__(self) -> None:
        # Action templates that the bandit selects from
        self._action_templates: List[str] = [
            "focus_bottleneck",
            "raise_ai_all",
            "reduce_uncertainty_worst",
            "set_wip_limits",
            "restructure_balance",
        ]
        self._counts: Dict[str, int] = {t: 0 for t in self._action_templates}
        self._rewards: Dict[str, float] = {t: 0.0 for t in self._action_templates}
        self._prev_output: float = 0.0
        self._total_decisions: int = 0
        self._restructured: bool = False

    @property
    def name(self) -> str:
        return "ai_sciops"

    @property
    def overhead_config(self) -> OverheadConfig:
        return OverheadConfig(base_cost=0.10, ai_infra_cost=0.20, human_coord_cost=0.05)

    def decide(self, state: PipelineState, timestep: int) -> ActionSet:
        actions = ActionSet()

        if timestep % self.DECIDE_INTERVAL != 0 or timestep == 0:
            return actions

        # Update reward for previous action
        current_output = state.net_output
        reward = current_output - self._prev_output
        self._prev_output = current_output

        if self._total_decisions > 0:
            # Attribute reward to the last selected template
            last_template = getattr(self, "_last_template", None)
            if last_template and last_template in self._counts:
                self._rewards[last_template] += reward
                self._counts[last_template] += 1

        # Select next action template via UCB1
        template = self._ucb1_select()
        self._last_template = template
        self._total_decisions += 1

        # Execute selected template
        if template == "focus_bottleneck":
            self._focus_bottleneck(state, actions)
        elif template == "raise_ai_all":
            self._raise_ai_all(state, actions)
        elif template == "reduce_uncertainty_worst":
            self._reduce_uncertainty(state, actions)
        elif template == "set_wip_limits":
            self._set_wip_limits(state, actions)
        elif template == "restructure_balance":
            self._restructure(state, actions)

        return actions

    def _ucb1_select(self) -> str:
        """Select action template using UCB1."""
        # Explore untried templates first
        for t in self._action_templates:
            if self._counts[t] == 0:
                return t

        total = sum(self._counts.values())
        best_score = -float("inf")
        best_template = self._action_templates[0]

        for t in self._action_templates:
            avg_reward = self._rewards[t] / max(1, self._counts[t])
            exploration = math.sqrt(2 * math.log(total) / self._counts[t])
            score = avg_reward + exploration
            if score > best_score:
                best_score = score
                best_template = t

        return best_template

    def _focus_bottleneck(self, state: PipelineState, actions: ActionSet) -> None:
        bottleneck = state.find_bottleneck()
        total = sum(s.resources for s in state.stages.values())
        bn_share = total * 0.40
        other_share = (total - bn_share) / max(1, len(STAGE_ORDER) - 1)

        for stage_name in STAGE_ORDER:
            if stage_name == bottleneck.name:
                actions.add(AllocateResources(stage_name, bn_share))
            else:
                actions.add(AllocateResources(stage_name, other_share))

    def _raise_ai_all(self, state: PipelineState, actions: ActionSet) -> None:
        for stage_name in STAGE_ORDER:
            stage = state.stages[stage_name]
            if stage.ai_level < 0.9:
                actions.add(AdjustAILevel(stage_name, 0.1))

    def _reduce_uncertainty(self, state: PipelineState, actions: ActionSet) -> None:
        # Invest in the stage with highest uncertainty
        worst = max(state.stages.values(), key=lambda s: s.uncertainty)
        actions.add(InvestUncertaintyReduction(worst.name, 0.4))

    def _set_wip_limits(self, state: PipelineState, actions: ActionSet) -> None:
        for stage_name in STAGE_ORDER:
            wip = state.stages[stage_name].wip_count
            # Set limit to current WIP + small buffer
            actions.add(AdjustWIPLimit(stage_name, max(3, wip + 2)))

    def _restructure(self, state: PipelineState, actions: ActionSet) -> None:
        if self._restructured:
            # Only restructure once
            self._focus_bottleneck(state, actions)
            return
        actions.add(Restructure(description="rebalance_pipeline", cost=0.5, downtime_ticks=2))
        self._restructured = True
        # After restructure, equalize resources
        total = sum(s.resources for s in state.stages.values())
        per_stage = total / len(STAGE_ORDER)
        for stage_name in STAGE_ORDER:
            actions.add(AllocateResources(stage_name, per_stage))

    def reset(self) -> None:
        self._counts = {t: 0 for t in self._action_templates}
        self._rewards = {t: 0.0 for t in self._action_templates}
        self._prev_output = 0.0
        self._total_decisions = 0
        self._restructured = False
