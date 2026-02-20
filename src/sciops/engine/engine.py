"""Core tick-based simulation engine."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from sciops.engine.actions import (
    Action,
    ActionSet,
    AdjustAILevel,
    AdjustWIPLimit,
    AllocateResources,
    InvestUncertaintyReduction,
    Restructure,
)
from sciops.pipeline.ai_capability import compute_ai_params
from sciops.pipeline.config import (
    STAGE_ORDER,
    FeedbackConfig,
    PipelineConfig,
    StageConfig,
    StageName,
)
from sciops.pipeline.research_unit import ResearchUnit
from sciops.pipeline.state import PipelineState, StageState


class SimulationEngine:
    """Tick-based simulation engine for the scientific pipeline."""

    def __init__(self, config: PipelineConfig, alpha: float = 0.0) -> None:
        self.config = config
        self.alpha = alpha
        self.ai_params = compute_ai_params(alpha)
        self._restructure_downtime: int = 0

    def create_initial_state(self) -> PipelineState:
        """Create an initialized pipeline state with equal resource allocation."""
        num_stages = len(STAGE_ORDER)
        per_stage = self.config.total_resources / num_stages

        stages: Dict[StageName, StageState] = {}
        for name in STAGE_ORDER:
            stage_cfg = self.config.stages[name]
            stages[name] = StageState(
                name=name,
                resources=per_stage,
                uncertainty=stage_cfg.uncertainty,
            )

        return PipelineState(stages=stages)

    def apply_actions(self, action_set: ActionSet, state: PipelineState) -> None:
        """Apply a set of strategy actions to the pipeline state."""
        for action in action_set:
            self._apply_single_action(action, state)

    def _apply_single_action(self, action: Action, state: PipelineState) -> None:
        if isinstance(action, AllocateResources):
            stage = state.stages[action.stage]
            stage.resources = max(0.0, action.amount)
            # Enforce total resource constraint
            total = sum(s.resources for s in state.stages.values())
            if total > self.config.total_resources:
                scale = self.config.total_resources / total
                for s in state.stages.values():
                    s.resources *= scale

        elif isinstance(action, AdjustAILevel):
            stage = state.stages[action.stage]
            stage.ai_level = max(0.0, min(1.0, stage.ai_level + action.delta))

        elif isinstance(action, InvestUncertaintyReduction):
            stage = state.stages[action.stage]
            ai_level = stage.ai_level
            efficiency = 0.1 * (1.0 + ai_level * 1.5)
            investment = max(0.0, action.amount)
            reduction = efficiency * investment / (1.0 + efficiency * investment)
            stage.uncertainty = max(0.05, stage.uncertainty * (1.0 - reduction))

        elif isinstance(action, AdjustWIPLimit):
            stage = state.stages[action.stage]
            stage.wip_limit = action.limit

        elif isinstance(action, Restructure):
            self._restructure_downtime = action.downtime_ticks
            state.cumulative_overhead += action.cost

    def step(self, state: PipelineState, rng: np.random.Generator) -> None:
        """Execute one simulation timestep."""
        # Handle restructure downtime
        if self._restructure_downtime > 0:
            self._restructure_downtime -= 1
            state.timestep += 1
            # Only tick existing units, no processing
            for stage in state.stages.values():
                for unit in stage.wip:
                    unit.tick()
            return

        # Phase 1: Arrivals
        self._process_arrivals(state, rng)

        # Phase 2: Process each stage (in reverse order to avoid double-processing)
        feedback_queue: List[tuple[ResearchUnit, StageName]] = []
        completed: List[ResearchUnit] = []

        for stage_name in reversed(STAGE_ORDER):
            stage_state = state.stages[stage_name]
            stage_cfg = self.config.stages[stage_name]

            processed, failed, feedback = self._process_stage(
                stage_state, stage_cfg, state, rng
            )

            feedback_queue.extend(feedback)

            # Move processed units to next stage or output
            next_stage = self._next_stage(stage_name)
            for unit in processed:
                if next_stage is None:
                    # Reached end of pipeline
                    completed.append(unit)
                else:
                    next_state = state.stages[next_stage]
                    if not next_state.is_at_wip_limit():
                        unit.advance_to(next_stage, state.timestep)
                        next_state.wip.append(unit)
                    # If at WIP limit, unit stays in current stage (backpressure)

        # Phase 3: Process feedback loops
        if self.config.feedback.enable_feedback:
            for unit, target_stage in feedback_queue:
                target_state = state.stages[target_stage]
                if not target_state.is_at_wip_limit():
                    unit.send_back_to(target_stage, state.timestep)
                    target_state.wip.append(unit)

        # Phase 4: Collect output
        for unit in completed:
            state.cumulative_output += 1
            quality_penalty = max(0.0, 1.0 - unit.loop_count * 0.05)
            state.net_output += unit.quality * quality_penalty
            state.completed_units.append(unit)

        # Phase 5: Tick time for remaining units
        for stage in state.stages.values():
            for unit in stage.wip:
                unit.tick()

        state.timestep += 1

    def _process_arrivals(self, state: PipelineState, rng: np.random.Generator) -> None:
        """Generate new research units entering the pipeline."""
        first_stage = state.stages[STAGE_ORDER[0]]
        if first_stage.is_at_wip_limit():
            return

        # Poisson arrivals
        n_arrivals = rng.poisson(self.config.arrival_rate)
        for _ in range(n_arrivals):
            if first_stage.is_at_wip_limit():
                break
            unit = state.create_unit()
            first_stage.wip.append(unit)

    def _process_stage(
        self,
        stage_state: StageState,
        stage_cfg: StageConfig,
        pipeline_state: PipelineState,
        rng: np.random.Generator,
    ) -> tuple[List[ResearchUnit], List[ResearchUnit], List[tuple[ResearchUnit, StageName]]]:
        """
        Process units in a single stage.

        Returns: (processed_units, failed_units, feedback_list)
        where feedback_list contains (unit, target_stage) pairs.
        """
        processed: List[ResearchUnit] = []
        failed: List[ResearchUnit] = []
        feedback: List[tuple[ResearchUnit, StageName]] = []
        remaining: List[ResearchUnit] = []

        # Compute effective throughput
        ai_level = stage_state.ai_level
        noise = rng.uniform(-1.0, 1.0) * stage_state.uncertainty
        base = stage_cfg.base_throughput * stage_state.resources

        # AI automation bonus
        ai_bonus = 1.0 + ai_level * stage_cfg.ai_automatable
        # Human review bottleneck
        human_factor = (1.0 - stage_cfg.human_review_needed) + stage_cfg.human_review_needed * (1.0 - ai_level * 0.5)
        # Uncertainty noise
        noise_factor = max(0.1, 1.0 - abs(noise))

        effective_throughput = base * ai_bonus * human_factor * noise_factor
        capacity = max(0, int(math.floor(effective_throughput)))

        # Failure and feedback rates
        ai_fail_reduction = ai_level * self.ai_params.failure_reduction_rate
        effective_failure_rate = stage_cfg.failure_rate * (1.0 - ai_fail_reduction)

        for unit in stage_state.wip:
            if len(processed) + len(feedback) >= capacity:
                remaining.append(unit)
                continue

            # Abandonment check
            if unit.loop_count >= self.config.feedback.max_loops:
                pipeline_state.abandoned_units.append(unit)
                stage_state.cumulative_failed += 1
                continue

            # Failure check
            if rng.random() < effective_failure_rate:
                failed.append(unit)
                stage_state.cumulative_failed += 1
                continue

            # Feedback routing
            fb_target = self._check_feedback(stage_state.name, unit, rng)
            if fb_target is not None:
                feedback.append((unit, fb_target))
            else:
                processed.append(unit)
                stage_state.cumulative_processed += 1

        stage_state.wip = remaining
        return processed, failed, feedback

    def _check_feedback(
        self,
        stage_name: StageName,
        unit: ResearchUnit,
        rng: np.random.Generator,
    ) -> StageName | None:
        """Check if a unit should be sent back via feedback loop."""
        if not self.config.feedback.enable_feedback:
            return None

        fb = self.config.feedback

        if stage_name == StageName.ANALYSIS:
            if rng.random() < fb.p_revision:
                return StageName.EXPERIMENT

        elif stage_name == StageName.REVIEW:
            r = rng.random()
            if r < fb.p_major_rejection:
                return StageName.HYPOTHESIS
            elif r < fb.p_major_rejection + fb.p_minor_revision:
                return StageName.WRITING

        return None

    def _next_stage(self, current: StageName) -> StageName | None:
        """Return the next stage in the pipeline, or None if at the end."""
        idx = STAGE_ORDER.index(current)
        if idx + 1 >= len(STAGE_ORDER):
            return None
        return STAGE_ORDER[idx + 1]
