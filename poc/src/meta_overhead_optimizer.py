"""
Meta-Overhead Optimizer: AI that Adjusts Management Costs
==========================================================
Explores the challenges of using AI to optimize management overhead itself.

Key challenges modeled:
1. Recursive overhead: AI optimization of management has its own cost
2. Noisy observability: AI can only estimate overhead from indirect signals
3. Delayed feedback: Management changes take time to manifest
4. Goodhart's Law: Optimizing proxy metrics can degrade true outcomes
5. Human trust dynamics: Aggressive automation erodes coordination quality

Each variant of the MetaAI optimizer represents a different level of
difficulty in the meta-optimization problem.
"""

import random
import math
from dataclasses import dataclass, field

from scientific_process import ProcessStep
from optimizers import Optimizer, OptimizationAction
from management_overhead import OverheadProfile


@dataclass
class MetaAIState:
    """State of the meta-AI management optimizer."""
    # Current overhead profile being applied (AI's control variables)
    current_profile: OverheadProfile = field(default_factory=OverheadProfile)
    # AI's internal estimate of what overhead is actually costing
    estimated_overhead: float = 0.0
    # True overhead (may differ from estimate due to observability limits)
    true_overhead: float = 0.0
    # Accumulated meta-optimization cost (the AI's own overhead)
    meta_cost_accumulated: float = 0.0
    # History of (action, observed_effect) pairs for learning
    adjustment_history: list = field(default_factory=list)
    # Human trust level (1.0 = full trust, 0.0 = no trust)
    human_trust: float = 1.0


class MetaAIOracleOptimizer(Optimizer):
    """
    Variant A: Oracle - Perfect Observability.

    The AI knows exactly what overhead each management activity costs
    and can observe the immediate effect of changes. This is the upper
    bound on how well meta-optimization can work.

    Uses Kanban-SciOps as base strategy with dynamically adjusted overhead.
    """

    def __init__(self):
        super().__init__("MetaAI-Oracle (Perfect Info)")
        self.state = MetaAIState()
        self.history: list[dict] = []
        # Start with Kanban-like overhead profile
        self.state.current_profile = OverheadProfile(
            base_cost=0.10,
            per_action_cost=0.02,
            complexity_scaling=0.2,
            ai_infrastructure_cost=0.15,
            learning_cost=0.15,
            human_coordination_cost=0.05,
        )
        self.meta_ai_cost_per_step = 0.05  # Cost of running the meta-AI itself

    def get_overhead_profile(self) -> OverheadProfile:
        return self.state.current_profile

    def get_meta_cost(self) -> float:
        return self.meta_ai_cost_per_step

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        for step in pipeline:
            step.throughput = step.config.base_throughput

        state = {
            "time_step": time_step,
            "throughputs": {p.config.name: p.effective_throughput() for p in pipeline},
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
        }
        self.history.append(state)

        # --- Meta-optimization: adjust overhead profile ---
        if time_step > 0 and time_step % 10 == 0:
            self._adjust_overhead(pipeline, time_step)

        # --- Base optimization (Kanban-like) ---
        progress = time_step / 100.0
        for step in pipeline:
            step.ai_assistance_level = min(
                0.85, step.config.ai_automatable * (0.3 + progress * 0.65)
            )

        # Resource allocation (bottleneck-aware)
        weights = {}
        for step in pipeline:
            wip_ratio = (step.work_in_progress + 0.1) / (step.effective_throughput() + 0.1)
            weights[step.config.name] = max(0.5, wip_ratio)
        total_weight = sum(weights.values())
        for step in pipeline:
            step.allocated_resources = (
                weights[step.config.name] / total_weight
            ) * total_resources

        # Process improvements at key points
        if time_step == 40:
            for step in pipeline:
                if step.ai_assistance_level > 0.5:
                    step.config.uncertainty *= 0.75
        if time_step == 60:
            for step in pipeline:
                if step.config.name == "Review":
                    step.config.human_review_needed *= 0.6
                elif step.config.name == "Writing":
                    step.config.human_review_needed *= 0.7
                elif step.config.name in ("Survey", "Hypothesis"):
                    step.config.base_throughput *= 1.2
                    step.throughput = step.config.base_throughput

        return pipeline

    def _adjust_overhead(self, pipeline, time_step):
        """Oracle: perfectly observes overhead and optimizes it."""
        p = self.state.current_profile

        # With perfect info, AI can find the minimum viable overhead
        # Reduce components that aren't providing proportional benefit
        if time_step >= 20:
            # Human coordination can be reduced as AI takes over
            p.human_coordination_cost = max(0.02, p.human_coordination_cost * 0.85)
        if time_step >= 40:
            # Complexity scaling gets handled by better AI models
            p.complexity_scaling = max(0.05, p.complexity_scaling * 0.8)
        if time_step >= 60:
            # Learning cost drops as system matures
            p.learning_cost = max(0.05, p.learning_cost * 0.7)

        # AI infra cost has diminishing returns floor
        p.ai_infrastructure_cost = max(0.08, p.ai_infrastructure_cost * 0.95)

        self.actions.append(OptimizationAction(
            time_step=time_step, target_process="meta",
            action_type="MetaAI-AdjustOH",
            description=f"Oracle adjusted OH: base={p.base_cost:.3f}, "
            f"human={p.human_coordination_cost:.3f}, ai_infra={p.ai_infrastructure_cost:.3f}",
        ))


class MetaAINoisyOptimizer(Optimizer):
    """
    Variant B: Noisy Observability.

    The AI cannot directly measure overhead cost. It can only observe
    throughput changes and must INFER what the overhead is doing.
    This creates a credit assignment problem: did throughput improve
    because of better research optimization, or because overhead decreased?

    Models the real-world challenge where management costs are deeply
    entangled with productive work.
    """

    def __init__(self):
        super().__init__("MetaAI-Noisy (Indirect Obs.)")
        self.state = MetaAIState()
        self.history: list[dict] = []
        self.state.current_profile = OverheadProfile(
            base_cost=0.10, per_action_cost=0.02, complexity_scaling=0.2,
            ai_infrastructure_cost=0.15, learning_cost=0.15,
            human_coordination_cost=0.05,
        )
        self.meta_ai_cost_per_step = 0.08  # Higher: more analysis needed
        self.throughput_history: list[float] = []
        self.overhead_estimates: list[float] = []
        self.noise_std = 0.15  # Observation noise

    def get_overhead_profile(self) -> OverheadProfile:
        return self.state.current_profile

    def get_meta_cost(self) -> float:
        return self.meta_ai_cost_per_step

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        for step in pipeline:
            step.throughput = step.config.base_throughput

        # Record system throughput (what AI can observe)
        if time_step > 0:
            last_tp = sum(p.effective_throughput() for p in pipeline)
            # Add noise to observation
            observed_tp = last_tp + random.gauss(0, self.noise_std)
            self.throughput_history.append(observed_tp)

        state = {
            "time_step": time_step,
            "throughputs": {p.config.name: p.effective_throughput() for p in pipeline},
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
        }
        self.history.append(state)

        # --- Meta-optimization with noisy signals ---
        if time_step > 10 and time_step % 10 == 0:
            self._noisy_adjust_overhead(pipeline, time_step)

        # Base optimization (same as Oracle)
        progress = time_step / 100.0
        for step in pipeline:
            step.ai_assistance_level = min(
                0.85, step.config.ai_automatable * (0.3 + progress * 0.65)
            )

        weights = {}
        for step in pipeline:
            wip_ratio = (step.work_in_progress + 0.1) / (step.effective_throughput() + 0.1)
            weights[step.config.name] = max(0.5, wip_ratio)
        total_weight = sum(weights.values())
        for step in pipeline:
            step.allocated_resources = (
                weights[step.config.name] / total_weight
            ) * total_resources

        if time_step == 40:
            for step in pipeline:
                if step.ai_assistance_level > 0.5:
                    step.config.uncertainty *= 0.75
        if time_step == 60:
            for step in pipeline:
                if step.config.name == "Review":
                    step.config.human_review_needed *= 0.6
                elif step.config.name == "Writing":
                    step.config.human_review_needed *= 0.7
                elif step.config.name in ("Survey", "Hypothesis"):
                    step.config.base_throughput *= 1.2
                    step.throughput = step.config.base_throughput

        return pipeline

    def _noisy_adjust_overhead(self, pipeline, time_step):
        """Adjust overhead based on noisy throughput observations."""
        p = self.state.current_profile

        if len(self.throughput_history) < 10:
            return

        # AI tries to estimate the effect of overhead changes
        recent = self.throughput_history[-10:]
        earlier = self.throughput_history[-20:-10] if len(self.throughput_history) >= 20 else self.throughput_history[:10]

        avg_recent = sum(recent) / len(recent)
        avg_earlier = sum(earlier) / len(earlier)
        delta = avg_recent - avg_earlier

        # AI's inference: if throughput improved, current overhead is okay
        # If throughput declined, maybe overhead is too high
        # But this is noisy and can lead to wrong conclusions
        if delta < -0.1:
            # Throughput dropped -- try reducing overhead
            # But sometimes this is wrong (maybe research just got harder)
            component = random.choice([
                "human_coordination_cost", "base_cost",
                "complexity_scaling", "ai_infrastructure_cost"
            ])
            old_val = getattr(p, component)
            # Cautious reduction due to uncertainty
            new_val = max(0.01, old_val * 0.9)
            setattr(p, component, new_val)

            self.actions.append(OptimizationAction(
                time_step=time_step, target_process="meta",
                action_type="MetaAI-NoisyAdjust",
                description=f"Noisy inference: tp_delta={delta:.3f}, "
                f"reduced {component} {old_val:.3f}→{new_val:.3f}",
            ))
        elif delta > 0.2:
            # Throughput improved -- but overconfident AI might reduce too much
            if random.random() < 0.4:
                # Sometimes AI over-reduces (Goodhart's Law)
                p.human_coordination_cost = max(0.01, p.human_coordination_cost * 0.8)
                self.actions.append(OptimizationAction(
                    time_step=time_step, target_process="meta",
                    action_type="MetaAI-OverReduce",
                    description=f"Overconfident reduction: human_coord→{p.human_coordination_cost:.3f}",
                ))


class MetaAIDelayedOptimizer(Optimizer):
    """
    Variant C: Delayed Feedback.

    Management changes take 10-15 steps to show their full effect.
    The AI must learn to account for this delay, or it will:
    1. Make a change
    2. See no immediate effect
    3. Make another change (oscillation)
    4. First change finally kicks in + second change = overshoot

    This models the real-world problem of organizational inertia:
    changing from Agile to Lean doesn't show results for months.
    """

    def __init__(self):
        super().__init__("MetaAI-Delayed (Lagged Feedback)")
        self.state = MetaAIState()
        self.history: list[dict] = []
        self.state.current_profile = OverheadProfile(
            base_cost=0.10, per_action_cost=0.02, complexity_scaling=0.2,
            ai_infrastructure_cost=0.15, learning_cost=0.15,
            human_coordination_cost=0.05,
        )
        self.meta_ai_cost_per_step = 0.06
        self.pending_changes: list[tuple[int, str, float, float]] = []  # (apply_at, field, old, new)
        self.feedback_delay = 12  # Steps before a change takes effect
        self.change_cooldown = 0  # Steps to wait before next change

    def get_overhead_profile(self) -> OverheadProfile:
        return self.state.current_profile

    def get_meta_cost(self) -> float:
        return self.meta_ai_cost_per_step

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        for step in pipeline:
            step.throughput = step.config.base_throughput

        # Apply delayed changes that have matured
        p = self.state.current_profile
        for change in list(self.pending_changes):
            apply_at, fld, old_val, new_val = change
            if time_step >= apply_at:
                # Gradual application (not instant)
                current = getattr(p, fld)
                # Move 30% toward target each step it's "active"
                steps_active = time_step - apply_at
                blend = min(1.0, steps_active * 0.3)
                blended = current * (1 - blend) + new_val * blend
                setattr(p, fld, blended)
                if blend >= 0.95:
                    self.pending_changes.remove(change)

        state = {
            "time_step": time_step,
            "throughputs": {p.config.name: p.effective_throughput() for p in pipeline},
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
        }
        self.history.append(state)

        # --- Meta-optimization with delayed feedback ---
        if self.change_cooldown > 0:
            self.change_cooldown -= 1

        if time_step > 10 and time_step % 8 == 0 and self.change_cooldown == 0:
            self._delayed_adjust_overhead(pipeline, time_step)

        # Base optimization
        progress = time_step / 100.0
        for step in pipeline:
            step.ai_assistance_level = min(
                0.85, step.config.ai_automatable * (0.3 + progress * 0.65)
            )

        weights = {}
        for step in pipeline:
            wip_ratio = (step.work_in_progress + 0.1) / (step.effective_throughput() + 0.1)
            weights[step.config.name] = max(0.5, wip_ratio)
        total_weight = sum(weights.values())
        for step in pipeline:
            step.allocated_resources = (
                weights[step.config.name] / total_weight
            ) * total_resources

        if time_step == 40:
            for step in pipeline:
                if step.ai_assistance_level > 0.5:
                    step.config.uncertainty *= 0.75
        if time_step == 60:
            for step in pipeline:
                if step.config.name == "Review":
                    step.config.human_review_needed *= 0.6
                elif step.config.name == "Writing":
                    step.config.human_review_needed *= 0.7
                elif step.config.name in ("Survey", "Hypothesis"):
                    step.config.base_throughput *= 1.2
                    step.throughput = step.config.base_throughput

        return pipeline

    def _delayed_adjust_overhead(self, pipeline, time_step):
        """Schedule overhead changes that take effect after a delay."""
        p = self.state.current_profile

        if len(self.history) < 15:
            return

        recent = self.history[-8:]
        earlier = self.history[-16:-8]
        avg_recent_tp = sum(
            min(h["throughputs"].values()) for h in recent
        ) / len(recent)
        avg_earlier_tp = sum(
            min(h["throughputs"].values()) for h in earlier
        ) / len(earlier)

        delta = avg_recent_tp - avg_earlier_tp

        if delta < 0:
            # AI sees decline, schedules a reduction
            # But doesn't know the PREVIOUS pending change might fix it
            targets = [
                ("human_coordination_cost", max(0.02, p.human_coordination_cost * 0.85)),
                ("base_cost", max(0.03, p.base_cost * 0.85)),
                ("complexity_scaling", max(0.05, p.complexity_scaling * 0.85)),
            ]
            field_name, new_val = random.choice(targets)
            old_val = getattr(p, field_name)

            apply_at = time_step + self.feedback_delay
            self.pending_changes.append((apply_at, field_name, old_val, new_val))
            self.change_cooldown = 5  # Don't change again too soon

            self.actions.append(OptimizationAction(
                time_step=time_step, target_process="meta",
                action_type="MetaAI-DelayedChange",
                description=f"Scheduled {field_name} {old_val:.3f}→{new_val:.3f} "
                f"(effective at step {apply_at}), delta={delta:.3f}",
            ))

            # Problem: if multiple pending changes overlap, they can overshoot
            if len(self.pending_changes) > 2:
                self.actions.append(OptimizationAction(
                    time_step=time_step, target_process="meta",
                    action_type="MetaAI-Oscillation",
                    description=f"WARNING: {len(self.pending_changes)} pending changes, "
                    f"risk of overshoot/oscillation",
                ))


class MetaAIRecursiveOptimizer(Optimizer):
    """
    Variant D: Recursive Overhead (Self-Referential Cost).

    The AI's own management optimization has a cost that GROWS
    as it tries to optimize more aggressively. This creates a
    fundamental limit: the meta-optimization cost must be less
    than the savings it produces.

    Models the real-world problem: hiring a management consultant
    to reduce management overhead... the consultant IS overhead.

    The AI faces a cost function:
        total_cost = management_OH + meta_AI_OH(optimization_intensity)
    where meta_AI_OH grows with optimization_intensity.
    """

    def __init__(self):
        super().__init__("MetaAI-Recursive (Self-Ref. Cost)")
        self.state = MetaAIState()
        self.history: list[dict] = []
        self.state.current_profile = OverheadProfile(
            base_cost=0.10, per_action_cost=0.02, complexity_scaling=0.2,
            ai_infrastructure_cost=0.15, learning_cost=0.15,
            human_coordination_cost=0.05,
        )
        self.optimization_intensity = 0.5  # 0.0 = no meta-opt, 1.0 = max
        self.base_meta_cost = 0.05
        self.intensity_history: list[float] = []

    def get_overhead_profile(self) -> OverheadProfile:
        return self.state.current_profile

    def get_meta_cost(self) -> float:
        """Meta cost grows quadratically with optimization intensity."""
        return self.base_meta_cost * (1.0 + self.optimization_intensity ** 2 * 3.0)

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        for step in pipeline:
            step.throughput = step.config.base_throughput

        state = {
            "time_step": time_step,
            "throughputs": {p.config.name: p.effective_throughput() for p in pipeline},
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
        }
        self.history.append(state)

        # --- Recursive meta-optimization ---
        if time_step > 5 and time_step % 8 == 0:
            self._recursive_adjust(pipeline, time_step)

        self.intensity_history.append(self.optimization_intensity)

        # Base optimization
        progress = time_step / 100.0
        for step in pipeline:
            step.ai_assistance_level = min(
                0.85, step.config.ai_automatable * (0.3 + progress * 0.65)
            )

        weights = {}
        for step in pipeline:
            wip_ratio = (step.work_in_progress + 0.1) / (step.effective_throughput() + 0.1)
            weights[step.config.name] = max(0.5, wip_ratio)
        total_weight = sum(weights.values())
        for step in pipeline:
            step.allocated_resources = (
                weights[step.config.name] / total_weight
            ) * total_resources

        if time_step == 40:
            for step in pipeline:
                if step.ai_assistance_level > 0.5:
                    step.config.uncertainty *= 0.75
        if time_step == 60:
            for step in pipeline:
                if step.config.name == "Review":
                    step.config.human_review_needed *= 0.6
                elif step.config.name == "Writing":
                    step.config.human_review_needed *= 0.7
                elif step.config.name in ("Survey", "Hypothesis"):
                    step.config.base_throughput *= 1.2
                    step.throughput = step.config.base_throughput

        return pipeline

    def _recursive_adjust(self, pipeline, time_step):
        """Adjust overhead, but each adjustment costs more."""
        p = self.state.current_profile

        # The reduction in management OH is proportional to intensity
        reduction_factor = 1.0 - self.optimization_intensity * 0.15

        if time_step >= 20:
            p.human_coordination_cost = max(0.02, p.human_coordination_cost * reduction_factor)
        if time_step >= 40:
            p.complexity_scaling = max(0.05, p.complexity_scaling * reduction_factor)
        if time_step >= 60:
            p.learning_cost = max(0.05, p.learning_cost * reduction_factor)

        # Now the key question: should AI increase or decrease intensity?
        # Higher intensity = more OH reduction but also more meta-cost
        meta_cost = self.get_meta_cost()
        current_oh = p.compute_overhead(time_step, len(pipeline), 1)

        # AI tries to find optimal intensity using gradient-free search
        # Test small increase and decrease
        self.optimization_intensity = min(1.0, self.optimization_intensity + 0.05)
        meta_cost_up = self.get_meta_cost()
        self.optimization_intensity -= 0.1
        meta_cost_down = self.get_meta_cost() if self.optimization_intensity >= 0 else float('inf')
        self.optimization_intensity += 0.05  # restore

        # Total cost = management OH + meta OH
        total_current = current_oh + meta_cost
        # Estimate benefit of higher vs lower intensity
        if meta_cost_up < current_oh * 0.8:
            # Meta cost is still small relative to management OH, increase
            self.optimization_intensity = min(1.0, self.optimization_intensity + 0.05)
        elif meta_cost > current_oh * 0.5:
            # Meta cost is becoming too large, decrease
            self.optimization_intensity = max(0.1, self.optimization_intensity - 0.08)

        self.actions.append(OptimizationAction(
            time_step=time_step, target_process="meta",
            action_type="MetaAI-Recursive",
            description=f"intensity={self.optimization_intensity:.2f}, "
            f"meta_cost={self.get_meta_cost():.3f}, mgmt_oh={current_oh:.3f}, "
            f"total={current_oh + self.get_meta_cost():.3f}",
        ))


class MetaAITrustDecayOptimizer(Optimizer):
    """
    Variant E: Human Trust Dynamics.

    When AI reduces management overhead (especially human coordination),
    it can erode the implicit trust and knowledge-sharing that happens
    in meetings and reviews. This causes:
    1. Increased uncertainty (fewer sanity checks)
    2. Higher failure rates (less peer review)
    3. Knowledge silos (less cross-team learning)

    The AI must balance:
    - Reducing overhead (saves resources) vs.
    - Maintaining trust (prevents quality degradation)

    This models the real challenge of AI replacing human management:
    you can't just remove meetings without replacing the functions
    those meetings served.
    """

    def __init__(self):
        super().__init__("MetaAI-TrustDecay (Human Factor)")
        self.state = MetaAIState()
        self.state.human_trust = 1.0
        self.history: list[dict] = []
        self.state.current_profile = OverheadProfile(
            base_cost=0.10, per_action_cost=0.02, complexity_scaling=0.2,
            ai_infrastructure_cost=0.15, learning_cost=0.15,
            human_coordination_cost=0.05,
        )
        self.meta_ai_cost_per_step = 0.06
        self.original_human_coord = 0.05
        self.trust_visible = False  # AI can't see trust directly at first
        self.trust_awareness_step = None

    def get_overhead_profile(self) -> OverheadProfile:
        return self.state.current_profile

    def get_meta_cost(self) -> float:
        return self.meta_ai_cost_per_step

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        for step in pipeline:
            step.throughput = step.config.base_throughput

        state = {
            "time_step": time_step,
            "throughputs": {p.config.name: p.effective_throughput() for p in pipeline},
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
            "trust": self.state.human_trust,
        }
        self.history.append(state)

        # --- Apply trust effects on pipeline quality ---
        # Low trust = higher uncertainty and failure rates
        trust = self.state.human_trust
        if trust < 0.8:
            trust_penalty = (0.8 - trust) * 0.5  # up to 0.25 penalty at trust=0.3
            for step in pipeline:
                # Uncertainty increases when humans don't coordinate
                # (simulates knowledge silos, missed errors)
                if step.config.name in ("Hypothesis", "Experiment", "Review"):
                    # These processes suffer most from lack of human coordination
                    effective_uncertainty = step.config.uncertainty * (1 + trust_penalty)
                    # Temporarily boost uncertainty for this step
                    step.config.uncertainty = min(0.8, effective_uncertainty)

        # --- Meta-optimization: adjust overhead ---
        if time_step > 5 and time_step % 10 == 0:
            self._trust_aware_adjust(pipeline, time_step)

        # --- Trust dynamics ---
        p = self.state.current_profile
        coord_reduction = 1.0 - (p.human_coordination_cost / max(0.01, self.original_human_coord))

        # Trust decays when coordination is reduced
        if coord_reduction > 0.3:
            decay_rate = coord_reduction * 0.03
            self.state.human_trust = max(0.3, self.state.human_trust - decay_rate)

        # Trust slowly recovers when coordination is maintained
        if coord_reduction < 0.1:
            self.state.human_trust = min(1.0, self.state.human_trust + 0.005)

        # --- AI detects trust problem (with delay) ---
        if not self.trust_visible and self.state.human_trust < 0.65:
            # AI starts seeing quality metrics degrade
            if len(self.history) >= 5:
                recent_tp = [
                    min(h["throughputs"].values()) for h in self.history[-5:]
                ]
                if len(recent_tp) >= 2 and recent_tp[-1] < recent_tp[0] * 0.9:
                    self.trust_visible = True
                    self.trust_awareness_step = time_step
                    self.actions.append(OptimizationAction(
                        time_step=time_step, target_process="meta",
                        action_type="MetaAI-TrustAlert",
                        description=f"AI detected quality degradation, "
                        f"trust={self.state.human_trust:.2f}. "
                        f"Investigating human coordination.",
                    ))

        # Base optimization
        progress = time_step / 100.0
        for step in pipeline:
            step.ai_assistance_level = min(
                0.85, step.config.ai_automatable * (0.3 + progress * 0.65)
            )

        weights = {}
        for step in pipeline:
            wip_ratio = (step.work_in_progress + 0.1) / (step.effective_throughput() + 0.1)
            weights[step.config.name] = max(0.5, wip_ratio)
        total_weight = sum(weights.values())
        for step in pipeline:
            step.allocated_resources = (
                weights[step.config.name] / total_weight
            ) * total_resources

        if time_step == 40:
            for step in pipeline:
                if step.ai_assistance_level > 0.5:
                    step.config.uncertainty *= 0.75
        if time_step == 60:
            for step in pipeline:
                if step.config.name == "Review":
                    step.config.human_review_needed *= 0.6
                elif step.config.name == "Writing":
                    step.config.human_review_needed *= 0.7
                elif step.config.name in ("Survey", "Hypothesis"):
                    step.config.base_throughput *= 1.2
                    step.throughput = step.config.base_throughput

        return pipeline

    def _trust_aware_adjust(self, pipeline, time_step):
        """Adjust overhead with awareness of trust dynamics."""
        p = self.state.current_profile

        if self.trust_visible:
            # AI now knows trust is a factor -- try to restore it
            if self.state.human_trust < 0.7:
                p.human_coordination_cost = min(
                    self.original_human_coord,
                    p.human_coordination_cost * 1.3  # Increase coordination
                )
                self.actions.append(OptimizationAction(
                    time_step=time_step, target_process="meta",
                    action_type="MetaAI-RestoreTrust",
                    description=f"Restoring coordination: {p.human_coordination_cost:.3f}, "
                    f"trust={self.state.human_trust:.2f}",
                ))
            # Still optimize other components
            p.complexity_scaling = max(0.05, p.complexity_scaling * 0.9)
            p.base_cost = max(0.03, p.base_cost * 0.9)
        else:
            # AI doesn't see trust yet -- aggressively reduces human overhead
            p.human_coordination_cost = max(0.01, p.human_coordination_cost * 0.7)
            p.base_cost = max(0.03, p.base_cost * 0.9)

            self.actions.append(OptimizationAction(
                time_step=time_step, target_process="meta",
                action_type="MetaAI-AggressiveReduce",
                description=f"Reduced human_coord→{p.human_coordination_cost:.3f} "
                f"(trust={self.state.human_trust:.2f}, unaware)",
            ))
