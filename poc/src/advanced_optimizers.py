"""
Advanced Process Optimization Strategies
==========================================
Three new strategies that address weaknesses found in the original AI-SciOps:

1. Kanban-SciOps: Pull-based flow control with WIP limits
2. Adaptive-SciOps: Data-driven stage transitions (not fixed time)
3. Holistic-SciOps: Combined best practices + aggressive parallelization
"""

import random
from dataclasses import dataclass, field

from scientific_process import ProcessStep
from optimizers import Optimizer, OptimizationAction


class KanbanSciOpsOptimizer(Optimizer):
    """
    Kanban + AI Science Operations optimizer.

    Key insight: The original AI-SciOps pushes work through the pipeline,
    causing WIP to accumulate at bottlenecks. Kanban uses a PULL system
    with WIP limits — downstream processes pull work only when ready.

    This prevents the WIP explosion seen in Baseline's Experiment (WIP > 60)
    and the Writing backlog in AI-SciOps (backlog > 1.0).

    Combines:
    - WIP limits per process (Kanban)
    - Input throttling when pipeline is congested
    - AI-driven dynamic WIP limit adjustment
    - Bottleneck-aware resource allocation
    """

    def __init__(self):
        super().__init__("Kanban-SciOps (Pull-based Flow)")
        self.wip_limits: dict[str, float] = {}
        self.history: list[dict] = []
        self.initial_wip_limit = 3.0

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        # Initialize WIP limits
        if not self.wip_limits:
            for step in pipeline:
                self.wip_limits[step.config.name] = self.initial_wip_limit

        # Reset throughputs to base (prevent cumulative degradation)
        for step in pipeline:
            step.throughput = step.config.base_throughput

        # Record state
        state = {
            "time_step": time_step,
            "throughputs": {p.config.name: p.effective_throughput() for p in pipeline},
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
        }
        self.history.append(state)

        # --- Kanban WIP enforcement ---
        # If a downstream process is congested, shift resources FROM upstream TO it
        congested = set()
        for step in pipeline:
            if step.work_in_progress > self.wip_limits[step.config.name]:
                congested.add(step.config.name)

        # --- AI-driven resource allocation ---
        # Allocate more to congested processes (pull-based)
        weights = {}
        for step in pipeline:
            wip_ratio = (step.work_in_progress + 0.1) / (step.effective_throughput() + 0.1)
            base_weight = max(0.5, wip_ratio)
            if step.config.name in congested:
                base_weight *= 1.5  # Pull: boost congested processes
            weights[step.config.name] = base_weight

        total_weight = sum(weights.values())
        for step in pipeline:
            step.allocated_resources = (
                weights[step.config.name] / total_weight
            ) * total_resources

        # --- Progressive AI assistance ---
        progress = time_step / 100.0
        for step in pipeline:
            step.ai_assistance_level = min(
                0.85, step.config.ai_automatable * (0.3 + progress * 0.65)
            )

        # --- Dynamic WIP limit adjustment ---
        if time_step > 10 and time_step % 10 == 0:
            recent = self.history[-10:]
            for step in pipeline:
                name = step.config.name
                avg_wip = sum(h["wip"][name] for h in recent) / len(recent)
                avg_tp = sum(h["throughputs"][name] for h in recent) / len(recent)

                # Tighten WIP limit if flow is smooth, loosen if starved
                if avg_wip < avg_tp * 0.5:
                    self.wip_limits[name] = max(1.0, self.wip_limits[name] * 0.9)
                elif avg_wip > self.wip_limits[name] * 0.8:
                    self.wip_limits[name] *= 1.1

                self.actions.append(
                    OptimizationAction(
                        time_step=time_step,
                        target_process=name,
                        action_type="Kanban-AdjustWIP",
                        description=f"WIP limit: {self.wip_limits[name]:.1f}, "
                        f"avg_wip: {avg_wip:.1f}, avg_tp: {avg_tp:.1f}",
                    )
                )

        # --- Process pruning at mid-point ---
        if time_step == 40:
            for step in pipeline:
                if step.ai_assistance_level > 0.5:
                    step.config.uncertainty *= 0.7
                    self.actions.append(
                        OptimizationAction(
                            time_step=time_step,
                            target_process=step.config.name,
                            action_type="Kanban-Prune",
                            description=f"Reduced uncertainty to {step.config.uncertainty:.2f}",
                        )
                    )

        # --- Meta-optimization at 60% ---
        if time_step == 60:
            for step in pipeline:
                if step.config.name == "Review":
                    step.config.human_review_needed *= 0.6
                elif step.config.name == "Writing":
                    step.config.human_review_needed *= 0.7
                elif step.config.name in ("Survey", "Hypothesis"):
                    step.config.base_throughput *= 1.2
                    step.throughput = step.config.base_throughput

            self.actions.append(
                OptimizationAction(
                    time_step=time_step,
                    target_process="system",
                    action_type="Kanban-Meta",
                    description="Meta-optimization: reduced review needs, parallelized upstream",
                )
            )

        return pipeline


class AdaptiveSciOpsOptimizer(Optimizer):
    """
    Adaptive AI Science Operations optimizer.

    Key insight: The original AI-SciOps uses fixed time boundaries for
    stage transitions (0/20/50/80). This is wasteful — if the system
    learns quickly, it should advance faster. If learning is slow, it
    should stay longer.

    Uses performance metrics to decide when to transition:
    - Stage 1→2: When throughput variance stabilizes
    - Stage 2→3: When resource allocation converges
    - Stage 3→4: When pruning effects are realized

    Also introduces:
    - Aggressive early AI deployment
    - Multi-bottleneck resolution (address top-2 bottlenecks simultaneously)
    - Writing-specific optimization (biggest hidden bottleneck)
    """

    def __init__(self):
        super().__init__("Adaptive-SciOps (Metric-driven)")
        self.history: list[dict] = []
        self.current_stage = 1
        self.learned_allocations: dict[str, float] = {}
        self.exploration_rate = 0.2
        self.process_pruned: set[str] = set()
        self.meta_done = False
        self.stage_transition_steps: list[int] = [0]

    def _should_advance_stage(self, time_step: int) -> bool:
        """Decide whether to advance to the next stage based on metrics."""
        if len(self.history) < 8:
            return False

        recent = self.history[-8:]

        if self.current_stage == 1:
            # Advance when throughput variance is low (system understood)
            throughputs = [
                min(h["throughputs"].values()) for h in recent
            ]
            variance = sum((t - sum(throughputs) / len(throughputs)) ** 2 for t in throughputs) / len(throughputs)
            return variance < 0.05 or time_step >= 15

        elif self.current_stage == 2:
            # Advance when allocation weights stabilize
            if len(self.history) < 12:
                return False
            old = self.history[-12]
            new = self.history[-1]
            delta = sum(
                abs(old["throughputs"][k] - new["throughputs"][k])
                for k in old["throughputs"]
            )
            return delta < 0.3 or time_step >= 40

        elif self.current_stage == 3:
            # Advance when pruning effects are realized (throughput jump)
            if len(self.stage_transition_steps) < 3:
                return False
            since_stage3 = time_step - self.stage_transition_steps[-1]
            return since_stage3 >= 15 or time_step >= 65

        return False

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        # Record state
        state = {
            "time_step": time_step,
            "throughputs": {p.config.name: p.effective_throughput() for p in pipeline},
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
        }
        self.history.append(state)

        # Check for stage advancement
        if self.current_stage < 4 and self._should_advance_stage(time_step):
            self.current_stage += 1
            self.stage_transition_steps.append(time_step)
            self.actions.append(
                OptimizationAction(
                    time_step=time_step,
                    target_process="system",
                    action_type=f"Stage{self.current_stage}-Transition",
                    description=f"Adaptive transition to Stage {self.current_stage} "
                    f"at step {time_step}",
                )
            )

        # Execute current stage
        if self.current_stage == 1:
            self._stage1(pipeline, time_step, total_resources)
        elif self.current_stage == 2:
            self._stage2(pipeline, time_step, total_resources)
        elif self.current_stage == 3:
            self._stage3(pipeline, time_step, total_resources)
        else:
            self._stage4(pipeline, time_step, total_resources)

        self.exploration_rate = max(0.03, self.exploration_rate * 0.99)
        return pipeline

    def _stage1(self, pipeline: list[ProcessStep], time_step: int, total_resources: float):
        """Aggressive early deployment — faster than original AI-SciOps."""
        # More aggressive AI from the start
        for step in pipeline:
            step.ai_assistance_level = min(0.5, step.config.ai_automatable * 0.5)

        # Multi-bottleneck: address top 2 bottlenecks
        sorted_steps = sorted(pipeline, key=lambda p: p.effective_throughput())
        bottleneck1 = sorted_steps[0]
        bottleneck2 = sorted_steps[1]

        for step in pipeline:
            if step.config.name == bottleneck1.config.name:
                step.allocated_resources = total_resources * 0.30
            elif step.config.name == bottleneck2.config.name:
                step.allocated_resources = total_resources * 0.22
            else:
                remaining = total_resources * 0.48
                step.allocated_resources = remaining / (len(pipeline) - 2)

    def _stage2(self, pipeline: list[ProcessStep], time_step: int, total_resources: float):
        """Learning-based allocation with Writing-specific optimization."""
        for step in pipeline:
            step.ai_assistance_level = min(0.75, step.config.ai_automatable * 0.75)

        # Learn from WIP patterns
        if len(self.history) >= 5:
            recent = self.history[-5:]
            for step in pipeline:
                name = step.config.name
                avg_wip = sum(h["wip"][name] for h in recent) / len(recent)
                avg_tp = sum(h["throughputs"][name] for h in recent) / len(recent)

                if avg_wip > avg_tp * 1.5:
                    self.learned_allocations[name] = self.learned_allocations.get(name, 1.0) * 1.15
                elif avg_wip < avg_tp * 0.3:
                    self.learned_allocations[name] = self.learned_allocations.get(name, 1.0) * 0.85

        # Special: Writing gets a boost (hidden bottleneck in original)
        self.learned_allocations["Writing"] = self.learned_allocations.get("Writing", 1.0) * 1.02
        self.learned_allocations["Review"] = self.learned_allocations.get("Review", 1.0) * 1.02

        total_weight = sum(self.learned_allocations.get(p.config.name, 1.0) for p in pipeline)
        for step in pipeline:
            weight = self.learned_allocations.get(step.config.name, 1.0)
            if random.random() < self.exploration_rate:
                weight *= random.uniform(0.85, 1.15)
            step.allocated_resources = (weight / total_weight) * total_resources

    def _stage3(self, pipeline: list[ProcessStep], time_step: int, total_resources: float):
        """Aggressive pruning — ALL processes with AI > 0.5 get pruned."""
        for step in pipeline:
            step.ai_assistance_level = min(0.9, step.config.ai_automatable * 0.95)

            # Prune more aggressively: threshold 0.5 instead of 0.7
            if step.ai_assistance_level > 0.5 and step.config.name not in self.process_pruned:
                old = step.config.uncertainty
                step.config.uncertainty *= 0.65
                step.config.failure_rate *= 0.8
                self.process_pruned.add(step.config.name)
                self.actions.append(
                    OptimizationAction(
                        time_step=time_step,
                        target_process=step.config.name,
                        action_type="Adaptive-Prune",
                        description=f"Pruned: uncertainty {old:.2f}→{step.config.uncertainty:.2f}, "
                        f"failure_rate reduced 20%",
                    )
                )

        # Continue learning-based allocation
        self._stage2(pipeline, time_step, total_resources)
        for step in pipeline:
            step.ai_assistance_level = min(0.9, step.config.ai_automatable * 0.95)

    def _stage4(self, pipeline: list[ProcessStep], time_step: int, total_resources: float):
        """Meta-optimization with Writing-specific intervention."""
        if not self.meta_done:
            self.meta_done = True
            for step in pipeline:
                if step.config.name in ("Survey", "Hypothesis"):
                    step.config.base_throughput *= 1.4  # Stronger parallelization
                    step.throughput = step.config.base_throughput
                elif step.config.name == "Review":
                    step.config.human_review_needed *= 0.5
                elif step.config.name == "Writing":
                    # Key fix: Writing bottleneck was ignored in original
                    step.config.human_review_needed *= 0.6
                    step.config.base_throughput *= 1.15
                    step.throughput = step.config.base_throughput
                elif step.config.name == "Experiment":
                    step.config.uncertainty *= 0.8

            self.actions.append(
                OptimizationAction(
                    time_step=time_step,
                    target_process="system",
                    action_type="Adaptive-Meta",
                    description="Meta-optimization: parallelization, Writing fix, Review reduction",
                )
            )

        self._stage3(pipeline, time_step, total_resources)


class HolisticSciOpsOptimizer(Optimizer):
    """
    Holistic AI Science Operations optimizer.

    Combines the best of all approaches into a unified strategy:
    - Kanban WIP limits (prevent accumulation)
    - Adaptive stage transitions (data-driven)
    - Aggressive parallelization from early on
    - Feedback loops (experimental failure improves hypothesis)
    - Continuous pruning (not one-shot)
    - Balanced human-AI allocation across ALL processes

    This represents the paper's ultimate vision: "Science Operations (SciOps)"
    as a fully integrated system that optimizes continuously.
    """

    def __init__(self):
        super().__init__("Holistic-SciOps (Integrated)")
        self.history: list[dict] = []
        self.wip_limits: dict[str, float] = {}
        self.learned_allocations: dict[str, float] = {}
        self.exploration_rate = 0.15
        self.pruning_applied: dict[str, int] = {}  # name -> times pruned
        self.max_prune_rounds = 3

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        # Reset throughputs to base (prevent cumulative degradation)
        for step in pipeline:
            step.throughput = step.config.base_throughput

        state = {
            "time_step": time_step,
            "throughputs": {p.config.name: p.effective_throughput() for p in pipeline},
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
            "backlogs": {p.config.name: p.human_review_backlog for p in pipeline},
        }
        self.history.append(state)

        # Initialize
        if not self.wip_limits:
            for step in pipeline:
                self.wip_limits[step.config.name] = 4.0

        # --- 1. Progressive AI assistance (fast ramp-up) ---
        progress = min(1.0, time_step / 60.0)  # Reach max by step 60
        for step in pipeline:
            target = step.config.ai_automatable * (0.4 + progress * 0.55)
            step.ai_assistance_level = min(0.9, target)

        # --- 2. Kanban flow control (resource-based, not throughput reduction) ---
        congested = set()
        for step in pipeline:
            if step.work_in_progress > self.wip_limits.get(step.config.name, 4.0):
                congested.add(step.config.name)

        # --- 3. Smart resource allocation (continuous learning) ---
        if len(self.history) >= 3:
            recent = self.history[-3:]
            for step in pipeline:
                name = step.config.name
                avg_wip = sum(h["wip"][name] for h in recent) / len(recent)
                avg_tp = sum(h["throughputs"][name] for h in recent) / len(recent)
                avg_backlog = sum(h["backlogs"][name] for h in recent) / len(recent)

                # Factor in both WIP and human review backlog + congestion
                pressure = (avg_wip + avg_backlog * 2) / (avg_tp + 0.1)
                if name in congested:
                    pressure *= 1.3  # Boost congested processes
                self.learned_allocations[name] = max(
                    0.3, self.learned_allocations.get(name, 1.0) * (1.0 + (pressure - 1.0) * 0.1)
                )

        # Apply allocations
        total_weight = sum(self.learned_allocations.get(p.config.name, 1.0) for p in pipeline)
        for step in pipeline:
            weight = self.learned_allocations.get(step.config.name, 1.0)
            if random.random() < self.exploration_rate:
                weight *= random.uniform(0.9, 1.1)
            step.allocated_resources = (weight / total_weight) * total_resources

        self.exploration_rate = max(0.02, self.exploration_rate * 0.995)

        # --- 4. Continuous pruning (every 20 steps) ---
        if time_step > 0 and time_step % 20 == 0:
            for step in pipeline:
                name = step.config.name
                rounds = self.pruning_applied.get(name, 0)
                if rounds < self.max_prune_rounds and step.ai_assistance_level > 0.4:
                    old_unc = step.config.uncertainty
                    old_fail = step.config.failure_rate
                    step.config.uncertainty *= 0.85
                    step.config.failure_rate *= 0.9
                    self.pruning_applied[name] = rounds + 1
                    self.actions.append(
                        OptimizationAction(
                            time_step=time_step,
                            target_process=name,
                            action_type="Holistic-Prune",
                            description=f"Round {rounds + 1}: uncertainty "
                            f"{old_unc:.3f}→{step.config.uncertainty:.3f}, "
                            f"failure {old_fail:.3f}→{step.config.failure_rate:.3f}",
                        )
                    )

        # --- 5. Early parallelization (step 30, not 80) ---
        if time_step == 30:
            for step in pipeline:
                if step.config.name in ("Survey", "Hypothesis"):
                    step.config.base_throughput *= 1.25
                    step.throughput = step.config.base_throughput
            self.actions.append(
                OptimizationAction(
                    time_step=30,
                    target_process="system",
                    action_type="Holistic-Parallelize",
                    description="Early parallelization of Survey + Hypothesis",
                )
            )

        # --- 6. Human review optimization (step 25) ---
        if time_step == 25:
            for step in pipeline:
                if step.config.name == "Review":
                    step.config.human_review_needed *= 0.55
                elif step.config.name == "Writing":
                    step.config.human_review_needed *= 0.65
                elif step.config.name == "Hypothesis":
                    step.config.human_review_needed *= 0.8
            self.actions.append(
                OptimizationAction(
                    time_step=25,
                    target_process="system",
                    action_type="Holistic-ReviewOpt",
                    description="AI meta-review deployed: Review→0.55x, Writing→0.65x, Hypothesis→0.8x",
                )
            )

        # --- 7. Dynamic WIP limit adjustment ---
        if time_step > 0 and time_step % 15 == 0 and len(self.history) >= 10:
            recent = self.history[-10:]
            for step in pipeline:
                name = step.config.name
                avg_wip = sum(h["wip"][name] for h in recent) / len(recent)
                tp = step.effective_throughput()
                # Target: WIP = 2x throughput (Little's Law optimal)
                target_wip = tp * 2.0
                self.wip_limits[name] = max(1.5, target_wip)

        # --- 8. Feedback loop: Experiment failures improve Hypothesis ---
        if time_step > 0 and time_step % 25 == 0:
            exp_step = next((s for s in pipeline if s.config.name == "Experiment"), None)
            hyp_step = next((s for s in pipeline if s.config.name == "Hypothesis"), None)
            if exp_step and hyp_step and exp_step.failed_units > 0:
                # Learning from failure reduces future uncertainty
                improvement = min(0.05, exp_step.failed_units * 0.002)
                hyp_step.config.uncertainty = max(0.05, hyp_step.config.uncertainty - improvement)
                self.actions.append(
                    OptimizationAction(
                        time_step=time_step,
                        target_process="Hypothesis",
                        action_type="Holistic-FeedbackLoop",
                        description=f"Experiment failures fed back: Hypothesis uncertainty "
                        f"→ {hyp_step.config.uncertainty:.3f}",
                    )
                )

        return pipeline
