"""
Process Optimization Strategies
================================
Implements three optimization approaches from the paper:
1. No optimization (Baseline)
2. TOC + PDCA (Rule-based, industrial management)
3. AI-SciOps (AI-driven autonomous optimization)
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from scientific_process import ProcessStep


@dataclass
class OptimizationAction:
    """Record of an optimization action taken."""

    time_step: int
    target_process: str
    action_type: str
    description: str
    parameter_changes: dict = field(default_factory=dict)


class Optimizer(ABC):
    """Base class for process optimizers."""

    def __init__(self, name: str):
        self.name = name
        self.actions: list[OptimizationAction] = []

    @abstractmethod
    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        pass

    def get_bottleneck(self, pipeline: list[ProcessStep]) -> ProcessStep:
        """Identify the bottleneck process (lowest effective throughput)."""
        return min(pipeline, key=lambda p: p.effective_throughput())


class BaselineOptimizer(Optimizer):
    """No optimization. Equal resource distribution."""

    def __init__(self):
        super().__init__("Baseline (No Optimization)")

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        # Equal distribution of resources
        per_process = total_resources / len(pipeline)
        for step in pipeline:
            step.allocated_resources = per_process
        return pipeline


class TOCPDCAOptimizer(Optimizer):
    """
    Theory of Constraints + PDCA cycle optimizer.

    Implements the 5 Focusing Steps of TOC:
    1. IDENTIFY the constraint
    2. EXPLOIT the constraint (maximize its output)
    3. SUBORDINATE everything else to the constraint
    4. ELEVATE the constraint (invest to increase capacity)
    5. REPEAT (go back to step 1 - the PDCA cycle)

    Combined with PDCA:
    - Plan: Identify bottleneck and plan resource reallocation
    - Do: Execute reallocation
    - Check: Measure throughput improvement
    - Act: Standardize if improved, revise if not
    """

    def __init__(self, pdca_cycle_length: int = 10):
        super().__init__("TOC + PDCA")
        self.pdca_cycle_length = pdca_cycle_length
        self.current_phase = "Plan"
        self.cycle_start = 0
        self.previous_throughput = 0.0
        self.planned_bottleneck: str | None = None

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        cycle_position = (time_step - self.cycle_start) % self.pdca_cycle_length

        if cycle_position == 0:
            # PLAN phase: Identify bottleneck
            self.current_phase = "Plan"
            bottleneck = self.get_bottleneck(pipeline)
            self.planned_bottleneck = bottleneck.config.name

            self.actions.append(
                OptimizationAction(
                    time_step=time_step,
                    target_process=bottleneck.config.name,
                    action_type="TOC-Identify",
                    description=f"Identified bottleneck: {bottleneck.config.name} "
                    f"(throughput: {bottleneck.effective_throughput():.2f})",
                )
            )

        elif cycle_position < self.pdca_cycle_length * 0.3:
            # DO phase: Reallocate resources to bottleneck
            self.current_phase = "Do"

        elif cycle_position < self.pdca_cycle_length * 0.7:
            # CHECK/STUDY phase: Monitor
            self.current_phase = "Check"

        else:
            # ACT phase: Standardize or revise
            self.current_phase = "Act"
            current_throughput = min(p.effective_throughput() for p in pipeline)
            if current_throughput > self.previous_throughput:
                self.actions.append(
                    OptimizationAction(
                        time_step=time_step,
                        target_process=self.planned_bottleneck or "system",
                        action_type="PDCA-Standardize",
                        description=f"Improvement confirmed: "
                        f"{self.previous_throughput:.2f} → {current_throughput:.2f}",
                    )
                )
            self.previous_throughput = current_throughput

        # TOC resource allocation: prioritize the bottleneck
        bottleneck = self.get_bottleneck(pipeline)
        for step in pipeline:
            if step.config.name == bottleneck.config.name:
                # EXPLOIT + ELEVATE: give more resources to bottleneck
                step.allocated_resources = total_resources * 0.35
            else:
                # SUBORDINATE: distribute remaining resources
                remaining = total_resources * 0.65
                step.allocated_resources = remaining / (len(pipeline) - 1)

        return pipeline


class AISciOpsOptimizer(Optimizer):
    """
    AI-driven Science Operations optimizer.

    Implements the paper's vision of AI autonomously optimizing
    the entire scientific process. This simulates an AI agent that:

    1. Monitors all process metrics continuously
    2. Identifies bottlenecks (TOC-like)
    3. Dynamically reallocates resources
    4. Adjusts AI assistance levels per process
    5. Can restructure processes (pruning/adding) - Stage 3 from paper
    6. Optimizes the meta-process itself - Stage 4 from paper

    Represents "Science Operations (SciOps)" concept from the paper.
    """

    def __init__(self):
        super().__init__("AI-SciOps (Autonomous Optimization)")
        self.history: list[dict] = []
        self.exploration_rate = 0.3  # Start with exploration
        self.learned_allocations: dict[str, float] = {}
        self.process_pruned: set[str] = set()
        self.meta_optimization_triggered = False

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        # Record current state
        state = {
            "time_step": time_step,
            "throughputs": {
                p.config.name: p.effective_throughput() for p in pipeline
            },
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
            "backlogs": {
                p.config.name: p.human_review_backlog for p in pipeline
            },
        }
        self.history.append(state)

        # Stage 1: AI-assisted process with human feedback (early phase)
        if time_step < 20:
            self._stage1_optimize(pipeline, time_step, total_resources)

        # Stage 2: Autonomous process optimization
        elif time_step < 50:
            self._stage2_optimize(pipeline, time_step, total_resources)

        # Stage 3: Process restructuring (pruning/addition)
        elif time_step < 80:
            self._stage3_optimize(pipeline, time_step, total_resources)

        # Stage 4: Meta-process optimization
        else:
            self._stage4_optimize(pipeline, time_step, total_resources)

        # Decay exploration rate over time (learning)
        self.exploration_rate = max(0.05, self.exploration_rate * 0.995)

        return pipeline

    def _stage1_optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ):
        """Stage 1: Process replacement with human feedback.

        AI assists but humans provide feedback on optimization direction.
        Moderate AI assistance, conservative resource allocation.
        """
        bottleneck = self.get_bottleneck(pipeline)

        for step in pipeline:
            # Gradually introduce AI assistance
            step.ai_assistance_level = min(
                0.3, step.config.ai_automatable * 0.3
            )

            if step.config.name == bottleneck.config.name:
                step.allocated_resources = total_resources * 0.3
            else:
                remaining = total_resources * 0.7
                step.allocated_resources = remaining / (len(pipeline) - 1)

        if time_step == 0:
            self.actions.append(
                OptimizationAction(
                    time_step=time_step,
                    target_process="system",
                    action_type="Stage1-Init",
                    description="Stage 1: AI-assisted optimization with human feedback",
                )
            )

    def _stage2_optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ):
        """Stage 2: Autonomous process optimization.

        AI takes over optimization decisions. Higher AI assistance levels.
        Dynamic resource allocation based on learned patterns.
        """
        if time_step == 20:
            self.actions.append(
                OptimizationAction(
                    time_step=time_step,
                    target_process="system",
                    action_type="Stage2-Init",
                    description="Stage 2: Transitioning to autonomous optimization",
                )
            )

        # Analyze historical data to find patterns
        if len(self.history) >= 5:
            recent = self.history[-5:]
            for step in pipeline:
                name = step.config.name
                avg_throughput = sum(
                    h["throughputs"][name] for h in recent
                ) / len(recent)
                avg_wip = sum(h["wip"][name] for h in recent) / len(recent)

                # AI learns optimal allocation
                if avg_wip > avg_throughput * 2:
                    # Work piling up: this needs more resources
                    self.learned_allocations[name] = (
                        self.learned_allocations.get(name, 1.0) * 1.1
                    )
                elif avg_wip < avg_throughput * 0.5:
                    # Underutilized: can reduce resources
                    self.learned_allocations[name] = (
                        self.learned_allocations.get(name, 1.0) * 0.9
                    )

        # Apply learned allocations with exploration
        total_weight = sum(self.learned_allocations.get(p.config.name, 1.0) for p in pipeline)
        for step in pipeline:
            name = step.config.name
            weight = self.learned_allocations.get(name, 1.0)

            if random.random() < self.exploration_rate:
                # Explore: random perturbation
                weight *= random.uniform(0.8, 1.2)

            step.allocated_resources = (weight / total_weight) * total_resources
            # Increase AI assistance
            step.ai_assistance_level = min(
                0.7, step.config.ai_automatable * 0.7
            )

    def _stage3_optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ):
        """Stage 3: Process pruning and restructuring.

        AI identifies redundant processes and restructures the pipeline.
        Near-maximum AI assistance for automatable tasks.
        """
        if time_step == 50:
            self.actions.append(
                OptimizationAction(
                    time_step=time_step,
                    target_process="system",
                    action_type="Stage3-Init",
                    description="Stage 3: Process restructuring - pruning inefficiencies",
                )
            )

        # Identify processes that can be streamlined
        for step in pipeline:
            # Maximize AI assistance
            step.ai_assistance_level = min(
                0.9, step.config.ai_automatable * 0.95
            )

            # Simulate process pruning: reduce unnecessary uncertainty
            # for highly AI-automated processes
            if step.ai_assistance_level > 0.7 and step.config.name not in self.process_pruned:
                original_uncertainty = step.config.uncertainty
                step.config.uncertainty *= 0.7  # AI reduces uncertainty
                self.process_pruned.add(step.config.name)
                self.actions.append(
                    OptimizationAction(
                        time_step=time_step,
                        target_process=step.config.name,
                        action_type="Stage3-Prune",
                        description=f"Reduced uncertainty: {original_uncertainty:.2f} → "
                        f"{step.config.uncertainty:.2f}",
                        parameter_changes={
                            "uncertainty": step.config.uncertainty
                        },
                    )
                )

        # Smart resource allocation based on accumulated learning
        self._stage2_optimize(pipeline, time_step, total_resources)
        # Override AI levels back to stage 3 levels
        for step in pipeline:
            step.ai_assistance_level = min(
                0.9, step.config.ai_automatable * 0.95
            )

    def _stage4_optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ):
        """Stage 4: Meta-process reorganization.

        AI optimizes the higher-level process structure itself.
        This represents the paper's vision of "meta-process reconstruction."
        """
        if not self.meta_optimization_triggered:
            self.meta_optimization_triggered = True
            self.actions.append(
                OptimizationAction(
                    time_step=time_step,
                    target_process="system",
                    action_type="Stage4-MetaOptimize",
                    description="Stage 4: Meta-process reorganization initiated. "
                    "AI is restructuring the scientific workflow itself.",
                )
            )

            # Meta-optimization: boost throughput of the entire pipeline
            # by recognizing that some processes can be parallelized
            # or merged (e.g., Survey + Hypothesis can overlap)
            for step in pipeline:
                if step.config.name in ("Survey", "Hypothesis"):
                    step.throughput *= 1.3  # Parallel execution boost
                    self.actions.append(
                        OptimizationAction(
                            time_step=time_step,
                            target_process=step.config.name,
                            action_type="Stage4-Parallelize",
                            description=f"Parallelized {step.config.name} with adjacent processes",
                        )
                    )
                elif step.config.name == "Review":
                    # AI-assisted meta-review reduces human bottleneck
                    step.config.human_review_needed *= 0.6
                    self.actions.append(
                        OptimizationAction(
                            time_step=time_step,
                            target_process="Review",
                            action_type="Stage4-MetaReview",
                            description="Introduced AI meta-review to reduce human review bottleneck",
                        )
                    )

        # Continue with optimized allocation
        self._stage3_optimize(pipeline, time_step, total_resources)
