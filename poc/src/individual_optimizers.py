"""
Individual Methodology Optimizers
==================================
Separates TOC and PDCA into standalone strategies for fair comparison.
Each methodology is implemented in its pure form without mixing.

1. TOC-only: Theory of Constraints (5 Focusing Steps)
2. PDCA-only: Plan-Do-Check-Act cycle
3. Kanban-only: Pull-based WIP-limited flow (no AI enhancements)
4. Agile-only: Sprint-based iterative management (no AI enhancements)
"""

import random

from scientific_process import ProcessStep
from optimizers import Optimizer, OptimizationAction


class TOCOnlyOptimizer(Optimizer):
    """
    Pure Theory of Constraints optimizer.

    Implements Goldratt's 5 Focusing Steps without PDCA overlay:
    1. IDENTIFY the system's constraint (bottleneck)
    2. EXPLOIT the constraint (maximize its output)
    3. SUBORDINATE everything else to the constraint
    4. ELEVATE the constraint (invest to increase capacity)
    5. REPEAT — if the constraint shifts, go back to step 1

    No AI assistance. No cycle-based improvement. Pure TOC resource reallocation.
    """

    def __init__(self):
        super().__init__("TOC (Theory of Constraints)")
        self.previous_bottleneck: str | None = None
        self.bottleneck_tenure: int = 0
        self.elevate_bonus: dict[str, float] = {}

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        # No AI assistance — pure management technique
        for step in pipeline:
            step.ai_assistance_level = 0.0

        # Step 1: IDENTIFY the constraint
        bottleneck = self.get_bottleneck(pipeline)
        bn_name = bottleneck.config.name

        if bn_name != self.previous_bottleneck:
            if self.previous_bottleneck is not None:
                self.actions.append(
                    OptimizationAction(
                        time_step=time_step,
                        target_process=bn_name,
                        action_type="TOC-Identify",
                        description=f"Constraint shifted: {self.previous_bottleneck} → {bn_name} "
                        f"(throughput: {bottleneck.effective_throughput():.2f})",
                    )
                )
            self.previous_bottleneck = bn_name
            self.bottleneck_tenure = 0
        else:
            self.bottleneck_tenure += 1

        # Step 2: EXPLOIT — maximize bottleneck output
        # Give the bottleneck the largest share of resources
        bottleneck_share = 0.35

        # Step 3: SUBORDINATE — other processes serve the constraint
        # Distribute remaining resources equally among non-bottleneck processes
        remaining_share = 1.0 - bottleneck_share
        for step in pipeline:
            if step.config.name == bn_name:
                step.allocated_resources = total_resources * bottleneck_share
            else:
                step.allocated_resources = (
                    total_resources * remaining_share / (len(pipeline) - 1)
                )

        # Step 4: ELEVATE — if bottleneck persists, invest more
        # After 15 consecutive steps as bottleneck, increase its share further
        if self.bottleneck_tenure > 15:
            elevation = min(0.10, (self.bottleneck_tenure - 15) * 0.005)
            self.elevate_bonus[bn_name] = elevation
            for step in pipeline:
                if step.config.name == bn_name:
                    step.allocated_resources = total_resources * (bottleneck_share + elevation)
                else:
                    step.allocated_resources = (
                        total_resources * (remaining_share - elevation) / (len(pipeline) - 1)
                    )

            if self.bottleneck_tenure == 16:
                self.actions.append(
                    OptimizationAction(
                        time_step=time_step,
                        target_process=bn_name,
                        action_type="TOC-Elevate",
                        description=f"Elevating constraint {bn_name}: "
                        f"additional {elevation:.1%} resources",
                    )
                )

        # Step 5: REPEAT — handled by re-identifying each step

        return pipeline


class PDCAOnlyOptimizer(Optimizer):
    """
    Pure PDCA (Plan-Do-Check-Act) cycle optimizer.

    Implements Deming's continuous improvement cycle without TOC's
    constraint-focused resource allocation:

    - Plan: Analyze metrics and plan improvements
    - Do: Execute the plan with balanced resource allocation
    - Check: Measure results against planned targets
    - Act: Standardize improvements or revise the plan

    No AI assistance. No bottleneck-focused allocation. Pure PDCA cycles.
    """

    def __init__(self, cycle_length: int = 10):
        super().__init__("PDCA (Plan-Do-Check-Act)")
        self.cycle_length = cycle_length
        self.cycle_number = 0
        self.phase = "Plan"
        self.plan_targets: dict[str, float] = {}
        self.previous_metrics: dict[str, float] = {}
        self.improvement_applied: dict[str, int] = {}

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        # No AI assistance — pure management technique
        for step in pipeline:
            step.ai_assistance_level = 0.0

        cycle_pos = time_step % self.cycle_length

        # --- PLAN: analyze and set targets ---
        if cycle_pos == 0:
            self.phase = "Plan"
            self.cycle_number += 1

            # Analyze current metrics
            for step in pipeline:
                self.previous_metrics[step.config.name] = step.effective_throughput()

            # Set improvement targets: 10% improvement on weakest processes
            sorted_procs = sorted(pipeline, key=lambda p: p.effective_throughput())
            self.plan_targets = {}
            for step in sorted_procs[:3]:  # Focus on bottom 3
                self.plan_targets[step.config.name] = (
                    step.effective_throughput() * 1.10
                )

            self.actions.append(
                OptimizationAction(
                    time_step=time_step,
                    target_process="system",
                    action_type="PDCA-Plan",
                    description=f"Cycle {self.cycle_number}: targets set for "
                    f"{list(self.plan_targets.keys())}",
                )
            )

        # --- DO: execute with slightly weighted allocation ---
        elif cycle_pos < int(self.cycle_length * 0.4):
            self.phase = "Do"

        # --- CHECK: measure results ---
        elif cycle_pos < int(self.cycle_length * 0.7):
            self.phase = "Check"

        # --- ACT: standardize or revise ---
        else:
            self.phase = "Act"
            if cycle_pos == int(self.cycle_length * 0.7):
                improved = []
                not_improved = []
                for name, target in self.plan_targets.items():
                    step = next(s for s in pipeline if s.config.name == name)
                    current = step.effective_throughput()
                    prev = self.previous_metrics.get(name, current)
                    if current >= prev:
                        improved.append(name)
                    else:
                        not_improved.append(name)

                if improved:
                    self.actions.append(
                        OptimizationAction(
                            time_step=time_step,
                            target_process="system",
                            action_type="PDCA-Act",
                            description=f"Standardize improvements: {improved}; "
                            f"revise: {not_improved}",
                        )
                    )

        # Resource allocation: slightly favor target processes during Do/Check
        if self.plan_targets and self.phase in ("Do", "Check"):
            target_names = set(self.plan_targets.keys())
            n_target = len(target_names)
            n_other = len(pipeline) - n_target
            # Give 60% of resources to target processes, 40% to others
            target_share = 0.60 / max(1, n_target)
            other_share = 0.40 / max(1, n_other)
            for step in pipeline:
                if step.config.name in target_names:
                    step.allocated_resources = total_resources * target_share
                else:
                    step.allocated_resources = total_resources * other_share
        else:
            # Equal distribution during Plan and Act phases
            per_process = total_resources / len(pipeline)
            for step in pipeline:
                step.allocated_resources = per_process

        # PDCA continuous improvement: every 3 cycles, reduce uncertainty
        # for processes that consistently improve (standardized gains)
        if self.cycle_number > 0 and self.cycle_number % 3 == 0 and cycle_pos == 0:
            for step in pipeline:
                rounds = self.improvement_applied.get(step.config.name, 0)
                if rounds < 3:
                    step.config.uncertainty *= 0.95
                    self.improvement_applied[step.config.name] = rounds + 1

        return pipeline


class KanbanOnlyOptimizer(Optimizer):
    """
    Pure Kanban optimizer (no AI, no SciOps enhancements).

    Implements core Kanban principles:
    1. Visualize work flow
    2. Limit Work In Progress (WIP)
    3. Manage flow (pull-based)
    4. Make policies explicit
    5. Improve collaboratively

    No AI assistance. No process pruning. No meta-optimization.
    Pure pull-based flow control with WIP limits.
    """

    def __init__(self, wip_limit: float = 3.0):
        super().__init__("Kanban (Pull-based Flow)")
        self.wip_limits: dict[str, float] = {}
        self.initial_wip_limit = wip_limit
        self.history: list[dict] = []

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        # No AI assistance — pure Kanban
        for step in pipeline:
            step.ai_assistance_level = 0.0
            step.throughput = step.config.base_throughput

        # Initialize WIP limits
        if not self.wip_limits:
            for step in pipeline:
                self.wip_limits[step.config.name] = self.initial_wip_limit

        # Record state
        state = {
            "time_step": time_step,
            "throughputs": {p.config.name: p.effective_throughput() for p in pipeline},
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
        }
        self.history.append(state)

        # --- WIP enforcement: detect congested processes ---
        congested = set()
        for step in pipeline:
            if step.work_in_progress > self.wip_limits[step.config.name]:
                congested.add(step.config.name)

        # --- Pull-based resource allocation ---
        # Congested (over WIP limit) processes get boosted
        # Starved processes get less
        weights = {}
        for step in pipeline:
            wip_ratio = (step.work_in_progress + 0.1) / (step.effective_throughput() + 0.1)
            base_weight = max(0.5, wip_ratio)
            if step.config.name in congested:
                base_weight *= 1.5  # Pull: boost congested
            weights[step.config.name] = base_weight

        total_weight = sum(weights.values())
        for step in pipeline:
            step.allocated_resources = (
                weights[step.config.name] / total_weight
            ) * total_resources

        # --- Dynamic WIP limit adjustment (every 15 steps) ---
        if time_step > 10 and time_step % 15 == 0 and len(self.history) >= 10:
            recent = self.history[-10:]
            for step in pipeline:
                name = step.config.name
                avg_wip = sum(h["wip"][name] for h in recent) / len(recent)
                avg_tp = sum(h["throughputs"][name] for h in recent) / len(recent)

                if avg_wip < avg_tp * 0.5:
                    self.wip_limits[name] = max(1.0, self.wip_limits[name] * 0.9)
                elif avg_wip > self.wip_limits[name] * 0.8:
                    self.wip_limits[name] *= 1.1

                self.actions.append(
                    OptimizationAction(
                        time_step=time_step,
                        target_process=name,
                        action_type="Kanban-WIP",
                        description=f"WIP limit: {self.wip_limits[name]:.1f}",
                    )
                )

        return pipeline


class AgileOnlyOptimizer(Optimizer):
    """
    Pure Agile/Scrum optimizer (no AI, no SciOps enhancements).

    Implements core Agile principles:
    1. Sprint-based iterative work
    2. Sprint planning with backlog prioritization
    3. Sprint review and retrospective
    4. Responding to change mid-sprint
    5. Velocity tracking

    No AI assistance. No process pruning. No meta-optimization.
    Pure Agile management framework.
    """

    def __init__(self, sprint_length: int = 8):
        super().__init__("Agile (Sprint-based)")
        self.sprint_length = sprint_length
        self.sprint_number = 0
        self.sprint_velocity: list[float] = []
        self.sprint_start_output = 0.0
        self.focus_processes: list[str] = []

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        # No AI assistance — pure Agile
        for step in pipeline:
            step.ai_assistance_level = 0.0

        sprint_position = time_step % self.sprint_length

        # --- Sprint Planning ---
        if sprint_position == 0:
            self.sprint_number += 1

            # Retrospective of previous sprint
            if self.sprint_velocity:
                last_vel = self.sprint_velocity[-1]
                avg_vel = sum(self.sprint_velocity) / len(self.sprint_velocity)
                if last_vel < avg_vel * 0.9:
                    self.actions.append(
                        OptimizationAction(
                            time_step=time_step,
                            target_process="system",
                            action_type="Agile-Retro",
                            description=f"Sprint {self.sprint_number - 1} velocity {last_vel:.2f} "
                            f"below avg {avg_vel:.2f}",
                        )
                    )

            # Sprint planning: prioritize 2 weakest processes
            sorted_procs = sorted(pipeline, key=lambda p: p.effective_throughput())
            self.focus_processes = [sorted_procs[0].config.name, sorted_procs[1].config.name]
            self.sprint_start_output = sum(p.completed_units for p in pipeline)

            self.actions.append(
                OptimizationAction(
                    time_step=time_step,
                    target_process="system",
                    action_type="Agile-Plan",
                    description=f"Sprint {self.sprint_number}: focus on {self.focus_processes}",
                )
            )

        # --- Sprint Review ---
        if sprint_position == self.sprint_length - 1:
            current_output = sum(p.completed_units for p in pipeline)
            velocity = current_output - self.sprint_start_output
            self.sprint_velocity.append(velocity)

        # --- Sprint Execution: focused resource allocation ---
        for step in pipeline:
            if step.config.name in self.focus_processes:
                step.allocated_resources = total_resources * 0.25
            else:
                remaining = total_resources * 0.50
                non_focus = len(pipeline) - len(self.focus_processes)
                step.allocated_resources = remaining / max(1, non_focus)

        # --- Mid-sprint adjustment (respond to change) ---
        if sprint_position == self.sprint_length // 2:
            current_bottleneck = min(pipeline, key=lambda p: p.effective_throughput())
            if current_bottleneck.config.name not in self.focus_processes:
                self.focus_processes[1] = current_bottleneck.config.name

        # --- Process improvement via retrospectives (every 5 sprints) ---
        if time_step > 0 and self.sprint_number % 5 == 0 and sprint_position == 0:
            for step in pipeline:
                if step.config.name in self.focus_processes:
                    step.config.uncertainty *= 0.95

        return pipeline
