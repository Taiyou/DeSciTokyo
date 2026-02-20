"""
Process Management Optimizers
==============================
Additional management methodologies from the paper:
1. Agile-Scrum: Sprint-based iterative development
2. Lean-SciOps: Waste elimination (muda) + continuous flow
3. SixSigma-SciOps: Statistical quality control via DMAIC
"""

import random
from dataclasses import dataclass

from scientific_process import ProcessStep
from optimizers import Optimizer, OptimizationAction


class AgileScrumOptimizer(Optimizer):
    """
    Agile/Scrum for Science (ScrumAdemia).

    Based on Franco et al. (2023) and Hidalgo (2019) from the paper.
    The paper's Section 2 discusses agile's 4 principles:
    1. Individuals and interactions over processes and tools
    2. Working products over comprehensive documentation
    3. Customer collaboration over contract negotiation
    4. Responding to change over following a plan

    Implementation:
    - Fixed-length sprints (sprint_length time steps)
    - Sprint planning: prioritize processes with highest WIP
    - Sprint execution: focused resource allocation
    - Sprint review: evaluate output and identify issues
    - Sprint retrospective: adjust allocation strategy
    - Backlog: ordered list of process improvement items
    """

    def __init__(self, sprint_length: int = 8):
        super().__init__("Agile-Scrum")
        self.sprint_length = sprint_length
        self.sprint_number = 0
        self.sprint_velocity: list[float] = []  # output per sprint
        self.sprint_start_output = 0.0
        self.current_sprint_output = 0.0
        self.focus_processes: list[str] = []
        self.history: list[dict] = []
        self.ai_level = 0.0

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        state = {
            "time_step": time_step,
            "throughputs": {p.config.name: p.effective_throughput() for p in pipeline},
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
        }
        self.history.append(state)

        sprint_position = time_step % self.sprint_length

        # --- Sprint Planning (start of sprint) ---
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
                            action_type="Scrum-Retro",
                            description=f"Sprint {self.sprint_number - 1} velocity {last_vel:.2f} "
                            f"below avg {avg_vel:.2f}, adjusting focus",
                        )
                    )

            # Sprint planning: identify top 2 bottleneck processes
            sorted_procs = sorted(pipeline, key=lambda p: p.effective_throughput())
            self.focus_processes = [sorted_procs[0].config.name, sorted_procs[1].config.name]

            self.sprint_start_output = sum(p.completed_units for p in pipeline)

            self.actions.append(
                OptimizationAction(
                    time_step=time_step,
                    target_process="system",
                    action_type="Scrum-Plan",
                    description=f"Sprint {self.sprint_number}: focus on {self.focus_processes}",
                )
            )

        # --- Sprint Review (end of sprint) ---
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

        # --- Progressive AI (Agile adapts tools gradually) ---
        self.ai_level = min(0.7, time_step / 120.0)
        for step in pipeline:
            step.ai_assistance_level = min(
                self.ai_level, step.config.ai_automatable * self.ai_level
            )

        # --- Responding to change: mid-sprint adjustment ---
        if sprint_position == self.sprint_length // 2:
            # Check if focus is still correct
            current_bottleneck = min(pipeline, key=lambda p: p.effective_throughput())
            if current_bottleneck.config.name not in self.focus_processes:
                self.focus_processes[1] = current_bottleneck.config.name

        # --- Process improvement via retrospectives ---
        if time_step > 0 and time_step % (self.sprint_length * 4) == 0:
            # Every 4 sprints, reduce uncertainty in focus processes
            for step in pipeline:
                if step.config.name in self.focus_processes and step.ai_assistance_level > 0.3:
                    step.config.uncertainty *= 0.9
                    self.actions.append(
                        OptimizationAction(
                            time_step=time_step,
                            target_process=step.config.name,
                            action_type="Scrum-Kaizen",
                            description=f"Retrospective improvement: uncertainty → "
                            f"{step.config.uncertainty:.3f}",
                        )
                    )

        return pipeline


class LeanSciOpsOptimizer(Optimizer):
    """
    Lean Management for Science.

    Based on the paper's mention of Wraae et al. (2024) scoping review
    of Lean management in research. Lean focuses on:

    1. Value Stream Mapping: identify value-adding vs waste activities
    2. Eliminate 7 wastes (muda): overproduction, waiting, transport,
       overprocessing, inventory (WIP), motion, defects
    3. Just-In-Time: produce only what's needed, when needed
    4. Continuous flow: minimize batch sizes
    5. Kaizen: continuous improvement

    In scientific context:
    - Waste = rework, failed experiments, WIP accumulation, unnecessary review
    - Value = completed research output
    - Flow = smooth progression through pipeline stages
    """

    def __init__(self):
        super().__init__("Lean-SciOps")
        self.history: list[dict] = []
        self.waste_metrics: dict[str, float] = {}
        self.kaizen_applied: dict[str, int] = {}
        self.value_stream_mapped = False

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        state = {
            "time_step": time_step,
            "throughputs": {p.config.name: p.effective_throughput() for p in pipeline},
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
            "rework": {p.config.name: p.rework_units for p in pipeline},
            "failures": {p.config.name: p.failed_units for p in pipeline},
        }
        self.history.append(state)

        # --- 1. Value Stream Mapping (one-time, step 5) ---
        if time_step == 5 and not self.value_stream_mapped:
            self.value_stream_mapped = True
            # Identify waste sources
            for step in pipeline:
                waste = step.config.uncertainty * 0.3 + step.config.failure_rate * 0.1
                self.waste_metrics[step.config.name] = waste
            self.actions.append(
                OptimizationAction(
                    time_step=time_step,
                    target_process="system",
                    action_type="Lean-VSM",
                    description=f"Value Stream Map: waste sources identified: "
                    f"{', '.join(f'{k}={v:.2f}' for k, v in sorted(self.waste_metrics.items(), key=lambda x: -x[1]))}",
                )
            )

        # --- 2. Progressive AI (lean uses AI to eliminate waste) ---
        progress = min(1.0, time_step / 80.0)
        for step in pipeline:
            step.ai_assistance_level = min(
                0.85, step.config.ai_automatable * (0.3 + progress * 0.6)
            )

        # --- 3. Eliminate waste: focus resources on high-waste processes ---
        if self.waste_metrics:
            # Weight allocation by waste (more waste = more resources to fix)
            weights = {}
            for step in pipeline:
                name = step.config.name
                waste = self.waste_metrics.get(name, 0.1)
                wip_waste = step.work_in_progress / (step.effective_throughput() + 0.1)
                weights[name] = max(0.3, 1.0 + waste * 2 + wip_waste * 0.5)

            total_weight = sum(weights.values())
            for step in pipeline:
                step.allocated_resources = (
                    weights[step.config.name] / total_weight
                ) * total_resources
        else:
            per_proc = total_resources / len(pipeline)
            for step in pipeline:
                step.allocated_resources = per_proc

        # --- 4. Kaizen: continuous waste reduction (every 15 steps) ---
        if time_step > 0 and time_step % 15 == 0:
            # Target the process with highest current waste
            worst = max(pipeline, key=lambda p: p.config.uncertainty + p.config.failure_rate)
            rounds = self.kaizen_applied.get(worst.config.name, 0)
            if rounds < 4 and worst.ai_assistance_level > 0.3:
                old_unc = worst.config.uncertainty
                old_fail = worst.config.failure_rate
                worst.config.uncertainty *= 0.88
                worst.config.failure_rate *= 0.9
                self.kaizen_applied[worst.config.name] = rounds + 1

                # Update waste metrics
                self.waste_metrics[worst.config.name] = (
                    worst.config.uncertainty * 0.3 + worst.config.failure_rate * 0.1
                )

                self.actions.append(
                    OptimizationAction(
                        time_step=time_step,
                        target_process=worst.config.name,
                        action_type="Lean-Kaizen",
                        description=f"Kaizen #{rounds + 1}: uncertainty {old_unc:.3f}→"
                        f"{worst.config.uncertainty:.3f}, failure {old_fail:.3f}→"
                        f"{worst.config.failure_rate:.3f}",
                    )
                )

        # --- 5. Just-In-Time: reduce human review for AI-capable processes ---
        if time_step == 35:
            for step in pipeline:
                if step.ai_assistance_level > 0.4:
                    old = step.config.human_review_needed
                    step.config.human_review_needed *= 0.75
                    self.actions.append(
                        OptimizationAction(
                            time_step=time_step,
                            target_process=step.config.name,
                            action_type="Lean-JIT",
                            description=f"JIT review reduction: {old:.2f}→"
                            f"{step.config.human_review_needed:.2f}",
                        )
                    )

        return pipeline


class SixSigmaSciOpsOptimizer(Optimizer):
    """
    Six Sigma DMAIC for Science.

    Uses the Define-Measure-Analyze-Improve-Control cycle to
    systematically reduce variation and defects in research processes.

    - Define: Set quality targets for each process
    - Measure: Collect process performance data
    - Analyze: Identify root causes of defects/variation
    - Improve: Implement targeted fixes
    - Control: Monitor and maintain improvements

    Key difference from other methods: focuses on QUALITY (reducing
    rework and failures) rather than just throughput/speed.
    """

    def __init__(self, dmaic_cycle_length: int = 20):
        super().__init__("SixSigma-SciOps")
        self.dmaic_cycle_length = dmaic_cycle_length
        self.cycle_number = 0
        self.phase = "Define"
        self.history: list[dict] = []
        self.quality_targets: dict[str, float] = {}
        self.root_causes: dict[str, str] = {}
        self.control_limits: dict[str, float] = {}

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        state = {
            "time_step": time_step,
            "throughputs": {p.config.name: p.effective_throughput() for p in pipeline},
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
            "rework": {p.config.name: p.rework_units for p in pipeline},
            "failures": {p.config.name: p.failed_units for p in pipeline},
        }
        self.history.append(state)

        cycle_pos = time_step % self.dmaic_cycle_length

        # --- DMAIC Phase Transitions ---
        if cycle_pos == 0:
            self.phase = "Define"
            self.cycle_number += 1
            # Define: set quality targets based on current state
            for step in pipeline:
                defect_rate = step.config.uncertainty + step.config.failure_rate
                self.quality_targets[step.config.name] = defect_rate * 0.8  # 20% improvement target
            self.actions.append(
                OptimizationAction(
                    time_step=time_step,
                    target_process="system",
                    action_type="6S-Define",
                    description=f"DMAIC cycle {self.cycle_number}: targets set",
                )
            )

        elif cycle_pos < self.dmaic_cycle_length * 0.2:
            self.phase = "Measure"

        elif cycle_pos < self.dmaic_cycle_length * 0.4:
            self.phase = "Analyze"
            # Analyze root causes: find process with highest defect rate
            if cycle_pos == int(self.dmaic_cycle_length * 0.2):
                for step in pipeline:
                    if step.config.uncertainty > step.config.failure_rate:
                        self.root_causes[step.config.name] = "high_uncertainty"
                    else:
                        self.root_causes[step.config.name] = "high_failure"

        elif cycle_pos < self.dmaic_cycle_length * 0.7:
            self.phase = "Improve"
            # Apply targeted improvements
            if cycle_pos == int(self.dmaic_cycle_length * 0.4):
                worst = max(
                    pipeline,
                    key=lambda p: p.config.uncertainty + p.config.failure_rate,
                )
                cause = self.root_causes.get(worst.config.name, "high_uncertainty")
                if cause == "high_uncertainty":
                    worst.config.uncertainty *= 0.85
                else:
                    worst.config.failure_rate *= 0.85
                self.actions.append(
                    OptimizationAction(
                        time_step=time_step,
                        target_process=worst.config.name,
                        action_type="6S-Improve",
                        description=f"Fix {cause}: uncertainty={worst.config.uncertainty:.3f}, "
                        f"failure={worst.config.failure_rate:.3f}",
                    )
                )

        else:
            self.phase = "Control"
            # Set control limits
            if cycle_pos == int(self.dmaic_cycle_length * 0.7):
                for step in pipeline:
                    self.control_limits[step.config.name] = step.effective_throughput() * 0.9

        # --- Progressive AI ---
        progress = min(1.0, time_step / 80.0)
        for step in pipeline:
            step.ai_assistance_level = min(
                0.8, step.config.ai_automatable * (0.25 + progress * 0.6)
            )

        # --- Resource allocation: balanced + quality focus ---
        weights = {}
        for step in pipeline:
            defect_rate = step.config.uncertainty + step.config.failure_rate
            quality_weight = 1.0 + defect_rate * 2  # More resources to lower-quality processes
            wip_weight = max(0.5, step.work_in_progress / (step.effective_throughput() + 0.1))
            weights[step.config.name] = quality_weight + wip_weight

        total_weight = sum(weights.values())
        for step in pipeline:
            step.allocated_resources = (
                weights[step.config.name] / total_weight
            ) * total_resources

        # --- Human review optimization at cycle 3 ---
        if self.cycle_number == 3 and cycle_pos == 0:
            for step in pipeline:
                if step.ai_assistance_level > 0.5:
                    step.config.human_review_needed *= 0.8
                    self.actions.append(
                        OptimizationAction(
                            time_step=time_step,
                            target_process=step.config.name,
                            action_type="6S-Control",
                            description=f"Review reduction: → {step.config.human_review_needed:.2f}",
                        )
                    )

        return pipeline
