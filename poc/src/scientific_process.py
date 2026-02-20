"""
Scientific Process Simulator
=============================
Models a scientific research pipeline as a series of processes,
each with throughput, uncertainty, and failure characteristics.

Based on the "Science in the Loop" framework:
- Survey → Hypothesis → Experiment → Analysis → Writing → Review
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProcessState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessConfig:
    """Configuration for a single scientific process step."""

    name: str
    base_throughput: float  # units per time step
    uncertainty: float  # 0.0 - 1.0, probability of needing rework
    failure_rate: float  # 0.0 - 1.0, probability of complete failure
    resource_cost: float  # resource units consumed per time step
    ai_automatable: float  # 0.0 - 1.0, degree to which AI can automate
    human_review_needed: float  # 0.0 - 1.0, fraction requiring human review
    min_throughput: float = 0.1
    max_throughput: float = 10.0


@dataclass
class ProcessStep:
    """Runtime state of a single process step in the pipeline."""

    config: ProcessConfig
    state: ProcessState = ProcessState.IDLE
    throughput: float = 0.0
    allocated_resources: float = 1.0
    ai_assistance_level: float = 0.0  # 0.0 = no AI, 1.0 = fully AI-driven
    work_in_progress: float = 0.0
    completed_units: float = 0.0
    failed_units: float = 0.0
    rework_units: float = 0.0
    human_review_backlog: float = 0.0
    cumulative_wait_time: float = 0.0

    def __post_init__(self):
        self.throughput = self.config.base_throughput

    def effective_throughput(self) -> float:
        """Calculate effective throughput considering AI assistance and resources."""
        base = self.throughput * self.allocated_resources

        # AI boosts throughput for automatable portions
        ai_boost = 1.0 + (
            self.ai_assistance_level * self.config.ai_automatable * 2.0
        )
        effective = base * ai_boost

        # Human review bottleneck: if AI assistance is high but review is needed,
        # the human becomes the bottleneck (key insight from the paper)
        if self.ai_assistance_level > 0.5 and self.config.human_review_needed > 0.3:
            review_bottleneck = 1.0 - (
                self.config.human_review_needed
                * self.ai_assistance_level
                * 0.5
            )
            effective *= max(0.2, review_bottleneck)

        return max(self.config.min_throughput, min(effective, self.config.max_throughput))

    def step(self, incoming_work: float) -> float:
        """Execute one time step. Returns completed output units."""
        self.work_in_progress += incoming_work

        if self.work_in_progress <= 0:
            self.state = ProcessState.IDLE
            self.cumulative_wait_time += 1
            return 0.0

        self.state = ProcessState.RUNNING
        capacity = self.effective_throughput()

        # Process work
        processable = min(self.work_in_progress, capacity)

        # Apply uncertainty (rework needed)
        rework_fraction = random.random()
        if rework_fraction < self.config.uncertainty * (1 - self.ai_assistance_level * 0.5):
            rework = processable * 0.3
            self.rework_units += rework
            self.work_in_progress += rework  # rework goes back to queue
            processable *= 0.7

        # Apply failure rate
        if random.random() < self.config.failure_rate * (1 - self.ai_assistance_level * 0.3):
            failed = processable * 0.1
            self.failed_units += failed
            processable -= failed

        # Human review backlog
        review_needed = processable * self.config.human_review_needed
        if self.ai_assistance_level > 0.3:
            self.human_review_backlog += review_needed * self.ai_assistance_level
            # Only a fraction gets reviewed per step
            reviewed = min(self.human_review_backlog, capacity * 0.3)
            self.human_review_backlog -= reviewed

        self.work_in_progress -= min(processable + review_needed, self.work_in_progress)
        self.completed_units += processable
        return processable


def create_default_pipeline() -> list[ProcessStep]:
    """Create a default scientific research pipeline.

    Models the typical flow:
    Survey → Hypothesis → Experiment → Analysis → Writing → Review

    Parameters are designed to create realistic bottleneck patterns:
    - Experiment has lowest throughput (physical world constraint)
    - Analysis has high AI automatability
    - Review has high human review requirement
    """
    configs = [
        ProcessConfig(
            name="Survey",
            base_throughput=2.0,
            uncertainty=0.2,
            failure_rate=0.05,
            resource_cost=1.0,
            ai_automatable=0.8,
            human_review_needed=0.2,
        ),
        ProcessConfig(
            name="Hypothesis",
            base_throughput=1.5,
            uncertainty=0.4,
            failure_rate=0.1,
            resource_cost=1.5,
            ai_automatable=0.6,
            human_review_needed=0.5,
        ),
        ProcessConfig(
            name="Experiment",
            base_throughput=0.8,
            uncertainty=0.5,
            failure_rate=0.15,
            resource_cost=3.0,
            ai_automatable=0.3,
            human_review_needed=0.3,
        ),
        ProcessConfig(
            name="Analysis",
            base_throughput=1.8,
            uncertainty=0.3,
            failure_rate=0.08,
            resource_cost=2.0,
            ai_automatable=0.9,
            human_review_needed=0.4,
        ),
        ProcessConfig(
            name="Writing",
            base_throughput=1.2,
            uncertainty=0.3,
            failure_rate=0.05,
            resource_cost=1.5,
            ai_automatable=0.7,
            human_review_needed=0.6,
        ),
        ProcessConfig(
            name="Review",
            base_throughput=0.6,
            uncertainty=0.2,
            failure_rate=0.1,
            resource_cost=1.0,
            ai_automatable=0.4,
            human_review_needed=0.8,
        ),
    ]
    return [ProcessStep(config=c) for c in configs]
