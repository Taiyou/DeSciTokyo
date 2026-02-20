"""
Probabilistic Process Model
============================
Replaces uniform random distributions with more realistic probability models:

1. Beta distributions for uncertainty (non-symmetric, realistic for research)
2. Poisson processes for event arrivals (failures, breakthroughs)
3. Inter-process correlation structures (experiment failure -> hypothesis rework)

This addresses Future-Work item "Probabilistic Model Refinement" from the roadmap.
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from scientific_process import ProcessConfig, ProcessState


# ============================================================
# Beta distribution sampling (pure Python, no scipy needed)
# ============================================================

def _gamma_sample(alpha: float) -> float:
    """Sample from Gamma(alpha, 1) using Marsaglia-Tsang method.

    Works for alpha >= 1. For alpha < 1, uses the relation:
    Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
    """
    if alpha < 1.0:
        u = random.random()
        return _gamma_sample(alpha + 1.0) * (u ** (1.0 / alpha))

    d = alpha - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)

    while True:
        x = random.gauss(0, 1)
        v = (1.0 + c * x) ** 3
        if v <= 0:
            continue
        u = random.random()
        if u < 1.0 - 0.0331 * (x * x) * (x * x):
            return d * v
        if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
            return d * v


def beta_sample(alpha: float, beta: float) -> float:
    """Sample from Beta(alpha, beta) distribution.

    Uses the relation: if X ~ Gamma(a,1) and Y ~ Gamma(b,1),
    then X/(X+Y) ~ Beta(a,b).
    """
    if alpha <= 0 or beta <= 0:
        return random.random()  # fallback
    x = _gamma_sample(alpha)
    y = _gamma_sample(beta)
    if x + y == 0:
        return 0.5
    return x / (x + y)


def poisson_sample(lam: float) -> int:
    """Sample from Poisson(lambda) distribution using Knuth's algorithm."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p < L:
            return k - 1


# ============================================================
# Process correlation model
# ============================================================

@dataclass
class CorrelationEvent:
    """An event that propagates between processes."""
    source_process: str
    target_process: str
    event_type: str  # "failure_propagation", "breakthrough_boost", "quality_cascade"
    magnitude: float  # 0.0-1.0
    delay: int  # time steps before effect manifests


CORRELATION_RULES = [
    # Experiment failure triggers hypothesis rework
    {"source": "Experiment", "target": "Hypothesis",
     "trigger": "failure", "effect": "rework_boost", "magnitude": 0.4, "delay": 1},
    # Good analysis quality improves writing speed
    {"source": "Analysis", "target": "Writing",
     "trigger": "high_quality", "effect": "throughput_boost", "magnitude": 0.2, "delay": 0},
    # Survey thoroughness reduces hypothesis uncertainty
    {"source": "Survey", "target": "Hypothesis",
     "trigger": "high_throughput", "effect": "uncertainty_reduction", "magnitude": 0.15, "delay": 0},
    # Writing rejection cascades back to analysis
    {"source": "Review", "target": "Analysis",
     "trigger": "failure", "effect": "rework_boost", "magnitude": 0.3, "delay": 1},
    # Experiment breakthrough boosts analysis throughput
    {"source": "Experiment", "target": "Analysis",
     "trigger": "breakthrough", "effect": "throughput_boost", "magnitude": 0.25, "delay": 0},
]


# ============================================================
# Enhanced process step with probabilistic model
# ============================================================

@dataclass
class ProbabilisticProcessStep:
    """Enhanced process step using Beta/Poisson distributions and correlations."""

    config: ProcessConfig
    state: ProcessState = ProcessState.IDLE
    throughput: float = 0.0
    allocated_resources: float = 1.0
    ai_assistance_level: float = 0.0
    work_in_progress: float = 0.0
    completed_units: float = 0.0
    failed_units: float = 0.0
    rework_units: float = 0.0
    human_review_backlog: float = 0.0
    cumulative_wait_time: float = 0.0

    # Beta distribution parameters (derived from config uncertainty)
    # Higher alpha = more likely to succeed, higher beta = more uncertainty
    uncertainty_alpha: float = 0.0
    uncertainty_beta: float = 0.0

    # Correlation effects accumulator
    pending_effects: list = field(default_factory=list)
    temp_throughput_boost: float = 0.0
    temp_uncertainty_reduction: float = 0.0
    temp_rework_extra: float = 0.0

    # Tracking for correlation triggers
    step_had_failure: bool = False
    step_had_breakthrough: bool = False
    step_throughput_ratio: float = 0.0

    def __post_init__(self):
        self.throughput = self.config.base_throughput
        # Derive Beta parameters from uncertainty
        # uncertainty=0.2 -> alpha=4, beta=1 (mostly successful)
        # uncertainty=0.5 -> alpha=2, beta=2 (symmetric, high variance)
        # uncertainty=0.8 -> alpha=1, beta=4 (mostly fails)
        u = self.config.uncertainty
        self.uncertainty_alpha = max(0.5, 2.0 * (1 - u) + 0.5)
        self.uncertainty_beta = max(0.5, 2.0 * u + 0.5)

    def effective_throughput(self) -> float:
        """Calculate effective throughput (same logic as original)."""
        base = self.throughput * self.allocated_resources
        ai_boost = 1.0 + (
            self.ai_assistance_level * self.config.ai_automatable * 2.0
        )
        effective = base * ai_boost

        # Apply temporary throughput boost from correlations
        effective *= (1.0 + self.temp_throughput_boost)

        if self.ai_assistance_level > 0.5 and self.config.human_review_needed > 0.3:
            review_bottleneck = 1.0 - (
                self.config.human_review_needed
                * self.ai_assistance_level
                * 0.5
            )
            effective *= max(0.2, review_bottleneck)

        return max(self.config.min_throughput, min(effective, self.config.max_throughput))

    def step(self, incoming_work: float) -> float:
        """Execute one time step with probabilistic model."""
        self.work_in_progress += incoming_work
        self.step_had_failure = False
        self.step_had_breakthrough = False

        # Apply and clear pending correlation effects
        self._apply_pending_effects()

        if self.work_in_progress <= 0:
            self.state = ProcessState.IDLE
            self.cumulative_wait_time += 1
            return 0.0

        self.state = ProcessState.RUNNING
        capacity = self.effective_throughput()
        processable = min(self.work_in_progress, capacity)

        # Track throughput ratio for correlation triggers
        self.step_throughput_ratio = capacity / self.config.base_throughput if self.config.base_throughput > 0 else 1.0

        # === Beta-distributed uncertainty ===
        # Sample from Beta distribution instead of uniform
        effective_alpha = self.uncertainty_alpha * (1 + self.ai_assistance_level * 0.5)
        effective_beta = self.uncertainty_beta * (1 - self.ai_assistance_level * 0.3)
        effective_beta = max(0.3, effective_beta - self.temp_uncertainty_reduction)

        rework_probability = beta_sample(effective_beta, effective_alpha)
        # Beta gives us probability; threshold at config uncertainty level
        if rework_probability > (1 - self.config.uncertainty):
            rework = processable * 0.3
            self.rework_units += rework
            self.work_in_progress += rework + self.temp_rework_extra
            processable *= 0.7

        # === Poisson-distributed failures ===
        # Expected failures per step based on failure rate
        failure_lambda = self.config.failure_rate * (1 - self.ai_assistance_level * 0.3)
        n_failures = poisson_sample(failure_lambda)
        if n_failures > 0:
            # Each failure event removes a fraction of processable work
            failure_fraction = min(0.5, n_failures * 0.1)
            failed = processable * failure_fraction
            self.failed_units += failed
            processable -= failed
            self.step_had_failure = True

        # === Breakthrough detection (Poisson, rare events) ===
        breakthrough_lambda = 0.02 * (1 + self.ai_assistance_level * 0.5)
        if poisson_sample(breakthrough_lambda) > 0:
            processable *= 1.3  # 30% boost on breakthrough
            self.step_had_breakthrough = True

        # Human review backlog (same as original)
        review_needed = processable * self.config.human_review_needed
        if self.ai_assistance_level > 0.3:
            self.human_review_backlog += review_needed * self.ai_assistance_level
            reviewed = min(self.human_review_backlog, capacity * 0.3)
            self.human_review_backlog -= reviewed

        self.work_in_progress -= min(processable + review_needed, self.work_in_progress)
        self.completed_units += processable

        # Reset temporary effects
        self.temp_throughput_boost = 0.0
        self.temp_uncertainty_reduction = 0.0
        self.temp_rework_extra = 0.0

        return processable

    def _apply_pending_effects(self):
        """Apply pending correlation effects from other processes."""
        remaining = []
        for effect in self.pending_effects:
            if effect["delay"] <= 0:
                etype = effect["type"]
                mag = effect["magnitude"]
                if etype == "throughput_boost":
                    self.temp_throughput_boost += mag
                elif etype == "uncertainty_reduction":
                    self.temp_uncertainty_reduction += mag
                elif etype == "rework_boost":
                    self.temp_rework_extra += mag
            else:
                effect["delay"] -= 1
                remaining.append(effect)
        self.pending_effects = remaining


def create_probabilistic_pipeline() -> list[ProbabilisticProcessStep]:
    """Create pipeline with probabilistic process steps."""
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
    return [ProbabilisticProcessStep(config=c) for c in configs]


class CorrelationEngine:
    """Manages inter-process correlation events."""

    def __init__(self, pipeline: list[ProbabilisticProcessStep]):
        self.pipeline = pipeline
        self.process_map = {p.config.name: p for p in pipeline}
        self.events_triggered: list[dict] = []

    def propagate_correlations(self, time_step: int):
        """Check for correlation triggers and propagate effects."""
        for rule in CORRELATION_RULES:
            source = self.process_map.get(rule["source"])
            target = self.process_map.get(rule["target"])
            if source is None or target is None:
                continue

            triggered = False
            if rule["trigger"] == "failure" and source.step_had_failure:
                triggered = True
            elif rule["trigger"] == "breakthrough" and source.step_had_breakthrough:
                triggered = True
            elif rule["trigger"] == "high_throughput" and source.step_throughput_ratio > 1.5:
                triggered = True
            elif rule["trigger"] == "high_quality" and source.step_throughput_ratio > 1.2 and not source.step_had_failure:
                triggered = True

            if triggered:
                target.pending_effects.append({
                    "type": rule["effect"],
                    "magnitude": rule["magnitude"],
                    "delay": rule["delay"],
                })
                self.events_triggered.append({
                    "time_step": time_step,
                    "source": rule["source"],
                    "target": rule["target"],
                    "trigger": rule["trigger"],
                    "effect": rule["effect"],
                })
