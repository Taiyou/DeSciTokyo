"""
Emerging challenge models for the research pipeline simulation.

Six models capturing unintended consequences of AI-augmented research:
1. QualityDriftModel (Goodhart's Law): proxy quality diverges from true quality
2. RedundancyModel: multiple parallel outputs decrease novelty
3. BlackBoxingModel: human understanding decays with AI involvement
4. TrustDecayModel: trust erodes on failures, recovers on successes
5. ObservabilityGapModel: higher AI complexity reduces observability
6. DelaySensitivityModel: feedback delays amplify at AI speeds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from sciops.pipeline.research_unit import ResearchUnit
from sciops.pipeline.state import PipelineState


@dataclass(frozen=True)
class ChallengeParams:
    """Parameters governing all challenge dynamics."""

    # Quality drift (Goodhart's Law)
    goodhart_drift_rate: float = 0.005
    goodhart_ai_amplification: float = 2.0

    # Scientific redundancy
    novelty_decay_rate: float = 0.01
    redundancy_penalty: float = 0.5

    # Black-boxing
    understanding_decay_rate: float = 0.02
    resilience_threshold: float = 0.3

    # Trust decay
    trust_decay_base: float = 0.01
    trust_recovery_rate: float = 0.005

    # Observability gap
    observability_decay_rate: float = 0.015

    # Delay sensitivity
    delay_amplification_exponent: float = 1.5


class QualityDriftModel:
    """Goodhart's Law: proxy quality diverges from true quality."""

    def __init__(self, params: ChallengeParams) -> None:
        self.params = params

    def apply(self, unit: ResearchUnit, ai_level: float, alpha: float) -> None:
        drift = self.params.goodhart_drift_rate * (
            1.0 + alpha * self.params.goodhart_ai_amplification
        )
        unit.proxy_quality = min(1.0, unit.proxy_quality + drift * ai_level)
        unit.quality = max(0.0, unit.quality - drift * ai_level * 0.1)

    def apply_batch(self, units: List[ResearchUnit], ai_level: float, alpha: float) -> None:
        for unit in units:
            self.apply(unit, ai_level, alpha)


class RedundancyModel:
    """Scientific redundancy: novelty decreases as more output is produced."""

    def __init__(self, params: ChallengeParams) -> None:
        self.params = params

    def compute_novelty_factor(self, cumulative_output: int) -> float:
        return float(np.exp(-self.params.novelty_decay_rate * cumulative_output))

    def compute_effective_output(self, cumulative_output: int) -> float:
        k = self.params.novelty_decay_rate
        if k <= 0:
            return float(cumulative_output)
        return (1.0 - np.exp(-k * cumulative_output)) / k


class BlackBoxingModel:
    """Human understanding decay: AI involvement erodes human comprehension."""

    def __init__(self, params: ChallengeParams) -> None:
        self.params = params

    def update_understanding(self, unit: ResearchUnit, ai_level: float) -> None:
        decay = self.params.understanding_decay_rate * ai_level
        unit.human_understanding = max(0.0, unit.human_understanding - decay)

    def update_batch(self, units: List[ResearchUnit], ai_level: float) -> None:
        for unit in units:
            self.update_understanding(unit, ai_level)

    def compute_resilience(self, state: PipelineState) -> float:
        all_units: List[ResearchUnit] = []
        for stage in state.stages.values():
            all_units.extend(stage.wip)
        if not all_units:
            return 1.0
        return float(np.mean([u.human_understanding for u in all_units]))

    def is_fragile(self, state: PipelineState) -> bool:
        return self.compute_resilience(state) < self.params.resilience_threshold


class TrustDecayModel:
    """Trust in AI decays on failures and recovers on successes."""

    def __init__(self, params: ChallengeParams) -> None:
        self.params = params
        self.trust_level: float = 1.0
        self._prev_total_failures: int = 0
        self._prev_total_successes: int = 0

    def update(self, state: PipelineState) -> float:
        current_failures = sum(s.cumulative_failed for s in state.stages.values())
        current_successes = state.cumulative_output

        new_failures = max(0, current_failures - self._prev_total_failures)
        new_successes = max(0, current_successes - self._prev_total_successes)

        self.trust_level -= self.params.trust_decay_base * new_failures
        self.trust_level += self.params.trust_recovery_rate * new_successes
        self.trust_level = float(np.clip(self.trust_level, 0.0, 1.0))

        self._prev_total_failures = current_failures
        self._prev_total_successes = current_successes
        return self.trust_level

    def reset(self) -> None:
        self.trust_level = 1.0
        self._prev_total_failures = 0
        self._prev_total_successes = 0


class ObservabilityGapModel:
    """As AI complexity grows, the system becomes less observable."""

    def __init__(self, params: ChallengeParams) -> None:
        self.params = params

    def compute_observation_noise(self, alpha: float, timestep: int) -> float:
        return self.params.observability_decay_rate * alpha * np.log1p(timestep)

    def apply_noise_to_wip(
        self,
        true_wip: Dict,
        alpha: float,
        timestep: int,
        rng: np.random.Generator,
    ) -> Dict:
        noise_level = self.compute_observation_noise(alpha, timestep)
        noisy = {}
        for stage_name, count in true_wip.items():
            noise = rng.normal(0, noise_level * max(1, count))
            noisy[stage_name] = max(0, int(round(count + noise)))
        return noisy


class DelaySensitivityModel:
    """Feedback delays become more consequential at higher AI speeds."""

    def __init__(self, params: ChallengeParams) -> None:
        self.params = params

    def compute_delay_cost(self, delay_ticks: int, alpha: float) -> float:
        base = max(0, delay_ticks)
        exponent = 1.0 + alpha * (self.params.delay_amplification_exponent - 1.0)
        return float(base**exponent)


class ChallengeSet:
    """Aggregate of all six challenge models."""

    def __init__(self, params: ChallengeParams | None = None) -> None:
        self.params = params or ChallengeParams()
        self.quality_drift = QualityDriftModel(self.params)
        self.redundancy = RedundancyModel(self.params)
        self.black_boxing = BlackBoxingModel(self.params)
        self.trust_decay = TrustDecayModel(self.params)
        self.observability_gap = ObservabilityGapModel(self.params)
        self.delay_sensitivity = DelaySensitivityModel(self.params)

    def update_all(self, state: PipelineState, alpha: float) -> Dict[str, float]:
        """Apply all challenge models for one tick. Returns metrics."""
        avg_ai = state.total_ai_level() / max(1, len(state.stages))

        # Quality drift on recent completions
        recent = state.completed_units[-20:] if state.completed_units else []
        self.quality_drift.apply_batch(recent, avg_ai, alpha)

        # Black-boxing: decay understanding for all active units
        for stage in state.stages.values():
            self.black_boxing.update_batch(stage.wip, stage.ai_level)

        # Trust decay
        trust = self.trust_decay.update(state)

        # Compute metrics
        resilience = self.black_boxing.compute_resilience(state)
        novelty = self.redundancy.compute_novelty_factor(state.cumulative_output)
        obs_noise = self.observability_gap.compute_observation_noise(alpha, state.timestep)

        return {
            "trust": trust,
            "resilience": resilience,
            "novelty_factor": novelty,
            "observability_noise": obs_noise,
            "is_fragile": float(self.black_boxing.is_fragile(state)),
        }

    def reset(self) -> None:
        self.trust_decay.reset()
