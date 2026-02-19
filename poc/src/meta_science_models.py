"""
Meta-Science Shared Models
===========================
Common data structures for all meta-science simulation experiments.
Models the entities and interactions in a multi-lab scientific ecosystem.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    """A research output that can be submitted for publication."""

    lab_id: int
    time_step: int
    quality: float  # 0.0-1.0, methodological soundness
    effect_size: float  # True underlying effect size
    reported_effect: float  # What the paper claims (may be inflated by p-hacking)
    novelty: float  # 0.0-1.0
    is_true: bool  # Ground truth: is the finding actually real?
    is_published: bool = False
    citations: int = 0
    replicated: Optional[bool] = None  # None = not yet tested
    retracted: bool = False
    openness: float = 0.0  # 0.0-1.0, how open the data/code is


@dataclass
class PublicationChannel:
    """Models a journal/publication venue with review standards."""

    name: str
    acceptance_rate: float = 0.2  # 0.0-1.0
    novelty_bias: float = 0.3  # How much novelty affects acceptance
    positive_result_bias: float = 0.4  # How much positive results are favored
    review_quality: float = 0.6  # Ability to detect flawed papers

    def evaluate(self, paper: Paper) -> bool:
        """Decide whether to accept a paper for publication.

        When positive_result_bias is 0 (fair review), acceptance depends mainly
        on quality. When bias > 0, papers with large reported effects and high
        novelty are favored, allowing false positives with inflated effects through.
        """
        # Base quality score: the core merit of the paper (always present)
        quality_score = paper.quality

        # Bias component: favors large reported effects and novelty
        # This is the key mechanism: when bias > 0, false positives with
        # inflated effects can score higher than their quality merits
        bias_bonus = (
            abs(paper.reported_effect) * self.positive_result_bias
            + paper.novelty * self.novelty_bias
        )

        # Review quality: ability to detect effect inflation (p-hacking)
        # High review_quality penalizes papers where reported >> true effect
        effect_gap = abs(paper.reported_effect - paper.effect_size)
        detection_penalty = effect_gap * self.review_quality * 0.3

        score = quality_score + bias_bonus - detection_penalty + random.gauss(0, 0.1)

        # Acceptance threshold: higher = more selective
        threshold = 1.0 - self.acceptance_rate
        return score > threshold


@dataclass
class EcosystemMetrics:
    """Metrics collected at each time step of the ecosystem simulation."""

    time_step: int
    total_papers_submitted: int
    total_papers_published: int
    truth_ratio: float  # Fraction of published papers that are true
    false_positive_rate: float  # Fraction of positive papers that are false
    mean_lab_reputation: float
    gini_funding: float  # Inequality of funding distribution
    knowledge_growth_rate: float  # New true papers this step
    ai_lab_publication_share: float  # Fraction of publications from AI labs
    total_community_output: float  # Sum of all lab pipeline outputs
    replication_rate: float = 0.0  # Fraction of papers replicated
    successful_replication_rate: float = 0.0  # Of replicated, fraction successful
    retraction_count: int = 0


def compute_gini(values: list[float]) -> float:
    """Compute the Gini coefficient of a list of values."""
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = 0.0
    gini_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        gini_sum += cumulative
    gini = (2 * gini_sum) / (n * total) - (n + 1) / n
    return max(0.0, min(1.0, gini))
