"""
Replication Engine
===================
Models the replication process in scientific research.
Papers can be replicated by other labs, with costs and incentives.
"""

import random
from dataclasses import dataclass, field

from meta_science_models import Paper


@dataclass
class ReplicationAttempt:
    """Record of a replication attempt."""

    original_paper: Paper
    replicating_lab_id: int
    time_step: int
    replicated: bool  # Did it successfully replicate?
    cost: float  # Resources consumed


class ReplicationEngine:
    """Manages replication attempts across the ecosystem."""

    def __init__(
        self,
        replication_cost_ratio: float = 3.0,
        ai_cost_discount: float = 0.6,
    ):
        self.replication_cost_ratio = replication_cost_ratio
        self.ai_cost_discount = ai_cost_discount
        self.attempts: list[ReplicationAttempt] = []

    def attempt_replication(
        self,
        paper: Paper,
        lab_id: int,
        time_step: int,
        lab_quality: float = 0.7,
        is_ai_lab: bool = False,
    ) -> ReplicationAttempt:
        """Attempt to replicate a paper."""
        # Cost calculation
        cost = self.replication_cost_ratio
        if is_ai_lab:
            cost *= (1.0 - self.ai_cost_discount)

        # Probability of successful replication
        if paper.is_true:
            # True findings replicate with high probability if quality is high
            p_replicate = 0.85 * paper.quality * lab_quality
        else:
            # False findings rarely replicate
            p_replicate = 0.1 * (1.0 - paper.quality)

        replicated = random.random() < p_replicate
        paper.replicated = replicated

        attempt = ReplicationAttempt(
            original_paper=paper,
            replicating_lab_id=lab_id,
            time_step=time_step,
            replicated=replicated,
            cost=cost,
        )
        self.attempts.append(attempt)
        return attempt

    def get_replication_stats(self) -> dict:
        """Compute summary replication statistics."""
        if not self.attempts:
            return {
                "total_attempts": 0,
                "successful": 0,
                "failed": 0,
                "replication_rate": 0.0,
                "total_cost": 0.0,
            }

        successful = sum(1 for a in self.attempts if a.replicated)
        return {
            "total_attempts": len(self.attempts),
            "successful": successful,
            "failed": len(self.attempts) - successful,
            "replication_rate": successful / len(self.attempts),
            "total_cost": sum(a.cost for a in self.attempts),
        }
