"""
Funding Allocation Models
==========================
Different mechanisms for distributing research funding across labs.
Each model implements a different allocation philosophy.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from meta_science_models import Paper


class FundingAllocator(ABC):
    """Base class for funding allocation mechanisms."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def allocate(
        self,
        lab_scores: list[dict],
        total_budget: float,
    ) -> list[float]:
        """Return funding amounts for each lab (same order as lab_scores).

        lab_scores contains dicts with keys:
            lab_id, reputation, recent_pubs, risk_appetite, current_funding
        """
        pass


class PeerReviewFunding(FundingAllocator):
    """Traditional grant review. Favors established labs with track records."""

    def __init__(self):
        super().__init__("Peer Review")

    def allocate(self, lab_scores, total_budget):
        scores = []
        for ls in lab_scores:
            proposal_quality = 0.7 * ls["reputation"] + 0.3 * random.random()
            score = (
                0.4 * ls["reputation"]
                + 0.3 * min(1.0, ls["recent_pubs"] / 5.0)
                + 0.3 * proposal_quality
            )
            scores.append(max(0.05, score))

        total_score = sum(scores)
        return [total_budget * s / total_score for s in scores]


class LotteryFunding(FundingAllocator):
    """Random allocation with minimum quality threshold."""

    def __init__(self, min_reputation: float = 0.2, n_winners_fraction: float = 0.6):
        super().__init__("Lottery")
        self.min_reputation = min_reputation
        self.n_winners_fraction = n_winners_fraction

    def allocate(self, lab_scores, total_budget):
        n_labs = len(lab_scores)
        eligible = [i for i, ls in enumerate(lab_scores)
                     if ls["reputation"] >= self.min_reputation]

        if not eligible:
            # Everyone gets equal
            return [total_budget / n_labs] * n_labs

        n_winners = max(1, int(len(eligible) * self.n_winners_fraction))
        winners = set(random.sample(eligible, min(n_winners, len(eligible))))

        # Base funding for all + bonus for winners
        base = total_budget * 0.3 / n_labs
        bonus_pool = total_budget * 0.7
        bonus = bonus_pool / max(1, len(winners))

        result = []
        for i in range(n_labs):
            if i in winners:
                result.append(base + bonus)
            else:
                result.append(base)
        return result


class SBIRStagedFunding(FundingAllocator):
    """Staged funding: small seed -> proof of concept -> full funding."""

    def __init__(self):
        super().__init__("SBIR Staged")
        self.lab_stages: dict[int, int] = {}  # lab_id -> current stage (0,1,2)
        self.stage_budgets = [0.15, 0.35, 1.0]  # Relative multipliers

    def allocate(self, lab_scores, total_budget):
        n_labs = len(lab_scores)

        # Initialize stages
        for ls in lab_scores:
            if ls["lab_id"] not in self.lab_stages:
                self.lab_stages[ls["lab_id"]] = 0

        # Advance stages based on performance
        for ls in lab_scores:
            lab_id = ls["lab_id"]
            stage = self.lab_stages[lab_id]
            if stage < 2 and ls["recent_pubs"] >= 1:
                # Good performance: advance stage
                if random.random() < 0.3:  # Not all advance
                    self.lab_stages[lab_id] = stage + 1
            elif stage > 0 and ls["recent_pubs"] == 0:
                # Poor performance: may regress
                if random.random() < 0.2:
                    self.lab_stages[lab_id] = max(0, stage - 1)

        # Allocate based on stages
        weights = []
        for ls in lab_scores:
            stage = self.lab_stages[ls["lab_id"]]
            weights.append(self.stage_budgets[stage])

        total_weight = sum(weights)
        return [total_budget * w / total_weight for w in weights]


class FROLongTermFunding(FundingAllocator):
    """Focused Research Organization: large, long-term, goal-directed funding."""

    def __init__(self, program_duration: int = 50, n_programs: int = 5):
        super().__init__("FRO Long-term")
        self.program_duration = program_duration
        self.n_programs = n_programs
        self.programs: dict[int, dict] = {}  # program_id -> {labs, remaining}
        self.cycle_count = 0

    def allocate(self, lab_scores, total_budget):
        n_labs = len(lab_scores)
        self.cycle_count += 1

        # Create new programs if needed
        if not self.programs or self.cycle_count % (self.program_duration // 20) == 0:
            self._create_programs(lab_scores)

        # Stable funding for program members
        program_budget = total_budget * 0.7
        general_budget = total_budget * 0.3

        funding = [general_budget / n_labs] * n_labs

        active_program_labs = set()
        for pid, prog in self.programs.items():
            prog["remaining"] -= 1
            if prog["remaining"] > 0:
                per_lab = program_budget / max(1, sum(
                    len(p["labs"]) for p in self.programs.values() if p["remaining"] > 0
                ))
                for lab_id in prog["labs"]:
                    if lab_id < n_labs:
                        funding[lab_id] += per_lab
                        active_program_labs.add(lab_id)

        # Remove expired programs
        self.programs = {
            pid: p for pid, p in self.programs.items() if p["remaining"] > 0
        }

        return funding

    def _create_programs(self, lab_scores):
        """Select labs for new programs."""
        available = [ls for ls in lab_scores
                      if ls["lab_id"] not in {
                          lid for p in self.programs.values()
                          for lid in p["labs"]
                      }]
        if len(available) < 2:
            return

        # Select high-potential labs (mix of risk appetites)
        sorted_by_potential = sorted(
            available,
            key=lambda ls: ls["reputation"] * 0.5 + ls["risk_appetite"] * 0.5,
            reverse=True,
        )
        n_select = min(len(sorted_by_potential), max(2, len(sorted_by_potential) // 3))
        selected = sorted_by_potential[:n_select]

        pid = len(self.programs)
        self.programs[pid] = {
            "labs": [ls["lab_id"] for ls in selected],
            "remaining": self.program_duration // 20,  # In funding cycles
        }
