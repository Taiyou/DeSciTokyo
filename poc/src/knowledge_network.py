"""
Knowledge Network
==================
Models how published knowledge flows between labs.
Open science creates positive externalities: shared knowledge
accelerates all labs, but creates free-rider incentives.
"""

import math
import random
from dataclasses import dataclass, field

from meta_science_models import Paper


@dataclass
class KnowledgeItem:
    """A unit of shared scientific knowledge."""

    paper: Paper
    openness: float  # 0.0=closed, 1.0=fully open (data+code+methods)
    reusability: float  # How easy for others to build on
    time_published: int
    times_reused: int = 0


class KnowledgeNetwork:
    """Models knowledge flow between labs via open publications."""

    def __init__(self, decay_rate: float = 0.01, boost_cap: float = 2.0):
        self.items: list[KnowledgeItem] = []
        self.decay_rate = decay_rate
        self.boost_cap = boost_cap

    def add_knowledge(self, paper: Paper, time_step: int) -> None:
        """Add a published paper to the knowledge network."""
        reusability = paper.quality * (0.5 + 0.5 * paper.openness)
        item = KnowledgeItem(
            paper=paper,
            openness=paper.openness,
            reusability=reusability,
            time_published=time_step,
        )
        self.items.append(item)

    def knowledge_boost(self, current_time: int, lab_openness: float = 0.0) -> float:
        """Calculate throughput boost from accessible knowledge.

        Open items are accessible to all labs.
        Closed items are only accessible to labs that also contribute (reciprocity).
        Uses sampling for efficiency when knowledge base is large.
        """
        if not self.items:
            return 1.0

        # Sample items for efficiency (max 100 items)
        if len(self.items) > 100:
            sampled = random.sample(self.items, 100)
            scale = len(self.items) / 100.0
        else:
            sampled = self.items
            scale = 1.0

        boost = 0.0
        for item in sampled:
            if item.openness < 0.3:
                if lab_openness < 0.3:
                    continue

            relevance = random.betavariate(2, 5)
            recency = math.exp(-self.decay_rate * (current_time - item.time_published))
            contribution = item.openness * item.reusability * relevance * recency
            boost += contribution * 0.01

        # Scale up if we sampled
        boost *= scale

        return min(self.boost_cap, 1.0 + boost)

    def total_accessible_knowledge(self, lab_openness: float = 0.0) -> int:
        """Count total accessible knowledge items."""
        count = 0
        for item in self.items:
            if item.openness >= 0.3 or lab_openness >= 0.3:
                count += 1
        return count
