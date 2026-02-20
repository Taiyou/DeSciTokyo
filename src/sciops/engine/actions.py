"""Unified action interface for all management strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from sciops.pipeline.config import StageName


@dataclass(frozen=True)
class AllocateResources:
    """Reallocate resources to a specific stage (within total_resources constraint)."""
    stage: StageName
    amount: float


@dataclass(frozen=True)
class AdjustAILevel:
    """Adjust the AI support level for a stage (incurs cost)."""
    stage: StageName
    delta: float


@dataclass(frozen=True)
class InvestUncertaintyReduction:
    """Invest in reducing uncertainty at a stage (diminishing returns)."""
    stage: StageName
    amount: float


@dataclass(frozen=True)
class AdjustWIPLimit:
    """Set or modify the WIP limit for a stage."""
    stage: StageName
    limit: Optional[int]


@dataclass(frozen=True)
class Restructure:
    """Pipeline structure modification (high cost + downtime)."""
    description: str
    cost: float = 0.5
    downtime_ticks: int = 3


Action = AllocateResources | AdjustAILevel | InvestUncertaintyReduction | AdjustWIPLimit | Restructure


@dataclass
class ActionSet:
    """Container for a strategy's actions in a single timestep."""
    actions: List[Action] = field(default_factory=list)

    def add(self, action: Action) -> None:
        self.actions.append(action)

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)
