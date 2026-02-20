"""Strategy factory: creates strategy instances by name."""

from __future__ import annotations

from typing import Dict, List, Type

from sciops.strategies.agile import AgileStrategy
from sciops.strategies.ai_assisted import AIAssistedStrategy
from sciops.strategies.ai_sciops import AISciOpsStrategy
from sciops.strategies.base import Strategy
from sciops.strategies.baseline import BaselineStrategy
from sciops.strategies.kanban import KanbanStrategy
from sciops.strategies.oracle import OracleStrategy
from sciops.strategies.toc_pdca import TOCPDCAStrategy

_REGISTRY: Dict[str, Type[Strategy]] = {
    "baseline": BaselineStrategy,
    "toc_pdca": TOCPDCAStrategy,
    "kanban": KanbanStrategy,
    "agile": AgileStrategy,
    "ai_assisted": AIAssistedStrategy,
    "ai_sciops": AISciOpsStrategy,
    "oracle": OracleStrategy,
}


def create_strategy(name: str) -> Strategy:
    """Create a strategy instance by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()


def list_strategies() -> List[str]:
    """Return all registered strategy names."""
    return list(_REGISTRY.keys())
