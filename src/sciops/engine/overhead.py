"""Management overhead model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OverheadConfig:
    """Overhead cost parameters for a management strategy."""
    base_cost: float = 0.0
    ai_infra_cost: float = 0.0
    human_coord_cost: float = 0.0


def compute_overhead(
    config: OverheadConfig,
    total_ai_level: float,
    num_active_stages: int,
    timestep: int,
) -> float:
    """
    Compute management overhead for one timestep.

    Total overhead = base_cost + ai_infra_cost * total_ai_level
                     + human_coord_cost * num_active_stages / 6
    """
    overhead = config.base_cost
    overhead += config.ai_infra_cost * total_ai_level
    overhead += config.human_coord_cost * num_active_stages / 6.0
    return max(0.0, overhead)
