"""MetricsCollector: records per-tick metrics during a simulation run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from sciops.pipeline.config import STAGE_ORDER, StageName
from sciops.pipeline.state import PipelineState


@dataclass
class TickMetrics:
    """Metrics recorded at a single timestep."""

    timestep: int
    wip_per_stage: Dict[StageName, int]
    cumulative_output: int
    net_output: float
    quality_true_mean: float
    quality_proxy_mean: float
    management_overhead: float
    resilience: float
    trust_level: float
    total_wip: int
    novelty_factor: float


class MetricsCollector:
    """Collects per-tick metrics and converts to arrays for storage."""

    def __init__(self) -> None:
        self.history: List[TickMetrics] = []

    def record(self, state: PipelineState, challenge_metrics: Dict[str, float]) -> None:
        recent = state.completed_units[-10:] if state.completed_units else []
        if recent:
            q_true = float(np.mean([u.quality for u in recent]))
            q_proxy = float(np.mean([u.proxy_quality for u in recent]))
        else:
            q_true = 1.0
            q_proxy = 1.0

        tick = TickMetrics(
            timestep=state.timestep,
            wip_per_stage={name: stage.wip_count for name, stage in state.stages.items()},
            cumulative_output=state.cumulative_output,
            net_output=state.net_output,
            quality_true_mean=q_true,
            quality_proxy_mean=q_proxy,
            management_overhead=state.management_overhead,
            resilience=challenge_metrics.get("resilience", 1.0),
            trust_level=challenge_metrics.get("trust", 1.0),
            total_wip=state.total_wip,
            novelty_factor=challenge_metrics.get("novelty_factor", 1.0),
        )
        self.history.append(tick)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert recorded history to numpy arrays."""
        n = len(self.history)
        if n == 0:
            return {}

        result = {
            "timestep": np.array([t.timestep for t in self.history]),
            "cumulative_output": np.array([t.cumulative_output for t in self.history]),
            "net_output": np.array([t.net_output for t in self.history]),
            "quality_true": np.array([t.quality_true_mean for t in self.history]),
            "quality_proxy": np.array([t.quality_proxy_mean for t in self.history]),
            "total_wip": np.array([t.total_wip for t in self.history]),
            "management_overhead": np.array([t.management_overhead for t in self.history]),
            "resilience": np.array([t.resilience for t in self.history]),
            "trust": np.array([t.trust_level for t in self.history]),
            "novelty_factor": np.array([t.novelty_factor for t in self.history]),
        }

        # WIP per stage as 2D array (num_steps x 6)
        wip_matrix = np.zeros((n, len(STAGE_ORDER)))
        for i, tick in enumerate(self.history):
            for j, name in enumerate(STAGE_ORDER):
                wip_matrix[i, j] = tick.wip_per_stage.get(name, 0)
        result["wip_per_stage"] = wip_matrix

        return result

    @property
    def num_ticks(self) -> int:
        return len(self.history)

    def clear(self) -> None:
        self.history = []
