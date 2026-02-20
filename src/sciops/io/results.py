"""Result data structures and storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunResult:
    """Scalar results from a single simulation run."""

    scenario: str
    strategy: str
    alpha: float
    seed: int
    cumulative_output: int
    net_output: float
    mean_cycle_time: float
    quality_true_mean: float
    quality_proxy_mean: float
    quality_drift: float
    total_management_overhead: float
    max_wip: int
    bottleneck_stage: str


@dataclass
class TimeSeriesResult:
    """Per-tick time series from a single simulation run."""

    scenario: str
    strategy: str
    alpha: float
    seed: int
    wip_per_stage: np.ndarray
    throughput_per_stage: np.ndarray
    cumulative_output: np.ndarray
    quality_true: np.ndarray
    quality_proxy: np.ndarray
    management_overhead: np.ndarray
    resilience: np.ndarray
    trust: np.ndarray


class ResultsStore:
    """Saves and loads simulation results as CSV/Parquet."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._scalar_path = self.output_dir / "scalar_results.csv"
        self._timeseries_dir = self.output_dir / "timeseries"

    def save_scalar_results(self, results: List[RunResult]) -> None:
        """Append scalar results to CSV."""
        rows = []
        for r in results:
            rows.append({
                "scenario": r.scenario,
                "strategy": r.strategy,
                "alpha": r.alpha,
                "seed": r.seed,
                "cumulative_output": r.cumulative_output,
                "net_output": r.net_output,
                "mean_cycle_time": r.mean_cycle_time,
                "quality_true_mean": r.quality_true_mean,
                "quality_proxy_mean": r.quality_proxy_mean,
                "quality_drift": r.quality_drift,
                "total_management_overhead": r.total_management_overhead,
                "max_wip": r.max_wip,
                "bottleneck_stage": r.bottleneck_stage,
            })
        df = pd.DataFrame(rows)
        if self._scalar_path.exists():
            df.to_csv(self._scalar_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self._scalar_path, index=False)

    def load_scalar_results(self) -> pd.DataFrame:
        """Load all scalar results."""
        if not self._scalar_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self._scalar_path)

    def save_timeseries(self, ts: TimeSeriesResult) -> None:
        """Save a single time series result as compressed numpy."""
        self._timeseries_dir.mkdir(parents=True, exist_ok=True)
        key = f"{ts.scenario}_{ts.strategy}_a{ts.alpha:.2f}_s{ts.seed}"
        path = self._timeseries_dir / f"{key}.npz"
        np.savez_compressed(
            path,
            wip_per_stage=ts.wip_per_stage,
            cumulative_output=ts.cumulative_output,
            quality_true=ts.quality_true,
            quality_proxy=ts.quality_proxy,
            management_overhead=ts.management_overhead,
            resilience=ts.resilience,
            trust=ts.trust,
        )

    def load_timeseries(self, scenario: str, strategy: str, alpha: float, seed: int) -> dict:
        """Load a single time series result."""
        key = f"{scenario}_{strategy}_a{alpha:.2f}_s{seed}"
        path = self._timeseries_dir / f"{key}.npz"
        if not path.exists():
            return {}
        return dict(np.load(path))
