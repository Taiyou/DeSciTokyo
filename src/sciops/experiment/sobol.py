"""Sobol sensitivity analysis experiment runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from joblib import Parallel, delayed
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import saltelli
from tqdm import tqdm

from sciops.experiment.run_config import RunConfig
from sciops.experiment.runner import execute_single_run
from sciops.metrics.challenges import ChallengeParams
from sciops.pipeline.config import (
    FeedbackConfig,
    PipelineConfig,
    StageConfig,
    StageName,
)

SOBOL_PROBLEM: Dict = {
    "num_vars": 14,
    "names": [
        "survey_throughput",
        "hypothesis_throughput",
        "experiment_throughput",
        "analysis_throughput",
        "writing_throughput",
        "review_throughput",
        "p_revision",
        "p_minor_revision",
        "p_major_rejection",
        "arrival_rate",
        "alpha",
        "goodhart_drift_rate",
        "understanding_decay_rate",
        "novelty_decay_rate",
    ],
    "bounds": [
        [1.0, 4.0],    # survey_throughput
        [0.5, 3.0],    # hypothesis_throughput
        [0.3, 2.0],    # experiment_throughput
        [0.8, 3.5],    # analysis_throughput
        [0.5, 2.5],    # writing_throughput
        [0.3, 1.5],    # review_throughput
        [0.0, 0.6],    # p_revision
        [0.0, 0.5],    # p_minor_revision
        [0.0, 0.15],   # p_major_rejection
        [1.0, 6.0],    # arrival_rate
        [0.0, 1.0],    # alpha
        [0.0, 0.02],   # goodhart_drift_rate
        [0.0, 0.05],   # understanding_decay_rate
        [0.0, 0.03],   # novelty_decay_rate
    ],
}

_DEFAULT_UNCERTAINTY = [0.2, 0.4, 0.5, 0.3, 0.3, 0.2]
_DEFAULT_FAILURE = [0.05, 0.1, 0.15, 0.08, 0.05, 0.1]
_DEFAULT_AI_AUTO = [0.8, 0.6, 0.3, 0.9, 0.7, 0.4]
_DEFAULT_HUMAN = [0.2, 0.5, 0.3, 0.4, 0.6, 0.8]

_STAGE_NAMES = [
    StageName.SURVEY,
    StageName.HYPOTHESIS,
    StageName.EXPERIMENT,
    StageName.ANALYSIS,
    StageName.WRITING,
    StageName.REVIEW,
]


def generate_sobol_samples(n: int = 1024) -> np.ndarray:
    """Generate Saltelli samples. Returns N * (2D + 2) sample points."""
    return saltelli.sample(SOBOL_PROBLEM, n, calc_second_order=True)


def _sobol_params_to_config(
    params: np.ndarray,
    seed: int,
    strategy: str,
) -> tuple[RunConfig, ChallengeParams]:
    """Convert a 14-element Sobol parameter vector to RunConfig + ChallengeParams."""
    stages = {}
    for i, name in enumerate(_STAGE_NAMES):
        stages[name] = StageConfig(
            name=name,
            base_throughput=float(params[i]),
            uncertainty=_DEFAULT_UNCERTAINTY[i],
            failure_rate=_DEFAULT_FAILURE[i],
            ai_automatable=_DEFAULT_AI_AUTO[i],
            human_review_needed=_DEFAULT_HUMAN[i],
        )

    feedback = FeedbackConfig(
        p_revision=float(params[6]),
        p_minor_revision=float(params[7]),
        p_major_rejection=float(params[8]),
        enable_feedback=True,
    )

    pipeline_config = PipelineConfig(
        stages=stages,
        feedback=feedback,
        arrival_rate=float(params[9]),
    )

    alpha = float(params[10])

    challenge_params = ChallengeParams(
        goodhart_drift_rate=float(params[11]),
        understanding_decay_rate=float(params[12]),
        novelty_decay_rate=float(params[13]),
    )

    run_config = RunConfig(
        scenario_name="sobol",
        strategy_name=strategy,
        alpha=alpha,
        seed=seed,
        pipeline_config=pipeline_config,
    )

    return run_config, challenge_params


def run_sobol_analysis(
    output_dir: Path,
    strategy: str = "ai_sciops",
    n: int = 1024,
    n_jobs: int = -1,
    seeds_per_sample: int = 10,
) -> Dict:
    """
    Run Sobol sensitivity analysis for a given strategy.

    Total runs = N * (2D + 2) * seeds_per_sample
    For N=1024, D=14, seeds=10: 1024 * 30 * 10 = 307,200 runs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = generate_sobol_samples(n)
    print(
        f"Sobol samples: {len(samples)} points x {seeds_per_sample} seeds = "
        f"{len(samples) * seeds_per_sample} total runs"
    )

    # Build all run configs
    all_tasks = []
    for i, params in enumerate(samples):
        for s in range(seeds_per_sample):
            run_cfg, ch_params = _sobol_params_to_config(
                params, seed=i * 10000 + s, strategy=strategy
            )
            all_tasks.append((run_cfg, ch_params))

    # Execute in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(execute_single_run)(cfg, ch, False)
        for cfg, ch in tqdm(all_tasks, desc="Sobol runs")
    )

    # Aggregate: mean net_output per sample point
    outputs = np.array([r[0].net_output for r in results])
    outputs = outputs.reshape(len(samples), seeds_per_sample).mean(axis=1)

    # Sobol analysis
    si = sobol_analyze.analyze(SOBOL_PROBLEM, outputs, calc_second_order=True)

    # Save indices
    si_serializable = {
        "S1": si["S1"].tolist(),
        "S1_conf": si["S1_conf"].tolist(),
        "ST": si["ST"].tolist(),
        "ST_conf": si["ST_conf"].tolist(),
        "names": SOBOL_PROBLEM["names"],
    }
    with open(output_dir / f"sobol_indices_{strategy}.json", "w") as f:
        json.dump(si_serializable, f, indent=2)

    print(f"Sobol analysis complete. Results saved to {output_dir}")
    return si
