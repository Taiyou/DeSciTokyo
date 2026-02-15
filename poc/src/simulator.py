"""
Scientific Process Optimization Simulator
==========================================
Main simulation engine that runs the scientific pipeline
under different optimization strategies and collects metrics.
"""

import json
import os
import random
from dataclasses import dataclass, field, asdict

from scientific_process import ProcessStep, create_default_pipeline
from optimizers import (
    Optimizer,
    BaselineOptimizer,
    TOCPDCAOptimizer,
    AISciOpsOptimizer,
)


@dataclass
class TimeStepMetrics:
    """Metrics collected at each time step."""

    time_step: int
    process_throughputs: dict[str, float]
    process_wip: dict[str, float]
    process_backlogs: dict[str, float]
    system_throughput: float  # output of the last process
    cumulative_output: float
    bottleneck_process: str
    bottleneck_throughput: float
    total_rework: float
    total_failures: float


@dataclass
class SimulationResult:
    """Complete results of a simulation run."""

    optimizer_name: str
    total_time_steps: int
    total_output: float
    metrics: list[TimeStepMetrics]
    optimization_actions: list[dict]
    final_state: dict


class Simulator:
    """Runs a scientific pipeline simulation with a given optimizer."""

    def __init__(
        self,
        optimizer: Optimizer,
        total_resources: float = 6.0,
        input_rate: float = 2.0,
        seed: int | None = None,
    ):
        self.optimizer = optimizer
        self.total_resources = total_resources
        self.input_rate = input_rate
        self.pipeline = create_default_pipeline()
        self.metrics: list[TimeStepMetrics] = []
        self.cumulative_output = 0.0

        if seed is not None:
            random.seed(seed)

    def run(self, time_steps: int = 100) -> SimulationResult:
        """Run the simulation for the specified number of time steps."""
        for t in range(time_steps):
            # Let optimizer adjust the pipeline
            self.pipeline = self.optimizer.optimize(
                self.pipeline, t, self.total_resources
            )

            # Feed work into the pipeline
            incoming = self.input_rate
            for i, step in enumerate(self.pipeline):
                output = step.step(incoming)
                incoming = output  # output of one step feeds the next

            # The output of the last step is the system output
            system_output = incoming
            self.cumulative_output += system_output

            # Collect metrics
            bottleneck = min(
                self.pipeline, key=lambda p: p.effective_throughput()
            )
            metrics = TimeStepMetrics(
                time_step=t,
                process_throughputs={
                    p.config.name: round(p.effective_throughput(), 4)
                    for p in self.pipeline
                },
                process_wip={
                    p.config.name: round(p.work_in_progress, 4)
                    for p in self.pipeline
                },
                process_backlogs={
                    p.config.name: round(p.human_review_backlog, 4)
                    for p in self.pipeline
                },
                system_throughput=round(system_output, 4),
                cumulative_output=round(self.cumulative_output, 4),
                bottleneck_process=bottleneck.config.name,
                bottleneck_throughput=round(
                    bottleneck.effective_throughput(), 4
                ),
                total_rework=round(
                    sum(p.rework_units for p in self.pipeline), 4
                ),
                total_failures=round(
                    sum(p.failed_units for p in self.pipeline), 4
                ),
            )
            self.metrics.append(metrics)

        # Compile results
        final_state = {}
        for step in self.pipeline:
            final_state[step.config.name] = {
                "throughput": round(step.effective_throughput(), 4),
                "completed_units": round(step.completed_units, 4),
                "failed_units": round(step.failed_units, 4),
                "rework_units": round(step.rework_units, 4),
                "human_review_backlog": round(step.human_review_backlog, 4),
                "ai_assistance_level": round(step.ai_assistance_level, 4),
                "allocated_resources": round(step.allocated_resources, 4),
            }

        return SimulationResult(
            optimizer_name=self.optimizer.name,
            total_time_steps=len(self.metrics),
            total_output=round(self.cumulative_output, 4),
            metrics=self.metrics,
            optimization_actions=[
                {
                    "time_step": a.time_step,
                    "target": a.target_process,
                    "type": a.action_type,
                    "description": a.description,
                }
                for a in self.optimizer.actions
            ],
            final_state=final_state,
        )


def run_experiment(
    time_steps: int = 100,
    total_resources: float = 6.0,
    input_rate: float = 2.0,
    seed: int = 42,
    output_dir: str = "results",
) -> dict[str, SimulationResult]:
    """Run the complete experiment with all three optimization strategies."""

    optimizers = [
        BaselineOptimizer(),
        TOCPDCAOptimizer(pdca_cycle_length=10),
        AISciOpsOptimizer(),
    ]

    results = {}
    for opt in optimizers:
        print(f"\n{'='*60}")
        print(f"Running: {opt.name}")
        print(f"{'='*60}")

        sim = Simulator(
            optimizer=opt,
            total_resources=total_resources,
            input_rate=input_rate,
            seed=seed,
        )
        result = sim.run(time_steps=time_steps)
        results[opt.name] = result

        print(f"  Total output: {result.total_output:.2f}")
        print(f"  Optimization actions: {len(result.optimization_actions)}")
        print(f"  Final state:")
        for name, state in result.final_state.items():
            print(f"    {name}: throughput={state['throughput']:.2f}, "
                  f"completed={state['completed_units']:.2f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    for name, result in results.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        filepath = os.path.join(output_dir, f"{safe_name}.json")
        with open(filepath, "w") as f:
            json.dump(
                {
                    "optimizer_name": result.optimizer_name,
                    "total_time_steps": result.total_time_steps,
                    "total_output": result.total_output,
                    "metrics": [asdict(m) for m in result.metrics],
                    "optimization_actions": result.optimization_actions,
                    "final_state": result.final_state,
                },
                f,
                indent=2,
            )
        print(f"\nSaved: {filepath}")

    return results


if __name__ == "__main__":
    results = run_experiment(
        time_steps=100,
        total_resources=6.0,
        input_rate=2.0,
        seed=42,
        output_dir=os.path.join(os.path.dirname(__file__), "..", "results"),
    )

    # Print comparison summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Strategy':<45} {'Total Output':>12}")
    print(f"{'-'*57}")
    for name, result in results.items():
        print(f"{name:<45} {result.total_output:>12.2f}")

    baseline_output = results["Baseline (No Optimization)"].total_output
    print(f"\nRelative to Baseline:")
    for name, result in results.items():
        if name != "Baseline (No Optimization)":
            improvement = (
                (result.total_output - baseline_output) / baseline_output * 100
            )
            print(f"  {name}: {improvement:+.1f}%")
