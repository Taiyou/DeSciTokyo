"""
Bottleneck-Persists World Experiment
=====================================
What happens when AI is highly capable but human review is still required?

This models a realistic scenario where:
- AI capabilities are high (same as AI-superior: better error detection, etc.)
- BUT organizational/regulatory/trust requirements mandate human review
- The human review bottleneck mechanism remains active

This contrasts with the AI-Superior world where the bottleneck was removed.

Three worlds compared:
1. Current: Low AI capability + bottleneck exists
2. Bottleneck-Persists: High AI capability + bottleneck exists  (THIS)
3. AI-Superior: High AI capability + bottleneck removed

Key question: How much of the AI-superior world's gains come from
AI capability vs. removing the human bottleneck?
"""

import os
import random
from dataclasses import dataclass

from scientific_process import ProcessConfig, ProcessStep, ProcessState
from optimizers import Optimizer, OptimizationAction, BaselineOptimizer
from advanced_optimizers import KanbanSciOpsOptimizer
from management_overhead import OverheadProfile, get_overhead_profile
from simulator import Simulator, SimulationResult, TimeStepMetrics
from meta_overhead_optimizer import (
    MetaAIOracleOptimizer,
    MetaAINoisyOptimizer,
    MetaAIDelayedOptimizer,
    MetaAIRecursiveOptimizer,
    MetaAITrustDecayOptimizer,
)


# ============================================================
# ProcessStep: AI is capable, but human bottleneck remains
# ============================================================

class BottleneckPersistsProcessStep(ProcessStep):
    """ProcessStep where AI is highly capable but human review is still required.

    Key differences from current world:
    - AI uncertainty reduction: 85% (was 50%)
    - AI failure reduction: 70% (was 30%)
    - AI boost coefficient: 2.5 (was 2.0)

    Key differences from AI-superior world:
    - Human review bottleneck: REMAINS (was removed)
    - Human review backlog: accumulates normally (was 90% reduced)
    - human_review_needed: moderate (0.15-0.50), not minimal (0.02-0.15)
    """

    def effective_throughput(self) -> float:
        """High AI capability BUT human review bottleneck persists."""
        base = self.throughput * self.allocated_resources

        # AI boost is stronger (same as AI-superior: 2.5x)
        ai_boost = 1.0 + (
            self.ai_assistance_level * self.config.ai_automatable * 2.5
        )
        effective = base * ai_boost

        # KEY: Human review bottleneck REMAINS
        # Same mechanism as current world — human review gates throughput
        if self.ai_assistance_level > 0.5 and self.config.human_review_needed > 0.3:
            review_bottleneck = 1.0 - (
                self.config.human_review_needed
                * self.ai_assistance_level
                * 0.5
            )
            effective *= max(0.2, review_bottleneck)

        return max(self.config.min_throughput, min(effective, self.config.max_throughput))

    def step(self, incoming_work: float) -> float:
        """High AI capability for error/failure reduction, but review backlog persists."""
        self.work_in_progress += incoming_work

        if self.work_in_progress <= 0:
            self.state = ProcessState.IDLE
            self.cumulative_wait_time += 1
            return 0.0

        self.state = ProcessState.RUNNING
        capacity = self.effective_throughput()
        processable = min(self.work_in_progress, capacity)

        # Uncertainty: AI reduces by 85% (same as AI-superior, was 50%)
        rework_fraction = random.random()
        if rework_fraction < self.config.uncertainty * (1 - self.ai_assistance_level * 0.85):
            rework = processable * 0.3
            self.rework_units += rework
            self.work_in_progress += rework
            processable *= 0.7

        # Failure: AI reduces by 70% (same as AI-superior, was 30%)
        if random.random() < self.config.failure_rate * (1 - self.ai_assistance_level * 0.7):
            failed = processable * 0.1
            self.failed_units += failed
            processable -= failed

        # Human review backlog: STILL ACCUMULATES (unlike AI-superior)
        # Same mechanism as current world
        review_needed = processable * self.config.human_review_needed
        if self.ai_assistance_level > 0.3:
            self.human_review_backlog += review_needed * self.ai_assistance_level
            reviewed = min(self.human_review_backlog, capacity * 0.3)
            self.human_review_backlog -= reviewed

        self.work_in_progress -= min(processable + review_needed, self.work_in_progress)
        self.completed_units += processable
        return processable


def create_bottleneck_persists_pipeline() -> list[ProcessStep]:
    """Pipeline where AI is capable but human review is still mandated.

    ai_automatable: same as AI-superior (high)
    human_review_needed: REDUCED from current but NOT eliminated
      - Current: 0.2-0.8 → Bottleneck-persists: 0.15-0.50
      - AI-superior had: 0.02-0.15
    """
    configs = [
        ProcessConfig(
            name="Survey",
            base_throughput=2.0,
            uncertainty=0.2,
            failure_rate=0.05,
            resource_cost=1.0,
            ai_automatable=0.95,      # same as AI-superior
            human_review_needed=0.15,  # reduced from 0.2, but > AI-sup's 0.05
        ),
        ProcessConfig(
            name="Hypothesis",
            base_throughput=1.5,
            uncertainty=0.4,
            failure_rate=0.1,
            resource_cost=1.5,
            ai_automatable=0.90,      # same as AI-superior
            human_review_needed=0.35,  # reduced from 0.5, but > AI-sup's 0.10
        ),
        ProcessConfig(
            name="Experiment",
            base_throughput=0.8,
            uncertainty=0.5,
            failure_rate=0.15,
            resource_cost=3.0,
            ai_automatable=0.70,      # same as AI-superior
            human_review_needed=0.25,  # reduced from 0.3, but > AI-sup's 0.15
        ),
        ProcessConfig(
            name="Analysis",
            base_throughput=1.8,
            uncertainty=0.3,
            failure_rate=0.08,
            resource_cost=2.0,
            ai_automatable=0.98,      # same as AI-superior
            human_review_needed=0.30,  # reduced from 0.4, but > AI-sup's 0.02
        ),
        ProcessConfig(
            name="Writing",
            base_throughput=1.2,
            uncertainty=0.3,
            failure_rate=0.05,
            resource_cost=1.5,
            ai_automatable=0.95,      # same as AI-superior
            human_review_needed=0.40,  # reduced from 0.6, but > AI-sup's 0.05
        ),
        ProcessConfig(
            name="Review",
            base_throughput=0.6,
            uncertainty=0.2,
            failure_rate=0.1,
            resource_cost=1.0,
            ai_automatable=0.85,      # same as AI-superior
            human_review_needed=0.50,  # reduced from 0.8, but > AI-sup's 0.10
        ),
    ]
    return [BottleneckPersistsProcessStep(config=c) for c in configs]


# ============================================================
# Simulator for bottleneck-persists world
# ============================================================

class BottleneckPersistsSimulator(Simulator):
    """Simulator using bottleneck-persists process steps."""

    def __init__(self, optimizer, total_resources=6.0, input_rate=2.0, seed=None):
        self.optimizer = optimizer
        self.total_resources = total_resources
        self.input_rate = input_rate
        self.pipeline = create_bottleneck_persists_pipeline()
        self.metrics: list[TimeStepMetrics] = []
        self.cumulative_output = 0.0
        self.overhead_per_step: list[float] = []
        self.meta_cost_per_step: list[float] = []
        self.total_overhead = 0.0
        self.total_meta_cost = 0.0
        self.trust_per_step: list[float] = []
        self.intensity_per_step: list[float] = []

        if seed is not None:
            random.seed(seed)

    def run(self, time_steps: int = 100) -> SimulationResult:
        prev_action_count = 0

        for t in range(time_steps):
            # Get overhead profile
            if hasattr(self.optimizer, 'get_overhead_profile'):
                profile = self.optimizer.get_overhead_profile()
                meta_cost = self.optimizer.get_meta_cost()
            else:
                profile = get_overhead_profile(self.optimizer.name)
                meta_cost = 0.0

            current_action_count = len(self.optimizer.actions)
            new_actions = current_action_count - prev_action_count
            prev_action_count = current_action_count

            is_transition = any(
                a.action_type.endswith("-Init") or a.action_type.endswith("-Transition")
                for a in self.optimizer.actions[-new_actions:]
            ) if new_actions > 0 else False

            overhead = profile.compute_overhead(
                time_step=t, num_processes=len(self.pipeline),
                num_actions_this_step=new_actions, is_stage_transition=is_transition,
            )

            self.overhead_per_step.append(overhead)
            self.meta_cost_per_step.append(meta_cost)
            self.total_overhead += overhead
            self.total_meta_cost += meta_cost

            if hasattr(self.optimizer, 'state') and hasattr(self.optimizer.state, 'human_trust'):
                self.trust_per_step.append(self.optimizer.state.human_trust)
            else:
                self.trust_per_step.append(1.0)

            if hasattr(self.optimizer, 'optimization_intensity'):
                self.intensity_per_step.append(self.optimizer.optimization_intensity)
            else:
                self.intensity_per_step.append(0.0)

            effective_resources = max(0.5, self.total_resources - overhead - meta_cost)
            self.pipeline = self.optimizer.optimize(self.pipeline, t, effective_resources)

            incoming = self.input_rate
            for step in self.pipeline:
                output = step.step(incoming)
                incoming = output

            system_output = incoming
            self.cumulative_output += system_output

            bottleneck = min(self.pipeline, key=lambda p: p.effective_throughput())
            metrics = TimeStepMetrics(
                time_step=t,
                process_throughputs={p.config.name: round(p.effective_throughput(), 4) for p in self.pipeline},
                process_wip={p.config.name: round(p.work_in_progress, 4) for p in self.pipeline},
                process_backlogs={p.config.name: round(p.human_review_backlog, 4) for p in self.pipeline},
                system_throughput=round(system_output, 4),
                cumulative_output=round(self.cumulative_output, 4),
                bottleneck_process=bottleneck.config.name,
                bottleneck_throughput=round(bottleneck.effective_throughput(), 4),
                total_rework=round(sum(p.rework_units for p in self.pipeline), 4),
                total_failures=round(sum(p.failed_units for p in self.pipeline), 4),
            )
            self.metrics.append(metrics)

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
                {"time_step": a.time_step, "target": a.target_process,
                 "type": a.action_type, "description": a.description}
                for a in self.optimizer.actions
            ],
            final_state=final_state,
        )


# ============================================================
# Optimizers adapted for bottleneck-persists world
# ============================================================

class BNPersistsTrustDecayOptimizer(MetaAITrustDecayOptimizer):
    """TrustDecay in bottleneck-persists world.

    The trust dynamics are MORE impactful here because:
    - AI is capable, so aggressive coordination cuts seem justified
    - But human review is still required, so trust loss DOES degrade quality
    - The tension between AI capability and review requirement is maximal
    """
    pass  # Same TrustDecay logic; the pipeline difference drives the results


class BNPersistsOracleOptimizer(MetaAIOracleOptimizer):
    """Oracle in bottleneck-persists world.

    With perfect info, the Oracle knows the bottleneck persists and
    optimizes within that constraint. It CANNOT remove the bottleneck
    but can minimize overhead around it.
    """

    def _adjust_overhead(self, pipeline, time_step):
        """Oracle recognizes human review is mandatory — optimizes around it."""
        p = self.state.current_profile

        # Can reduce some human coordination but NOT eliminate it
        # (review is mandated, so coordination for it must remain)
        if time_step >= 20:
            # Reduce less aggressively than AI-superior Oracle
            p.human_coordination_cost = max(0.02, p.human_coordination_cost * 0.90)
        if time_step >= 40:
            p.complexity_scaling = max(0.05, p.complexity_scaling * 0.85)
        if time_step >= 60:
            p.learning_cost = max(0.05, p.learning_cost * 0.75)

        p.ai_infrastructure_cost = max(0.08, p.ai_infrastructure_cost * 0.95)

        self.actions.append(OptimizationAction(
            time_step=time_step, target_process="meta",
            action_type="MetaAI-BNPersistsAdjust",
            description=f"BN-persists Oracle OH: human={p.human_coordination_cost:.3f}, "
            f"base={p.base_cost:.3f}, ai_infra={p.ai_infrastructure_cost:.3f}",
        ))


class BNPersistsRecursiveOptimizer(MetaAIRecursiveOptimizer):
    """Recursive cost in bottleneck-persists world."""
    pass  # Same recursive dynamics; bottleneck constrains the gains


# ============================================================
# Run functions
# ============================================================

def run_bottleneck_persists_world(seed=42):
    """Run all variants in bottleneck-persists world."""
    from run_ai_superior import AISuperiorSimulator

    optimizers = [
        ("Baseline (BN-Per)", BaselineOptimizer(), BottleneckPersistsSimulator),
        ("Kanban (BN-Per)", KanbanSciOpsOptimizer(), BottleneckPersistsSimulator),
        ("Oracle (BN-Per)", BNPersistsOracleOptimizer(), BottleneckPersistsSimulator),
        ("TrustDecay (BN-Per)", BNPersistsTrustDecayOptimizer(), BottleneckPersistsSimulator),
        ("Recursive (BN-Per)", BNPersistsRecursiveOptimizer(), BottleneckPersistsSimulator),
        ("Noisy (BN-Per)", MetaAINoisyOptimizer(), BottleneckPersistsSimulator),
        ("Delayed (BN-Per)", MetaAIDelayedOptimizer(), BottleneckPersistsSimulator),
    ]

    results = {}
    sims = {}
    for label, opt, SimClass in optimizers:
        random.seed(seed)
        sim = SimClass(optimizer=opt, seed=seed)
        result = sim.run(time_steps=100)
        results[label] = result
        sims[label] = sim

    return results, sims


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT: Bottleneck-Persists World")
    print("  AI is capable but human review is still mandated")
    print("=" * 60)

    results, sims = run_bottleneck_persists_world(seed=42)

    print(f"\n{'Variant':<25} {'Output':>8} {'MgmtOH':>8} {'MetaOH':>8} {'TotalOH':>8}")
    print(f"{'-'*65}")
    for k in sorted(results.keys(), key=lambda x: results[x].total_output, reverse=True):
        r = results[k]
        sim = sims[k]
        meta = sim.total_meta_cost if hasattr(sim, 'total_meta_cost') else 0.0
        print(f"{k:<25} {r.total_output:>8.1f} {sim.total_overhead:>8.1f} "
              f"{meta:>8.1f} {sim.total_overhead + meta:>8.1f}")
