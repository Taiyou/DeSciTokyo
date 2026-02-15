"""
AI-Superior World Experiment
==============================
What happens when AI is fundamentally more trustworthy than humans?

Current model assumptions that change:
1. Human review bottleneck (ProcessStep.effective_throughput line 71-77)
   → DISAPPEARS: AI can self-verify, no human review needed
2. ai_automatable: 0.3-0.9 → 0.85-0.99 (almost everything automatable)
3. human_review_needed: 0.2-0.8 → 0.0-0.15 (minimal human oversight)
4. AI reduces uncertainty by 50% max → 85% (AI catches more errors)
5. AI reduces failure by 30% max → 70% (AI prevents more failures)

This creates a fundamentally different optimization landscape where:
- The human-AI coordination problem vanishes
- Overhead from human management becomes pure waste
- The recursive cost and observability challenges remain
- But trust decay becomes irrelevant
"""

import os
import random
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
# Modified ProcessStep for AI-Superior world
# ============================================================

class AIAsGoodProcessStep(ProcessStep):
    """ProcessStep where AI is as capable as humans — no review bottleneck."""

    def effective_throughput(self) -> float:
        """AI-superior version: no human review bottleneck."""
        base = self.throughput * self.allocated_resources

        # AI boost is stronger (AI is more capable)
        ai_boost = 1.0 + (
            self.ai_assistance_level * self.config.ai_automatable * 2.5  # was 2.0
        )
        effective = base * ai_boost

        # KEY CHANGE: Human review bottleneck is REMOVED
        # AI can self-verify its outputs; no human gating needed
        # (The original bottleneck was: effective *= max(0.2, review_bottleneck))

        return max(self.config.min_throughput, min(effective, self.config.max_throughput))

    def step(self, incoming_work: float) -> float:
        """AI-superior: stronger uncertainty/failure reduction, no review backlog."""
        self.work_in_progress += incoming_work

        if self.work_in_progress <= 0:
            self.state = ProcessState.IDLE
            self.cumulative_wait_time += 1
            return 0.0

        self.state = ProcessState.RUNNING
        capacity = self.effective_throughput()
        processable = min(self.work_in_progress, capacity)

        # Uncertainty: AI reduces by 85% (was 50%)
        rework_fraction = random.random()
        if rework_fraction < self.config.uncertainty * (1 - self.ai_assistance_level * 0.85):
            rework = processable * 0.3
            self.rework_units += rework
            self.work_in_progress += rework
            processable *= 0.7

        # Failure: AI reduces by 70% (was 30%)
        if random.random() < self.config.failure_rate * (1 - self.ai_assistance_level * 0.7):
            failed = processable * 0.1
            self.failed_units += failed
            processable -= failed

        # Human review backlog: MINIMAL
        # AI reviews its own work; tiny fraction still goes to human for audit
        review_needed = processable * self.config.human_review_needed * 0.1  # 90% reduction
        if self.ai_assistance_level > 0.3:
            self.human_review_backlog += review_needed * 0.2  # Minimal accumulation
            reviewed = min(self.human_review_backlog, capacity * 0.5)  # Fast processing
            self.human_review_backlog -= reviewed

        self.work_in_progress -= min(processable + review_needed, self.work_in_progress)
        self.completed_units += processable
        return processable


def create_ai_superior_pipeline() -> list[ProcessStep]:
    """Pipeline where AI capability exceeds human capability.

    Key differences from default:
    - ai_automatable: much higher across the board
    - human_review_needed: dramatically reduced
    - Experiment still has physical-world constraints (can't automate physics)
    """
    configs = [
        ProcessConfig(
            name="Survey",
            base_throughput=2.0,
            uncertainty=0.2,
            failure_rate=0.05,
            resource_cost=1.0,
            ai_automatable=0.95,      # was 0.8
            human_review_needed=0.05,  # was 0.2
        ),
        ProcessConfig(
            name="Hypothesis",
            base_throughput=1.5,
            uncertainty=0.4,
            failure_rate=0.1,
            resource_cost=1.5,
            ai_automatable=0.90,      # was 0.6
            human_review_needed=0.10,  # was 0.5
        ),
        ProcessConfig(
            name="Experiment",
            base_throughput=0.8,
            uncertainty=0.5,
            failure_rate=0.15,
            resource_cost=3.0,
            ai_automatable=0.70,      # was 0.3 — still needs physical world
            human_review_needed=0.15,  # was 0.3 — safety oversight
        ),
        ProcessConfig(
            name="Analysis",
            base_throughput=1.8,
            uncertainty=0.3,
            failure_rate=0.08,
            resource_cost=2.0,
            ai_automatable=0.98,      # was 0.9
            human_review_needed=0.02,  # was 0.4
        ),
        ProcessConfig(
            name="Writing",
            base_throughput=1.2,
            uncertainty=0.3,
            failure_rate=0.05,
            resource_cost=1.5,
            ai_automatable=0.95,      # was 0.7
            human_review_needed=0.05,  # was 0.6
        ),
        ProcessConfig(
            name="Review",
            base_throughput=0.6,
            uncertainty=0.2,
            failure_rate=0.1,
            resource_cost=1.0,
            ai_automatable=0.85,      # was 0.4
            human_review_needed=0.10,  # was 0.8
        ),
    ]
    return [AIAsGoodProcessStep(config=c) for c in configs]


# ============================================================
# AI-Superior Optimizers (same logic, but pipeline uses AI-superior steps)
# ============================================================

class AISuperiorKanbanOptimizer(KanbanSciOpsOptimizer):
    """Kanban with AI-superior pipeline understanding."""
    pass  # Same logic, different pipeline


class AISuperiorOracleOptimizer(MetaAIOracleOptimizer):
    """Oracle meta-AI in AI-superior world."""

    def _adjust_overhead(self, pipeline, time_step):
        """In AI-superior world, can reduce human overhead much more aggressively."""
        p = self.state.current_profile

        # Human coordination is almost unnecessary
        if time_step >= 10:
            p.human_coordination_cost = max(0.005, p.human_coordination_cost * 0.7)
        if time_step >= 20:
            p.base_cost = max(0.02, p.base_cost * 0.8)
        if time_step >= 30:
            p.complexity_scaling = max(0.03, p.complexity_scaling * 0.7)
        if time_step >= 40:
            p.learning_cost = max(0.03, p.learning_cost * 0.6)

        # AI infra cost is the ONLY significant remaining cost
        # And it has value — don't reduce it below useful level
        p.ai_infrastructure_cost = max(0.10, p.ai_infrastructure_cost * 0.97)

        self.actions.append(OptimizationAction(
            time_step=time_step, target_process="meta",
            action_type="MetaAI-SuperiorAdjust",
            description=f"AI-superior OH: human={p.human_coordination_cost:.4f}, "
            f"base={p.base_cost:.3f}, ai_infra={p.ai_infrastructure_cost:.3f}",
        ))


class AISuperiorTrustDecayOptimizer(MetaAITrustDecayOptimizer):
    """Trust decay in AI-superior world.

    Key difference: when trust in HUMANS decays, it doesn't matter
    because AI can compensate. The trust dynamic is reversed.
    """

    def optimize(
        self, pipeline: list[ProcessStep], time_step: int, total_resources: float
    ) -> list[ProcessStep]:
        for step in pipeline:
            step.throughput = step.config.base_throughput

        state = {
            "time_step": time_step,
            "throughputs": {p.config.name: p.effective_throughput() for p in pipeline},
            "wip": {p.config.name: p.work_in_progress for p in pipeline},
            "trust": self.state.human_trust,
        }
        self.history.append(state)

        # KEY CHANGE: Trust decay does NOT penalize pipeline quality
        # because AI compensates for reduced human coordination
        # (In the original, low trust increased uncertainty/failure)
        # Here: AI handles it, so no penalty applied

        # Meta-optimization: same aggressive reduction
        if time_step > 5 and time_step % 10 == 0:
            self._trust_aware_adjust(pipeline, time_step)

        # Trust still decays (humans notice they're being sidelined)
        p = self.state.current_profile
        coord_reduction = 1.0 - (p.human_coordination_cost / max(0.01, self.original_human_coord))
        if coord_reduction > 0.3:
            self.state.human_trust = max(0.3, self.state.human_trust - coord_reduction * 0.03)
        if coord_reduction < 0.1:
            self.state.human_trust = min(1.0, self.state.human_trust + 0.005)

        # But AI doesn't need to detect or fix trust — it's irrelevant
        # (No trust_visible check needed)

        # Base optimization
        progress = time_step / 100.0
        for step in pipeline:
            step.ai_assistance_level = min(
                0.95,  # Higher max (was 0.85)
                step.config.ai_automatable * (0.5 + progress * 0.50)
            )

        weights = {}
        for step in pipeline:
            wip_ratio = (step.work_in_progress + 0.1) / (step.effective_throughput() + 0.1)
            weights[step.config.name] = max(0.5, wip_ratio)
        total_weight = sum(weights.values())
        for step in pipeline:
            step.allocated_resources = (
                weights[step.config.name] / total_weight
            ) * total_resources

        if time_step == 30:
            for step in pipeline:
                if step.ai_assistance_level > 0.5:
                    step.config.uncertainty *= 0.7
        if time_step == 50:
            for step in pipeline:
                if step.config.name in ("Survey", "Hypothesis"):
                    step.config.base_throughput *= 1.3
                    step.throughput = step.config.base_throughput
                elif step.config.name in ("Review", "Writing"):
                    step.config.human_review_needed *= 0.5

        return pipeline


class AISuperiorRecursiveOptimizer(MetaAIRecursiveOptimizer):
    """Recursive cost in AI-superior world.

    The recursive cost problem PERSISTS even when AI is superior,
    because the meta-optimization cost is about AI compute, not humans.
    """

    def _recursive_adjust(self, pipeline, time_step):
        """Same recursive dynamics, but human costs are removed faster."""
        p = self.state.current_profile

        reduction_factor = 1.0 - self.optimization_intensity * 0.20  # Stronger (was 0.15)

        # Human costs drop much faster
        if time_step >= 10:
            p.human_coordination_cost = max(0.005, p.human_coordination_cost * 0.6)
        if time_step >= 20:
            p.base_cost = max(0.02, p.base_cost * reduction_factor)
        if time_step >= 30:
            p.complexity_scaling = max(0.03, p.complexity_scaling * reduction_factor)
        if time_step >= 40:
            p.learning_cost = max(0.03, p.learning_cost * reduction_factor)

        # BUT: AI infrastructure cost remains and meta-cost still grows
        meta_cost = self.get_meta_cost()
        current_oh = p.compute_overhead(time_step, len(pipeline), 1)

        if meta_cost_up := self.base_meta_cost * (1.0 + min(1.0, self.optimization_intensity + 0.05) ** 2 * 3.0):
            if meta_cost_up < current_oh * 0.8:
                self.optimization_intensity = min(1.0, self.optimization_intensity + 0.05)
            elif meta_cost > current_oh * 0.5:
                self.optimization_intensity = max(0.1, self.optimization_intensity - 0.08)

        self.actions.append(OptimizationAction(
            time_step=time_step, target_process="meta",
            action_type="MetaAI-SuperiorRecursive",
            description=f"intensity={self.optimization_intensity:.2f}, "
            f"meta_cost={self.get_meta_cost():.3f}, mgmt_oh={current_oh:.3f}",
        ))


# ============================================================
# Simulator that uses AI-superior pipeline
# ============================================================

class AISuperiorSimulator(Simulator):
    """Simulator using AI-superior process steps."""

    def __init__(self, optimizer, total_resources=6.0, input_rate=2.0, seed=None):
        # Don't call super().__init__ because it creates default pipeline
        self.optimizer = optimizer
        self.total_resources = total_resources
        self.input_rate = input_rate
        self.pipeline = create_ai_superior_pipeline()  # KEY CHANGE
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
# Run both worlds
# ============================================================

def run_current_world(seed=42):
    """Rerun from run_meta_overhead for comparison."""
    from run_meta_overhead import run_all
    return run_all(seed=seed)


def run_ai_superior_world(seed=42):
    """Run all variants in AI-superior world."""
    optimizers = [
        ("Baseline (AI-Sup)", BaselineOptimizer(), AISuperiorSimulator),
        ("Kanban (AI-Sup)", KanbanSciOpsOptimizer(), AISuperiorSimulator),
        ("Oracle (AI-Sup)", AISuperiorOracleOptimizer(), AISuperiorSimulator),
        ("TrustDecay (AI-Sup)", AISuperiorTrustDecayOptimizer(), AISuperiorSimulator),
        ("Recursive (AI-Sup)", AISuperiorRecursiveOptimizer(), AISuperiorSimulator),
        ("Noisy (AI-Sup)", MetaAINoisyOptimizer(), AISuperiorSimulator),
        ("Delayed (AI-Sup)", MetaAIDelayedOptimizer(), AISuperiorSimulator),
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


def print_comparison(current_results, current_sims, superior_results, superior_sims):
    """Print side-by-side comparison."""

    print(f"\n{'='*120}")
    print("WORLD COMPARISON: Current (Human Bottleneck) vs. AI-Superior (No Human Bottleneck)")
    print(f"{'='*120}")

    # Current world
    print(f"\n--- CURRENT WORLD (Human review bottleneck exists) ---")
    print(f"{'Variant':<35} {'Output':>8} {'MgmtOH':>8} {'MetaOH':>8} {'TotalOH':>8}")
    print(f"{'-'*75}")
    for k in sorted(current_results.keys(), key=lambda x: current_results[x].total_output, reverse=True):
        r = current_results[k]
        sim = current_sims[k]
        meta = sim.total_meta_cost if hasattr(sim, 'total_meta_cost') else 0.0
        print(f"{k:<35} {r.total_output:>8.1f} {sim.total_overhead:>8.1f} {meta:>8.1f} {sim.total_overhead + meta:>8.1f}")

    # AI-superior world
    print(f"\n--- AI-SUPERIOR WORLD (AI self-verifies, no human bottleneck) ---")
    print(f"{'Variant':<35} {'Output':>8} {'MgmtOH':>8} {'MetaOH':>8} {'TotalOH':>8}")
    print(f"{'-'*75}")
    for k in sorted(superior_results.keys(), key=lambda x: superior_results[x].total_output, reverse=True):
        r = superior_results[k]
        sim = superior_sims[k]
        meta = sim.total_meta_cost if hasattr(sim, 'total_meta_cost') else 0.0
        print(f"{k:<35} {r.total_output:>8.1f} {sim.total_overhead:>8.1f} {meta:>8.1f} {sim.total_overhead + meta:>8.1f}")

    # Key comparisons
    print(f"\n--- KEY INSIGHTS ---")

    curr_baseline = current_results["Baseline"].total_output
    sup_baseline = superior_results["Baseline (AI-Sup)"].total_output
    print(f"  Baseline improvement (AI-Sup vs Current): {curr_baseline:.1f} → {sup_baseline:.1f} "
          f"({(sup_baseline - curr_baseline) / curr_baseline * 100:+.1f}%)")

    curr_best = max(current_results.values(), key=lambda x: x.total_output)
    sup_best = max(superior_results.values(), key=lambda x: x.total_output)
    print(f"  Best strategy improvement: {curr_best.total_output:.1f} → {sup_best.total_output:.1f} "
          f"({(sup_best.total_output - curr_best.total_output) / curr_best.total_output * 100:+.1f}%)")

    # Which challenges remain?
    print(f"\n--- CHALLENGE PERSISTENCE ---")
    sup_oracle = superior_results.get("Oracle (AI-Sup)")
    sup_noisy = superior_results.get("Noisy (AI-Sup)")
    sup_delayed = superior_results.get("Delayed (AI-Sup)")
    sup_recursive = superior_results.get("Recursive (AI-Sup)")
    sup_trust = superior_results.get("TrustDecay (AI-Sup)")

    if sup_oracle and sup_noisy:
        gap = sup_oracle.total_output - sup_noisy.total_output
        print(f"  Noisy vs Oracle gap: {gap:.1f} ({'PERSISTS' if gap > 2 else 'REDUCED' if gap > 0.5 else 'RESOLVED'})")
    if sup_oracle and sup_delayed:
        gap = sup_oracle.total_output - sup_delayed.total_output
        print(f"  Delayed vs Oracle gap: {gap:.1f} ({'PERSISTS' if gap > 2 else 'REDUCED' if gap > 0.5 else 'RESOLVED'})")
    if sup_oracle and sup_recursive:
        gap = sup_oracle.total_output - sup_recursive.total_output
        rec_sim = superior_sims.get("Recursive (AI-Sup)")
        meta_oh = rec_sim.total_meta_cost if rec_sim else 0
        print(f"  Recursive vs Oracle gap: {gap:.1f}, meta-OH: {meta_oh:.1f} "
              f"({'PERSISTS' if gap > 2 else 'REDUCED' if gap > 0.5 else 'RESOLVED'})")
    if sup_oracle and sup_trust:
        gap = sup_oracle.total_output - sup_trust.total_output
        print(f"  TrustDecay vs Oracle gap: {gap:.1f} ({'PERSISTS' if gap > 2 else 'REDUCED' if gap > 0.5 else 'RESOLVED'})")


def generate_figures(current_results, current_sims, superior_results, superior_sims, output_dir):
    """Generate comparison figures."""
    os.makedirs(output_dir, exist_ok=True)

    COLORS_CURR = {
        "Baseline": "#bdc3c7",
        "Kanban (Fixed OH)": "#2ecc71",
        "MetaAI-Oracle": "#3498db",
        "MetaAI-Noisy": "#e67e22",
        "MetaAI-Delayed": "#9b59b6",
        "MetaAI-Recursive": "#e74c3c",
        "MetaAI-TrustDecay": "#1abc9c",
    }
    COLORS_SUP = {
        "Baseline (AI-Sup)": "#bdc3c7",
        "Kanban (AI-Sup)": "#2ecc71",
        "Oracle (AI-Sup)": "#3498db",
        "Noisy (AI-Sup)": "#e67e22",
        "Delayed (AI-Sup)": "#9b59b6",
        "Recursive (AI-Sup)": "#e74c3c",
        "TrustDecay (AI-Sup)": "#1abc9c",
    }

    # === Figure 1: Side-by-side bar chart ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Current world
    curr_order = sorted(current_results.keys(),
                        key=lambda k: current_results[k].total_output, reverse=True)
    x1 = np.arange(len(curr_order))
    vals1 = [current_results[k].total_output for k in curr_order]
    cols1 = [COLORS_CURR.get(k, "gray") for k in curr_order]

    ax1.bar(x1, vals1, color=cols1, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(vals1):
        ax1.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels([k.replace("MetaAI-", "Meta:\n") for k in curr_order], fontsize=7)
    ax1.set_ylabel("Total Research Output")
    ax1.set_title("Current World\n(Human Review Bottleneck)", fontsize=12)
    ax1.set_ylim(0, max(vals1) * 1.3)
    ax1.grid(True, axis="y", alpha=0.3)

    # AI-superior world
    sup_order = sorted(superior_results.keys(),
                       key=lambda k: superior_results[k].total_output, reverse=True)
    x2 = np.arange(len(sup_order))
    vals2 = [superior_results[k].total_output for k in sup_order]
    cols2 = [COLORS_SUP.get(k, "gray") for k in sup_order]

    ax2.bar(x2, vals2, color=cols2, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(vals2):
        ax2.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([k.replace(" (AI-Sup)", "\n(AI-Sup)") for k in sup_order], fontsize=7)
    ax2.set_ylabel("Total Research Output")
    ax2.set_title("AI-Superior World\n(No Human Bottleneck)", fontsize=12)
    ax2.set_ylim(0, max(vals2) * 1.3)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("How AI Superiority Changes the Optimization Landscape", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v5_01_world_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Figure 2: Challenge persistence heatmap ===
    fig, ax = plt.subplots(figsize=(12, 6))

    challenges = ["Noisy\n(Observability)", "Delayed\n(Feedback)", "Recursive\n(Self-Ref.)", "TrustDecay\n(Human)"]
    curr_challenge_names = ["MetaAI-Noisy", "MetaAI-Delayed", "MetaAI-Recursive", "MetaAI-TrustDecay"]
    sup_challenge_names = ["Noisy (AI-Sup)", "Delayed (AI-Sup)", "Recursive (AI-Sup)", "TrustDecay (AI-Sup)"]

    curr_oracle = current_results.get("MetaAI-Oracle")
    sup_oracle = superior_results.get("Oracle (AI-Sup)")

    if curr_oracle and sup_oracle:
        curr_gaps = []
        sup_gaps = []
        for cn, sn in zip(curr_challenge_names, sup_challenge_names):
            if cn in current_results:
                curr_gaps.append(curr_oracle.total_output - current_results[cn].total_output)
            else:
                curr_gaps.append(0)
            if sn in superior_results:
                sup_gaps.append(sup_oracle.total_output - superior_results[sn].total_output)
            else:
                sup_gaps.append(0)

        x = np.arange(len(challenges))
        width = 0.35

        bars1 = ax.bar(x - width/2, curr_gaps, width, label="Current World",
                        color="#e74c3c", alpha=0.7, edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width/2, sup_gaps, width, label="AI-Superior World",
                        color="#3498db", alpha=0.7, edgecolor="black", linewidth=0.5)

        for b, v in zip(bars1, curr_gaps):
            ax.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.1f}",
                    ha="center", fontsize=9, fontweight="bold")
        for b, v in zip(bars2, sup_gaps):
            ax.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.1f}",
                    ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(challenges, fontsize=10)
        ax.set_ylabel("Performance Gap vs Oracle", fontsize=12)
        ax.set_title("Challenge Persistence: Which Problems Survive AI Superiority?", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(y=2.0, color="red", linestyle=":", alpha=0.5, label="Significance threshold")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v5_02_challenge_persistence.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Figure 3: Cumulative output comparison ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for k in curr_order:
        data = current_results[k]
        steps = [m.time_step for m in data.metrics]
        cumul = [m.cumulative_output for m in data.metrics]
        lw = 2.5 if "Oracle" in k or "TrustDecay" in k else 1.2
        ax1.plot(steps, cumul, label=k.replace("MetaAI-", ""),
                 color=COLORS_CURR.get(k, "gray"), linewidth=lw)

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Cumulative Output")
    ax1.set_title("Current World")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)

    for k in sup_order:
        data = superior_results[k]
        steps = [m.time_step for m in data.metrics]
        cumul = [m.cumulative_output for m in data.metrics]
        lw = 2.5 if "Oracle" in k or "TrustDecay" in k else 1.2
        ax2.plot(steps, cumul, label=k.replace(" (AI-Sup)", ""),
                 color=COLORS_SUP.get(k, "gray"), linewidth=lw)

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Cumulative Output")
    ax2.set_title("AI-Superior World")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Cumulative Output Trajectories: Two Worlds", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v5_03_cumulative_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Figure 4: Overhead efficiency scatter (both worlds) ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Current world
    curr_baseline_out = current_results["Baseline"].total_output
    curr_non_bl = [k for k in curr_order if k != "Baseline"]
    curr_gains = [current_results[k].total_output - curr_baseline_out for k in curr_non_bl]
    curr_ohs = [
        current_sims[k].total_overhead + (current_sims[k].total_meta_cost if hasattr(current_sims[k], 'total_meta_cost') else 0)
        for k in curr_non_bl
    ]
    for i, k in enumerate(curr_non_bl):
        ax1.scatter(curr_ohs[i], curr_gains[i], c=COLORS_CURR.get(k, "gray"),
                    s=180, edgecolors="black", linewidths=1, zorder=5)
        ax1.annotate(k.replace("MetaAI-", ""), (curr_ohs[i], curr_gains[i]),
                     textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax1.set_xlabel("Total Overhead")
    ax1.set_ylabel("Output Gain vs Baseline")
    ax1.set_title("Current World: Efficiency")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linewidth=0.5)

    # AI-superior world
    sup_baseline_out = superior_results["Baseline (AI-Sup)"].total_output
    sup_non_bl = [k for k in sup_order if "Baseline" not in k]
    sup_gains = [superior_results[k].total_output - sup_baseline_out for k in sup_non_bl]
    sup_ohs = [
        superior_sims[k].total_overhead + (superior_sims[k].total_meta_cost if hasattr(superior_sims[k], 'total_meta_cost') else 0)
        for k in sup_non_bl
    ]
    for i, k in enumerate(sup_non_bl):
        ax2.scatter(sup_ohs[i], sup_gains[i], c=COLORS_SUP.get(k, "gray"),
                    s=180, edgecolors="black", linewidths=1, zorder=5)
        ax2.annotate(k.replace(" (AI-Sup)", ""), (sup_ohs[i], sup_gains[i]),
                     textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax2.set_xlabel("Total Overhead")
    ax2.set_ylabel("Output Gain vs Baseline")
    ax2.set_title("AI-Superior World: Efficiency")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linewidth=0.5)

    fig.suptitle("Efficiency Frontiers: Current vs. AI-Superior World", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v5_04_efficiency_both_worlds.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === Figure 5: Structure of remaining challenges ===
    fig, ax = plt.subplots(figsize=(14, 7))

    # Show what percentage of the original gap remains in AI-superior world
    if curr_oracle and sup_oracle:
        labels = ["Noisy\n(Observability)", "Delayed\n(Feedback Lag)",
                  "Recursive\n(Self-Referential)", "TrustDecay\n(Human Factor)"]

        curr_g = curr_gaps
        sup_g = sup_gaps
        persistence_pct = []
        for cg, sg in zip(curr_g, sup_g):
            if abs(cg) < 0.01:
                persistence_pct.append(0)
            else:
                persistence_pct.append(sg / cg * 100 if cg != 0 else 0)

        colors = ["#e67e22", "#9b59b6", "#e74c3c", "#1abc9c"]
        bars = ax.bar(np.arange(len(labels)), persistence_pct, color=colors,
                      edgecolor="black", linewidth=1)

        for b, pct, cg, sg in zip(bars, persistence_pct, curr_g, sup_g):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
                    f"{pct:.0f}%\n(gap: {cg:.1f}→{sg:.1f})",
                    ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Challenge Persistence (%)", fontsize=12)
        ax.set_title("Which Challenges Survive When AI Exceeds Human Capability?", fontsize=14)
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="100% = unchanged")
        ax.axhline(y=50, color="orange", linestyle=":", alpha=0.5, label="50% = halved")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(persistence_pct) * 1.4 if persistence_pct else 150)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "v5_05_challenge_survival.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures_v5")

    print("=" * 60)
    print("EXPERIMENT: AI-Superior World vs. Current World")
    print("=" * 60)

    print("\n--- Running Current World ---")
    current_results, current_sims = run_current_world(seed=42)

    print("\n--- Running AI-Superior World ---")
    superior_results, superior_sims = run_ai_superior_world(seed=42)

    print_comparison(current_results, current_sims, superior_results, superior_sims)
    generate_figures(current_results, current_sims, superior_results, superior_sims, output_dir)
