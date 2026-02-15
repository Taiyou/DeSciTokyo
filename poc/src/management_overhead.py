"""
Management Overhead Model
==========================
Models the cost of management activities themselves.

In reality, every management method consumes resources:
- PDCA: meetings, data collection, analysis time
- Agile: sprint planning, standups, retrospectives, backlog grooming
- TOC: bottleneck analysis, constraint identification
- AI-SciOps: monitoring infrastructure, AI training, human review of AI decisions
- Lean: value stream mapping, waste analysis, kaizen events
- Six Sigma: measurement, statistical analysis, DMAIC cycles

This module provides overhead calculation that reduces the effective
resources available for actual research work.
"""

from dataclasses import dataclass


@dataclass
class OverheadProfile:
    """Defines the management overhead characteristics of a strategy.

    Attributes:
        base_cost: Fixed resource cost per time step for management activities.
            Represents standing meetings, monitoring infrastructure, etc.
        per_action_cost: Additional cost each time an optimization action is taken.
            Represents the cost of analysis, planning, and decision-making.
        complexity_scaling: How much overhead grows as pipeline size increases.
            0.0 = no scaling, 1.0 = linear with number of processes.
        ai_infrastructure_cost: Cost of maintaining AI systems (training, inference).
            Only applies to AI-augmented strategies.
        learning_cost: One-time cost when entering a new stage or mode.
            Represents setup, training, or transition costs.
        human_coordination_cost: Cost of human coordination per time step.
            Higher for methods requiring more human interaction (Agile, PDCA).
    """
    base_cost: float = 0.0
    per_action_cost: float = 0.0
    complexity_scaling: float = 0.0
    ai_infrastructure_cost: float = 0.0
    learning_cost: float = 0.0
    human_coordination_cost: float = 0.0

    def compute_overhead(
        self,
        time_step: int,
        num_processes: int,
        num_actions_this_step: int,
        is_stage_transition: bool = False,
    ) -> float:
        """Compute total management overhead for this time step.

        Returns resource units consumed by management (subtracted from total).
        """
        overhead = self.base_cost
        overhead += self.per_action_cost * num_actions_this_step
        overhead += self.complexity_scaling * num_processes * 0.05
        overhead += self.ai_infrastructure_cost
        overhead += self.human_coordination_cost

        if is_stage_transition:
            overhead += self.learning_cost

        return overhead


# --- Overhead profiles for each strategy ---

OVERHEAD_PROFILES = {
    "Baseline (No Optimization)": OverheadProfile(
        # No management = no overhead
        base_cost=0.0,
        per_action_cost=0.0,
        complexity_scaling=0.0,
        ai_infrastructure_cost=0.0,
        learning_cost=0.0,
        human_coordination_cost=0.0,
    ),
    "TOC + PDCA": OverheadProfile(
        # Regular meetings (PDCA cycle), bottleneck analysis
        base_cost=0.15,
        per_action_cost=0.05,
        complexity_scaling=0.3,
        ai_infrastructure_cost=0.0,
        learning_cost=0.1,
        human_coordination_cost=0.15,
    ),
    "Agile-Scrum": OverheadProfile(
        # Sprint planning, daily standups, retrospectives, backlog grooming
        # Agile has highest human coordination cost
        base_cost=0.2,
        per_action_cost=0.03,
        complexity_scaling=0.4,
        ai_infrastructure_cost=0.0,
        learning_cost=0.15,
        human_coordination_cost=0.25,
    ),
    "Lean-SciOps": OverheadProfile(
        # Value stream mapping, waste analysis, kaizen events
        base_cost=0.1,
        per_action_cost=0.04,
        complexity_scaling=0.2,
        ai_infrastructure_cost=0.0,
        learning_cost=0.2,  # Initial value stream mapping is expensive
        human_coordination_cost=0.1,
    ),
    "SixSigma-SciOps": OverheadProfile(
        # DMAIC cycles, statistical measurement, control charts
        base_cost=0.15,
        per_action_cost=0.06,
        complexity_scaling=0.3,
        ai_infrastructure_cost=0.0,
        learning_cost=0.25,  # Define/Measure phases are costly
        human_coordination_cost=0.15,
    ),
    "AI-SciOps (Autonomous Optimization)": OverheadProfile(
        # AI monitoring, model training, human review of AI decisions
        base_cost=0.1,
        per_action_cost=0.02,
        complexity_scaling=0.2,
        ai_infrastructure_cost=0.2,
        learning_cost=0.3,  # Stage transitions require retraining
        human_coordination_cost=0.05,
    ),
    "Kanban-SciOps (Pull-based Flow)": OverheadProfile(
        # WIP monitoring, board maintenance, flow metrics
        base_cost=0.1,
        per_action_cost=0.02,
        complexity_scaling=0.2,
        ai_infrastructure_cost=0.15,
        learning_cost=0.15,
        human_coordination_cost=0.05,
    ),
    "Adaptive-SciOps (Metric-driven)": OverheadProfile(
        # Continuous metric monitoring, AI learning, stage analysis
        base_cost=0.1,
        per_action_cost=0.02,
        complexity_scaling=0.2,
        ai_infrastructure_cost=0.25,  # More AI analysis than basic SciOps
        learning_cost=0.2,
        human_coordination_cost=0.05,
    ),
    "Holistic-SciOps (Integrated)": OverheadProfile(
        # Most complex: Kanban + learning + pruning + feedback loops
        base_cost=0.15,
        per_action_cost=0.03,
        complexity_scaling=0.3,
        ai_infrastructure_cost=0.3,  # Most AI infrastructure
        learning_cost=0.2,
        human_coordination_cost=0.08,
    ),
}


def get_overhead_profile(strategy_name: str) -> OverheadProfile:
    """Get overhead profile for a strategy, with default fallback."""
    return OVERHEAD_PROFILES.get(strategy_name, OverheadProfile())
