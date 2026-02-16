"""
Meta-Science Ecosystem Simulator
==================================
Simulates N research labs competing in a shared publication system.
Each lab has its own 6-stage pipeline and optimization strategy.

The ecosystem manages:
1. Lab-level pipeline simulation (reusing existing Simulator mechanics)
2. Paper generation from pipeline output
3. Publication review with configurable biases
4. Knowledge base accumulation
5. Funding redistribution based on publications
"""

import math
import random
from dataclasses import dataclass, field

from scientific_process import ProcessStep, ProcessConfig, create_default_pipeline
from optimizers import Optimizer, BaselineOptimizer
from meta_science_models import (
    Paper,
    PublicationChannel,
    EcosystemMetrics,
    compute_gini,
)


@dataclass
class LabState:
    """Runtime state of a single research lab in the ecosystem."""

    lab_id: int
    pipeline: list[ProcessStep]
    optimizer: Optimizer
    funding: float  # Current funding (used as total_resources)
    reputation: float = 1.0
    is_ai_lab: bool = False
    papers_submitted: list[Paper] = field(default_factory=list)
    papers_published: list[Paper] = field(default_factory=list)
    cumulative_output: float = 0.0
    openness: float = 0.0  # For open science experiment
    risk_appetite: float = 0.5  # For funding experiment

    # p-hacking behavior
    p_hacking_intensity: float = 0.0

    def pipeline_quality(self) -> float:
        """Estimate quality from pipeline state (low rework/failure = high quality)."""
        total_completed = sum(s.completed_units for s in self.pipeline) + 0.01
        total_rework = sum(s.rework_units for s in self.pipeline)
        total_failed = sum(s.failed_units for s in self.pipeline)
        quality = 1.0 - (total_rework + total_failed) / (total_completed + total_rework + total_failed)
        return max(0.1, min(1.0, quality))


class MetaScienceEcosystem:
    """
    Simulates a scientific community of N labs.

    Each lab runs its own 6-stage pipeline with its own optimizer.
    Labs produce papers, which are evaluated by a publication channel.
    Published papers accumulate in the knowledge base.
    Funding is redistributed based on publication performance.
    """

    def __init__(
        self,
        n_labs: int = 50,
        publication_channel: PublicationChannel | None = None,
        base_funding: float = 6.0,
        input_rate: float = 2.0,
        base_truth_rate: float = 0.5,
        paper_threshold: float = 0.5,
        funding_cycle: int = 20,
        ai_fraction: float = 0.3,
        p_hacking_intensity: float = 0.0,
        seed: int | None = None,
    ):
        if seed is not None:
            random.seed(seed)

        self.n_labs = n_labs
        self.publication_channel = publication_channel or PublicationChannel(name="Default Journal")
        self.base_funding = base_funding
        self.input_rate = input_rate
        self.base_truth_rate = base_truth_rate
        self.paper_threshold = paper_threshold
        self.funding_cycle = funding_cycle
        self.ai_fraction = ai_fraction

        # Create labs
        self.labs: list[LabState] = []
        self._create_labs(p_hacking_intensity)

        # Global state
        self.knowledge_base: list[Paper] = []
        self.metrics_history: list[EcosystemMetrics] = []
        self.time_step = 0

    def _create_labs(self, p_hacking_intensity: float) -> None:
        """Create N labs with diverse strategies."""
        n_ai = int(self.n_labs * self.ai_fraction)
        for i in range(self.n_labs):
            pipeline = create_default_pipeline()
            is_ai = i < n_ai

            if is_ai:
                # AI labs use a simplified AI-SciOps-like behavior
                optimizer = BaselineOptimizer()
                # We'll apply AI assistance directly
            else:
                optimizer = BaselineOptimizer()

            lab = LabState(
                lab_id=i,
                pipeline=pipeline,
                optimizer=optimizer,
                funding=self.base_funding,
                is_ai_lab=is_ai,
                p_hacking_intensity=p_hacking_intensity,
            )
            self.labs.append(lab)

    def run(self, time_steps: int = 200) -> list[EcosystemMetrics]:
        """Run the ecosystem simulation."""
        for t in range(time_steps):
            self.time_step = t
            self._step()
        return self.metrics_history

    def _step(self) -> None:
        """Execute one global time step."""
        t = self.time_step
        papers_this_step: list[Paper] = []
        total_output = 0.0

        # 1. Run each lab's pipeline
        for lab in self.labs:
            output = self._run_lab(lab, t)
            total_output += output

            # 2. Generate papers from output
            if output > self.paper_threshold:
                paper = self._generate_paper(lab, t, output)
                lab.papers_submitted.append(paper)
                papers_this_step.append(paper)

        # 3. Publication review
        published_this_step = []
        for paper in papers_this_step:
            if self.publication_channel.evaluate(paper):
                paper.is_published = True
                lab = self.labs[paper.lab_id]
                lab.papers_published.append(paper)
                self.knowledge_base.append(paper)
                published_this_step.append(paper)

        # 4. Update reputations
        self._update_reputations(published_this_step)

        # 5. Redistribute funding periodically
        if t > 0 and t % self.funding_cycle == 0:
            self._redistribute_funding()

        # 6. Collect metrics
        self._collect_metrics(t, papers_this_step, published_this_step, total_output)

    def _run_lab(self, lab: LabState, time_step: int) -> float:
        """Run one time step for a single lab's pipeline."""
        # Apply AI assistance for AI labs
        if lab.is_ai_lab:
            ai_progress = min(1.0, time_step / 100.0)
            for step in lab.pipeline:
                step.ai_assistance_level = min(
                    0.8 * ai_progress, step.config.ai_automatable * 0.9
                )
        else:
            for step in lab.pipeline:
                step.ai_assistance_level = 0.0

        # Optimize pipeline
        lab.pipeline = lab.optimizer.optimize(lab.pipeline, time_step, lab.funding)

        # Feed work through pipeline
        incoming = self.input_rate
        for step in lab.pipeline:
            output = step.step(incoming)
            incoming = output

        lab.cumulative_output += incoming
        return incoming

    def _generate_paper(self, lab: LabState, time_step: int, output: float) -> Paper:
        """Convert pipeline output into a Paper object."""
        quality = lab.pipeline_quality()

        # True effect: base truth rate (independent of quality to avoid confound)
        is_true = random.random() < self.base_truth_rate
        # True findings have moderate positive effects; false findings near-zero
        if is_true:
            effect_size = abs(random.gauss(0.4, 0.2))
        else:
            effect_size = abs(random.gauss(0.05, 0.1))

        # p-hacking: inflate reported effect (especially for null results)
        reported_effect = effect_size
        if lab.p_hacking_intensity > 0:
            pressure = 0.3 + max(0, (0.7 - lab.reputation) * lab.p_hacking_intensity)
            if not is_true:
                # False findings get inflated more (selective reporting)
                reported_effect = effect_size + pressure * 0.3
            else:
                reported_effect = effect_size + pressure * 0.05

        novelty = random.betavariate(2, 5)  # Most papers are incremental

        return Paper(
            lab_id=lab.lab_id,
            time_step=time_step,
            quality=quality,
            effect_size=effect_size,
            reported_effect=reported_effect,
            novelty=novelty,
            is_true=is_true,
            openness=lab.openness,
        )

    def _update_reputations(self, published: list[Paper]) -> None:
        """Update lab reputations based on publications."""
        for paper in published:
            lab = self.labs[paper.lab_id]
            # Publishing increases reputation
            lab.reputation = min(2.0, lab.reputation + 0.02)

        # Slight decay for non-publishing labs
        publishing_labs = {p.lab_id for p in published}
        for lab in self.labs:
            if lab.lab_id not in publishing_labs:
                lab.reputation = max(0.1, lab.reputation * 0.998)

    def _redistribute_funding(self) -> None:
        """Redistribute funding based on reputation and publication record."""
        total_budget = sum(lab.funding for lab in self.labs)

        # Score each lab
        scores = []
        for lab in self.labs:
            recent_pubs = len([
                p for p in lab.papers_published
                if p.time_step > self.time_step - self.funding_cycle
            ])
            score = 0.5 * lab.reputation + 0.5 * (recent_pubs / max(1, self.funding_cycle / 10))
            scores.append(max(0.1, score))

        total_score = sum(scores)
        for lab, score in zip(self.labs, scores):
            lab.funding = max(1.0, total_budget * score / total_score)

    def _collect_metrics(
        self,
        time_step: int,
        submitted: list[Paper],
        published: list[Paper],
        total_output: float,
    ) -> None:
        """Collect ecosystem-level metrics."""
        all_published = [p for p in self.knowledge_base if not p.retracted]

        # Truth ratio
        if all_published:
            truth_ratio = sum(1 for p in all_published if p.is_true) / len(all_published)
        else:
            truth_ratio = 1.0

        # False positive rate: of papers claiming positive effect, how many are false?
        positive_papers = [p for p in all_published if p.reported_effect > 0.2]
        if positive_papers:
            false_positive_rate = sum(1 for p in positive_papers if not p.is_true) / len(positive_papers)
        else:
            false_positive_rate = 0.0

        # Funding inequality
        funding_values = [lab.funding for lab in self.labs]
        gini = compute_gini(funding_values)

        # Knowledge growth: true papers published this step
        true_this_step = sum(1 for p in published if p.is_true)

        # AI lab publication share
        ai_pubs = sum(1 for p in published if self.labs[p.lab_id].is_ai_lab)
        ai_share = ai_pubs / max(1, len(published))

        metrics = EcosystemMetrics(
            time_step=time_step,
            total_papers_submitted=len(submitted),
            total_papers_published=len(published),
            truth_ratio=truth_ratio,
            false_positive_rate=false_positive_rate,
            mean_lab_reputation=sum(l.reputation for l in self.labs) / self.n_labs,
            gini_funding=gini,
            knowledge_growth_rate=float(true_this_step),
            ai_lab_publication_share=ai_share,
            total_community_output=total_output,
        )
        self.metrics_history.append(metrics)
