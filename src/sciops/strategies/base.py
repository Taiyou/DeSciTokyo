"""Abstract base class for management strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sciops.engine.actions import ActionSet
from sciops.engine.overhead import OverheadConfig
from sciops.pipeline.state import PipelineState


class Strategy(ABC):
    """Base class for all management strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""

    @property
    @abstractmethod
    def overhead_config(self) -> OverheadConfig:
        """Overhead cost configuration for this strategy."""

    @abstractmethod
    def decide(self, state: PipelineState, timestep: int) -> ActionSet:
        """
        Decide which actions to take given the current pipeline state.

        Args:
            state: Current pipeline state (observable).
            timestep: Current simulation timestep.

        Returns:
            ActionSet containing zero or more actions.
        """

    def reset(self) -> None:
        """Reset any internal state between runs (override if stateful)."""
