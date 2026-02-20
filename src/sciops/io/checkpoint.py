"""Checkpoint/resume for long-running experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Set, Tuple


class CheckpointManager:
    """Tracks completed run keys for checkpoint/resume."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_file = self.checkpoint_dir / "completed_keys.json"
        self._completed: Set[Tuple[str, str, float, int]] = set()

    def load(self) -> None:
        """Load checkpoint state from disk."""
        if self._checkpoint_file.exists():
            with open(self._checkpoint_file) as f:
                data = json.load(f)
            self._completed = {tuple(k) for k in data}  # type: ignore[arg-type]

    def save(self) -> None:
        """Save checkpoint state to disk."""
        with open(self._checkpoint_file, "w") as f:
            json.dump([list(k) for k in self._completed], f)

    def mark_completed(self, key: Tuple[str, str, float, int]) -> None:
        self._completed.add(key)

    def mark_batch_completed(self, keys: list) -> None:
        for key in keys:
            self._completed.add(tuple(key))

    def is_completed(self, key: Tuple[str, str, float, int]) -> bool:
        return key in self._completed

    def get_all_completed(self) -> Set[Tuple[str, str, float, int]]:
        return self._completed

    @property
    def num_completed(self) -> int:
        return len(self._completed)
