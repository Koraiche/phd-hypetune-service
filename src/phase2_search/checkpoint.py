"""Crash-safe replay checkpoints for deterministic optimization runs.

Every completed objective evaluation is atomically persisted.  After a
restart, rerunning the optimizer with the same seed replays those recorded
costs in order, reconstructing its in-memory state without repeating costly
hardware measurements.  New evaluations continue from the first unfinished
point.  A metadata mismatch fails closed so stale measurements cannot be
silently mixed into a changed experiment.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .result import OptimizeResult


class CheckpointMismatchError(RuntimeError):
    """The checkpoint does not describe the run currently being requested."""


def _atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _encoded_cost(cost: float) -> float | str:
    if math.isinf(cost):
        return "inf" if cost > 0 else "-inf"
    if math.isnan(cost):
        return "nan"
    return float(cost)


def _decoded_cost(cost: float | str) -> float:
    return float(cost)


class ReplayCheckpointObjective:
    """Callable objective that records and deterministically replays calls."""

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        checkpoint_dir: Path,
        metadata: Mapping[str, object],
    ) -> None:
        self.objective_fn = objective_fn
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metadata_path = self.checkpoint_dir / "metadata.json"
        self.trace_path = self.checkpoint_dir / "evaluations.json"
        expected = dict(metadata)

        if self.metadata_path.exists():
            with open(self.metadata_path, encoding="utf-8") as handle:
                actual = json.load(handle)
            if actual != expected:
                raise CheckpointMismatchError(
                    f"checkpoint metadata mismatch in {self.checkpoint_dir}; "
                    "move the old checkpoint aside or restore the original run settings"
                )
        else:
            _atomic_json_write(self.metadata_path, expected)

        if self.trace_path.exists():
            with open(self.trace_path, encoding="utf-8") as handle:
                self.records = json.load(handle)
        else:
            self.records = []
        self.initial_record_count = len(self.records)
        self.cursor = 0

    @property
    def replayed_evaluations(self) -> int:
        return min(self.cursor, self.initial_record_count)

    def __call__(self, x: np.ndarray) -> float:
        point = np.asarray(x, dtype=float).tolist()
        if self.cursor < len(self.records):
            record = self.records[self.cursor]
            recorded_point = np.asarray(record["x"], dtype=float)
            if not np.array_equal(np.asarray(point), recorded_point):
                raise CheckpointMismatchError(
                    f"objective call {self.cursor + 1} diverged from the saved trace in "
                    f"{self.checkpoint_dir}; optimizer code or settings changed"
                )
            self.cursor += 1
            return _decoded_cost(record["cost"])

        cost = float(self.objective_fn(np.asarray(x, dtype=float)))
        self.records.append({"x": point, "cost": _encoded_cost(cost)})
        self.cursor += 1
        _atomic_json_write(self.trace_path, self.records)
        return cost


def save_result(path: Path, result: OptimizeResult) -> None:
    _atomic_json_write(path, {
        "best_x": np.asarray(result.best_x, dtype=float).tolist(),
        "best_cost": _encoded_cost(float(result.best_cost)),
        "convergence": [_encoded_cost(float(value)) for value in result.convergence],
    })


def load_result(path: Path) -> OptimizeResult:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    return OptimizeResult(
        best_x=np.asarray(value["best_x"], dtype=float),
        best_cost=_decoded_cost(value["best_cost"]),
        convergence=[_decoded_cost(item) for item in value["convergence"]],
    )


def run_with_checkpoint(
    optimize_fn: Callable[..., OptimizeResult],
    objective_fn: Callable[[np.ndarray], float],
    checkpoint_dir: Path,
    metadata: Mapping[str, object],
    **optimize_kwargs,
) -> tuple[OptimizeResult, int, bool]:
    """Run or resume an optimizer.

    Returns ``(result, replayed_evaluations, loaded_complete_result)``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    result_path = checkpoint_dir / "result.json"
    replay = ReplayCheckpointObjective(objective_fn, checkpoint_dir, metadata)
    if result_path.exists():
        return load_result(result_path), len(replay.records), True

    result = optimize_fn(replay, **optimize_kwargs)
    save_result(result_path, result)
    return result, replay.replayed_evaluations, False
