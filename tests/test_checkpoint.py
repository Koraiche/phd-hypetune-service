import numpy as np
import pytest

from phase2_search.checkpoint import (
    CheckpointMismatchError,
    ReplayCheckpointObjective,
    run_with_checkpoint,
)
from phase2_search.result import OptimizeResult


def test_replays_completed_evaluations_without_calling_hardware(tmp_path):
    calls = []

    def measured_objective(x):
        calls.append(x.copy())
        return float(np.sum(x ** 2))

    checkpoint = tmp_path / "run"
    first = ReplayCheckpointObjective(measured_objective, checkpoint, {"seed": 7})
    assert first(np.array([1.0, 2.0])) == 5.0
    assert first(np.array([3.0, 4.0])) == 25.0
    assert len(calls) == 2

    def must_not_run(_x):
        raise AssertionError("a saved hardware evaluation was repeated")

    resumed = ReplayCheckpointObjective(must_not_run, checkpoint, {"seed": 7})
    assert resumed(np.array([1.0, 2.0])) == 5.0
    assert resumed(np.array([3.0, 4.0])) == 25.0
    assert resumed.replayed_evaluations == 2


def test_rejects_changed_metadata_or_call_sequence(tmp_path):
    checkpoint = tmp_path / "run"
    saved = ReplayCheckpointObjective(lambda _x: 1.0, checkpoint, {"seed": 7})
    saved(np.array([1.0]))

    with pytest.raises(CheckpointMismatchError, match="metadata mismatch"):
        ReplayCheckpointObjective(lambda _x: 1.0, checkpoint, {"seed": 8})

    resumed = ReplayCheckpointObjective(lambda _x: 1.0, checkpoint, {"seed": 7})
    with pytest.raises(CheckpointMismatchError, match="diverged"):
        resumed(np.array([2.0]))


def test_persists_infinite_rejection_cost(tmp_path):
    checkpoint = tmp_path / "run"
    first = ReplayCheckpointObjective(lambda _x: np.inf, checkpoint, {"seed": 1})
    assert first(np.array([1.0])) == np.inf

    resumed = ReplayCheckpointObjective(lambda _x: 0.0, checkpoint, {"seed": 1})
    assert resumed(np.array([1.0])) == np.inf


def test_run_resumes_mid_optimizer_and_then_loads_completed_result(tmp_path):
    points = [np.array([3.0]), np.array([2.0]), np.array([1.0])]

    def simple_optimizer(objective_fn):
        costs = [objective_fn(point) for point in points]
        best = int(np.argmin(costs))
        return OptimizeResult(points[best], costs[best], list(np.minimum.accumulate(costs)))

    new_measurements = 0

    def interrupted_objective(x):
        nonlocal new_measurements
        new_measurements += 1
        if new_measurements == 3:
            raise RuntimeError("simulated power loss")
        return float(x[0])

    checkpoint = tmp_path / "run"
    with pytest.raises(RuntimeError, match="power loss"):
        run_with_checkpoint(simple_optimizer, interrupted_objective, checkpoint, {"seed": 4})

    resumed_measurements = 0

    def resumed_objective(x):
        nonlocal resumed_measurements
        resumed_measurements += 1
        return float(x[0])

    result, replayed, loaded = run_with_checkpoint(
        simple_optimizer, resumed_objective, checkpoint, {"seed": 4}
    )
    assert result.best_cost == 1.0
    assert replayed == 2
    assert resumed_measurements == 1
    assert loaded is False

    result, replayed, loaded = run_with_checkpoint(
        simple_optimizer,
        lambda _x: pytest.fail("completed checkpoint should not rerun the objective"),
        checkpoint,
        {"seed": 4},
    )
    assert result.best_cost == 1.0
    assert replayed == 3
    assert loaded is True
