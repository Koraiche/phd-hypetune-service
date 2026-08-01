import math

import numpy as np
import pytest

from phase2_search.objective import (
    EvaluationResult,
    NormalizationRanges,
    ObjectiveWeights,
    calibrate_normalization_ranges,
    make_objective,
)
from phase2_search.parameter_space import FloatParam, ParameterSpace


class LinearEvaluator:
    """v in [0,10] -> latency=v, quality=-v, constant memory/energy.
    Feasible iff v <= 8, so both the rejection path and the coupling
    between latency and quality are exercised by a single dial."""

    def evaluate(self, config):
        v = config["v"]
        feasible = v <= 8.0
        return EvaluationResult(
            latency_ms_per_token=v,
            quality=-v,
            peak_rss_mb=100.0,
            energy_j_per_token=1.0,
            feasible=feasible,
            reason="" if feasible else "over budget",
        )


@pytest.fixture
def space():
    return ParameterSpace([FloatParam("v", 0.0, 10.0)])


def test_calibration_excludes_infeasible_samples(space):
    ranges = calibrate_normalization_ranges(space, LinearEvaluator(), n_samples=500, seed=0)
    assert ranges.latency[1] <= 8.0
    assert ranges.latency[0] >= 0.0


def test_infeasible_config_returns_inf_cost(space):
    ranges = NormalizationRanges(latency=(0, 8), quality=(-8, 0), memory=(100, 100), energy=(1, 1))
    f = make_objective(space, LinearEvaluator(), ranges)
    assert f(np.array([9.0])) == math.inf


def test_objective_weighted_sum_matches_hand_computation(space):
    ranges = NormalizationRanges(latency=(0.0, 8.0), quality=(-8.0, 0.0), memory=(100.0, 100.0), energy=(1.0, 1.0))
    weights = ObjectiveWeights(alpha=0.5, beta=0.3, gamma=0.1, delta=0.1)
    f = make_objective(space, LinearEvaluator(), ranges, weights)

    cost = f(np.array([4.0]))

    lat_norm = (4.0 - 0.0) / (8.0 - 0.0)
    qual_norm = (-4.0 - (-8.0)) / (0.0 - (-8.0))
    mem_norm = 0.0  # degenerate range (100, 100) is defined to normalize to 0.0
    energy_norm = 0.0  # degenerate range (1, 1), same convention
    expected = 0.5 * lat_norm + 0.3 * (1 - qual_norm) + 0.1 * mem_norm + 0.1 * energy_norm

    assert cost == pytest.approx(expected)


def test_higher_latency_increases_cost_all_else_equal(space):
    ranges = NormalizationRanges(latency=(0.0, 8.0), quality=(-8.0, 0.0), memory=(100.0, 100.0), energy=(1.0, 1.0))
    f = make_objective(space, LinearEvaluator(), ranges)
    costs = [f(np.array([v])) for v in [0.0, 2.0, 4.0, 6.0, 8.0]]
    assert costs == sorted(costs)
    assert costs[0] < costs[-1]
