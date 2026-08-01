"""Phase 2 objective function (article_draft.md Section 3.6): scalarized,
constrained multi-objective scoring of a runtime configuration.

Deliberately decoupled from any specific evaluator (mock or real
llama.cpp-on-Pi) via the Evaluator protocol below, so the normalization,
weighting, and hard-constraint logic here is fully unit-testable without
hardware, and swaps to real measurement later by passing a different
Evaluator instance into make_objective — nothing else in this module changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, Tuple

import numpy as np

from .parameter_space import ParameterSpace


@dataclass
class EvaluationResult:
    """Raw measurement for one runtime configuration."""

    latency_ms_per_token: float
    quality: float  # higher is better (e.g. -perplexity, or task accuracy)
    peak_rss_mb: float
    energy_j_per_token: float
    feasible: bool = True
    reason: str = ""  # why infeasible, if not feasible


class Evaluator(Protocol):
    def evaluate(self, config: dict) -> EvaluationResult: ...


@dataclass
class ObjectiveWeights:
    """alpha/beta/gamma/delta from Section 3.6. Defaults are illustrative,
    not tuned — Section 3.6 flags the real weight choice as an open question
    (or a candidate for reporting a Pareto front instead of one scalarization)."""

    alpha: float = 0.35  # latency
    beta: float = 0.35  # quality
    gamma: float = 0.2  # memory
    delta: float = 0.1  # energy


@dataclass
class NormalizationRanges:
    latency: Tuple[float, float]
    quality: Tuple[float, float]
    memory: Tuple[float, float]
    energy: Tuple[float, float]

    @staticmethod
    def _norm(value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def normalize(self, raw: EvaluationResult) -> dict:
        return {
            "latency": self._norm(raw.latency_ms_per_token, *self.latency),
            "quality": self._norm(raw.quality, *self.quality),
            "memory": self._norm(raw.peak_rss_mb, *self.memory),
            "energy": self._norm(raw.energy_j_per_token, *self.energy),
        }


def calibrate_normalization_ranges(
    space: ParameterSpace,
    evaluator: Evaluator,
    n_samples: int = 200,
    seed: int = 0,
) -> NormalizationRanges:
    """Estimate min-max ranges for each raw metric by random sampling,
    exactly as Section 3.6 requires ("min-max scaled over observed range").

    Infeasible samples are excluded from range estimation — an infeasible
    config never produced a real quality/latency/memory/energy observation,
    it produced a rejection, and including it would distort the scale for
    every feasible config's score.
    """
    rng = np.random.default_rng(seed)
    lat, qual, mem, energy = [], [], [], []
    for _ in range(n_samples):
        x = rng.uniform(space.lb, space.ub)
        result = evaluator.evaluate(space.decode(x))
        if not result.feasible:
            continue
        lat.append(result.latency_ms_per_token)
        qual.append(result.quality)
        mem.append(result.peak_rss_mb)
        energy.append(result.energy_j_per_token)

    if not lat:
        raise RuntimeError(
            f"All {n_samples} calibration samples were infeasible — the "
            f"parameter space and feasibility constraint disagree; check "
            f"the RAM ceiling / evaluator before running the real search."
        )

    return NormalizationRanges(
        latency=(min(lat), max(lat)),
        quality=(min(qual), max(qual)),
        memory=(min(mem), max(mem)),
        energy=(min(energy), max(energy)),
    )


def make_objective(
    space: ParameterSpace,
    evaluator: Evaluator,
    ranges: NormalizationRanges,
    weights: ObjectiveWeights = ObjectiveWeights(),
) -> Callable[[np.ndarray], float]:
    """Build f(x) for PSO/GWO/DE to minimize.

    Hard constraints are enforced by rejection (cost = +inf), not a finite
    penalty term: per Section 3.6, an infeasible configuration (RAM ceiling
    exceeded, crash, timeout) must never be numerically outweighed by an
    otherwise-good quality score in a weighted sum.
    """

    def objective(x: np.ndarray) -> float:
        result = evaluator.evaluate(space.decode(x))
        if not result.feasible:
            return math.inf

        n = ranges.normalize(result)
        return (
            weights.alpha * n["latency"]
            + weights.beta * (1.0 - n["quality"])
            + weights.gamma * n["memory"]
            + weights.delta * n["energy"]
        )

    return objective
