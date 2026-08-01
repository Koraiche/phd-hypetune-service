"""Differential evolution, DE/rand/1/bin (Storn & Price, 1997).

Mutation scale F is resampled uniformly in [beta_min, beta_max] per target
vector per generation (a common self-adaptive variant), matching the
beta_min/beta_max/pCR parameterization used in the source paper's Table 4.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .result import OptimizeResult


def optimize(
    objective_fn: Callable[[np.ndarray], float],
    lb,
    ub,
    dim: int,
    pop_size: int = 20,
    max_iter: int = 100,
    seed: Optional[int] = None,
    beta_min: float = 0.2,
    beta_max: float = 0.9,
    pCR: float = 0.1,
) -> OptimizeResult:
    rng = np.random.default_rng(seed)
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,))
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,))

    X = rng.uniform(lb, ub, size=(pop_size, dim))
    fitness = np.array([objective_fn(x) for x in X])
    best_idx = int(np.argmin(fitness))
    best_x = X[best_idx].copy()
    best_cost = float(fitness[best_idx])

    convergence = []
    all_idx = np.arange(pop_size)
    for _ in range(max_iter):
        for i in range(pop_size):
            candidates = all_idx[all_idx != i]
            r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
            F = rng.uniform(beta_min, beta_max)
            mutant = np.clip(X[r1] + F * (X[r2] - X[r3]), lb, ub)

            cross_mask = rng.random(dim) < pCR
            if not cross_mask.any():
                cross_mask[rng.integers(dim)] = True
            trial = np.where(cross_mask, mutant, X[i])

            trial_cost = objective_fn(trial)
            if trial_cost < fitness[i]:
                X[i] = trial
                fitness[i] = trial_cost
                if trial_cost < best_cost:
                    best_cost = float(trial_cost)
                    best_x = trial.copy()

        convergence.append(best_cost)

    return OptimizeResult(best_x=best_x, best_cost=best_cost, convergence=convergence)
