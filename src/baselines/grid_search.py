"""Grid search baseline (article_draft.md Section 4.3).

A grid must use the same resolution on every axis, so a fixed evaluation
`budget` only translates to an per-axis resolution of floor(budget**(1/dim))
— the realized evaluation count is n_per_dim**dim, which is generally LESS
than `budget` and shrinks fast as dimensionality grows. This is reported
explicitly (via `OptimizeResult.convergence`'s actual length) rather than
padded to match budget, because that resolution collapse under
dimensionality is precisely the classical motivation for metaheuristics
over grid search (see article Section 1) — hiding it would undercut the
comparison's honesty.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Optional

import numpy as np

from phase2_search.result import OptimizeResult


def optimize(
    objective_fn: Callable[[np.ndarray], float],
    lb,
    ub,
    dim: int,
    budget: int = 4000,
    seed: Optional[int] = None,
) -> OptimizeResult:
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,))
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,))

    n_per_dim = max(2, int(math.floor(budget ** (1.0 / dim))))
    axes = [np.linspace(lb[d], ub[d], n_per_dim) for d in range(dim)]
    grid = np.array(list(itertools.product(*axes)))

    # Evaluation order doesn't change the final answer (grid search is
    # exhaustive and deterministic), but it does change the shape of the
    # best-so-far convergence curve. Shuffle with the given seed so the
    # curve is a representative rather than worst/best-case ordering.
    rng = np.random.default_rng(seed)
    grid = grid[rng.permutation(len(grid))]

    best_cost = np.inf
    best_x = grid[0].copy()
    convergence = []
    for x in grid:
        cost = objective_fn(x)
        if cost < best_cost:
            best_cost = cost
            best_x = x.copy()
        convergence.append(float(best_cost))

    return OptimizeResult(best_x=best_x, best_cost=float(best_cost), convergence=convergence)
