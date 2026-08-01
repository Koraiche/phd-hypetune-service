"""Random search baseline (article_draft.md Section 4.3)."""
from __future__ import annotations

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
    rng = np.random.default_rng(seed)
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,))
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,))

    best_cost = np.inf
    best_x = None
    convergence = []
    for _ in range(budget):
        x = rng.uniform(lb, ub, size=dim)
        cost = objective_fn(x)
        if cost < best_cost:
            best_cost = cost
            best_x = x.copy()
        convergence.append(float(best_cost))

    return OptimizeResult(best_x=best_x, best_cost=float(best_cost), convergence=convergence)
