"""Grey wolf optimizer (Mirjalili, Mirjalili & Lewis, 2014).

Position update is guided by the three best-known solutions (alpha, beta,
delta), with the exploration/exploitation control parameter `a` decaying
linearly from 2 to 0 over the run.
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
) -> OptimizeResult:
    rng = np.random.default_rng(seed)
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,))
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,))

    X = rng.uniform(lb, ub, size=(pop_size, dim))

    alpha_pos = np.zeros(dim)
    beta_pos = np.zeros(dim)
    delta_pos = np.zeros(dim)
    alpha_score = np.inf
    beta_score = np.inf
    delta_score = np.inf

    convergence = []
    for it in range(max_iter):
        fitness = np.array([objective_fn(x) for x in X])
        for i in range(pop_size):
            f = fitness[i]
            if f < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = f, X[i].copy()
            elif f < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = f, X[i].copy()
            elif f < delta_score:
                delta_score, delta_pos = f, X[i].copy()

        a = 2.0 - it * (2.0 / max_iter)

        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        A1, C1 = 2 * a * r1 - a, 2 * r2
        D_alpha = np.abs(C1 * alpha_pos - X)
        X1 = alpha_pos - A1 * D_alpha

        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        A2, C2 = 2 * a * r1 - a, 2 * r2
        D_beta = np.abs(C2 * beta_pos - X)
        X2 = beta_pos - A2 * D_beta

        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        A3, C3 = 2 * a * r1 - a, 2 * r2
        D_delta = np.abs(C3 * delta_pos - X)
        X3 = delta_pos - A3 * D_delta

        X = np.clip((X1 + X2 + X3) / 3.0, lb, ub)
        convergence.append(float(alpha_score))

    return OptimizeResult(best_x=alpha_pos, best_cost=float(alpha_score), convergence=convergence)
