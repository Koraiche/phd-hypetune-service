"""Particle swarm optimization (Kennedy & Eberhart, 1995).

Global-best PSO with inertia-weight damping and velocity clamping. The
damping schedule and clamp are standard stabilization additions on top of
the base algorithm (not present in the minimal textbook form) and are
called out here rather than silently assumed.
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
    w: float = 1.0,
    wdamp: float = 0.99,
    c1: float = 2.0,
    c2: float = 2.0,
) -> OptimizeResult:
    rng = np.random.default_rng(seed)
    lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,))
    ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,))
    vmax = 0.2 * (ub - lb)

    X = rng.uniform(lb, ub, size=(pop_size, dim))
    V = np.zeros((pop_size, dim))
    fitness = np.array([objective_fn(x) for x in X])

    pbest_x = X.copy()
    pbest_cost = fitness.copy()
    gbest_idx = int(np.argmin(pbest_cost))
    gbest_x = pbest_x[gbest_idx].copy()
    gbest_cost = float(pbest_cost[gbest_idx])

    convergence = []
    w_cur = w
    for _ in range(max_iter):
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        V = w_cur * V + c1 * r1 * (pbest_x - X) + c2 * r2 * (gbest_x - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, lb, ub)

        fitness = np.array([objective_fn(x) for x in X])
        improved = fitness < pbest_cost
        pbest_x[improved] = X[improved]
        pbest_cost[improved] = fitness[improved]

        gbest_idx = int(np.argmin(pbest_cost))
        if pbest_cost[gbest_idx] < gbest_cost:
            gbest_cost = float(pbest_cost[gbest_idx])
            gbest_x = pbest_x[gbest_idx].copy()

        w_cur *= wdamp
        convergence.append(gbest_cost)

    return OptimizeResult(best_x=gbest_x, best_cost=gbest_cost, convergence=convergence)
