"""Synthetic benchmark functions for validating PSO/GWO/DE before they are
trusted on the real (noisy, hardware-measured) Phase 2 objective.

Mirrors article_draft.md Section 3.8 / the source paper's validation set:
Sphere (smooth, convex, N-dimensional), Goldstein-Price (multimodal, fixed
2D), and a custom "FindValues" error function (the same shape of problem as
minimizing a real fitted-parameter error, at N=5 dimensions with asymmetric
per-dimension bounds).
"""
from __future__ import annotations

import numpy as np

FIND_VALUES_TARGET = np.array([21.0, 555.0, -106.0, 74.0, -701.0])
FIND_VALUES_LB = np.array([0.0, 200.0, -200.0, -100.0, -900.0])
FIND_VALUES_UB = np.array([100.0, 700.0, 50.0, 200.0, -400.0])


def sphere(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def goldstein_price(x) -> float:
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != 2:
        raise ValueError("Goldstein-Price is defined for dim=2 only")
    x1, x2 = x[0], x[1]
    term1 = 1 + (x1 + x2 + 1) ** 2 * (
        19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
    )
    term2 = 30 + (2 * x1 - 3 * x2) ** 2 * (
        18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
    )
    return float(term1 * term2)


def find_values(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum((x - FIND_VALUES_TARGET) ** 2))


# Registry consumed by the validation driver: each entry is a complete,
# self-contained problem spec (function + bounds + dimensionality + known
# global minimum, for reporting how close each optimizer gets).
FUNCTIONS = [
    {"name": "Sphere", "fn": sphere, "dim": 10, "lb": -10.0, "ub": 10.0, "global_min": 0.0},
    {"name": "Goldstein-Price", "fn": goldstein_price, "dim": 2, "lb": -2.0, "ub": 2.0, "global_min": 3.0},
    {"name": "FindValues", "fn": find_values, "dim": 5, "lb": FIND_VALUES_LB, "ub": FIND_VALUES_UB, "global_min": 0.0},
]
