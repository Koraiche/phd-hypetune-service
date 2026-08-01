"""Shared result type for all Phase 2 metaheuristics.

Kept separate so pso.py / gwo.py / de.py don't each redefine it, and so
Phase 2's real (hardware-measured) objective and the synthetic validation
functions in src/validation/ can consume optimizer output identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptimizeResult:
    best_x: np.ndarray
    best_cost: float
    convergence: list = field(default_factory=list)  # best-cost-so-far, one entry per iteration
