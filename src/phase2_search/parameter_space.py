"""Continuous <-> discrete/categorical parameter encoding for Phase 2.

PSO/GWO/DE all operate on a continuous bounded vector x in R^d. The real
Phase 2 search space (article_draft.md Section 3.5) is mixed: integer
parameters (threads, n_batch, n_ctx) and a categorical parameter
(kv_cache_type). This module is the single place that translates between
the optimizer's continuous vector and the actual runtime configuration dict
passed to an Evaluator, so pso.py/gwo.py/de.py never need to know the
search space is anything but a plain bounded box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np


@dataclass(frozen=True)
class IntParam:
    name: str
    lo: int
    hi: int  # inclusive


@dataclass(frozen=True)
class FloatParam:
    name: str
    lo: float
    hi: float


@dataclass(frozen=True)
class CategoricalParam:
    name: str
    choices: Sequence[str]


Param = Union[IntParam, FloatParam, CategoricalParam]


class ParameterSpace:
    """Ordered collection of parameters; optimizer dimensionality == len(params)."""

    def __init__(self, params: Sequence[Param]):
        self.params = list(params)

    @property
    def dim(self) -> int:
        return len(self.params)

    @property
    def lb(self) -> np.ndarray:
        return np.array([0.0 if isinstance(p, CategoricalParam) else float(p.lo) for p in self.params])

    @property
    def ub(self) -> np.ndarray:
        # Categorical upper bound is nudged just below len(choices) so that
        # floor() at the exact upper bound never produces an out-of-range index.
        return np.array([
            len(p.choices) - 1e-9 if isinstance(p, CategoricalParam) else float(p.hi)
            for p in self.params
        ])

    def decode(self, x: np.ndarray) -> dict:
        config = {}
        for value, p in zip(x, self.params):
            if isinstance(p, IntParam):
                config[p.name] = int(round(float(np.clip(value, p.lo, p.hi))))
            elif isinstance(p, FloatParam):
                config[p.name] = float(np.clip(value, p.lo, p.hi))
            elif isinstance(p, CategoricalParam):
                idx = int(np.clip(np.floor(value), 0, len(p.choices) - 1))
                config[p.name] = p.choices[idx]
            else:
                raise TypeError(f"Unknown parameter type: {p!r}")
        return config
