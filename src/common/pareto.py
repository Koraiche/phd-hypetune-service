"""Generic multi-objective Pareto non-dominated filter.

Used by Phase 1 to select which quantization variants are carried into
Phase 2, per article_draft.md Section 3.4: retain only variants that are not
strictly dominated on every axis (quality, speed, memory) by some other
variant, instead of committing to a single "best" quantization before
runtime tuning is known.
"""
from __future__ import annotations

from typing import Literal, Mapping, Sequence

Direction = Literal["max", "min"]


def dominates(a: Mapping[str, float], b: Mapping[str, float], axes: Mapping[str, Direction]) -> bool:
    """True if candidate `a` dominates `b`: at least as good on every axis,
    strictly better on at least one."""
    at_least_as_good = True
    strictly_better = False
    for axis, direction in axes.items():
        av, bv = a[axis], b[axis]
        if direction == "min":
            av, bv = -av, -bv
        if av < bv:
            at_least_as_good = False
            break
        if av > bv:
            strictly_better = True
    return at_least_as_good and strictly_better


def pareto_front(candidates: Sequence[Mapping], axes: Mapping[str, Direction], id_key: str = "name") -> list:
    """Return the subset of `candidates` not dominated by any other candidate."""
    retained = []
    for cand in candidates:
        if not any(
            other[id_key] != cand[id_key] and dominates(other, cand, axes)
            for other in candidates
        ):
            retained.append(cand)
    return retained
