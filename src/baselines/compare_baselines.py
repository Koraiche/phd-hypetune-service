"""Compare PSO/GWO/DE against grid search and random search under a matched
evaluation budget, on the same synthetic functions used for optimizer
validation (article_draft.md Section 3.8/5.2).

This is a METHODOLOGY REHEARSAL, not the real Section 4.3/5.6 result — that
comparison has to run on the actual hardware-measured Phase 2 objective once
Raspberry Pi access is available. What this script *does* legitimately
establish now: that the baseline implementations are correct, and whether
the paper's core motivating claim ("metaheuristics beat naive search under
equal budget") holds at all on hard synthetic problems before betting the
real experiment design on it.

Usage:
    python src/baselines/compare_baselines.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines import grid_search, random_search  # noqa: E402
from phase2_search import de, gwo, pso  # noqa: E402
from validation.synthetic_functions import FUNCTIONS  # noqa: E402

POP_SIZE = 20
MAX_ITER = 200
BUDGET = POP_SIZE * MAX_ITER  # 4000 — matched evaluation budget across all 5 methods
N_RUNS = 10
BASE_SEED = 42

# Fixed categorical order, extending the PSO/GWO/DE colors already used in
# run_validation.py's figures (same entity = same color everywhere) with the
# palette's slots 4-5 for the two new baseline methods.
METHODS = [
    ("PSO", lambda fn, lb, ub, dim, seed: pso.optimize(fn, lb, ub, dim, pop_size=POP_SIZE, max_iter=MAX_ITER, seed=seed), "#2a78d6", True),
    ("GWO", lambda fn, lb, ub, dim, seed: gwo.optimize(fn, lb, ub, dim, pop_size=POP_SIZE, max_iter=MAX_ITER, seed=seed), "#eb6834", True),
    ("DE", lambda fn, lb, ub, dim, seed: de.optimize(fn, lb, ub, dim, pop_size=POP_SIZE, max_iter=MAX_ITER, seed=seed), "#1baf7a", True),
    ("Random", lambda fn, lb, ub, dim, seed: random_search.optimize(fn, lb, ub, dim, budget=BUDGET, seed=seed), "#eda100", True),
    ("Grid", lambda fn, lb, ub, dim, seed: grid_search.optimize(fn, lb, ub, dim, budget=BUDGET, seed=seed), "#e87ba4", False),
]

CHART_SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"


def _style_axes(ax) -> None:
    ax.set_facecolor(CHART_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=SECONDARY_INK, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.title.set_color(PRIMARY_INK)
    ax.yaxis.label.set_color(SECONDARY_INK)


def _slug(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def run_all(results_dir: Path, figures_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for spec in FUNCTIONS:
        name, fn, dim = spec["name"], spec["fn"], spec["dim"]
        lb, ub, gmin = spec["lb"], spec["ub"], spec["global_min"]
        slug = _slug(name)

        finals: dict[str, np.ndarray] = {}
        realized_evals: dict[str, int] = {}

        for method_name, run_fn, _color, is_stochastic in METHODS:
            n_runs = N_RUNS if is_stochastic else 1
            costs = []
            eval_count = None
            for run in range(n_runs):
                res = run_fn(fn, lb, ub, dim, BASE_SEED + run)
                costs.append(res.best_cost)
                eval_count = len(res.convergence)
            finals[method_name] = np.array(costs)
            realized_evals[method_name] = eval_count

            summary_rows.append({
                "function": name,
                "method": method_name,
                "best": float(np.min(costs)),
                "median": float(np.median(costs)),
                "worst": float(np.max(costs)),
                "std": float(np.std(costs)),
                "n_runs": n_runs,
                "nominal_budget": BUDGET,
                "realized_evaluations": eval_count,
                "global_min": gmin,
            })

        final_csv = results_dir / f"final_costs_{slug}.csv"
        with open(final_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run"] + [m for m, _, _, _ in METHODS])
            max_runs = max(len(finals[m]) for m, _, _, _ in METHODS)
            for run in range(max_runs):
                row = [run + 1]
                for m, _, _, _ in METHODS:
                    row.append(finals[m][run] if run < len(finals[m]) else "")
                writer.writerow(row)

        labels = [m for m, _, _, _ in METHODS]
        data = [finals[m] for m in labels]
        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
        fig.patch.set_facecolor(CHART_SURFACE)
        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color=PRIMARY_INK, linewidth=1.5),
            whiskerprops=dict(color=MUTED),
            capprops=dict(color=MUTED),
            flierprops=dict(markeredgecolor=MUTED, markersize=4),
        )
        color_by_method = {m: c for m, _, c, _ in METHODS}
        for patch, method_name in zip(bp["boxes"], labels):
            color = color_by_method[method_name]
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        _style_axes(ax)
        ax.set_ylabel(f"Best cost after {BUDGET} evaluations")
        ax.set_title(
            f"Baseline comparison — {name}  (Grid: n=1, deterministic)",
            fontsize=10.5, loc="left", color=PRIMARY_INK,
        )
        fig.tight_layout()
        fig.savefig(figures_dir / f"baseline_comparison_{slug}.png", facecolor=fig.get_facecolor())
        plt.close(fig)

        grid_evals = realized_evals["Grid"]
        print(
            f"[baselines] {name} (dim={dim}, global min={gmin}): "
            f"grid realized {grid_evals}/{BUDGET} evals "
            f"(resolution {round(grid_evals ** (1.0 / dim))}/axis) -> best={finals['Grid'][0]:.4g}; "
            + ", ".join(f"{m} median={np.median(finals[m]):.4g}" for m in ["Random", "PSO", "GWO", "DE"])
        )

    summary_csv = results_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[baselines] wrote {summary_csv}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_all(repo_root / "results" / "baselines", repo_root / "results" / "figures")
