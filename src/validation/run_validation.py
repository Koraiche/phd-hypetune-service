"""Validate PSO/GWO/DE on synthetic benchmark functions before trusting them
on the real, hardware-measured Phase 2 objective (article_draft.md Section
3.8). Produces the data and figures for Section 5.2.

This step has no hardware dependency at all — it is pure numerical
optimization against closed-form functions — so it can (and should) be run
and finalized before Raspberry Pi access is available.

Usage:
    python src/validation/run_validation.py
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
from phase2_search import de, gwo, pso  # noqa: E402
from validation.synthetic_functions import FUNCTIONS  # noqa: E402

# Fixed categorical assignment, consistent across every figure this project
# produces — colors identify the algorithm, never a re-cycled/re-ranked hue.
# Palette: dataviz skill's reference categorical slots 1/2/3 (blue/orange/
# aqua), documented as colorblind-safe all-pairs in both light and dark mode.
ALGORITHMS = [
    ("PSO", pso.optimize, "#2a78d6"),
    ("GWO", gwo.optimize, "#eb6834"),
    ("DE", de.optimize, "#1baf7a"),
]

POP_SIZE = 20
MAX_ITER = 200
N_RUNS = 10
BASE_SEED = 42
LOG_FLOOR = 1e-12  # convergence values are clipped to this before log-scaling

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
    ax.xaxis.label.set_color(SECONDARY_INK)
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

        convergences: dict[str, np.ndarray] = {}
        finals: dict[str, np.ndarray] = {}

        for algo_name, algo_fn, _color in ALGORITHMS:
            runs_conv, runs_final = [], []
            for run in range(N_RUNS):
                res = algo_fn(fn, lb, ub, dim, pop_size=POP_SIZE, max_iter=MAX_ITER, seed=BASE_SEED + run)
                runs_conv.append(res.convergence)
                runs_final.append(res.best_cost)
            convergences[algo_name] = np.array(runs_conv)
            finals[algo_name] = np.array(runs_final)

            summary_rows.append({
                "function": name,
                "algorithm": algo_name,
                "best": float(np.min(runs_final)),
                "median": float(np.median(runs_final)),
                "worst": float(np.max(runs_final)),
                "std": float(np.std(runs_final)),
                "global_min": gmin,
            })

        # --- convergence: CSV (median trajectory per algorithm) ---
        medians = {a: np.median(convergences[a], axis=0) for a, _, _ in ALGORITHMS}
        conv_csv = results_dir / f"convergence_{slug}.csv"
        with open(conv_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration"] + [a for a, _, _ in ALGORITHMS])
            for it in range(MAX_ITER):
                writer.writerow([it + 1] + [medians[a][it] for a, _, _ in ALGORITHMS])

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        fig.patch.set_facecolor(CHART_SURFACE)
        for algo_name, _, color in ALGORITHMS:
            y = np.clip(medians[algo_name], LOG_FLOOR, None)
            ax.plot(range(1, MAX_ITER + 1), y, color=color, linewidth=2, label=algo_name)
        ax.set_yscale("log")
        _style_axes(ax)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"Best cost, median of {N_RUNS} runs (log scale)")
        ax.set_title(f"Convergence — {name}", fontsize=11, loc="left", color=PRIMARY_INK)
        ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK)
        fig.tight_layout()
        fig.savefig(figures_dir / f"convergence_{slug}.png", facecolor=fig.get_facecolor())
        plt.close(fig)

        # --- robustness: CSV (final best cost per run) + boxplot ---
        final_csv = results_dir / f"final_costs_{slug}.csv"
        with open(final_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run"] + [a for a, _, _ in ALGORITHMS])
            for run in range(N_RUNS):
                writer.writerow([run + 1] + [finals[a][run] for a, _, _ in ALGORITHMS])

        labels = [a for a, _, _ in ALGORITHMS]
        data = [finals[a] for a in labels]
        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
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
        for patch, algo_name in zip(bp["boxes"], labels):
            color = dict((a, c) for a, _, c in ALGORITHMS)[algo_name]
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        _style_axes(ax)
        ax.set_ylabel(f"Best cost found ({N_RUNS} independent runs)")
        ax.set_title(f"Robustness — {name}", fontsize=11, loc="left", color=PRIMARY_INK)
        fig.tight_layout()
        fig.savefig(figures_dir / f"boxplot_{slug}.png", facecolor=fig.get_facecolor())
        plt.close(fig)

        best_str = ", ".join(f"{a}={np.min(finals[a]):.4g}" for a, _, _ in ALGORITHMS)
        print(f"[validate] {name} (global min={gmin}): best-of-10-runs {best_str}")

    summary_csv = results_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[validate] wrote {summary_csv}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_all(repo_root / "results" / "validation", repo_root / "results" / "figures")
