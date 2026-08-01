"""End-to-end Phase 2 PLUMBING REHEARSAL using the synthetic MockLlamaCppEvaluator.

============================== DISCLAIMER ==================================
Every number this script produces comes from mock_evaluator.py's made-up
cost surface, NOT from a real TinyLlama run or real Raspberry Pi hardware.
Its purpose is narrow: prove that parameter encode/decode, feasibility
rejection, normalization calibration, and PSO/GWO/DE all wire together
correctly end to end, BEFORE the same code is pointed at a real Evaluator
backed by llama.cpp on the Pi. Nothing printed or plotted here may be cited
as a real Section 5 result in article_draft.md — see the explicitly labeled
"mock rehearsal" subsection where these numbers are discussed.
==============================================================================

Usage:
    python src/phase2_search/run_mock_rehearsal.py
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
from phase2_search.mock_evaluator import MockLlamaCppEvaluator  # noqa: E402
from phase2_search.objective import calibrate_normalization_ranges, make_objective  # noqa: E402
from phase2_search.parameter_space import CategoricalParam, IntParam, ParameterSpace  # noqa: E402

POP_SIZE = 20
MAX_ITER = 100
N_RUNS = 10
BASE_SEED = 7

ALGORITHMS = [
    ("PSO", pso.optimize, "#2a78d6"),
    ("GWO", gwo.optimize, "#eb6834"),
    ("DE", de.optimize, "#1baf7a"),
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
    ax.xaxis.label.set_color(SECONDARY_INK)
    ax.yaxis.label.set_color(SECONDARY_INK)


def build_space() -> ParameterSpace:
    return ParameterSpace([
        IntParam("threads", 1, 4),
        IntParam("n_batch", 32, 512),
        IntParam("n_ctx", 512, 4096),
        CategoricalParam("kv_cache_type", ["f16", "q8_0", "q4_0"]),
    ])


def main() -> None:
    print("=" * 78)
    print("MOCK PHASE 2 REHEARSAL - synthetic evaluator, not real hardware data.")
    print("=" * 78)

    repo_root = Path(__file__).resolve().parent.parent.parent
    results_dir = repo_root / "results" / "mock_phase2"
    figures_dir = repo_root / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    space = build_space()
    evaluator = MockLlamaCppEvaluator()
    ranges = calibrate_normalization_ranges(space, evaluator, n_samples=300, seed=0)
    print(f"[mock-rehearsal] calibrated ranges: {ranges}")
    objective = make_objective(space, evaluator, ranges)

    convergences = {}
    finals = {}
    best_overall = {}

    for algo_name, algo_fn, _color in ALGORITHMS:
        runs_conv, runs_final = [], []
        run_best_x, run_best_cost = None, np.inf
        for run in range(N_RUNS):
            res = algo_fn(objective, space.lb, space.ub, space.dim, pop_size=POP_SIZE, max_iter=MAX_ITER, seed=BASE_SEED + run)
            runs_conv.append(res.convergence)
            runs_final.append(res.best_cost)
            if res.best_cost < run_best_cost:
                run_best_cost = res.best_cost
                run_best_x = res.best_x
        convergences[algo_name] = np.array(runs_conv)
        finals[algo_name] = np.array(runs_final)
        best_overall[algo_name] = (run_best_x, run_best_cost)

    # --- convergence figure ---
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor(CHART_SURFACE)
    for algo_name, _, color in ALGORITHMS:
        median_curve = np.median(convergences[algo_name], axis=0)
        ax.plot(range(1, MAX_ITER + 1), median_curve, color=color, linewidth=2, label=algo_name)
    _style_axes(ax)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"Best cost, median of {N_RUNS} runs")
    ax.set_title("MOCK rehearsal — convergence (synthetic objective)", fontsize=10.5, loc="left", color=PRIMARY_INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK)
    fig.tight_layout()
    fig.savefig(figures_dir / "mock_phase2_convergence.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    # --- robustness boxplot ---
    labels = [a for a, _, _ in ALGORITHMS]
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    fig.patch.set_facecolor(CHART_SURFACE)
    bp = ax.boxplot(
        [finals[a] for a in labels], tick_labels=labels, patch_artist=True, widths=0.5,
        medianprops=dict(color=PRIMARY_INK, linewidth=1.5),
        whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
        flierprops=dict(markeredgecolor=MUTED, markersize=4),
    )
    color_by_algo = {a: c for a, _, c in ALGORITHMS}
    for patch, algo_name in zip(bp["boxes"], labels):
        patch.set_facecolor(color_by_algo[algo_name])
        patch.set_alpha(0.35)
        patch.set_edgecolor(color_by_algo[algo_name])
    _style_axes(ax)
    ax.set_ylabel(f"Best cost found ({N_RUNS} independent runs)")
    ax.set_title("MOCK rehearsal — robustness (synthetic objective)", fontsize=10.5, loc="left", color=PRIMARY_INK)
    fig.tight_layout()
    fig.savefig(figures_dir / "mock_phase2_boxplot.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    # --- best-config table ---
    table_csv = results_dir / "best_configs.csv"
    with open(table_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "cost", "threads", "n_batch", "n_ctx", "kv_cache_type",
                          "latency_ms_per_token", "quality", "peak_rss_mb", "energy_j_per_token", "feasible"])
        for algo_name, _, _ in ALGORITHMS:
            best_x, best_cost = best_overall[algo_name]
            config = space.decode(best_x)
            raw = evaluator.evaluate(config)
            writer.writerow([
                algo_name, best_cost, config["threads"], config["n_batch"], config["n_ctx"], config["kv_cache_type"],
                round(raw.latency_ms_per_token, 2), round(raw.quality, 4), round(raw.peak_rss_mb, 1),
                round(raw.energy_j_per_token, 4), raw.feasible,
            ])
            print(f"[mock-rehearsal] {algo_name} best: cost={best_cost:.4f} config={config} "
                  f"latency={raw.latency_ms_per_token:.1f}ms/tok mem={raw.peak_rss_mb:.0f}MB")

    print(f"[mock-rehearsal] wrote {table_csv}")

    # Brute-force sanity check: with only 4 parameters and cheap evaluations,
    # a fine grid scan is affordable here (unlike the real dim=10 Sphere case
    # in Section 5.2.1) and lets us confirm the optimizers actually found the
    # true optimum of this mock surface, rather than just agreeing with each
    # other by coincidence.
    import itertools
    best_grid_cost = np.inf
    best_grid_config = None
    for threads in range(1, 5):
        for n_batch in np.linspace(32, 512, 10):
            for n_ctx in np.linspace(512, 4096, 20):
                for kv in ["f16", "q8_0", "q4_0"]:
                    x = np.array([threads, n_batch, n_ctx, {"f16": 0, "q8_0": 1, "q4_0": 2}[kv]])
                    cost = objective(x)
                    if cost < best_grid_cost:
                        best_grid_cost = cost
                        best_grid_config = space.decode(x)
    print(f"[mock-rehearsal] brute-force grid best: cost={best_grid_cost:.4f} config={best_grid_config}")
    algo_best_cost = min(c for _, c in best_overall.values())
    matches = abs(algo_best_cost - best_grid_cost) < 1e-6
    print(f"[mock-rehearsal] optimizers matched brute-force optimum: {matches}")

    print("[mock-rehearsal] DONE. Remember: synthetic data - see disclaimer above.")


if __name__ == "__main__":
    main()
