"""REAL Phase 2 search on actual Raspberry Pi 5 hardware — the production
run this project has been building toward. Uses LlamaCppEvaluator (real
llama-bench + llama-perplexity measurements, real PMIC energy, real peak
RSS) instead of the synthetic MockLlamaCppEvaluator.

Adaptive budget: real evaluations cost real seconds (unlike the
millisecond-scale synthetic/mock evaluations), so this script calibrates
per-evaluation wall-clock cost first (a handful of random samples, which
also serve as normalization-range calibration per Section 3.6), then picks
a population/iteration budget that targets a bounded total runtime instead
of reusing the synthetic validation's fixed 20x200 budget — directly
implementing the cost-argument in article_draft.md Section 3.7.

Usage:
    python src/phase2_search/run_real_phase2.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines import grid_search, random_search  # noqa: E402
from phase2_search import de, gwo, pso  # noqa: E402
from phase2_search.llama_cpp_evaluator import LlamaCppEvaluator  # noqa: E402
from phase2_search.objective import calibrate_normalization_ranges, make_objective  # noqa: E402
from phase2_search.parameter_space import CategoricalParam, IntParam, ParameterSpace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BIN_DIR = Path("/home/fkoraiche/llama.cpp/build/bin")
BASE_MODEL_DIR = REPO_ROOT / "models"
QUALITY_CORPUS = REPO_ROOT / "data" / "prompts" / "wikitext2_quality_slice.txt"

N_CALIBRATION_SAMPLES = 12
TARGET_MINUTES_PER_ALGO = 40  # per algorithm, per retained variant
MIN_POP, MIN_ITER = 6, 8
N_RUNS = 3  # independent runs per algorithm (fewer than synthetic's 10 — real cost)
BASE_SEED = 100

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
        # Smaller than the mock rehearsal's 512-4096: n_ctx maps to llama-bench's
        # -p (n_prompt, see llama_cpp_evaluator.py), and bench wall-clock cost
        # scales with it. Kept modest to keep per-evaluation cost tractable for
        # a real overnight search budget.
        IntParam("n_ctx", 128, 2048),
        CategoricalParam("kv_cache_type", ["f16", "q8_0", "q4_0"]),
    ])


def run_variant(variant_name: str, results_dir: Path, figures_dir: Path) -> None:
    print(f"\n{'=' * 70}\nREAL PHASE 2 SEARCH — variant {variant_name}\n{'=' * 70}")
    model_path = BASE_MODEL_DIR / f"tinyllama-1.1b-{variant_name}.gguf"
    if not model_path.exists():
        print(f"[real-phase2] SKIP {variant_name}: {model_path} not found")
        return

    evaluator = LlamaCppEvaluator(
        bench_bin=str(BIN_DIR / "llama-bench"),
        perplexity_bin=str(BIN_DIR / "llama-perplexity"),
        model_gguf=model_path,
        quality_corpus=QUALITY_CORPUS,
    )
    space = build_space()

    print(f"[real-phase2] calibrating normalization ranges + per-eval cost ({N_CALIBRATION_SAMPLES} samples)...")
    t0 = time.monotonic()
    ranges = calibrate_normalization_ranges(space, evaluator, n_samples=N_CALIBRATION_SAMPLES, seed=BASE_SEED)
    calib_elapsed = time.monotonic() - t0
    per_eval_s = calib_elapsed / N_CALIBRATION_SAMPLES
    print(f"[real-phase2] ranges: {ranges}")
    print(f"[real-phase2] measured per-eval cost: {per_eval_s:.2f}s")

    objective = make_objective(space, evaluator, ranges)
    pop = MIN_POP
    target_evals = max(MIN_POP * MIN_ITER, int((TARGET_MINUTES_PER_ALGO * 60) / per_eval_s))
    iters = max(MIN_ITER, target_evals // pop)
    print(f"[real-phase2] budget: population={pop}, iterations={iters} "
          f"(~{pop * iters * per_eval_s / 60:.1f} min/run x {N_RUNS} runs x 3 algorithms)")

    variant_results_dir = results_dir / variant_name
    variant_results_dir.mkdir(parents=True, exist_ok=True)

    convergences, finals, best_overall = {}, {}, {}
    for algo_name, algo_fn, _color in ALGORITHMS:
        print(f"[real-phase2] --- {algo_name} ---")
        runs_conv, runs_final = [], []
        run_best_x, run_best_cost = None, np.inf
        for run in range(N_RUNS):
            t_run = time.monotonic()
            res = algo_fn(objective, space.lb, space.ub, space.dim, pop_size=pop, max_iter=iters, seed=BASE_SEED + run)
            print(f"[real-phase2] {algo_name} run {run + 1}/{N_RUNS}: best={res.best_cost:.4f} "
                  f"({time.monotonic() - t_run:.0f}s)")
            runs_conv.append(res.convergence)
            runs_final.append(res.best_cost)
            if res.best_cost < run_best_cost:
                run_best_cost = res.best_cost
                run_best_x = res.best_x
        convergences[algo_name] = np.array(runs_conv)
        finals[algo_name] = np.array(runs_final)
        best_overall[algo_name] = (run_best_x, run_best_cost)

    # Baselines at the matched total-evaluation budget
    budget = pop * iters
    print(f"[real-phase2] --- baselines (budget={budget}) ---")
    t_r = time.monotonic()
    random_res = random_search.optimize(objective, space.lb, space.ub, space.dim, budget=budget, seed=BASE_SEED)
    print(f"[real-phase2] random search: best={random_res.best_cost:.4f} ({time.monotonic() - t_r:.0f}s)")
    t_g = time.monotonic()
    grid_res = grid_search.optimize(objective, space.lb, space.ub, space.dim, budget=budget, seed=BASE_SEED)
    print(f"[real-phase2] grid search: best={grid_res.best_cost:.4f}, "
          f"realized_evals={len(grid_res.convergence)} ({time.monotonic() - t_g:.0f}s)")

    # --- persist raw data ---
    conv_csv = variant_results_dir / "convergence.csv"
    with open(conv_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration"] + [a for a, _, _ in ALGORITHMS])
        medians = {a: np.median(convergences[a], axis=0) for a, _, _ in ALGORITHMS}
        for it in range(iters):
            writer.writerow([it + 1] + [medians[a][it] for a, _, _ in ALGORITHMS])

    final_csv = variant_results_dir / "final_costs.csv"
    with open(final_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run"] + [a for a, _, _ in ALGORITHMS])
        for run in range(N_RUNS):
            writer.writerow([run + 1] + [finals[a][run] for a, _, _ in ALGORITHMS])

    best_csv = variant_results_dir / "best_configs.csv"
    with open(best_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "cost", "threads", "n_batch", "n_ctx", "kv_cache_type",
                          "latency_ms_per_token", "quality_neg_ppl", "peak_rss_mb", "energy_j_per_token", "feasible"])
        for method_name, best_x, best_cost in (
            [(a, *best_overall[a]) for a, _, _ in ALGORITHMS]
            + [("Random", random_res.best_x, random_res.best_cost), ("Grid", grid_res.best_x, grid_res.best_cost)]
        ):
            config = space.decode(best_x)
            raw = evaluator.evaluate(config)
            writer.writerow([
                method_name, best_cost, config["threads"], config["n_batch"], config["n_ctx"], config["kv_cache_type"],
                round(raw.latency_ms_per_token, 2), round(raw.quality, 4), round(raw.peak_rss_mb, 1),
                round(raw.energy_j_per_token, 4), raw.feasible,
            ])
            print(f"[real-phase2] {method_name} best: cost={best_cost:.4f} config={config} "
                  f"latency={raw.latency_ms_per_token:.1f}ms/tok mem={raw.peak_rss_mb:.0f}MB "
                  f"energy={raw.energy_j_per_token:.3f}J/tok")

    meta = {
        "variant": variant_name,
        "population": pop,
        "iterations": iters,
        "n_runs": N_RUNS,
        "per_eval_cost_s": per_eval_s,
        "budget": budget,
        "grid_realized_evals": len(grid_res.convergence),
        "normalization_ranges": {
            "latency": list(ranges.latency), "quality": list(ranges.quality),
            "memory": list(ranges.memory), "energy": list(ranges.energy),
        },
    }
    with open(variant_results_dir / "meta.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    # --- figures ---
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor(CHART_SURFACE)
    for algo_name, _, color in ALGORITHMS:
        ax.plot(range(1, iters + 1), medians[algo_name], color=color, linewidth=2, label=algo_name)
    _style_axes(ax)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"Best cost, median of {N_RUNS} runs")
    ax.set_title(f"REAL Phase 2 convergence — {variant_name}", fontsize=10.5, loc="left", color=PRIMARY_INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=SECONDARY_INK)
    fig.tight_layout()
    fig.savefig(figures_dir / f"real_phase2_convergence_{variant_name}.png", facecolor=fig.get_facecolor())
    plt.close(fig)

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
    ax.set_ylabel(f"Best cost found ({N_RUNS} runs)")
    ax.set_title(f"REAL Phase 2 robustness — {variant_name}", fontsize=10.5, loc="left", color=PRIMARY_INK)
    fig.tight_layout()
    fig.savefig(figures_dir / f"real_phase2_boxplot_{variant_name}.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[real-phase2] {variant_name} DONE. Wrote {variant_results_dir}/")


def main() -> None:
    retained_path = REPO_ROOT / "results" / "phase1_retained.yaml"
    with open(retained_path) as f:
        retained = yaml.safe_load(f)["retained_variants"]
    print(f"[real-phase2] retained variants from Phase 1: {retained}")

    results_dir = REPO_ROOT / "results" / "phase2_runs"
    figures_dir = REPO_ROOT / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    for variant_name in retained:
        run_variant(variant_name, results_dir, figures_dir)

    print("\n[real-phase2] ALL VARIANTS DONE.")


if __name__ == "__main__":
    main()
