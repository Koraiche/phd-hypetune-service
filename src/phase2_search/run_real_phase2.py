"""REAL Phase 2 search on actual Raspberry Pi 5 hardware.

REVISED 2026-08-16 after a first attempt ran 6+ hours on a single variant
with zero incremental visibility and had to be killed. Root cause: the
original calibration step estimated per-evaluation cost by averaging over
random samples that included the expensive "cold" per-KV-cache-type quality
calculation (each ~5-10 min) alongside cheap warm-cache evaluations (~20s),
inflating the average; combined with a population*iteration floor of 48
evaluations at real per-eval costs of tens of seconds, times 3 runs times 3
algorithms times 3 variants, the total simply never fit in an overnight
budget. Fixed here by:

1. Pre-computing quality for all 3 KV-cache types explicitly and separately,
   BEFORE calibration, so calibration only ever times warm-cache (fast)
   evaluations and gives an accurate, uncontaminated per-eval cost estimate.
2. A much smaller, explicitly time-boxed budget (see TARGET_MINUTES_*).
3. n_ctx search range capped at 1024 (not 2048) to bound worst-case
   per-evaluation cost — larger prompt sizes cost real minutes, confirmed by
   direct measurement (see diag_eval.py output referenced in article).
4. Unbuffered, flushed, incremental logging AND incremental result writes
   (after every run, not just at the end of a whole variant) — the original
   script's silence for 6+ hours was itself a design flaw, not just bad luck.
5. Reduced power-sampling frequency (1.0s instead of 0.2s) — confirmed by
   direct measurement to cut instrumentation overhead to ~6% of a bare call
   rather than a larger, unmeasured overhead at the original interval.

Usage:
    python3 -u src/phase2_search/run_real_phase2.py
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

N_CALIBRATION_SAMPLES = 8
TARGET_MINUTES_PER_ALGO = 12  # per algorithm, per retained variant -- deliberately tight
MIN_POP, MIN_ITER = 4, 5  # floor of 20 evaluations, not 48
N_RUNS = 2  # reduced from a synthetic-validation-scale 10; real cost, disclosed
BASE_SEED = 100
KV_CACHE_TYPES = ["f16", "q8_0", "q4_0"]

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


def log(msg: str) -> None:
    print(msg, flush=True)


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
        IntParam("n_ctx", 128, 1024),  # capped at 1024, not 2048 -- see module docstring
        CategoricalParam("kv_cache_type", ["f16", "q8_0", "q4_0"]),
    ])


def precompute_quality(evaluator: LlamaCppEvaluator) -> None:
    for kv in KV_CACHE_TYPES:
        t0 = time.monotonic()
        q = evaluator._quality_for(kv)
        log(f"[real-phase2]   quality[{kv}] = {q:.4f} ({time.monotonic() - t0:.0f}s)")


def run_variant(variant_name: str, results_dir: Path, figures_dir: Path) -> None:
    log(f"\n{'=' * 70}\nREAL PHASE 2 SEARCH — variant {variant_name}\n{'=' * 70}")
    model_path = BASE_MODEL_DIR / f"tinyllama-1.1b-{variant_name}.gguf"
    if not model_path.exists():
        log(f"[real-phase2] SKIP {variant_name}: {model_path} not found")
        return

    evaluator = LlamaCppEvaluator(
        bench_bin=str(BIN_DIR / "llama-bench"),
        perplexity_bin=str(BIN_DIR / "llama-perplexity"),
        model_gguf=model_path,
        quality_corpus=QUALITY_CORPUS,
    )
    space = build_space()

    log("[real-phase2] pre-computing quality per KV-cache type (one-time cost, cached)...")
    precompute_quality(evaluator)

    log(f"[real-phase2] calibrating normalization ranges + per-eval cost ({N_CALIBRATION_SAMPLES} warm-cache samples)...")
    t0 = time.monotonic()
    ranges = calibrate_normalization_ranges(space, evaluator, n_samples=N_CALIBRATION_SAMPLES, seed=BASE_SEED)
    per_eval_s = (time.monotonic() - t0) / N_CALIBRATION_SAMPLES
    log(f"[real-phase2] ranges: {ranges}")
    log(f"[real-phase2] measured per-eval cost (warm cache): {per_eval_s:.2f}s")

    objective = make_objective(space, evaluator, ranges)
    pop = MIN_POP
    target_evals = max(MIN_POP * MIN_ITER, int((TARGET_MINUTES_PER_ALGO * 60) / per_eval_s))
    iters = max(MIN_ITER, target_evals // pop)
    total_est_min = pop * iters * per_eval_s * N_RUNS * len(ALGORITHMS) / 60
    log(f"[real-phase2] budget: population={pop}, iterations={iters}, {N_RUNS} runs x {len(ALGORITHMS)} algorithms "
        f"(~{total_est_min:.0f} min estimated for this variant)")

    variant_results_dir = results_dir / variant_name
    variant_results_dir.mkdir(parents=True, exist_ok=True)

    convergences, finals, best_overall = {}, {}, {}
    for algo_name, algo_fn, _color in ALGORITHMS:
        log(f"[real-phase2] --- {algo_name} ---")
        runs_conv, runs_final = [], []
        run_best_x, run_best_cost = None, np.inf
        for run in range(N_RUNS):
            t_run = time.monotonic()
            res = algo_fn(objective, space.lb, space.ub, space.dim, pop_size=pop, max_iter=iters, seed=BASE_SEED + run)
            log(f"[real-phase2] {algo_name} run {run + 1}/{N_RUNS}: best={res.best_cost:.4f} "
                f"({time.monotonic() - t_run:.0f}s)")
            runs_conv.append(res.convergence)
            runs_final.append(res.best_cost)
            if res.best_cost < run_best_cost:
                run_best_cost = res.best_cost
                run_best_x = res.best_x
        convergences[algo_name] = np.array(runs_conv)
        finals[algo_name] = np.array(runs_final)
        best_overall[algo_name] = (run_best_x, run_best_cost)

        # Incremental write after EVERY algorithm, not just at the end of the
        # whole variant — the original script's silence for 6+ hours with no
        # partial output was itself the design flaw that forced a kill.
        partial_csv = variant_results_dir / "final_costs_partial.csv"
        with open(partial_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run"] + list(finals.keys()))
            for run in range(N_RUNS):
                writer.writerow([run + 1] + [finals[a][run] if a in finals else "" for a in finals])
        log(f"[real-phase2] wrote {partial_csv} (incremental progress marker)")

    # Baselines at the matched total-evaluation budget
    budget = pop * iters
    log(f"[real-phase2] --- baselines (budget={budget}) ---")
    t_r = time.monotonic()
    random_res = random_search.optimize(objective, space.lb, space.ub, space.dim, budget=budget, seed=BASE_SEED)
    log(f"[real-phase2] random search: best={random_res.best_cost:.4f} ({time.monotonic() - t_r:.0f}s)")
    t_g = time.monotonic()
    grid_res = grid_search.optimize(objective, space.lb, space.ub, space.dim, budget=budget, seed=BASE_SEED)
    log(f"[real-phase2] grid search: best={grid_res.best_cost:.4f}, "
        f"realized_evals={len(grid_res.convergence)} ({time.monotonic() - t_g:.0f}s)")

    # --- persist raw data ---
    medians = {a: np.median(convergences[a], axis=0) for a, _, _ in ALGORITHMS}
    conv_csv = variant_results_dir / "convergence.csv"
    with open(conv_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration"] + [a for a, _, _ in ALGORITHMS])
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
            log(f"[real-phase2] {method_name} best: cost={best_cost:.4f} config={config} "
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

    log(f"[real-phase2] {variant_name} DONE. Wrote {variant_results_dir}/")


# Phase 1 (2026-08-16) found all 7 quant variants Pareto-nondominated —
# quality/speed/memory trade off perfectly monotonically with quantization
# level, so nothing is strictly dominated (see results/phase1_table.csv and
# article_draft.md Section 5.1). Running the real Phase 2 metaheuristic
# search across all 7 is not tractable in a single-night compute budget at
# the per-evaluation costs measured here (real hardware, not synthetic).
# SCOPE DECISION, documented rather than silently narrowed: search 3
# variants spanning the retained frontier — Q2_K (fast/small extreme),
# Q4_K_M (the K-quant "sweet spot" the wider community treats as a default),
# and Q8_0 (quality extreme) — rather than all 7, and even this list is
# processed in priority order so a partial run still yields the single most
# representative variant complete.
PHASE2_VARIANT_SUBSET = ["Q4_K_M", "Q2_K", "Q8_0"]


def main() -> None:
    retained_path = REPO_ROOT / "results" / "phase1_retained.yaml"
    with open(retained_path) as f:
        retained = yaml.safe_load(f)["retained_variants"]
    log(f"[real-phase2] retained variants from Phase 1: {retained}")
    log(f"[real-phase2] running real search on subset (priority order): {PHASE2_VARIANT_SUBSET}")

    results_dir = REPO_ROOT / "results" / "phase2_runs"
    figures_dir = REPO_ROOT / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    for variant_name in PHASE2_VARIANT_SUBSET:
        run_variant(variant_name, results_dir, figures_dir)

    log("\n[real-phase2] ALL VARIANTS DONE.")


if __name__ == "__main__":
    main()