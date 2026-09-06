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

POST-EXPERIMENT HARDENING 2026-09-07 (not used to produce the checked-in
article results): search measurements now use repeated llama-bench runs,
selected candidates receive a separate higher-repetition verification before
any winner is reported, calibration uses 24 samples, every hardware call is
temperature-gated, and deterministic objective-call traces make every
optimizer and baseline resumable after a process or device crash.

Usage:
    python3 -u src/phase2_search/run_real_phase2.py
"""
from __future__ import annotations

import csv
import os
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
from measurement.thermal_guard import ThermalGuard  # noqa: E402
from phase2_search import de, gwo, pso  # noqa: E402
from phase2_search.checkpoint import run_with_checkpoint  # noqa: E402
from phase2_search.llama_cpp_evaluator import LlamaCppEvaluator  # noqa: E402
from phase2_search.objective import (  # noqa: E402
    NormalizationRanges,
    calibrate_normalization_ranges,
    make_objective,
    score_evaluation,
)
from phase2_search.parameter_space import CategoricalParam, IntParam, ParameterSpace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BIN_DIR = Path("/home/fkoraiche/llama.cpp/build/bin")
BASE_MODEL_DIR = REPO_ROOT / "models"
QUALITY_CORPUS = REPO_ROOT / "data" / "prompts" / "wikitext2_quality_slice.txt"

N_CALIBRATION_SAMPLES = 24
TARGET_MINUTES_PER_ALGO = 12  # per algorithm, per retained variant -- deliberately tight
MIN_POP, MIN_ITER = 4, 5  # floor of 20 evaluations, not 48
N_RUNS = 2  # reduced from a synthetic-validation-scale 10; real cost, disclosed
BASE_SEED = 100
KV_CACHE_TYPES = ["f16", "q8_0", "q4_0"]
SEARCH_BENCH_REPETITIONS = 3
VERIFICATION_BENCH_REPETITIONS = 5
VERIFICATION_ABS_TOLERANCE = 0.03
VERIFICATION_REL_TOLERANCE = 0.25

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


def write_yaml_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(temporary, "w") as f:
        yaml.safe_dump(value, f, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def verification_is_consistent(search_cost: float, verified_cost: float) -> tuple[bool, float, float]:
    """Apply the predeclared absolute/relative winner-verification rule."""
    if not (np.isfinite(search_cost) and np.isfinite(verified_cost)):
        return False, np.inf, np.inf
    absolute_gap = abs(verified_cost - search_cost)
    scale = max(abs(verified_cost), abs(search_cost), 1e-12)
    relative_gap = absolute_gap / scale
    consistent = (
        absolute_gap <= VERIFICATION_ABS_TOLERANCE
        or relative_gap <= VERIFICATION_REL_TOLERANCE
    )
    return consistent, absolute_gap, relative_gap


def checkpoint_metadata(
    variant_name: str,
    method: str,
    seed: int,
    space: ParameterSpace,
    ranges,
    **budget,
) -> dict:
    return {
        "schema_version": 1,
        "variant": variant_name,
        "method": method,
        "seed": seed,
        "bounds": {"lower": space.lb.tolist(), "upper": space.ub.tolist()},
        "normalization_ranges": {
            "latency": list(ranges.latency),
            "quality": list(ranges.quality),
            "memory": list(ranges.memory),
            "energy": list(ranges.energy),
        },
        "search_bench_repetitions": SEARCH_BENCH_REPETITIONS,
        **budget,
    }


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

    variant_results_dir = results_dir / variant_name
    variant_results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = variant_results_dir / "checkpoints"

    thermal_guard = ThermalGuard(logger=log)
    evaluator = LlamaCppEvaluator(
        bench_bin=str(BIN_DIR / "llama-bench"),
        perplexity_bin=str(BIN_DIR / "llama-perplexity"),
        model_gguf=model_path,
        quality_corpus=QUALITY_CORPUS,
        bench_repetitions=SEARCH_BENCH_REPETITIONS,
        thermal_guard=thermal_guard,
    )
    space = build_space()

    calibration_path = checkpoint_root / "calibration.yaml"
    expected_settings = {
        "samples": N_CALIBRATION_SAMPLES,
        "search_bench_repetitions": SEARCH_BENCH_REPETITIONS,
    }
    if calibration_path.exists():
        with open(calibration_path) as f:
            calibration = yaml.safe_load(f)
        if calibration.get("settings") != expected_settings:
            raise RuntimeError(
                f"calibration checkpoint settings differ in {calibration_path}; "
                "move the old checkpoints aside or restore the original settings"
            )
        saved_ranges = calibration["normalization_ranges"]
        ranges = NormalizationRanges(
            latency=tuple(saved_ranges["latency"]),
            quality=tuple(saved_ranges["quality"]),
            memory=tuple(saved_ranges["memory"]),
            energy=tuple(saved_ranges["energy"]),
        )
        per_eval_s = float(calibration["per_eval_cost_s"])
        evaluator._quality_cache.update(calibration["quality_cache"])
        log(f"[real-phase2] loaded calibration and quality cache from {calibration_path}")
    else:
        log("[real-phase2] pre-computing quality per KV-cache type (one-time cost, cached)...")
        precompute_quality(evaluator)
        log(
            f"[real-phase2] calibrating normalization ranges + per-eval cost "
            f"({N_CALIBRATION_SAMPLES} warm-cache samples)..."
        )
        t0 = time.monotonic()
        ranges = calibrate_normalization_ranges(
            space, evaluator, n_samples=N_CALIBRATION_SAMPLES, seed=BASE_SEED
        )
        per_eval_s = (time.monotonic() - t0) / N_CALIBRATION_SAMPLES
        calibration = {
            "settings": expected_settings,
            "per_eval_cost_s": per_eval_s,
            "quality_cache": evaluator._quality_cache,
            "normalization_ranges": {
                "latency": list(ranges.latency),
                "quality": list(ranges.quality),
                "memory": list(ranges.memory),
                "energy": list(ranges.energy),
            },
        }
        write_yaml_atomic(calibration_path, calibration)
        log(f"[real-phase2] wrote reusable calibration checkpoint {calibration_path}")
    log(f"[real-phase2] ranges: {ranges}")
    log(f"[real-phase2] measured per-eval cost (warm cache): {per_eval_s:.2f}s")

    objective = make_objective(space, evaluator, ranges)
    pop = MIN_POP
    target_evals = max(MIN_POP * MIN_ITER, int((TARGET_MINUTES_PER_ALGO * 60) / per_eval_s))
    iters = max(MIN_ITER, target_evals // pop)
    total_est_min = pop * iters * per_eval_s * N_RUNS * len(ALGORITHMS) / 60
    log(f"[real-phase2] budget: population={pop}, iterations={iters}, {N_RUNS} runs x {len(ALGORITHMS)} algorithms "
        f"(~{total_est_min:.0f} min estimated for this variant)")

    convergences, finals, best_overall = {}, {}, {}
    for algo_name, algo_fn, _color in ALGORITHMS:
        log(f"[real-phase2] --- {algo_name} ---")
        runs_conv, runs_final = [], []
        run_best_x, run_best_cost = None, np.inf
        for run in range(N_RUNS):
            t_run = time.monotonic()
            seed = BASE_SEED + run
            res, replayed, loaded = run_with_checkpoint(
                algo_fn,
                objective,
                checkpoint_root / algo_name.lower() / f"run_{run + 1}",
                checkpoint_metadata(
                    variant_name, algo_name, seed, space, ranges,
                    population=pop, iterations=iters,
                ),
                lb=space.lb,
                ub=space.ub,
                dim=space.dim,
                pop_size=pop,
                max_iter=iters,
                seed=seed,
            )
            resume_note = "loaded completed checkpoint" if loaded else f"replayed {replayed} evaluations"
            log(
                f"[real-phase2] {algo_name} run {run + 1}/{N_RUNS}: best={res.best_cost:.4f} "
                f"({time.monotonic() - t_run:.0f}s; {resume_note})"
            )
            runs_conv.append(res.convergence)
            runs_final.append(res.best_cost)
            if res.best_cost < run_best_cost:
                run_best_cost = res.best_cost
                run_best_x = res.best_x

            partial_csv = variant_results_dir / "final_costs_partial.csv"
            partial_names = list(finals) + [algo_name]
            partial_values = {**finals, algo_name: np.asarray(runs_final)}
            with open(partial_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["run"] + partial_names)
                for completed_run in range(N_RUNS):
                    writer.writerow(
                        [completed_run + 1]
                        + [
                            partial_values[name][completed_run]
                            if completed_run < len(partial_values[name]) else ""
                            for name in partial_names
                        ]
                    )
            log(f"[real-phase2] wrote {partial_csv} (incremental progress marker)")
        convergences[algo_name] = np.array(runs_conv)
        finals[algo_name] = np.array(runs_final)
        best_overall[algo_name] = (run_best_x, run_best_cost)

    # Baselines at the matched total-evaluation budget
    budget = pop * iters
    log(f"[real-phase2] --- baselines (budget={budget}) ---")
    t_r = time.monotonic()
    random_res, random_replayed, random_loaded = run_with_checkpoint(
        random_search.optimize,
        objective,
        checkpoint_root / "random",
        checkpoint_metadata(variant_name, "Random", BASE_SEED, space, ranges, budget=budget),
        lb=space.lb, ub=space.ub, dim=space.dim, budget=budget, seed=BASE_SEED,
    )
    random_note = "loaded completed checkpoint" if random_loaded else f"replayed {random_replayed} evaluations"
    log(
        f"[real-phase2] random search: best={random_res.best_cost:.4f} "
        f"({time.monotonic() - t_r:.0f}s; {random_note})"
    )
    t_g = time.monotonic()
    grid_res, grid_replayed, grid_loaded = run_with_checkpoint(
        grid_search.optimize,
        objective,
        checkpoint_root / "grid",
        checkpoint_metadata(variant_name, "Grid", BASE_SEED, space, ranges, budget=budget),
        lb=space.lb, ub=space.ub, dim=space.dim, budget=budget, seed=BASE_SEED,
    )
    grid_note = "loaded completed checkpoint" if grid_loaded else f"replayed {grid_replayed} evaluations"
    log(f"[real-phase2] grid search: best={grid_res.best_cost:.4f}, "
        f"realized_evals={len(grid_res.convergence)} ({time.monotonic() - t_g:.0f}s; {grid_note})")

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
    verification_evaluator = LlamaCppEvaluator(
        bench_bin=evaluator.bench_bin,
        perplexity_bin=evaluator.perplexity_bin,
        model_gguf=evaluator.model_gguf,
        quality_corpus=evaluator.quality_corpus,
        quality_ctx_size=evaluator.quality_ctx_size,
        bench_repetitions=VERIFICATION_BENCH_REPETITIONS,
        n_gen=evaluator.n_gen,
        ram_ceiling_mb=evaluator.ram_ceiling_mb,
        thermal_guard=thermal_guard,
        _quality_cache=dict(evaluator._quality_cache),
    )
    verification_rows = []
    with open(best_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "cost", "search_cost", "verified_cost", "absolute_gap", "relative_gap",
            "verification_status", "verification_repetitions", "threads", "n_batch", "n_ctx", "kv_cache_type",
            "latency_ms_per_token", "quality_neg_ppl", "peak_rss_mb", "energy_j_per_token", "feasible",
        ])
        for method_name, best_x, search_cost in (
            [(a, *best_overall[a]) for a, _, _ in ALGORITHMS]
            + [("Random", random_res.best_x, random_res.best_cost), ("Grid", grid_res.best_x, grid_res.best_cost)]
        ):
            config = space.decode(best_x)
            raw = verification_evaluator.evaluate(config)
            verified_cost = score_evaluation(raw, ranges)
            consistent, absolute_gap, relative_gap = verification_is_consistent(search_cost, verified_cost)
            status = "verified" if consistent else "unstable"
            writer.writerow([
                method_name, verified_cost, search_cost, verified_cost, absolute_gap, relative_gap,
                status, VERIFICATION_BENCH_REPETITIONS,
                config["threads"], config["n_batch"], config["n_ctx"], config["kv_cache_type"],
                round(raw.latency_ms_per_token, 2), round(raw.quality, 4), round(raw.peak_rss_mb, 1),
                round(raw.energy_j_per_token, 4), raw.feasible,
            ])
            verification_rows.append({
                "method": method_name,
                "search_cost": float(search_cost),
                "verified_cost": float(verified_cost),
                "verification_status": status,
                "absolute_gap": float(absolute_gap),
                "relative_gap": float(relative_gap),
                "config": config,
            })
            log(f"[real-phase2] {method_name} candidate: search={search_cost:.4f}, "
                f"verified={verified_cost:.4f}, status={status}, config={config} "
                f"latency={raw.latency_ms_per_token:.1f}ms/tok mem={raw.peak_rss_mb:.0f}MB "
                f"energy={raw.energy_j_per_token:.3f}J/tok")

    verified_candidates = [row for row in verification_rows if row["verification_status"] == "verified"]
    winner_summary = {
        "status": "verified_winner" if verified_candidates else "no_verified_winner",
        "verification_absolute_tolerance": VERIFICATION_ABS_TOLERANCE,
        "verification_relative_tolerance": VERIFICATION_REL_TOLERANCE,
        "verification_repetitions": VERIFICATION_BENCH_REPETITIONS,
    }
    if verified_candidates:
        winner = min(verified_candidates, key=lambda row: row["verified_cost"])
        winner_summary["winner"] = winner
        log(f"[real-phase2] verified winner: {winner['method']} at {winner['verified_cost']:.4f}")
    else:
        log("[real-phase2] no candidate passed verification; no winner will be reported")
    write_yaml_atomic(variant_results_dir / "winner.yaml", winner_summary)

    meta = {
        "variant": variant_name,
        "population": pop,
        "iterations": iters,
        "n_runs": N_RUNS,
        "per_eval_cost_s": per_eval_s,
        "budget": budget,
        "grid_realized_evals": len(grid_res.convergence),
        "calibration_samples": N_CALIBRATION_SAMPLES,
        "search_bench_repetitions": SEARCH_BENCH_REPETITIONS,
        "verification_bench_repetitions": VERIFICATION_BENCH_REPETITIONS,
        "verification_absolute_tolerance": VERIFICATION_ABS_TOLERANCE,
        "verification_relative_tolerance": VERIFICATION_REL_TOLERANCE,
        "normalization_ranges": {
            "latency": list(ranges.latency), "quality": list(ranges.quality),
            "memory": list(ranges.memory), "energy": list(ranges.energy),
        },
    }
    write_yaml_atomic(variant_results_dir / "meta.yaml", meta)

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
# SCOPE DECISION, documented rather than silently narrowed: the first pass
# searched 3 variants spanning the retained frontier — Q2_K (fast/small
# extreme), Q4_K_M (the K-quant "sweet spot"), and Q8_0 (quality extreme) —
# all of which are now complete (see results/phase2_runs/{Q2_K,Q4_K_M,Q8_0}/).
# EXTENSION (2026-08-17): with the pipeline proven stable across a full
# unattended run (including surviving a mid-run Pi crash with no data loss,
# since results were already written to disk), extending to the remaining 4
# Pareto-retained variants to get full 7/7 coverage rather than 3/7.
PHASE2_VARIANT_SUBSET = ["Q3_K_M", "Q4_0", "Q5_K_M", "Q6_K"]


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
