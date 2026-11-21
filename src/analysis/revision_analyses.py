"""Offline analyses added for the ISI 47011 major revision.

Everything here re-uses data already on disk (no hardware access):

  A. budget_table          evaluation counts, distinct configurations and
                           estimated device time per method (Reviewer 2, pt 7)
  B. noise_decomposition   hardware noise vs. selection bias, from the
                           hardened run's repeated measurements (Reviewer 1, pt 8)
  C. weight_sensitivity    re-scoring the recorded best configurations under
                           alternative objective weights (Reviewer 1, pt 4)
  D. cross_variant         common-scale comparison of all (variant, runtime)
                           pairs: the outer step of the two-phase problem
                           (Reviewer 1, pts 1 and 6)
  E. rank_statistics       Friedman / Wilcoxon tests across variants (Reviewer 1, pt 7)
  F. synthetic_study       noisy, small-budget controlled experiment with a
                           known ground truth (Reviewer 1, pts 5, 7, 8)
  G. energy_sampling       sampling density and truncation bound of the PMIC
                           energy integration (Reviewer 2, pt 5)

Usage:  python src/analysis/revision_analyses.py
Writes CSV/JSON files to results/revision/ and prints the key numbers.
"""
from __future__ import annotations

import collections
import csv
import glob
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from baselines import grid_search, random_search  # noqa: E402
from phase2_search import de, gwo, pso  # noqa: E402
from phase2_search.parameter_space import CategoricalParam, IntParam, ParameterSpace  # noqa: E402

ORIG = ROOT / "results" / "phase2_runs"
HARD = ROOT / "results" / "phase2_runs_hardened"
OUT = ROOT / "results" / "revision"
VARIANTS = ["Q2_K", "Q3_K_M", "Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
METHODS = ["PSO", "GWO", "DE", "Random", "Grid"]
METAHEURISTICS = ["PSO", "GWO", "DE"]
DEFAULT_W = (0.35, 0.35, 0.20, 0.10)  # latency, quality, memory, energy
SPACE = ParameterSpace([
    IntParam("threads", 1, 4),
    IntParam("n_batch", 32, 512),
    IntParam("n_ctx", 128, 1024),
    CategoricalParam("kv_cache_type", ["f16", "q8_0", "q4_0"]),
])
DISCRETE_SPACE_SIZE = 4 * (512 - 32 + 1) * (1024 - 128 + 1) * 3


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def norm(v: float, lo: float, hi: float) -> float:
    return 0.0 if hi <= lo else float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def score(row: dict, ranges: dict, w=DEFAULT_W) -> float:
    a, b, g, d = w
    return (a * norm(float(row["latency_ms_per_token"]), *ranges["latency"])
            + b * (1.0 - norm(float(row["quality_neg_ppl"]), *ranges["quality"]))
            + g * norm(float(row["peak_rss_mb"]), *ranges["memory"])
            + d * norm(float(row["energy_j_per_token"]), *ranges["energy"]))


def evals_per_run(method: str, pop: int, iters: int, budget: int, grid_evals: int) -> int:
    # PSO and DE evaluate the initial population before the first iteration;
    # GWO evaluates inside its iteration loop only (see src/phase2_search/*.py).
    return {"PSO": pop * (iters + 1), "DE": pop * (iters + 1), "GWO": pop * iters,
            "Random": budget, "Grid": grid_evals}[method]


# --------------------------------------------------------------------------- A
def budget_table() -> list[dict]:
    rows = []
    for v in VARIANTS:
        m = load_yaml(ORIG / v / "meta.yaml")
        pop, iters, budget = m["population"], m["iterations"], m["budget"]
        t = m["per_eval_cost_s"]
        n_calib = 8  # original protocol; see article Section 3
        for meth in METHODS:
            runs = 2 if meth in METAHEURISTICS else 1
            per_run = evals_per_run(meth, pop, iters, budget, m["grid_realized_evals"])
            rows.append({"protocol": "original", "variant": v, "method": meth, "runs": runs,
                         "evals_per_run": per_run, "evals_total": per_run * runs,
                         "distinct_configs": "", "threads_kv_cells_visited": "",
                         "per_eval_s": round(t, 1),
                         "device_min_est": round(per_run * runs * t / 60, 1),
                         "calibration_evals": n_calib})
    for v in sorted(os.listdir(HARD)):
        cp = HARD / v / "checkpoints"
        if not cp.is_dir():
            continue
        m = load_yaml(HARD / v / "meta.yaml")
        t = m["per_eval_cost_s"]
        for meth in METHODS:
            files = sorted(glob.glob(str(cp / meth.lower() / "run_*" / "evaluations.json")))
            if not files:
                files = [str(cp / meth.lower() / "evaluations.json")]
            evs = [json.load(open(f)) for f in files]
            confs = set()
            cells = set()
            for ev in evs:
                for e in ev:
                    c = SPACE.decode(np.array(e["x"]))
                    confs.add((c["threads"], c["n_batch"], c["n_ctx"], c["kv_cache_type"]))
                    cells.add((c["threads"], c["kv_cache_type"]))
            total = sum(len(ev) for ev in evs)
            rows.append({"protocol": "hardened", "variant": v, "method": meth, "runs": len(evs),
                         "evals_per_run": len(evs[0]), "evals_total": total,
                         "distinct_configs": len(confs), "threads_kv_cells_visited": len(cells),
                         "per_eval_s": round(t, 1), "device_min_est": round(total * t / 60, 1),
                         "calibration_evals": m["calibration_samples"]})
    write_csv(OUT / "budget_table.csv", rows)
    return rows


# --------------------------------------------------------------------------- B
def hardened_logs(v: str):
    cp = HARD / v / "checkpoints"
    for meth in METHODS:
        files = sorted(glob.glob(str(cp / meth.lower() / "run_*" / "evaluations.json")))
        if not files:
            files = [str(cp / meth.lower() / "evaluations.json")]
        for f in files:
            yield meth, f, json.load(open(f))


def noise_decomposition() -> dict:
    groups_all, rel_noise, gaps = [], [], []
    per_variant = {}
    for v in sorted(os.listdir(HARD)):
        if not (HARD / v / "checkpoints").is_dir():
            continue
        groups, random_costs = [], []
        for meth, f, ev in hardened_logs(v):
            g = collections.defaultdict(list)
            for e in ev:
                c = SPACE.decode(np.array(e["x"]))
                g[(c["threads"], c["n_batch"], c["n_ctx"], c["kv_cache_type"])].append(e["cost"])
                if meth == "Random":
                    random_costs.append(e["cost"])
            groups += [vals for vals in g.values() if len(vals) > 1]
        # Pooled within-configuration SD (one-way ANOVA residual).
        ss = sum(((np.array(x) - np.mean(x)) ** 2).sum() for x in groups)
        dof = sum(len(x) - 1 for x in groups)
        sigma = float(np.sqrt(ss / dof))
        q75, q25 = np.percentile(random_costs, [75, 25])
        iqr = float(q75 - q25)
        rel_noise.append(sigma / iqr)
        groups_all += groups
        rows = read_csv(HARD / v / "best_configs.csv")
        vgaps = {}
        for r in rows:
            gap = float(r["verified_cost"]) - float(r["search_cost"])
            vgaps[r["method"]] = gap
            gaps.append({"variant": v, "method": r["method"], "search": float(r["search_cost"]),
                         "verified": float(r["verified_cost"]), "gap": gap,
                         "status": r["verification_status"]})
        mh = np.mean([vgaps[m] for m in METAHEURISTICS])
        bl = np.mean([vgaps[m] for m in ("Random", "Grid")])
        per_variant[v] = {"sigma": sigma, "dup_groups": len(groups), "random_iqr": iqr,
                          "sigma_over_iqr": sigma / iqr, "mean_gap_mh": float(mh),
                          "mean_gap_baselines": float(bl), "excess_mh": float(mh - bl)}
    ss = sum(((np.array(x) - np.mean(x)) ** 2).sum() for x in groups_all)
    dof = sum(len(x) - 1 for x in groups_all)
    pos = sum(1 for g in gaps if g["gap"] > 0)
    common = float(np.mean([p["mean_gap_baselines"] for p in per_variant.values()]))
    mh_mean = float(np.mean([p["mean_gap_mh"] for p in per_variant.values()]))
    out = {
        "pooled_sigma": float(np.sqrt(ss / dof)),
        "dup_groups": len(groups_all),
        "dup_measurements": int(sum(len(x) for x in groups_all)),
        "sigma_over_random_iqr_mean": float(np.mean(rel_noise)),
        "gaps_positive": pos, "gaps_total": len(gaps),
        "sign_test_p_one_sided": float(stats.binomtest(pos, len(gaps), 0.5, alternative="greater").pvalue),
        "mean_gap_mh": mh_mean, "mean_gap_baselines": common,
        "excess_mh": mh_mean - common,
        "per_variant": per_variant,
    }
    write_csv(OUT / "hardened_gaps.csv", gaps)
    with open(OUT / "noise_decomposition.json", "w") as f:
        json.dump(out, f, indent=2)

    # Original protocol: Table 6 style ratios (reconstructed / recorded).
    ratios = []
    for v in VARIANTS:
        m = load_yaml(ORIG / v / "meta.yaml")
        for r in read_csv(ORIG / v / "best_configs.csv"):
            rec = float(r["cost"])
            recon = score(r, m["normalization_ranges"])
            ratios.append({"variant": v, "method": r["method"], "recorded": rec,
                           "reconstructed": recon,
                           "ratio": (recon / rec) if rec > 0 else float("inf")})
    write_csv(OUT / "original_reconciliation.csv", ratios)
    return out


# --------------------------------------------------------------------------- C
def simplex(step: float = 0.05, floor: float = 0.05):
    n = int(round(1 / step))
    k = int(round(floor / step))
    for a in range(k, n + 1):
        for b in range(k, n + 1 - a):
            for c in range(k, n + 1 - a - b):
                d = n - a - b - c
                if d >= k:
                    yield (a * step, b * step, c * step, d * step)


SCENARIOS = {
    "default (0.35/0.35/0.20/0.10)": DEFAULT_W,
    "equal (0.25 each)": (0.25, 0.25, 0.25, 0.25),
    "latency-led (0.55/0.25/0.10/0.10)": (0.55, 0.25, 0.10, 0.10),
    "quality-led (0.25/0.55/0.10/0.10)": (0.25, 0.55, 0.10, 0.10),
    "memory-led (0.25/0.25/0.40/0.10)": (0.25, 0.25, 0.40, 0.10),
    "energy-led (0.25/0.25/0.10/0.40)": (0.25, 0.25, 0.10, 0.40),
}


def weight_sensitivity() -> dict:
    """Re-score each method's recorded best configuration (raw re-measured
    metrics) under alternative weights, per variant, with that variant's
    own calibration ranges.  This re-ranks the configurations the searches
    found; it does not re-run the searches under the new weights."""
    W = list(simplex())
    win_share = {m: [] for m in METHODS}
    taus, changed = [], 0
    scen_rows = []
    for v in VARIANTS:
        rng = load_yaml(ORIG / v / "meta.yaml")["normalization_ranges"]
        rows = {r["method"]: r for r in read_csv(ORIG / v / "best_configs.csv")}
        base = np.array([score(rows[m], rng) for m in METHODS])
        wins = collections.Counter()
        for w in W:
            s = np.array([score(rows[m], rng, w) for m in METHODS])
            wins[METHODS[int(np.argmin(s))]] += 1
            tau = stats.kendalltau(base, s).statistic
            taus.append(tau if np.isfinite(tau) else 1.0)
            changed += int(np.argmin(s) != np.argmin(base))
        for m in METHODS:
            win_share[m].append(wins[m] / len(W))
        for name, w in SCENARIOS.items():
            s = [score(rows[m], rng, w) for m in METHODS]
            scen_rows.append({"variant": v, "scenario": name, "winner": METHODS[int(np.argmin(s))],
                              **{m: round(x, 4) for m, x in zip(METHODS, s)}})
    write_csv(OUT / "weight_scenarios.csv", scen_rows)
    # Physical spread of each metric inside one variant, relative to its
    # level: shows which normalized terms magnify tiny physical differences.
    spread = []
    for v in VARIANTS:
        rng = load_yaml(ORIG / v / "meta.yaml")["normalization_ranges"]
        spread.append({"variant": v, **{k: round(100 * (hi - lo) / abs(hi), 2)
                                        for k, (lo, hi) in rng.items()}})
    write_csv(OUT / "calibration_relative_spread_pct.csv", spread)
    out = {
        "n_weight_vectors": len(W),
        "winner_changes_share": changed / (len(W) * len(VARIANTS)),
        "median_kendall_tau": float(np.median(taus)),
        "share_of_vectors_won_mean_over_variants": {m: float(np.mean(win_share[m])) for m in METHODS},
        "default_winner_per_scenario_agreement": {
            name: sum(1 for r in scen_rows if r["scenario"] == name
                      and r["winner"] == next(x["winner"] for x in scen_rows
                                              if x["variant"] == r["variant"]
                                              and x["scenario"].startswith("default")))
            for name in SCENARIOS},
    }
    with open(OUT / "weight_sensitivity.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# --------------------------------------------------------------------------- D
def cross_variant() -> dict:
    """Outer step of the decomposed problem: compare all 35 (variant, method)
    configurations on one physical scale (global min-max over the 35 raw
    re-measurements), so that variants can be compared with each other."""
    allrows = []
    for v in VARIANTS:
        for r in read_csv(ORIG / v / "best_configs.csv"):
            allrows.append({"variant": v, **r})
    keys = {"latency": "latency_ms_per_token", "quality": "quality_neg_ppl",
            "memory": "peak_rss_mb", "energy": "energy_j_per_token"}
    g = {k: (min(float(r[c]) for r in allrows), max(float(r[c]) for r in allrows))
         for k, c in keys.items()}
    for r in allrows:
        r["global_cost"] = score(r, g)
    allrows.sort(key=lambda r: r["global_cost"])
    best_per_variant = {}
    for r in allrows:
        best_per_variant.setdefault(r["variant"], r)
    table = [{"variant": v, "method": r["method"], "threads": r["threads"], "n_batch": r["n_batch"],
              "n_p": r["n_ctx"], "kv": r["kv_cache_type"],
              "latency_ms": float(r["latency_ms_per_token"]), "ppl": -float(r["quality_neg_ppl"]),
              "rss_mb": float(r["peak_rss_mb"]), "energy_j": float(r["energy_j_per_token"]),
              "global_cost": round(r["global_cost"], 4)}
             for v, r in sorted(best_per_variant.items(), key=lambda kv: kv[1]["global_cost"])]
    write_csv(OUT / "cross_variant.csv", table)
    W = list(simplex())
    wins = collections.Counter()
    for w in W:
        best = min(allrows, key=lambda r: score(r, g, w))
        wins[best["variant"]] += 1
    scen = {}
    for name, w in SCENARIOS.items():
        best = min(allrows, key=lambda r: score(r, g, w))
        scen[name] = f'{best["variant"]}/{best["method"]}'
    out = {"global_ranges": g, "ranking": table,
           "variant_win_share": {v: wins[v] / len(W) for v in VARIANTS},
           "scenario_winner": scen}
    with open(OUT / "cross_variant.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# --------------------------------------------------------------------------- E
def rank_statistics() -> dict:
    rec = {m: [] for m in METHODS}
    recon = {m: [] for m in METHODS}
    for v in VARIANTS:
        fc = read_csv(ORIG / v / "final_costs.csv")
        rng = load_yaml(ORIG / v / "meta.yaml")["normalization_ranges"]
        bc = {r["method"]: r for r in read_csv(ORIG / v / "best_configs.csv")}
        for m in METHODS:
            if m in METAHEURISTICS:
                rec[m].append(float(np.median([float(r[m]) for r in fc])))
            else:
                rec[m].append(float(bc[m]["cost"]))
            recon[m].append(score(bc[m], rng))
    out = {}
    for label, d in (("recorded", rec), ("reconstructed", recon)):
        f3 = stats.friedmanchisquare(*[d[m] for m in METAHEURISTICS])
        f5 = stats.friedmanchisquare(*[d[m] for m in METHODS])
        ranks = np.array([stats.rankdata([d[m][i] for m in METHODS]) for i in range(len(VARIANTS))])
        n, k = ranks.shape
        w_kendall = 12 * ((ranks.sum(0) - n * (k + 1) / 2) ** 2).sum() / (n ** 2 * (k ** 3 - k))
        best_mh = [min(d[m][i] for m in METAHEURISTICS) for i in range(len(VARIANTS))]
        wil = stats.wilcoxon(best_mh, d["Random"], alternative="less")
        out[label] = {"friedman_3_mh": {"stat": float(f3.statistic), "p": float(f3.pvalue)},
                      "friedman_5": {"stat": float(f5.statistic), "p": float(f5.pvalue)},
                      "kendall_w_5": float(w_kendall),
                      "mean_rank_5": {m: float(r) for m, r in zip(METHODS, ranks.mean(0))},
                      "wilcoxon_bestMH_vs_random_less": {"stat": float(wil.statistic), "p": float(wil.pvalue)}}
    with open(OUT / "rank_statistics.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# --------------------------------------------------------------------------- F
FV_T = np.array([21.0, 555.0, -106.0, 74.0])
FV_LB = np.array([0.0, 200.0, -200.0, -100.0])
FV_UB = np.array([100.0, 700.0, 50.0, 200.0])

SYN_FUNCS = {
    "Sphere-4D": (lambda x: float(np.sum(np.asarray(x) ** 2)), np.full(4, -10.0), np.full(4, 10.0)),
    "FindValues-4D": (lambda x: float(np.sum((np.asarray(x) - FV_T) ** 2)), FV_LB, FV_UB),
    "Rastrigin-4D": (lambda x: float(10 * 4 + np.sum(np.asarray(x) ** 2 - 10 * np.cos(2 * np.pi * np.asarray(x)))),
                     np.full(4, -5.12), np.full(4, 5.12)),
}


def run_algo(name, fn, lb, ub, pop, iters, seed):
    if name == "PSO":
        return pso.optimize(fn, lb, ub, 4, pop_size=pop, max_iter=iters, seed=seed)
    if name == "GWO":
        return gwo.optimize(fn, lb, ub, 4, pop_size=pop, max_iter=iters, seed=seed)
    if name == "DE":
        return de.optimize(fn, lb, ub, 4, pop_size=pop, max_iter=iters, seed=seed)
    if name == "Random":
        return random_search.optimize(fn, lb, ub, 4, budget=pop * iters, seed=seed)
    return grid_search.optimize(fn, lb, ub, 4, budget=pop * iters, seed=seed)


def synthetic_study(rel_noise: float, seeds: int = 100) -> dict:
    rows, summary = [], {}
    for fname, (f, lb, ub) in SYN_FUNCS.items():
        u = np.random.default_rng(12345).uniform(lb, ub, size=(20000, 4))
        vals = np.array([f(x) for x in u])
        iqr = float(np.subtract(*np.percentile(vals, [75, 25])))
        for iters in (5, 8):
            for noise_label, r in (("noise-free", 0.0), ("hardware-calibrated", rel_noise)):
                sigma = r * iqr
                res = {m: {"true": [], "gap": []} for m in METHODS}
                for s in range(seeds):
                    for mi, m in enumerate(METHODS):
                        nrng = np.random.default_rng(10_000 * (s + 1) + mi)
                        obj = (lambda x, f=f, nrng=nrng, sigma=sigma: f(x) + sigma * nrng.standard_normal())
                        out = run_algo(m, obj, lb, ub, 4, iters, s)
                        true = f(out.best_x)
                        res[m]["true"].append(true / iqr)
                        res[m]["gap"].append((true - out.best_cost) / sigma if sigma > 0 else 0.0)
                T = np.array([res[m]["true"] for m in METHODS]).T  # seeds x methods
                fr = stats.friedmanchisquare(*T.T)
                ranks = np.array([stats.rankdata(t) for t in T])
                first = collections.Counter(METHODS[int(np.argmin(t))] for t in T)
                # Reliability of the paper's design: 2 runs per method, pick
                # the method with the lower median; how often does that match
                # the method that is best by the 100-seed median?
                true_best = METHODS[int(np.argmin(np.median(T, axis=0)))]
                prng = np.random.default_rng(7)
                hit = 0
                for _ in range(2000):
                    idx = prng.choice(seeds, size=2, replace=False)
                    hit += METHODS[int(np.argmin(np.median(T[idx], axis=0)))] == true_best
                key = f"{fname}|iters={iters}|{noise_label}"
                fr4 = stats.friedmanchisquare(*[T[:, METHODS.index(m)] for m in ("PSO", "GWO", "DE", "Random")])
                pair = {}
                for m in METAHEURISTICS:
                    p = stats.wilcoxon(T[:, METHODS.index(m)], T[:, METHODS.index("Random")]).pvalue
                    pair[m] = float(p)
                # Holm correction over the three metaheuristic-vs-random tests.
                order = sorted(pair, key=pair.get)
                holm, running = {}, 0.0
                for i, m in enumerate(order):
                    running = max(running, min(1.0, (len(order) - i) * pair[m]))
                    holm[m] = running
                summary[key] = {
                    "friedman_p": float(fr.pvalue),
                    "friedman_p_without_grid": float(fr4.pvalue),
                    "wilcoxon_vs_random_holm_p": holm,
                    "mean_rank": {m: float(x) for m, x in zip(METHODS, ranks.mean(0))},
                    "share_first": {m: first[m] / seeds for m in METHODS},
                    "true_best_by_median": true_best,
                    "two_run_design_hit_rate": hit / 2000,
                }
                for m in METHODS:
                    rows.append({"function": fname, "iters": iters, "noise": noise_label,
                                 "method": m,
                                 "median_true_over_iqr": float(np.median(res[m]["true"])),
                                 "iqr_true_over_iqr": float(np.subtract(*np.percentile(res[m]["true"], [75, 25]))),
                                 "mean_gap_over_sigma": float(np.mean(res[m]["gap"])),
                                 "share_gap_positive": float(np.mean(np.array(res[m]["gap"]) > 0)) if sigma > 0 else "",
                                 "share_first": first[m] / seeds})
    write_csv(OUT / "synthetic_study.csv", rows)
    with open(OUT / "synthetic_study.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------- G
def energy_sampling() -> list[dict]:
    rows = []
    for v in VARIANTS:
        t = load_yaml(ORIG / v / "meta.yaml")["per_eval_cost_s"]
        rows.append({"protocol": "original", "variant": v, "per_eval_s": round(t, 1),
                     "power_samples_per_eval": int(t // 1.0) + 1,
                     "max_tail_truncation_pct": round(100 * 1.0 / t, 1)})
    for v in sorted(os.listdir(HARD)):
        if (HARD / v / "meta.yaml").exists():
            t = load_yaml(HARD / v / "meta.yaml")["per_eval_cost_s"]
            rows.append({"protocol": "hardened", "variant": v, "per_eval_s": round(t, 1),
                         "power_samples_per_eval": int(t // 1.0) + 1,
                         "max_tail_truncation_pct": round(100 * 1.0 / t, 1)})
    write_csv(OUT / "energy_sampling.csv", rows)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("A. budget");             b = budget_table()
    for r in b:
        print("  ", r)
    print("B. noise decomposition"); n = noise_decomposition()
    print(json.dumps(n, indent=1))
    print("C. weight sensitivity");  print(json.dumps(weight_sensitivity(), indent=1))
    print("D. cross-variant");       c = cross_variant()
    print(json.dumps({k: c[k] for k in ("variant_win_share", "scenario_winner")}, indent=1))
    for r in c["ranking"]:
        print("  ", r)
    print("E. rank statistics");     print(json.dumps(rank_statistics(), indent=1))
    print("F. synthetic study");     s = synthetic_study(n["sigma_over_random_iqr_mean"])
    print(json.dumps(s, indent=1))
    print("G. energy sampling")
    for r in energy_sampling():
        print("  ", r)


if __name__ == "__main__":
    main()
