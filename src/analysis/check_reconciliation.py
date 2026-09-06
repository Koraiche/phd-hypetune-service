"""Cross-checks each Phase 2 method's recorded search-time cost against a
recomputation of that same method's scalarized cost from its own reported
raw metrics (best_configs.csv).

Written after noticing GWO's near-zero recorded cost on Q8_0 (Section 5.4)
did not reconcile with its own raw latency/quality/memory/energy values.
Running this check across every completed variant showed the discrepancy
is general -- consistent with the optimizer's curse, not an isolated
anomaly -- because run_real_phase2.py's best_configs.csv raw columns come
from a fresh evaluator.evaluate(config) call after the search completes,
not the exact measurement that produced the recorded search-time cost.
See article_draft.md Section 5.5 for the full account and results table.

Usage: python src/analysis/check_reconciliation.py
"""
import csv
import glob
import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(REPO_ROOT, "results", "phase2_runs")
ALPHA, BETA, GAMMA, DELTA = 0.35, 0.35, 0.2, 0.1  # ObjectiveWeights defaults


def clip01(x):
    return max(0.0, min(1.0, x))


def main():
    for variant_dir in sorted(glob.glob(os.path.join(BASE, "*"))):
        variant = os.path.basename(variant_dir)
        meta_path = os.path.join(variant_dir, "meta.yaml")
        best_path = os.path.join(variant_dir, "best_configs.csv")
        if not (os.path.exists(meta_path) and os.path.exists(best_path)):
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        r = meta["normalization_ranges"]
        lat_lo, lat_hi = r["latency"]
        q_lo, q_hi = r["quality"]
        mem_lo, mem_hi = r["memory"]
        en_lo, en_hi = r["energy"]
        print(f"\n=== {variant} ===")
        with open(best_path) as f:
            for row in csv.DictReader(f):
                lat = float(row["latency_ms_per_token"])
                q = float(row["quality_neg_ppl"])
                mem = float(row["peak_rss_mb"])
                en = float(row["energy_j_per_token"])
                # New runs preserve both values explicitly; historical CSVs
                # only have ``cost``, which was the search-time minimum.
                recorded = float(row.get("search_cost") or row["cost"])
                lat_n = clip01((lat - lat_lo) / (lat_hi - lat_lo))
                q_n = clip01((q - q_lo) / (q_hi - q_lo))
                mem_n = clip01((mem - mem_lo) / (mem_hi - mem_lo))
                en_n = clip01((en - en_lo) / (en_hi - en_lo))
                recon = ALPHA * lat_n + BETA * (1 - q_n) + GAMMA * mem_n + DELTA * en_n
                ratio = recon / recorded if recorded > 1e-9 else float("inf")
                verified = row.get("verified_cost")
                verified_text = f"  verified={float(verified):.4f}" if verified else ""
                print(
                    f"  {row['method']:8s} recorded={recorded:.4f}  "
                    f"reconstructed={recon:.4f}{verified_text}  ratio={ratio:6.2f}x"
                )


if __name__ == "__main__":
    main()
