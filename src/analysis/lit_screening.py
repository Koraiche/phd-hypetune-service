"""Screening of the OpenAlex records retrieved by lit_search.py (ISI revision,
Reviewer 2, pt 2).

Stage 1 excludes non-peer-reviewed record types.  Stage 2 flags the four gap
criteria from title and abstract with keyword rules (a: population-based
metaheuristic; b: search space containing both compression and runtime
parameters; c: each candidate measured on a physical low-power device;
d: LLM as deployed model).  Every record flagged on at least two criteria was
then assessed manually from its abstract where needed; the manual decisions are listed in
MANUAL below and override the keyword flags.  Records flagged on three or
more criteria cannot pass without a manual entry; records flagged on fewer
cannot meet three criteria and keep their keyword flags.  Output:
results/revision/litsearch/screening.csv

Usage:  python src/analysis/lit_screening.py
"""
import collections
import csv
import json
import re
from pathlib import Path

LS = Path(__file__).resolve().parents[2] / "results" / "revision" / "litsearch"
PEER_REVIEWED = {"article", "conference-paper", "book-chapter", "software-paper"}

KW = {
    "a": r"particle swarm|grey wolf|gray wolf|differential evolution|genetic algorithm|evolutionary|metaheuristic|swarm",
    "b_comp": r"quantiz|compression|mixed[- ]precision|pruning",
    "b_run": r"batch size|thread|context length|kv[- ]cache|runtime config|inference config|scheduling|tensor parallel|deployment config",
    "c": r"raspberry pi|jetson|smartphone|android|galaxy|arm cpu|arm devices|microcontroller|physical testbed|on real devices|on-device measurement|edge board",
    "d": r"large language model|\bllms?\b|small language model|language model",
}

# Manual assessment (from abstracts) of every record flagged on >= 2 criteria.
# key: DOI suffix -> (a, b, c, d, note)
MANUAL = {
    "10.1109/iscas66217.2026.11562390": (1, 0, 1, 1, "GA mixed-precision search for LLMs on ARM CPUs; operators hand-optimized, runtime parameters not searched"),
    "10.1109/tpds.2026.3703757": (1, 1, 0, 1, "FlexiTensor: hybrid evolutionary search of placement, tensor splitting, threads, quantization; candidates evaluated with pre-profiled device data"),
    "10.3390/electronics15061312": (1, 0, 1, 1, "offline PSO design of a softmax approximation for TinyLlama, measured on an NPU; no compression/runtime search"),
    "10.1109/gaiss66401.2026.11779656": (0, 0, 1, 1, "rule-based runtime switching of precision and GC-aware scheduling on Android; no search"),
    "10.1109/icip51287.2024.10647455": (1, 0, 1, 0, "evolutionary PTQ of a VQA transformer deployed on a phone; not an LLM, no runtime search"),
    "10.1109/icassp48485.2024.10446281": (1, 0, 1, 0, "evolutionary PTQ calibration for CNNs deployed on a phone"),
    "10.1109/ccnc65079.2026.11366617": (1, 1, 0, 0, "GA for joint mixed precision and model splitting of CNNs in multi-hop IoT; simulated network"),
    "10.4018/ijsir.388948": (1, 0, 0, 0, "PSO mixed-precision quantization of CNNs via supernet evaluation"),
    "10.1016/j.array.2026.101023": (1, 0, 0, 0, "GA mixed-precision floating-point quantization of a CNN accelerator"),
    "10.1109/jiot.2026.3675866": (0, 0, 0, 1, "survey"),
    "10.1109/ccwc67433.2026.11393726": (0, 0, 0, 1, "survey of on-device SLM acceleration; no search, no own measurements"),
    "10.1145/3845562.3845593": (0, 0, 0, 1, "review of on-device PTQ for LLMs; no search"),
    "10.1109/tmc.2026.3697502": (0, 0, 1, 1, "layer-skipping serving system on devices; no metaheuristic or configuration search"),
    "10.1016/j.softx.2026.102764": (0, 0, 1, 1, "offline translation app with a 4-bit LLM on a smartphone; no search"),
    "10.1109/mipro70003.2026.11591742": (0, 0, 1, 1, "prefill/decode characterization on Raspberry Pi 5 and Jetson; no search"),
    "10.1109/les.2026.3727282": (0, 0, 1, 1, "runtime memory repartitioning in llama.cpp on Jetson; no search"),
    "10.1109/ijcnn69854.2026.11729483": (0, 0, 1, 1, "scene-language pipeline with a 4B LLM on edge hardware; no search"),
}


def flags(text):
    t = (text or "").lower()
    a = bool(re.search(KW["a"], t))
    b = bool(re.search(KW["b_comp"], t)) and bool(re.search(KW["b_run"], t))
    c = bool(re.search(KW["c"], t))
    d = bool(re.search(KW["d"], t))
    return [int(a), int(b), int(c), int(d)]


def main():
    recs = json.load(open(LS / "all_unique.json", encoding="utf-8"))
    rows, counts = [], collections.Counter()
    for r in recs:
        doi = (r.get("doi") or "").lower().replace("https://doi.org/", "")
        base = {"openalex_id": r["id"], "doi": doi, "year": r["year"], "type": r["type"],
                "venue": r.get("venue") or "", "title": r["title"], "queries": ",".join(r["queries"])}
        if r["type"] not in PEER_REVIEWED:
            rows.append({**base, "stage": "excluded: record type", "a": "", "b": "", "c": "", "d": "",
                         "criteria_met": "", "assessment": "keyword", "note": ""})
            counts["excluded_type"] += 1
            continue
        counts["screened"] += 1
        f = flags(f"{r['title']} {r.get('abstract', '')}")
        assessment, note = "keyword", ""
        if doi in MANUAL:
            *f, note = MANUAL[doi]
            assessment = "manual"
        elif sum(f) >= 3:
            raise SystemExit(f"record {doi} is flagged on >= 3 criteria and needs a manual assessment")
        n = sum(f)
        counts[f"met_{n}"] += 1
        rows.append({**base, "stage": "screened", "a": f[0], "b": f[1], "c": f[2], "d": f[3],
                     "criteria_met": n, "assessment": assessment, "note": note})
    with open(LS / "screening.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(dict(counts))
    for r in rows:
        if r["stage"] == "screened" and (r["criteria_met"] or 0) >= 2:
            print(r["criteria_met"], r["assessment"], r["doi"], "|", r["title"][:90])


if __name__ == "__main__":
    main()
