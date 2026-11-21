"""Documented literature search for the ISI 47011 revision (Reviewer 2, pts 1-2).

Runs fixed boolean title/abstract queries against the OpenAlex API, saves
every returned record (id, DOI, title, abstract, year, type, venue) to
results/revision/litsearch/<query>.json, and prints the hit counts.  The
query strings, filters and run date are recorded in queries.json so the
search can be repeated exactly.

Usage:  python src/analysis/lit_search.py [count|fetch]
"""
import datetime, json, sys, time, urllib.parse, urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results" / "revision" / "litsearch"
A = '("particle swarm" OR "grey wolf" OR "gray wolf" OR "differential evolution" OR "genetic algorithm" OR "evolutionary algorithm" OR metaheuristic OR "swarm intelligence")'
B = '(quantization OR quantized OR "model compression" OR "mixed precision" OR "mixed-precision")'
C = '("runtime configuration" OR "inference configuration" OR "batch size" OR "thread count" OR "number of threads" OR "context length" OR "KV cache" OR "configuration tuning" OR autotuning OR "auto-tuning" OR "inference engine" OR "deployment configuration")'
D = '("Raspberry Pi" OR "edge device" OR "edge devices" OR "embedded device" OR "embedded system" OR "single-board" OR microcontroller OR "on-device" OR "edge computing")'
E = '("large language model" OR "language model" OR LLM OR LLMs OR "small language model")'
QUERIES = {
    "Q1_opt_compression_edge_llm": f"{A} AND {B} AND {D} AND {E}",
    "Q2_opt_runtime_edge_llm": f"{A} AND {C} AND {D} AND {E}",
    "Q3_compression_runtime_edge_llm": f"{B} AND {C} AND {D} AND {E}",
    "Q4_opt_compression_runtime_edge": f"{A} AND {B} AND {C} AND {D}",
    "Q5_opt_edge_llm": f"{A} AND {D} AND {E}",
    "Q6_opt_compression_edge": f"{A} AND {B} AND {D}",
}
FILTER = "publication_year:2019-2026,language:en"
FIELDS = "id,doi,title,publication_year,type,primary_location,abstract_inverted_index"


def url(q, cursor="*", per_page=200):
    f = f"title_and_abstract.search:{q},{FILTER}"
    return ("https://api.openalex.org/works?" +
            urllib.parse.urlencode({"filter": f, "per-page": per_page, "cursor": cursor, "select": FIELDS}))


def get(u):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(u, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # rate limits / transient errors
            time.sleep(2 + 3 * attempt)
            err = e
    raise err


def abstract(inv):
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def main(mode):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"database": "OpenAlex (api.openalex.org)", "field": "title_and_abstract.search",
            "filters": FILTER, "run_date": datetime.date.today().isoformat(), "queries": QUERIES, "counts": {}}
    for name, q in QUERIES.items():
        first = get(url(q, per_page=1 if mode == "count" else 200))
        n = first["meta"]["count"]
        meta["counts"][name] = n
        print(f"{name}: {n}")
        if mode != "fetch":
            continue
        recs, page = [], first
        while True:
            for w in page["results"]:
                loc = (w.get("primary_location") or {}).get("source") or {}
                recs.append({"id": w["id"], "doi": w.get("doi"), "title": w.get("title"),
                             "year": w.get("publication_year"), "type": w.get("type"),
                             "venue": loc.get("display_name"), "venue_type": loc.get("type"),
                             "abstract": abstract(w.get("abstract_inverted_index"))})
            cur = page["meta"].get("next_cursor")
            if not cur or not page["results"]:
                break
            page = get(url(q, cursor=cur))
        with open(OUT / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=1, ensure_ascii=False)
    with open(OUT / "queries.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "count")
