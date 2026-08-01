"""Phase 1: exhaustive quantization-variant sweep (article_draft.md Section 3.4).

For each candidate GGUF quantization of TinyLlama, measures perplexity, peak
RSS, load-adjacent latency proxy (tokens/s), and file size — once per
variant, no search involved. Produces results/phase1_table.csv (Section 5.1)
and results/phase1_retained.yaml, the Pareto-nondominated variant set handed
to Phase 2's metaheuristic runtime search.

Usage:
    python src/phase1_quant_sweep.py --config config/phase1.yaml
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.llama_cpp_runner import (  # noqa: E402
    LlamaCppError,
    file_size_mb,
    quantize,
    run_bench,
    run_perplexity,
    run_with_peak_rss,
)
from common.pareto import pareto_front  # noqa: E402


def ensure_base_model(cfg: dict, repo_root: Path) -> Path:
    local_path = repo_root / cfg["base_model"]["local_path"]
    if local_path.exists():
        return local_path

    hf_filename = cfg["base_model"].get("hf_filename")
    if not hf_filename:
        raise FileNotFoundError(
            f"Base F16 GGUF not found at {local_path} and no hf_filename configured "
            f"to download one. Either place a converted GGUF there yourself (see "
            f"README.md 'Getting a base F16 GGUF'), or set base_model.hf_filename "
            f"in config/phase1.yaml to a real file in base_model.hf_repo."
        )

    from huggingface_hub import hf_hub_download  # imported lazily: optional dependency path
    import shutil

    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=cfg["base_model"]["hf_repo"], filename=hf_filename)
    shutil.copyfile(downloaded, local_path)
    return local_path


def run_phase1(config_path: Path) -> None:
    repo_root = config_path.resolve().parent.parent
    cfg = yaml.safe_load(config_path.read_text())

    bin_dir = Path(cfg["llama_cpp"]["bin_dir"])
    quantize_bin = str(bin_dir / cfg["llama_cpp"]["quantize_bin"])
    bench_bin = str(bin_dir / cfg["llama_cpp"]["bench_bin"])
    perplexity_bin = str(bin_dir / cfg["llama_cpp"]["perplexity_bin"])

    models_dir = repo_root / cfg["output"]["models_dir"]
    results_dir = repo_root / cfg["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = repo_root / cfg["perplexity"]["corpus_path"]
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Perplexity corpus not found at {corpus_path}. A smoke-test sample ships "
            f"in data/prompts/ — for real Section 5.1 numbers, point "
            f"perplexity.corpus_path at a held-out corpus (e.g. wikitext-2-raw)."
        )

    base_model_path = ensure_base_model(cfg, repo_root)

    rows = []
    for quant_type in cfg["quant_variants"]:
        print(f"[phase1] === {quant_type} ===", flush=True)
        quant_path = models_dir / f"tinyllama-1.1b-{quant_type}.gguf"

        try:
            quantize(quantize_bin, base_model_path, quant_path, quant_type)
        except LlamaCppError as exc:
            print(f"[phase1] SKIP {quant_type}: quantization failed: {exc}", file=sys.stderr)
            continue

        try:
            bench = run_bench(
                bench_bin,
                quant_path,
                threads=cfg["bench"]["threads"],
                n_prompt=cfg["bench"]["n_prompt"],
                n_gen=cfg["bench"]["n_gen"],
                repetitions=cfg["bench"]["repetitions"],
            )
        except LlamaCppError as exc:
            print(f"[phase1] SKIP {quant_type}: bench failed: {exc}", file=sys.stderr)
            continue

        try:
            ppl = run_perplexity(
                perplexity_bin, quant_path, corpus_path, ctx_size=cfg["perplexity"]["ctx_size"]
            )
        except LlamaCppError as exc:
            print(f"[phase1] SKIP {quant_type}: perplexity failed: {exc}", file=sys.stderr)
            continue

        # Reuse a bench invocation to sample peak RSS under load, rather than an
        # idle-load measurement — this is what actually determines OOM risk on
        # the Pi (Section 3.6's hard RAM-ceiling constraint).
        rss_probe = run_with_peak_rss(
            [
                bench_bin, "-m", str(quant_path),
                "-p", str(cfg["bench"]["n_prompt"]), "-n", str(cfg["bench"]["n_gen"]),
                "-t", str(cfg["bench"]["threads"]), "-r", "1",
            ]
        )

        rows.append({
            "name": quant_type,
            "quality": -ppl.ppl,  # higher = better, for Pareto axis consistency
            "perplexity": ppl.ppl,
            "perplexity_stderr": ppl.ppl_stderr,
            "tokens_per_sec": bench.gen_tokens_per_sec,
            "prompt_tokens_per_sec": bench.prompt_tokens_per_sec,
            "peak_rss_mb": rss_probe.peak_rss_bytes / (1024 * 1024),
            "file_size_mb": file_size_mb(quant_path),
        })
        print(f"[phase1] {quant_type}: PPL={ppl.ppl:.4f} tok/s={bench.gen_tokens_per_sec} "
              f"peak_rss={rows[-1]['peak_rss_mb']:.1f}MB size={rows[-1]['file_size_mb']:.1f}MB")

    if not rows:
        raise RuntimeError("No quant variant completed successfully — nothing to report or retain.")

    table_csv = repo_root / cfg["output"]["table_csv"]
    with open(table_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[phase1] wrote {table_csv}")

    axes = cfg["pareto_axes"]
    retained = pareto_front(rows, axes=axes, id_key="name")
    retained_yaml = repo_root / cfg["output"]["retained_yaml"]
    retained_yaml.write_text(yaml.safe_dump({"retained_variants": [r["name"] for r in retained]}))
    print(f"[phase1] retained for Phase 2: {[r['name'] for r in retained]}")
    print(f"[phase1] wrote {retained_yaml}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/phase1.yaml"))
    args = parser.parse_args()
    run_phase1(args.config)


if __name__ == "__main__":
    main()
