from pathlib import Path

from common.llama_cpp_runner import BenchResult, InstrumentedBenchResult
from phase2_search.llama_cpp_evaluator import LlamaCppEvaluator


def test_energy_is_normalized_by_generated_tokens_across_repetitions(monkeypatch):
    measured = InstrumentedBenchResult(
        bench=BenchResult(prompt_tokens_per_sec=100.0, gen_tokens_per_sec=10.0, raw_json=[]),
        peak_rss_bytes=100 * 1024 * 1024,
        energy_joules=96.0,
        avg_power_watts=6.0,
        wall_time_s=16.0,
    )
    monkeypatch.setattr(
        "phase2_search.llama_cpp_evaluator.run_bench_instrumented",
        lambda *args, **kwargs: measured,
    )
    evaluator = LlamaCppEvaluator(
        bench_bin="llama-bench",
        perplexity_bin="llama-perplexity",
        model_gguf=Path("model.gguf"),
        quality_corpus=Path("corpus.txt"),
        bench_repetitions=3,
        n_gen=64,
        _quality_cache={"f16": -10.0},
    )

    result = evaluator.evaluate({"threads": 4, "n_batch": 32, "n_ctx": 128, "kv_cache_type": "f16"})

    assert result.energy_j_per_token == 0.5
    assert result.latency_ms_per_token == 100.0
