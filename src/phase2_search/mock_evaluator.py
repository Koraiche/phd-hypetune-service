"""SYNTHETIC evaluator standing in for real llama.cpp-on-Pi measurement.

Every number in this module is a made-up, physically-motivated toy cost
surface (diminishing-returns thread speedup, KV-cache memory roughly linear
in context length, quantized-cache quality penalty, power roughly linear in
active threads) — NOT derived from any measurement, real TinyLlama run, or
Raspberry Pi. Its only purpose is to exercise the Phase 2 search loop
end to end (parameter encode/decode, feasibility rejection, PSO/GWO/DE
wiring, calibration, scalarization) before Raspberry Pi hardware is
available.

No result produced using this evaluator may be reported as a real finding
in article_draft.md Section 5 — see the explicit disclaimer in the section
that reports mock-rehearsal results.
"""
from __future__ import annotations

from dataclasses import dataclass

from .objective import EvaluationResult

# Toy RAM ceiling for this synthetic rehearsal only — NOT the real Pi 5 limit.
MOCK_RAM_CEILING_MB = 900.0

_KV_CACHE_QUALITY_PENALTY = {"f16": 0.0, "q8_0": 0.03, "q4_0": 0.12}
_KV_CACHE_MEMORY_FACTOR = {"f16": 1.0, "q8_0": 0.55, "q4_0": 0.30}
_KV_CACHE_LATENCY_FACTOR = {"f16": 1.0, "q8_0": 0.97, "q4_0": 0.93}


@dataclass
class MockLlamaCppEvaluator:
    base_perplexity: float = 6.0  # a plausible-looking TinyLlama-scale PPL
    base_latency_ms: float = 220.0  # ms/token at threads=1, small ctx/batch
    base_model_mb: float = 700.0  # roughly a Q4-ish TinyLlama-1.1B GGUF, resident

    def evaluate(self, config: dict) -> EvaluationResult:
        threads = config["threads"]
        n_batch = config["n_batch"]
        n_ctx = config["n_ctx"]
        kv = config["kv_cache_type"]

        thread_speedup = threads**0.7  # diminishing returns per extra core
        ctx_penalty = 1.0 + (n_ctx / 4096.0) * 0.4
        batch_penalty = 1.0 + (n_batch / 512.0) * 0.1
        latency = (
            self.base_latency_ms
            / thread_speedup
            * ctx_penalty
            * batch_penalty
            * _KV_CACHE_LATENCY_FACTOR[kv]
        )

        kv_cache_mb = n_ctx * 0.08 * _KV_CACHE_MEMORY_FACTOR[kv]
        batch_mb = n_batch * 0.05
        peak_rss_mb = self.base_model_mb + kv_cache_mb + batch_mb

        # Quality is independent of threads/batch, as in reality — only the
        # lossy quantized KV cache degrades it.
        quality = -(self.base_perplexity * (1.0 + _KV_CACHE_QUALITY_PENALTY[kv]))

        power_w = 3.0 + 1.2 * threads  # more active cores draw more power
        energy_j_per_token = power_w * (latency / 1000.0)

        feasible = peak_rss_mb <= MOCK_RAM_CEILING_MB
        reason = "" if feasible else f"peak_rss_mb={peak_rss_mb:.1f} > ceiling={MOCK_RAM_CEILING_MB}"

        return EvaluationResult(
            latency_ms_per_token=latency,
            quality=quality,
            peak_rss_mb=peak_rss_mb,
            energy_j_per_token=energy_j_per_token,
            feasible=feasible,
            reason=reason,
        )
