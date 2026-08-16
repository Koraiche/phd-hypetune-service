"""Real Evaluator for Phase 2, backed by actual llama.cpp measurements on
the Raspberry Pi 5 target. Implements the same Evaluator protocol as
mock_evaluator.MockLlamaCppEvaluator (see objective.py) — this is the
production swap-in article_draft.md Section 5.2.2 said would happen "once
Phase 1 + Pi data exist."

Key design decisions, each a deliberate adaptation to what real measurement
on this hardware actually costs and actually exposes, not silent guesses:

- llama-bench has NO -c/--ctx-size flag (confirmed against a real build,
  2026-08-16). The Phase 2 `n_ctx` search parameter is realized as -p
  (n_prompt) instead — the amount of context llama-bench actually
  processes/allocates for. See run_bench_instrumented's docstring.

- Quality (perplexity) is the one measurement that's genuinely expensive
  (Phase 1's full-wikitext-2-corpus perplexity runs took 1-3+ hours PER
  variant) and the one dimension of the Phase 2 search space that does NOT
  affect it is threads/n_batch/n_ctx — only kv_cache_type does (it changes
  the numerical precision of attention). So quality is measured ONCE per
  distinct kv_cache_type value seen (at most 3: f16/q8_0/q4_0), cached, and
  reused for every candidate sharing that value — turning an O(evaluations)
  cost into an O(3) cost. This is what makes a real metaheuristic search
  budget (dozens to hundreds of evaluations) tractable at all; without it,
  Phase 2 would inherit Phase 1's hours-per-measurement cost and become
  infeasible on an overnight budget, exactly the cost-asymmetry problem
  article_draft.md Section 3.7 already argues motivates the two-phase design
  in the first place — this is that argument applied one level deeper.

- The quality corpus is a fixed slice of the real wikitext-2-raw-v1 test set
  (not the full ~1.28MB corpus Phase 1 used) — real held-out text, just less
  of it, chosen to be fast enough for repeated per-kv-type measurement while
  still comfortably exceeding llama-perplexity's minimum token requirement.

- The RAM ceiling is a real number for this device: 8GB total, 6144MB
  (6GB) ceiling leaves ~2GB headroom for the OS and other processes, not an
  arbitrary round number.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llama_cpp_runner import LlamaCppError, run_bench_instrumented, run_perplexity  # noqa: E402
from phase2_search.objective import EvaluationResult  # noqa: E402

RAM_CEILING_MB = 6144.0


@dataclass
class LlamaCppEvaluator:
    bench_bin: str
    perplexity_bin: str
    model_gguf: Path
    quality_corpus: Path
    quality_ctx_size: int = 512
    bench_repetitions: int = 1
    n_gen: int = 64
    ram_ceiling_mb: float = RAM_CEILING_MB
    _quality_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def _quality_for(self, kv_cache_type: str) -> float:
        if kv_cache_type not in self._quality_cache:
            result = run_perplexity(
                self.perplexity_bin, self.model_gguf, self.quality_corpus,
                ctx_size=self.quality_ctx_size,
                cache_type_k=kv_cache_type, cache_type_v=kv_cache_type,
            )
            self._quality_cache[kv_cache_type] = -result.ppl
        return self._quality_cache[kv_cache_type]

    def evaluate(self, config: dict) -> EvaluationResult:
        threads = config["threads"]
        n_batch = config["n_batch"]
        n_ctx = config["n_ctx"]
        kv = config["kv_cache_type"]

        try:
            result = run_bench_instrumented(
                self.bench_bin, self.model_gguf, threads=threads,
                n_prompt=n_ctx, n_gen=self.n_gen, repetitions=self.bench_repetitions,
                batch_size=n_batch, cache_type_k=kv, cache_type_v=kv,
            )
        except LlamaCppError as exc:
            return EvaluationResult(
                latency_ms_per_token=0.0, quality=0.0, peak_rss_mb=0.0, energy_j_per_token=0.0,
                feasible=False, reason=f"llama-bench failed: {exc}",
            )

        peak_rss_mb = result.peak_rss_bytes / (1024 * 1024)
        feasible = peak_rss_mb <= self.ram_ceiling_mb

        gen_tps = result.bench.gen_tokens_per_sec or 1e-6
        latency_ms_per_token = 1000.0 / gen_tps
        energy_j_per_token = result.energy_joules / self.n_gen if self.n_gen else 0.0

        return EvaluationResult(
            latency_ms_per_token=latency_ms_per_token,
            quality=self._quality_for(kv),
            peak_rss_mb=peak_rss_mb,
            energy_j_per_token=energy_j_per_token,
            feasible=feasible,
            reason="" if feasible else f"peak_rss_mb={peak_rss_mb:.1f} > ceiling={self.ram_ceiling_mb}",
        )
