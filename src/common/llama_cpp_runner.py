"""Thin wrappers around llama.cpp CLI binaries (llama-quantize, llama-bench,
llama-perplexity) plus cross-platform peak-RSS measurement.

These wrap subprocess calls rather than a Python binding (llama-cpp-python)
by design: the goal is to measure exactly what will run on the Raspberry Pi
target, and CLI binaries avoid a compiled Python extension that would need
to build cleanly on ARM64 (see article_draft.md, Section 3 discussion).

NOTE on CLI flags: llama.cpp's flag names have changed across releases
(e.g. the perplexity/bench tool names and some flags). The flags used below
match a recent llama.cpp release as of this writing. Before trusting any
numbers produced by this module, run each binary with --help on your actual
build and confirm the flags still match — this is flagged explicitly rather
than assumed, per the project's placeholder/verify convention.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil


class LlamaCppError(RuntimeError):
    """Raised when a llama.cpp CLI invocation fails or its output can't be parsed."""


@dataclass
class BenchResult:
    prompt_tokens_per_sec: Optional[float]
    gen_tokens_per_sec: Optional[float]
    raw_json: list


@dataclass
class PerplexityResult:
    ppl: float
    ppl_stderr: Optional[float]


@dataclass
class RunWithRssResult:
    returncode: int
    stdout: str
    stderr: str
    peak_rss_bytes: int
    wall_time_s: float


def run_with_peak_rss(cmd: list[str], sample_interval_s: float = 0.05, timeout_s: Optional[float] = None) -> RunWithRssResult:
    """Run a subprocess while polling its (and its children's) RSS, cross-platform.

    Used instead of `/usr/bin/time -v` so the same code path works during
    local development on Windows and on the Raspberry Pi target.
    """
    start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_rss = 0
    stop_flag = threading.Event()

    def _poll():
        nonlocal peak_rss
        try:
            ps_proc = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            return
        while not stop_flag.is_set():
            try:
                procs = [ps_proc] + ps_proc.children(recursive=True)
                total = sum(p.memory_info().rss for p in procs if p.is_running())
                peak_rss = max(peak_rss, total)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if stop_flag.wait(sample_interval_s):
                break

    sampler = threading.Thread(target=_poll, daemon=True)
    sampler.start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        stop_flag.set()
        sampler.join()
        raise LlamaCppError(f"Command timed out after {timeout_s}s: {' '.join(cmd)}")
    stop_flag.set()
    sampler.join()

    return RunWithRssResult(
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        peak_rss_bytes=peak_rss,
        wall_time_s=time.monotonic() - start,
    )


def quantize(quantize_bin: str, input_gguf: Path, output_gguf: Path, quant_type: str) -> None:
    """Run llama-quantize to produce a quantized GGUF from an F16 source."""
    if output_gguf.exists():
        return
    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [quantize_bin, str(input_gguf), str(output_gguf), quant_type]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise LlamaCppError(
            f"llama-quantize failed for {quant_type}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


_PP_TEST_RE = re.compile(r"^pp\d+$")
_TG_TEST_RE = re.compile(r"^tg\d+$")


def run_bench(bench_bin: str, model_gguf: Path, threads: int, n_prompt: int, n_gen: int, repetitions: int = 3) -> BenchResult:
    """Run llama-bench with JSON output and extract prompt/gen tokens-per-second.

    llama-bench runs a "pp<N>" (prompt processing) test and a "tg<N>" (text
    generation) test by default and emits one JSON row per test per repetition
    when -o json is used. We average avg_ts across repetitions per test type.
    """
    cmd = [
        bench_bin,
        "-m", str(model_gguf),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-t", str(threads),
        "-r", str(repetitions),
        "-o", "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise LlamaCppError(f"llama-bench failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LlamaCppError(
            f"Could not parse llama-bench JSON output (flag names may have changed "
            f"in your llama.cpp version — check `llama-bench --help`):\n{result.stdout}"
        ) from exc

    pp_values = [row["avg_ts"] for row in rows if _PP_TEST_RE.match(row.get("test", ""))]
    tg_values = [row["avg_ts"] for row in rows if _TG_TEST_RE.match(row.get("test", ""))]

    return BenchResult(
        prompt_tokens_per_sec=sum(pp_values) / len(pp_values) if pp_values else None,
        gen_tokens_per_sec=sum(tg_values) / len(tg_values) if tg_values else None,
        raw_json=rows,
    )


_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")


def run_perplexity(perplexity_bin: str, model_gguf: Path, corpus_path: Path, ctx_size: int) -> PerplexityResult:
    """Run llama-perplexity and extract the final PPL estimate."""
    cmd = [
        perplexity_bin,
        "-m", str(model_gguf),
        "-f", str(corpus_path),
        "-c", str(ctx_size),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise LlamaCppError(f"llama-perplexity failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    match = _PPL_RE.search(result.stdout) or _PPL_RE.search(result.stderr)
    if not match:
        raise LlamaCppError(
            f"Could not find 'Final estimate: PPL = ...' in llama-perplexity output "
            f"(flag/output format may have changed in your llama.cpp version):\n{result.stdout}"
        )
    return PerplexityResult(ppl=float(match.group(1)), ppl_stderr=float(match.group(2)))


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)
