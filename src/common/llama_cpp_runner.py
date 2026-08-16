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


def quantize(quantize_bin: str, input_gguf: Path, output_gguf: Path, quant_type: str, allow_requantize: bool = True) -> None:
    """Run llama-quantize to produce a quantized GGUF from a source GGUF.

    allow_requantize defaults to True because this project's base_model is
    Q8_0 (see config/phase1.yaml's documented deviation from true F16), not
    F32/F16 — llama-quantize refuses to requantize an already-quantized
    source unless --allow-requantize is passed explicitly. Confirmed against
    a real build's --help output 2026-08-15, not assumed.
    """
    if output_gguf.exists():
        return
    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [quantize_bin]
    if allow_requantize:
        cmd.append("--allow-requantize")
    cmd += [str(input_gguf), str(output_gguf), quant_type]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise LlamaCppError(
            f"llama-quantize failed for {quant_type}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _build_bench_cmd(
    bench_bin: str, model_gguf: Path, threads: int, n_prompt: int, n_gen: int,
    repetitions: int = 3, batch_size: Optional[int] = None,
    cache_type_k: Optional[str] = None, cache_type_v: Optional[str] = None,
) -> list[str]:
    cmd = [
        bench_bin,
        "-m", str(model_gguf),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-t", str(threads),
        "-r", str(repetitions),
        "-o", "json",
    ]
    # llama-bench has NO -c/--ctx-size flag (confirmed against real --help,
    # 2026-08-16) — context/KV-cache size is sized implicitly from -p/-n/-d.
    # Phase 2's n_ctx search parameter is therefore realized as -p (n_prompt)
    # here: the amount of context llama-bench actually processes/allocates
    # for. This is a deliberate, documented re-interpretation, not an
    # oversight — see article_draft.md Section 3.5.
    if batch_size is not None:
        cmd += ["-b", str(batch_size)]
    if cache_type_k is not None:
        cmd += ["-ctk", cache_type_k]
    if cache_type_v is not None:
        cmd += ["-ctv", cache_type_v]
    return cmd


def _parse_bench_json(stdout: str) -> BenchResult:
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise LlamaCppError(
            f"Could not parse llama-bench JSON output (flag names may have changed "
            f"in your llama.cpp version — check `llama-bench --help`):\n{stdout}"
        ) from exc

    # CONFIRMED against a real build (2026-08-15): there is no "test" string
    # field (no "pp512"/"tg128" labels) — rows are distinguished by
    # n_prompt > 0 (pp row, n_gen == 0) vs n_gen > 0 (tg row, n_prompt == 0).
    pp_values = [row["avg_ts"] for row in rows if row.get("n_prompt", 0) > 0 and row.get("n_gen", 0) == 0]
    tg_values = [row["avg_ts"] for row in rows if row.get("n_gen", 0) > 0 and row.get("n_prompt", 0) == 0]

    return BenchResult(
        prompt_tokens_per_sec=sum(pp_values) / len(pp_values) if pp_values else None,
        gen_tokens_per_sec=sum(tg_values) / len(tg_values) if tg_values else None,
        raw_json=rows,
    )


def run_bench(
    bench_bin: str, model_gguf: Path, threads: int, n_prompt: int, n_gen: int,
    repetitions: int = 3, batch_size: Optional[int] = None,
    cache_type_k: Optional[str] = None, cache_type_v: Optional[str] = None,
) -> BenchResult:
    """Run llama-bench with JSON output and extract prompt/gen tokens-per-second."""
    cmd = _build_bench_cmd(bench_bin, model_gguf, threads, n_prompt, n_gen, repetitions, batch_size, cache_type_k, cache_type_v)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise LlamaCppError(f"llama-bench failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return _parse_bench_json(result.stdout)


@dataclass
class InstrumentedBenchResult:
    bench: BenchResult
    peak_rss_bytes: int
    energy_joules: float
    avg_power_watts: float
    wall_time_s: float


def run_bench_instrumented(
    bench_bin: str, model_gguf: Path, threads: int, n_prompt: int, n_gen: int,
    repetitions: int = 1, batch_size: Optional[int] = None,
    cache_type_k: Optional[str] = None, cache_type_v: Optional[str] = None,
    rss_interval_s: float = 0.05, power_interval_s: float = 1.0,
    vcgencmd_bin: str = "vcgencmd", timeout_s: Optional[float] = None,
) -> InstrumentedBenchResult:
    """One llama-bench invocation, instrumented with concurrent peak-RSS and
    PMIC power sampling — the real Phase 2 evaluator's core measurement,
    combining latency (via bench JSON), memory, and energy in a single
    subprocess run rather than three separate model loads per evaluation.
    """
    from measurement.power_logger import read_pmic_rails, total_power_watts

    cmd = _build_bench_cmd(bench_bin, model_gguf, threads, n_prompt, n_gen, repetitions, batch_size, cache_type_k, cache_type_v)

    peak_rss = 0
    power_samples: list[tuple[float, float]] = []
    stop_flag = threading.Event()

    start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def _poll_rss():
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
            if stop_flag.wait(rss_interval_s):
                break

    def _poll_power():
        while not stop_flag.is_set():
            try:
                rails = read_pmic_rails(vcgencmd_bin)
                power_samples.append((time.monotonic(), total_power_watts(rails)))
            except Exception:
                pass
            if stop_flag.wait(power_interval_s):
                break

    rss_thread = threading.Thread(target=_poll_rss, daemon=True)
    power_thread = threading.Thread(target=_poll_power, daemon=True)
    rss_thread.start()
    power_thread.start()

    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        stop_flag.set()
        rss_thread.join()
        power_thread.join()
        raise LlamaCppError(f"Command timed out after {timeout_s}s: {' '.join(cmd)}")

    stop_flag.set()
    rss_thread.join()
    power_thread.join()
    wall_time = time.monotonic() - start

    if proc.returncode != 0:
        raise LlamaCppError(f"llama-bench failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

    energy = 0.0
    if len(power_samples) >= 2:
        for (t0, p0), (t1, p1) in zip(power_samples, power_samples[1:]):
            energy += (p0 + p1) / 2.0 * (t1 - t0)
        avg_power = energy / wall_time if wall_time > 0 else 0.0
    elif power_samples:
        avg_power = power_samples[0][1]
        energy = avg_power * wall_time
    else:
        avg_power = 0.0

    return InstrumentedBenchResult(
        bench=_parse_bench_json(stdout),
        peak_rss_bytes=peak_rss,
        energy_joules=energy,
        avg_power_watts=avg_power,
        wall_time_s=wall_time,
    )


_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")


def run_perplexity(
    perplexity_bin: str, model_gguf: Path, corpus_path: Path, ctx_size: int,
    cache_type_k: Optional[str] = None, cache_type_v: Optional[str] = None,
) -> PerplexityResult:
    """Run llama-perplexity and extract the final PPL estimate.

    cache_type_k/v (confirmed present on llama-perplexity, same flags as
    llama-bench) let quality be measured under a given KV-cache quantization
    — the one Phase 2 search parameter that actually affects quality.
    """
    cmd = [
        perplexity_bin,
        "-m", str(model_gguf),
        "-f", str(corpus_path),
        "-c", str(ctx_size),
    ]
    if cache_type_k is not None:
        cmd += ["-ctk", cache_type_k]
    if cache_type_v is not None:
        cmd += ["-ctv", cache_type_v]
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
