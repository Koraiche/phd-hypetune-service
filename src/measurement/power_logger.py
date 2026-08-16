"""Real energy measurement via the Raspberry Pi 5's PMIC ADC telemetry.

Discovered 2026-08-16: `vcgencmd pmic_read_adc` reports live voltage and
current for every power rail on the Pi 5 (VDD_CORE, DDR, 1V8/1V1/0V8 system
rails, WiFi, HDMI, etc.). Summing V*I across all rails gives a real,
physically grounded total system power draw — no external power meter or
INA219 sensor needed, which is a meaningfully better position than
article_draft.md Section 3.9 originally assumed when it listed
"USB-C inline power meter / INA219 sensor on GPIO" as the only options.

This does NOT capture power drawn upstream of the rails the PMIC monitors
(e.g. losses before the 5V input), so it is a lower bound on wall-socket
power, not an absolute one — worth stating plainly rather than implying
whole-device power draw.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

def read_pmic_rails(vcgencmd_bin: str = "vcgencmd") -> dict:
    """Parse `vcgencmd pmic_read_adc` into {rail_name: {"A": amps, "V": volts}}."""
    result = subprocess.run([vcgencmd_bin, "pmic_read_adc"], capture_output=True, text=True, timeout=5)
    if result.returncode != 0:
        raise RuntimeError(f"vcgencmd pmic_read_adc failed: {result.stderr}")

    rails: dict = {}
    for line in result.stdout.splitlines():
        m = re.match(r"^\s*(\S+)_(A|V)\s+(current|volt)\(\d+\)=([\d.]+)[AV]\s*$", line)
        if not m:
            continue
        rail, kind = m.group(1), m.group(2)
        value = float(m.group(4))
        rails.setdefault(rail, {})[kind] = value
    return rails


def total_power_watts(rails: dict) -> float:
    """Sum V*I across every rail that has both a current and voltage reading."""
    total = 0.0
    for _rail, vals in rails.items():
        if "A" in vals and "V" in vals:
            total += vals["A"] * vals["V"]
    return total


@dataclass
class EnergyMeasurement:
    energy_joules: float
    avg_power_watts: float
    peak_power_watts: float
    wall_time_s: float
    n_samples: int
    samples_watts: list = field(default_factory=list)


def run_with_energy(
    cmd: list[str],
    sample_interval_s: float = 0.2,
    vcgencmd_bin: str = "vcgencmd",
    timeout_s: Optional[float] = None,
) -> tuple[subprocess.CompletedProcess, EnergyMeasurement]:
    """Run a subprocess while sampling total PMIC power, integrating to joules.

    Trapezoidal integration over the wall-clock-timestamped power samples.
    Mirrors llama_cpp_runner.run_with_peak_rss's polling-thread structure.
    """
    samples: list[tuple[float, float]] = []  # (timestamp, watts)
    stop_flag = threading.Event()

    def _poll():
        while not stop_flag.is_set():
            try:
                rails = read_pmic_rails(vcgencmd_bin)
                samples.append((time.monotonic(), total_power_watts(rails)))
            except Exception:
                pass
            if stop_flag.wait(sample_interval_s):
                break

    sampler = threading.Thread(target=_poll, daemon=True)
    start = time.monotonic()
    sampler.start()

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)

    stop_flag.set()
    sampler.join()
    wall_time = time.monotonic() - start

    if len(samples) < 2:
        # Workload finished faster than one sample interval — fall back to a
        # single instantaneous reading times wall time, clearly worse
        # resolution but not a crash.
        watts = samples[0][1] if samples else 0.0
        energy = watts * wall_time
        return proc, EnergyMeasurement(
            energy_joules=energy, avg_power_watts=watts, peak_power_watts=watts,
            wall_time_s=wall_time, n_samples=len(samples), samples_watts=[s[1] for s in samples],
        )

    energy = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        energy += (p0 + p1) / 2.0 * (t1 - t0)

    watts_list = [w for _, w in samples]
    avg_power = energy / wall_time if wall_time > 0 else 0.0

    return proc, EnergyMeasurement(
        energy_joules=energy,
        avg_power_watts=avg_power,
        peak_power_watts=max(watts_list),
        wall_time_s=wall_time,
        n_samples=len(samples),
        samples_watts=watts_list,
    )
