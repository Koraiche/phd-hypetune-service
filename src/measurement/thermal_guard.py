"""Temperature gate for long Raspberry Pi hardware-in-the-loop runs.

The guard is deliberately called at the evaluator boundary, so calibration,
metaheuristics, and baselines all receive the same protection.  A high
temperature starts a cooldown; hysteresis then waits for a lower resume
temperature to avoid repeatedly crossing the limit between evaluations.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable


class ThermalGuardError(RuntimeError):
    """Raised when temperature cannot be read or a cooldown times out."""


def read_soc_temp_c(vcgencmd_bin: str = "vcgencmd") -> float:
    try:
        result = subprocess.run(
            [vcgencmd_bin, "measure_temp"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ThermalGuardError(f"could not run {vcgencmd_bin} measure_temp: {exc}") from exc

    match = re.search(r"temp=([\d.]+)", result.stdout)
    if result.returncode != 0 or match is None:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit status {result.returncode}"
        raise ThermalGuardError(f"could not read SoC temperature: {detail}")
    return float(match.group(1))


@dataclass
class ThermalGuard:
    """Block a new hardware evaluation until the SoC is cool enough."""

    max_temp_c: float = 75.0
    resume_temp_c: float = 65.0
    check_interval_s: float = 15.0
    max_wait_s: float = 900.0
    vcgencmd_bin: str = "vcgencmd"
    logger: Callable[[str], None] = print
    temperature_reader: Callable[[], float] | None = field(default=None, repr=False)
    sleeper: Callable[[float], None] = field(default=time.sleep, repr=False)

    def __post_init__(self) -> None:
        if self.resume_temp_c > self.max_temp_c:
            raise ValueError("resume_temp_c must be <= max_temp_c")
        if self.check_interval_s <= 0 or self.max_wait_s < 0:
            raise ValueError("thermal-guard intervals must be positive")

    def _read(self) -> float:
        if self.temperature_reader is not None:
            return self.temperature_reader()
        return read_soc_temp_c(self.vcgencmd_bin)

    def wait_until_safe(self, context: str = "hardware evaluation") -> float:
        temp = self._read()
        if temp < self.max_temp_c:
            return temp

        self.logger(
            f"[thermal-guard] {temp:.1f}C before {context}; cooling to "
            f"{self.resume_temp_c:.1f}C or below"
        )
        waited = 0.0
        while temp > self.resume_temp_c:
            if waited >= self.max_wait_s:
                raise ThermalGuardError(
                    f"SoC remained at {temp:.1f}C after {waited:.0f}s while waiting "
                    f"to start {context}; refusing to run hot"
                )
            delay = min(self.check_interval_s, self.max_wait_s - waited)
            self.sleeper(delay)
            waited += delay
            temp = self._read()

        self.logger(f"[thermal-guard] {temp:.1f}C after {waited:.0f}s; starting {context}")
        return temp
