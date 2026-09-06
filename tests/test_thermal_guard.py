import pytest

from measurement.thermal_guard import ThermalGuard, ThermalGuardError


def sequence_reader(values):
    iterator = iter(values)
    return lambda: next(iterator)


def test_returns_immediately_below_limit():
    sleeps = []
    guard = ThermalGuard(
        temperature_reader=sequence_reader([64.0]),
        sleeper=sleeps.append,
        logger=lambda _message: None,
    )

    assert guard.wait_until_safe("test") == 64.0
    assert sleeps == []


def test_high_temperature_waits_through_hysteresis():
    sleeps = []
    guard = ThermalGuard(
        max_temp_c=75.0,
        resume_temp_c=65.0,
        check_interval_s=10,
        max_wait_s=30,
        temperature_reader=sequence_reader([76.0, 70.0, 64.0]),
        sleeper=sleeps.append,
        logger=lambda _message: None,
    )

    assert guard.wait_until_safe("test") == 64.0
    assert sleeps == [10, 10]


def test_timeout_refuses_to_run_hot():
    guard = ThermalGuard(
        max_temp_c=75.0,
        resume_temp_c=65.0,
        check_interval_s=10,
        max_wait_s=20,
        temperature_reader=sequence_reader([80.0, 79.0, 78.0]),
        sleeper=lambda _seconds: None,
        logger=lambda _message: None,
    )

    with pytest.raises(ThermalGuardError, match="refusing to run hot"):
        guard.wait_until_safe("test")
