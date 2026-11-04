"""D-103 class-A windows and residuals (LR15-2) on synthetic series."""

from __future__ import annotations

import math
import statistics

import pytest

from .derived_helpers import const_series, outcome, patch_const, ramp, series_outcomes

SPEED = "did.VEHICLE_SPEED"
RPM = "did.ENGINE_SPEED"
THROTTLE = "did.THROTTLE_POS"
COOLANT = "did.COOLANT_TEMP"
BATT = "did.BATTERY_VOLTAGE"
FUEL = "did.FUEL_LEVEL"
ACCEL = "derived.vehicle_accel"
AZ = "sensor.imu_accel_z"


def _rpm_pattern(stop: int, fn) -> tuple:
    return ramp(0, stop, 110, lambda t: fn(t // 110))


def _idle_feeds(rpm_fn, stop=40000, speed=0.0, coolant=90.0) -> dict:
    return {
        RPM: _rpm_pattern(stop, rpm_fn),
        SPEED: const_series(0, stop, 100, speed),
        COOLANT: const_series(0, stop, 1000, coolant),
    }


# --- idle_rpm_std -----------------------------------------------------------------------------


def test_idle_rpm_std_is_the_population_std_of_the_window():
    values = {i: 1350.0 + 10.0 * (i % 7) for i in range(400)}
    f = _idle_feeds(lambda i: values[i])
    t = 30030  # a grid time on the 110 ms rpm samples
    [(value, flag)] = series_outcomes("idle_rpm_std", f, [t])
    window = [values[i] for i in range(400) if t - 10000 < i * 110 <= t]
    assert flag is None and value == pytest.approx(statistics.pstdev(window))


def test_idle_rpm_std_leaves_out_samples_that_are_not_warm_idle():
    # moving: no sample at rest; cold: no warm sample; revving: outside the idle band
    for kwargs in ({"speed": 20.0}, {"coolant": 60.0}):
        f = _idle_feeds(lambda i: 1400.0 + 20.0 * (i % 2), **kwargs)
        assert series_outcomes("idle_rpm_std", f, [30000]) == [(None, "insufficient_samples")]
    f = _idle_feeds(lambda i: 3000.0 + 20.0 * (i % 2))
    assert series_outcomes("idle_rpm_std", f, [30000]) == [(None, "insufficient_samples")]


def test_idle_rpm_std_stale_rpm_is_null():
    f = _idle_feeds(lambda i: 1400.0 + 20.0 * (i % 2))
    f[RPM] = ([0, 110], [1400.0, 1420.0])
    assert series_outcomes("idle_rpm_std", f, [30000]) == [(None, "stale_input")]


def test_idle_rpm_std_follows_the_warm_and_band_constants(monkeypatch):
    f = _idle_feeds(lambda i: 1400.0 + 20.0 * (i % 2))
    assert series_outcomes("idle_rpm_std", f, [30000])[0][0] == pytest.approx(10.0, abs=0.5)
    patch_const(monkeypatch, "derived_operating_coolant_degc", 95.0)
    assert series_outcomes("idle_rpm_std", f, [30000]) == [(None, "insufficient_samples")]
    patch_const(monkeypatch, "derived_operating_coolant_degc", 80.0)
    patch_const(monkeypatch, "derived_idle_rpm_max", 1410.0)  # the upper samples drop out
    assert series_outcomes("idle_rpm_std", f, [30000])[0] == (0.0, None)


# --- idle_rpm_drift ---------------------------------------------------------------------------


def test_idle_rpm_drift_is_the_window_mean_minus_the_first_full_window():
    f = _idle_feeds(lambda i: 1400.0 if i * 110 < 60000 else 1450.0, stop=130000)
    out = dict(
        zip([30000, 125000], series_outcomes("idle_rpm_drift", f, [30000, 125000]), strict=True)
    )
    assert out[30000] == (None, "insufficient_samples")  # the first full window is not complete
    assert out[125000][0] == pytest.approx(50.0) and out[125000][1] is None


def test_idle_rpm_drift_follows_the_idle_band(monkeypatch):
    f = _idle_feeds(lambda i: 1400.0 if i * 110 < 60000 else 1450.0, stop=130000)
    patch_const(monkeypatch, "derived_idle_rpm_max", 1420.0)  # the 1450 samples are not idle
    assert series_outcomes("idle_rpm_drift", f, [125000]) == [(None, "insufficient_samples")]


# --- charging_plateau_voltage -----------------------------------------------------------------


def _charging() -> dict:
    return {
        RPM: ramp(0, 20000, 100, lambda t: 5000.0 if t < 15000 else 2000.0),
        BATT: ramp(0, 20000, 1000, lambda t: 14.2 if t < 15000 else 12.4),
    }


def test_charging_plateau_is_the_window_mean_while_the_rpm_is_above_the_minimum(monkeypatch):
    value, flag = outcome("charging_plateau_voltage", _charging(), 20000)
    assert flag is None and value == pytest.approx(14.2)  # t = 11000..14000 only
    patch_const(monkeypatch, "charging_plateau_rpm_min", 1500.0)
    value, _ = outcome("charging_plateau_voltage", _charging(), 20000)
    assert value == pytest.approx((4 * 14.2 + 6 * 12.4) / 10)


def test_charging_plateau_without_any_charging_sample_is_insufficient():
    f = {RPM: const_series(0, 20000, 100, 1000.0), BATT: const_series(0, 20000, 1000, 12.4)}
    assert outcome("charging_plateau_voltage", f, 20000) == (None, "insufficient_samples")


# --- fuel_level_filtered ----------------------------------------------------------------------


def _fuel() -> dict:
    return {
        FUEL: ramp(0, 20000, 1000, lambda t: 40.0 if t in (5000, 6000) else 50.0),
        ACCEL: ramp(0, 20000, 100, lambda t: 2.0 if 4500 <= t <= 6500 else 0.0),
    }


def test_fuel_level_filtered_leaves_out_the_samples_under_acceleration(monkeypatch):
    value, flag = outcome("fuel_level_filtered", _fuel(), 20000)
    assert flag is None and value == pytest.approx(50.0)
    patch_const(monkeypatch, "fuel_level_filtered_accel_max_mps2", 3.0)
    assert outcome("fuel_level_filtered", _fuel(), 20000)[0] == pytest.approx(
        (19 * 50.0 + 2 * 40.0) / 21
    )


def test_fuel_level_filtered_with_every_sample_under_acceleration_is_insufficient():
    f = {FUEL: const_series(0, 20000, 1000, 50.0), ACCEL: const_series(0, 20000, 100, 3.0)}
    assert outcome("fuel_level_filtered", f, 20000) == (None, "insufficient_samples")


# --- road_roughness ---------------------------------------------------------------------------


def _az(frequency_hz: float, amplitude: float = 2.0, offset: float = 9.80665) -> dict:
    return {
        AZ: ramp(
            0,
            5000,
            10,
            lambda t: offset + amplitude * math.sin(2 * math.pi * frequency_hz * t / 1000.0),
        ),
        SPEED: const_series(0, 5000, 100, 50.0),
    }


def _gain(frequency: float, low: float, high: float) -> float:
    """Analytical gain of the first-order high-pass (corner low) and low-pass (corner high)."""
    return (
        (frequency / low)
        / math.sqrt(1 + (frequency / low) ** 2)
        / math.sqrt(1 + (frequency / high) ** 2)
    )


def test_road_roughness_passes_the_band_and_rejects_gravity_and_slow_motion():
    in_band = series_outcomes("road_roughness", _az(10.0), [4000])[0]
    expected = 2.0 / math.sqrt(2) * _gain(10.0, 5.0, 20.0)
    assert in_band[1] is None and in_band[0] == pytest.approx(expected, rel=0.06)
    slow = series_outcomes("road_roughness", _az(0.2), [4000])[0][0]
    assert slow < 0.1 * in_band[0]  # constant 1 g and a 0.2 Hz heave are outside the band


def test_road_roughness_follows_the_band_constants(monkeypatch):
    baseline = series_outcomes("road_roughness", _az(10.0), [4000])[0][0]
    patch_const(monkeypatch, "road_roughness_band_high_hz", 40.0)
    widened = series_outcomes("road_roughness", _az(10.0), [4000])[0][0]
    assert widened == pytest.approx(2.0 / math.sqrt(2) * _gain(10.0, 5.0, 40.0), rel=0.06)
    assert widened > baseline * 1.05
    patch_const(monkeypatch, "road_roughness_band_low_hz", 15.0)  # 10 Hz now below the band
    assert series_outcomes("road_roughness", _az(10.0), [4000])[0][0] < widened * 0.7


def test_road_roughness_only_while_moving_and_fresh(monkeypatch):
    f = _az(10.0)
    f[SPEED] = const_series(0, 5000, 100, 1.0)
    assert series_outcomes("road_roughness", f, [4000]) == [(None, "not_moving")]
    f[SPEED] = const_series(0, 5000, 100, 50.0)
    f[AZ] = (f[AZ][0][:100], f[AZ][1][:100])  # ends at 990 ms
    assert series_outcomes("road_roughness", f, [4000]) == [(None, "stale_input")]
    f = _az(10.0)
    patch_const(monkeypatch, "derived_moving_speed_min_kmh", 60.0)
    assert series_outcomes("road_roughness", f, [4000]) == [(None, "not_moving")]


# --- heat_soak_delta --------------------------------------------------------------------------


def test_heat_soak_delta_at_low_speed(monkeypatch):
    f = {
        "did.INTAKE_AIR_TEMP": const_series(0, 1000, 100, 48.0),
        "did.AMBIENT_AIR_TEMP": const_series(0, 1000, 100, 30.0),
        SPEED: const_series(0, 1000, 100, 5.0),
    }
    assert outcome("heat_soak_delta", f, 500) == (18.0, None)
    f[SPEED] = const_series(0, 1000, 100, 30.0)
    assert outcome("heat_soak_delta", f, 500) == (None, "not_applicable")
    patch_const(monkeypatch, "heat_soak_delta_speed_max_kmh", 40.0)
    assert outcome("heat_soak_delta", f, 500) == (18.0, None)


# --- idle_map_residual ------------------------------------------------------------------------


def test_idle_map_residual_at_closed_throttle_in_the_idle_band(monkeypatch):
    def feeds(map_kpa=40.0, throttle=0.0, rpm=1400.0) -> dict:
        return {
            "did.INTAKE_MAP": const_series(0, 1000, 100, map_kpa),
            THROTTLE: const_series(0, 1000, 100, throttle),
            RPM: const_series(0, 1000, 100, rpm),
        }

    assert outcome("idle_map_residual", feeds(), 500) == (5.0, None)  # 40 - 35
    assert outcome("idle_map_residual", feeds(throttle=10.0), 500) == (None, "not_applicable")
    assert outcome("idle_map_residual", feeds(rpm=3000.0), 500) == (None, "not_applicable")
    patch_const(monkeypatch, "idle_map_residual_expected_kpa", 30.0)
    assert outcome("idle_map_residual", feeds(), 500) == (10.0, None)
    stale = feeds()
    stale["did.INTAKE_MAP"] = ([0], [40.0])
    assert outcome("idle_map_residual", stale, 20000) == (None, "stale_input")
