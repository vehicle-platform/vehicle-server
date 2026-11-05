"""D-079 health-window summaries on synthetic rides (no real data)."""

from __future__ import annotations

import math

import pytest

from vehicle_server.defs import vehicle_derived
from vehicle_server.derived import engine, summary
from vehicle_server.derived.health_window_summaries import (
    FLAG_NO_WINDOW,
    NOT_IMPLEMENTED_WINDOW_REASONS,
)

from .derived_helpers import FakeInputs, const_series, patch_const

SPECS = vehicle_derived.SIGNALS
AMB = 20.0
TAU_S = 300.0
TAU_MS = TAU_S * 1000.0


def _run(series: dict, outputs: list[engine.SignalOutput] | None = None):
    return summary.evaluate(FakeInputs(series), engine.Evaluation(outputs=outputs or []))


def _value(result, name: str) -> tuple[float | None, str | None]:
    entry = result.coverage[name]
    assert entry["status"] == "computed", entry
    return entry["value"], entry["flag"]


def _ride():
    rpm_times = list(range(0, 3_001, 100))
    rpm_values = [0.0] * len(rpm_times[:-1]) + [1_200.0]
    for t in range(3_100, 30_000, 100):
        rpm_times.append(t)
        rpm_values.append(1_200.0 if t < 16_500 else 1_300.0)
    rpm_times.append(30_000)
    rpm_values.append(0.0)
    for t in range(30_100, 91_001, 100):
        rpm_times.append(t)
        rpm_values.append(0.0)

    coolant_times = list(range(0, 30_001, 1_000))
    coolant_values = [85.0] * len(coolant_times)
    for t in range(31_000, 91_001, 1_000):
        coolant_times.append(t)
        coolant_values.append(AMB + (85.0 - AMB) * math.exp(-(t - 30_000) / TAU_MS))

    ambient_times = list(range(0, 90_001, 5_000)) + [91_000]
    map_times = list(range(0, 91_001, 1_000))
    map_values = [110.0 if 17_000 <= t <= 30_000 else 100.0 for t in map_times]
    stft_times = [*range(3_000, 16_001, 1_000), 16_500, *range(17_000, 30_000, 1_000)]
    stft_values = [4.0 if t < 16_500 else 6.0 for t in stft_times]

    series = {
        "did.ENGINE_SPEED": (rpm_times, rpm_values),
        "did.COOLANT_TEMP": (coolant_times, coolant_values),
        "did.AMBIENT_AIR_TEMP": (ambient_times, [AMB] * len(ambient_times)),
        "did.INTAKE_AIR_TEMP": const_series(0, 91_000, 1_000, 80.0),
        "did.INTAKE_MAP": (map_times, map_values),
        "did.BAROMETRIC_PRESSURE": const_series(0, 91_000, 1_000, 100.0),
        "did.BATTERY_VOLTAGE": const_series(0, 91_000, 1_000, 12.6),
        "did.SHORT_TERM_FUEL_TRIM_B1": (stft_times, stft_values),
        "did.VEHICLE_SPEED": const_series(0, 91_000, 100, 0.0),
        "did.THROTTLE_POS": const_series(0, 91_000, 1_000, 10.0),
    }
    gear = engine.SignalOutput(
        "gear_estimate",
        SPECS["gear_estimate"]["unit"],
        list(range(0, 91_001, 1_000)),
        [0.0] * 92,
        [None] * 92,
    )
    return series, [gear]


def test_every_computed_window_signal_on_one_synthetic_ride():
    series, outputs = _ride()
    result = _run(series, outputs)

    expected = {
        "window_key_on_rest_voltage": 12.6,
        "window_key_on_sensor_spread": 65.0,
        "window_key_on_map_baro_delta": 0.0,
        "window_key_on_warm_restart": 1.0,
        "window_key_on_count": 1.0,
        "window_crank_count": 1.0,
        "window_warm_idle_rpm_spread": 100.0,
        "window_warm_idle_rpm_drift": 100.0,
        "window_warm_idle_map_spread": 10.0,
        "window_warm_idle_stft_mean": 5.0,
        "window_warm_idle_stft_std": math.sqrt(28.0 / 27.0),
        "window_warm_idle_count": 1.0,
        "window_cruise_count": 0.0,
        "window_post_stop_coolant_start": 85.0,
        "window_post_stop_coolant_end": AMB + (85.0 - AMB) * math.exp(-61.0 / TAU_S),
        "window_post_stop_coolant_tau": TAU_S,
        "window_post_stop_count": 1.0,
    }
    for name, value in expected.items():
        assert _value(result, name) == (pytest.approx(value), None), name
    assert _value(result, "window_post_stop_coolant_tau")[0] == pytest.approx(TAU_S, rel=1e-6)


def test_no_qualifying_window_reports_no_window():
    result = _run(
        {
            "did.BATTERY_VOLTAGE": const_series(0, 2_000, 1_000, 12.6),
            "did.ENGINE_SPEED": const_series(0, 2_000, 1_000, 1_200.0),
        }
    )

    assert _value(result, "window_key_on_rest_voltage") == (None, FLAG_NO_WINDOW)
    assert _value(result, "window_key_on_count") == (0.0, None)


def test_out_of_range_value_is_null_not_clamped():
    result = _run(
        {
            "did.BATTERY_VOLTAGE": const_series(0, 2_000, 1_000, 20.0),
            "did.ENGINE_SPEED": const_series(0, 2_000, 1_000, 0.0),
        }
    )

    assert _value(result, "window_key_on_rest_voltage") == (None, "out_of_range")


def test_cold_soak_constant_patch_changes_warm_restart(monkeypatch):
    series, outputs = _ride()
    assert _value(_run(series, outputs), "window_key_on_warm_restart") == (1.0, None)

    patch_const(monkeypatch, "window_cold_soak_max_delta_degc", 100.0)

    assert _value(_run(series, outputs), "window_key_on_warm_restart") == (0.0, None)


def test_every_ready_window_signal_is_computed_or_has_a_specific_reason():
    computed = {
        name for name, spec in SPECS.items() if spec.get("window") and name in summary.SUMMARIES
    }
    assert len(computed) == 17
    for name in computed:
        assert name not in summary.NOT_IMPLEMENTED_REASONS

    for name, spec in SPECS.items():
        if (
            spec.get("window")
            and spec["ready_phase0"]
            and not spec.get("pending_inputs")
            and name not in computed
        ):
            reason = summary.NOT_IMPLEMENTED_REASONS.get(name)
            assert reason and reason != summary._WINDOW_REASON, name

    assert set(NOT_IMPLEMENTED_WINDOW_REASONS) == {
        "window_cruise_final_drive_residual",
        "window_cruise_ve_residual",
    }
