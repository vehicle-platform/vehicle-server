"""D-103 class-B ride summaries (LR15-4) against known answers on synthetic rides."""

from __future__ import annotations

import pytest

from vehicle_server.defs import vehicle_derived
from vehicle_server.derived import engine, summary
from vehicle_server.gps_fix import fix_values_by_name

from .derived_helpers import FakeInputs, const_series, patch_const, ramp

SPECS = vehicle_derived.SIGNALS
FIX3D = fix_values_by_name()["fix3d"]
FIX2D = fix_values_by_name()["fix2d"]


def _out(name: str, times: list[int], values: list[float | None]) -> engine.SignalOutput:
    return engine.SignalOutput(name, SPECS[name]["unit"], times, values, [None] * len(times))


def _run(series: dict | None = None, outputs: list[engine.SignalOutput] | None = None):
    evaluation = engine.Evaluation(outputs=outputs or [])
    return summary.evaluate(FakeInputs(series or {}), evaluation)


def _value(result, name: str) -> tuple[float | None, str | None]:
    entry = result.coverage[name]
    assert entry["status"] == "computed", entry
    return entry["value"], entry["flag"]


# --- usage of the class-A flags and events ----------------------------------------------------


def test_flag_times_count_the_seconds_the_flag_is_set_and_never_over_a_gap():
    rows = _out("over_rev", [0, 110, 220, 330], [0.0, 1.0, 1.0, 0.0])
    assert _value(_run(outputs=[rows]), "ride_over_rev_time") == (pytest.approx(0.22), None)
    gap = _out("over_rev", [0, 110, 1000], [1.0, 1.0, 1.0])  # 330 ms stale limit caps the gap
    assert _value(_run(outputs=[gap]), "ride_over_rev_time") == (pytest.approx(0.44), None)
    cold = _out("cold_engine_abuse", [0, 110, 220], [0.0, 0.0, 0.0])
    assert _value(_run(outputs=[cold]), "ride_cold_abuse_time") == (0.0, None)
    none = _out("over_rev", [0, 110], [None, None])
    assert _value(_run(outputs=[none]), "ride_over_rev_time") == (None, "invalid_input")


def test_max_first_and_last_of_the_class_a_outputs():
    soak = _out("heat_soak_delta", [0, 110, 220], [5.0, 18.0, None])
    idle_std = _out("idle_rpm_std", [0, 110], [12.0, 30.0])
    drift = _out("idle_rpm_drift", [0, 110, 220], [4.0, 9.0, None])
    dip = _out("cranking_dip", [1000, 5000], [9.5, 8.0])
    rest = _out("battery_rest_voltage", [900, 4900], [12.6, 12.0])
    result = _run(outputs=[soak, idle_std, drift, dip, rest])
    assert _value(result, "ride_heat_soak_max")[0] == 18.0
    assert _value(result, "ride_idle_rpm_std")[0] == 30.0
    assert _value(result, "ride_idle_rpm_drift")[0] == 9.0
    assert _value(result, "ride_cranking_dip")[0] == 9.5
    assert _value(result, "ride_rest_voltage")[0] == 12.6
    empty = _run(outputs=[_out("cranking_dip", [], [])])
    assert _value(empty, "ride_cranking_dip") == (None, "invalid_input")


def test_shift_and_launch_counts():
    shifts = _out("shift_event", [1000, 2000, 3000], [400.0, 300.0, None])
    launches = _out("launch_event", [], [])
    result = _run(outputs=[shifts, launches])
    assert _value(result, "ride_shift_count")[0] == 2.0
    assert _value(result, "ride_launch_count")[0] == 0.0


# --- thermal ----------------------------------------------------------------------------------


def _thermal(coolant: list[float]) -> dict:
    times = list(range(0, 1000 * len(coolant), 1000))
    return {
        "did.COOLANT_TEMP": (times, coolant),
        "did.RUN_TIME_SINCE_START": (times, [t / 1000.0 + 5.0 for t in times]),
    }


def test_warmup_time_is_the_run_time_at_the_first_warm_coolant_sample(monkeypatch):
    series = _thermal([20.0, 50.0, 85.0, 90.0])
    assert _value(_run(series), "ride_warmup_time") == (7.0, None)
    patch_const(monkeypatch, "derived_operating_coolant_degc", 88.0)
    assert _value(_run(series), "ride_warmup_time") == (8.0, None)
    patch_const(monkeypatch, "derived_operating_coolant_degc", 95.0)
    assert _value(_run(series), "ride_warmup_time") == (None, "never_warm")


def test_warmup_time_is_invalid_when_the_ride_starts_warm():
    assert _value(_run(_thermal([85.0, 90.0])), "ride_warmup_time") == (None, "starts_warm")


def test_thermal_cycles_need_cold_then_hot_with_hysteresis(monkeypatch):
    assert (
        _value(_run(_thermal([30.0, 60.0, 85.0, 70.0, 45.0, 90.0])), "ride_thermal_cycle_count")[0]
        == 2.0
    )
    assert _value(_run(_thermal([85.0, 90.0, 60.0, 85.0])), "ride_thermal_cycle_count")[0] == 0.0
    series = _thermal([60.0, 85.0, 45.0, 85.0])
    assert (
        _value(_run(series), "ride_thermal_cycle_count")[0] == 1.0
    )  # only the second ride from cold
    patch_const(monkeypatch, "derived_cold_coolant_degc", 70.0)
    assert _value(_run(series), "ride_thermal_cycle_count")[0] == 2.0


# --- throttle-torque lag ----------------------------------------------------------------------


def _lag_series(delays: list[int], step_times: list[int]) -> dict:
    """Throttle steps 10 -> 60 % at the step times (20 ms samples); the torque follows each by
    its own delay."""
    end = max(step_times) + 3000

    def throttle(t: int) -> float:
        high = any(s <= t < s + 1500 for s in step_times)
        return 60.0 if high else 10.0

    def torque(t: int) -> float:
        high = any(s + d <= t < s + d + 1500 for s, d in zip(step_times, delays, strict=True))
        return 70.0 if high else 20.0

    return {
        "did.THROTTLE_POS": ramp(0, end, 20, throttle),
        "did.ACTUAL_TORQUE_PCT": ramp(0, end, 20, torque),
        "did.ENGINE_SPEED": const_series(0, end, 20, 3000.0),
    }


def test_throttle_torque_lag_is_the_median_over_the_steps(monkeypatch):
    assert _value(_run(_lag_series([200], [1000])), "throttle_torque_lag") == (200.0, None)
    series = _lag_series([100, 300, 200], [1000, 6000, 11000])
    assert _value(_run(series), "throttle_torque_lag") == (200.0, None)
    even = _lag_series([100, 300], [1000, 6000])
    assert _value(_run(even), "throttle_torque_lag") == (200.0, None)
    patch_const(
        monkeypatch, "throttle_torque_lag_step_min_pct", 60.0
    )  # the 50 % steps are too small
    assert _value(_run(series), "throttle_torque_lag") == (None, "no_step")


def test_throttle_torque_lag_without_a_step_is_invalid_and_a_slow_rise_is_no_step(monkeypatch):
    flat = _lag_series([200], [1000])
    flat["did.THROTTLE_POS"] = const_series(0, 4000, 20, 30.0)
    assert _value(_run(flat), "throttle_torque_lag") == (None, "no_step")
    series = _lag_series([200], [1000])
    patch_const(monkeypatch, "throttle_torque_lag_step_window_ms", 0)  # the rise takes one sample
    assert _value(_run(series), "throttle_torque_lag") == (None, "no_step")


# --- speedometer error and effective tyre radius ----------------------------------------------


def _steady(
    ecu: float = 100.0,
    gnss_mps: float = 26.0,
    fix: float = FIX3D,
    accuracy: float = 0.1,
    end: int = 20000,
    jitter: float = 0.0,
) -> dict:
    return {
        "did.VEHICLE_SPEED": ramp(
            0, end, 100, lambda t: ecu + (jitter if t // 100 % 2 else -jitter)
        ),
        "sensor.gps_ground_speed": const_series(0, end, 100, gnss_mps),
        "sensor.gps_speed_accuracy": const_series(0, end, 100, accuracy),
        "sensor.gps_fix_type": const_series(0, end, 100, fix),
    }


def test_speedometer_error_and_tyre_radius_at_steady_speed():
    result = _run(_steady())
    error, flag = _value(result, "speedometer_error")
    assert flag is None and error == pytest.approx((100.0 - 93.6) / 93.6 * 100.0)
    radius = _value(result, "effective_tyre_radius")[0]
    assert radius == pytest.approx(0.311 * 93.6 / 100.0)  # r_dynamic x GNSS / ECU


def test_calibration_leaves_out_samples_without_a_good_fix_or_steady_speed():
    for series in (_steady(fix=FIX2D), _steady(accuracy=0.5), _steady(end=5000), _steady(ecu=20.0)):
        assert _value(_run(series), "speedometer_error") == (None, "no_steady_samples")
        assert _value(_run(series), "effective_tyre_radius") == (None, "no_steady_samples")


def test_calibration_follows_its_constants(monkeypatch):
    jittery = _steady(jitter=2.0)  # spread 4 km/h
    assert _value(_run(jittery), "speedometer_error")[1] is None
    patch_const(monkeypatch, "derived_steady_speed_spread_kmh", 3.0)
    assert _value(_run(jittery), "speedometer_error") == (None, "no_steady_samples")
    patch_const(monkeypatch, "derived_steady_speed_spread_kmh", 4.0)
    patch_const(monkeypatch, "derived_gnss_speed_accuracy_max_mps", 0.05)
    assert _value(_run(jittery), "speedometer_error") == (None, "no_steady_samples")
    patch_const(monkeypatch, "derived_gnss_speed_accuracy_max_mps", 0.3)
    patch_const(monkeypatch, "derived_steady_min_duration_ms", 30000)
    assert _value(_run(jittery), "speedometer_error") == (None, "no_steady_samples")
    patch_const(monkeypatch, "derived_steady_min_duration_ms", 10000)
    patch_const(monkeypatch, "derived_calibration_speed_min_kmh", 120.0)
    assert _value(_run(jittery), "speedometer_error") == (None, "no_steady_samples")
    patch_const(monkeypatch, "derived_calibration_speed_min_kmh", 30.0)
    patch_const(monkeypatch, "rear_wheel_radius_dynamic_m", 0.3)
    assert _value(_run(_steady()), "effective_tyre_radius")[0] == pytest.approx(0.3 * 0.936)


# --- rider style ------------------------------------------------------------------------------


def _style_outputs(jerk: float = 5.0, brake: float = -6.0) -> list[engine.SignalOutput]:
    return [
        _out("vehicle_jerk", [0, 110, 220], [jerk, -jerk, jerk]),
        _out("braking_event", [500], [brake]),
        _out("shift_event", [], []),
        _out("launch_event", [], []),
    ]


def _throttle_ramp(rate_pct_s: float) -> dict:
    return {"did.THROTTLE_POS": ramp(0, 1000, 100, lambda t: rate_pct_s * t / 1000.0)}


def test_aggressiveness_is_the_mean_of_the_clamped_features(monkeypatch):
    # throttle 200 %/s (ref 200) -> 1; jerk 5 (ref 10) -> 0.5; braking 6 (ref 6) -> 1
    result = _run(_throttle_ramp(200.0), _style_outputs())
    assert _value(result, "ride_aggressiveness_score")[0] == pytest.approx(100 * (1 + 0.5 + 1) / 3)
    over = _run(_throttle_ramp(600.0), _style_outputs(jerk=50.0, brake=-20.0))
    assert _value(over, "ride_aggressiveness_score")[0] == pytest.approx(
        100.0
    )  # clamped, not scaled
    patch_const(monkeypatch, "ride_aggressiveness_score_jerk_ref_mps3", 5.0)
    again = _run(_throttle_ramp(200.0), _style_outputs())
    assert _value(again, "ride_aggressiveness_score")[0] == pytest.approx(100.0)
    patch_const(monkeypatch, "ride_aggressiveness_score_decel_ref_mps2", 12.0)
    last = _run(_throttle_ramp(200.0), _style_outputs())
    assert _value(last, "ride_aggressiveness_score")[0] == pytest.approx(100 * (1 + 1 + 0.5) / 3)


def test_aggressiveness_ignores_features_without_data_and_is_invalid_without_any():
    only_throttle = _run(
        _throttle_ramp(100.0),
        _style_outputs()[2:] + [_out("vehicle_jerk", [0], [None]), _out("braking_event", [], [])],
    )
    assert _value(only_throttle, "ride_aggressiveness_score")[0] == pytest.approx(50.0)
    flat = {"did.THROTTLE_POS": const_series(0, 1000, 100, 20.0)}
    nothing = _run(
        flat,
        _style_outputs()[2:] + [_out("vehicle_jerk", [0], [None]), _out("braking_event", [], [])],
    )
    assert _value(nothing, "ride_aggressiveness_score") == (None, "no_data")


def test_cornering_aggressiveness(monkeypatch):
    lateral = _out("lateral_accel", [0, 110, 220], [3.0, -3.0, 3.0])
    lean = {"sensor.lean_angle": const_series(0, 1000, 100, -20.0)}
    assert _value(_run(lean, [lateral]), "ride_cornering_aggressiveness")[0] == pytest.approx(50.0)
    patch_const(monkeypatch, "ride_cornering_aggressiveness_lean_ref_deg", 20.0)
    assert _value(_run(lean, [lateral]), "ride_cornering_aggressiveness")[0] == pytest.approx(75.0)
    empty = _out("lateral_accel", [0], [None])
    assert _value(_run(lean, [empty]), "ride_cornering_aggressiveness") == (None, "no_data")


# --- ECU and bus health -----------------------------------------------------------------------


def test_can_error_count_sums_the_positive_increments_of_both_levels():
    times = [0, 100, 200, 300]
    series = {
        "sensor.can_tx_error_count": (times, [0.0, 1.0, 1.0, 3.0]),
        "sensor.can_rx_error_count": (times, [5.0, 5.0, 7.0, 6.0]),
    }
    assert _value(_run(series), "ride_can_error_count") == (5.0, None)  # 3 + 2, the drop adds 0
    flat = {k: (times, [255.0] * 4) for k in series}
    assert _value(_run(flat), "ride_can_error_count") == (0.0, None)  # saturated: nothing


def test_unanswered_did_count_restarts_its_baseline_after_a_reboot_and_saturates():
    times = [0, 100, 200, 300, 400]
    series = {"sensor.unanswered_did_count": (times, [0.0, 2.0, 5.0, 1.0, 4.0])}
    assert _value(_run(series), "ride_unanswered_did_count") == (8.0, None)  # 5 + (4 - 1)
    big = {"sensor.unanswered_did_count": ([0, 100], [0.0, 1000.0])}
    assert _value(_run(big), "ride_unanswered_did_count") == (255.0, None)


def test_the_summaries_left_out_report_their_reason():
    result = _run()
    for name in (
        "ride_fan_on_temperature",
        "ride_final_drive_ratio",
        "ride_tester_nrc78_count",
        "coast_down_cda",
        "coast_down_rolling_resistance",
    ):
        assert result.coverage[name] == {
            "status": "not_implemented",
            "reason": summary.NOT_IMPLEMENTED_REASONS[name],
        }
