"""Class-B ride summaries (part 3), against hand-computed values on synthetic series."""

from __future__ import annotations

import pytest

from vehicle_server.defs import vehicle_catalog, vehicle_cl250, vehicle_constants, vehicle_derived
from vehicle_server.derived import engine, summary
from vehicle_server.derived.context import Feed, InputInvalid
from vehicle_server.derived.formulas import FORMULAS
from vehicle_server.derived.inputs import resolve_sensor
from vehicle_server.derived.summary import NOT_IMPLEMENTED_REASONS, SUMMARIES, SummaryEnv
from vehicle_server.gps_fix import fix_values_by_name, usable_fix_values

from .derived_helpers import FakeInputs, const_series, ramp

SPECS = vehicle_derived.SIGNALS
UNSTORED_SENSORS = {"lean_angle"}
CLASS_B = [name for name, spec in SPECS.items() if spec["class"] == "B"]
SPEED_MPS = "sensor.gps_ground_speed"
FIX = "sensor.gps_fix_type"
RPM = "did.ENGINE_SPEED"
LOAD = "did.ENGINE_LOAD"
COOLANT = "did.COOLANT_TEMP"


def _out(name: str, times: list[int], values: list[float | None]) -> engine.SignalOutput:
    return engine.SignalOutput(name, SPECS[name]["unit"], times, values, [None] * len(times))


def _run(
    series: dict | None = None,
    outputs: list[engine.SignalOutput] | None = None,
    specs: dict | None = None,
) -> summary.SummaryEvaluation:
    evaluation = engine.Evaluation(outputs=outputs or [])
    return summary.evaluate(FakeInputs(series or {}), evaluation, specs)


def _scalar(result: summary.SummaryEvaluation, name: str) -> tuple[float | None, str | None]:
    entry = result.coverage[name]
    assert entry["status"] == "computed", entry
    return entry["value"], entry["flag"]


def _cells(result: summary.SummaryEvaluation, name: str) -> dict[int, float | None]:
    return {row.cell: row.value for row in result.rows if row.signal == name}


# --- the registry matches the defs ---------------------------------------------------------


def test_every_summary_is_a_ready_class_b_signal_with_declared_resolvable_inputs():
    for name in SUMMARIES:
        spec = SPECS[name]
        assert spec["class"] == "B"
        assert spec["ready_phase0"] and not spec.get("pending_inputs")
        for inp in spec["inputs"]:
            kind, _, key = inp.partition(".")
            if kind == "did":
                assert key in vehicle_catalog.SIGNALS, inp
            elif kind == "derived":
                assert key in SPECS, inp
            elif kind == "const":
                assert key in vehicle_constants.CONSTANTS, inp
            else:
                assert kind == "sensor" and key in vehicle_derived.SENSORS, inp
                if key in UNSTORED_SENSORS:  # not stored: missing_inputs in a real session
                    assert resolve_sensor(key) is None
                else:
                    assert resolve_sensor(key) is not None, f"{inp} has no parquet column"
        for axis in spec.get("bins") or []:
            assert axis["input"] in spec["inputs"]
        assert spec["record"] in vehicle_derived.RIDE_SUMMARY_RECORDS


def test_summaries_only_read_declared_inputs():
    """SummaryEnv refuses an undeclared input (KeyError): run every summary once over data
    for each declared input and require that nothing but a data problem is raised."""
    for name, entry in SUMMARIES.items():
        spec = SPECS[name]
        feeds: dict[str, Feed] = {}
        outputs: dict[str, engine.SignalOutput] = {}
        results: dict[str, float | None] = {}
        for inp in spec["inputs"]:
            kind, _, key = inp.partition(".")
            if kind == "derived" and SPECS[key]["class"] == "A":
                outputs[key] = _out(key, [0, 10, 20], [1.0, 2.0, 3.0])
            elif kind == "derived":
                results[key] = 1.0
            else:
                value = next(iter(usable_fix_values())) if inp == FIX else 1.0
                times, values = const_series(0, 5000, 10, value)
                feeds[inp] = Feed(times, values, 300)
        env = SummaryEnv(spec, feeds, outputs, results)
        try:
            entry.fn(env)
        except InputInvalid:
            pass
        undeclared = next(
            n for n in ("derived.trip_time", "did.THROTTLE_POS") if n not in env.inputs
        )
        with pytest.raises(KeyError):
            env.valid(undeclared)
        with pytest.raises(KeyError):
            env.samples(undeclared)


def test_not_implemented_reasons_cover_every_ready_class_b_signal_without_a_summary():
    pending = {n for n in CLASS_B if n not in SUMMARIES and SPECS[n]["ready_phase0"]} - {
        n for n in CLASS_B if SPECS[n].get("pending_inputs")
    }
    assert set(NOT_IMPLEMENTED_REASONS) == pending
    result = _run()
    for name in pending:
        entry = result.coverage[name]
        assert entry == {"status": "not_implemented", "reason": NOT_IMPLEMENTED_REASONS[name]}
    # every "input not implemented" class-A input is indeed not computed by the reference
    for name, reason in NOT_IMPLEMENTED_REASONS.items():
        if reason.startswith("input not implemented: derived."):
            missing = reason.removeprefix("input not implemented: derived.")
            assert f"derived.{missing}" in SPECS[name]["inputs"]
            assert missing not in FORMULAS


def test_exact_reasons():
    reasons = NOT_IMPLEMENTED_REASONS
    assert set(reasons) - {n for n in reasons if n.startswith("window_")} == {
        "ride_fan_on_temperature",
        "ride_final_drive_ratio",
        "ride_tester_nrc78_count",
        "coast_down_cda",
        "coast_down_rolling_resistance",
    }
    assert reasons["coast_down_cda"] == reasons["coast_down_rolling_resistance"]
    assert reasons["coast_down_cda"].startswith("definition: coast-down speed band")
    assert all(reason.startswith("definition:") for reason in reasons.values())


def test_not_ready_and_missing_inputs():
    result = _run()
    for name in CLASS_B:
        spec = SPECS[name]
        if not spec["ready_phase0"] or spec.get("pending_inputs"):
            assert result.coverage[name]["status"] == "not_ready"
            assert result.coverage[name]["pending_inputs"] == list(spec.get("pending_inputs") or [])
    assert result.coverage["ride_ltft_mean"] == {
        "status": "missing_inputs",
        "missing": ["did.LONG_TERM_FUEL_TRIM_B1"],
    }
    # a class-B input that is not computed is missing too
    assert result.coverage["ride_fuel_consumption"]["missing"] == [
        "derived.ride_fuel_used",
        "derived.ride_distance",
    ]
    assert not result.rows
    assert list(result.coverage) == [n for n in SPECS if n in CLASS_B]


def test_class_a_signals_are_not_in_the_summary_coverage():
    assert set(_run().coverage) == set(CLASS_B)


# --- distance, time, stops -------------------------------------------------------------------


def test_usage_takes_the_last_valid_value_and_counts_valid_stops():
    result = _run(
        outputs=[
            _out("trip_distance", [0, 20, 40], [0.0, 1.5, None]),
            _out("trip_time", [0, 20], [0.0, 90.0]),
            _out("idle_time", [0, 20], [0.0, 12.0]),
            _out("stop_event", [10, 20, 30], [2.0, None, 3.0]),
        ]
    )
    assert _scalar(result, "ride_distance") == (1.5, None)
    assert _scalar(result, "ride_time") == (90.0, None)
    assert _scalar(result, "ride_idle_time") == (12.0, None)
    assert _scalar(result, "ride_stop_count") == (2.0, None)


def test_no_stop_events_is_zero_stops_and_no_valid_row_is_invalid():
    result = _run(
        outputs=[_out("stop_event", [], []), _out("trip_distance", [0, 10], [None, None])]
    )
    assert _scalar(result, "ride_stop_count") == (0.0, None)
    assert _scalar(result, "ride_distance") == (None, "invalid_input")


# --- fuel ------------------------------------------------------------------------------------


def test_fuel_used_integrates_litres_per_hour():
    # 3600 L/h = 1 L/s held 1 s twice (the last sample holds nothing)
    fuel = {"did.ENGINE_FUEL_RATE": ([0, 1000, 2000], [3600.0, 3600.0, 3600.0])}
    assert _scalar(_run(fuel), "ride_fuel_used") == (pytest.approx(2.0), None)


def test_fuel_used_integration_over_a_gap_adds_nothing():
    limit = vehicle_cl250.DID_STALE_AFTER_MS["ENGINE_FUEL_RATE"]
    fuel = {"did.ENGINE_FUEL_RATE": ([0, 20000], [3600.0, 3600.0])}
    # held only for the freshness limit, not for the 20 s gap
    assert _scalar(_run(fuel), "ride_fuel_used") == (pytest.approx(limit / 1000.0), None)


def test_did_input_is_held_for_its_own_limit_not_three_polls():
    # D-087: the hold limit is the DID's own stale_after_ms; with a limit narrower than 3 polls
    # (as VEHICLE_SPEED has: 300 ms vs 330 ms) a gap of 3 polls is not bridged beyond it
    poll = vehicle_catalog.SIGNALS["ENGINE_FUEL_RATE"]["poll_period_ms"]
    own = poll  # narrower than the old 3 x poll
    fuel = {"did.ENGINE_FUEL_RATE": ([0, 3 * poll], [3600.0, 3600.0])}
    evaluation = engine.Evaluation()
    result = summary.evaluate(FakeInputs(fuel), evaluation, did_limits={"ENGINE_FUEL_RATE": own})
    assert _scalar(result, "ride_fuel_used") == (pytest.approx(own / 1000.0), None)


def test_fuel_consumption_is_litres_per_100_km():
    fuel = {"did.ENGINE_FUEL_RATE": ([0, 1000, 2000], [3600.0, 3600.0, 3600.0])}
    result = _run(fuel, [_out("trip_distance", [0, 10], [0.0, 40.0])])
    assert _scalar(result, "ride_fuel_consumption") == (pytest.approx(5.0), None)  # 2 L / 40 km


def test_fuel_consumption_at_standstill_and_with_an_invalid_input():
    fuel = {"did.ENGINE_FUEL_RATE": ([0, 1000], [3600.0, 3600.0])}
    result = _run(fuel, [_out("trip_distance", [0, 10], [0.0, 0.0])])
    assert _scalar(result, "ride_fuel_consumption") == (None, "standstill")
    result = _run(fuel, [_out("trip_distance", [0], [None])])
    assert _scalar(result, "ride_fuel_consumption") == (None, "invalid_input")


def test_trim_statistics():
    ltft = {"did.LONG_TERM_FUEL_TRIM_B1": ([0, 100, 200], [-2.0, 0.0, 5.0])}
    stft = {"did.SHORT_TERM_FUEL_TRIM_B1": (list(range(8)), [2.0, 4, 4, 4, 5, 5, 7, 9])}
    result = _run(ltft | stft)
    assert _scalar(result, "ride_ltft_mean") == (pytest.approx(1.0), None)
    assert _scalar(result, "ride_stft_std") == (pytest.approx(2.0), None)  # population std
    one = _run({"did.SHORT_TERM_FUEL_TRIM_B1": ([0], [3.0])})
    assert _scalar(one, "ride_stft_std") == (None, "insufficient_samples")


# --- time in cells -----------------------------------------------------------------------------


def test_operating_point_time_cells_are_row_major_and_outside_values_are_reported():
    # speed 750 rpm bins, load 6.25 % bins: 1000 rpm -> 1, 50 % -> 8 -> cell 1 * 16 + 8 = 24
    # 12000 rpm is the excluded upper edge -> outside; 0 rpm / 0 % is cell 0
    times = [0, 100, 200, 300, 400, 500]
    rpm = [1000.0, 1000.0, 1000.0, 12000.0, 0.0, 0.0]
    load = [50.0] * 4 + [0.0, 0.0]
    result = _run({RPM: (times, rpm), LOAD: (times, load)})
    cells = _cells(result, "operating_point_time")
    assert len(cells) == 256
    assert cells[24] == pytest.approx(0.3)
    assert cells[0] == pytest.approx(0.1)  # t=400 held 100 ms; the last sample holds nothing
    assert sum(v for v in cells.values() if v) == pytest.approx(0.4)
    entry = result.coverage["operating_point_time"]
    assert entry["cells"] == 256 and entry["nonzero_cells"] == 2
    assert entry["total_s"] == pytest.approx(0.4)
    assert entry["outside_s"] == pytest.approx(0.1)  # t=300 (12000 rpm)
    assert "value" not in entry
    row = next(r for r in result.rows if r.cell == 24)
    assert row.edges == [(750.0, 1500.0), (50.0, 56.25)]
    assert next(r for r in result.rows if r.cell == 255).edges == [
        (11250.0, 12000.0),
        (93.75, 100.0),
    ]
    assert summary.cell_axes(SPECS["operating_point_time"]["bins"], 24) == [1, 8]


def test_time_at_temperature_holds_never_beyond_freshness():
    limit = vehicle_cl250.DID_STALE_AFTER_MS["COOLANT_TEMP"]
    # 85 degC -> bin 5 of 10 (30..130), 95 -> bin 6; the 8 s gap after t=2000 adds only `limit`
    result = _run({COOLANT: ([0, 1000, 2000, 10000], [85.0, 85.0, 95.0, 95.0])})
    cells = _cells(result, "ride_coolant_time_at_temp")
    assert len(cells) == 10
    assert cells[5] == pytest.approx(2.0)
    assert cells[6] == pytest.approx(limit / 1000.0)
    assert result.coverage["ride_coolant_time_at_temp"]["outside_s"] == 0.0
    below = _run({COOLANT: ([0, 1000], [20.0, 20.0])})
    assert below.coverage["ride_coolant_time_at_temp"]["outside_s"] == pytest.approx(1.0)
    assert below.coverage["ride_coolant_time_at_temp"]["nonzero_cells"] == 0


def test_oil_time_at_temperature_uses_its_own_bins():
    # 30..150 in 12 bins of 10: 100 degC -> bin 7
    oil = {"did.ENGINE_OIL_TEMP": ([0, 1000, 2000], [100.0, 100.0, 100.0])}
    cells = _cells(_run(oil), "ride_oil_time_at_temp")
    assert len(cells) == 12 and cells[7] == pytest.approx(2.0)
    row = next(r for r in _run(oil).rows if r.cell == 7)
    assert row.edges == [(100.0, 110.0)]


# --- thermal, GNSS, bus ------------------------------------------------------------------------


def test_coolant_rate_max_and_start_end_temperatures():
    series = {
        COOLANT: ([0, 1000, 2000], [20.0, 50.0, 80.0]),
        "did.INTAKE_AIR_TEMP": ([0, 500], [31.0, 35.0]),
    }
    result = _run(series, [_out("coolant_rate", [0, 1, 2, 3], [0.5, None, 2.5, 1.0])])
    assert _scalar(result, "ride_coolant_rate_max") == (2.5, None)
    assert _scalar(result, "ride_start_coolant_temp") == (20.0, None)
    assert _scalar(result, "ride_end_coolant_temp") == (80.0, None)
    assert _scalar(result, "ride_start_intake_temp") == (31.0, None)


def test_distance_gps_integrates_speed_and_adds_nothing_over_a_gap():
    usable = next(iter(usable_fix_values()))
    times, values = const_series(0, 10000, 100, 10.0)  # 10 m/s for 10 s = 0.1 km
    series = {SPEED_MPS: (times, values), FIX: (times, [usable] * len(times))}
    assert _scalar(_run(series), "ride_distance_gps") == (pytest.approx(0.1), None)
    gap = {SPEED_MPS: ([0, 5000], [10.0, 10.0]), FIX: ([0, 5000], [usable, usable])}
    # held 300 ms (SENSORS stale_after_ms) at 10 m/s = 3 m
    assert _scalar(_run(gap), "ride_distance_gps") == (pytest.approx(0.003), None)


def test_distance_gps_skips_a_no_fix_stretch_mid_ride():
    usable = next(iter(usable_fix_values()))
    unusable = next(v for v in fix_values_by_name().values() if v not in usable_fix_values())
    times, values = const_series(0, 20000, 100, 10.0)  # 10 m/s for 20 s = 0.2 km
    fix = [unusable if 5000 <= t < 15000 else usable for t in times]
    result = _run({SPEED_MPS: (times, values), FIX: (times, fix)})
    # the 10 s no-fix stretch contributes zero: half the full distance
    assert _scalar(result, "ride_distance_gps") == (pytest.approx(0.1), None)


def test_distance_gps_ignores_speed_samples_after_the_fix_type_goes_stale():
    usable = next(iter(usable_fix_values()))
    times, values = const_series(0, 10000, 100, 10.0)
    fix_times = list(range(0, 2001, 100))
    result = _run({SPEED_MPS: (times, values), FIX: (fix_times, [usable] * len(fix_times))})
    # counted from 0 to 2000 ms plus the 300 ms fix freshness = 2.3 s at 10 m/s = 0.023 km
    assert _scalar(result, "ride_distance_gps") == (pytest.approx(0.023), None)


def test_distance_gps_with_no_usable_fix_is_missing_inputs():
    unusable = next(v for v in fix_values_by_name().values() if v not in usable_fix_values())
    times, values = const_series(0, 10000, 100, 10.0)
    result = _run({SPEED_MPS: (times, values), FIX: (times, [unusable] * len(times))})
    assert result.coverage["ride_distance_gps"] == {
        "status": "missing_inputs",
        "missing": [FIX],
    }


def test_bus_off_count_sums_increments_and_a_restart_restarts_from_the_new_value():
    count = "sensor.can_bus_off_count"
    assert _scalar(_run({count: ([0, 1, 2, 3, 4], [0.0, 0, 2, 2, 5])}), "ride_bus_off_count") == (
        5.0,
        None,
    )
    # 0 -> 3 (+3), 3 -> 1 restart (+1), 1 -> 1, 1 -> 4 (+3)
    restart = {count: ([0, 1, 2, 3], [0.0, 3, 1, 1])}
    restart[count][0].append(4)
    restart[count][1].append(4.0)
    assert _scalar(_run(restart), "ride_bus_off_count") == (7.0, None)


def test_tester_maxima():
    series = {
        "sensor.tester_rtt_max": ([0, 1, 2], [10.0, 40.0, 25.0]),
        "sensor.tester_step_gap_max": ([0, 1], [5.0, 3.0]),
    }
    result = _run(series)
    assert _scalar(result, "ride_tester_rtt_max") == (40.0, None)
    assert _scalar(result, "ride_tester_step_gap_max") == (5.0, None)


# --- range -------------------------------------------------------------------------------------


def test_scalar_outside_the_defs_range_is_null_with_out_of_range():
    result = _run(outputs=[_out("trip_distance", [0], [3000.0])])  # range 0 .. 2000 km
    assert _scalar(result, "ride_distance") == (None, "out_of_range")
    row = next(r for r in result.rows if r.signal == "ride_distance")
    assert row.value is None and row.flag == "out_of_range"
    negative = _run(outputs=[_out("coolant_rate", [0], [-1.0])])  # range 0 .. 30
    assert _scalar(negative, "ride_coolant_rate_max") == (None, "out_of_range")


def test_a_cell_outside_the_defs_range_is_null_with_out_of_range():
    narrow = {**SPECS, "ride_coolant_time_at_temp": {**SPECS["ride_coolant_time_at_temp"]}}
    narrow["ride_coolant_time_at_temp"]["range"] = [0.0, 1.0]
    result = _run({COOLANT: ([0, 1000, 2000, 3000], [85.0] * 4)}, specs=narrow)
    rows = {r.cell: r for r in result.rows if r.signal == "ride_coolant_time_at_temp"}
    assert rows[5].value is None and rows[5].flag == "out_of_range"  # 3 s > 1 s
    assert rows[0].value == 0.0 and rows[0].flag is None
    entry = result.coverage["ride_coolant_time_at_temp"]
    assert entry["nonzero_cells"] == 0 and entry["total_s"] == 0.0


def test_a_failing_summary_is_an_error_entry_and_does_not_stop_the_others():
    def boom(_env):
        raise RuntimeError("boom")

    summaries = {**SUMMARIES, "ride_time": summary.Summary(boom)}
    result = summary.evaluate(
        FakeInputs({}),
        engine.Evaluation(
            outputs=[_out("trip_time", [0], [5.0]), _out("trip_distance", [0], [1.0])]
        ),
        summaries=summaries,
    )
    assert result.coverage["ride_time"] == {"status": "error", "message": "RuntimeError: boom"}
    assert _scalar(result, "ride_distance") == (1.0, None)


def test_a_ramp_crossing_a_bin_edge_splits_the_time():
    # 31 .. 41 degC every 100 ms: samples 31..39 in bin 0, 40 and 41 in bin 1; the last holds 0 s
    times, values = ramp(0, 1000, 100, lambda t: 31.0 + t / 100)
    cells = _cells(_run({COOLANT: (times, values)}), "ride_coolant_time_at_temp")
    assert cells[0] == pytest.approx(0.9)
    assert cells[1] == pytest.approx(0.1)
