"""The derived-signal engine: ordering, staleness, readiness, range and grid rules."""

from __future__ import annotations

import pytest

from vehicle_server.defs import vehicle_cl250, vehicle_derived
from vehicle_server.derived import engine
from vehicle_server.derived.context import At
from vehicle_server.derived.d103_common import NOT_IMPLEMENTED_REASONS_A
from vehicle_server.derived.formulas import Formula

from .derived_helpers import FakeInputs, const_series, ramp


def _spec(inputs: list[str], **kw):
    return {
        "class": "A",
        "unit": "u",
        "range": [-100.0, 100.0],
        "inputs": inputs,
        "period_ms": 100,
        "stale_after_ms": 250,
        "ready_phase0": True,
        **kw,
    }


# The DID table's limits of the made-up DIDs of these tests (D-070: a DID input is judged by
# the DID's own stale_after_ms, not by the signal's); 250 ms = the `_spec` default.
LIMITS = {"X": 250, "A": 250, "B": 250}


def _point(fn):
    return Formula("point", fn)


def _by_name(evaluation: engine.Evaluation) -> dict[str, engine.SignalOutput]:
    return {o.name: o for o in evaluation.outputs}


# --- dependency order ------------------------------------------------------------


def test_dependency_order_puts_every_derived_input_first_in_the_real_defs():
    order = engine.dependency_order(vehicle_derived.SIGNALS)
    assert set(order) == set(vehicle_derived.SIGNALS)
    position = {name: i for i, name in enumerate(order)}
    for name, spec in vehicle_derived.SIGNALS.items():
        for inp in spec["inputs"]:
            if inp.startswith("derived."):
                assert position[inp.removeprefix("derived.")] < position[name], (name, inp)


def test_dependency_cycle_is_rejected():
    specs = {"a": _spec(["derived.b"]), "b": _spec(["derived.a"])}
    with pytest.raises(ValueError, match="cycle"):
        engine.dependency_order(specs)


def test_a_consumer_listed_before_its_producer_still_sees_the_producer():
    specs = {
        "third": _spec(["derived.second"]),
        "second": _spec(["derived.first"]),
        "first": _spec(["did.X"]),
    }
    formulas = {
        "first": _point(lambda at: at.value("did.X") + 1.0),
        "second": _point(lambda at: at.value("derived.first") * 2.0),
        "third": _point(lambda at: at.value("derived.second") + 10.0),
    }
    result = engine.evaluate(
        FakeInputs({"did.X": const_series(0, 500, 50, 1.0)}),
        specs,
        formulas,
        {},
        LIMITS,
    )
    assert [o.name for o in result.outputs] == ["first", "second", "third"]
    assert set(_by_name(result)["third"].values) == {14.0}


# --- staleness -------------------------------------------------------------------


def test_stale_input_gives_null_not_a_held_value():
    # X: 0..200 ms, then nothing until 1000..1200 ms; stale limit of the DID 250 ms
    times = [0, 100, 200, 1000, 1100, 1200]
    specs = {"s": _spec(["did.X"])}
    formulas = {"s": _point(lambda at: at.value("did.X"))}
    result = engine.evaluate(FakeInputs({"did.X": (times, [5.0] * 6)}), specs, formulas, {}, LIMITS)
    out = _by_name(result)["s"]
    by_time = dict(zip(out.times, zip(out.values, out.flags, strict=True), strict=True))
    assert by_time[200] == (5.0, None)
    assert by_time[400] == (5.0, None)  # age 200 <= 250
    assert by_time[500] == (None, "stale_input")  # age 300 > 250
    assert by_time[900] == (None, "stale_input")
    assert by_time[1000] == (5.0, None)


def test_the_stale_limit_of_a_did_is_its_own_not_the_signals():
    # VEHICLE_SPEED's own limit is 300 ms; the signal's (6 s here) must not widen it
    own = vehicle_cl250.DID_STALE_AFTER_MS["VEHICLE_SPEED"]
    assert own == 300
    specs = {"s": _spec(["did.VEHICLE_SPEED"], period_ms=100, stale_after_ms=6000)}
    formulas = {"s": _point(lambda at: at.value("did.VEHICLE_SPEED"))}
    times = [0, 1000]  # one sample at 0, the next only at 1000: a 1 s gap
    result = engine.evaluate(
        FakeInputs({"did.VEHICLE_SPEED": (times, [5.0, 5.0])}), specs, formulas, {}
    )
    out = _by_name(result)["s"]
    by_time = dict(zip(out.times, zip(out.values, out.flags, strict=True), strict=True))
    assert by_time[300] == (5.0, None)  # age 300 <= own limit
    assert by_time[400] == (None, "stale_input")  # age 400 > 300, though <= 6000
    assert by_time[900] == (None, "stale_input")
    assert by_time[1000] == (5.0, None)


def test_the_fuel_consumption_inputs_use_the_dids_own_limits():
    # a 6 s old VEHICLE_SPEED must not feed fuel_consumption_instant (own limit 300 ms)
    spec = vehicle_derived.SIGNALS["fuel_consumption_instant"]
    assert spec["stale_after_ms"] > vehicle_cl250.DID_STALE_AFTER_MS["VEHICLE_SPEED"]

    stale = engine.evaluate(
        FakeInputs(
            {
                "did.ENGINE_FUEL_RATE": const_series(0, 8000, 100, 3.0),
                "did.VEHICLE_SPEED": ([0, 7000], [36.0, 36.0]),
            }
        )
    )
    out = _by_name(stale)["fuel_consumption_instant"]
    by_time = dict(zip(out.times, zip(out.values, out.flags, strict=True), strict=True))
    assert by_time[6600] == (None, "stale_input")  # speed sample 6.6 s old


def test_the_stale_limit_of_a_sensor_comes_from_sensors_not_the_signal():
    specs = {"s": _spec(["sensor.x"], stale_after_ms=10_000)}
    formulas = {"s": _point(lambda at: at.value("sensor.x"))}
    sensors = {"x": {"stale_after_ms": 30}}
    result = engine.evaluate(
        FakeInputs({"sensor.x": ([0, 200], [1.0, 1.0])}), specs, formulas, sensors
    )
    out = _by_name(result)["s"]
    assert dict(zip(out.times, out.values, strict=True)) == {
        0: 1.0,
        100: None,
        200: 1.0,
    }


def test_an_invalid_derived_input_makes_the_consumer_invalid():
    # engine_torque is out of range (50 % of 100 Nm = 50 Nm > 30 Nm) -> torque_gap is null
    inputs = {
        "did.ACTUAL_TORQUE_PCT": const_series(0, 4000, 500, 50.0),
        "did.DEMAND_TORQUE_PCT": const_series(0, 4000, 500, 25.0),
        "did.REFERENCE_TORQUE": const_series(0, 4000, 500, 100.0),
    }
    result = engine.evaluate(FakeInputs(inputs))
    out = _by_name(result)
    assert set(out["engine_torque"].flags) == {"out_of_range"}
    assert set(out["torque_gap"].flags) == {"invalid_input"}
    assert result.coverage["torque_gap"]["valid_rows"] == 0


def test_gear_estimate_follows_the_invalid_ratio_at_standstill():
    inputs = {
        "did.ENGINE_SPEED": const_series(0, 1000, 100, 1500.0),
        "did.VEHICLE_SPEED": const_series(0, 1000, 100, 0.0),
    }
    out = _by_name(engine.evaluate(FakeInputs(inputs)))
    assert set(out["gearbox_ratio_measured"].flags) == {"standstill"}
    assert set(out["gear_estimate"].flags) == {"invalid_input"}


# --- readiness, class, events ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "reason"),
    [("engine_vibration_rms", "ready_phase0"), ("friction_torque", "ready_phase0")],
)
def test_signals_that_are_not_ready_are_never_evaluated(name: str, reason: str):
    spec = vehicle_derived.SIGNALS[name]
    assert not spec["ready_phase0"] and spec["pending_inputs"]
    # every input present: still no row, whatever formulas say
    inputs = {
        inp: const_series(0, 5000, 10, 1.0)
        for inp in spec["inputs"]
        if inp.startswith(("did.", "sensor."))
    }
    formulas = {name: _point(lambda at: 1.0)}
    result = engine.evaluate(FakeInputs(inputs), formulas=formulas)
    assert name not in _by_name(result)
    assert result.coverage[name]["status"] == engine.NOT_READY
    assert result.coverage[name]["reason"] == reason
    assert result.coverage[name]["pending_inputs"] == spec["pending_inputs"]


def test_pending_inputs_alone_make_a_signal_not_ready():
    specs = {"p": _spec(["did.X"], pending_inputs=["something undefined"])}
    result = engine.evaluate(
        FakeInputs({"did.X": const_series(0, 500, 50, 1.0)}),
        specs,
        {"p": _point(lambda at: 1.0)},
        {},
    )
    assert result.outputs == []
    assert result.coverage["p"]["status"] == engine.NOT_READY
    assert result.coverage["p"]["reason"] == "pending_inputs"


def test_other_classes_events_and_unimplemented_signals_are_not_implemented():
    result = engine.evaluate(FakeInputs({}))
    coverage = result.coverage
    assert coverage["speedometer_error"] == {
        "class": "B",
        "status": engine.NOT_IMPLEMENTED,
        "reason": "class B",
    }
    assert coverage["ride_distance"]["status"] == engine.NOT_IMPLEMENTED
    assert coverage["shift_event"]["status"] == engine.MISSING_INPUTS  # D-103 events are computed
    assert coverage["data_quality_mask"] == {
        "class": "A",
        "status": engine.NOT_IMPLEMENTED,
        "reason": NOT_IMPLEMENTED_REASONS_A["data_quality_mask"],
    }
    assert coverage["stop_event"]["status"] == engine.MISSING_INPUTS  # events are computed now
    assert coverage["clutch_state"]["status"] == engine.MISSING_INPUTS
    assert set(coverage) == set(vehicle_derived.SIGNALS)


# --- range -----------------------------------------------------------------------


def test_out_of_range_result_is_null_with_a_flag():
    specs = {"r": _spec(["did.X"], range=[0.0, 10.0])}
    formulas = {"r": _point(lambda at: at.value("did.X"))}
    values = [5.0, 11.0, -1.0, 10.0, float("nan")]
    result = engine.evaluate(
        FakeInputs({"did.X": ([0, 100, 200, 300, 400], values)}), specs, formulas, {}, LIMITS
    )
    out = _by_name(result)["r"]
    assert out.values == [5.0, None, None, 10.0, None]
    assert out.flags == [None, "out_of_range", "out_of_range", None, "out_of_range"]
    assert result.coverage["r"]["flag_counts"] == {"out_of_range": 3}
    assert result.coverage["r"]["valid_rows"] == 2


# --- grid ------------------------------------------------------------------------


def test_the_output_grid_uses_the_period_from_the_defs():
    inputs = {
        "did.ENGINE_SPEED": const_series(0, 5000, 50, 3000.0),
        "did.VEHICLE_SPEED": const_series(0, 5000, 50, 36.0),
    }
    result = engine.evaluate(FakeInputs(inputs))
    out = _by_name(result)
    for name in ("gearbox_ratio_measured", "trip_distance"):
        period = vehicle_derived.SIGNALS[name]["period_ms"]
        times = out[name].times
        assert {b - a for a, b in zip(times, times[1:], strict=False)} == {period}
    assert out["gearbox_ratio_measured"].times[0] == 0
    assert out["gearbox_ratio_measured"].times[-1] == 4950  # last grid point <= end (5000)


def test_the_grid_spans_the_overlap_of_the_inputs():
    specs = {"s": _spec(["did.A", "did.B"], period_ms=100)}
    formulas = {"s": _point(lambda at: at.value("did.A") + at.value("did.B"))}
    inputs = {"did.A": const_series(100, 600, 50, 1.0), "did.B": const_series(300, 900, 50, 2.0)}
    out = _by_name(engine.evaluate(FakeInputs(inputs), specs, formulas, {}, LIMITS))["s"]
    assert out.times == [300, 400, 500, 600]
    assert set(out.values) == {3.0}


def test_inputs_without_a_common_span_are_missing_inputs():
    specs = {"s": _spec(["did.A", "did.B"])}
    formulas = {"s": _point(lambda at: 0.0)}
    inputs = {"did.A": const_series(0, 100, 50, 1.0), "did.B": const_series(900, 1000, 50, 2.0)}
    result = engine.evaluate(FakeInputs(inputs), specs, formulas, {}, LIMITS)
    assert result.coverage["s"]["status"] == engine.MISSING_INPUTS
    assert result.coverage["s"]["reason"] == "no common time span"


# --- missing inputs and errors ---------------------------------------------------


def test_absent_inputs_are_listed_and_propagate_to_dependants():
    inputs = {"did.ENGINE_SPEED": const_series(0, 1000, 100, 3000.0)}
    result = engine.evaluate(FakeInputs(inputs))
    ratio = result.coverage["gearbox_ratio_measured"]
    assert ratio["status"] == engine.MISSING_INPUTS
    assert ratio["missing"] == ["did.VEHICLE_SPEED"]
    assert result.coverage["gear_estimate"]["missing"] == ["derived.gearbox_ratio_measured"]
    assert result.coverage["trip_time"]["status"] == engine.COMPUTED


def test_a_failing_formula_is_an_error_and_does_not_stop_the_others():
    def boom(at: At) -> float:
        raise RuntimeError("bad formula")

    specs = {"bad": _spec(["did.X"]), "good": _spec(["did.X"])}
    formulas = {"bad": _point(boom), "good": _point(lambda at: 1.0)}
    result = engine.evaluate(
        FakeInputs({"did.X": const_series(0, 500, 50, 1.0)}),
        specs,
        formulas,
        {},
        LIMITS,
    )
    assert result.coverage["bad"] == {
        "class": "A",
        "status": engine.ERROR,
        "message": "RuntimeError: bad formula",
    }
    assert result.coverage["good"]["status"] == engine.COMPUTED


def test_provisional_constants_are_reported():
    inputs = {
        "did.ENGINE_SPEED": const_series(0, 1000, 100, 3000.0),
        "did.VEHICLE_SPEED": ramp(0, 1000, 100, lambda t: 30.0),
    }
    entry = engine.evaluate(FakeInputs(inputs)).coverage["gearbox_ratio_measured"]
    assert entry["provisional_constants"] == ["rear_wheel_radius_dynamic_m"]


# --- event formulas ----------------------------------------------------------------


def test_an_event_formula_has_no_grid_and_its_rows_are_the_event_times():
    specs = {
        "e": _spec(["did.VEHICLE_SPEED"], range=[0.0, 5.0], period_ms=None, stale_after_ms=None)
    }
    specs["e"]["event"] = True
    formulas = {"e": Formula("event", lambda env: [(100, 3.0, None), (700, 7.0, None)])}
    result = engine.evaluate(
        FakeInputs({"did.VEHICLE_SPEED": const_series(0, 1000, 100, 1.0)}), specs, formulas, {}
    )
    out = _by_name(result)["e"]
    assert out.times == [100, 700]
    assert out.values == [3.0, None] and out.flags == [None, "out_of_range"]
    entry = result.coverage["e"]
    assert entry["status"] == engine.COMPUTED and entry["rows"] == 2 and entry["valid_rows"] == 1
    assert "period_ms" not in entry and entry["flag_counts"] == {"out_of_range": 1}


def test_an_event_with_no_formula_is_not_implemented_with_reason_event():
    specs = {"e": _spec(["did.X"], event=True)}
    result = engine.evaluate(FakeInputs({}), specs, {}, {})
    assert result.coverage["e"]["reason"] == "event"


def test_every_event_signal_has_a_formula():
    coverage = engine.evaluate(FakeInputs({})).coverage
    events = {n for n, s in vehicle_derived.SIGNALS.items() if s.get("event")}
    assert {"stop_event", "shift_event", "launch_event", "braking_event"} <= events
    for name in events:
        assert coverage[name]["status"] in (engine.MISSING_INPUTS, engine.NOT_READY), name


def test_class_a_signals_left_out_of_d103_report_their_reason():
    coverage = engine.evaluate(FakeInputs({})).coverage
    assert set(NOT_IMPLEMENTED_REASONS_A) == {"data_quality_mask"}
    for name, reason in NOT_IMPLEMENTED_REASONS_A.items():
        assert coverage[name]["status"] == engine.NOT_IMPLEMENTED
        assert coverage[name]["reason"] == reason
