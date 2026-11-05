"""D-079 health-window state machine over synthetic series (no real ride data)."""

from __future__ import annotations

from vehicle_server.defs import vehicle_derived
from vehicle_server.derived.health_windows import (
    CLOSE_CONDITION,
    CLOSE_MAX_DURATION,
    CLOSE_SESSION_END,
    CLOSE_STABLE,
    CLOSE_UNKNOWN,
    WindowInput,
    evaluate_window,
    input_names,
)

WINDOWS = vehicle_derived.WINDOWS
RPM = "did.ENGINE_SPEED"
COOLANT = "did.COOLANT_TEMP"
SPEED = "did.VEHICLE_SPEED"
GEAR = "derived.gear_estimate"
THROTTLE = "did.THROTTLE_POS"


def _input(times: list[int], values: list[float | None], stale_after_ms: int = 120_000):
    return WindowInput(tuple(times), tuple(values), stale_after_ms)


def _run(key: str, series: dict[str, WindowInput], trigger: str | None = None):
    spec = WINDOWS[key]
    return evaluate_window(
        spec, {name: series[name] for name in input_names(spec, trigger)}, trigger
    )


def test_key_on_at_most_one_and_session_end():
    result = _run("W1_KEY_ON", {RPM: _input([0, 2_500], [0.0, 0.0])})

    assert len(result.instances) == 1
    instance = result.instances[0]
    assert (instance.open_ms, instance.close_ms, instance.close_reason) == (
        0,
        2_500,
        CLOSE_SESSION_END,
    )
    assert instance.qualifies
    assert result.first_qualifying == instance
    assert result.qualifying_count == 1


def test_key_on_needs_the_condition_at_the_first_sample():
    result = _run("W1_KEY_ON", {RPM: _input([0, 2_500], [1_200.0, 0.0])})

    assert result.instances == ()
    assert result.first_qualifying is None
    assert result.qualifying_count == 0


def test_condition_flicker_rearms_only_after_the_condition_fails():
    result = _run(
        "W3_WARM_IDLE",
        {
            COOLANT: _input([0], [85.0]),
            SPEED: _input([0], [0.0]),
            RPM: _input([0, 1_000, 2_000, 3_000, 4_000], [1_200.0, 1_200.0, 0.0, 1_200.0, 1_200.0]),
        },
    )

    assert [(i.open_ms, i.close_ms, i.close_reason) for i in result.instances] == [
        (0, 2_000, CLOSE_CONDITION),
        (3_000, 4_000, CLOSE_SESSION_END),
    ]
    assert result.qualifying_count == 0


def test_stable_spread_closes_the_instance():
    result = _run(
        "W4_CRUISE",
        {
            GEAR: _input([0], [4.0]),
            SPEED: _input([0, 1_000, 2_000], [50.0, 50.0, 55.0]),
            THROTTLE: _input([0], [10.0]),
        },
    )

    assert len(result.instances) == 1
    assert result.instances[0].close_reason == CLOSE_STABLE
    assert result.instances[0].close_ms == 2_000
    assert not result.instances[0].qualifies


def test_max_duration_closes_and_condition_rearms():
    result = _run(
        "W3_WARM_IDLE",
        {
            COOLANT: _input([0], [85.0]),
            SPEED: _input([0], [0.0]),
            RPM: _input(
                list(range(0, 63_001, 1_000)),
                [1_200.0] * 61 + [0.0, 1_200.0, 1_200.0],
            ),
        },
    )

    assert [(i.open_ms, i.close_ms, i.close_reason, i.qualifies) for i in result.instances] == [
        (0, 60_000, CLOSE_MAX_DURATION, True),
        (62_000, 63_000, CLOSE_SESSION_END, False),
    ]
    assert result.qualifying_count == 1


def test_stale_condition_input_is_not_met():
    result = _run(
        "W3_WARM_IDLE",
        {
            COOLANT: _input([0], [85.0], stale_after_ms=200),
            SPEED: _input([0, 300], [0.0, 0.0]),
            RPM: _input([0, 300], [1_200.0, 1_200.0]),
        },
    )

    assert len(result.instances) == 1
    assert result.instances[0].close_reason == CLOSE_CONDITION
    assert result.instances[0].close_ms == 300


def test_engine_start_needs_two_consecutive_valid_samples():
    broken = _run(
        "W2_CRANK",
        {RPM: _input([0, 100, 200, 300], [0.0, None, 1_200.0, 1_300.0])},
        RPM,
    )
    assert broken.instances == ()

    valid = _run("W2_CRANK", {RPM: _input([0, 100, 200], [0.0, 1_200.0, 1_300.0])}, RPM)
    assert len(valid.instances) == 1
    assert valid.instances[0].open_ms == 100
    assert valid.instances[0].close_reason == CLOSE_SESSION_END
    assert valid.instances[0].qualifies


def test_engine_start_unknown_closes():
    result = _run(
        "W2_CRANK",
        {RPM: _input([0, 100, 500], [0.0, 1_200.0, None])},
        RPM,
    )

    assert len(result.instances) == 1
    assert result.instances[0].close_reason == CLOSE_UNKNOWN
    assert result.instances[0].close_ms == 500


def test_engine_stop_session_end_qualifies():
    result = _run(
        "W5_POST_STOP",
        {RPM: _input([0, 1_000, 61_000], [1_500.0, 0.0, 0.0])},
        RPM,
    )

    assert len(result.instances) == 1
    assert result.instances[0].open_ms == 1_000
    assert result.instances[0].close_ms == 61_000
    assert result.instances[0].close_reason == CLOSE_SESSION_END
    assert result.instances[0].qualifies


def test_min_duration_boundary():
    below = _run("W1_KEY_ON", {RPM: _input([0, 1_999], [0.0, 0.0])})
    at = _run("W1_KEY_ON", {RPM: _input([0, 2_000], [0.0, 0.0])})

    assert not below.instances[0].qualifies
    assert at.instances[0].qualifies


def test_patching_a_window_threshold_changes_the_result(monkeypatch):
    series = {RPM: _input([0, 2_500], [0.0, 0.0])}
    assert _run("W1_KEY_ON", series).qualifying_count == 1

    monkeypatch.setitem(WINDOWS["W1_KEY_ON"], "min_duration_ms", 3_000)

    assert _run("W1_KEY_ON", series).qualifying_count == 0
