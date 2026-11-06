"""Seeded properties of the provisional D-079 window state machine.

The qualifying rule under test is the README's: an instance qualifies when its duration is at
least `min_duration_ms`; a `max_duration_ms` close also qualifies, and no instance lasts longer
than `max_duration_ms`. The inputs are synthetic piecewise-constant series that cross each
window's own condition / stable thresholds, with random gaps around each input's staleness.
No state-machine code is changed here.
"""

from __future__ import annotations

import random
from bisect import bisect_right

import pytest

from vehicle_server.defs import vehicle_derived
from vehicle_server.derived import summary
from vehicle_server.derived.health_windows import (
    CLOSE_CONDITION,
    CLOSE_ENGINE_ZERO,
    CLOSE_MAX_DURATION,
    CLOSE_SESSION_END,
    CLOSE_STABLE,
    CLOSE_UNKNOWN,
    WindowInput,
    WindowResult,
    evaluate_window,
    input_names,
)

WINDOWS = vehicle_derived.WINDOWS
CLOSE_REASONS = {
    CLOSE_CONDITION,
    CLOSE_STABLE,
    CLOSE_MAX_DURATION,
    CLOSE_ENGINE_ZERO,
    CLOSE_UNKNOWN,
    CLOSE_SESSION_END,
}
SEEDS = range(200)


def _candidates(spec: dict, name: str, trigger: str | None) -> list[float | None]:
    values: list[float | None] = []
    for condition in spec.get("conditions") or ():
        if condition["input"] != name:
            continue
        value = float(condition["value"])
        if condition["op"] in ("eq", "ne"):
            values.extend((value,) * 4)
        elif condition["op"] in ("gt", "ge"):
            values.extend((value, value + 1.0, value + 5.0, value + 20.0))
        else:
            values.extend((value, value - 1.0, value - 5.0, value - 20.0))
        values.append(value * 0.5)
    if any(stable["input"] == name for stable in spec.get("stable") or ()):
        values.extend((40.0, 50.0, 60.0, 70.0))
    if name == trigger:
        values.extend((0.0, 1_000.0, 1_200.0, 1_500.0, 1_800.0))
    if name == "derived.gear_estimate":
        values.extend((3.0, 4.0, 5.0, 6.0))
    if name == "did.VEHICLE_SPEED":
        values.extend((0.0, 30.0, 50.0, 60.0))
    if name == "did.THROTTLE_POS":
        values.extend((5.0, 10.0, 20.0))
    if name == "did.COOLANT_TEMP":
        values.extend((70.0, 80.0, 85.0, 120.0))
    values.extend((None, None))
    return values


def _timeline(rng: random.Random) -> int:
    return rng.choice((60_000, 120_000, 180_000))


def _input(
    rng: random.Random, spec: dict, name: str, trigger: str | None, session_len: int
) -> WindowInput:
    stale_after_ms = rng.choice((1_000, 5_000, 20_000, 120_000))
    choices = _candidates(spec, name, trigger)
    current = rng.choice(choices)
    times: list[int] = []
    values: list[float | None] = []
    t = 0
    while t <= session_len:
        stop = min(session_len + 1, t + rng.choice((2_000, 5_000, 10_000, 30_000, 60_000, 120_000)))
        while t < stop:
            times.append(t)
            values.append(current)
            t += rng.choice((500, 1_000, 2_500, 5_000))
        if rng.random() < 0.15:
            t += rng.choice((stale_after_ms + 1, stale_after_ms * 2))
        if rng.random() < 0.7:
            current = rng.choice(choices)
    return WindowInput(tuple(times), tuple(values), stale_after_ms)


def _run(
    key: str, rng: random.Random, seed: int | None = None
) -> tuple[WindowResult, dict[str, WindowInput], str | None]:
    spec = WINDOWS[key]
    trigger = summary._window_trigger(spec)  # the production trigger (D-069 item 3)
    names = input_names(spec, trigger)
    session_len = _timeline(rng)
    if seed is not None and trigger is not None:
        forced = _forced_trigger(key, trigger, session_len, seed)
        if forced is not None:
            session_len, inputs = forced
            return evaluate_window(spec, inputs, trigger), inputs, trigger
    inputs = {name: _input(rng, spec, name, trigger, session_len) for name in names}
    return evaluate_window(spec, inputs, trigger), inputs, trigger


def _forced_trigger(
    key: str, trigger: str, session_len: int, seed: int
) -> tuple[int, dict[str, WindowInput]] | None:
    """Deterministic trigger series for the close reasons random seeds do not always hit."""
    if key == "W2_CRANK":
        if seed == 0:  # engine_start -> exactly 0 while open
            return session_len, {
                trigger: WindowInput(
                    (0, 1_000, 2_000, 3_000, 4_000),
                    (0.0, 1_200.0, 1_200.0, 0.0, 0.0),
                    120_000,
                )
            }
        if seed == 1:  # open near the end, no transition closes it
            return 60_000, {
                trigger: WindowInput(
                    (0, 58_000, 59_000, 60_000),
                    (0.0, 0.0, 1_200.0, 1_200.0),
                    120_000,
                )
            }
        if seed == 2:  # the trigger goes stale while the instance is open
            return session_len, {
                trigger: WindowInput((0, 1_000, 2_000), (0.0, 1_200.0, 1_200.0), 500)
            }
    if key == "W5_POST_STOP" and seed == 3:
        long_len = int(WINDOWS[key]["max_duration_ms"]) + 60_000
        return long_len, {trigger: WindowInput((0, 1_000, long_len), (1_200.0, 0.0, 0.0), long_len)}
    return None


def _value_at(series: WindowInput, t: int) -> float | None:
    index = bisect_right(series.times, t) - 1
    if index < 0 or t - series.times[index] > series.stale_after_ms:
        return None
    return series.values[index]


def _reachable_reasons(spec: dict) -> set[str]:
    reasons = {CLOSE_MAX_DURATION, CLOSE_SESSION_END}
    if spec["opens_on"] == "engine_start":
        reasons.update((CLOSE_ENGINE_ZERO, CLOSE_UNKNOWN))
    else:
        reasons.add(CLOSE_CONDITION)
        if spec.get("stable"):
            reasons.add(CLOSE_STABLE)
    return reasons


@pytest.mark.parametrize("key", list(WINDOWS))
def test_engine_zero_and_reason_coverage(key: str):
    spec = WINDOWS[key]
    trigger_name = summary._window_trigger(spec)
    seen: set[str] = set()
    for seed in SEEDS:
        result, inputs, trigger = _run(key, random.Random(seed), seed)
        seen.update(instance.close_reason for instance in result.instances)
        if trigger_name is None:
            continue
        trigger_input = inputs[trigger_name]
        for instance in result.instances:
            if instance.close_reason == CLOSE_ENGINE_ZERO:
                zero = next(
                    t
                    for t, value in zip(trigger_input.times, trigger_input.values, strict=True)
                    if t >= instance.open_ms and value == 0.0
                )
                assert instance.close_ms == zero
            if (
                spec["opens_on"] != "engine_stop"
                and _value_at(trigger_input, instance.open_ms) == 0.0
            ):
                pytest.fail(f"{key} opened while the trigger was 0")
    assert _reachable_reasons(spec) <= seen, (key, seen)


def _assert_properties(key: str, result: WindowResult) -> None:
    spec = WINDOWS[key]
    previous_end: int | None = None
    for instance in result.instances:
        assert instance.open_ms <= instance.close_ms
        assert instance.duration_ms == instance.close_ms - instance.open_ms
        assert instance.close_reason in CLOSE_REASONS
        if previous_end is not None:
            assert previous_end <= instance.open_ms
        previous_end = instance.close_ms

    qualifying = [instance for instance in result.instances if instance.qualifies]
    assert result.qualifying_count == len(qualifying)
    if qualifying:
        assert result.first_qualifying == qualifying[0]
    else:
        assert result.first_qualifying is None
    for instance in qualifying:
        assert spec["min_duration_ms"] <= instance.duration_ms <= spec["max_duration_ms"]


@pytest.mark.parametrize("key", list(WINDOWS))
def test_seeded_series_are_not_vacuous(key: str):
    results = [_run(key, random.Random(seed), seed)[0] for seed in SEEDS]

    assert any(result.instances for result in results)
    assert any(result.qualifying_count for result in results)


@pytest.mark.parametrize("key", list(WINDOWS))
def test_seeded_window_properties(key: str):
    for seed in SEEDS:
        result, inputs, trigger = _run(key, random.Random(seed), seed)
        _assert_properties(key, result)

        again = evaluate_window(WINDOWS[key], inputs, trigger)
        assert again == result


@pytest.mark.parametrize("key", list(WINDOWS))
def test_time_shift_moves_every_instance(key: str):
    shift = 1_000_000
    for seed in SEEDS:
        result, inputs, trigger = _run(key, random.Random(seed), seed)
        shifted = {
            name: WindowInput(
                tuple(t + shift for t in series.times), series.values, series.stale_after_ms
            )
            for name, series in inputs.items()
        }
        moved = evaluate_window(WINDOWS[key], shifted, trigger)

        assert [(i.open_ms, i.close_ms, i.close_reason, i.qualifies) for i in moved.instances] == [
            (i.open_ms + shift, i.close_ms + shift, i.close_reason, i.qualifies)
            for i in result.instances
        ]
        assert moved.qualifying_count == result.qualifying_count
        assert (moved.first_qualifying is None) == (result.first_qualifying is None)
        if moved.first_qualifying is not None and result.first_qualifying is not None:
            assert moved.first_qualifying.open_ms == result.first_qualifying.open_ms + shift
            assert moved.first_qualifying.close_ms == result.first_qualifying.close_ms + shift


@pytest.mark.parametrize("key", list(WINDOWS))
def test_no_inputs_or_all_stale_never_qualifies(key: str):
    spec = WINDOWS[key]
    trigger = summary._window_trigger(spec)
    names = input_names(spec, trigger)

    empty = {name: WindowInput((), (), 120_000) for name in names}
    assert evaluate_window(spec, empty, trigger).qualifying_count == 0

    stale = {name: WindowInput((0,), (None,), 0) for name in names}
    assert evaluate_window(spec, stale, trigger).qualifying_count == 0
