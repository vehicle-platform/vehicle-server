"""D-079 health-window state machine over time-stamped samples (reference only).

The window definitions (conditions, stable bands, durations and `opens_on`) come from
vehicle_derived.WINDOWS; this module only applies them. It is pure: no I/O, no clock, no
session state. A window is evaluated on the merged sample times of its condition, stable and
trigger inputs. At every time each input has its latest valid sample when that sample is not
older than the input's stale_after_ms; otherwise it is unknown and a condition with an unknown
input is not met (D-079: never a held value).

One instance is kept as (open_ms, close_ms, close_reason, qualifies). The result also carries
the first qualifying instance and the number of qualifying instances, because the class-B
summaries report one value over the first instance and one count per window. Re-arm follows
D-079: a `condition` window may open again only after its conditions have not been met at
least once, while the engine transition windows re-arm on their next transition.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CLOSE_CONDITION = "condition"
CLOSE_STABLE = "stable"
CLOSE_MAX_DURATION = "max_duration"
CLOSE_ENGINE_ZERO = "engine_zero"
CLOSE_UNKNOWN = "unknown"
CLOSE_SESSION_END = "session_end"

_TRANSITION_OPENERS = ("engine_start", "engine_stop")


@dataclass(frozen=True)
class WindowInput:
    """Samples of one window input (times ascending, value None = invalid sample)."""

    times: tuple[int, ...]
    values: tuple[float | None, ...]
    stale_after_ms: int


@dataclass(frozen=True)
class WindowInstance:
    """One closed window instance; duration = close_ms - open_ms."""

    open_ms: int
    close_ms: int
    close_reason: str
    qualifies: bool

    @property
    def duration_ms(self) -> int:
        return self.close_ms - self.open_ms


@dataclass(frozen=True)
class WindowResult:
    """All closed instances of one window, its first qualifying one and the qualifying count."""

    instances: tuple[WindowInstance, ...] = ()
    first_qualifying: WindowInstance | None = None
    qualifying_count: int = 0
    inputs: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> WindowResult:
        return cls()


def input_names(spec: Mapping[str, Any], trigger_input: str | None = None) -> tuple[str, ...]:
    """The window inputs to feed, in a stable order (conditions, stable, trigger)."""
    names: list[str] = []
    for condition in spec.get("conditions") or ():
        if condition["input"] not in names:
            names.append(condition["input"])
    for stable in spec.get("stable") or ():
        if stable["input"] not in names:
            names.append(stable["input"])
    if trigger_input is not None and trigger_input not in names:
        names.append(trigger_input)
    return tuple(names)


def _latest(sample: WindowInput, t: int) -> float | None:
    index = bisect_right(sample.times, t) - 1
    if index < 0 or t - sample.times[index] > sample.stale_after_ms:
        return None
    return sample.values[index]


def _sample_at(sample: WindowInput, t: int) -> tuple[bool, float | None]:
    index = bisect_left(sample.times, t)
    if index == len(sample.times) or sample.times[index] != t:
        return False, None
    return True, sample.values[index]


def _compare(value: float, op: str, expected: float) -> bool:
    if op == "eq":
        return value == expected
    if op == "ne":
        return value != expected
    if op == "gt":
        return value > expected
    if op == "ge":
        return value >= expected
    if op == "lt":
        return value < expected
    if op == "le":
        return value <= expected
    raise ValueError(f"unknown window condition op: {op!r}")


def _conditions_hold(spec: Mapping[str, Any], inputs: Mapping[str, WindowInput], t: int) -> bool:
    for condition in spec.get("conditions") or ():
        value = _latest(inputs[condition["input"]], t)
        if value is None or not _compare(value, condition["op"], condition["value"]):
            return False
    return True


def _stable_available(spec: Mapping[str, Any], inputs: Mapping[str, WindowInput], t: int) -> bool:
    return all(
        _latest(inputs[stable["input"]], t) is not None for stable in spec.get("stable") or ()
    )


def _stable_bounds(
    spec: Mapping[str, Any], inputs: Mapping[str, WindowInput], t: int
) -> dict[str, tuple[float, float]]:
    return {
        stable["input"]: (_latest(inputs[stable["input"]], t),) * 2  # type: ignore[misc]
        for stable in spec.get("stable") or ()
    }


def _can_open(spec: Mapping[str, Any], inputs: Mapping[str, WindowInput], t: int) -> bool:
    return _conditions_hold(spec, inputs, t) and _stable_available(spec, inputs, t)


def _stable_close_reason(
    spec: Mapping[str, Any],
    inputs: Mapping[str, WindowInput],
    t: int,
    bounds: dict[str, tuple[float, float]],
) -> str | None:
    for stable in spec.get("stable") or ():
        name, high = stable["input"], stable["max_spread"]
        value = _latest(inputs[name], t)
        if value is None:
            return CLOSE_CONDITION
        low, previous_high = bounds[name]
        low, previous_high = min(low, value), max(previous_high, value)
        bounds[name] = (low, previous_high)
        if previous_high - low > high:
            return CLOSE_STABLE
    return None


def _trigger_transition(
    opener: str, previous_valid: bool, previous: float | None, current: float | None
) -> bool:
    if not previous_valid or current is None:
        return False
    if opener == "engine_start":
        return previous == 0.0 and current > 0.0
    return previous is not None and previous > 0.0 and current == 0.0


def evaluate_window(
    spec: Mapping[str, Any],
    inputs: Mapping[str, WindowInput],
    trigger_input: str | None = None,
) -> WindowResult:
    """Evaluates one WINDOWS entry over its inputs (see the module docstring)."""
    opener = spec["opens_on"]
    names = input_names(spec, trigger_input)
    if opener in _TRANSITION_OPENERS and trigger_input is None:
        raise ValueError(f"{opener} window needs a trigger input")
    missing = [name for name in names if name not in inputs]
    if missing:
        raise KeyError(f"missing window inputs: {missing}")

    times = sorted({t for name in names for t in inputs[name].times})
    if not times:
        return WindowResult.empty()

    trigger = inputs[trigger_input] if trigger_input is not None else None
    instances: list[WindowInstance] = []
    open_ms: int | None = None
    bounds: dict[str, tuple[float, float]] = {}
    can_open = True
    key_on_seen = False
    previous_valid = False
    previous_value: float | None = None
    previous_time: int | None = None

    def close(t: int, reason: str) -> None:
        nonlocal open_ms, bounds
        if open_ms is None:
            return
        duration = t - open_ms
        qualifies = reason == CLOSE_MAX_DURATION or duration >= spec["min_duration_ms"]
        instances.append(WindowInstance(open_ms, t, reason, qualifies))
        open_ms = None
        bounds = {}

    def open(t: int) -> None:
        nonlocal open_ms, bounds
        open_ms = t
        bounds = _stable_bounds(spec, inputs, t)

    for t in times:
        transitioned = False
        if opener in _TRANSITION_OPENERS and trigger is not None:
            is_sample, current = _sample_at(trigger, t)
            if is_sample:
                if current is None:
                    previous_valid = False
                else:
                    consecutive = (
                        previous_time is not None and t - previous_time <= trigger.stale_after_ms
                    )
                    if consecutive:
                        transitioned = _trigger_transition(
                            opener, previous_valid, previous_value, current
                        )
                    previous_valid = True
                    previous_value = current
                    previous_time = t

        was_open = open_ms is not None
        if was_open:
            if t - open_ms >= spec["max_duration_ms"]:
                close(t, CLOSE_MAX_DURATION)
            elif opener in ("condition", "key_on", "engine_stop"):
                if not _conditions_hold(spec, inputs, t):
                    close(t, CLOSE_CONDITION)
                else:
                    reason = _stable_close_reason(spec, inputs, t, bounds)
                    if reason is not None:
                        close(t, reason)
            elif opener == "engine_start" and trigger is not None:
                value = _latest(trigger, t)
                if value is None:
                    close(t, CLOSE_UNKNOWN)
                elif value == 0.0:
                    close(t, CLOSE_ENGINE_ZERO)

        if open_ms is None:
            if opener == "condition":
                if was_open:
                    can_open = False
                met = _can_open(spec, inputs, t)
                if not met:
                    can_open = True
                elif can_open:
                    open(t)
            elif opener == "key_on":
                if not key_on_seen:
                    key_on_seen = True
                    if _can_open(spec, inputs, t):
                        open(t)
            elif transitioned and _can_open(spec, inputs, t):
                open(t)

    if open_ms is not None:
        close(times[-1], CLOSE_SESSION_END)

    first = next((instance for instance in instances if instance.qualifies), None)
    return WindowResult(
        tuple(instances),
        first,
        sum(instance.qualifies for instance in instances),
        names,
    )
