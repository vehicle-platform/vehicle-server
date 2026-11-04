"""Synthetic inputs for the derived-signal tests (no real ride data, no position)."""

from __future__ import annotations

from vehicle_server.defs import vehicle_cl250, vehicle_derived
from vehicle_server.derived.context import At, Env, Feed
from vehicle_server.derived.formulas import FORMULAS
from vehicle_server.derived.inputs import Series

Samples = tuple[list[int], list[float | None]]


class FakeInputs:
    """An InputSource over in-memory series: {"did.VEHICLE_SPEED": (times, values), ...}."""

    def __init__(self, series: dict[str, Samples]):
        self._series = {
            name: Series(list(times), list(values)) for name, (times, values) in series.items()
        }

    def series(self, input_name: str) -> Series | None:
        return self._series.get(input_name)


def ramp(start: int, stop: int, step: int, fn) -> Samples:
    times = list(range(start, stop + 1, step))
    return times, [fn(t) for t in times]


def const_series(start: int, stop: int, step: int, value: float) -> Samples:
    return ramp(start, stop, step, lambda _t: value)


def stale_limit(signal: str, input_name: str) -> int:
    """The stale limit the engine uses for an input of `signal`."""
    kind, _, key = input_name.partition(".")
    if kind == "sensor":
        return vehicle_derived.SENSORS[key]["stale_after_ms"]
    if kind == "derived":
        return vehicle_derived.SIGNALS[key]["stale_after_ms"]
    return vehicle_cl250.DID_STALE_AFTER_MS[key]  # D-070 / D-087: the DID's own limit


def env_for(signal: str, feeds: dict[str, Samples]) -> Env:
    """The Env of a signal over hand-made feeds (stale limits as the engine uses them)."""
    spec = vehicle_derived.SIGNALS[signal]
    built = {
        name: Feed(list(times), list(values), stale_limit(signal, name))
        for name, (times, values) in feeds.items()
    }
    return Env(list(spec["inputs"]), built, spec.get("window_ms"))


def point_value(signal: str, feeds: dict[str, Samples], t: int) -> float:
    """Evaluates one point formula at time t over hand-made feeds."""
    return FORMULAS[signal].fn(At(env_for(signal, feeds), t))


def patch_const(monkeypatch, name: str, value: float) -> None:
    """Makes the constant accessor return `value` for const.<name> (the defs stay untouched)."""
    from vehicle_server.defs import vehicle_constants

    old = vehicle_constants.CONSTANTS[name]
    monkeypatch.setitem(vehicle_constants.CONSTANTS, name, (value, *old[1:]))


def outcome(signal: str, feeds: dict[str, Samples], t: int) -> tuple[float | None, str | None]:
    """(value, None) or (None, flag) of one point formula at t, as the engine records it."""
    from vehicle_server.derived.context import InputInvalid

    try:
        return point_value(signal, feeds, t), None
    except InputInvalid as exc:
        return None, exc.flag


def event_rows(
    signal: str, feeds: dict[str, Samples]
) -> list[tuple[int, float | None, str | None]]:
    """The rows of an event formula over hand-made feeds."""
    return FORMULAS[signal].fn(env_for(signal, feeds))


def series_outcomes(signal: str, feeds: dict[str, Samples], grid: list[int]):
    """The outcomes of a series formula over hand-made feeds."""
    return FORMULAS[signal].fn(env_for(signal, feeds), grid)
