"""Per-ride summaries of the class-B derived signals (D-070 item 2, part 3).

Phase 0 has no ride segmentation: one stored session is one ride, so every summary covers
the whole session. Names, units, ranges, records and bins come from
vehicle_defs.vehicle_derived.SIGNALS (class "B"); this module only holds the arithmetic.

Summaries are evaluated AFTER the class-A evaluation (engine.evaluate), in dependency order
(ride_fuel_consumption reads ride_fuel_used and ride_distance). A summary function sees:
    - the valid rows of the class-A outputs it declares as derived.* inputs,
    - the scalar results of the class-B signals it declares as derived.* inputs,
    - the raw series of its did.* / sensor.* inputs (same InputSource as engine.evaluate).
It returns a scalar (value | None, flag | None) or, for a signal with defs `bins`, a Binned
(one value per cell, row-major, the first bins entry is the major axis). Time integrals reuse
formulas.hold_segments: a value is held between samples but never beyond the input's
freshness, so a gap adds nothing. The freshness limit of a did.* input of a class-B signal
(no stale_after_ms of its own) is the DID's own stale_after_ms (D-087); sensors use SENSORS.
Statuses are those of the engine; every value outside the defs `range` is null with the flag
"out_of_range" (each cell too). Only declared inputs may be read, and every declared input
must be read (D-095; tests check both). A summary whose declared input has no usable data
raises MissingInputs, reported as the engine's missing_inputs status.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from vehicle_server.defs import vehicle_cl250, vehicle_derived
from vehicle_server.derived.context import (
    FLAG_INSUFFICIENT_SAMPLES,
    FLAG_INVALID_INPUT,
    FLAG_OUT_OF_RANGE,
    Env,
    Feed,
    InputInvalid,
)
from vehicle_server.derived.engine import (
    COMPUTED,
    ERROR,
    MISSING_INPUTS,
    NOT_IMPLEMENTED,
    NOT_READY,
    Evaluation,
    InputSource,
    SignalOutput,
    dependency_order,
)
from vehicle_server.derived.formulas import FLAG_STANDSTILL, Outcome, hold_segments
from vehicle_server.gps_fix import usable_fix_values

# Unit conversions (not vehicle data).
SECONDS_PER_HOUR = 3600.0
M_PER_KM = 1000.0
LITRES_PER_100KM_FACTOR = 100.0

# --- input names (all of them declared by the defs; tests check this) -------------------
_TRIP_DISTANCE = "derived.trip_distance"
_TRIP_TIME = "derived.trip_time"
_IDLE_TIME = "derived.idle_time"
_STOP_EVENT = "derived.stop_event"
_COOLANT_RATE = "derived.coolant_rate"
_FUEL_USED = "derived.ride_fuel_used"
_DISTANCE = "derived.ride_distance"
_FUEL_RATE = "did.ENGINE_FUEL_RATE"
_LTFT = "did.LONG_TERM_FUEL_TRIM_B1"
_STFT = "did.SHORT_TERM_FUEL_TRIM_B1"
_COOLANT = "did.COOLANT_TEMP"
_INTAKE = "did.INTAKE_AIR_TEMP"
_GPS_SPEED = "sensor.gps_ground_speed"
_GPS_FIX = "sensor.gps_fix_type"
_BUS_OFF = "sensor.can_bus_off_count"
_RTT_MAX = "sensor.tester_rtt_max"
_STEP_GAP_MAX = "sensor.tester_step_gap_max"


@dataclass
class Binned:
    """Result of a binned summary: seconds (or values) per cell, row-major, and the seconds
    held outside every cell."""

    cells: list[Outcome]
    outside_s: float = 0.0


@dataclass(frozen=True)
class Summary:
    fn: Callable[[SummaryEnv], Outcome | Binned]


SUMMARIES: dict[str, Summary] = {}


def _summary(name: str) -> Callable[[Callable], Callable]:
    def register(fn: Callable) -> Callable:
        SUMMARIES[name] = Summary(fn)
        return fn

    return register


def _input_not_implemented(name: str) -> str:
    return f"input not implemented: derived.{name}"


# Reason of every class-B signal that has no summary yet (the README lists them too).
_COAST_DOWN_REASON = "definition: coast-down speed band not defined (D-103 gave none)"
NOT_IMPLEMENTED_REASONS: dict[str, str] = {
    "ride_fan_on_temperature": (
        "definition: the rule needs a steady-RPM condition; ENGINE_SPEED is not a declared input "
        "and no steady-RPM threshold exists"
    ),
    "ride_final_drive_ratio": (
        "definition: turning the measured ratio into a final-drive ratio needs the nominal gear "
        "ratios (const.gear_ratio_N are not declared inputs)"
    ),
    "ride_tester_nrc78_count": (
        "definition: the rule sums per rttDid, rttDid is not a declared input"
    ),
    "coast_down_cda": _COAST_DOWN_REASON,
    "coast_down_rolling_resistance": _COAST_DOWN_REASON,
}

# D-079 health-window signals (defs v0.16.0): each reports one window instance (`window` key in
# the defs), which needs the window state machine; the reference has none yet.
_WINDOW_REASON = "definition: D-079 window instance (window state machine not implemented)"
NOT_IMPLEMENTED_REASONS.update(
    {
        name: _WINDOW_REASON
        for name, spec in vehicle_derived.SIGNALS.items()
        if spec["class"] == "B"
        and spec.get("window")
        and spec["ready_phase0"]
        and not spec.get("pending_inputs")
    }
)


class MissingInputs(Exception):
    """A declared input has no usable data here; reported as the existing missing_inputs."""

    def __init__(self, inputs: list[str]):
        super().__init__(", ".join(inputs))
        self.inputs = inputs


class SummaryEnv(Env):
    """The declared inputs of one class-B signal: did/sensor feeds, the valid rows of its
    class-A inputs and the scalar results of its class-B inputs."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        feeds: dict[str, Feed],
        outputs: Mapping[str, SignalOutput],
        results: Mapping[str, float | None],
    ):
        super().__init__(list(spec["inputs"]), feeds, spec.get("window_ms"))
        self.bins: list[Mapping[str, Any]] = list(spec.get("bins") or [])
        self._outputs = outputs
        self._results = results

    def _declared(self, name: str) -> str:
        if name not in self.inputs:
            raise KeyError(f"{name!r} is not a declared input (declared: {self.inputs})")
        return name.removeprefix("derived.")

    def valid(self, name: str) -> list[float]:
        """Valid values (time order) of a declared class-A input."""
        key = self._declared(name)
        return [v for v in self._outputs[key].values if v is not None]

    def result(self, name: str) -> float | None:
        """Scalar result of a declared class-B input (None = invalid)."""
        return self._results[self._declared(name)]

    def rows(self, name: str) -> tuple[list[int], list[float | None]]:
        """(times, values) of a declared class-A input, invalid rows (None) included."""
        output = self._outputs[self._declared(name)]
        return output.times, output.values

    def samples(self, name: str) -> list[float]:
        """All samples (time order) of a declared did.* / sensor.* input."""
        return list(self.feed(name).values)  # type: ignore[arg-type]


# --- helpers ------------------------------------------------------------------------------


def _last_valid(env: SummaryEnv, name: str) -> Outcome:
    values = env.valid(name)
    if not values:
        raise InputInvalid(FLAG_INVALID_INPUT)
    return values[-1], None


def _integral(env: SummaryEnv, name: str, per_second: Callable[[float], float]) -> Outcome:
    """Time integral of per_second(value) over the held segments of one input."""
    _, held, spans = hold_segments(env, [name])
    return sum(per_second(v[0]) * s for v, s in zip(held, spans, strict=True)), None


def _axis_index(value: float, axis: Mapping[str, Any]) -> int | None:
    """Index of the uniform bin [min, max) of one axis containing value, None if outside."""
    low, high, count = axis["min"], axis["max"], axis["count"]
    if not low <= value < high:
        return None
    return min(count - 1, int((value - low) / (high - low) * count))


def bin_edges(axis: Mapping[str, Any], index: int) -> tuple[float, float]:
    """Lower and upper edge of bin `index` of an axis (the last upper edge is `max`)."""
    low, high, count = axis["min"], axis["max"], axis["count"]
    width = (high - low) / count
    return low + index * width, high if index == count - 1 else low + (index + 1) * width


def cell_axes(bins: list[Mapping[str, Any]], cell: int) -> list[int]:
    """Per-axis bin indices of a row-major cell index (the first axis is the major one)."""
    indices: list[int] = []
    for axis in reversed(bins):
        cell, i = divmod(cell, axis["count"])
        indices.append(i)
    return indices[::-1]


def cell_count(bins: list[Mapping[str, Any]]) -> int:
    return math.prod(axis["count"] for axis in bins)


def _time_in_bins(env: SummaryEnv) -> Binned:
    """Seconds per cell of the defs `bins`: the held values of every bins input (hold rule
    of hold_segments) are attributed to their cell; a held value outside an axis range is
    counted in no cell, its seconds go to outside_s."""
    names = [axis["input"] for axis in env.bins]
    _, held, spans = hold_segments(env, names)
    seconds = [0.0] * cell_count(env.bins)
    outside = 0.0
    for values, span in zip(held, spans, strict=True):
        cell = 0
        for value, axis in zip(values, env.bins, strict=True):
            i = _axis_index(value, axis)
            if i is None:
                cell = -1
                break
            cell = cell * axis["count"] + i
        if cell < 0:
            outside += span
        else:
            seconds[cell] += span
    return Binned([(s, None) for s in seconds], outside)


# --- distance, time, stops ------------------------------------------------------------------


@_summary("ride_distance")
def ride_distance(env: SummaryEnv) -> Outcome:
    """Last valid derived.trip_distance, km."""
    return _last_valid(env, _TRIP_DISTANCE)


@_summary("ride_time")
def ride_time(env: SummaryEnv) -> Outcome:
    """Last valid derived.trip_time, s."""
    return _last_valid(env, _TRIP_TIME)


@_summary("ride_idle_time")
def ride_idle_time(env: SummaryEnv) -> Outcome:
    """Last valid derived.idle_time, s."""
    return _last_valid(env, _IDLE_TIME)


@_summary("ride_stop_count")
def ride_stop_count(env: SummaryEnv) -> Outcome:
    """Number of valid derived.stop_event rows."""
    return float(len(env.valid(_STOP_EVENT))), None


# --- fuel -----------------------------------------------------------------------------------


@_summary("ride_fuel_used")
def ride_fuel_used(env: SummaryEnv) -> Outcome:
    """Time integral of the fuel rate, L (L/h divided by 3600 s/h per second)."""
    return _integral(env, _FUEL_RATE, lambda rate: rate / SECONDS_PER_HOUR)


@_summary("ride_fuel_consumption")
def ride_fuel_consumption(env: SummaryEnv) -> Outcome:
    """ride_fuel_used / ride_distance * 100, L/100km; distance <= 0 is "standstill"."""
    used, distance = env.result(_FUEL_USED), env.result(_DISTANCE)
    if used is None or distance is None:
        raise InputInvalid(FLAG_INVALID_INPUT)
    if distance <= 0:
        raise InputInvalid(FLAG_STANDSTILL)
    return used / distance * LITRES_PER_100KM_FACTOR, None


@_summary("ride_ltft_mean")
def ride_ltft_mean(env: SummaryEnv) -> Outcome:
    """Arithmetic mean of the valid long-term fuel trim samples (bank 1), %."""
    samples = env.samples(_LTFT)
    return sum(samples) / len(samples), None


@_summary("ride_stft_std")
def ride_stft_std(env: SummaryEnv) -> Outcome:
    """Population standard deviation of the valid short-term fuel trim samples (bank 1), %;
    fewer than two samples are "insufficient_samples"."""
    samples = env.samples(_STFT)
    if len(samples) < 2:
        raise InputInvalid(FLAG_INSUFFICIENT_SAMPLES)
    mean = sum(samples) / len(samples)
    return math.sqrt(sum((v - mean) ** 2 for v in samples) / len(samples)), None


# --- time in cells (operating point, temperature) ---------------------------------------------


@_summary("operating_point_time")
def operating_point_time(env: SummaryEnv) -> Binned:
    """Seconds per (ENGINE_SPEED, ENGINE_LOAD) cell: uniform bins [min, max) per axis from the
    defs, cell = speed bin * load count + load bin; the time of a held value is attributed to
    its cell, a value outside an axis range to no cell (outside_s)."""
    return _time_in_bins(env)


@_summary("ride_coolant_time_at_temp")
def ride_coolant_time_at_temp(env: SummaryEnv) -> Binned:
    """Seconds per coolant temperature bin [min, max) of the defs; outside values in outside_s."""
    return _time_in_bins(env)


@_summary("ride_oil_time_at_temp")
def ride_oil_time_at_temp(env: SummaryEnv) -> Binned:
    """Seconds per engine oil temperature bin [min, max) of the defs; outside values in
    outside_s."""
    return _time_in_bins(env)


# --- thermal ----------------------------------------------------------------------------------


@_summary("ride_coolant_rate_max")
def ride_coolant_rate_max(env: SummaryEnv) -> Outcome:
    """Largest valid derived.coolant_rate, degC/min."""
    values = env.valid(_COOLANT_RATE)
    if not values:
        raise InputInvalid(FLAG_INVALID_INPUT)
    return max(values), None


@_summary("ride_start_coolant_temp")
def ride_start_coolant_temp(env: SummaryEnv) -> Outcome:
    """First valid coolant temperature sample, degC."""
    return env.samples(_COOLANT)[0], None


@_summary("ride_end_coolant_temp")
def ride_end_coolant_temp(env: SummaryEnv) -> Outcome:
    """Last valid coolant temperature sample, degC."""
    return env.samples(_COOLANT)[-1], None


@_summary("ride_start_intake_temp")
def ride_start_intake_temp(env: SummaryEnv) -> Outcome:
    """First valid intake air temperature sample, degC."""
    return env.samples(_INTAKE)[0], None


# --- GNSS and bus health ------------------------------------------------------------------------


@_summary("ride_distance_gps")
def ride_distance_gps(env: SummaryEnv) -> Outcome:
    """Time integral of the GNSS ground speed, km (m/s divided by 1000 m/km per second),
    counted only while the latest fix-type sample is usable and fresh by the /track rule.
    A gap without a usable fix adds nothing; speed only, never a position. A ride with no
    usable fix at all is missing_inputs, not 0.0."""
    fix = env.feed(_GPS_FIX)
    usable = usable_fix_values()
    gated = [value if value in usable else None for value in fix.values]
    if all(value is None for value in gated):
        raise MissingInputs([_GPS_FIX])
    # An unusable fix type reads as an invalid sample, so hold_segments drops the stretch;
    # the shared env stays untouched.
    feeds = {_GPS_SPEED: env.feed(_GPS_SPEED), _GPS_FIX: Feed(fix.times, gated, fix.stale_after_ms)}
    _, held, spans = hold_segments(Env(list(feeds), feeds, None), list(feeds))
    return sum(speed / M_PER_KM * span for (speed, _), span in zip(held, spans, strict=True)), None


@_summary("ride_bus_off_count")
def ride_bus_off_count(env: SummaryEnv) -> Outcome:
    """Sum of the increments of the cumulative bus-off counter between consecutive samples
    (the first sample is the baseline). A decrease means the node restarted: the new value
    itself is then the increment."""
    samples = env.samples(_BUS_OFF)
    total = 0.0
    for previous, current in zip(samples, samples[1:], strict=False):
        total += current - previous if current >= previous else current
    return total, None


@_summary("ride_tester_rtt_max")
def ride_tester_rtt_max(env: SummaryEnv) -> Outcome:
    """Largest valid tester round-trip maximum sample, ms."""
    return max(env.samples(_RTT_MAX)), None


@_summary("ride_tester_step_gap_max")
def ride_tester_step_gap_max(env: SummaryEnv) -> Outcome:
    """Largest valid tester step-gap maximum sample, ms."""
    return max(env.samples(_STEP_GAP_MAX)), None


# --- evaluation ---------------------------------------------------------------------------------


@dataclass
class SummaryRow:
    """One scalar (cell None) or one cell of a binned summary."""

    signal: str
    unit: str
    cell: int | None
    edges: list[tuple[float, float]]  # per bins axis (empty for scalars)
    value: float | None
    flag: str | None


@dataclass
class SummaryEvaluation:
    rows: list[SummaryRow] = field(default_factory=list)
    coverage: dict[str, dict[str, Any]] = field(default_factory=dict)  # class B, defs order


def _checked(value: float | None, flag: str | None, rng: list[float]) -> Outcome:
    if value is not None and not (math.isfinite(value) and rng[0] <= value <= rng[1]):
        return None, FLAG_OUT_OF_RANGE
    return value, flag


def _feeds_of(
    spec: Mapping[str, Any],
    inputs: InputSource,
    outputs: Mapping[str, SignalOutput],
    results: Mapping[str, float | None],
    specs: Mapping[str, Mapping[str, Any]],
    sensors: Mapping[str, Mapping[str, Any]],
    did_limits: Mapping[str, int],
) -> tuple[dict[str, Feed], list[str]]:
    """The feeds of the did/sensor inputs and the list of inputs that are missing here."""
    feeds: dict[str, Feed] = {}
    missing: list[str] = []
    for inp in spec["inputs"]:
        kind, _, key = inp.partition(".")
        if kind == "const":
            continue
        if kind == "derived":
            present = key in (outputs if specs[key]["class"] == "A" else results)
            if not present:
                missing.append(inp)
            continue
        series = inputs.series(inp)
        if series is None:
            missing.append(inp)
            continue
        stale = sensors[key]["stale_after_ms"] if kind == "sensor" else did_limits[key]
        feeds[inp] = Feed(series.times, list(series.values), stale)
    return feeds, missing


def _summarize_one(
    name: str,
    spec: Mapping[str, Any],
    summary: Summary,
    env: SummaryEnv,
    result: SummaryEvaluation,
    results: dict[str, float | None],
) -> dict[str, Any]:
    unit, rng = spec["unit"], list(spec["range"])
    try:
        outcome: Outcome | Binned = summary.fn(env)
    except InputInvalid as exc:
        outcome = (None, exc.flag)
    if isinstance(outcome, Binned):
        cells = [_checked(value, flag, rng) for value, flag in outcome.cells]
        for index, (value, flag) in enumerate(cells):
            axes = cell_axes(env.bins, index)
            edges = [bin_edges(axis, i) for axis, i in zip(env.bins, axes, strict=True)]
            result.rows.append(SummaryRow(name, unit, index, edges, value, flag))
        valid = [v for v, _ in cells if v is not None]
        return {
            "status": COMPUTED,
            "unit": unit,
            "cells": len(cells),
            "nonzero_cells": sum(v > 0 for v in valid),
            "total_s": sum(valid),
            "outside_s": outcome.outside_s,
        }
    value, flag = _checked(outcome[0], outcome[1], rng)
    result.rows.append(SummaryRow(name, unit, None, [], value, flag))
    results[name] = value
    return {"status": COMPUTED, "value": value, "unit": unit, "flag": flag}


def evaluate(
    inputs: InputSource,
    evaluation: Evaluation,
    specs: Mapping[str, Mapping[str, Any]] | None = None,
    summaries: Mapping[str, Summary] | None = None,
    sensors: Mapping[str, Mapping[str, Any]] | None = None,
    did_limits: Mapping[str, int] | None = None,
) -> SummaryEvaluation:
    """Evaluates every class-B signal of `specs` (default: the pinned defs) over `inputs`
    and the class-A `evaluation`."""
    specs = specs if specs is not None else vehicle_derived.SIGNALS
    summaries = summaries if summaries is not None else SUMMARIES
    sensors = sensors if sensors is not None else vehicle_derived.SENSORS
    did_limits = did_limits if did_limits is not None else vehicle_cl250.DID_STALE_AFTER_MS

    outputs = {o.name: o for o in evaluation.outputs}
    results: dict[str, float | None] = {}  # computed class-B scalars
    result = SummaryEvaluation()
    entries: dict[str, dict[str, Any]] = {}
    for name in dependency_order(specs):
        spec = specs[name]
        if spec["class"] != "B":
            continue
        pending = list(spec.get("pending_inputs") or [])
        if not spec.get("ready_phase0", False) or pending:
            reason = "ready_phase0" if not spec.get("ready_phase0", False) else "pending_inputs"
            entries[name] = {"status": NOT_READY, "reason": reason, "pending_inputs": pending}
        elif name not in summaries:
            reason = NOT_IMPLEMENTED_REASONS.get(name, "no summary")
            entries[name] = {"status": NOT_IMPLEMENTED, "reason": reason}
        else:
            try:
                feeds, missing = _feeds_of(
                    spec, inputs, outputs, results, specs, sensors, did_limits
                )
                if missing:
                    entries[name] = {"status": MISSING_INPUTS, "missing": missing}
                else:
                    env = SummaryEnv(spec, feeds, outputs, results)
                    entries[name] = _summarize_one(
                        name, spec, summaries[name], env, result, results
                    )
            except MissingInputs as exc:
                entries[name] = {"status": MISSING_INPUTS, "missing": list(exc.inputs)}
            except Exception as exc:  # one failing summary must not stop the others
                entries[name] = {"status": ERROR, "message": f"{type(exc).__name__}: {exc}"}
    result.coverage = {name: entries[name] for name in specs if name in entries}  # defs order
    return result


# The D-103 summaries register themselves in SUMMARIES; the module needs the registry above.
from vehicle_server.derived import d103_summaries  # noqa: E402, F401
