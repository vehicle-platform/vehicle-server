"""Evaluates the derived signals of a session in dependency order (D-070 item 2).

For every signal of vehicle_defs.vehicle_derived.SIGNALS the engine decides a status:
    computed         a formula exists and its inputs have a common span in this session
    not_implemented  class B/C/D, an event signal, or no formula in formulas.py yet
    not_ready        ready_phase0 is False or pending_inputs is set (never evaluated)
    missing_inputs   an input is absent here (no file / column / samples, or a derived input
                     that was not computed), or the inputs share no time span
    error            the evaluation raised
An event signal (formula kind "event") has no grid: its rows are at the event times.
A computed signal is evaluated on a grid of period_ms (from the defs) from the latest
first-sample time to the earliest last-sample time of its inputs. Each output row is
valid or null with a flag: a stale input (age above that input's stale_after_ms; for a
did.* input the signal's own, for sensors SENSORS', for derived inputs the producer's),
an invalid derived input, too few samples in a window, or a result outside the defs
`range` ("out_of_range"). A held value is never presented as fresh.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from vehicle_server.defs import vehicle_cl250, vehicle_constants, vehicle_derived
from vehicle_server.derived.context import (
    FLAG_OUT_OF_RANGE,
    At,
    Env,
    Feed,
    InputInvalid,
)
from vehicle_server.derived.d103_common import NOT_IMPLEMENTED_REASONS_A
from vehicle_server.derived.formulas import FORMULAS, Formula
from vehicle_server.derived.inputs import Series

COMPUTED = "computed"
NOT_IMPLEMENTED = "not_implemented"
NOT_READY = "not_ready"
MISSING_INPUTS = "missing_inputs"
ERROR = "error"

FLAG_MATH_ERROR = "math_error"


class InputSource(Protocol):
    def series(self, input_name: str) -> Series | None: ...


@dataclass
class SignalOutput:
    """One computed signal on its grid; value None = invalid, with the flag saying why."""

    name: str
    unit: str
    times: list[int]
    values: list[float | None]
    flags: list[str | None]

    @property
    def valid_count(self) -> int:
        return sum(v is not None for v in self.values)


@dataclass
class Evaluation:
    outputs: list[SignalOutput] = field(default_factory=list)  # in evaluation order
    coverage: dict[str, dict[str, Any]] = field(default_factory=dict)


def dependency_order(specs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Signal names so that every derived.* input comes before its consumer (defs order is
    kept where there is no dependency). Raises ValueError on a cycle or an unknown input."""
    order: list[str] = []
    state: dict[str, int] = {}  # 1 = visiting, 2 = done

    def visit(name: str, trail: tuple[str, ...]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise ValueError(f"dependency cycle: {' -> '.join((*trail, name))}")
        state[name] = 1
        for inp in specs[name]["inputs"]:
            if inp.startswith("derived."):
                dep = inp.removeprefix("derived.")
                if dep not in specs:
                    raise ValueError(f"{name}: unknown derived input {inp!r}")
                visit(dep, (*trail, name))
        state[name] = 2
        order.append(name)

    for name in specs:
        visit(name, ())
    return order


def _provisional_constants(spec: Mapping[str, Any]) -> list[str]:
    names = [i.removeprefix("const.") for i in spec["inputs"] if i.startswith("const.")]
    known = vehicle_constants.CONSTANTS
    return [n for n in names if n in known and known[n][2] == "provisional"]


def _pre_status(
    spec: Mapping[str, Any], name: str, formulas: Mapping[str, Formula]
) -> dict[str, Any] | None:
    """The coverage entry of a signal that is not evaluated at all, else None."""
    entry: dict[str, Any] = {"class": spec["class"]}
    if spec["class"] != "A":
        return {**entry, "status": NOT_IMPLEMENTED, "reason": f"class {spec['class']}"}
    pending = list(spec.get("pending_inputs") or [])
    if not spec.get("ready_phase0", False) or pending:
        reason = "ready_phase0" if not spec.get("ready_phase0", False) else "pending_inputs"
        return {**entry, "status": NOT_READY, "reason": reason, "pending_inputs": pending}
    if name not in formulas:
        default = "event" if spec.get("event") else "no formula"
        reason = NOT_IMPLEMENTED_REASONS_A.get(name, default)
        return {**entry, "status": NOT_IMPLEMENTED, "reason": reason}
    return None


def _flag_counts(flags: list[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for flag in flags:
        if flag:
            counts[flag] = counts.get(flag, 0) + 1
    return counts


def _evaluate_one(
    name: str,
    spec: Mapping[str, Any],
    formula: Formula,
    inputs: InputSource,
    done: Mapping[str, SignalOutput],
    specs: Mapping[str, Mapping[str, Any]],
    sensors: Mapping[str, Mapping[str, Any]],
    did_limits: Mapping[str, int],
) -> tuple[SignalOutput | None, dict[str, Any]]:
    feeds: dict[str, Feed] = {}
    missing: list[str] = []
    for inp in spec["inputs"]:
        kind, _, key = inp.partition(".")
        if kind == "const":
            continue
        if kind == "derived":
            dep = done.get(key)
            if dep is None:
                missing.append(inp)
            else:
                feeds[inp] = Feed(dep.times, dep.values, specs[key]["stale_after_ms"])
            continue
        series = inputs.series(inp)
        if series is None:
            missing.append(inp)
            continue
        if kind == "sensor":
            stale = sensors[key]["stale_after_ms"]
        else:  # D-070 item 2 / D-087: a DID input (event or not) is judged by its own limit
            stale = did_limits[key]
        feeds[inp] = Feed(series.times, list(series.values), stale)
    if missing:
        return None, {"status": MISSING_INPUTS, "missing": missing}
    if not feeds:
        return None, {"status": MISSING_INPUTS, "missing": [], "reason": "no series inputs"}

    env = Env(list(spec["inputs"]), feeds, spec.get("window_ms"))
    is_event = formula.kind == "event"
    if is_event:  # no grid: one row per event, at the event time
        events = formula.fn(env)
        grid = [t for t, _, _ in events]
        outcomes = [(value, flag) for _, value, flag in events]
    else:
        start = max(f.times[0] for f in feeds.values())
        end = min(f.times[-1] for f in feeds.values())
        if start > end:
            return None, {"status": MISSING_INPUTS, "missing": [], "reason": "no common time span"}
        grid = list(range(start, end + 1, spec["period_ms"]))
        if formula.kind == "series":
            outcomes = formula.fn(env, grid)
        else:
            outcomes = []
            for t in grid:
                try:
                    outcomes.append((formula.fn(At(env, t)), None))
                except InputInvalid as exc:
                    outcomes.append((None, exc.flag))
                except ArithmeticError:
                    outcomes.append((None, FLAG_MATH_ERROR))

    low, high = spec["range"]
    values: list[float | None] = []
    flags: list[str | None] = []
    for value, flag in outcomes:
        if value is not None and not (math.isfinite(value) and low <= value <= high):
            value, flag = None, FLAG_OUT_OF_RANGE
        values.append(value)
        flags.append(flag)
    output = SignalOutput(name, spec["unit"], grid, values, flags)
    entry: dict[str, Any] = {"status": COMPUTED, "unit": spec["unit"]}
    if not is_event:
        entry["period_ms"] = spec["period_ms"]
    entry |= {
        "rows": len(grid),
        "valid_rows": output.valid_count,
        "flag_counts": _flag_counts(flags),
    }
    return output, entry


def evaluate(
    inputs: InputSource,
    specs: Mapping[str, Mapping[str, Any]] | None = None,
    formulas: Mapping[str, Formula] | None = None,
    sensors: Mapping[str, Mapping[str, Any]] | None = None,
    did_limits: Mapping[str, int] | None = None,
) -> Evaluation:
    """Evaluates every signal of `specs` (default: the pinned defs) over `inputs`.

    `did_limits` is the DID table's stale_after_ms per DID name (default: the generated
    vehicle_cl250.DID_STALE_AFTER_MS), the age limit of every DID input (D-087: events and
    class B too).
    """
    specs = specs if specs is not None else vehicle_derived.SIGNALS
    formulas = formulas if formulas is not None else FORMULAS
    sensors = sensors if sensors is not None else vehicle_derived.SENSORS
    did_limits = did_limits if did_limits is not None else vehicle_cl250.DID_STALE_AFTER_MS

    result = Evaluation()
    done: dict[str, SignalOutput] = {}
    for name in dependency_order(specs):
        spec = specs[name]
        entry = _pre_status(spec, name, formulas)
        if entry is None:
            try:
                output, entry = _evaluate_one(
                    name, spec, formulas[name], inputs, done, specs, sensors, did_limits
                )
            except Exception as exc:  # one failing formula must not stop the others
                output, entry = None, {"status": ERROR, "message": f"{type(exc).__name__}: {exc}"}
            entry = {"class": spec["class"], **entry}
            provisional = _provisional_constants(spec)
            if provisional and entry["status"] == COMPUTED:
                entry["provisional_constants"] = provisional
            if output is not None:
                done[name] = output
                result.outputs.append(output)
        result.coverage[name] = entry
    result.coverage = {name: result.coverage[name] for name in specs}  # defs order
    return result
