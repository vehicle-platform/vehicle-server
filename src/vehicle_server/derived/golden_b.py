"""Class-B (per-ride) golden vectors for the rt-core host tests (D-070 item 2, C3 part 2a).

Extends golden.py: the same scenarios and inputs, but the expected outputs are the per-ride
summaries of summary.py (the Python reference of the class-B signals) instead of the class-A
rows. Files (all committed into vehicle-rt-core under tests/host/vectors/, banner and
content-sha256 as class A):

    derived_vectors_b.h      types: ride marker, class-B output row (the flag enum, the input
                             row and the tolerance row come from derived_vectors.h)
    derived_<scenario>_b.h   rides table, expected closed values per ride, tolerances

Ride semantics (D-087 item 2): on rt-core a ride is one D-075 stop epoch. The vector start
opens the ride and its end closes it, so a single-ride scenario has one row in the rides table
(start 0, end = the scenario duration) and no epoch pseudo-input. The inputs of those four
scenarios are the ones of derived_<scenario>.h (the class-A header); the two_rides scenario is
self-contained (derived_two_rides_b.h also carries its inputs) because no class-A header
exists for it. The reference evaluates every ride on its own (ride-local time, fresh state);
that is the contract rt-core must meet when it opens a new ride.

Two-ride marker: the rides table {ride, start_ms, end_ms} IS the epoch-marker table. At
start_ms of ride N the test drives rt-core's epoch counter (a new D-075 epoch: the ECU was
absent, then confirmed again); at end_ms it ends the ride (ECU absent) and reads the closed
values of that ride. Between end_ms of one ride and start_ms of the next there are no inputs
at all (an ECU-absent stretch) and the gap is longer than ecu_absent_timeout_ms of the CL250
tester policy (asserted; exported as DERIVED_<SCENARIO>_ECU_ABSENT_TIMEOUT_MS).

Selection rule: class B, a summary in SUMMARIES, ready_rt_core true, scalar (no `bins`), no
sensor.* input (never a GPS name), every derived input produced by a selected signal. The
expected value of a signal in a ride is the closed value at end_ms. `element` is 0 for every
row (reserved so multi-element signals can be added without a type change).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vehicle_server.defs import vehicle_cl250, vehicle_derived
from vehicle_server.derived import golden, summary
from vehicle_server.derived.engine import COMPUTED, ERROR
from vehicle_server.derived.engine import evaluate as evaluate_class_a
from vehicle_server.derived.golden import Scenario
from vehicle_server.derived.inputs import Series

TYPES_HEADER_B = "derived_vectors_b.h"

SELECTION_RULE_B = (
    "class B, a summary in SUMMARIES, ready_rt_core true, scalar (no bins), no sensor.* input "
    "(never a GPS name), derived inputs from selected signals; value = closed value at ride end."
)

# Class-B signals that are skipped, with the reason (listed in the README and the report).
SKIP_BINNED = "binned (multi-element, 2a is scalar-only)"
SKIP_PER_GEAR = "per gear (multi-element, 2a is scalar-only)"
SKIP_SENSOR = "sensor input (vectors carry DIDs only)"
SKIP_NO_SUMMARY = "no summary in the reference"
SKIP_NOT_READY = "not ready_rt_core / ready_phase0 / pending inputs"
SKIP_HEALTH_WINDOW = "D-079 health window (excluded until rt-core implements the windows)"

# Gap between two rides of the two-ride scenario (no inputs at all), ms.
TWO_RIDES_GAP_MS = 8_000


@dataclass(frozen=True)
class Ride:
    """One ride of a scenario: the DID profile it replays (ride-local time) and where the
    ride sits on the vector timeline."""

    source: Scenario
    start_ms: int

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.source.duration_ms


@dataclass(frozen=True)
class BScenario:
    name: str
    description: str
    rides: tuple[Ride, ...]
    own_inputs: bool  # True: the _b header carries the inputs (no class-A header)


def _two_rides() -> BScenario:
    first = golden.SCENARIOS_BY_NAME["accel_gears"]
    second = golden.SCENARIOS_BY_NAME["edge_values"]
    start = first.duration_ms + TWO_RIDES_GAP_MS
    return BScenario(
        "two_rides",
        "Two rides (two D-075 stop epochs): accel_gears (ride 0, 0..30 s), an ECU-absent gap "
        f"of {TWO_RIDES_GAP_MS / 1000:g} s without any input, then edge_values (ride 1, "
        f"{start / 1000:g}..{(start + second.duration_ms) / 1000:g} s). Both closed values "
        "are expected; ride 1 starts from a fresh state.",
        (Ride(first, 0), Ride(second, start)),
        own_inputs=True,
    )


def b_scenarios() -> tuple[BScenario, ...]:
    singles = tuple(
        BScenario(s.name, s.description, (Ride(s, 0),), own_inputs=False) for s in golden.SCENARIOS
    )
    return (*singles, _two_rides())


# --- selection -----------------------------------------------------------------------------


def classify() -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    """(selected class-B specs in defs order, skipped class-B name -> reason)."""
    selected: dict[str, Mapping[str, Any]] = {}
    skipped: dict[str, str] = {}
    class_a = golden.select_signals()
    for name, spec in vehicle_derived.SIGNALS.items():
        if spec["class"] != "B":
            continue
        inputs = spec["inputs"]
        if spec.get("window"):
            skipped[name] = SKIP_HEALTH_WINDOW
        elif name not in summary.SUMMARIES:
            skipped[name] = SKIP_NO_SUMMARY
        elif (
            not spec.get("ready_rt_core", False)
            or not spec.get("ready_phase0", False)
            or spec.get("pending_inputs")
        ) and not any(i.startswith("sensor.") for i in inputs):
            skipped[name] = SKIP_NOT_READY
        elif any(i.startswith("sensor.") for i in inputs):
            skipped[name] = SKIP_SENSOR
        elif not spec.get("ready_rt_core", False):
            skipped[name] = SKIP_NOT_READY
        elif spec.get("bins"):
            skipped[name] = SKIP_BINNED
        elif spec.get("per_gear"):
            skipped[name] = SKIP_PER_GEAR
        else:
            selected[name] = spec
    for name, spec in selected.items():
        for inp in spec["inputs"]:
            assert "gps" not in inp.lower(), f"GPS input {inp!r} in selected signal {name}"
            assert not inp.startswith("sensor."), f"sensor input {inp!r} in signal {name}"
            if inp.startswith("derived."):
                dep = inp.removeprefix("derived.")
                assert dep in class_a or dep in selected, f"{name}: input {inp!r} not selected"
    return selected, skipped


# --- evaluation ----------------------------------------------------------------------------


@dataclass
class RideRow:
    ride: int
    signal: str
    valid: bool
    flag: str | None
    value: float


def evaluate_ride(
    rows: list[Any],
    class_a: Mapping[str, Mapping[str, Any]],
    class_b: Mapping[str, Mapping[str, Any]],
    ride: int,
) -> list[RideRow]:
    """Class-A then class-B evaluation of one ride from its (ride-local) input rows."""
    per_did: dict[str, tuple[list[int], list[float]]] = {}
    for t, _, v, did in rows:
        times, values = per_did.setdefault(did, ([], []))
        times.append(t)
        values.append(v)
    inputs = golden._Inputs({f"did.{d}": Series(t, v) for d, (t, v) in per_did.items()})
    evaluation = evaluate_class_a(inputs, specs=class_a)
    for name, entry in evaluation.coverage.items():
        if entry["status"] == ERROR:
            raise RuntimeError(f"ride {ride}: {name} failed: {entry['message']}")
    specs = {**class_a, **class_b}
    result = summary.evaluate(inputs, evaluation, specs=specs)
    out: list[RideRow] = []
    for name in class_b:
        entry = result.coverage[name]
        if entry["status"] != COMPUTED:
            raise RuntimeError(f"ride {ride}: {name} is {entry['status']}: {entry}")
    for row in result.rows:
        if row.signal not in class_b:
            continue
        if row.value is None:
            if row.flag is None:
                raise RuntimeError(f"ride {ride}: {row.signal} invalid without a flag")
            out.append(RideRow(ride, row.signal, False, row.flag, 0.0))
        else:
            out.append(RideRow(ride, row.signal, True, row.flag, golden.to_float32(row.value)))
    return out


def ride_inputs(
    ride: Ride, dids: list[str], did_idx: Mapping[str, int]
) -> list[tuple[int, int, float, str]]:
    """The ride's input rows in ride-local time (what a standalone run of it sees)."""
    return golden.input_rows(ride.source, dids, did_idx)


# --- C rendering ---------------------------------------------------------------------------


def types_header_b() -> str:
    return """\
/* Shared types of the class-B (per-ride) golden vectors for the derived-signal host tests
 * (D-070 item 2, C3 part 2a). Needs derived_vectors.h (input row, flag enum, tolerance row).
 * A ride is one D-075 stop epoch (D-087 item 2): the rides table is the epoch-marker table.
 * At start_ms of ride N the test opens a new epoch; at end_ms it ends the ride (ECU absent)
 * and compares the closed values of that ride. Between rides there are no inputs.
 * signal is VEHICLE_CL250_DERIVED_<NAME>; element is the element index of a multi-element
 * signal (always 0 now: only scalar signals are in the vectors). An invalid row has
 * valid = false, a flag and value 0.0f. abs_tol is derived_vector_tol_t (resolution / 2). */

#ifndef DERIVED_VECTORS_B_H
#define DERIVED_VECTORS_B_H

#include "derived_vectors.h"

typedef struct {
    uint16_t ride;
    uint32_t start_ms;
    uint32_t end_ms;
} derived_vector_ride_t;

typedef struct {
    uint16_t ride;
    uint16_t signal;
    uint16_t element;
    bool valid;
    uint8_t flag;
    float value;
} derived_vector_b_output_t;

#endif /* DERIVED_VECTORS_B_H */
"""


def scenario_header_b(
    sc: BScenario,
    inputs: list[tuple[int, int, float, str]],
    outputs: list[RideRow],
    selected: Mapping[str, Mapping[str, Any]],
    did_idx: Mapping[str, int],
    sig_idx: Mapping[str, int],
) -> str:
    n, upper = sc.name, sc.name.upper()
    ordered = sorted(outputs, key=lambda r: (r.ride, sig_idx[r.signal]))
    present = sorted({r.signal for r in outputs}, key=lambda s: sig_idx[s])
    where = (
        "Inputs: below (ascending time, ties in DID table order, none between rides)."
        if sc.own_inputs
        else f"Inputs: those of derived_{n}.h (one ride, 0..{sc.rides[0].end_ms} ms)."
    )
    lines = [
        f"/* Scenario {n} (class B): {sc.description}",
        f" * {where}",
        f" * Outputs: the closed value of every selected class-B signal per ride at end_ms. "
        f"Selection rule: {SELECTION_RULE_B}",
        " * Tolerance: resolution / 2 of the defs per signal present. Deterministic, no GPS, "
        "no real ride. */",
        "",
        f"#ifndef DERIVED_{upper}_B_H",
        f"#define DERIVED_{upper}_B_H",
        "",
        '#include "derived_vectors_b.h"',
        "",
        f"#define DERIVED_{upper}_RIDE_COUNT ({len(sc.rides)}u)",
        f"#define DERIVED_{upper}_B_OUTPUT_COUNT ({len(ordered)}u)",
        f"#define DERIVED_{upper}_B_TOL_COUNT ({len(present)}u)",
        f"#define DERIVED_{upper}_ECU_ABSENT_TIMEOUT_MS ({vehicle_cl250.ECU_ABSENT_TIMEOUT_MS}u)",
    ]
    if sc.own_inputs:
        lines.append(f"#define DERIVED_{upper}_B_INPUT_COUNT ({len(inputs)}u)")
    lines += ["", f"static const derived_vector_ride_t derived_{n}_rides[] = {{"]
    for i, ride in enumerate(sc.rides):
        lines.append(f"    {{ {i}u, {ride.start_ms}u, {ride.end_ms}u }},")
    lines += ["};", ""]
    if sc.own_inputs:
        lines.append(f"static const derived_vector_input_t derived_{n}_b_inputs[] = {{")
        for t, i, v, did in inputs:
            assert did_idx[did] == i
            lines.append(f"    {{ {t}u, VEHICLE_CL250_IDX_{did}, {golden.c_float(v)} }},")
        lines += ["};", ""]
    lines.append(f"static const derived_vector_b_output_t derived_{n}_b_outputs[] = {{")
    known = set(golden.FLAG_ORDER)
    for r in ordered:
        if r.flag is not None and r.flag not in known:
            raise RuntimeError(f"{n}: {r.signal} carries unknown flag {r.flag!r}")
        flag = golden.flag_constant(r.flag) if r.flag else "DERIVED_VECTOR_FLAG_NONE"
        macro = f"VEHICLE_CL250_DERIVED_{r.signal.upper()}"
        valid = "true" if r.valid else "false"
        lines.append(
            f"    {{ {r.ride}u, {macro}, 0u, {valid}, {flag}, {golden.c_float(r.value)} }},"
        )
    lines += ["};", "", f"static const derived_vector_tol_t derived_{n}_b_tol[] = {{"]
    for s in present:
        tol = selected[s]["resolution"] / 2.0
        lines.append(f"    {{ VEHICLE_CL250_DERIVED_{s.upper()}, {golden.c_float(tol)} }},")
    lines += ["};", "", f"#endif /* DERIVED_{upper}_B_H */", ""]
    return "\n".join(lines)


# --- generation ----------------------------------------------------------------------------


@dataclass
class ClassBGeneration:
    files: list[golden.GeneratedFile]
    selected: list[str]
    skipped: dict[str, str]
    valid_rows: dict[str, int]  # signal -> valid (ride, scenario) results
    ride_values: dict[str, list[list[RideRow]]]  # scenario -> per ride rows


def generate_b() -> ClassBGeneration:
    selected_b, skipped = classify()
    class_a = golden.select_signals()
    dids = golden.needed_dids(class_a)
    did_idx, sig_idx = golden.did_indices(), golden.derived_indices()
    for spec in selected_b.values():
        for inp in spec["inputs"]:
            if inp.startswith("did."):
                assert inp.removeprefix("did.") in dids, f"{inp} is not fed by the vectors"
    missing = [s for s in selected_b if s not in sig_idx]
    if missing:
        raise RuntimeError(f"not in the gen/ C enums: {missing}")

    def body_file(name: str, body: str) -> golden.GeneratedFile:
        return golden.GeneratedFile(
            name, golden.with_banner(body), hashlib.sha256(body.encode()).hexdigest()
        )

    files = [body_file(TYPES_HEADER_B, types_header_b())]
    valid_rows: dict[str, int] = dict.fromkeys(selected_b, 0)
    ride_values: dict[str, list[list[RideRow]]] = {}
    for sc in b_scenarios():
        all_rows: list[RideRow] = []
        abs_inputs: list[tuple[int, int, float, str]] = []
        per_ride: list[list[RideRow]] = []
        for index, ride in enumerate(sc.rides):
            local = ride_inputs(ride, dids, did_idx)
            rows = evaluate_ride(local, class_a, selected_b, index)
            per_ride.append(rows)
            all_rows += rows
            for r in rows:
                valid_rows[r.signal] += r.valid
            abs_inputs += [(t + ride.start_ms, i, v, d) for t, i, v, d in local]
        for a, b in zip(sc.rides, sc.rides[1:], strict=False):
            assert b.start_ms - a.end_ms > vehicle_cl250.ECU_ABSENT_TIMEOUT_MS, sc.name
        abs_inputs.sort(key=lambda r: r[0])  # stable: ties keep DID table order
        ride_values[sc.name] = per_ride
        body = scenario_header_b(sc, abs_inputs, all_rows, selected_b, did_idx, sig_idx)
        files.append(body_file(f"derived_{sc.name}_b.h", body))
    never = [s for s, n in valid_rows.items() if n == 0]
    if never:
        raise RuntimeError(f"class-B signals never valid in any ride: {never}")
    return ClassBGeneration(files, list(selected_b), skipped, valid_rows, ride_values)


def report_b(gen: ClassBGeneration) -> str:
    lines = [f"class B: {len(gen.selected)} signals computed (valid results: see below)"]
    lines += [f"  {s:<26} valid results={gen.valid_rows[s]}" for s in gen.selected]
    lines.append(f"class B skipped ({len(gen.skipped)}):")
    reasons: dict[str, list[str]] = {}
    for name, why in gen.skipped.items():
        reasons.setdefault(why, []).append(name)
    lines += [f"  {why}: {len(names)}" for why, names in reasons.items()]
    return "\n".join(lines)
