"""Golden input/output vectors for the vehicle-rt-core host tests (D-070 item 2).

The Python reference (formulas.py, evaluated by engine.py) is the oracle of the C
implementation on rt-core. This module runs it over a few synthetic scenarios and writes
the inputs and the outputs as generated C headers, which are committed into vehicle-rt-core
under tests/host/vectors/ (rt-core never depends on vehicle-server; the host tests read no
files and parse no strings):

    derived_vectors.h        shared types: input / output / tolerance rows and the flag enum
    derived_<scenario>.h     one scenario: inputs, expected outputs, tolerances

Selection rule (the signals of the vectors): class A, a formula in FORMULAS, `ready_rt_core`
true in the defs and not an unsupported kind. Point and series signals have one row per
output grid time; stop_event (event kind) is included too, with one row per event at the
event time (value = duration in s). A signal with a `sensor.*` input is never selected (the
vectors carry DIDs only), and a GPS name is never an input of a selected signal (D-033,
invariant 7); both are asserted. The inputs are the DIDs of the selected signals sampled at
their own poll_period_ms (vehicle_defs.vehicle_catalog), nothing else; no position, no real ride.

Everything is deterministic: the scenarios are closed-form profiles plus a hash-based
jitter (no random module), so the same defs pin and formulas always give the same bytes.
Inputs are quantised to the DID's raw resolution and rounded to float32 before the reference
runs, so the reference sees exactly the values the C test feeds. Tolerance of an output:
resolution / 2 of the defs (abs_tol). The content-sha256 of a file covers everything after
its banner line, so it does not change with the server commit; tests/golden/MANIFEST.sha256
records them and tests/test_golden.py fails on drift.

    uv run vehicle-server golden --out <vehicle-rt-core>/tests/host/vectors
    python -m vehicle_server.derived.golden --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import struct
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vehicle_server.defs import defs_version, vehicle_catalog, vehicle_derived
from vehicle_server.derived import context, d103_common, d103_summaries, engine, formulas
from vehicle_server.derived.formulas import FORMULAS
from vehicle_server.derived.inputs import Series, constant

GENERATOR = "vehicle_server.derived.golden"
TYPES_HEADER = "derived_vectors.h"
MANIFEST_RELATIVE = Path("tests") / "golden" / "MANIFEST.sha256"
DEFS_C_RT_CORE = Path(vehicle_catalog.__file__).resolve().parents[2] / "c" / "rt_core"
REPO_ROOT = Path(__file__).resolve().parents[3]

SELECTION_RULE = (
    "class A, a formula in FORMULAS, ready_rt_core true, no sensor.* input (never a GPS name); "
    "point/series signals at every grid time, stop_event at its event times."
)

# DIDs fed to every scenario although no selected signal reads them (idle_warmup: "battery
# steady"); the host test must ignore a DID nobody consumes.
EXTRA_DIDS = ("BATTERY_VOLTAGE", "RUN_TIME_SINCE_START")

# The flags the vectors can carry, in enum order. Checked against the FLAG_* constants of
# context.py / formulas.py / engine.py: a flag added there fails the generator until it is
# listed here (the C enum must know every flag the reference can emit).
FLAG_ORDER = (
    "stale_input",
    "invalid_input",
    "insufficient_samples",
    "out_of_range",
    "math_error",
    "standstill",
    "straight_line",
    "no_consumption",
    "no_power",
    "engine_off",
    "not_moving",
    "no_gear",
    "not_applicable",
    "no_data",
    "no_step",
    "starts_warm",
    "never_warm",
    "no_steady_samples",
    "no_fan_step",
)


def flag_constant(flag: str) -> str:
    return f"DERIVED_VECTOR_FLAG_{flag.upper()}"


def used_flags() -> set[str]:
    """Every FLAG_* string constant of the reference modules."""
    found: set[str] = set()
    for module in (context, formulas, engine, d103_common, d103_summaries):
        for name, value in vars(module).items():
            if name.startswith("FLAG_") and isinstance(value, str):
                found.add(value)
    return found


# --- selection -------------------------------------------------------------------------------


def select_signals() -> dict[str, Mapping[str, Any]]:
    """The specs of the signals in the vectors (see SELECTION_RULE), defs order."""
    selected: dict[str, Mapping[str, Any]] = {}
    for name, spec in vehicle_derived.SIGNALS.items():
        if spec["class"] != "A" or name not in FORMULAS or not spec.get("ready_rt_core", False):
            continue
        if any(i.startswith("sensor.") for i in spec["inputs"]):
            continue
        selected[name] = spec
    for name, spec in selected.items():
        for inp in spec["inputs"]:
            assert "gps" not in inp.lower(), f"GPS input {inp!r} in selected signal {name}"
            assert not inp.startswith("sensor."), f"sensor input {inp!r} in signal {name}"
            if inp.startswith("derived."):
                assert inp.removeprefix("derived.") in selected, (
                    f"{name}: derived input {inp!r} is not selected"
                )
    return selected


def needed_dids(selected: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """DIDs read by the selected signals plus EXTRA_DIDS, in vehicle-catalog order."""
    names = {i.removeprefix("did.") for spec in selected.values() for i in spec["inputs"]} & set(
        vehicle_catalog.SIGNALS
    )
    unknown = {
        i.removeprefix("did.")
        for spec in selected.values()
        for i in spec["inputs"]
        if i.startswith("did.")
    } - set(vehicle_catalog.SIGNALS)
    assert not unknown, f"DIDs not in the vehicle catalog: {sorted(unknown)}"
    names |= set(EXTRA_DIDS)
    return [n for n in vehicle_catalog.SIGNALS if n in names]


# --- header symbol tables (the gen/ enums are the only source of indices) ---------------------


def _read_header(name: str) -> str:
    return (DEFS_C_RT_CORE / name).read_text()


def did_indices() -> dict[str, int]:
    text = _read_header("vehicle_cl250.h")
    return {m[1]: int(m[2]) for m in re.finditer(r"VEHICLE_CL250_IDX_(\w+) = (\d+),", text)}


def derived_indices() -> dict[str, int]:
    text = _read_header("vehicle_cl250_derived.h")
    return {
        m[1].lower(): int(m[2])
        for m in re.finditer(r"#define VEHICLE_CL250_DERIVED_(\w+) \((\d+)u\)", text)
        if m[1] != "COUNT"
    }


# --- input quantisation and float printing ---------------------------------------------------


def to_float32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def quantise(did: str, value: float) -> float:
    """The value the ECU would report: clamped to the DID range, on the raw grid, as float32."""
    s = vehicle_catalog.SIGNALS[did]
    step = s["factor_num"] / s["factor_den"]
    raw = round((min(max(value, s["min"]), s["max"]) - s["offset"]) / step)
    raw = min(
        max(raw, math.ceil((s["min"] - s["offset"]) / step)), int((s["max"] - s["offset"]) / step)
    )
    return to_float32(raw * step + s["offset"])


def c_float(x: float) -> str:
    """A C float literal that round-trips the float32 of x (%.9g plus f; never inf / nan)."""
    if not math.isfinite(x):
        raise ValueError(f"non-finite value {x!r} in a golden vector")
    x = to_float32(x) + 0.0  # -0.0 -> 0.0
    text = f"{x:.9g}"
    if not any(c in text for c in ".e"):
        text += ".0"
    return text + "f"


# --- scenarios -------------------------------------------------------------------------------

Pts = list[tuple[float, float]]
Jitter = Callable[[str], float]
Model = Callable[[float, Jitter], dict[str, float]]


def interp(points: Pts, t: float) -> float:
    """Piecewise-linear value at t seconds (held outside the points)."""
    if t <= points[0][0]:
        return points[0][1]
    for (t0, v0), (t1, v1) in zip(points, points[1:], strict=False):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return points[-1][1]


def rpm_per_kmh(gear: int) -> float:
    """Engine rpm per km/h in a gear, from the defs gear ratios, reductions and tyre radius."""
    wheel_rpm = 1.0 / 3.6 / (2.0 * math.pi * constant("rear_wheel_radius_dynamic_m")) * 60.0
    reduction = constant("primary_reduction_ratio") * constant("secondary_reduction_ratio")
    return wheel_rpm * reduction * constant(f"gear_ratio_{gear}")


def jitter_fn(scenario: str, t_ms: int) -> Jitter:
    """Deterministic jitter in [-1, 1) per (scenario, DID, time); no random module."""

    def jit(name: str) -> float:
        digest = hashlib.sha256(f"{scenario}:{name}:{t_ms}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**63 - 1.0

    return jit


@dataclass(frozen=True)
class Profile:
    """Closed-form description of a ride (times in s)."""

    speed: Pts  # km/h
    throttle: Pts  # accelerator pedal, %
    rpm_free: Pts  # rpm while no gear is engaged (idle, clutch slip)
    gears: list[tuple[float, int]] = field(default_factory=lambda: [(0.0, 0)])  # (from, gear)
    engine_on_s: float | None = None  # key-on without the engine until then (D-103 start)
    engine_off_s: float | None = None
    coolant: Pts = field(default_factory=lambda: [(0.0, 70.0), (30.0, 73.6)])
    oil: Pts = field(default_factory=lambda: [(0.0, 55.0), (30.0, 56.8)])
    baro_kpa: Pts = field(default_factory=lambda: [(0.0, 99.0)])
    iat: Pts = field(default_factory=lambda: [(0.0, 28.0), (30.0, 29.5)])
    # Hook for the edge cases: mutates the DID values of one instant.
    tweak: Callable[[float, dict[str, float]], None] | None = None


def make_model(p: Profile) -> Model:
    def model(t: float, jit: Jitter) -> dict[str, float]:
        speed = interp(p.speed, t)
        pedal = interp(p.throttle, t)
        gear = [g for start, g in p.gears if t >= start][-1]
        engine_on = (p.engine_on_s is None or t >= p.engine_on_s) and (
            p.engine_off_s is None or t < p.engine_off_s
        )
        if not engine_on:
            rpm, pedal = 0.0, 0.0
        elif gear:
            rpm = speed * rpm_per_kmh(gear) + 8.0 * jit("rpm")
        else:
            rpm = interp(p.rpm_free, t) + 12.0 * jit("rpm")
        load = rpm / 1000.0 * pedal / 100.0
        maf = (1.3 + 1.6 * load) * (1.0 + 0.03 * jit("maf")) if engine_on else 0.0
        enrich = 0.92 if pedal > 80.0 else 1.0
        fuel = maf * 3.6 / (enrich * 14.7 * 0.745) * (1.0 + 0.03 * jit("fuel"))
        throttle_a = 12.0 + 0.62 * pedal
        torque = (8.0 + 0.8 * pedal + 1.5 * jit("torque")) if engine_on else 0.0
        values = {
            "ENGINE_SPEED": rpm,
            "VEHICLE_SPEED": speed,
            "ACTUAL_TORQUE_PCT": torque,
            "DEMAND_TORQUE_PCT": torque + 2.0 + 0.04 * pedal if engine_on else 0.0,
            "REFERENCE_TORQUE": 23.0,
            "BAROMETRIC_PRESSURE": interp(p.baro_kpa, t),
            "INTAKE_AIR_TEMP": interp(p.iat, t),
            "COMMANDED_EQUIV_RATIO": enrich,
            "MAF_RATE": maf,
            "ENGINE_FUEL_RATE": fuel,
            "FUEL_LEVEL": 60.0,
            "SHORT_TERM_FUEL_TRIM_B1": 2.5 * jit("stft"),
            "LONG_TERM_FUEL_TRIM_B1": 2.0,
            "THROTTLE_POS": throttle_a,
            "ABSOLUTE_THROTTLE_POS_B": throttle_a + 0.4 + 0.2 * jit("thr_b"),
            "RELATIVE_THROTTLE_POS": (throttle_a - 12.0) / 88.0 * 100.0,
            "COMMANDED_THROTTLE": throttle_a + 0.5 * jit("thr_cmd"),
            "ACCEL_PEDAL_POS_D": 0.9 * pedal + 1.0,
            "RELATIVE_ACCEL_PEDAL_POS": pedal,
            "COOLANT_TEMP": interp(p.coolant, t),
            "ENGINE_OIL_TEMP": interp(p.oil, t),
            "BATTERY_VOLTAGE": 14.1 + 0.02 * jit("battery") if engine_on else 12.4,
            # D-103 inputs: load and manifold pressure follow the pedal, the ambient is steady.
            "ENGINE_LOAD": (15.0 + 0.8 * pedal + 1.0 * jit("load")) if engine_on else 0.0,
            "INTAKE_MAP": (30.0 + 0.65 * pedal + 0.5 * jit("map")) if engine_on else 99.0,
            "AMBIENT_AIR_TEMP": 24.0,
            "RUN_TIME_SINCE_START": max(0.0, t - (p.engine_on_s or 0.0)),
        }
        if p.tweak is not None:
            p.tweak(t, values)
        return values

    return model


# Samples of a DID that never arrive: {did: [(from_ms, to_ms)]} (to exclusive).
Gaps = dict[str, list[tuple[int, int]]]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    duration_ms: int
    model: Model
    gaps: Gaps = field(default_factory=dict)


# Launch and acceleration through gears 1..4. The shifts are 0.4 s ramps; the rpm follows the
# engaged gear, so gear_estimate resolves (clutch slip below 3.5 s gives out-of-range ratios).
_ACCEL_PROFILE = Profile(
    speed=[
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 6.0),
        (3.5, 17.0),
        (7.0, 32.0),
        (7.4, 32.5),
        (11.0, 49.0),
        (11.4, 49.5),
        (16.0, 67.0),
        (16.4, 67.5),
        (23.0, 88.0),
        (30.0, 95.0),
    ],  # fmt: skip
    throttle=[
        (0.0, 0.0),
        (1.0, 0.0),
        (1.5, 30.0),
        (3.5, 60.0),
        (7.0, 90.0),
        (7.4, 10.0),
        (7.7, 85.0),
        (11.0, 90.0),
        (11.4, 10.0),
        (11.7, 80.0),
        (16.0, 85.0),
        (16.4, 10.0),
        (16.7, 70.0),
        (23.0, 75.0),
        (26.0, 30.0),
        (30.0, 20.0),
    ],  # fmt: skip
    rpm_free=[(0.0, 1450.0), (1.0, 1450.0), (2.0, 3500.0), (3.5, 3680.0)],
    gears=[(0.0, 0), (3.5, 1), (7.2, 2), (11.2, 3), (16.2, 4)],
    baro_kpa=[(0.0, 99.0), (17.9, 99.0), (18.1, 98.0)],
)

_IDLE_PROFILE = Profile(
    speed=[(0.0, 0.0)],
    throttle=[(0.0, 0.0)],
    rpm_free=[(0.0, 1600.0), (20.0, 1450.0)],
    coolant=[(0.0, 62.0), (20.0, 65.0)],
    oil=[(0.0, 45.0), (20.0, 46.6)],
)


def _edge_tweak(t: float, v: dict[str, float]) -> None:
    if 6.0 <= t < 12.0:  # standing, no torque: no_power (bsfc), standstill (consumption)
        v["ACTUAL_TORQUE_PCT"] = v["DEMAND_TORQUE_PCT"] = 0.0
    if 12.0 <= t < 13.5:  # crawling at 1 km/h on 2.5 L/h: consumption out of range
        v["ENGINE_FUEL_RATE"] = 2.5
    if 15.5 <= t < 18.5:  # MAF spike: fuel_rate_maf out of range
        v["MAF_RATE"] = 50.0
    if 18.0 <= t < 22.0:  # implausible barometer: air_density, altitude out of range
        v["BAROMETRIC_PRESSURE"] = 40.0


_EDGE_PROFILE = Profile(
    speed=[
        (0.0, 25.0),
        (4.0, 25.0),
        (6.0, 0.0),
        (12.0, 0.0),
        (12.2, 1.0),
        (13.5, 1.0),
        (15.0, 12.0),
        (18.0, 30.0),
        (21.0, 30.0),
        (24.0, 0.0),
        (30.0, 0.0),
    ],  # fmt: skip
    throttle=[
        (0.0, 15.0),
        (4.0, 15.0),
        (4.5, 0.0),
        (12.0, 0.0),
        (12.2, 12.0),
        (13.5, 12.0),
        (15.0, 35.0),
        (18.0, 50.0),
        (21.0, 30.0),
        (21.5, 0.0),
        (30.0, 0.0),
    ],  # fmt: skip
    rpm_free=[(0.0, 1450.0)],
    gears=[(0.0, 2), (5.0, 0), (15.0, 2), (23.0, 0)],
    engine_off_s=24.5,
    tweak=_edge_tweak,
)


def _start_tweak(t: float, v: dict[str, float]) -> None:
    """Cold start, closed throttle at idle, a fuel top-up at rest (D-103 events)."""
    if 1.0 <= t < 1.4:  # cranking
        v["ENGINE_SPEED"] = 300.0
    elif 1.4 <= t < 1.6:
        v["ENGINE_SPEED"] = 700.0
    if 1.0 <= t < 1.9:  # the starter pulls the battery down
        v["BATTERY_VOLTAGE"] = 9.6
    if v["RELATIVE_ACCEL_PEDAL_POS"] == 0.0:  # closed throttle plate
        v["THROTTLE_POS"] = 1.0
    v["FUEL_LEVEL"] = 40.0 if t < 14.0 else 70.0
    if t >= 60.0:  # the cooling fan switches on at rest: a 0.3 V step in the battery voltage
        v["BATTERY_VOLTAGE"] -= 0.3


_START_PROFILE = Profile(
    speed=[
        (0.0, 0.0),
        (3.0, 0.0),
        (6.0, 30.0),
        (8.0, 30.0),
        (10.0, 0.0),
        (17.5, 0.0),
        (18.5, 20.0),
        (20.0, 20.0),
        (21.0, 0.0),
        (90.0, 0.0),
    ],  # fmt: skip
    throttle=[
        (0.0, 0.0),
        (3.0, 0.0),
        (3.5, 30.0),
        (8.0, 25.0),
        (8.5, 0.0),
        (17.5, 0.0),
        (18.0, 15.0),
        (20.0, 10.0),
        (20.5, 0.0),
        (90.0, 0.0),
    ],  # fmt: skip
    rpm_free=[(0.0, 1400.0), (90.0, 1400.0)],
    engine_on_s=1.0,
    coolant=[(0.0, 45.0), (30.0, 95.0)],
    tweak=_start_tweak,
)

SCENARIOS = (
    Scenario(
        "idle_warmup",
        "Engine at idle, vehicle standing, coolant and oil rising, battery steady (20 s).",
        20_000,
        make_model(_IDLE_PROFILE),
    ),
    Scenario(
        "accel_gears",
        "Launch and acceleration through gears 1..4 with the rpm from the defs gear ratios, "
        "reductions and dynamic tyre radius (30 s).",
        30_000,
        make_model(_ACCEL_PROFILE),
    ),
    Scenario(
        "stale_gap",
        "As accel_gears, but VEHICLE_SPEED samples are missing from 14.0 s to 21.0 s (longer "
        "than every stale limit that reads it), then resume (30 s).",
        30_000,
        make_model(_ACCEL_PROFILE),
        gaps={"VEHICLE_SPEED": [(14_000, 21_000)]},
    ),
    Scenario(
        "start_idle_refuel",
        "Cold key-on, cranking start, launch, braking, rest with a fuel top-up, a second move, "
        "then warm idle with the coolant above the operating temperature and a cooling-fan voltage "
        "step (D-103 start, launch, braking, refuel, idle statistics, fan) (90 s).",
        90_000,
        make_model(_START_PROFILE),
    ),
    Scenario(
        "edge_values",
        "Cruise, stop with the engine on, standing with zero torque, crawl, out-of-range "
        "MAF / barometer, stop, engine off (standstill, no_power, out_of_range, engine_off, "
        "stop_event) (30 s).",
        30_000,
        make_model(_EDGE_PROFILE),
    ),
)


SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}


def input_rows(scenario: Scenario, dids: list[str], idx: Mapping[str, int]) -> list[Any]:
    """(time_ms, did_idx, value) of every DID at its own poll period, ascending time, ties in
    DID table order."""
    rows = []
    for order, did in enumerate(dids):
        period = vehicle_catalog.SIGNALS[did]["poll_period_ms"]
        gaps = scenario.gaps.get(did, [])
        for t in range(0, scenario.duration_ms, period):
            if any(a <= t < b for a, b in gaps):
                continue
            values = scenario.model(t / 1000.0, jitter_fn(scenario.name, t))
            rows.append((t, order, idx[did], did, quantise(did, values[did])))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [(t, i, v, did) for t, _, i, did, v in rows]


class _Inputs:
    """An InputSource over in-memory DID series."""

    def __init__(self, series: dict[str, Series]):
        self._series = series

    def series(self, input_name: str) -> Series | None:
        return self._series.get(input_name)


@dataclass
class OutputRow:
    time_ms: int
    signal: str
    valid: bool
    flag: str | None
    value: float


def evaluate_scenario(
    scenario: Scenario, rows: list[Any], selected: Mapping[str, Mapping[str, Any]]
) -> tuple[list[OutputRow], dict[str, int]]:
    """Runs the reference over the input rows; returns the output rows (time, then defs
    order is fixed later by the signal index) and the valid-row count per selected signal."""
    per_did: dict[str, tuple[list[int], list[float]]] = {}
    for t, _, v, did in rows:
        times, values = per_did.setdefault(did, ([], []))
        times.append(t)
        values.append(v)
    inputs = _Inputs({f"did.{d}": Series(t, v) for d, (t, v) in per_did.items()})
    result = engine.evaluate(inputs, specs=selected)
    out: list[OutputRow] = []
    valid: dict[str, int] = dict.fromkeys(selected, 0)
    for name, entry in result.coverage.items():
        if entry["status"] == engine.ERROR:
            raise RuntimeError(f"{scenario.name}: {name} failed: {entry['message']}")
    for signal in result.outputs:
        for t, value, flag in zip(signal.times, signal.values, signal.flags, strict=True):
            if value is None:
                if flag is None:
                    raise RuntimeError(f"{scenario.name}: {signal.name} invalid without a flag")
                out.append(OutputRow(t, signal.name, False, flag, 0.0))
            else:
                valid[signal.name] += 1
                out.append(OutputRow(t, signal.name, True, flag, to_float32(value)))
    return out, valid


# --- C rendering -----------------------------------------------------------------------------


def banner(sha: str) -> str:
    return (
        f"/* GENERATED by vehicle-server ({GENERATOR}) from vehicle-defs {defs_version()}; "
        f"the expected outputs use that version's stale_after_ms, so test against a defs pin "
        f"with the same derived table (v0.14.0 or later). "
        f"DO NOT EDIT. content-sha256: {sha} */\n"
    )


def with_banner(body: str) -> str:
    """The file text: banner line (hash of the body) followed by the body."""
    return banner(hashlib.sha256(body.encode()).hexdigest()) + body


BANNER_RE = re.compile(r"\A/\* GENERATED by .* content-sha256: ([0-9a-f]{64}) \*/\n")


def split_banner(text: str) -> tuple[str, str]:
    """(content-sha256 stated in the banner, body)."""
    m = BANNER_RE.match(text)
    if m is None:
        raise ValueError("no golden banner")
    return m[1], text[m.end() :]


def types_header() -> str:
    flags = ["NONE", *(f.upper() for f in FLAG_ORDER)]
    enum = ",\n".join(f"    DERIVED_VECTOR_FLAG_{f} = {i}" for i, f in enumerate(flags))
    body = f"""\
/* Shared types of the golden vectors for the derived-signal host tests (D-070 item 2).
 * Rows are plain C data: no strings, no files. did_idx is VEHICLE_CL250_IDX_<NAME> of
 * vehicle_cl250.h, signal is VEHICLE_CL250_DERIVED_<NAME> of vehicle_cl250_derived.h.
 * An invalid output row has valid = false, a flag saying why and value 0.0f.
 * abs_tol is the largest accepted |C - expected| of a valid row (defs resolution / 2). */

#ifndef DERIVED_VECTORS_H
#define DERIVED_VECTORS_H

#include <stdbool.h>
#include <stdint.h>

#include "vehicle_cl250.h"
#include "vehicle_cl250_derived.h"

typedef struct {{
    uint32_t time_ms;
    uint16_t did_idx;
    float value;
}} derived_vector_input_t;

typedef enum {{
{enum}
}} derived_vector_flag_t;

typedef struct {{
    uint32_t time_ms;
    uint16_t signal;
    bool valid;
    uint8_t flag;
    float value;
}} derived_vector_output_t;

typedef struct {{
    uint16_t signal;
    float abs_tol;
}} derived_vector_tol_t;

#endif /* DERIVED_VECTORS_H */
"""
    return body


def scenario_header(
    scenario: Scenario,
    rows: list[Any],
    outputs: list[OutputRow],
    selected: Mapping[str, Mapping[str, Any]],
    did_idx: Mapping[str, int],
    sig_idx: Mapping[str, int],
) -> str:
    n = scenario.name
    upper = n.upper()
    ordered = sorted(outputs, key=lambda r: (r.time_ms, sig_idx[r.signal]))
    present = sorted({r.signal for r in outputs}, key=lambda s: sig_idx[s])
    lines = [
        f"/* Scenario {n}: {scenario.description}",
        f" * Inputs: the DIDs of the selected signals (+ {', '.join(EXTRA_DIDS)}) at their own "
        "poll_period_ms, ascending time, ties in DID table order.",
        f" * Outputs: every row of the selected signals. Selection rule: {SELECTION_RULE}",
        " * Tolerance: resolution / 2 of the defs per signal present. Deterministic, no GPS, "
        "no real ride. */",
        "",
        f"#ifndef DERIVED_{upper}_H",
        f"#define DERIVED_{upper}_H",
        "",
        '#include "derived_vectors.h"',
        "",
        f"#define DERIVED_{upper}_INPUT_COUNT ({len(rows)}u)",
        f"#define DERIVED_{upper}_OUTPUT_COUNT ({len(ordered)}u)",
        f"#define DERIVED_{upper}_TOL_COUNT ({len(present)}u)",
        "",
        f"static const derived_vector_input_t derived_{n}_inputs[] = {{",
    ]
    for t, i, v, did in rows:
        macro = f"VEHICLE_CL250_IDX_{did}"
        assert did_idx[did] == i
        lines.append(f"    {{ {t}u, {macro}, {c_float(v)} }},")
    lines += ["};", "", f"static const derived_vector_output_t derived_{n}_outputs[] = {{"]
    known = set(FLAG_ORDER)
    for r in ordered:
        if r.flag is not None and r.flag not in known:
            raise RuntimeError(f"{n}: {r.signal} carries unknown flag {r.flag!r}")
        flag = flag_constant(r.flag) if r.flag else "DERIVED_VECTOR_FLAG_NONE"
        macro = f"VEHICLE_CL250_DERIVED_{r.signal.upper()}"
        valid = "true" if r.valid else "false"
        lines.append(f"    {{ {r.time_ms}u, {macro}, {valid}, {flag}, {c_float(r.value)} }},")
    lines += ["};", "", f"static const derived_vector_tol_t derived_{n}_tol[] = {{"]
    for s in present:
        tol = selected[s]["resolution"] / 2.0
        lines.append(f"    {{ VEHICLE_CL250_DERIVED_{s.upper()}, {c_float(tol)} }},")
    lines += ["};", "", f"#endif /* DERIVED_{upper}_H */", ""]
    return "\n".join(lines)


# --- generation ------------------------------------------------------------------------------


@dataclass
class GeneratedFile:
    name: str
    text: str
    sha256: str  # content-sha256 (of the body after the banner)


@dataclass
class Generation:
    files: list[GeneratedFile]
    selected: list[str]
    coverage: dict[str, dict[str, int]]  # signal -> scenario -> valid rows
    stats: dict[str, tuple[int, int]]  # scenario -> (input rows, output rows)
    class_b: Any = None  # golden_b.ClassBGeneration


def _file(name: str, body: str) -> GeneratedFile:
    return GeneratedFile(name, with_banner(body), hashlib.sha256(body.encode()).hexdigest())


def generate() -> Generation:
    """Builds every file in memory; raises if a selected signal is never valid."""
    unknown = used_flags() - set(FLAG_ORDER)
    if unknown:
        raise RuntimeError(f"flags of the reference missing in FLAG_ORDER: {sorted(unknown)}")
    selected = select_signals()
    dids = needed_dids(selected)
    did_idx, sig_idx = did_indices(), derived_indices()
    missing = [d for d in dids if d not in did_idx] + [s for s in selected if s not in sig_idx]
    if missing:
        raise RuntimeError(f"not in the gen/ C enums: {missing}")

    files = [_file(TYPES_HEADER, types_header())]
    coverage: dict[str, dict[str, int]] = {s: {} for s in selected}
    stats: dict[str, tuple[int, int]] = {}
    for scenario in SCENARIOS:
        rows = input_rows(scenario, dids, did_idx)
        outputs, valid = evaluate_scenario(scenario, rows, selected)
        for signal, count in valid.items():
            coverage[signal][scenario.name] = count
        stats[scenario.name] = (len(rows), len(outputs))
        body = scenario_header(scenario, rows, outputs, selected, did_idx, sig_idx)
        files.append(_file(f"derived_{scenario.name}.h", body))
    never = [s for s, per in coverage.items() if not any(per.values())]
    if never:
        raise RuntimeError(f"selected signals never valid in any scenario: {never}")
    # class B and the D-098 records extend this module (lazy imports: no cycle)
    from vehicle_server.derived import golden_b, golden_records

    class_b = golden_b.generate_b()
    files += class_b.files
    files.append(golden_records.generate_records(class_b))
    return Generation(files, list(selected), coverage, stats, class_b)


def manifest_text(files: list[GeneratedFile]) -> str:
    lines = [f"{f.sha256}  {f.name}" for f in files]
    return "\n".join([*lines, f"defs {defs_version()}"]) + "\n"


def coverage_table(gen: Generation) -> str:
    names = [s.name for s in SCENARIOS]
    width = max(len(s) for s in gen.selected)
    head = f"{'signal':<{width}}  " + "  ".join(f"{n:>11}" for n in names)
    rows = [head]
    for s in gen.selected:
        rows.append(f"{s:<{width}}  " + "  ".join(f"{gen.coverage[s][n]:>11}" for n in names))
    return "\n".join(rows)


def write(out_dir: Path, manifest: Path | None) -> Generation:
    gen = generate()
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in gen.files:
        (out_dir / f.name).write_text(f.text)
    if manifest is not None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(manifest_text(gen.files))
    return gen


def report(gen: Generation, out_dir: Path, manifest: Path | None) -> str:
    lines = [f"golden vectors from vehicle-defs {defs_version()} -> {out_dir}"]
    for f in gen.files:
        size = len((out_dir / f.name).read_bytes())
        scenario = f.name.removeprefix("derived_").removesuffix(".h")
        counts = gen.stats.get(scenario)
        extra = f"  inputs={counts[0]} outputs={counts[1]}" if counts else ""
        lines.append(f"  {f.name:<28} {size / 1024:7.1f} KB  sha256={f.sha256[:12]}{extra}")
    lines.append(f"selected signals ({len(gen.selected)}): valid rows per scenario")
    lines.append(coverage_table(gen))
    if gen.class_b is not None:
        from vehicle_server.derived import golden_b

        lines.append(golden_b.report_b(gen.class_b))
    if manifest is not None:
        lines.append(f"manifest: {manifest}")
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="directory for the generated headers")
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / MANIFEST_RELATIVE),
        help="where to write MANIFEST.sha256 (default: tests/golden/ of this repo); "
        "'-' writes none",
    )


def run(out: str, manifest: str) -> int:
    out_dir = Path(out)
    manifest_path = None if manifest == "-" else Path(manifest)
    try:
        gen = write(out_dir, manifest_path)
    except (RuntimeError, AssertionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report(gen, out_dir, manifest_path))
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m vehicle_server.derived.golden")
    add_arguments(parser)
    args = parser.parse_args(argv)
    sys.exit(run(args.out, args.manifest))


if __name__ == "__main__":
    main()
