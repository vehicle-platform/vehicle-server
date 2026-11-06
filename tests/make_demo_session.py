#!/usr/bin/env python3
"""Writes one deterministic synthetic session that exercises every D-079 window.

Phases, in order: key-on rest (W1), crank with one stall and one stale engine-speed gap
(W2, closing one instance by `engine_zero` and one by `unknown`), cold start and warm-up,
a qualifying warm idle (W3), a short warm idle that does not qualify, steady cruises in three
gears (W4 and the per-gear final-drive summary), two D-075 stops, and a final engine stop
followed by 80 s of key-on coolant decay (W5, closing at the session end).

The session is synthetic and has no GPS block or position column. Names, DIDs, scales and
window thresholds come from the pinned vehicle-defs catalog; the date defaults to a fixed
day so two runs are byte-identical.

    uv run python -m tests.make_demo_session --out <dir>
    uv run python -m tests.make_demo_session --out <dir> --zip
    uv run python -m tests.make_demo_session --out <dir> --date 2026-01-01
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from vehicle_server.decode import SIGNALS_HEADER, V4_HEADER
from vehicle_server.defs import ble as defs_ble
from vehicle_server.defs import (
    vehicle_catalog,
    vehicle_cl250,
    vehicle_constants,
    vehicle_derived,
)

from .make_fixtures import vsig_row_cells, write_csv

DEFAULT_DATE = date(2026, 1, 1)
START_HOUR_UTC = 9
DURATION_MS = 20 * 60 * 1000
TICK_MS = 100
AMBIENT_C = 20.0
COOLANT_TAU_MS = 300_000.0

FAST_DIDS = ("ENGINE_SPEED", "VEHICLE_SPEED", "THROTTLE_POS")
SLOW_DIDS = (
    "COOLANT_TEMP",
    "INTAKE_MAP",
    "BATTERY_VOLTAGE",
    "AMBIENT_AIR_TEMP",
    "INTAKE_AIR_TEMP",
    "BAROMETRIC_PRESSURE",
    "SHORT_TERM_FUEL_TRIM_B1",
)
ALL_DIDS = (*FAST_DIDS, *SLOW_DIDS)


def _const(name: str) -> float:
    return float(vehicle_constants.CONSTANTS[name][0])


def _condition(key: str, input_name: str, op: str | None = None) -> dict[str, object]:
    for condition in vehicle_derived.WINDOWS[key].get("conditions") or ():
        if condition["input"] == input_name and (op is None or condition["op"] == op):
            return condition
    raise KeyError(f"{key} has no condition on {input_name} ({op})")


def _stable(key: str, input_name: str) -> dict[str, object]:
    for stable in vehicle_derived.WINDOWS[key].get("stable") or ():
        if stable["input"] == input_name:
            return stable
    raise KeyError(f"{key} has no stable band on {input_name}")


W1_MIN = int(vehicle_derived.WINDOWS["W1_KEY_ON"]["min_duration_ms"])
W2_MIN = int(vehicle_derived.WINDOWS["W2_CRANK"]["min_duration_ms"])
W3_MIN = int(vehicle_derived.WINDOWS["W3_WARM_IDLE"]["min_duration_ms"])
W3_MAX = int(vehicle_derived.WINDOWS["W3_WARM_IDLE"]["max_duration_ms"])
W4_MIN = int(vehicle_derived.WINDOWS["W4_CRUISE"]["min_duration_ms"])
W4_MAX = int(vehicle_derived.WINDOWS["W4_CRUISE"]["max_duration_ms"])
W5_MIN = int(vehicle_derived.WINDOWS["W5_POST_STOP"]["min_duration_ms"])

COOLANT_WARM = float(_condition("W3_WARM_IDLE", "did.COOLANT_TEMP", "gt")["value"])
COOLANT_TARGET = COOLANT_WARM + 5.0
IDLE_RPM = float(_condition("W3_WARM_IDLE", "did.ENGINE_SPEED", "ge")["value"]) + 200.0
IDLE_RPM_MAX = float(_condition("W3_WARM_IDLE", "did.ENGINE_SPEED", "le")["value"])
if IDLE_RPM > IDLE_RPM_MAX:
    IDLE_RPM = (IDLE_RPM_MAX + COOLANT_WARM) / 2.0

CRUISE_GEAR_MIN = int(_condition("W4_CRUISE", "derived.gear_estimate", "ge")["value"])
SPEED_SPREAD = float(_stable("W4_CRUISE", "did.VEHICLE_SPEED")["max_spread"])
THROTTLE_SPREAD = float(_stable("W4_CRUISE", "did.THROTTLE_POS")["max_spread"])
CALIBRATION_SPEED = _const("derived_calibration_speed_min_kmh")
GEAR_COUNT = int(_const("gear_count"))
CRUISE_GEARS = tuple(range(CRUISE_GEAR_MIN, min(GEAR_COUNT, CRUISE_GEAR_MIN + 3) + 1))
REARM_GEAR = CRUISE_GEAR_MIN - 1
ENGINE_STALE_MS = int(vehicle_cl250.DID_STALE_AFTER_MS["ENGINE_SPEED"])

PRIMARY = _const("primary_reduction_ratio")
SECONDARY = _const("secondary_reduction_ratio")
WHEEL_RADIUS = _const("rear_wheel_radius_dynamic_m")


def _ceil_tick(value: int) -> int:
    return ((value + TICK_MS - 1) // TICK_MS) * TICK_MS


def _cruise_speed(gear: int) -> float:
    return CALIBRATION_SPEED + 10.0 + 10.0 * (gear - CRUISE_GEAR_MIN)


CRUISE_SPEEDS = {gear: _cruise_speed(gear) for gear in CRUISE_GEARS}
REARM_SPEED = CALIBRATION_SPEED + 10.0


def _rpm(gear: int, speed_kmh: float) -> float:
    ratio = _const(f"gear_ratio_{gear}")
    wheel_rpm = (speed_kmh / 3.6) / (2.0 * math.pi * WHEEL_RADIUS) * 60.0
    return ratio * wheel_rpm * PRIMARY * SECONDARY


CRUISE_RPM = {gear: _rpm(gear, speed) for gear, speed in CRUISE_SPEEDS.items()}
REARM_RPM = _rpm(REARM_GEAR, REARM_SPEED)
ALL_RPM = {**CRUISE_RPM, REARM_GEAR: REARM_RPM}
ALL_SPEEDS = {**CRUISE_SPEEDS, REARM_GEAR: REARM_SPEED}

W1_END = W1_MIN + 1_000
STALL_AT = W1_END + max(W2_MIN + 400, 1_000)
RESTART_AT = STALL_AT + 1_000
ENGINE_GAP = (
    RESTART_AT + 100,
    RESTART_AT + _ceil_tick(ENGINE_STALE_MS + 100),
)
WARM_START = RESTART_AT
WARM_AT = WARM_START + 60_000
IDLE_QUAL_END = WARM_AT + min(W3_MAX - 5_000, max(2 * W3_MIN, 30_000))
MOVE_AFTER_IDLE = IDLE_QUAL_END + 5_000
SHORT_IDLE_END = MOVE_AFTER_IDLE + max(W3_MIN // 2, 5_000)
CRUISE_START = SHORT_IDLE_END
CRUISE_DURATION = min(W4_MAX - 5_000, max(2 * W4_MIN, 20_000))
REARM_DURATION = 5_000
STOP1 = 300_000
STOP1_END = 312_000
STOP2 = 600_000
STOP2_END = 615_000
ENGINE_STOP = 1_120_000
END = DURATION_MS

CRUISE_SEGMENTS: list[tuple[int, int, int]] = []
cursor = CRUISE_START
for index, gear in enumerate(CRUISE_GEARS):
    CRUISE_SEGMENTS.append((cursor, cursor + CRUISE_DURATION, gear))
    cursor += CRUISE_DURATION
    if index < len(CRUISE_GEARS) - 1:
        CRUISE_SEGMENTS.append((cursor, cursor + REARM_DURATION, REARM_GEAR))
        cursor += REARM_DURATION
CRUISE_SEGMENTS.append((cursor, STOP1, CRUISE_GEARS[0]))

POST_STOP_SEGMENTS = (
    (STOP1_END, STOP1_END + REARM_DURATION, REARM_GEAR),
    (STOP1_END + REARM_DURATION, STOP2, CRUISE_GEARS[0]),
    (STOP2_END, STOP2_END + REARM_DURATION, REARM_GEAR),
    (STOP2_END + REARM_DURATION, ENGINE_STOP, CRUISE_GEARS[0]),
)


def _raw(name: str, value: float) -> int:
    entry = vehicle_catalog.SIGNALS[name]
    return int(round((value - entry["offset"]) * entry["factor_den"] / entry["factor_num"]))


def _coolant_at(t_ms: int) -> float:
    if t_ms < WARM_START:
        return AMBIENT_C
    if t_ms < WARM_AT:
        return AMBIENT_C + (COOLANT_TARGET - AMBIENT_C) * (t_ms - WARM_START) / (
            WARM_AT - WARM_START
        )
    if t_ms < ENGINE_STOP:
        return COOLANT_TARGET
    return AMBIENT_C + (COOLANT_TARGET - AMBIENT_C) * math.exp(
        -(t_ms - ENGINE_STOP) / COOLANT_TAU_MS
    )


def _profile(t_ms: int) -> tuple[float, float, float]:
    if t_ms < W1_END:
        return 0.0, 0.0, 0.0
    if t_ms < STALL_AT:
        return IDLE_RPM, 0.0, 0.0
    if t_ms < RESTART_AT:
        return 0.0, 0.0, 0.0
    if t_ms < IDLE_QUAL_END:
        return IDLE_RPM, 0.0, 0.0
    if t_ms < MOVE_AFTER_IDLE:
        return REARM_RPM, REARM_SPEED, 10.0
    if t_ms < SHORT_IDLE_END:
        return IDLE_RPM, 0.0, 0.0
    for start, end, gear in CRUISE_SEGMENTS:
        if start <= t_ms < end:
            return ALL_RPM[gear], ALL_SPEEDS[gear], 10.0
    for start, end, gear in POST_STOP_SEGMENTS:
        if start <= t_ms < end:
            return ALL_RPM[gear], ALL_SPEEDS[gear], 10.0
    if STOP1 <= t_ms < STOP1_END or STOP2 <= t_ms < STOP2_END:
        return IDLE_RPM, 0.0, 0.0
    return 0.0, 0.0, 0.0


def _value(name: str, t_ms: int) -> float:
    rpm, speed, throttle = _profile(t_ms)
    if name == "ENGINE_SPEED":
        return rpm
    if name == "VEHICLE_SPEED":
        return speed
    if name == "THROTTLE_POS":
        return throttle
    if name == "COOLANT_TEMP":
        return _coolant_at(t_ms)
    if name == "INTAKE_MAP":
        return 100.0
    if name == "BATTERY_VOLTAGE":
        return 12.6
    if name == "AMBIENT_AIR_TEMP":
        return AMBIENT_C
    if name == "INTAKE_AIR_TEMP":
        return AMBIENT_C
    if name == "BAROMETRIC_PRESSURE":
        return 100.0
    if name == "SHORT_TERM_FUEL_TRIM_B1":
        return 5.0
    raise KeyError(name)


def _session_id(session_date: date) -> str:
    return f"{session_date:%Y%m%d}-{START_HOUR_UTC:02d}0000-5e55"


def _session_name(session_date: date) -> str:
    return f"synthetic-demo-{session_date:%Y%m%d}"


def _utc_iso(start: datetime, t_ms: int) -> str:
    value = start + timedelta(milliseconds=t_ms)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _engine_speed_gap(t_ms: int) -> bool:
    return ENGINE_GAP[0] <= t_ms < ENGINE_GAP[1]


def _signal_rows(start: datetime) -> list[list[str]]:
    rows: list[list[str]] = []
    block_seq = 0
    for t_ms in range(0, DURATION_MS + 1, TICK_MS):
        due = ALL_DIDS if t_ms % 1_000 == 0 else FAST_DIDS
        names = [name for name in due if not (name == "ENGINE_SPEED" and _engine_speed_gap(t_ms))]
        if not names:
            continue
        first_record = ALL_DIDS.index(names[0])
        rx = _utc_iso(start, t_ms)
        seq = block_seq % 256
        block_seq += 1
        for index, name in enumerate(names):
            rows.append(
                vsig_row_cells(
                    rx,
                    t_ms,
                    seq,
                    0,
                    first_record,
                    vehicle_catalog.SIGNALS[name]["did"],
                    "valid",
                    0,
                    _raw(name, _value(name, t_ms)),
                    index == 0,
                    table_len=len(ALL_DIDS),
                )
            )
    return rows


def _meta(session_date: date, start: datetime) -> dict[str, object]:
    return {
        "session_id": _session_id(session_date),
        "created_utc": _utc_iso(start, 0),
        "rider_name": "synthetic-rider",
        "rider_weight_kg": 75,
        "extra_load_kg": 0,
        "ambient_temp_c": AMBIENT_C,
        "weather": "dry",
        "tire_pressure_front_bar": 2.2,
        "tire_pressure_rear_bar": 2.5,
        "fuel_level": 0.8,
        "vehicle_config": "stock",
        "condition_label": "synthetic-demo",
        "route_type": "desk demo",
        "note": "Synthetic session from tests/make_demo_session.py; never a real ride.",
        "app_version": "synthetic-demo",
        "ble_schema_version": defs_ble.CURRENT_VERSION,
        "vehicle_signal_block_version": defs_ble.VSIG_BLOCK_VERSION,
        "device_name": "CL250-synthetic",
        "synthetic": True,
    }


def _summary() -> dict[str, object]:
    return {
        "duration_ms": DURATION_MS,
        "packet_count": 0,
        "lost_count": 0,
        "loss_percent": 0.0,
        "decode_error_count": 0,
        "disconnect_count": 0,
        "synthetic": True,
    }


def write_session(directory: Path, session_date: date = DEFAULT_DATE) -> Path:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    start = datetime(
        session_date.year,
        session_date.month,
        session_date.day,
        START_HOUR_UTC,
        tzinfo=UTC,
    )
    write_csv(directory / "telemetry.csv", V4_HEADER, [])
    write_csv(directory / "signals.csv", SIGNALS_HEADER, _signal_rows(start))
    write_csv(
        directory / "events.csv",
        ["rx_utc_iso", "rx_mono_ms", "event", "detail"],
        [
            [_utc_iso(start, 0), "0", "recording_started", ""],
            [_utc_iso(start, DURATION_MS), str(DURATION_MS), "recording_stopped", ""],
        ],
    )
    (directory / "meta.json").write_text(json.dumps(_meta(session_date, start), indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(_summary(), indent=2) + "\n")
    return directory


def _zip_directory(directory: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as archive:
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    return destination


def build(out: Path, as_zip: bool = False, session_date: date = DEFAULT_DATE) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    if not as_zip:
        return write_session(out / _session_name(session_date), session_date)

    with tempfile.TemporaryDirectory(prefix=".synthetic-demo-", dir=out) as tmp:
        directory = write_session(Path(tmp) / _session_name(session_date), session_date)
        return _zip_directory(directory, out / f"{_session_name(session_date)}.zip")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--zip", action="store_true", help="write a .zip instead of a folder")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=DEFAULT_DATE,
        help="session date, YYYY-MM-DD (default: the fixed 2026-01-01)",
    )
    args = parser.parse_args(argv)
    path = build(args.out.resolve(), args.zip, args.date)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
