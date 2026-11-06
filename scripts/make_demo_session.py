#!/usr/bin/env python3
"""Writes one deterministic synthetic session that exercises every D-079 window.

The session has no GPS block and no position data. Values are built from the
generated vehicle catalog (never from hand-written byte offsets), and the
signals.csv rows use the same wire layout the server decodes:

    uv run python scripts/make_demo_session.py --out <dir>
    uv run python scripts/make_demo_session.py --out <dir> --zip

The output is named synthetic-demo-<YYYYMMDD> (a folder, or a .zip with the
session files at its root). Ingestion accepts both.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.make_fixtures import write_csv  # noqa: E402

from vehicle_server.decode import SIGNALS_HEADER, V4_HEADER, vsig_state  # noqa: E402
from vehicle_server.defs import ble as defs_ble  # noqa: E402
from vehicle_server.defs import vehicle_catalog  # noqa: E402

DEMO_DATE = "20261106"
SESSION_NAME = f"synthetic-demo-{DEMO_DATE}"
SESSION_ID = "20261106-090000-5e55"
START = datetime(2026, 11, 6, 9, 0, 0, tzinfo=UTC)
DURATION_MS = 20 * 60 * 1000
TICK_MS = 100

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
AMBIENT = 20.0
COOLANT_TAU_MS = 300_000.0
ENGINE_SPEED_GAP_MS = (303_000, 308_000)


def _raw(name: str, value: float) -> int:
    entry = vehicle_catalog.SIGNALS[name]
    return int(round((value - entry["offset"]) * entry["factor_den"] / entry["factor_num"]))


def _decode(name: str, raw: int) -> float:
    entry = vehicle_catalog.SIGNALS[name]
    return vehicle_catalog.decode(name, raw.to_bytes(entry["length"], "big"))


def _coolant_at(t_ms: int) -> float:
    if t_ms < 3_000:
        return AMBIENT
    if t_ms < 63_000:
        return AMBIENT + (85.0 - AMBIENT) * (t_ms - 3_000) / 60_000.0
    if t_ms < 1_120_000:
        return 85.0
    return AMBIENT + (85.0 - AMBIENT) * math.exp(-(t_ms - 1_120_000) / COOLANT_TAU_MS)


def _profile(t_ms: int) -> tuple[float, float, float]:
    """(rpm, speed km/h, throttle %) for the synthetic ride."""
    if t_ms < 3_000:
        return 0.0, 0.0, 0.0
    if t_ms < 96_000:
        return 1_200.0, 0.0, 0.0
    if t_ms < 120_000:
        return 5_124.0, 60.0, 10.0
    if t_ms < 125_000:
        return 4_175.0, 40.0, 10.0
    if t_ms < 150_000:
        return 5_163.0, 70.0, 10.0
    if t_ms < 155_000:
        return 4_175.0, 40.0, 10.0
    if t_ms < 180_000:
        return 5_253.0, 80.0, 10.0
    if t_ms < 185_000:
        return 4_175.0, 40.0, 10.0
    if t_ms < 300_000:
        return 5_124.0, 60.0, 10.0
    if t_ms < 312_000:
        return 1_200.0, 0.0, 0.0
    if t_ms < 315_000:
        return 4_175.0, 40.0, 10.0
    if t_ms < 600_000:
        return 5_124.0, 60.0, 10.0
    if t_ms < 615_000:
        return 1_200.0, 0.0, 0.0
    if t_ms < 620_000:
        return 4_175.0, 40.0, 10.0
    if t_ms < 1_120_000:
        return 5_124.0, 60.0, 10.0
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
        return AMBIENT
    if name == "INTAKE_AIR_TEMP":
        return AMBIENT
    if name == "BAROMETRIC_PRESSURE":
        return 100.0
    if name == "SHORT_TERM_FUEL_TRIM_B1":
        return 5.0
    raise KeyError(name)


def _utc_iso(t_ms: int) -> str:
    value = START + timedelta(milliseconds=t_ms)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _engine_speed_gap(t_ms: int) -> bool:
    return ENGINE_SPEED_GAP_MS[0] <= t_ms < ENGINE_SPEED_GAP_MS[1]


def _signal_rows() -> list[list[str]]:
    state = str(vsig_state("valid"))
    rows: list[list[str]] = []
    block_seq = 0
    for t_ms in range(0, DURATION_MS + 1, TICK_MS):
        due = ALL_DIDS if t_ms % 1_000 == 0 else FAST_DIDS
        names = [name for name in due if not (name == "ENGINE_SPEED" and _engine_speed_gap(t_ms))]
        if not names:
            continue
        first_record = ALL_DIDS.index(names[0])
        rx = _utc_iso(t_ms)
        seq = block_seq % 256
        block_seq += 1
        for name in names:
            raw = _raw(name, _value(name, t_ms))
            entry = vehicle_catalog.SIGNALS[name]
            rows.append(
                [
                    rx,
                    str(t_ms),
                    str(seq),
                    "0",
                    str(t_ms),
                    str(len(ALL_DIDS)),
                    str(first_record),
                    str(entry["did"]),
                    state,
                    "0",
                    str(raw),
                    name,
                    repr(_decode(name, raw)),
                ]
            )
    return rows


def _meta() -> dict[str, object]:
    return {
        "session_id": SESSION_ID,
        "created_utc": _utc_iso(0),
        "rider_name": "synthetic-rider",
        "rider_weight_kg": 75,
        "extra_load_kg": 0,
        "ambient_temp_c": AMBIENT,
        "weather": "dry",
        "tire_pressure_front_bar": 2.2,
        "tire_pressure_rear_bar": 2.5,
        "fuel_level": 0.8,
        "vehicle_config": "stock",
        "condition_label": "synthetic-demo",
        "route_type": "desk demo",
        "note": "Synthetic session generated by scripts/make_demo_session.py; never a real ride.",
        "app_version": "synthetic-demo",
        "ble_schema_version": 4,
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


def write_session(directory: Path) -> Path:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    write_csv(directory / "telemetry.csv", V4_HEADER, [])
    write_csv(directory / "signals.csv", SIGNALS_HEADER, _signal_rows())
    write_csv(
        directory / "events.csv",
        ["rx_utc_iso", "rx_mono_ms", "event", "detail"],
        [
            [_utc_iso(0), "0", "recording_started", ""],
            [_utc_iso(DURATION_MS), str(DURATION_MS), "recording_stopped", ""],
        ],
    )
    (directory / "meta.json").write_text(json.dumps(_meta(), indent=2) + "\n")
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
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    return destination


def build(out: Path, as_zip: bool = False) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    if not as_zip:
        return write_session(out / SESSION_NAME)

    with tempfile.TemporaryDirectory(prefix=".synthetic-demo-", dir=out) as tmp:
        directory = write_session(Path(tmp) / SESSION_NAME)
        return _zip_directory(directory, out / f"{SESSION_NAME}.zip")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--zip", action="store_true", help="write a .zip instead of a folder")
    args = parser.parse_args(argv)
    path = build(args.out.resolve(), args.zip)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
