#!/usr/bin/env python3
"""Generates the committed test fixture sessions under tests/fixtures/.

Builds raw_hex telemetry rows FROM THE SCHEMA (ble_schema.pack_fields), never
from hand-written byte offsets, so the fixtures exercise the same code path
the decoder uses. Run this again (and commit the result) whenever the fixture
scenarios need to change:

    uv run python tests/make_fixtures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from vehicle_server import ble_schema
from vehicle_server.decode import GPS_HEADER, SIGNALS_HEADER, V4_EXTRA_HEADER
from vehicle_server.defs import ble as defs_ble
from vehicle_server.defs import vehicle_catalog, vehicle_cl250

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SCHEMA = ble_schema.load_default_schema()


def bits_to_int(bit_defs: list[dict], set_names: set[str]) -> int:
    value = 0
    for bit in bit_defs:
        if bit["name"] in set_names:
            value |= 1 << bit["bit"]
    return value


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# v2 session: 16-byte raw_hex rows, two lost-packet gaps, one stale (invalid) value.
# ---------------------------------------------------------------------------

V2_SESSION_ID = "20260115-120000-a1b2"

FLAG_BITS = SCHEMA["flags"]["bits"]
V2_ALL_VALID = {
    "rpmValid",
    "speedValid",
    "coolantTempValid",
    "throttlePosValid",
    "batteryVoltValid",
    "ecuPresent",
}


def make_v2_fixture() -> None:
    out_dir = FIXTURES_DIR / "v2_session"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = SCHEMA["lowMtuFallback"]["fields"]
    total_bytes = SCHEMA["lowMtuFallback"]["totalBytes"]

    # seq sequence has two single-packet gaps: 0,1,2,4,5,7,8 (lost after seq=2 and seq=5).
    seqs = [0, 1, 2, 4, 5, 7, 8]
    stale_row_index = 3  # seq=4: battery is stale (invalid), value kept from before.

    telemetry_rows = []
    for i, seq in enumerate(seqs):
        rpm = 3000 + i * 10
        speed = 60
        coolant = 85
        tps = 30
        battery_raw = 12400  # 12.4 V
        valid_names = set(V2_ALL_VALID)
        if i == stale_row_index:
            valid_names.discard("batteryVoltValid")
        flags_value = bits_to_int(FLAG_BITS, valid_names)

        values = {
            "version": 2,
            "seq": seq,
            "rpm": rpm,
            "speed": speed,
            "coolantTemp": coolant,
            "throttlePos": tps,
            "batteryVolt": battery_raw,
            "leanAngle": -32768,
            "maxLeanRight": -32768,
            "maxLeanLeft": -32768,
            "flags": flags_value,
        }
        raw = ble_schema.pack_fields(values, fields, total_bytes)
        rx_mono_ms = 1_000 + i * 100

        telemetry_rows.append(
            [
                f"2026-01-15T12:00:{i:02d}.000Z",
                str(rx_mono_ms),
                str(seq),
                "0" if i == 0 else str(seq - seqs[i - 1] - 1),
                raw.hex(),
                str(rpm),
                str(speed),
                str(coolant),
                str(tps),
                f"{battery_raw / 1000:.3f}",
                "",
                "",
                "",  # lean fields, deprecated, always empty for v2 fixture output
                "1",
                "1",
                "1",
                "1",
                "0" if i == stale_row_index else "1",
                "",
                "1",
                "",
            ]
        )

    write_csv(out_dir / "telemetry.csv", _v2_telemetry_header(), telemetry_rows)
    write_csv(
        out_dir / "events.csv",
        ["rx_utc_iso", "rx_mono_ms", "event", "detail"],
        [
            ["2026-01-15T12:00:00.000Z", "1000", "recording_started", ""],
            ["2026-01-15T12:00:02.200Z", "1200", "packet_gap", "lost=1"],
            ["2026-01-15T12:00:05.500Z", "1500", "packet_gap", "lost=1"],
            [
                f"2026-01-15T12:00:{len(seqs):02d}.000Z",
                str(1000 + len(seqs) * 100),
                "recording_stopped",
                "",
            ],
        ],
    )

    meta = {
        "session_id": V2_SESSION_ID,
        "created_utc": "2026-01-15T12:00:00.000Z",
        "rider_name": "test-rider",
        "rider_weight_kg": 75,
        "extra_load_kg": 0,
        "ambient_temp_c": 18,
        "weather": "dry",
        "tire_pressure_front_bar": 2.2,
        "tire_pressure_rear_bar": 2.5,
        "fuel_level": 0.8,
        "vehicle_config": "stock",
        "condition_label": "healthy",
        "route_type": "urban",
        "note": "v2 fixture: two single-packet gaps, one stale battery reading",
        "app_version": "1.4.0",
        "ble_schema_version": 2,
        "device_name": "CL250-fixture",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    lost = sum((seqs[i] - seqs[i - 1] - 1) for i in range(1, len(seqs)))
    summary = {
        "duration_ms": 1000 + len(seqs) * 100,
        "packet_count": len(seqs),
        "lost_count": lost,
        "loss_percent": round(lost / (lost + len(seqs)) * 100, 4),
        "decode_error_count": 0,
        "disconnect_count": 0,
        "first_packet_utc": "2026-01-15T12:00:00.000Z",
        "last_packet_utc": f"2026-01-15T12:00:{len(seqs):02d}.000Z",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _v2_telemetry_header() -> list[str]:
    return [
        "rx_utc_iso",
        "rx_mono_ms",
        "seq",
        "lost_since_prev",
        "raw_hex",
        "rpm",
        "speed_kmh",
        "coolant_c",
        "tps_pct",
        "battery_v",
        "lean_deg",
        "max_lean_right_deg",
        "max_lean_left_deg",
        "rpm_valid",
        "speed_valid",
        "coolant_valid",
        "tps_valid",
        "battery_valid",
        "lean_valid",
        "ecu_present",
        "decode_error",
    ]


# ---------------------------------------------------------------------------
# v3 session: 37-byte raw_hex rows with changing CAN health, one out-of-range
# value, plus imu.csv with a missing-sample gap that wraps sample_index.
# ---------------------------------------------------------------------------

V3_SESSION_ID = "20260115-130000-c3d4"

CAN_STATE_BY_NAME = {v["name"]: v["value"] for v in SCHEMA["canHealth"]["busState"]["values"]}
CAN_FLAG_BITS = SCHEMA["canHealth"]["canFlags"]["bits"]


def make_v3_fixture() -> None:
    out_dir = FIXTURES_DIR / "v3_session"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = defs_ble.telemetry_fields(3)
    total_bytes = defs_ble.TOTAL_BYTES_BY_VERSION[3]

    # seq has one single-packet gap: 10,11,13,14,15 (lost after seq=11).
    seqs = [10, 11, 13, 14, 15]
    out_of_range_row_index = 2  # coolant below DIDS min (-40 degC)

    can_rows = [
        {
            "state": "running",
            "tec": 0,
            "rec": 0,
            "bus_off": 0,
            "unanswered": 0,
            "flags": {"pollerEnabled"},
        },
        {
            "state": "errorWarning",
            "tec": 100,
            "rec": 50,
            "bus_off": 0,
            "unanswered": 0,
            "flags": {"pollerEnabled"},
        },
        {
            "state": "busOff",
            "tec": 255,
            "rec": 255,
            "bus_off": 1,
            "unanswered": 2,
            "flags": {"pollerEnabled"},
        },
        {
            "state": "stopped",
            "tec": 255,
            "rec": 255,
            "bus_off": 1,
            "unanswered": 2,
            "flags": {"pollerEnabled", "latchedBusOff"},
        },
        {
            "state": "errorWarning",
            "tec": 20,
            "rec": 15,
            "bus_off": 1,
            "unanswered": 2,
            "flags": {"pollerEnabled"},
        },
    ]

    telemetry_rows = []
    for i, seq in enumerate(seqs):
        rpm = 4000 + i * 5
        speed = 90
        coolant = -100 if i == out_of_range_row_index else 88
        tps = 40
        battery_raw = 12600
        device_time_ms = 5_000 + i * 100
        can = can_rows[i]

        values = {
            "version": 3,
            "seq": seq,
            "deviceTimeMs": device_time_ms,
            "rpm": rpm,
            "speed": speed,
            "coolantTemp": coolant,
            "throttlePos": tps,
            "batteryVolt": battery_raw,
            "leanAngle": -32768,
            "maxLeanRight": -32768,
            "maxLeanLeft": -32768,
            "flags": bits_to_int(
                FLAG_BITS,
                {
                    "rpmValid",
                    "speedValid",
                    "coolantTempValid",
                    "throttlePosValid",
                    "batteryVoltValid",
                    "ecuPresent",
                    "imuActive",
                },
            ),
            "rpmAgeMs": 10,
            "speedAgeMs": 15,
            "coolantTempAgeMs": 200,
            "throttlePosAgeMs": 20,
            "batteryVoltAgeMs": 400,
            "canBusState": CAN_STATE_BY_NAME[can["state"]],
            "canTxErrorCount": can["tec"],
            "canRxErrorCount": can["rec"],
            "canBusOffCount": can["bus_off"],
            "unansweredDidCount": can["unanswered"],
            "canFlags": bits_to_int(CAN_FLAG_BITS, can["flags"]),
        }
        raw = ble_schema.pack_fields(values, fields, total_bytes)
        rx_mono_ms = 5_000 + i * 100

        telemetry_rows.append(
            [
                f"2026-01-15T13:00:{i:02d}.000Z",
                str(rx_mono_ms),
                str(seq),
                "0" if i == 0 else str(seq - seqs[i - 1] - 1),
                raw.hex(),
                str(rpm),
                str(speed),
                str(coolant),
                str(tps),
                f"{battery_raw / 1000:.3f}",
                "",
                "",
                "",  # deprecated lean columns
                "1",
                "1",
                "1",
                "1",
                "1",
                "",
                "1",
                "",
                "3",
                str(device_time_ms),
                "10",
                "15",
                "200",
                "20",
                "400",
                "1",
                str(values["canBusState"]),
                str(can["tec"]),
                str(can["rec"]),
                str(can["bus_off"]),
                str(can["unanswered"]),
                str(values["canFlags"]),
            ]
        )

    write_csv(out_dir / "telemetry.csv", _v3_telemetry_header(), telemetry_rows)
    write_csv(
        out_dir / "events.csv",
        ["rx_utc_iso", "rx_mono_ms", "event", "detail"],
        [
            ["2026-01-15T13:00:00.000Z", "5000", "recording_started", ""],
            ["2026-01-15T13:00:00.000Z", "5000", "packet_version", "version=3"],
            [
                "2026-01-15T13:00:01.000Z",
                "5100",
                "can_health",
                "state=errorWarning;tec=100;rec=50;bus_off=0;flags=1",
            ],
            [
                "2026-01-15T13:00:02.000Z",
                "5200",
                "can_health",
                "state=busOff;tec=255;rec=255;bus_off=1;flags=1",
            ],
            ["2026-01-15T13:00:02.000Z", "5200", "did_unanswered", "count=2;delta=2"],
            [
                "2026-01-15T13:00:03.000Z",
                "5300",
                "can_health",
                "state=stopped;tec=255;rec=255;bus_off=1;flags=5",
            ],
            ["2026-01-15T13:00:04.000Z", "5400", "recording_stopped", ""],
        ],
    )

    make_v3_imu_csv(out_dir / "imu.csv")

    meta = {
        "session_id": V3_SESSION_ID,
        "created_utc": "2026-01-15T13:00:00.000Z",
        "rider_name": "test-rider",
        "rider_weight_kg": 75,
        "extra_load_kg": 5,
        "ambient_temp_c": 20,
        "weather": "dry",
        "tire_pressure_front_bar": 2.2,
        "tire_pressure_rear_bar": 2.5,
        "fuel_level": 0.6,
        "vehicle_config": "stock",
        "condition_label": "fault:cornering-can-health",
        "route_type": "closed course",
        "note": "v3 fixture: CAN health changes, one out-of-range coolant reading, "
        "IMU block gap wrapping sample_index",
        "app_version": "2.0.0",
        "ble_schema_version": 3,
        "device_name": "CL250-fixture",
        "imu_block_version": 1,
        "requested_mtu": 185,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    lost = sum(seqs[i] - seqs[i - 1] - 1 for i in range(1, len(seqs)))
    summary = {
        "duration_ms": 5000 + len(seqs) * 100,
        "packet_count": len(seqs),
        "lost_count": lost,
        "loss_percent": round(lost / (lost + len(seqs)) * 100, 4),
        "decode_error_count": 0,
        "disconnect_count": 0,
        "first_packet_utc": "2026-01-15T13:00:00.000Z",
        "last_packet_utc": f"2026-01-15T13:00:{len(seqs):02d}.000Z",
        "packet_versions": {"3": len(seqs)},
        "imu_block_count": 2,
        "imu_sample_count": 10,
        "imu_missing_samples": 4,
        "imu_loss_percent": round(4 / 14 * 100, 4),
        "mtu": 185,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _v3_telemetry_header() -> list[str]:
    return _v2_telemetry_header() + [
        "packet_version",
        "device_time_ms",
        "rpm_age_ms",
        "speed_age_ms",
        "coolant_age_ms",
        "tps_age_ms",
        "battery_age_ms",
        "imu_active",
        "can_bus_state",
        "can_tec",
        "can_rec",
        "can_bus_off_count",
        "unanswered_did_count",
        "can_flags",
    ]


IMU_FLAG_BITS = SCHEMA["imuBlock"]["flags"]["bits"]


def make_v3_imu_csv(path: Path) -> None:
    """Two 5-sample blocks; block B starts after a 4-sample gap that wraps the
    uint16 sample_index counter (65530..65534, gap, then 3..7). Block B also
    carries `sensorReconfigured` (the node re-initialized the IMU after
    losing its configuration), to exercise the report's IMU flag counters."""
    accel_lsb = SCHEMA["imuBlock"]["scale"]["accel"]["lsbPerUnit"]
    gyro_lsb = SCHEMA["imuBlock"]["scale"]["gyro"]["lsbPerUnit"]
    sample_period_ms = SCHEMA["imuBlock"]["samplePeriodMs"]

    rows = []
    blocks = [
        {
            "block_seq": 0,
            "first_sample_index": 65530,
            "device_time_ms": 1000,
            "missing": 0,
            "flags": set(),
        },
        {
            "block_seq": 1,
            "first_sample_index": 3,
            "device_time_ms": 1090,
            "missing": 4,
            "flags": {"sensorReconfigured"},
        },
    ]
    for block in blocks:
        block_flags_value = bits_to_int(IMU_FLAG_BITS, block["flags"])
        for i in range(5):
            sample_index = (block["first_sample_index"] + i) % 65536
            device_time_ms = block["device_time_ms"] + i * sample_period_ms
            ax_raw, ay_raw, az_raw = 100 + i, -50 + i, 4096
            gx_raw, gy_raw, gz_raw = 10 + i, -10 - i, 5
            rows.append(
                [
                    f"2026-01-15T13:00:{block['block_seq']:02d}.{i:02d}0Z",
                    str(1000 + block["block_seq"] * 100 + i * sample_period_ms),
                    str(block["block_seq"]),
                    str(block_flags_value),
                    str(block["missing"] if i == 0 else 0),
                    str(sample_index),
                    str(device_time_ms),
                    str(ax_raw),
                    str(ay_raw),
                    str(az_raw),
                    str(gx_raw),
                    str(gy_raw),
                    str(gz_raw),
                    f"{ax_raw / accel_lsb:.6f}",
                    f"{ay_raw / accel_lsb:.6f}",
                    f"{az_raw / accel_lsb:.6f}",
                    f"{gx_raw / gyro_lsb:.6f}",
                    f"{gy_raw / gyro_lsb:.6f}",
                    f"{gz_raw / gyro_lsb:.6f}",
                ]
            )

    write_csv(
        path,
        [
            "rx_utc_iso",
            "rx_mono_ms",
            "block_seq",
            "block_flags",
            "block_missing_samples",
            "sample_index",
            "device_time_ms",
            "ax_raw",
            "ay_raw",
            "az_raw",
            "gx_raw",
            "gy_raw",
            "gz_raw",
            "ax_g",
            "ay_g",
            "az_g",
            "gx_dps",
            "gy_dps",
            "gz_dps",
        ],
        rows,
    )


# ---------------------------------------------------------------------------
# v4 session: the 43-column telemetry.csv. Row 0 is a version 3 packet (the 8
# tester columns empty, as the app writes them), rows 1..6 are version 4
# packets with a rotating round-trip DID. DIDs come from the generated defs.
# ---------------------------------------------------------------------------

V4_SESSION_ID = "20260115-140000-e5f6"
SATURATED_U32 = 0xFFFFFFFF
RTT_DID_A = vehicle_cl250.DIDS["ENGINE_SPEED"][0]
RTT_DID_B = vehicle_cl250.DIDS["VEHICLE_SPEED"][0]
RTT_DID_C = vehicle_cl250.DIDS["THROTTLE_POS"][0]

# One entry per row: (version, tester statistics by schema field name).
V4_ROWS: list[tuple[int, dict[str, int]]] = [
    (3, {}),
    (
        4,
        dict(
            stepGapMaxMs=12,
            stepGapOverCount=0,
            rttDid=RTT_DID_A,
            rttMinMs=5,
            rttMaxMs=9,
            rttSumMs=70,
            rttCount=10,
            rttNrc78Count=0,
        ),
    ),
    (
        4,
        dict(
            stepGapMaxMs=12,
            stepGapOverCount=0,
            rttDid=RTT_DID_B,
            rttMinMs=3,
            rttMaxMs=8,
            rttSumMs=50,
            rttCount=10,
            rttNrc78Count=0,
        ),
    ),
    (
        4,
        dict(
            stepGapMaxMs=15,
            stepGapOverCount=1,
            rttDid=RTT_DID_A,
            rttMinMs=4,
            rttMaxMs=11,
            rttSumMs=100,
            rttCount=14,
            rttNrc78Count=1,
        ),
    ),
    # DID C: only NRC 0x78 answers so far, no round-trip sample.
    (
        4,
        dict(
            stepGapMaxMs=15,
            stepGapOverCount=1,
            rttDid=RTT_DID_C,
            rttMinMs=65535,
            rttMaxMs=0,
            rttSumMs=0,
            rttCount=0,
            rttNrc78Count=3,
        ),
    ),
    # DID B again, with the sum and the count saturated (no average).
    (
        4,
        dict(
            stepGapMaxMs=20,
            stepGapOverCount=2,
            rttDid=RTT_DID_B,
            rttMinMs=3,
            rttMaxMs=60000,
            rttSumMs=SATURATED_U32,
            rttCount=SATURATED_U32,
            rttNrc78Count=0,
        ),
    ),
    # No record in this packet.
    (
        4,
        dict(
            stepGapMaxMs=20,
            stepGapOverCount=2,
            rttDid=0,
            rttMinMs=65535,
            rttMaxMs=0,
            rttSumMs=0,
            rttCount=0,
            rttNrc78Count=0,
        ),
    ),
]


def make_v4_fixture() -> None:
    out_dir = FIXTURES_DIR / "v4_session"
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = [20 + i for i in range(len(V4_ROWS))]
    valid = {
        "rpmValid",
        "speedValid",
        "coolantTempValid",
        "throttlePosValid",
        "batteryVoltValid",
        "ecuPresent",
    }
    running = CAN_STATE_BY_NAME["running"]
    can_flags = bits_to_int(CAN_FLAG_BITS, {"pollerEnabled"})

    rows = []
    for i, (version, tester) in enumerate(V4_ROWS):
        values = {
            "version": version,
            "seq": seqs[i],
            "deviceTimeMs": 9_000 + i * 100,
            "rpm": 2000 + i,
            "speed": 40,
            "coolantTemp": 80,
            "throttlePos": 10,
            "batteryVolt": 12500,
            "leanAngle": -32768,
            "maxLeanRight": -32768,
            "maxLeanLeft": -32768,
            "flags": bits_to_int(FLAG_BITS, valid),
            "rpmAgeMs": 10,
            "speedAgeMs": 15,
            "coolantTempAgeMs": 200,
            "throttlePosAgeMs": 20,
            "batteryVoltAgeMs": 400,
            "canBusState": running,
            "canTxErrorCount": 0,
            "canRxErrorCount": 0,
            "canBusOffCount": 0,
            "unansweredDidCount": 0,
            "canFlags": can_flags,
            **tester,
        }
        raw = ble_schema.pack_fields(
            values, defs_ble.telemetry_fields(version), defs_ble.TOTAL_BYTES_BY_VERSION[version]
        )
        tester_cells = [
            str(tester[ble_schema_field]) if tester else ""
            for ble_schema_field in _v4_schema_fields()
        ]
        rows.append(
            [
                f"2026-01-15T14:00:{i:02d}.000Z",
                str(9_000 + i * 100),
                str(seqs[i]),
                "0",
                raw.hex(),
                str(values["rpm"]),
                "40",
                "80",
                "10",
                "12.500",
                "",
                "",
                "",
                "1",
                "1",
                "1",
                "1",
                "1",
                "",
                "1",
                "",
                str(version),
                str(values["deviceTimeMs"]),
                "10",
                "15",
                "200",
                "20",
                "400",
                "0",
                str(running),
                "0",
                "0",
                "0",
                "0",
                str(can_flags),
                *tester_cells,
            ]
        )

    write_csv(out_dir / "telemetry.csv", _v3_telemetry_header() + V4_EXTRA_HEADER, rows)
    write_csv(
        out_dir / "events.csv",
        ["rx_utc_iso", "rx_mono_ms", "event", "detail"],
        [
            ["2026-01-15T14:00:00.000Z", "9000", "recording_started", ""],
            ["2026-01-15T14:00:00.000Z", "9000", "packet_version", "version=3"],
            ["2026-01-15T14:00:01.000Z", "9100", "packet_version", "version=4"],
            ["2026-01-15T14:00:07.000Z", "9700", "recording_stopped", ""],
        ],
    )
    meta = {
        "session_id": V4_SESSION_ID,
        "created_utc": "2026-01-15T14:00:00.000Z",
        "rider_name": "test-rider",
        "rider_weight_kg": 75,
        "extra_load_kg": 0,
        "ambient_temp_c": 20,
        "weather": "dry",
        "tire_pressure_front_bar": 2.2,
        "tire_pressure_rear_bar": 2.5,
        "fuel_level": 0.5,
        "vehicle_config": "stock",
        "condition_label": "healthy",
        "route_type": "closed course",
        "note": "v4 fixture: one version 3 packet then version 4 tester statistics",
        "app_version": "2.1.0",
        "ble_schema_version": 4,
        "device_name": "CL250-fixture",
        "imu_block_version": 1,
        "requested_mtu": 185,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    summary = {
        "duration_ms": 9_000 + len(rows) * 100,
        "packet_count": len(rows),
        "lost_count": 0,
        "loss_percent": 0.0,
        "decode_error_count": 0,
        "disconnect_count": 0,
        "first_packet_utc": "2026-01-15T14:00:00.000Z",
        "last_packet_utc": f"2026-01-15T14:00:{len(rows) - 1:02d}.000Z",
        "packet_versions": {"3": 1, "4": len(rows) - 1},
        "imu_block_count": 0,
        "imu_sample_count": 0,
        "imu_missing_samples": 0,
        "imu_loss_percent": 0.0,
        "mtu": 185,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _v4_schema_fields() -> list[str]:
    """Schema field names of the version 4 tester columns, in telemetry.csv order."""
    return [f["name"] for f in ble_schema.tester_stat_fields(SCHEMA)]


# ---------------------------------------------------------------------------
# v4 + GPS session (D-060): the v4 telemetry plus gps.csv. Scenario: no fix,
# 2-D fix, then 3-D fix; a 2-block seq gap (253, 254 lost or MTU-skipped); the
# 255 -> 0 seq wrap (no gap); one parseError and one uartOverflow block; a
# gnssDeadReckoning block; a fix3d block without gnssFixOk (not usable).
# ---------------------------------------------------------------------------

V4_GPS_SESSION_ID = "20260115-150000-a7b8"
GPS = SCHEMA["gpsBlock"]
GPS_FIX = {v["name"]: v["value"] for v in GPS["fixType"]["values"]}
GPS_FLAG_BITS = GPS["flags"]["bits"]
GPS_SPEED_LSB = GPS["scale"]["speed"]["lsbPerUnit"]
GPS_HEADING_LSB = GPS["scale"]["heading"]["lsbPerUnit"]

# (seq, deviceTimeMs, groundSpeed mm/s, headingOfMotion 1e-5 deg, speedAccuracy,
#  headingAccuracy, fixType name, numSv, set flag names)
V4_GPS_BLOCKS = [
    (250, 9_000, 0, 0, 5_000, 18_000_000, "noFix", 0, set()),
    (251, 9_100, 150, 0, 2_500, 9_000_000, "fix2d", 5, set()),
    (252, 9_200, 11_111, 9_012_345, 300, 50_000, "fix3d", 9, {"gnssFixOk"}),
    (255, 9_500, 11_200, 9_050_000, 280, 48_000, "fix3d", 10, {"gnssFixOk", "parseError"}),
    (0, 9_600, 11_250, 9_100_000, 280, 47_000, "fix3d", 10, {"gnssFixOk", "uartOverflow"}),
    (1, 9_700, 11_300, 9_150_000, 290, 46_000, "gnssDeadReckoning", 10, {"gnssFixOk"}),
    (2, 9_800, 11_310, 9_160_000, 900, 300_000, "fix3d", 4, set()),
]


def make_v4_gps_fixture() -> None:
    out_dir = FIXTURES_DIR / "v4_gps_session"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = FIXTURES_DIR / "v4_session"
    (out_dir / "telemetry.csv").write_text((src_dir / "telemetry.csv").read_text())

    rows = []
    gap_events = []
    prev_seq = None
    for i, (seq, t_ms, speed, heading, s_acc, h_acc, fix, num_sv, flag_names) in enumerate(
        V4_GPS_BLOCKS
    ):
        flags = bits_to_int(GPS_FLAG_BITS, flag_names)
        values = {
            "version": GPS["version"],
            "seq": seq,
            "deviceTimeMs": t_ms,
            "groundSpeed": speed,
            "headingOfMotion": heading,
            "speedAccuracy": s_acc,
            "headingAccuracy": h_acc,
            "fixType": GPS_FIX[fix],
            "numSv": num_sv,
            "flags": flags,
            "reserved": 0,
        }
        raw = ble_schema.pack_fields(values, defs_ble.gps_fields(), defs_ble.GPS_TOTAL_BYTES)
        lost = 0 if prev_seq is None else (seq - prev_seq - 1) % 256
        prev_seq = seq
        rx_utc = f"2026-01-15T15:00:{i:02d}.000Z"
        if lost:
            gap_events.append([rx_utc, str(t_ms), "gps_gap", f"missing={lost};seq={seq}"])
        rows.append(
            [
                rx_utc,
                str(t_ms),
                str(seq),
                str(lost),
                raw.hex(),
                str(t_ms),
                str(speed),
                str(heading),
                str(s_acc),
                str(h_acc),
                str(GPS_FIX[fix]),
                str(num_sv),
                str(flags),
                repr(speed / GPS_SPEED_LSB),
                repr(heading / GPS_HEADING_LSB),
                repr(s_acc / GPS_SPEED_LSB),
                repr(h_acc / GPS_HEADING_LSB),
                "1" if "gnssFixOk" in flag_names else "0",
                "1" if "parseError" in flag_names else "0",
                "1" if "uartOverflow" in flag_names else "0",
            ]
        )
    write_csv(out_dir / "gps.csv", GPS_HEADER, rows)

    write_csv(
        out_dir / "events.csv",
        ["rx_utc_iso", "rx_mono_ms", "event", "detail"],
        [
            ["2026-01-15T15:00:00.000Z", "9000", "recording_started", ""],
            ["2026-01-15T15:00:00.000Z", "9000", "packet_version", "version=3"],
            ["2026-01-15T15:00:01.000Z", "9100", "packet_version", "version=4"],
            *gap_events,
            ["2026-01-15T15:00:07.000Z", "9700", "recording_stopped", ""],
        ],
    )

    meta = json.loads((src_dir / "meta.json").read_text())
    meta.update(
        {
            "session_id": V4_GPS_SESSION_ID,
            "created_utc": "2026-01-15T15:00:00.000Z",
            "note": "v4 + GPS fixture: speed and heading blocks (D-060), no position",
            "app_version": "2.2.0",
            "gps_block_version": GPS["version"],
        }
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    summary = json.loads((src_dir / "summary.json").read_text())
    summary.update(
        {
            "first_packet_utc": "2026-01-15T15:00:00.000Z",
            "last_packet_utc": "2026-01-15T15:00:06.000Z",
            "gps_block_count": len(rows),
            "gps_lost_or_skipped_blocks": 2,
            "gps_lost_or_skipped_percent": round(2 / (len(rows) + 2) * 100, 4),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


# ---------------------------------------------------------------------------
# v4 + vehicle signals session (D-066 item 4): the v4 telemetry plus signals.csv.
# Scenario: a sender DID table of 8 DIDs sent as blocks of 4 records, so each table
# pass is two blocks. The first pass (seq 10, 11) has valid polled DIDs, a stale one,
# a candidate DID (ENGINE_LOAD, D-066 item 3) in state valid, a negative response
# (FUEL_LEVEL, NRC 0x31) and a DID missing from the catalog (a newer defs). Seq 12 and
# 13 are lost; the second pass (seq 14, 15) has a skipped DID and new values.
# ---------------------------------------------------------------------------

V4_VSIG_SESSION_ID = "20260115-160000-c9d0"
VSIG = SCHEMA["vehicleSignalBlock"]
VSIG_STATE = {v["name"]: v["value"] for v in VSIG["state"]["values"]}
# A DID no catalog contains: what a sender built from a newer defs would add.
VSIG_UNKNOWN_DID = max(vehicle_catalog.BY_DID) + 1
VSIG_NRC_OUT_OF_RANGE = 0x31


def _vsig_did(name: str) -> int:
    return vehicle_catalog.SIGNALS[name]["did"]


# The sender's DID table, in table order.
VSIG_TABLE = [
    _vsig_did("ENGINE_SPEED"),
    _vsig_did("VEHICLE_SPEED"),
    _vsig_did("THROTTLE_POS"),
    _vsig_did("COOLANT_TEMP"),
    _vsig_did("BATTERY_VOLTAGE"),
    _vsig_did("ENGINE_LOAD"),
    _vsig_did("FUEL_LEVEL"),
    VSIG_UNKNOWN_DID,
]

# (block seq, rx_mono_ms, {DID: (state name, age_ms, raw)}), records in table order.
V4_VSIG_BLOCKS = [
    (
        10,
        9_100,
        {
            _vsig_did("ENGINE_SPEED"): ("valid", 40, 12_000),
            _vsig_did("VEHICLE_SPEED"): ("valid", 60, 40),
            _vsig_did("THROTTLE_POS"): ("valid", 120, 26),
            _vsig_did("COOLANT_TEMP"): ("stale", 3_000, 120),
        },
    ),
    (
        11,
        9_200,
        {
            _vsig_did("BATTERY_VOLTAGE"): ("valid", 500, 12_500),
            _vsig_did("ENGINE_LOAD"): ("valid", 90, 102),
            _vsig_did("FUEL_LEVEL"): ("negativeResponse", 65_535, VSIG_NRC_OUT_OF_RANGE),
            VSIG_UNKNOWN_DID: ("valid", 70, 77),
        },
    ),
    (
        14,
        9_500,
        {
            _vsig_did("ENGINE_SPEED"): ("valid", 30, 12_400),
            _vsig_did("VEHICLE_SPEED"): ("valid", 50, 42),
            _vsig_did("THROTTLE_POS"): ("skipped", 65_535, 0),
            _vsig_did("COOLANT_TEMP"): ("valid", 700, 125),
        },
    ),
    (
        15,
        9_600,
        {
            _vsig_did("BATTERY_VOLTAGE"): ("valid", 300, 12_480),
            _vsig_did("ENGINE_LOAD"): ("valid", 100, 128),
            _vsig_did("FUEL_LEVEL"): ("negativeResponse", 65_535, VSIG_NRC_OUT_OF_RANGE),
            VSIG_UNKNOWN_DID: ("valid", 80, 78),
        },
    ),
]


def vsig_row_cells(
    rx_utc: str,
    rx_mono_ms: int,
    seq: int,
    lost: int,
    first_record: int,
    did: int,
    state: str,
    age_ms: int,
    raw: int,
    first_in_block: bool,
    table_len: int = len(VSIG_TABLE),
) -> list[str]:
    """One signals.csv row; `signal` / `value` as the app writes them (catalog decode)."""
    name = vehicle_catalog.BY_DID.get(did)
    value = ""
    if name and VSIG_STATE[state] in (VSIG_STATE["valid"], VSIG_STATE["stale"]):
        entry = vehicle_catalog.SIGNALS[name]
        value = repr(vehicle_catalog.decode(name, raw.to_bytes(entry["length"], "big")))
    return [
        rx_utc,
        str(rx_mono_ms),
        str(seq),
        str(lost if first_in_block else 0),
        str(rx_mono_ms),
        str(table_len),
        str(first_record),
        str(did),
        str(VSIG_STATE[state]),
        str(age_ms),
        str(raw),
        name or "",
        value,
    ]


def make_v4_vsig_fixture() -> None:
    out_dir = FIXTURES_DIR / "v4_vsig_session"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = FIXTURES_DIR / "v4_session"
    (out_dir / "telemetry.csv").write_text((src_dir / "telemetry.csv").read_text())

    rows = []
    gap_events = []
    prev_seq = None
    for i, (seq, rx_mono_ms, records) in enumerate(V4_VSIG_BLOCKS):
        lost = 0 if prev_seq is None else (seq - prev_seq - 1) % 256
        prev_seq = seq
        rx_utc = f"2026-01-15T16:00:{i + 1:02d}.000Z"
        if lost:
            gap_events.append([rx_utc, str(rx_mono_ms), "vsig_gap", f"missing={lost};seq={seq}"])
        first_record = VSIG_TABLE.index(next(iter(records)))
        for j, (did, (state, age_ms, raw)) in enumerate(records.items()):
            rows.append(
                vsig_row_cells(
                    rx_utc, rx_mono_ms, seq, lost, first_record, did, state, age_ms, raw, j == 0
                )
            )
    write_csv(out_dir / "signals.csv", SIGNALS_HEADER, rows)

    write_csv(
        out_dir / "events.csv",
        ["rx_utc_iso", "rx_mono_ms", "event", "detail"],
        [
            ["2026-01-15T16:00:00.000Z", "9000", "recording_started", ""],
            ["2026-01-15T16:00:00.000Z", "9000", "packet_version", "version=3"],
            ["2026-01-15T16:00:01.000Z", "9100", "packet_version", "version=4"],
            *gap_events,
            ["2026-01-15T16:00:07.000Z", "9700", "recording_stopped", ""],
        ],
    )

    meta = json.loads((src_dir / "meta.json").read_text())
    meta.update(
        {
            "session_id": V4_VSIG_SESSION_ID,
            "created_utc": "2026-01-15T16:00:00.000Z",
            "note": "v4 + vehicle signals fixture: D-066 item 4 blocks, synthetic values",
            "app_version": "2.3.0",
            "vehicle_signal_block_version": VSIG["version"],
        }
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    summary = json.loads((src_dir / "summary.json").read_text())
    summary.update(
        {
            "first_packet_utc": "2026-01-15T16:00:00.000Z",
            "last_packet_utc": "2026-01-15T16:00:06.000Z",
            "vehicle_signal_block_count": len(V4_VSIG_BLOCKS),
            "vehicle_signal_record_count": len(rows),
            "vehicle_signal_lost_blocks": 2,
            "vehicle_signal_lost_percent": round(2 / (len(V4_VSIG_BLOCKS) + 2) * 100, 4),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    make_v2_fixture()
    make_v3_fixture()
    make_v4_fixture()
    make_v4_gps_fixture()
    make_v4_vsig_fixture()
    print(f"Wrote fixtures to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
