from __future__ import annotations

from pathlib import Path

import pytest

from vehicle_server import ble_schema
from vehicle_server.config import Settings
from vehicle_server.defs import ble as defs_ble
from vehicle_server.ingest import ingest_path
from vehicle_server.report import render_text


def test_v2_report_exact_loss_and_stale_value(settings: Settings, fixtures_dir: Path):
    result = ingest_path(fixtures_dir / "v2_session", settings)
    report = result.report

    assert report["packet_count"] == 7
    assert report["loss"]["lost_count"] == 2
    assert report["loss"]["loss_percent"] == pytest.approx(2 / 9 * 100)
    assert report["telemetry_layout"] == "v2"
    assert report["consistency"]["decode_error_rows"] == []
    assert report["consistency"]["mismatch_rows"] == []

    battery = report["valid_ratios"]["battery_voltage"]
    assert battery["total_count"] == 7
    assert battery["valid_count"] == 6
    assert battery["valid_percent"] == pytest.approx(6 / 7 * 100)


def test_v3_report_exact_can_health_and_imu_gap(settings: Settings, fixtures_dir: Path):
    result = ingest_path(fixtures_dir / "v3_session", settings)
    report = result.report

    assert report["packet_count"] == 5
    assert report["loss"]["lost_count"] == 1
    assert report["loss"]["loss_percent"] == pytest.approx(1 / 6 * 100)

    coolant = report["range_checks"]["coolant_temp"]
    assert coolant["out_of_range_count"] == 1
    assert coolant["checked_count"] == 5

    can = report["can_health"]
    assert can["max_tec"] == 255
    assert can["max_rec"] == 255
    assert can["max_bus_off_count"] == 1
    assert can["max_unanswered_did_count"] == 2
    assert can["latched_bus_off"] is True
    assert can["latched_foreign_tester"] is False

    imu = report["imu"]
    assert imu["sample_count"] == 10
    assert imu["missing_samples"] == 4
    assert imu["loss_percent"] == pytest.approx(4 / 14 * 100)
    assert imu["scale_mismatch_rows"] == []
    assert imu["block_count"] == 2
    assert imu["blocks_with_sensor_reconfigured"] == 1
    assert imu["blocks_with_device_overflow"] == 0
    assert imu["blocks_with_read_error"] == 0

    assert report["status"] == "warn"
    assert "one or more signals had out-of-range values" in report["reasons"]
    assert "4 missing IMU sample(s)" in report["reasons"]
    assert any("sensor reconfiguration" in reason for reason in report["reasons"])


def test_report_fails_on_decode_error(settings: Settings, fixtures_dir: Path, tmp_path: Path):
    from .conftest import copy_fixture

    session_dir = copy_fixture(fixtures_dir / "v2_session", tmp_path / "session")
    telemetry_path = session_dir / "telemetry.csv"
    lines = telemetry_path.read_text().splitlines()
    # Corrupt the raw_hex of the first data row so it fails to decode.
    header, first, *rest = lines
    fields = first.split(",")
    fields[4] = "00"  # too short to be a valid v2/v3 payload
    lines = [header, ",".join(fields), *rest]
    telemetry_path.write_text("\n".join(lines) + "\n")

    result = ingest_path(session_dir, settings)
    assert result.report["status"] == "fail"
    assert len(result.report["consistency"]["decode_error_rows"]) == 1


def test_v4_report_tester_stats(settings: Settings, fixtures_dir: Path):
    result = ingest_path(fixtures_dir / "v4_session", settings)
    report = result.report

    assert report["telemetry_layout"] == "v4"
    assert report["packet_versions"] == {"3": 1, "4": 6}
    assert report["schema_versions_seen"] == [3, 4]
    assert report["consistency"] == {"decode_error_rows": [], "mismatch_rows": []}
    assert report["status"] == "ok"

    stats = report["tester_stats"]
    assert stats["step_gap_max_ms"] == 20
    assert stats["step_gap_over_count"] == 2
    assert list(stats["rtt"]) == ["0xF40C", "0xF40D", "0xF411"]

    engine = stats["rtt"]["0xF40C"]
    assert engine["name"] == "ENGINE_SPEED"
    assert (engine["min_ms"], engine["max_ms"]) == (4, 11)
    assert engine["avg_ms"] == pytest.approx(100 / 14)
    assert (engine["count"], engine["nrc78_count"]) == (14, 1)

    # Saturated sum and count: no average.
    speed = stats["rtt"]["0xF40D"]
    assert speed["avg_ms"] is None
    assert speed["count"] == 0xFFFFFFFF
    assert (speed["min_ms"], speed["max_ms"]) == (3, 60000)

    # Only NRC 0x78 answers: no sample, so no min/max/avg.
    throttle = stats["rtt"]["0xF411"]
    assert (throttle["min_ms"], throttle["max_ms"], throttle["avg_ms"]) == (None, None, None)
    assert (throttle["count"], throttle["nrc78_count"]) == (0, 3)


def test_v4_report_text_shows_tester_stats_with_hex_dids(settings: Settings, fixtures_dir: Path):
    result = ingest_path(fixtures_dir / "v4_session", settings)
    text = render_text(result.report)
    assert "tester stats: step gap max 20 ms, over client_step_max_ms 2" in text
    assert "0xF40C (ENGINE_SPEED): rtt min 4 ms max 11 ms avg 7.1 ms, 14 sample(s)" in text
    assert "NRC 0x78 1" in text
    assert "0xF40D (VEHICLE_SPEED): rtt min 3 ms max 60000 ms avg n/a" in text
    assert "0xF411 (THROTTLE_POS): rtt min n/a max n/a avg n/a, 0 sample(s), NRC 0x78 3" in text


def test_report_without_tester_columns_has_no_tester_section(
    settings: Settings, fixtures_dir: Path
):
    for name in ("v2_session", "v3_session"):
        result = ingest_path(fixtures_dir / name, settings)
        assert result.report["tester_stats"] is None
        assert "tester stats" not in render_text(result.report)


def test_v4_app_columns_disagreeing_with_raw_hex_are_flagged(
    settings: Settings, fixtures_dir: Path, tmp_path: Path
):
    from .conftest import copy_fixture

    session_dir = copy_fixture(fixtures_dir / "v4_session", tmp_path / "session")
    telemetry_path = session_dir / "telemetry.csv"
    header, *rows = telemetry_path.read_text().splitlines()
    columns = header.split(",")
    fields = rows[1].split(",")  # first version 4 row
    fields[columns.index("rtt_max_ms")] = "999"  # raw_hex says 9
    rows[1] = ",".join(fields)
    telemetry_path.write_text("\n".join([header, *rows]) + "\n")

    result = ingest_path(session_dir, settings)
    assert result.report["consistency"]["mismatch_rows"] == [1]
    assert result.report["status"] == "warn"
    # The authoritative value (raw_hex) is what the report uses.
    assert result.report["tester_stats"]["rtt"]["0xF40C"]["max_ms"] == 11


SCHEMA = ble_schema.load_default_schema()
ECU_PRESENT_BIT = next(bit["bit"] for bit in SCHEMA["flags"]["bits"] if bit["name"] == "ecuPresent")


def _rewrite_ecu_present(session_dir: Path, states: list[bool]) -> None:
    telemetry_path = session_dir / "telemetry.csv"
    header, *rows = telemetry_path.read_text().splitlines()
    columns = header.split(",")
    raw_index = columns.index("raw_hex")
    field_defs = defs_ble.telemetry_fields(3)
    total_bytes = defs_ble.TOTAL_BYTES_BY_VERSION[3]
    for i, (row, present) in enumerate(zip(rows, states, strict=True)):
        fields = row.split(",")
        values = ble_schema.unpack_fields(bytes.fromhex(fields[raw_index]), field_defs)
        if present:
            values["flags"] |= 1 << ECU_PRESENT_BIT
        else:
            values["flags"] &= ~(1 << ECU_PRESENT_BIT)
        fields[raw_index] = ble_schema.pack_fields(values, field_defs, total_bytes).hex()
        rows[i] = ",".join(fields)
    telemetry_path.write_text("\n".join([header, *rows]) + "\n")


def test_v3_ecu_present_on_all_can_health_rows(settings: Settings, fixtures_dir: Path):
    result = ingest_path(fixtures_dir / "v3_session", settings)
    can = result.report["can_health"]

    assert can["ecu_present_rows"] == 5
    assert can["ecu_absent_rows"] == 0
    assert can["ecu_present_last"] is True
    assert "ecu present=5/5 absent=0 last=True" in render_text(result.report)


def test_v3_ecu_absent_on_all_can_health_rows(
    settings: Settings, fixtures_dir: Path, tmp_path: Path
):
    from .conftest import copy_fixture

    session_dir = copy_fixture(fixtures_dir / "v3_session", tmp_path / "session")
    _rewrite_ecu_present(session_dir, [False] * 5)

    result = ingest_path(session_dir, settings)
    can = result.report["can_health"]

    assert can["ecu_present_rows"] == 0
    assert can["ecu_absent_rows"] == 5
    assert can["ecu_present_last"] is False
    assert "ecu present=0/5 absent=5 last=False" in render_text(result.report)


def test_v3_ecu_present_then_absent(settings: Settings, fixtures_dir: Path, tmp_path: Path):
    from .conftest import copy_fixture

    session_dir = copy_fixture(fixtures_dir / "v3_session", tmp_path / "session")
    _rewrite_ecu_present(session_dir, [True, True, True, False, False])

    result = ingest_path(session_dir, settings)
    can = result.report["can_health"]

    assert can["ecu_present_rows"] == 3
    assert can["ecu_absent_rows"] == 2
    assert can["ecu_present_last"] is False
    assert "ecu present=3/5 absent=2 last=False" in render_text(result.report)


def test_v2_has_no_can_health_and_ecu_keys_are_absent(settings: Settings, fixtures_dir: Path):
    result = ingest_path(fixtures_dir / "v2_session", settings)

    assert result.report["can_health"] is None
    assert "CAN health" not in render_text(result.report)


def test_old_report_text_marks_ecu_presence_unknown(settings: Settings, fixtures_dir: Path):
    result = ingest_path(fixtures_dir / "v3_session", settings)
    old_report = dict(result.report)
    old_report["can_health"] = dict(result.report["can_health"])
    for key in ("ecu_present_rows", "ecu_absent_rows", "ecu_present_last"):
        del old_report["can_health"][key]

    assert "ecu present=n/a (old report)" in render_text(old_report)
