"""Derived signals through ingest, the report block, the index and the CLI (synthetic data)."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from vehicle_server import ble_schema, cli, derived, index
from vehicle_server.config import Settings
from vehicle_server.decode import IMU_HEADER
from vehicle_server.defs import vehicle_derived
from vehicle_server.derived import engine
from vehicle_server.derived.inputs import SessionInputs, _to_series, _unwrap, resolve_sensor
from vehicle_server.derived.summary import NOT_IMPLEMENTED_REASONS
from vehicle_server.ingest import ingest_path, rederive_session
from vehicle_server.report import render_text

from .conftest import copy_fixture
from .make_fixtures import write_csv

SCHEMA = ble_schema.load_default_schema()
ACCEL_LSB = SCHEMA["imuBlock"]["scale"]["accel"]["lsbPerUnit"]
GYRO_LSB = SCHEMA["imuBlock"]["scale"]["gyro"]["lsbPerUnit"]
IMU_SAMPLES = 100
IMU_START_MS = 9000


def _write_imu_csv(path: Path) -> None:
    """100 IMU samples (10 ms apart) over the time span of the v4_vsig_session signals:
    constant 0.1 g longitudinal and 5 deg/s yaw."""
    rows = []
    for i in range(IMU_SAMPLES):
        ax_raw, gz_raw = round(0.1 * ACCEL_LSB), round(5 * GYRO_LSB)
        rows.append(
            [
                f"2026-01-15T16:00:{i // 50:02d}.000Z",
                str(IMU_START_MS + i * 10),
                str(i // 5),
                "0",
                "0",
                str(i),
                str(IMU_START_MS + i * 10),
                str(ax_raw),
                "0",
                str(ACCEL_LSB),
                "0",
                "0",
                str(gz_raw),
                f"{ax_raw / ACCEL_LSB:.6f}",
                "0.000000",
                f"{ACCEL_LSB / ACCEL_LSB:.6f}",
                "0.000000",
                "0.000000",
                f"{gz_raw / GYRO_LSB:.6f}",
            ]
        )
    write_csv(path, IMU_HEADER, rows)


@pytest.fixture
def imu_vsig_session(fixtures_dir: Path, tmp_path: Path) -> Path:
    """v4_vsig_session (signals.csv) plus an imu.csv on the same time axis."""
    session = copy_fixture(fixtures_dir / "v4_vsig_session", tmp_path / "imu_vsig")
    _write_imu_csv(session / "imu.csv")
    return session


def _derived_table(settings: Settings, session_id: str):
    return pq.read_table(settings.sessions_dir / session_id / "parquet" / "derived.parquet")


def _rows(table, signal: str) -> list[dict]:
    return [r for r in table.to_pylist() if r["signal"] == signal]


# --- ingest -------------------------------------------------------------------------------


def test_ingest_writes_derived_parquet_and_report_block(settings: Settings, imu_vsig_session: Path):
    result = ingest_path(imu_vsig_session, settings)
    table = _derived_table(settings, result.session_id)

    assert table.column_names == [
        "session_id",
        "device_time_ms",
        "signal",
        "value",
        "unit",
        "valid",
        "flag",
    ]
    assert set(table["session_id"].to_pylist()) == {result.session_id}
    assert str(table.schema.field("value").type) == "double"

    # gearbox ratio: 3000 rpm at 40 km/h (rpm 12000 raw / 4, speed raw 40) -> gear 5
    ratio = [r for r in _rows(table, "gearbox_ratio_measured") if r["valid"]]
    assert ratio and ratio[0]["value"] == pytest.approx(1.18571, abs=1e-4)
    assert ratio[0]["unit"] == "ratio" and ratio[0]["flag"] is None
    gear = [r for r in _rows(table, "gear_estimate") if r["valid"]]
    assert gear[0]["value"] == 5.0

    # IMU signals: yaw rate is the 5 deg/s of the fixture, on the 20 ms grid of the defs
    yaw = _rows(table, "yaw_rate")
    assert [r["device_time_ms"] for r in yaw[:3]] == [9000, 9020, 9040]
    assert yaw[0]["value"] == pytest.approx(5.0, rel=5e-3)  # raw LSB quantization
    # lateral acceleration: v * omega at a valid speed sample (40 km/h, 5 deg/s)
    lateral = {r["device_time_ms"]: r for r in _rows(table, "lateral_accel")}
    assert lateral[9040]["value"] == pytest.approx(40 / 3.6 * math.radians(5.0), rel=5e-3)
    # the speed sample at 9040 is stale once its age exceeds the signal's stale_after_ms
    stale_t = 9040 + vehicle_derived.SIGNALS["lateral_accel"]["stale_after_ms"] + 10
    assert lateral[stale_t]["valid"] is False and lateral[stale_t]["flag"] == "stale_input"

    block = result.report["derived"]
    assert block["defs_version"] == result.report["defs_version"]
    assert block["rows"] == table.num_rows
    assert block["valid_rows"] == sum(table["valid"].to_pylist())
    signals = block["signals"]
    assert set(signals) == set(vehicle_derived.SIGNALS)
    assert signals["gearbox_ratio_measured"]["status"] == "computed"
    assert signals["gearbox_ratio_measured"]["rows"] == len(_rows(table, "gearbox_ratio_measured"))
    assert signals["engine_vibration_rms"]["status"] == "not_ready"
    assert signals["engine_vibration_rms"]["pending_inputs"]
    assert signals["speedometer_error"]["status"] == "not_implemented"
    assert signals["engine_torque"]["status"] == "missing_inputs"
    assert "did.REFERENCE_TORQUE" in signals["engine_torque"]["missing"]
    assert sum(block["summary"].values()) == len(vehicle_derived.SIGNALS)
    assert not any(s["status"] == "error" for s in signals.values())
    # nothing was computed for a signal that is not computed
    assert {r["signal"] for r in table.to_pylist()} == {
        n for n, s in signals.items() if s["status"] == "computed" and s["rows"]
    }

    on_disk = json.loads((settings.sessions_dir / result.session_id / "report.json").read_text())
    assert on_disk["derived"] == block
    assert f"derived signals: {block['rows']} row(s)" in render_text(result.report)


def test_index_has_derived_follows_the_file(settings: Settings, imu_vsig_session: Path):
    result = ingest_path(imu_vsig_session, settings)
    row = index.get_session(settings.index_db_path, result.session_id)
    assert row is not None and row.has_derived is True
    assert index.as_dict(row)["has_derived"] is True


def test_a_session_without_derivable_inputs_has_no_derived_parquet(
    settings: Settings, fixtures_dir: Path
):
    result = ingest_path(fixtures_dir / "v3_session", settings)  # imu only, 90 ms span
    row = index.get_session(settings.index_db_path, result.session_id)
    path = settings.sessions_dir / result.session_id / "parquet" / "derived.parquet"
    assert row is not None and row.has_derived is path.is_file()
    assert "error" not in result.report["derived"]


def test_ingest_survives_a_derive_failure(
    settings: Settings, imu_vsig_session: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reference = ingest_path(imu_vsig_session, Settings(tmp_path / "ref", "t", 100))

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(derived, "derive_session", boom)
    result = ingest_path(imu_vsig_session, settings)

    assert result.created is True
    assert result.report["status"] == reference.report["status"]
    assert result.report["derived"] == {"error": "RuntimeError: boom"}
    assert "derived signals: error: RuntimeError: boom" in render_text(result.report)
    session_dir = settings.sessions_dir / result.session_id
    assert not (session_dir / "parquet" / "derived.parquet").exists()
    assert (session_dir / "parquet" / "telemetry.parquet").is_file()
    on_disk = json.loads((session_dir / "report.json").read_text())
    assert on_disk["derived"] == {"error": "RuntimeError: boom"}
    row = index.get_session(settings.index_db_path, result.session_id)
    assert row is not None and row.status == reference.report["status"]
    assert row.has_derived is False


def test_identical_reupload_returns_the_stored_derived_block(
    settings: Settings, imu_vsig_session: Path
):
    first = ingest_path(imu_vsig_session, settings)
    again = ingest_path(imu_vsig_session, settings)
    assert again.created is False and again.report["derived"] == first.report["derived"]


# --- class-B ride summaries -------------------------------------------------------------------


def _summary_table(settings: Settings, session_id: str):
    path = settings.sessions_dir / session_id / "parquet" / "derived_summary.parquet"
    return pq.read_table(path)


def test_ingest_writes_the_ride_summary_parquet_and_report_block(
    settings: Settings, imu_vsig_session: Path
):
    result = ingest_path(imu_vsig_session, settings)
    table = _summary_table(settings, result.session_id)

    assert table.schema.names == [
        "session_id",
        "record",
        "did",
        "signal",
        "cell",
        "bin0_low",
        "bin0_high",
        "bin1_low",
        "bin1_high",
        "value",
        "unit",
        "valid",
        "flag",
    ]
    types = {f.name: str(f.type) for f in table.schema}
    assert types["did"] == "int64" and types["cell"] == "int64"
    assert types["value"] == "double" and types["bin1_high"] == "double"
    assert types["valid"] == "bool" and types["record"] == "string"
    assert set(table["session_id"].to_pylist()) == {result.session_id}

    rows = table.to_pylist()
    scalar = next(r for r in rows if r["signal"] == "ride_stop_count")
    assert scalar["record"] == "RIDE_USAGE"
    assert scalar["did"] == vehicle_derived.RIDE_SUMMARY_RECORDS["RIDE_USAGE"][0]
    assert scalar["cell"] is None and scalar["bin0_low"] is None and scalar["bin1_high"] is None
    assert scalar["value"] == 0.0 and scalar["valid"] is True and scalar["unit"] == "count"
    cells = [r for r in rows if r["signal"] == "operating_point_time"]
    assert [r["cell"] for r in cells] == list(range(256))
    assert cells[24]["bin0_low"] == 750.0 and cells[24]["bin1_high"] == 56.25
    assert sum(r["value"] for r in cells) == pytest.approx(0.31)
    one_axis = [r for r in rows if r["signal"] == "ride_coolant_time_at_temp"]
    assert len(one_axis) == 10 and one_axis[0]["bin1_low"] is None
    assert {r["signal"] for r in rows}.isdisjoint({"ride_warmup_time", "ride_fuel_used"})

    block = result.report["derived_summary"]
    assert block["defs_version"] == result.report["defs_version"]
    assert block["ride_definition"] == "session"
    class_b = [n for n, spec in vehicle_derived.SIGNALS.items() if spec["class"] == "B"]
    assert sum(block["summary"].values()) == len(class_b)
    assert set(block["records"]) == set(vehicle_derived.RIDE_SUMMARY_RECORDS)
    usage = block["records"]["RIDE_USAGE"]
    assert usage["did"] == vehicle_derived.RIDE_SUMMARY_RECORDS["RIDE_USAGE"][0]
    assert usage["signals"]["ride_stop_count"] == {
        "status": "computed",
        "value": 0.0,
        "unit": "count",
        "flag": None,
    }
    binned = block["records"]["RIDE_OPERATING_POINTS"]["signals"]["operating_point_time"]
    assert binned["cells"] == 256 and binned["nonzero_cells"] == 1
    assert binned["total_s"] == pytest.approx(0.31) and binned["outside_s"] == 0.0
    assert "value" not in binned and "values" not in binned
    thermal = block["records"]["RIDE_THERMAL"]["signals"]
    assert thermal["ride_fan_on_temperature"] == {
        "status": "not_implemented",
        "reason": NOT_IMPLEMENTED_REASONS["ride_fan_on_temperature"],
    }
    assert thermal["ride_warmup_time"]["status"] == "missing_inputs"
    assert thermal["ride_oil_time_at_temp"]["status"] == "missing_inputs"
    assert "did.ENGINE_OIL_TEMP" in thermal["ride_oil_time_at_temp"]["missing"]
    assert block["records"]["RIDE_FUELING"]["signals"]["fuel_trim_map"]["status"] == "not_ready"

    on_disk = json.loads((settings.sessions_dir / result.session_id / "report.json").read_text())
    assert on_disk["derived_summary"] == block
    assert "ride summaries: one session = one ride (" in render_text(result.report)


def test_a_session_without_summary_rows_has_no_summary_parquet(
    settings: Settings, imu_vsig_session: Path
):
    result = ingest_path(imu_vsig_session, settings)
    session_dir = settings.sessions_dir / result.session_id
    path = session_dir / "parquet" / "derived_summary.parquet"
    assert path.is_file()
    for name in ("signals", "telemetry", "imu"):  # nothing left to summarize
        (session_dir / "parquet" / f"{name}.parquet").unlink(missing_ok=True)
    block, has_rows = derived.store_summary(session_dir, result.session_id)
    assert has_rows is False and not path.exists()
    assert "error" not in block and "computed" not in block["summary"]


def test_ingest_survives_a_summary_failure(
    settings: Settings, imu_vsig_session: Path, monkeypatch: pytest.MonkeyPatch
):
    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(derived, "summarize_session", boom)
    result = ingest_path(imu_vsig_session, settings)

    assert result.created is True and result.report["derived_summary"] == {
        "error": "RuntimeError: boom"
    }
    assert "ride summaries: error: RuntimeError: boom" in render_text(result.report)
    session_dir = settings.sessions_dir / result.session_id
    assert not (session_dir / "parquet" / "derived_summary.parquet").exists()
    assert (session_dir / "parquet" / "derived.parquet").is_file()  # class A unaffected
    on_disk = json.loads((session_dir / "report.json").read_text())
    assert on_disk["derived_summary"] == {"error": "RuntimeError: boom"}
    assert "error" not in result.report["derived"]


def test_ingest_evaluates_class_a_once(
    settings: Settings, imu_vsig_session: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = []
    real = engine.evaluate

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "evaluate", counting)
    result = ingest_path(imu_vsig_session, settings)
    assert "error" not in result.report["derived"]
    assert "error" not in result.report["derived_summary"]
    assert len(calls) == 1  # derived.parquet and the summaries share one evaluation
    calls.clear()
    block, summary_block = rederive_session(settings, result.session_id)
    assert len(calls) == 1
    assert block == result.report["derived"] and summary_block == result.report["derived_summary"]


def test_a_failed_shared_evaluation_is_recorded_in_both_blocks(
    settings: Settings, imu_vsig_session: Path, monkeypatch: pytest.MonkeyPatch
):
    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "evaluate", boom)
    result = ingest_path(imu_vsig_session, settings)

    assert result.created is True
    assert result.report["derived"] == {"error": "RuntimeError: boom"}
    assert result.report["derived_summary"] == {"error": "RuntimeError: boom"}
    session_dir = settings.sessions_dir / result.session_id
    assert not (session_dir / "parquet" / "derived.parquet").exists()
    assert not (session_dir / "parquet" / "derived_summary.parquet").exists()
    row = index.get_session(settings.index_db_path, result.session_id)
    assert row is not None and row.has_derived is False


# --- index migration ------------------------------------------------------------------------


def test_has_derived_is_added_to_an_existing_index(tmp_path: Path):
    db = tmp_path / "index.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, created_utc TEXT, "
        "imported_utc TEXT NOT NULL, content_hash TEXT NOT NULL, app_version TEXT, "
        "ble_schema_version INTEGER, packet_count INTEGER, loss_percent REAL, status TEXT, "
        "has_imu INTEGER NOT NULL DEFAULT 0, defs_version TEXT)"
    )
    conn.execute(
        "INSERT INTO sessions (session_id, imported_utc, content_hash) VALUES ('old', 'x', 'h')"
    )
    conn.commit()
    conn.close()

    row = index.get_session(db, "old")
    assert row is not None and row.has_derived is False and row.has_vehicle_signals is False
    index.set_has_derived(db, "old", True)
    assert index.list_sessions(db)[0].has_derived is True


# --- CLI ------------------------------------------------------------------------------------


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]) -> int:
    monkeypatch.setenv("VEHICLE_DATA_DIR", str(tmp_path / "data"))
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)
    return exc_info.value.code


def test_cli_derive_recomputes_one_session_and_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, imu_vsig_session: Path, capsys
):
    assert _run(monkeypatch, tmp_path, ["import", str(imu_vsig_session)]) == 0
    session_id = json.loads((imu_vsig_session / "meta.json").read_text())["session_id"]
    session_dir = tmp_path / "data" / "sessions" / session_id
    db = tmp_path / "data" / "index.sqlite"

    # lose the derived output, then recompute it
    (session_dir / "parquet" / "derived.parquet").unlink()
    index.set_has_derived(db, session_id, False)
    report = json.loads((session_dir / "report.json").read_text())
    del report["derived"]
    (session_dir / "report.json").write_text(json.dumps(report))

    capsys.readouterr()
    assert _run(monkeypatch, tmp_path, ["derive", session_id]) == 0
    out = capsys.readouterr().out
    assert session_id in out and "computed" in out
    assert f"{session_id}  derived summary  " in out
    assert (session_dir / "parquet" / "derived_summary.parquet").is_file()
    assert json.loads((session_dir / "report.json").read_text())["derived_summary"]["records"]
    assert (session_dir / "parquet" / "derived.parquet").is_file()
    assert index.get_session(db, session_id).has_derived is True
    assert json.loads((session_dir / "report.json").read_text())["derived"]["signals"]

    (session_dir / "parquet" / "derived.parquet").unlink()
    assert _run(monkeypatch, tmp_path, ["derive", "--all"]) == 0
    assert (session_dir / "parquet" / "derived.parquet").is_file()

    capsys.readouterr()
    assert _run(monkeypatch, tmp_path, ["list"]) == 0
    assert "derived=yes" in capsys.readouterr().out


def test_cli_derive_unknown_session_and_bad_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    assert _run(monkeypatch, tmp_path, ["derive", "20260101-000000-abcd"]) == 1
    assert _run(monkeypatch, tmp_path, ["derive"]) == 2
    assert _run(monkeypatch, tmp_path, ["derive", "--all", "20260101-000000-abcd"]) == 2
    assert _run(monkeypatch, tmp_path, ["derive", "--all"]) == 0  # nothing stored: nothing to do


# --- inputs ---------------------------------------------------------------------------------


def test_did_series_comes_from_signals_parquet_at_the_sample_time(
    settings: Settings, imu_vsig_session: Path
):
    result = ingest_path(imu_vsig_session, settings)
    inputs = SessionInputs(settings.sessions_dir / result.session_id)
    speed = inputs.series("did.VEHICLE_SPEED")
    # records: device time 9100 age 60 -> 9040 (raw 40); 9500 age 50 -> 9450 (raw 42)
    assert speed.times == [9040, 9450] and speed.values == [40.0, 42.0]
    # a stale value is not a sample: COOLANT_TEMP has one valid record (9500 - 700 = 8800)
    assert inputs.series("did.COOLANT_TEMP").times == [8800]
    assert inputs.series("did.REFERENCE_TORQUE") is None


def test_sensor_series_are_scaled_into_the_sensor_units(settings: Settings, imu_vsig_session: Path):
    result = ingest_path(imu_vsig_session, settings)
    session_dir = settings.sessions_dir / result.session_id
    imu = pq.read_table(session_dir / "parquet" / "imu.parquet")
    inputs = SessionInputs(session_dir)

    accel = inputs.series("sensor.imu_accel_x")
    assert accel.times[0] == IMU_START_MS and len(accel) == IMU_SAMPLES
    assert accel.values[0] == pytest.approx(imu["ax_g"][0].as_py() * 9.80665)  # g -> m/s2
    assert inputs.series("sensor.imu_gyro_z").values[0] == pytest.approx(imu["gz_dps"][0].as_py())
    assert inputs.series("sensor.lean_angle") is None  # not a stored column


def test_telemetry_is_the_fallback_for_dids_without_signals_csv(
    settings: Settings, fixtures_dir: Path
):
    result = ingest_path(fixtures_dir / "v4_session", settings)
    inputs = SessionInputs(settings.sessions_dir / result.session_id)
    rpm = inputs.series("did.ENGINE_SPEED")
    assert rpm is not None and rpm.times == sorted(set(rpm.times))
    assert inputs.series("did.REFERENCE_TORQUE") is None


def test_gps_sensors_read_speed_and_heading_never_a_position(
    settings: Settings, fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    result = ingest_path(fixtures_dir / "v4_gps_session", settings)
    inputs = SessionInputs(settings.sessions_dir / result.session_id)
    assert inputs.series("sensor.gps_ground_speed") is not None
    assert resolve_sensor("gps_ground_speed").column == "ground_speed_mps"
    assert resolve_sensor("gps_heading").column == "heading_of_motion_deg"
    assert resolve_sensor("gps_fix_type").column == "fix_type"
    monkeypatch.setitem(
        vehicle_derived.SENSORS,
        "gps_lat",
        {"unit": "deg", "stale_after_ms": 300, "phase0": "ble.gpsBlock.latitude"},
    )
    assert resolve_sensor("gps_lat") is None


def test_series_helpers_drop_gaps_dedupe_and_unwrap():
    series = _to_series([30, 10, 20, 20, None, 40], [3.0, 1.0, 2.0, 2.5, 9.0, None])
    assert series.times == [10, 20, 30] and series.values == [1.0, 2.5, 3.0]
    wrap = 2**32
    assert _unwrap([wrap - 20, wrap - 10, 5, 15]) == [wrap - 20, wrap - 10, wrap + 5, wrap + 15]


def test_ride_bus_off_count_is_computed_from_the_stored_can_health(
    settings: Settings, fixtures_dir: Path
):
    # v3_session: can_bus_off_count 0, 0, 1, 1, 1 -> one bus-off event in the ride.
    result = ingest_path(fixtures_dir / "v3_session", settings)
    health = result.report["derived_summary"]["records"]["RIDE_ECU_HEALTH"]["signals"]
    assert health["ride_bus_off_count"] == {
        "status": "computed",
        "value": 1.0,
        "unit": "count",
        "flag": None,
    }
