"""D-079 health windows end to end: synthetic signals.csv -> ingest -> report/API/CLI."""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vehicle_server import cli
from vehicle_server.api import create_app
from vehicle_server.config import Settings
from vehicle_server.decode import SIGNALS_HEADER, V4_HEADER, vsig_state
from vehicle_server.defs import vehicle_catalog
from vehicle_server.ingest import ingest_path
from vehicle_server.report import render_text

from .make_fixtures import write_csv

AUTH = {"Authorization": "Bearer test-token"}
SESSION_ID = "20260115-160000-e2e0"
AMB = 20.0
TAU_S = 300.0
START = datetime.datetime(2026, 1, 15, 16, 0, 0, tzinfo=datetime.UTC)
DIDS = (
    "ENGINE_SPEED",
    "VEHICLE_SPEED",
    "COOLANT_TEMP",
    "INTAKE_MAP",
    "BATTERY_VOLTAGE",
    "AMBIENT_AIR_TEMP",
    "INTAKE_AIR_TEMP",
    "BAROMETRIC_PRESSURE",
    "SHORT_TERM_FUEL_TRIM_B1",
    "THROTTLE_POS",
)


def _value_at(name: str, t: int) -> float:
    if name == "ENGINE_SPEED":
        if t < 3_000 or t >= 30_000:
            return 0.0
        return 1_200.0 if t < 16_500 else 1_300.0
    if name == "COOLANT_TEMP":
        if t <= 30_000:
            return 85.0
        return AMB + (85.0 - AMB) * math.exp(-(t - 30_000) / (TAU_S * 1_000))
    if name == "INTAKE_MAP":
        return 110.0 if 16_500 <= t < 30_000 else 100.0
    if name == "BATTERY_VOLTAGE":
        return 12.6
    if name == "AMBIENT_AIR_TEMP":
        return AMB
    if name == "INTAKE_AIR_TEMP":
        return 80.0
    if name == "BAROMETRIC_PRESSURE":
        return 100.0
    if name == "SHORT_TERM_FUEL_TRIM_B1":
        return 4.0 if t < 16_500 else 6.0
    if name == "THROTTLE_POS":
        return 10.0
    return 0.0  # VEHICLE_SPEED


def _raw(name: str, value: float) -> int:
    entry = vehicle_catalog.SIGNALS[name]
    return round((value - entry["offset"]) * entry["factor_den"] / entry["factor_num"])


def _write_health_window_session(root: Path) -> Path:
    session = root / "health-window-session"
    session.mkdir()
    write_csv(session / "telemetry.csv", V4_HEADER, [])

    rows = []
    state = vsig_state("valid")
    for i, t in enumerate(range(0, 91_001, 100)):
        rx = (
            (START + datetime.timedelta(milliseconds=t))
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        for first_record, name in enumerate(DIDS):
            raw = _raw(name, _value_at(name, t))
            entry = vehicle_catalog.SIGNALS[name]
            value = vehicle_catalog.decode(name, raw.to_bytes(entry["length"], "big"))
            rows.append(
                [
                    rx,
                    str(t),
                    str(i % 256),
                    "0",
                    str(t),
                    str(len(DIDS)),
                    str(first_record),
                    str(entry["did"]),
                    str(state),
                    "0",
                    str(raw),
                    name,
                    repr(value),
                ]
            )
    write_csv(session / "signals.csv", SIGNALS_HEADER, rows)

    write_csv(
        session / "events.csv",
        ["rx_utc_iso", "rx_mono_ms", "event", "detail"],
        [
            [rx, "0", "recording_started", ""],
            [rx, "91000", "recording_stopped", ""],
        ],
    )
    (session / "meta.json").write_text(
        json.dumps(
            {
                "session_id": SESSION_ID,
                "created_utc": "2026-01-15T16:00:00.000Z",
                "app_version": "e2e",
                "ble_schema_version": 4,
            }
        )
    )
    (session / "summary.json").write_text(
        json.dumps({"duration_ms": 91_000, "packet_count": 0, "lost_count": 0})
    )
    return session


def test_health_windows_from_ingest_to_report_api_and_cli(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    result = ingest_path(_write_health_window_session(tmp_path), settings)
    report = json.loads((settings.sessions_dir / SESSION_ID / "report.json").read_text())

    summary_block = report["derived_summary"]
    health_windows = summary_block["health_windows"]
    assert set(summary_block) == {
        "defs_version",
        "ride_definition",
        "summary",
        "records",
        "health_windows",
    }
    assert health_windows["t0_device_ms"] == 0
    assert list(health_windows["windows"]) == [
        "W1_KEY_ON",
        "W2_CRANK",
        "W3_WARM_IDLE",
        "W4_CRUISE",
        "W5_POST_STOP",
    ]
    assert health_windows["windows"]["W1_KEY_ON"]["instances"] == [
        {
            "start_ms": 0,
            "end_ms": 3_000,
            "duration_ms": 3_000,
            "close_reason": "condition",
            "qualifies": True,
            "first_qualifying": True,
        }
    ]
    assert health_windows["windows"]["W3_WARM_IDLE"]["instances"] == [
        {
            "start_ms": 3_000,
            "end_ms": 30_000,
            "duration_ms": 27_000,
            "close_reason": "condition",
            "qualifies": True,
            "first_qualifying": True,
        }
    ]
    assert health_windows["windows"]["W5_POST_STOP"]["instances"] == [
        {
            "start_ms": 30_000,
            "end_ms": 91_000,
            "duration_ms": 61_000,
            "close_reason": "session_end",
            "qualifies": True,
            "first_qualifying": True,
        }
    ]
    assert health_windows["windows"]["W4_CRUISE"]["instances"] == []

    by_record = summary_block["records"]
    key_on = by_record["RIDE_WINDOW_KEY_ON"]["signals"]
    warm_idle = by_record["RIDE_WINDOW_WARM_IDLE"]["signals"]
    post_stop = by_record["RIDE_WINDOW_POST_STOP"]["signals"]

    assert key_on["window_key_on_rest_voltage"]["value"] == pytest.approx(12.6)
    assert key_on["window_key_on_count"]["value"] == 1.0
    assert warm_idle["window_warm_idle_rpm_spread"]["value"] == pytest.approx(100.0)
    assert warm_idle["window_warm_idle_rpm_drift"]["value"] == pytest.approx(100.0)
    assert warm_idle["window_warm_idle_map_spread"]["value"] == pytest.approx(10.0)
    assert warm_idle["window_warm_idle_stft_mean"]["value"] == pytest.approx(5.078125)
    assert warm_idle["window_warm_idle_count"]["value"] == 1.0
    assert post_stop["window_post_stop_coolant_start"]["value"] == pytest.approx(85.0)
    assert post_stop["window_post_stop_coolant_end"]["value"] == pytest.approx(73.0)
    assert post_stop["window_post_stop_coolant_tau"]["value"] == pytest.approx(TAU_S, rel=0.01)
    assert post_stop["window_post_stop_count"]["value"] == 1.0
    assert result.report == report

    client = TestClient(create_app(settings))
    summary = client.get(f"/sessions/{SESSION_ID}/summary", headers=AUTH)
    assert summary.status_code == 200
    series_t0 = client.get(f"/sessions/{SESSION_ID}/series", headers=AUTH).json()["t0_device_ms"]
    assert health_windows["t0_device_ms"] == series_t0
    api_signals = {entry["name"]: entry for entry in summary.json()["signals"]}
    assert api_signals["window_key_on_rest_voltage"]["value"] == pytest.approx(12.6)
    assert api_signals["window_warm_idle_rpm_drift"]["value"] == pytest.approx(100.0)
    assert api_signals["window_post_stop_coolant_tau"]["value"] == pytest.approx(TAU_S, rel=0.01)

    rides = client.get("/rides", headers=AUTH)
    assert rides.status_code == 200
    row = next(r for r in rides.json()["rows"] if r["session_id"] == SESSION_ID)
    assert row["values"]["window_key_on_rest_voltage"] == pytest.approx(12.6)
    assert row["values"]["window_warm_idle_rpm_spread"] == pytest.approx(100.0)
    assert row["values"]["window_post_stop_coolant_start"] == pytest.approx(85.0)

    monkeypatch.setenv("VEHICLE_DATA_DIR", str(settings.data_dir))
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["derive", SESSION_ID])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert SESSION_ID in out and "derived summary" in out
    rerun = json.loads((settings.sessions_dir / SESSION_ID / "report.json").read_text())
    assert (
        rerun["derived_summary"]["records"]["RIDE_WINDOW_POST_STOP"]["signals"][
            "window_post_stop_count"
        ]["value"]
        == 1.0
    )
    assert rerun["derived_summary"]["health_windows"] == health_windows

    text = render_text(report)
    assert "ride summaries: one session = one ride" in text
    assert "W1_KEY_ON: 1 instance, 1 qualifying (first 0.0-3.0 s)" in text
    assert "W3_WARM_IDLE: 1 instance, 1 qualifying (first 3.0-30.0 s)" in text
    assert "W4_CRUISE: 0 instances, 0 qualifying\n" in text + "\n"
    assert "W5_POST_STOP: 1 instance, 1 qualifying (first 30.0-91.0 s)" in text


def _evaluation_with_w3(open_ms: int, close_ms: int):
    from vehicle_server.derived import summary
    from vehicle_server.derived.health_windows import WindowInstance, WindowResult

    instance = WindowInstance(open_ms, close_ms, "condition", True)
    return summary.SummaryEvaluation(
        windows={"W3_WARM_IDLE": WindowResult((instance,), instance, 1)}
    )


def test_health_windows_block_subtracts_the_series_t0(monkeypatch, tmp_path: Path):
    from vehicle_server import derived, series

    monkeypatch.setattr(series, "load_all", lambda _session_dir: {})
    monkeypatch.setattr(series, "t0_of", lambda _all: 1_000_000)
    block = derived._health_windows_block(_evaluation_with_w3(1_003_000, 1_030_000), tmp_path)

    assert block["t0_device_ms"] == 1_000_000
    assert block["windows"]["W3_WARM_IDLE"]["instances"] == [
        {
            "start_ms": 3_000,
            "end_ms": 30_000,
            "duration_ms": 27_000,
            "close_reason": "condition",
            "qualifies": True,
            "first_qualifying": True,
        }
    ]


def test_health_windows_failure_stays_in_its_own_key(monkeypatch, tmp_path: Path):
    from vehicle_server import derived, series

    def broken(_session_dir):
        raise OSError("derived.parquet unreadable")

    monkeypatch.setattr(series, "load_all", broken)
    block = derived._health_windows_block(_evaluation_with_w3(0, 1), tmp_path)

    assert block == {"error": "OSError: derived.parquet unreadable"}
