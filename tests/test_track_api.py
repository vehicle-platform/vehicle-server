"""GET /sessions/{id}/track (TR-2) and its cross-check with ride_distance_gps (TR-3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vehicle_server import ble_schema, series
from vehicle_server.api import create_app
from vehicle_server.config import Settings
from vehicle_server.decode import (
    GPS_HEADER,
)
from vehicle_server.defs import ble as defs_ble
from vehicle_server.ingest import ingest_path

from .conftest import copy_fixture

AUTH = {"Authorization": "Bearer test-token"}
GPS = ble_schema.load_default_schema()["gpsBlock"]
FIX = {entry["name"]: entry["value"] for entry in GPS["fixType"]["values"]}
SPEED_LSB = GPS["scale"]["speed"]["lsbPerUnit"]
HEADING_LSB = GPS["scale"]["heading"]["lsbPerUnit"]
GNSS_FIX_OK_BIT = next(bit["bit"] for bit in GPS["flags"]["bits"] if bit["name"] == "gnssFixOk")


def _gps_rows(points: list[tuple[int, float, float]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for seq, (t, speed, heading) in enumerate(points):
        speed_raw = round(speed * SPEED_LSB)
        heading_raw = round(heading * HEADING_LSB)
        values = {field["name"]: 0 for field in GPS["fields"]}
        values.update(
            {
                "version": GPS["version"],
                "seq": seq,
                "deviceTimeMs": t,
                "groundSpeed": speed_raw,
                "headingOfMotion": heading_raw,
                "speedAccuracy": 300,
                "headingAccuracy": 50_000,
                "fixType": FIX["fix3d"],
                "numSv": 10,
                "flags": 1 << GNSS_FIX_OK_BIT,
                "reserved": 0,
            }
        )
        raw = ble_schema.pack_fields(values, defs_ble.gps_fields(), defs_ble.GPS_TOTAL_BYTES)
        row = {
            "rx_utc_iso": f"2026-01-15T15:00:{seq % 60:02d}.000Z",
            "rx_mono_ms": str(t),
            "seq": str(seq),
            "lost_since_prev": "0",
            "raw_hex": raw.hex(),
            "device_time_ms": str(t),
            "ground_speed": str(speed_raw),
            "heading_of_motion": str(heading_raw),
            "speed_accuracy": "300",
            "heading_accuracy": "50000",
            "fix_type": str(FIX["fix3d"]),
            "num_sv": "10",
            "flags": str(1 << GNSS_FIX_OK_BIT),
            "ground_speed_mps": repr(speed_raw / SPEED_LSB),
            "heading_of_motion_deg": repr(heading_raw / HEADING_LSB),
            "speed_accuracy_mps": repr(300 / SPEED_LSB),
            "heading_accuracy_deg": repr(50_000 / HEADING_LSB),
            "gnss_fix_ok": "1",
            "parse_error": "0",
            "uart_overflow": "0",
        }
        rows.append([row[column] for column in GPS_HEADER])
    return rows


def _make_gps_session(
    fixtures_dir: Path, tmp_path: Path, points: list[tuple[int, float, float]]
) -> Path:
    session = copy_fixture(fixtures_dir / "v4_gps_session", tmp_path / "gps")
    rows = _gps_rows(points)
    (session / "gps.csv").write_text(
        "\n".join([",".join(GPS_HEADER), *(",".join(row) for row in rows)]) + "\n"
    )
    summary_path = session / "summary.json"
    report = json.loads(summary_path.read_text())
    report["gps_block_count"] = len(rows)
    report["gps_lost_or_skipped_blocks"] = 0
    report["gps_lost_or_skipped_percent"] = 0.0
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    return session


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


@pytest.fixture
def gps_session_id(settings: Settings, fixtures_dir: Path) -> str:
    return ingest_path(fixtures_dir / "v4_gps_session", settings).session_id


@pytest.fixture
def many_session_id(settings: Settings, fixtures_dir: Path, tmp_path: Path) -> str:
    points = [(9200 + i * 100, 10.0, (360.0 * i / 100) % 360.0) for i in range(100)]
    return ingest_path(_make_gps_session(fixtures_dir, tmp_path, points), settings).session_id


def _get(client: TestClient, session_id: str, **params):
    return client.get(f"/sessions/{session_id}/track", headers=AUTH, params=params)


def test_track_requires_token(client: TestClient, gps_session_id: str):
    assert client.get(f"/sessions/{gps_session_id}/track").status_code == 401
    assert client.get(f"/sessions/{gps_session_id}/track?token=test-token").status_code == 401


@pytest.mark.parametrize("bad_id", ["00000000-000000-ffff", "not-a-session", "..%2f..%2fetc"])
def test_track_unknown_or_malformed_session_is_404(client: TestClient, bad_id: str):
    assert _get(client, bad_id).status_code == 404


def test_track_session_without_gps_is_404(
    settings: Settings, client: TestClient, fixtures_dir: Path
):
    session_id = ingest_path(fixtures_dir / "v4_session", settings).session_id
    resp = _get(client, session_id)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "session has no GPS"


def test_track_t0_is_the_series_t0(client: TestClient, gps_session_id: str):
    series_body = client.get(f"/sessions/{gps_session_id}/series", headers=AUTH).json()
    track = _get(client, gps_session_id).json()
    assert track["t0_device_ms"] == series_body["t0_device_ms"]
    assert track["method"] == "dead_reckoning"
    assert track["frame"] == "relative_enu_from_first_fix"
    assert track["segments_total"] == 1
    assert track["path_length_m"] == pytest.approx(5.59665)
    segment = track["segments"][0]
    assert segment["gap_before_ms"] is None
    assert len(segment["t_ms"]) == len(segment["x_m"]) == len(segment["y_m"])


def test_track_max_points_is_respected(client: TestClient, many_session_id: str):
    body = _get(client, many_session_id, max_points=10).json()
    assert body["downsampled"] is True
    assert body["points_total"] == 100
    assert sum(len(segment["t_ms"]) for segment in body["segments"]) <= 10


def test_track_color_uses_the_defs_unit_and_nulls_after_staleness(
    client: TestClient, gps_session_id: str, monkeypatch: pytest.MonkeyPatch
):
    original = series.load_all

    def with_lean_angle(session_dir: Path):
        available = original(session_dir)
        t0 = series.t0_of(available)
        available["sensor.lean_angle"] = series.SeriesData(
            "sensor.lean_angle", "sensor", "deg", [t0 + 600, t0 + 700], [12.0, 15.0]
        )
        return available

    monkeypatch.setattr("vehicle_server.api.series.load_all", with_lean_angle)
    body = _get(client, gps_session_id, color="sensor.lean_angle").json()
    assert body["color"] == {"name": "sensor.lean_angle", "unit": "deg"}
    assert body["segments"][0]["color_v"] == [12.0, 15.0, 15.0, None]


@pytest.mark.parametrize("name", ["sensor.gps_latitude", "did.NOPE"])
def test_track_position_looking_or_unknown_color_is_refused(
    client: TestClient, gps_session_id: str, name: str
):
    resp = _get(client, gps_session_id, color=name)
    assert resp.status_code == 404
    assert resp.json()["detail"] == f"unknown or unavailable series: {name}"


def test_track_window_is_applied_before_integration(client: TestClient, gps_session_id: str):
    body = _get(client, gps_session_id, from_ms=700, to_ms=1000).json()
    segment = body["segments"][0]
    assert segment["t_ms"] == [900, 1000]
    assert body["path_length_m"] > 0


def test_track_max_points_out_of_range_is_422(client: TestClient, gps_session_id: str):
    assert _get(client, gps_session_id, max_points=9).status_code == 422
    assert _get(client, gps_session_id, max_points=20001).status_code == 422
    assert _get(client, gps_session_id, min_heading_speed_mps=11).status_code == 422
