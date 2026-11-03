"""TR-3: compare /track path length with the implemented derived ride_distance_gps."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vehicle_server import gps_track
from vehicle_server.api import create_app
from vehicle_server.config import Settings
from vehicle_server.defs import ble as defs_ble
from vehicle_server.derived import summary
from vehicle_server.derived.engine import Evaluation
from vehicle_server.derived.inputs import Series, SessionInputs
from vehicle_server.ingest import ingest_path

AUTH = {"Authorization": "Bearer test-token"}
FIX = {
    entry["name"]: float(entry["value"])
    for entry in defs_ble.SCHEMA["gpsBlock"]["fixType"]["values"]
}


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


@pytest.fixture
def gps_session_id(settings: Settings, fixtures_dir: Path) -> str:
    return ingest_path(fixtures_dir / "v4_gps_session", settings).session_id


def test_track_path_length_agrees_with_ride_distance_gps_on_a_stored_session(
    settings: Settings, client: TestClient, gps_session_id: str
):
    # Both /track and the summary use the same usable/fresh fix rule (D-095). The summary
    # follows the hold rule of formulas.hold_segments: its breakpoints are the union of the
    # speed and fix-type sample times, so the last speed sample (9700 ms) is held up to the
    # fix-type sample at 9800 ms while still fresh (11.3 m/s * 0.1 s = 1.13 m). /track ends
    # at the last speed sample and uses trapezoids, so the summary is 0.00111165 km longer
    # on this short fixture; both stay far inside the defs 0.01 km resolution.
    result = summary.evaluate(SessionInputs(settings.sessions_dir / gps_session_id), Evaluation())
    distance_km = result.coverage["ride_distance_gps"]["value"]
    track = client.get(f"/sessions/{gps_session_id}/track", headers=AUTH).json()
    track_km = track["path_length_m"] / 1000.0
    assert distance_km == pytest.approx(0.0067083, abs=1e-6)
    assert track_km - distance_km == pytest.approx(-0.00111165, abs=1e-6)
    assert abs(track_km - distance_km) <= 0.01


def test_track_path_length_agrees_with_ride_distance_gps_for_constant_speed():
    # With a constant speed and a usable fix at every sample, the trapezoid and the
    # held integral are equal; this isolates the two integration rules.
    times = list(range(0, 10_001, 100))

    class Inputs:
        def series(self, name: str) -> Series | None:
            if name == "sensor.gps_ground_speed":
                return Series(times, [10.0] * len(times))
            if name == "sensor.gps_fix_type":
                return Series(times, [FIX["fix3d"]] * len(times))
            return None

    result = summary.evaluate(Inputs(), Evaluation())
    distance_km = result.coverage["ride_distance_gps"]["value"]
    assert distance_km == pytest.approx(0.1)

    track = gps_track.build_track(
        gps_track.TrackSeries(times, [10.0] * len(times)),
        gps_track.TrackSeries(times, [0.0] * len(times)),
        gps_track.TrackSeries(times, [FIX["fix3d"]] * len(times)),
    )
    assert track.path_length_m / 1000.0 == pytest.approx(distance_km, abs=0.001)
