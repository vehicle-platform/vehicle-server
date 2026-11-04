from __future__ import annotations

import io
import json
import shutil
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vehicle_server import __version__
from vehicle_server.api import create_app
from vehicle_server.config import Settings

from .conftest import zip_fixture

AUTH = {"Authorization": "Bearer test-token"}


def _zip_bytes(fixtures_dir: Path, name: str, tmp_path: Path) -> bytes:
    zip_path = zip_fixture(fixtures_dir / name, tmp_path / f"{name}.zip")
    return zip_path.read_bytes()


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def test_health_requires_no_auth(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_openapi_version_matches_package_and_pyproject(client: TestClient):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert resp.json()["info"]["version"] == __version__ == pyproject["project"]["version"]


def test_protected_endpoint_without_token_is_401(client: TestClient):
    resp = client.get("/sessions")
    assert resp.status_code == 401


def test_protected_endpoint_with_wrong_token_is_401(client: TestClient):
    resp = client.get("/sessions", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_api_returns_503_when_token_not_configured(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", api_token=None, max_upload_mb=100)
    client = TestClient(create_app(settings))
    resp = client.get("/sessions", headers=AUTH)
    assert resp.status_code == 503


def test_create_session_then_list_and_report(
    client: TestClient, fixtures_dir: Path, tmp_path: Path
):
    body = _zip_bytes(fixtures_dir, "v2_session", tmp_path)
    resp = client.post(
        "/sessions",
        headers=AUTH,
        files={"archive": ("session.zip", io.BytesIO(body), "application/zip")},
    )
    assert resp.status_code == 201
    report = resp.json()
    assert report["session_id"] == "20260115-120000-a1b2"

    listed = client.get("/sessions", headers=AUTH)
    assert listed.status_code == 200
    assert any(row["session_id"] == "20260115-120000-a1b2" for row in listed.json())

    got_report = client.get("/sessions/20260115-120000-a1b2/report", headers=AUTH)
    assert got_report.status_code == 200
    assert got_report.json()["session_id"] == "20260115-120000-a1b2"

    text_report = client.get(
        "/sessions/20260115-120000-a1b2/report", headers=AUTH, params={"format": "text"}
    )
    assert text_report.status_code == 200
    assert "20260115-120000-a1b2" in text_report.text


def test_report_unknown_session_is_404(client: TestClient):
    resp = client.get("/sessions/00000000-000000-ffff/report", headers=AUTH)
    assert resp.status_code == 404


def test_create_session_idempotent_replay_is_200(
    client: TestClient, fixtures_dir: Path, tmp_path: Path
):
    body = _zip_bytes(fixtures_dir, "v3_session", tmp_path)
    files = {"archive": ("session.zip", io.BytesIO(body), "application/zip")}
    first = client.post("/sessions", headers=AUTH, files=files)
    assert first.status_code == 201

    files = {"archive": ("session.zip", io.BytesIO(body), "application/zip")}
    second = client.post("/sessions", headers=AUTH, files=files)
    assert second.status_code == 200


def test_create_session_conflict_is_409(client: TestClient, fixtures_dir: Path, tmp_path: Path):
    body = _zip_bytes(fixtures_dir, "v3_session", tmp_path)
    files = {"archive": ("session.zip", io.BytesIO(body), "application/zip")}
    first = client.post("/sessions", headers=AUTH, files=files)
    assert first.status_code == 201

    mutated_dir = tmp_path / "mutated"
    shutil.copytree(fixtures_dir / "v3_session", mutated_dir)
    events_path = mutated_dir / "events.csv"
    events_path.write_text(events_path.read_text() + "2026-01-15T13:00:05.000Z,5500,note,changed\n")
    mutated_zip = zip_fixture(mutated_dir, tmp_path / "mutated.zip")

    files = {"archive": ("session.zip", io.BytesIO(mutated_zip.read_bytes()), "application/zip")}
    resp = client.post("/sessions", headers=AUTH, files=files)
    assert resp.status_code == 409


def test_create_session_invalid_is_422(client: TestClient, fixtures_dir: Path, tmp_path: Path):
    mutated_dir = tmp_path / "mutated"
    shutil.copytree(fixtures_dir / "v2_session", mutated_dir)
    (mutated_dir / "summary.json").unlink()
    mutated_zip = zip_fixture(mutated_dir, tmp_path / "mutated.zip")

    files = {"archive": ("session.zip", io.BytesIO(mutated_zip.read_bytes()), "application/zip")}
    resp = client.post("/sessions", headers=AUTH, files=files)
    assert resp.status_code == 422


def test_create_session_too_large_is_413(fixtures_dir: Path, tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", api_token="test-token", max_upload_mb=0)
    client = TestClient(create_app(settings))
    body = _zip_bytes(fixtures_dir, "v2_session", tmp_path)
    resp = client.post(
        "/sessions",
        headers=AUTH,
        files={"archive": ("session.zip", io.BytesIO(body), "application/zip")},
    )
    assert resp.status_code == 413


def _upload(client: TestClient, fixtures_dir: Path, tmp_path: Path, name: str) -> None:
    body = _zip_bytes(fixtures_dir, name, tmp_path)
    resp = client.post(
        "/sessions",
        headers=AUTH,
        files={"archive": ("session.zip", io.BytesIO(body), "application/zip")},
    )
    assert resp.status_code == 201


@pytest.mark.parametrize("gate", ["B0", "V0", "V1", "v0"])
def test_bringup_returns_the_result_for_each_gate(
    client: TestClient, fixtures_dir: Path, tmp_path: Path, gate: str
):
    _upload(client, fixtures_dir, tmp_path, "v4_session")
    resp = client.get("/sessions/20260115-140000-e5f6/bringup", headers=AUTH, params={"gate": gate})
    assert resp.status_code == 200
    data = resp.json()
    assert data["gate"] == gate.upper()
    assert data["session_id"] == "20260115-140000-e5f6"
    assert data["verdict"] in ("pass", "fail", "incomplete")
    assert all(
        set(c) == {"id", "status", "measured", "expected", "source", "note"} for c in data["checks"]
    )


def test_bringup_matches_the_cli_serialiser(client: TestClient, fixtures_dir: Path, tmp_path: Path):
    from vehicle_server import bringup

    _upload(client, fixtures_dir, tmp_path, "v4_session")
    resp = client.get("/sessions/20260115-140000-e5f6/bringup", headers=AUTH, params={"gate": "V0"})
    session_dir = client.app.state.settings.sessions_dir / "20260115-140000-e5f6"
    report = json.loads((session_dir / "report.json").read_text())
    expected = bringup.check_gate("V0", report, bringup.load_inputs(session_dir)).to_dict()
    assert resp.json() == json.loads(json.dumps(expected))


def test_bringup_needs_the_token(client: TestClient, fixtures_dir: Path, tmp_path: Path):
    _upload(client, fixtures_dir, tmp_path, "v4_session")
    url = "/sessions/20260115-140000-e5f6/bringup"
    assert client.get(url, params={"gate": "V0"}).status_code == 401
    assert (
        client.get(url, params={"gate": "V0"}, headers={"Authorization": "Bearer nope"}).status_code
        == 401
    )


def test_bringup_unknown_session_is_404(client: TestClient):
    resp = client.get("/sessions/00000000-000000-ffff/bringup", headers=AUTH, params={"gate": "V0"})
    assert resp.status_code == 404


@pytest.mark.parametrize("params", [{}, {"gate": "V2"}])
def test_bringup_missing_or_unknown_gate_is_422(
    client: TestClient, fixtures_dir: Path, tmp_path: Path, params: dict
):
    _upload(client, fixtures_dir, tmp_path, "v4_session")
    resp = client.get("/sessions/20260115-140000-e5f6/bringup", headers=AUTH, params=params)
    assert resp.status_code == 422
    assert "gate must be one of" in resp.json()["detail"]
