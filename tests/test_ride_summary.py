"""Read-only ride-summary API: /sessions/{id}/summary and /rides (RS-1, RS-2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from vehicle_server import index, ride_summary
from vehicle_server.api import create_app
from vehicle_server.config import Settings
from vehicle_server.defs import vehicle_derived
from vehicle_server.derived import SUMMARY_SCHEMA
from vehicle_server.derived.summary import bin_edges, cell_axes, cell_count
from vehicle_server.ingest import ingest_path

AUTH = {"Authorization": "Bearer test-token"}
SESSION_ID = "20260115-120000-abcd"
SECOND_ID = "20260116-120000-abcd"
THIRD_ID = "20260117-120000-abcd"


def _write_report(
    settings: Settings, session_id: str, block: dict[str, Any], defs_version: str = "vtest"
) -> Path:
    session_dir = settings.sessions_dir / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "report.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "defs_version": defs_version,
                "derived_summary": block,
            }
        )
    )
    return session_dir


def _write_rows(settings: Settings, session_id: str, rows: list[dict[str, Any]]) -> None:
    session_dir = settings.sessions_dir / session_id
    parquet = session_dir / "parquet"
    parquet.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SUMMARY_SCHEMA), parquet / "derived_summary.parquet"
    )


def _add_index(settings: Settings, session_id: str, created_utc: str | None) -> None:
    index.upsert_session(
        settings.index_db_path,
        index.SessionIndexRow(
            session_id=session_id,
            created_utc=created_utc,
            imported_utc="2026-01-20T00:00:00+00:00",
            content_hash=session_id,
            app_version="test",
            ble_schema_version=4,
            packet_count=10,
            loss_percent=0.0,
            status="ok",
            has_imu=False,
            defs_version="vtest",
        ),
    )


def _block(signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "defs_version": "vtest",
        "ride_definition": "session",
        "records": {"RIDE_USAGE": {"did": 1, "signals": signals}},
    }


def _binned_rows(
    session_id: str, values: list[float | None], *, shift_first_edge: bool = False
) -> list[dict[str, Any]]:
    spec = vehicle_derived.SIGNALS["ride_coolant_time_at_temp"]
    axis = spec["bins"][0]
    rows = []
    for cell, value in enumerate(values):
        low, high = bin_edges(axis, cell)
        if shift_first_edge and cell == 0:
            low = 0.0
        rows.append(
            {
                "session_id": session_id,
                "record": spec["record"],
                "did": vehicle_derived.RIDE_SUMMARY_RECORDS[spec["record"]][0],
                "signal": "ride_coolant_time_at_temp",
                "cell": cell,
                "bin0_low": low,
                "bin0_high": high,
                "bin1_low": None,
                "bin1_high": None,
                "value": value,
                "unit": spec["unit"],
                "valid": value is not None,
                "flag": None,
            }
        )
    return rows


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def _summary(signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "ride_distance": {"status": "computed", "value": 12.3, "unit": "km", "flag": None},
        "ride_time": {"status": "computed", "value": 90.0, "unit": "s", "flag": None},
        "fuel_trim_map": {
            "status": "not_ready",
            "reason": "ready_phase0",
            "pending_inputs": ["fuel-system status"],
        },
        "ride_fuel_used": {
            "status": "missing_inputs",
            "missing": ["did.ENGINE_FUEL_RATE"],
        },
        **signals,
    }


def _get_summary(client: TestClient, session_id: str = SESSION_ID):
    return client.get(f"/sessions/{session_id}/summary", headers=AUTH)


def _get_rides(client: TestClient, **params):
    return client.get("/rides", headers=AUTH, params=params)


def test_summary_requires_token(client: TestClient):
    assert client.get(f"/sessions/{SESSION_ID}/summary").status_code == 401


@pytest.mark.parametrize("bad_id", ["00000000-000000-ffff", "not-a-session", "..%2f..%2fetc"])
def test_summary_unknown_or_malformed_session_is_404(client: TestClient, bad_id: str):
    assert _get_summary(client, bad_id).status_code == 404


def test_summary_without_block_or_with_error_is_404(settings: Settings, client: TestClient):
    no_block = settings.sessions_dir / SESSION_ID
    no_block.mkdir(parents=True)
    (no_block / "report.json").write_text(json.dumps({"defs_version": "vtest"}))
    resp = _get_summary(client)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "session has no ride summary"

    error = settings.sessions_dir / SECOND_ID
    error.mkdir(parents=True)
    (error / "report.json").write_text(
        json.dumps({"defs_version": "vtest", "derived_summary": {"error": "RuntimeError: boom"}})
    )
    resp = _get_summary(client, SECOND_ID)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "RuntimeError: boom"


def test_summary_scalars_are_in_defs_order_and_keep_status_fields(
    settings: Settings, client: TestClient
):
    signals = _summary(
        {
            "ride_distance": {
                "status": "computed",
                "value": 12.3,
                "unit": "km",
                "flag": None,
            },
            "ride_time": {"status": "computed", "value": 90.0, "unit": "s", "flag": None},
        }
    )
    # Store in reverse order on purpose: the response follows the defs order.
    _write_report(settings, SESSION_ID, _block(dict(reversed(list(signals.items())))))
    body = _get_summary(client).json()
    names = [entry["name"] for entry in body["signals"]]
    assert names == [name for name in vehicle_derived.SIGNALS if name in signals]
    assert body["ride_definition"] == "session"
    assert body["defs_version"] == "vtest"
    by_name = {entry["name"]: entry for entry in body["signals"]}
    assert by_name["ride_distance"] == {
        "name": "ride_distance",
        "status": "computed",
        "unit": "km",
        "range": [0.0, 2000.0],
        "kind": "scalar",
        "value": 12.3,
        "flag": None,
    }
    assert by_name["fuel_trim_map"] == {
        "name": "fuel_trim_map",
        "status": "not_ready",
        "reason": "ready_phase0",
        "pending_inputs": ["fuel-system status"],
    }
    assert by_name["ride_fuel_used"] == {
        "name": "ride_fuel_used",
        "status": "missing_inputs",
        "missing": ["did.ENGINE_FUEL_RATE"],
    }


def test_summary_binned_cells_and_axes(settings: Settings, client: TestClient):
    signals = _summary({})
    signals["ride_coolant_time_at_temp"] = {
        "status": "computed",
        "unit": "s",
        "cells": 10,
        "nonzero_cells": 1,
        "total_s": 10.0,
        "outside_s": 2.0,
    }
    _write_report(settings, SESSION_ID, _block(signals))
    values = [1.0] + [None] * 9
    _write_rows(settings, SESSION_ID, _binned_rows(SESSION_ID, values))

    body = _get_summary(client).json()
    entry = next(e for e in body["signals"] if e["name"] == "ride_coolant_time_at_temp")
    spec = vehicle_derived.SIGNALS["ride_coolant_time_at_temp"]
    axis = spec["bins"][0]
    assert entry["kind"] == "binned"
    assert entry["range"] == list(spec["range"])
    assert entry["total_s"] == 10.0
    assert entry["outside_s"] == 2.0
    assert entry["axes"] == [
        {
            "input": "did.COOLANT_TEMP",
            "unit": "degC",
            "edges": [list(bin_edges(axis, i)) for i in range(axis["count"])],
        }
    ]
    assert len(entry["cells"]) == 10
    assert entry["cells"][0] == {"index": 0, "axes": [0], "value": 1.0, "flag": None}
    assert entry["cells"][1] == {"index": 1, "axes": [1], "value": None, "flag": None}
    assert "defs_mismatch" not in entry


def test_summary_two_axis_bins_are_row_major(settings: Settings, client: TestClient):
    spec = vehicle_derived.SIGNALS["operating_point_time"]
    bins = spec["bins"]
    rows = []
    for cell in range(cell_count(bins)):
        indices = cell_axes(bins, cell)
        edges = [bin_edges(bins[axis], index) for axis, index in enumerate(indices)]
        rows.append(
            {
                "session_id": SESSION_ID,
                "record": spec["record"],
                "did": 1,
                "signal": "operating_point_time",
                "cell": cell,
                "bin0_low": edges[0][0],
                "bin0_high": edges[0][1],
                "bin1_low": edges[1][0],
                "bin1_high": edges[1][1],
                "value": float(cell),
                "unit": spec["unit"],
                "valid": True,
                "flag": None,
            }
        )
    signals = _summary({})
    signals["operating_point_time"] = {
        "status": "computed",
        "unit": spec["unit"],
        "cells": cell_count(bins),
        "nonzero_cells": 1,
        "total_s": 1.0,
        "outside_s": 0.0,
    }
    _write_report(settings, SESSION_ID, _block(signals))
    _write_rows(settings, SESSION_ID, rows)
    entry = next(
        e for e in _get_summary(client).json()["signals"] if e["name"] == "operating_point_time"
    )
    assert [axis["input"] for axis in entry["axes"]] == ["did.ENGINE_SPEED", "did.ENGINE_LOAD"]
    assert [axis["unit"] for axis in entry["axes"]] == ["rpm", "%"]
    assert entry["axes"][0]["edges"][1] == [750.0, 1500.0]
    assert entry["axes"][1]["edges"][1] == [6.25, 12.5]
    assert entry["cells"][16 * 16 - 1] == {
        "index": 255,
        "axes": [15, 15],
        "value": 255.0,
        "flag": None,
    }
    assert "defs_mismatch" not in entry


def test_summary_returns_stored_edges_and_marks_a_defs_mismatch(
    settings: Settings, client: TestClient
):
    signals = _summary({})
    signals["ride_coolant_time_at_temp"] = {
        "status": "computed",
        "unit": "s",
        "cells": 10,
        "nonzero_cells": 1,
        "total_s": 10.0,
        "outside_s": 2.0,
    }
    _write_report(settings, SESSION_ID, _block(signals))
    _write_rows(
        settings,
        SESSION_ID,
        _binned_rows(SESSION_ID, [1.0] + [None] * 9, shift_first_edge=True),
    )
    entry = next(
        e
        for e in _get_summary(client).json()["signals"]
        if e["name"] == "ride_coolant_time_at_temp"
    )
    assert entry["defs_mismatch"] is True
    assert entry["axes"][0]["edges"][0] == [0.0, 40.0]


def test_summary_scalar_values_agree_with_the_parquet_rows(
    settings: Settings, client: TestClient, fixtures_dir: Path
):
    session_id = ingest_path(fixtures_dir / "v4_session", settings).session_id
    body = _get_summary(client, session_id).json()
    parquet = settings.sessions_dir / session_id / "parquet" / "derived_summary.parquet"
    rows = pq.read_table(parquet).to_pylist()
    scalar_rows = {
        row["signal"]: row
        for row in rows
        if row["cell"] is None and row["signal"] in vehicle_derived.SIGNALS
    }
    for entry in body["signals"]:
        if entry.get("status") == "computed" and entry.get("kind") == "scalar":
            row = scalar_rows[entry["name"]]
            assert row["value"] == entry["value"]
            assert row["flag"] == entry["flag"]
            assert row["unit"] == entry["unit"]


def test_rides_requires_token(client: TestClient):
    assert client.get("/rides").status_code == 401


def _seed_rides(settings: Settings) -> None:
    computed = _summary(
        {
            "ride_coolant_rate_max": {
                "status": "computed",
                "value": None,
                "unit": "degC/min",
                "flag": "invalid_input",
            }
        }
    )
    _write_report(settings, SESSION_ID, _block(computed))
    _add_index(settings, SESSION_ID, "2026-01-03T10:00:00+00:00")

    not_ready = _summary(
        {"ride_distance": {"status": "not_ready", "reason": "ready_phase0", "pending_inputs": []}}
    )
    _write_report(settings, SECOND_ID, _block(not_ready))
    _add_index(settings, SECOND_ID, "2026-01-02T10:00:00+00:00")

    _add_index(settings, THIRD_ID, "2026-01-01T10:00:00+00:00")


def test_rides_returns_scalar_columns_and_skips_missing_summaries(
    settings: Settings, client: TestClient
):
    _seed_rides(settings)
    body = _get_rides(client).json()
    assert body["skipped"] == 1
    assert [row["session_id"] for row in body["rows"]] == [SECOND_ID, SESSION_ID]
    assert [col["name"] for col in body["columns"]] == ride_summary.scalar_names()
    first, second = body["rows"]
    assert first["values"]["ride_distance"] is None
    assert first["flags"]["ride_distance"] == "not_ready"
    assert second["values"]["ride_distance"] == 12.3
    assert second["values"]["ride_time"] == 90.0
    assert second["values"]["ride_coolant_rate_max"] is None
    assert second["flags"]["ride_coolant_rate_max"] == "invalid_input"


def test_rides_names_are_checked_and_keep_the_requested_order(
    settings: Settings, client: TestClient
):
    _seed_rides(settings)
    body = _get_rides(client, names="ride_time,ride_distance").json()
    assert [col["name"] for col in body["columns"]] == ["ride_time", "ride_distance"]
    assert [col["unit"] for col in body["columns"]] == ["s", "km"]

    for params in (
        {"names": "nope"},
        {"names": "ride_distance,nope"},
        {"names": "ride_coolant_time_at_temp"},
        {"names": ""},
    ):
        resp = _get_rides(client, **params)
        assert resp.status_code == 422, params
    assert "nope" in _get_rides(client, names="nope").json()["detail"]
    assert (
        "binned summaries" in _get_rides(client, names="ride_coolant_time_at_temp").json()["detail"]
    )


def test_rides_date_range_is_inclusive_and_utc(settings: Settings, client: TestClient):
    _seed_rides(settings)
    body = _get_rides(client, **{"from": "2026-01-02", "to": "2026-01-02"}).json()
    assert [row["session_id"] for row in body["rows"]] == [SECOND_ID]
    body = _get_rides(
        client,
        **{"from": "2026-01-02T09:00:00+01:00", "to": "2026-01-02T23:59:59Z"},
    ).json()
    assert [row["session_id"] for row in body["rows"]] == [SECOND_ID]
    body = _get_rides(client, **{"from": "2026-01-04"}).json()
    assert body["rows"] == []


@pytest.mark.parametrize("value", ["not-a-date", "2026-13-01", "2026-01-01T99:00:00"])
def test_rides_invalid_date_is_422(client: TestClient, value: str):
    resp = _get_rides(client, **{"from": value})
    assert resp.status_code == 422


def test_rides_binned_columns_have_no_cell_values(settings: Settings, client: TestClient):
    _write_report(settings, SESSION_ID, _block(_summary({})))
    _write_rows(settings, SESSION_ID, _binned_rows(SESSION_ID, [1.0] + [None] * 9))
    _add_index(settings, SESSION_ID, "2026-01-03T10:00:00+00:00")
    body = _get_rides(client, names="ride_distance").json()
    assert body["rows"][0]["values"] == {"ride_distance": 12.3}
    assert list(body["rows"][0]["values"]) == ["ride_distance"]


def test_rides_missing_value_in_an_older_report_is_null_with_status(
    settings: Settings, client: TestClient
):
    _write_report(
        settings,
        SESSION_ID,
        _block({"ride_distance": {"status": "computed", "value": 1.0, "unit": "km", "flag": None}}),
    )
    _add_index(settings, SESSION_ID, "2026-01-03T10:00:00+00:00")
    body = _get_rides(client, names="ride_distance,ride_time").json()
    assert body["rows"][0]["values"] == {"ride_distance": 1.0, "ride_time": None}
    assert body["rows"][0]["flags"] == {"ride_distance": None, "ride_time": "missing"}


def test_rides_skipped_counts_only_sessions_inside_the_range():
    block = {"records": {"r": {"signals": {}}}}
    sessions = [
        ("out-old", "2026-01-01T10:00:00+00:00", None),
        ("in-new", "2026-02-01T10:00:00+00:00", None),
        ("in-ok", "2026-02-02T10:00:00+00:00", block),
    ]
    start = ride_summary.parse_utc_bound("2026-02-01", end=False)
    body = ride_summary.build_rides(sessions, [], start, None)
    assert [row["session_id"] for row in body["rows"]] == ["in-ok"]
    assert body["skipped"] == 1
