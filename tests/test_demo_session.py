"""The synthetic demo session generator: deterministic output and every D-079 summary."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.make_demo_session import build
from vehicle_server.config import Settings
from vehicle_server.defs import vehicle_derived
from vehicle_server.ingest import ingest_path
from vehicle_server.position import is_position_name
from vehicle_server.ride_summary import load_ride_summary

SESSION_ID = "20260101-090000-5e55"
RAW_FILES = ("telemetry.csv", "signals.csv", "events.csv")


def _entries(report: dict) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for record in report["derived_summary"]["records"].values():
        entries.update(record["signals"])
    return entries


def _tree(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in sorted(path.iterdir()) if item.is_file()}


def test_demo_session_closes_every_window_and_is_deterministic(tmp_path: Path):
    session = build(tmp_path / "one")
    assert session.name == "synthetic-demo-20260101"

    again = build(tmp_path / "two")
    assert _tree(session) == _tree(again)

    zipped = build(tmp_path / "zip-one", as_zip=True)
    zipped_again = build(tmp_path / "zip-two", as_zip=True)
    assert zipped.name == "synthetic-demo-20260101.zip"
    assert zipped.read_bytes() == zipped_again.read_bytes()

    settings = Settings(data_dir=tmp_path / "data", api_token="test-token", max_upload_mb=100)
    result = ingest_path(session, settings)
    report = result.report

    assert report["status"] == "ok"
    assert "error" not in report["derived"]
    assert "error" not in report["derived_summary"]

    entries = _entries(report)
    for key, spec in vehicle_derived.WINDOWS.items():
        computed = [
            entries[name].get("value")
            for name in spec["signals"]
            if entries.get(name, {}).get("status") == "computed"
        ]
        assert any(value is not None for value in computed), key

    windows = report["derived_summary"]["health_windows"]["windows"]
    assert list(windows) == list(vehicle_derived.WINDOWS)
    for key, window in windows.items():
        assert window["instances"], key
        assert any(instance["qualifies"] for instance in window["instances"]), key

    reasons = {
        instance["close_reason"] for window in windows.values() for instance in window["instances"]
    }
    assert {"engine_zero", "unknown"} <= reasons

    summary = load_ride_summary(settings.sessions_dir / SESSION_ID, SESSION_ID)
    by_name = {entry["name"]: entry for entry in summary["signals"]}
    per_gear = by_name["ride_final_drive_ratio"]
    assert sum(gear["value"] is not None for gear in per_gear["gears"]) >= 3
    assert by_name["ride_stop_count"]["value"] >= 2

    raw = settings.sessions_dir / SESSION_ID / "raw"
    assert not (raw / "gps.csv").exists()
    for name in RAW_FILES:
        with (raw / name).open(newline="") as handle:
            header = next(csv.reader(handle))
        assert not any(is_position_name(column) for column in header), name

    assert zipped.is_file()
