from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from vehicle_server import cli


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]) -> int:
    monkeypatch.setenv("VEHICLE_DATA_DIR", str(tmp_path / "data"))
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)
    return exc_info.value.code


def test_cli_import_ok_exits_zero_even_with_warnings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixtures_dir: Path, capsys
):
    code = _run(monkeypatch, tmp_path, ["import", str(fixtures_dir / "v2_session")])
    assert code == 0
    out = capsys.readouterr().out
    assert "20260115-120000-a1b2" in out


def test_cli_import_conflict_exits_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixtures_dir: Path
):
    _run(monkeypatch, tmp_path, ["import", str(fixtures_dir / "v2_session")])

    mutated_dir = tmp_path / "mutated"
    shutil.copytree(fixtures_dir / "v2_session", mutated_dir)
    events_path = mutated_dir / "events.csv"
    events_path.write_text(events_path.read_text() + "2026-01-15T12:00:07.000Z,1700,note,changed\n")

    code = _run(monkeypatch, tmp_path, ["import", str(mutated_dir)])
    assert code == 2


def test_cli_import_invalid_exits_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixtures_dir: Path
):
    mutated_dir = tmp_path / "mutated"
    shutil.copytree(fixtures_dir / "v2_session", mutated_dir)
    (mutated_dir / "summary.json").unlink()

    code = _run(monkeypatch, tmp_path, ["import", str(mutated_dir)])
    assert code == 1


def test_cli_report_after_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixtures_dir: Path, capsys
):
    _run(monkeypatch, tmp_path, ["import", str(fixtures_dir / "v3_session")])
    code = _run(monkeypatch, tmp_path, ["report", "20260115-130000-c3d4", "--format", "json"])
    assert code == 0
    out = capsys.readouterr().out
    assert "20260115-130000-c3d4" in out


def test_cli_report_unknown_session_exits_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    code = _run(monkeypatch, tmp_path, ["report", "00000000-000000-ffff"])
    assert code == 1


def test_cli_list_after_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixtures_dir: Path, capsys
):
    _run(monkeypatch, tmp_path, ["import", str(fixtures_dir / "v2_session")])
    code = _run(monkeypatch, tmp_path, ["list"])
    assert code == 0
    out = capsys.readouterr().out
    assert "20260115-120000-a1b2" in out


@pytest.mark.parametrize(
    ("fixture", "session_id", "gate", "code"),
    [
        ("v4_session", "20260115-140000-e5f6", "V0", 0),  # pass (tester stats only warn)
        ("v3_session", "20260115-130000-c3d4", "V0", 1),  # latched bus-off -> fail
        ("v2_session", "20260115-120000-a1b2", "V0", 2),  # no CAN health -> incomplete
    ],
)
def test_cli_bringup_exit_code_follows_the_verdict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixtures_dir: Path,
    capsys,
    fixture,
    session_id,
    gate,
    code,
):
    _run(monkeypatch, tmp_path, ["import", str(fixtures_dir / fixture)])
    capsys.readouterr()
    assert _run(monkeypatch, tmp_path, ["bringup", session_id, "--gate", gate]) == code
    out = capsys.readouterr().out
    assert f"Bring-up gate {gate} for {session_id}" in out
    assert "Verdict:" in out


def test_cli_bringup_json_and_lower_case_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fixtures_dir: Path, capsys
):
    _run(monkeypatch, tmp_path, ["import", str(fixtures_dir / "v4_session")])
    capsys.readouterr()
    _run(monkeypatch, tmp_path, ["bringup", "20260115-140000-e5f6", "--gate", "v1", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["gate"] == "V1"
    assert data["needs_you"] == ["rpm_throttle_blip"]


def test_cli_bringup_unknown_session_exits_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    assert _run(monkeypatch, tmp_path, ["bringup", "00000000-000000-ffff", "--gate", "B0"]) == 1


def test_cli_bringup_rejects_an_unknown_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    assert _run(monkeypatch, tmp_path, ["bringup", "00000000-000000-ffff", "--gate", "V2"]) == 2
