"""Bring-up gate check (vehicle_server.bringup): pure checks on synthetic report dicts."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from vehicle_server import bringup
from vehicle_server.defs import vehicle_cl250
from vehicle_server.ingest import ingest_path

from .conftest import copy_fixture

POLLED = ["ENGINE_SPEED", "VEHICLE_SPEED", "COOLANT_TEMP", "THROTTLE_POS", "BATTERY_VOLTAGE"]
LONG_ENOUGH = bringup.BringupInputs(
    can_health_duration_s=float(bringup.V1_MIN_DURATION_S),
    engine_speed={"count": 10, "min": 1200.0, "p50": 1400.0, "max": 3000.0},
)


def _clean_report() -> dict:
    """A report every gate passes (apart from its manual checks)."""
    return {
        "session_id": "20260115-140000-e5f6",
        "status": "ok",
        "reasons": ["no issues found"],
        "ordering": {"rx_mono_ms_violations": [], "device_time_ms_resets": []},
        "valid_ratios": {
            name.lower(): {"total_count": 10, "valid_count": 10, "valid_percent": 100.0}
            for name in POLLED
        },
        "range_checks": {
            name.lower(): {"checked_count": 10, "out_of_range_count": 0, "out_of_range_percent": 0}
            for name in POLLED
        },
        "can_health": {
            "max_tec": 0,
            "max_rec": 0,
            "max_bus_off_count": 0,
            "max_unanswered_did_count": 0,
            "latched_foreign_tester": False,
            "latched_bus_off": False,
            "ecu_present_rows": 10,
            "ecu_absent_rows": 0,
            "ecu_present_last": True,
            "bus_states_seen": ["running"],
        },
        "tester_stats": {
            "step_gap_max_ms": vehicle_cl250.CLIENT_STEP_MAX_MS,
            "step_gap_over_count": 0,
            "rtt": {
                "0xF40C": {
                    "did": 0xF40C,
                    "name": "ENGINE_SPEED",
                    "min_ms": 5,
                    "max_ms": vehicle_cl250.ASSUMED_ROUND_TRIP_MS,
                    "avg_ms": 8.0,
                    "count": 10,
                    "nrc78_count": 0,
                }
            },
        },
        "imu": {
            "sample_count": 100,
            "missing_samples": 0,
            "loss_percent": 0.0,
            "effective_rate_hz": 100.0,
        },
    }


def _status(result: bringup.BringupResult, check_id: str) -> str:
    return next(check.status for check in result.checks if check.id == check_id)


@pytest.mark.parametrize("gate", bringup.GATES)
def test_clean_session_passes_every_gate(gate: str):
    result = bringup.check_gate(gate, _clean_report(), LONG_ENOUGH)
    assert result.verdict == "pass"
    assert result.warnings == []
    assert result.gate == gate


def test_gate_is_case_insensitive_and_unknown_gate_raises():
    assert bringup.check_gate("v0", _clean_report()).gate == "V0"
    with pytest.raises(ValueError, match="unknown gate"):
        bringup.check_gate("V2", _clean_report())


def test_check_order_is_fixed():
    ids = [check.id for check in bringup.check_gate("V1", _clean_report(), LONG_ENOUGH).checks]
    assert ids == [
        "ecu_present",
        "no_latch",
        "no_bus_off",
        "duration",
        "rpm_throttle_blip",
        "step_gap",
        "round_trip",
    ]


@pytest.mark.parametrize(
    ("gate", "check_id", "mutate"),
    [
        ("B0", "report_status", lambda r: r.update(status="fail")),
        ("B0", "imu_gap_count", lambda r: r["imu"].update(missing_samples=1)),
        ("V0", "ecu_present", lambda r: r["can_health"].update(ecu_absent_rows=1)),
        ("V0", "ecu_present", lambda r: r["can_health"].update(ecu_present_rows=0)),
        ("V0", "dids_valid", lambda r: r["valid_ratios"]["coolant_temp"].update(valid_count=0)),
        (
            "V0",
            "dids_valid",
            lambda r: r["range_checks"]["throttle_pos"].update(out_of_range_count=1),
        ),
        ("V0", "no_latch", lambda r: r["can_health"].update(latched_foreign_tester=True)),
        ("V0", "no_latch", lambda r: r["can_health"].update(latched_bus_off=True)),
        ("V0", "no_bus_off", lambda r: r["can_health"].update(max_bus_off_count=1)),
        ("V0", "no_bus_off", lambda r: r["can_health"].update(bus_states_seen=["busOff"])),
        ("V1", "ecu_present", lambda r: r["can_health"].update(ecu_absent_rows=2)),
        ("V1", "no_latch", lambda r: r["can_health"].update(latched_bus_off=True)),
        ("V1", "no_bus_off", lambda r: r["can_health"].update(max_bus_off_count=3)),
    ],
)
def test_each_criterion_failing_alone_fails_the_gate(gate, check_id, mutate):
    report = _clean_report()
    mutate(report)
    result = bringup.check_gate(gate, report, LONG_ENOUGH)
    assert result.verdict == "fail"
    assert [c.id for c in result.checks if c.status == "fail"] == [check_id]


def test_report_warn_is_a_warning_not_a_failure():
    report = _clean_report()
    report["status"] = "warn"
    result = bringup.check_gate("B0", report)
    assert result.verdict == "pass"
    assert result.warnings == ["report_status"]


def test_short_session_fails_v1_duration():
    short = bringup.BringupInputs(can_health_duration_s=bringup.V1_MIN_DURATION_S - 0.1)
    result = bringup.check_gate("V1", _clean_report(), short)
    assert result.verdict == "fail"
    assert _status(result, "duration") == "fail"


def test_device_time_reset_makes_duration_unknown():
    report = _clean_report()
    report["ordering"]["device_time_ms_resets"] = [{"row": 3, "prev": 9000, "curr": 10}]
    result = bringup.check_gate("V1", report, LONG_ENOUGH)
    assert _status(result, "duration") == "unknown"
    assert result.verdict == "incomplete"


@pytest.mark.parametrize("gate", ["V0", "V1"])
def test_tester_statistics_only_warn(gate: str):
    report = _clean_report()
    report["tester_stats"]["step_gap_over_count"] = 4
    report["tester_stats"]["rtt"]["0xF40C"]["max_ms"] = vehicle_cl250.ASSUMED_ROUND_TRIP_MS + 1
    result = bringup.check_gate(gate, report, LONG_ENOUGH)
    assert result.verdict == "pass"
    assert result.warnings == ["step_gap", "round_trip"]
    round_trip = next(c for c in result.checks if c.id == "round_trip")
    assert "ENGINE_SPEED" in round_trip.note


def test_report_without_ecu_presence_keys_is_incomplete():
    report = _clean_report()
    for key in ("ecu_present_rows", "ecu_absent_rows", "ecu_present_last"):
        del report["can_health"][key]
    result = bringup.check_gate("V0", report)
    assert _status(result, "ecu_present") == "unknown"
    assert result.verdict == "incomplete"


@pytest.mark.parametrize(
    ("gate", "block", "check_ids"),
    [
        ("B0", "imu", ["imu_gap_count"]),
        ("V0", "can_health", ["ecu_present", "no_latch", "no_bus_off"]),
        ("V0", "tester_stats", ["step_gap", "round_trip"]),
        ("V1", "can_health", ["ecu_present", "no_latch", "no_bus_off"]),
    ],
)
def test_missing_data_is_unknown_with_a_reason(gate, block, check_ids):
    report = _clean_report()
    report[block] = None
    result = bringup.check_gate(gate, report, LONG_ENOUGH)
    assert result.verdict == "incomplete"
    unknown = [c for c in result.checks if c.status == "unknown"]
    assert [c.id for c in unknown] == check_ids
    assert all(c.note for c in unknown)


def test_missing_duration_is_unknown():
    result = bringup.check_gate("V1", _clean_report(), bringup.BringupInputs())
    assert _status(result, "duration") == "unknown"


def test_manual_checks_never_change_the_verdict():
    for gate, manual in (("B0", ["at_rest"]), ("V1", ["rpm_throttle_blip"])):
        result = bringup.check_gate(gate, _clean_report(), LONG_ENOUGH)
        assert result.needs_you == manual
        assert result.verdict == "pass"
        assert f"pass, pending {len(manual)} manual check(s)" in bringup.format_text(result)


def test_v1_shows_the_engine_speed_spread():
    result = bringup.check_gate("V1", _clean_report(), LONG_ENOUGH)
    blip = next(c for c in result.checks if c.id == "rpm_throttle_blip")
    assert blip.measured == LONG_ENOUGH.engine_speed
    assert "ENGINE_SPEED" in blip.expected


def test_polled_dids_come_from_defs(monkeypatch: pytest.MonkeyPatch):
    assert bringup._polled_signals() == POLLED  # the BLE schema's defsSignal fields
    monkeypatch.setattr(bringup, "_polled_signals", lambda: ["ENGINE_SPEED", "ENGINE_LOAD"])
    result = bringup.check_gate("V0", _clean_report())
    dids = next(c for c in result.checks if c.id == "dids_valid")
    assert list(dids.measured) == ["ENGINE_SPEED", "ENGINE_LOAD"]
    assert dids.status == "unknown"  # ENGINE_LOAD is not in the report


def test_format_text_has_one_line_per_check_and_a_verdict():
    result = bringup.check_gate("V0", _clean_report())
    lines = bringup.format_text(result).splitlines()
    assert len(lines) == 1 + len(result.checks) + 1
    assert lines[1].split()[0] == "PASS"
    assert lines[-1] == "Verdict: pass"


def test_to_dict_is_json_shaped():
    data = bringup.check_gate("B0", copy.deepcopy(_clean_report())).to_dict()
    assert set(data) == {"session_id", "gate", "verdict", "checks", "needs_you", "warnings"}
    assert set(data["checks"][0]) == {"id", "status", "measured", "expected", "source", "note"}


def test_load_inputs_reads_a_stored_v4_session(settings, fixtures_dir: Path, tmp_path: Path):
    session = copy_fixture(fixtures_dir / "v4_session", tmp_path / "v4_session")
    result = ingest_path(session, settings)
    inputs = bringup.load_inputs(settings.sessions_dir / result.report["session_id"])
    assert inputs.can_health_duration_s is not None and inputs.can_health_duration_s > 0
    assert inputs.engine_speed is not None
    assert inputs.engine_speed["min"] <= inputs.engine_speed["p50"] <= inputs.engine_speed["max"]


def test_load_inputs_of_a_v2_session_has_no_duration(settings, fixtures_dir: Path, tmp_path: Path):
    session = copy_fixture(fixtures_dir / "v2_session", tmp_path / "v2_session")
    result = ingest_path(session, settings)
    inputs = bringup.load_inputs(settings.sessions_dir / result.report["session_id"])
    assert inputs.can_health_duration_s is None
