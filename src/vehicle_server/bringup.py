"""Bring-up gate check of a stored session (GET /sessions/{id}/bringup, `vehicle-server bringup`).

Answers whether a session passes gate B0, V0 or V1 of vehicle-defs
`docs/hardware-integration.md` §6, one `Check` per "Pass when" criterion, plus the §7.2
tester statistics as warnings. Read-only: the check functions take the stored report dict
and a few values the caller read from the session's parquet files (`load_inputs`); they do
no I/O themselves. GPS is never read (D-076).

Statuses: pass | fail | warn | manual | unknown. A criterion whose report field does not
exist (an older report, a v2 session without CAN health, no IMU, no v4 tester data) is
`unknown` with the reason, never `pass`. A criterion the doc gives no number for ("at rest",
"RPM changes with a throttle blip") is `manual`: the measured values are shown and the user
judges; a manual check never changes the verdict.

Every threshold comes from vehicle_defs or is quoted from §6 with its section.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from vehicle_server import ble_schema, signals
from vehicle_server.defs import vehicle_cl250
from vehicle_server.derived.inputs import SessionInputs, _unwrap
from vehicle_server.series import _stats

GATES = ("B0", "V0", "V1")
STATUSES = ("pass", "fail", "warn", "manual", "unknown")

_DOC = "hardware-integration.md"

# hardware-integration.md §6 V1: "CAN health clean for ≥ 5 min".
V1_MIN_DURATION_S = 5 * 60

# The engine-speed DID whose spread V1 shows for the throttle-blip check (defs name).
ENGINE_SPEED = "ENGINE_SPEED"
if ENGINE_SPEED not in vehicle_cl250.DIDS:
    raise ImportError(f"vehicle_defs has no DID {ENGINE_SPEED}")

# The bus-off state by its BLE schema enum name (canHealth.busState.values).
_BUS_OFF_STATE = "busOff"
if _BUS_OFF_STATE not in {
    value["name"] for value in ble_schema.load_default_schema()["canHealth"]["busState"]["values"]
}:
    raise ImportError(f"the BLE schema has no bus state {_BUS_OFF_STATE}")


@dataclass(frozen=True)
class Check:
    id: str
    status: str
    measured: Any
    expected: str
    source: str
    note: str


@dataclass(frozen=True)
class BringupInputs:
    """Values read from the session's parquet files by `load_inputs` (None = not available).

    can_health_duration_s: span of device_time_ms over the telemetry rows that carry CAN
    health (v3+), wrap removed; None with fewer than two such rows.
    engine_speed: count/min/p50/max of the ENGINE_SPEED series (the /series and /stats
    input), or None when the session has no valid sample.
    """

    can_health_duration_s: float | None = None
    engine_speed: dict[str, float] | None = None


@dataclass(frozen=True)
class BringupResult:
    session_id: str
    gate: str
    verdict: str
    checks: list[Check]
    needs_you: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """The one JSON shape of the CLI (--json) and the API."""
        return asdict(self)


def _polled_signals() -> list[str]:
    """The defs DIDs the telemetry packet carries (the BLE schema's defsSignal fields)."""
    return signals.known_defs_signals(ble_schema.load_default_schema())


# --- the criteria -------------------------------------------------------------


def _report_status(report: dict) -> Check:
    status = report.get("status")
    mapped = {"ok": "pass", "warn": "warn", "fail": "fail"}.get(status, "unknown")
    return Check(
        "report_status",
        mapped,
        {"status": status, "reasons": report.get("reasons", [])},
        "report status ok",
        f"{_DOC} §6 B0",
        "The session report's own verdict; its reasons list what is not ok.",
    )


def _imu_gaps(report: dict) -> Check:
    imu = report.get("imu")
    source = f"{_DOC} §6 B0"
    expected = "IMU gap count 0"
    if not imu:
        return Check("imu_gap_count", "unknown", None, expected, source, "The session has no IMU.")
    missing = imu["missing_samples"]
    return Check(
        "imu_gap_count",
        "pass" if missing == 0 else "fail",
        {"missing_samples": missing, "loss_percent": imu["loss_percent"]},
        expected,
        source,
        "IMU samples missing from sample_index (report imu.missing_samples).",
    )


def _at_rest(report: dict) -> Check:
    imu = report.get("imu")
    measured = (
        None
        if not imu
        else {
            "missing_samples": imu["missing_samples"],
            "effective_rate_hz": imu["effective_rate_hz"],
        }
    )
    return Check(
        "at_rest",
        "manual",
        measured,
        "the unit lay still on the desk for the whole recording",
        f"{_DOC} §6 B0",
        "The doc gives no number for 'at rest'; confirm the unit was not moved.",
    )


def _can_health_unknown(check_id: str, expected: str, source: str, report: dict) -> Check | None:
    if report.get("can_health"):
        return None
    return Check(
        check_id,
        "unknown",
        None,
        expected,
        source,
        "The session has no CAN health rows (telemetry v3+ needed).",
    )


def _ecu_present(report: dict, gate: str) -> Check:
    expected = "ecuPresent set on every CAN health row"
    source = f"{_DOC} §6 {gate}"
    missing = _can_health_unknown("ecu_present", expected, source, report)
    if missing:
        return missing
    can = report["can_health"]
    if "ecu_present_rows" not in can:
        return Check(
            "ecu_present",
            "unknown",
            None,
            expected,
            source,
            "The report was built before it counted ECU presence (ecu_present_rows).",
        )
    present, absent = can["ecu_present_rows"], can["ecu_absent_rows"]
    if not present and not absent:
        return Check(
            "ecu_present",
            "unknown",
            {"present_rows": 0, "absent_rows": 0, "last": can["ecu_present_last"]},
            expected,
            source,
            "No CAN health row carried a decodable ecuPresent flag.",
        )
    return Check(
        "ecu_present",
        "pass" if present and not absent else "fail",
        {"present_rows": present, "absent_rows": absent, "last": can["ecu_present_last"]},
        expected,
        source,
        "absent_rows = rows with the ecuPresent flag clear; telemetry.parquet shows where.",
    )


def _dids_valid(report: dict, gate: str) -> Check:
    expected = "every polled DID has a valid sample and no out-of-range sample"
    source = f"{_DOC} §6 {gate}; DIDs from the BLE schema defsSignal fields"
    valid_ratios = report.get("valid_ratios") or {}
    range_checks = report.get("range_checks") or {}
    measured: dict[str, Any] = {}
    status = "pass"
    problems: list[str] = []
    for defs_signal in _polled_signals():
        key = signals.signal_key(defs_signal)
        ratio, ranges = valid_ratios.get(key), range_checks.get(key)
        if ratio is None or ranges is None:
            measured[defs_signal] = None
            problems.append(f"{defs_signal} not in the report")
            if status == "pass":
                status = "unknown"
            continue
        measured[defs_signal] = {
            "valid_percent": ratio["valid_percent"],
            "valid_count": ratio["valid_count"],
            "out_of_range_count": ranges["out_of_range_count"],
        }
        if ratio["valid_count"] == 0 or ranges["out_of_range_count"]:
            status = "fail"
            problems.append(defs_signal)
    note = (
        "All polled DIDs valid."
        if not problems
        else f"Not valid or out of range: {', '.join(problems)}."
    )
    return Check("dids_valid", status, measured, expected, source, note)


def _no_latch(report: dict, gate: str) -> Check:
    expected = "latchedForeignTester and latchedBusOff never set"
    source = f"{_DOC} §6 {gate}"
    missing = _can_health_unknown("no_latch", expected, source, report)
    if missing:
        return missing
    can = report["can_health"]
    foreign, bus_off = can["latched_foreign_tester"], can["latched_bus_off"]
    return Check(
        "no_latch",
        "fail" if foreign or bus_off else "pass",
        {"latched_foreign_tester": foreign, "latched_bus_off": bus_off},
        expected,
        source,
        "A latch stops the poller until reboot (D-030).",
    )


def _no_bus_off(report: dict, gate: str) -> Check:
    expected = f"bus-off count 0 and no {_BUS_OFF_STATE} state"
    source = f"{_DOC} §6 {gate}"
    missing = _can_health_unknown("no_bus_off", expected, source, report)
    if missing:
        return missing
    can = report["can_health"]
    count, states = can["max_bus_off_count"], can["bus_states_seen"]
    return Check(
        "no_bus_off",
        "fail" if count or _BUS_OFF_STATE in states else "pass",
        {"max_bus_off_count": count, "bus_states_seen": states},
        expected,
        source,
        "busOffCount is cumulative since the node's boot.",
    )


def _duration(report: dict, extra: BringupInputs) -> Check:
    expected = f"CAN health rows span >= {V1_MIN_DURATION_S} s"
    source = f"{_DOC} §6 V1"
    resets = (report.get("ordering") or {}).get("device_time_ms_resets") or []
    if resets:
        return Check(
            "duration",
            "unknown",
            {"device_time_ms_resets": len(resets)},
            expected,
            source,
            "The node rebooted during the session, so its clock span is not one run.",
        )
    seconds = extra.can_health_duration_s
    if seconds is None:
        return Check(
            "duration",
            "unknown",
            None,
            expected,
            source,
            "Fewer than two telemetry rows with CAN health.",
        )
    return Check(
        "duration",
        "pass" if seconds >= V1_MIN_DURATION_S else "fail",
        {"seconds": seconds},
        expected,
        source,
        "Span of the node clock over the CAN health rows; 'clean' is the latch and bus-off "
        "checks above, which cover the whole session.",
    )


def _throttle_blip(extra: BringupInputs) -> Check:
    if extra.engine_speed is None:
        return Check(
            "rpm_throttle_blip",
            "unknown",
            None,
            f"{ENGINE_SPEED} rises with a throttle blip and falls back to idle",
            f"{_DOC} §6 V1",
            f"The session has no valid {ENGINE_SPEED} sample.",
        )
    return Check(
        "rpm_throttle_blip",
        "manual",
        extra.engine_speed,
        f"{ENGINE_SPEED} rises with a throttle blip and falls back to idle",
        f"{_DOC} §6 V1",
        "The doc gives no number; check the spread here or the series in the analyzer.",
    )


def _step_gap(report: dict) -> Check:
    tester = report.get("tester_stats")
    expected = (
        f"no poller step gap above CLIENT_STEP_MAX_MS ({vehicle_cl250.CLIENT_STEP_MAX_MS} ms)"
    )
    source = f"{_DOC} §7.2; vehicle_cl250.CLIENT_STEP_MAX_MS"
    if not tester:
        return Check("step_gap", "unknown", None, expected, source, "No v4 tester statistics.")
    over = tester["step_gap_over_count"]
    return Check(
        "step_gap",
        "warn" if over else "pass",
        {"step_gap_max_ms": tester["step_gap_max_ms"], "step_gap_over_count": over},
        expected,
        source,
        (
            f"The D-053 {vehicle_cl250.CLIENT_STEP_MAX_MS} ms step assumption does not hold "
            "for conn in this session; compare with the G0.1 serial report."
            if over
            else "Every poller step gap was within the assumption."
        ),
    )


def _round_trip(report: dict) -> Check:
    tester = report.get("tester_stats")
    limit = vehicle_cl250.ASSUMED_ROUND_TRIP_MS
    expected = f"every DID's round-trip max_ms <= ASSUMED_ROUND_TRIP_MS ({limit} ms)"
    source = f"{_DOC} §7.2; vehicle_cl250.ASSUMED_ROUND_TRIP_MS"
    if not tester:
        return Check("round_trip", "unknown", None, expected, source, "No v4 tester statistics.")
    rtt = tester["rtt"]
    if not any(entry["max_ms"] is not None for entry in rtt.values()):
        return Check("round_trip", "unknown", rtt, expected, source, "No round-trip sample yet.")
    slow = [
        entry["name"] or did
        for did, entry in rtt.items()
        if entry["max_ms"] is not None and entry["max_ms"] > limit
    ]
    measured = {entry["name"] or did: entry["max_ms"] for did, entry in rtt.items()}
    return Check(
        "round_trip",
        "warn" if slow else "pass",
        measured,
        expected,
        source,
        (
            f"Slower than assumed: {', '.join(slow)}; this changes the D-029 budget (A-4)."
            if slow
            else "Every DID answered within the assumed round trip."
        ),
    )


# --- the gates -----------------------------------------------------------------


def _can_checks(report: dict, gate: str) -> list[Check]:
    return [_ecu_present(report, gate), _no_latch(report, gate), _no_bus_off(report, gate)]


def _gate_checks(gate: str, report: dict, extra: BringupInputs) -> list[Check]:
    if gate == "B0":
        return [_report_status(report), _imu_gaps(report), _at_rest(report)]
    if gate == "V0":
        ecu, latch, bus_off = _can_checks(report, gate)
        return [
            ecu,
            _dids_valid(report, gate),
            latch,
            bus_off,
            _step_gap(report),
            _round_trip(report),
        ]
    return [
        *_can_checks(report, gate),
        _duration(report, extra),
        _throttle_blip(extra),
        _step_gap(report),
        _round_trip(report),
    ]


def check_gate(gate: str, report: dict, extra: BringupInputs | None = None) -> BringupResult:
    """The gate verdict of one stored report: fail > incomplete (an unknown) > pass."""
    gate = gate.upper()
    if gate not in GATES:
        raise ValueError(f"unknown gate {gate!r}; expected one of {', '.join(GATES)}")
    checks = _gate_checks(gate, report, extra or BringupInputs())
    statuses = {check.status for check in checks}
    if "fail" in statuses:
        verdict = "fail"
    elif "unknown" in statuses:
        verdict = "incomplete"
    else:
        verdict = "pass"
    return BringupResult(
        session_id=report.get("session_id", ""),
        gate=gate,
        verdict=verdict,
        checks=checks,
        needs_you=[check.id for check in checks if check.status == "manual"],
        warnings=[check.id for check in checks if check.status == "warn"],
    )


def format_text(result: BringupResult) -> str:
    lines = [f"Bring-up gate {result.gate} for {result.session_id}"]
    width = max(len(check.id) for check in result.checks)
    for check in result.checks:
        lines.append(
            f"  {check.status.upper():<7} {check.id:<{width}}  measured={check.measured}  "
            f"expected: {check.expected}"
        )
    verdict = result.verdict
    if result.needs_you:
        verdict += (
            f", pending {len(result.needs_you)} manual check(s): {', '.join(result.needs_you)}"
        )
    if result.warnings:
        verdict += f"; warnings: {', '.join(result.warnings)}"
    lines.append(f"Verdict: {verdict}")
    return "\n".join(lines) + "\n"


# --- reading the session (the only I/O, done by the CLI and the API) -----------


def _can_health_duration_s(session_dir: Path) -> float | None:
    path = session_dir / "parquet" / "telemetry.parquet"
    if not path.is_file() or "can_bus_state" not in pq.read_schema(path).names:
        return None  # no telemetry, or stored before the CAN health columns existed
    table = pq.read_table(path, columns=["device_time_ms", "can_bus_state"])
    times = [
        t
        for t, state in zip(
            table["device_time_ms"].to_pylist(), table["can_bus_state"].to_pylist(), strict=True
        )
        if t is not None and state is not None
    ]
    if len(times) < 2:
        return None
    unwrapped = _unwrap(times)
    return (unwrapped[-1] - unwrapped[0]) / 1000.0


def _engine_speed_stats(session_dir: Path) -> dict[str, float] | None:
    series = SessionInputs(session_dir).series(f"did.{ENGINE_SPEED}")
    if series is None:
        return None
    stats = _stats(series.values)
    return {key: stats[key] for key in ("count", "min", "p50", "max")}


def load_inputs(session_dir: Path) -> BringupInputs:
    """Reads what V1 needs from telemetry.parquet / signals.parquet (never gps.parquet)."""
    return BringupInputs(
        can_health_duration_s=_can_health_duration_s(session_dir),
        engine_speed=_engine_speed_stats(session_dir),
    )
