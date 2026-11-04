"""Builds the per-session validation report (report.json) and its text rendering.

Re-derives everything from raw_hex / *_raw columns (never trusts the app's own
decoded columns as ground truth) and cross-checks the two, per the ingest
contract: "raw_hex is authoritative; the server re-decodes it with the schema
and ignores the app's decoded columns except for a consistency check."
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from vehicle_server import ble_schema, signals
from vehicle_server.decode import (
    VSIG_NEGATIVE_RESPONSE,
    VSIG_VALID,
    decode_gps_csv,
    decode_imu_csv,
    decode_signals_csv,
    decode_telemetry_csv,
    decoded_tester_columns,
    group_signal_blocks,
)
from vehicle_server.defs import ble as defs_ble
from vehicle_server.defs import defs_version

# rx_mono_ms / device_time_ms gaps larger than this multiple of notifyPeriodMs
# are reported (schema rule referenced by the ingest contract).
GAP_MULTIPLIER = 3

# A device_time_ms decrease is treated as the documented uint32 wraparound
# (~49.7 days) only if the previous value was already near the top of the
# range and the wrapped-around delta is itself plausible; anything else is a
# reboot/reset event, reported separately from ordering violations.
_UINT32_MAX = 2**32
_WRAP_NEAR_MAX_MARGIN = 1_000_000

# Tester-stat sentinels (schema `testerStats.roundTrip` and the field descriptions):
# min 65535 = no sample yet; sum/count at 0xFFFFFFFF are saturated, so no average.
_RTT_MIN_NO_SAMPLE = 0xFFFF
_UINT32_SATURATED = 0xFFFFFFFF


def _tester_stats_summary(rows: list[Any]) -> dict[str, Any] | None:
    """Last tester statistics of the session (version 4 rows only), or None.

    step_gap_*: the last version 4 row's values (they are since-boot counters).
    rtt: the packet carries one rotating DID record; keep the last record seen
    per DID (rtt_did 0 = no record in that packet), keyed by the DID in hex.
    """
    stats_rows = [decoded_tester_columns(row.decoded) for row in rows if row.decoded]
    stats_rows = [s for s in stats_rows if s["step_gap_max_ms"] is not None]
    if not stats_rows:
        return None

    last_by_did: dict[int, dict[str, int]] = {}
    for stats in stats_rows:
        if stats["rtt_did"]:
            last_by_did[stats["rtt_did"]] = stats

    rtt: dict[str, dict[str, Any]] = {}
    for did in sorted(last_by_did):
        stats = last_by_did[did]
        count = stats["rtt_count"]
        total = stats["rtt_sum_ms"]
        has_sample = count > 0
        average_valid = has_sample and _UINT32_SATURATED not in (count, total)
        rtt[f"0x{did:04X}"] = {
            "did": did,
            "name": signals.did_name(did),
            "min_ms": stats["rtt_min_ms"] if has_sample else None,
            "max_ms": stats["rtt_max_ms"] if has_sample else None,
            "avg_ms": total / count if average_valid else None,
            "count": count,
            "nrc78_count": stats["rtt_nrc78_count"],
        }

    last = stats_rows[-1]
    return {
        "step_gap_max_ms": last["step_gap_max_ms"],
        "step_gap_over_count": last["step_gap_over_count"],
        "rtt": rtt,
    }


# gpsBlock.fixType.rule: the speed check of D-060 item 4 uses only blocks with
# these fix types (schema value names) and flags.gnssFixOk set.
_GPS_SPEED_CHECK_FIX_TYPES = ("fix3d", "gnssDeadReckoning")


def _gps_summary(gps_csv: Path, schema: dict[str, Any]) -> dict[str, Any]:
    """GPS block statistics (D-060), all re-derived from raw_hex.

    `lost_or_skipped_blocks` comes from seq gaps, which count both blocks lost
    on the radio and blocks the node skipped because the MTU was too small
    (gpsBlock.sequenceRule, D-062 item 2), so it is not a radio-loss figure.
    The parseError / uartOverflow flags cover the interval since the previous
    block, sent or skipped.
    """
    gps_rows = decode_gps_csv(gps_csv, schema)
    decoded_rows = [row for row in gps_rows if row.decoded]

    lost_total = 0
    prev_seq = None
    for row in gps_rows:
        if prev_seq is not None:
            lost_total += (row.seq - prev_seq - 1) % 256
        prev_seq = row.seq
    total_expected = lost_total + len(gps_rows)

    effective_rate_hz = None
    if len(decoded_rows) >= 2:
        duration_ms = (
            decoded_rows[-1].decoded["device_time_ms"] - decoded_rows[0].decoded["device_time_ms"]
        ) % _UINT32_MAX
        if duration_ms:
            effective_rate_hz = (len(decoded_rows) - 1) / (duration_ms / 1000.0)

    fix_types: dict[str, int] = {}
    usable = 0
    for row in decoded_rows:
        name = row.decoded["fix_type_name"] or f"unknown({row.decoded['raw']['fixType']})"
        fix_types[name] = fix_types.get(name, 0) + 1
        if name in _GPS_SPEED_CHECK_FIX_TYPES and row.decoded["flags"]["gnssFixOk"]:
            usable += 1

    return {
        "block_count": len(gps_rows),
        "lost_or_skipped_blocks": lost_total,
        "lost_or_skipped_percent": (lost_total / total_expected * 100) if total_expected else 0.0,
        "effective_rate_hz": effective_rate_hz,
        "decode_error_rows": [row.row_index for row in gps_rows if row.decode_error],
        "mismatch_rows": [row.row_index for row in gps_rows if row.mismatches],
        "fix_types": dict(sorted(fix_types.items())),
        "speed_check_usable_blocks": usable,
        "speed_check_usable_percent": (usable / len(decoded_rows) * 100) if decoded_rows else 0.0,
        "blocks_with_parse_error": sum(row.decoded["flags"]["parseError"] for row in decoded_rows),
        "blocks_with_uart_overflow": sum(
            row.decoded["flags"]["uartOverflow"] for row in decoded_rows
        ),
    }


_VSIG_VALID = VSIG_VALID
_VSIG_NEGATIVE_RESPONSE = VSIG_NEGATIVE_RESPONSE


def _vsig_gap_events(events_csv: Path) -> tuple[list[tuple[int, int]], list[str]]:
    """The `vsig_gap` events of events.csv as (seq, missing) pairs, plus unparsable details."""
    gaps: list[tuple[int, int]] = []
    malformed: list[str] = []
    if not events_csv.exists():
        return gaps, malformed
    with events_csv.open(newline="") as fh:
        for record in csv.DictReader(fh):
            if record.get("event") != "vsig_gap":
                continue
            detail = record.get("detail") or ""
            try:
                parts = dict(item.split("=", 1) for item in detail.split(";"))
                gaps.append((int(parts["seq"]), int(parts["missing"])))
            except (KeyError, ValueError):
                malformed.append(detail)
    return gaps, malformed


def _vsig_consistency_issues(
    blocks: list[list[Any]],
    recomputed_lost: dict[int, int],
    lost_total: float,
    block_count: int,
    record_count: int,
    lost_percent: float,
    events_csv: Path,
    summary: dict[str, Any],
) -> list[str]:
    """Cross-checks the app's own figures (E-18 item 4) against the re-derived ones.

    The three places the app reports vehicle-signal loss must agree with each other
    and with `block_seq` (vehicleSignalBlock.sequenceRule): the `lost_since_prev`
    column of signals.csv, the `vsig_gap` events of events.csv and the
    `vehicle_signal_*` keys of summary.json. Keys the app did not write are skipped.
    """
    issues: list[str] = []

    # lost_since_prev: carried on the first row of a block only (0 on the others, the
    # session_recorder contract) and equal to the block_seq gap.
    app_lost_by_seq: dict[int, int] = {}
    for block in blocks:
        seq = next((r.block_seq for r in block if r.block_seq is not None), None)
        if seq is None:
            continue
        values = [r.app_lost_since_prev or 0 for r in block]
        if any(values[1:]):
            issues.append(f"block seq {seq}: lost_since_prev is non-zero beyond the first row")
        app_lost_by_seq[seq] = values[0]
    for seq, app_lost in app_lost_by_seq.items():
        expected = recomputed_lost.get(seq)
        if expected is not None and app_lost != expected:
            issues.append(
                f"block seq {seq}: lost_since_prev {app_lost} but block_seq gap is {expected}"
            )

    # vsig_gap events: exactly one per block that follows a gap, with the same count.
    events, malformed = _vsig_gap_events(events_csv)
    for detail in malformed:
        issues.append(f"vsig_gap event with unparsable detail {detail!r}")
    events_by_seq: dict[int, int] = {}
    for seq, missing in events:
        if seq in events_by_seq:
            issues.append(f"more than one vsig_gap event for block seq {seq}")
        events_by_seq[seq] = missing
    for seq, expected in recomputed_lost.items():
        if expected and seq not in events_by_seq:
            issues.append(f"no vsig_gap event for block seq {seq} ({expected} block(s) lost)")
        elif seq in events_by_seq and events_by_seq[seq] != expected:
            issues.append(
                f"vsig_gap event for block seq {seq} says missing={events_by_seq[seq]} "
                f"but block_seq gap is {expected}"
            )
    for seq in events_by_seq:
        if seq not in recomputed_lost:
            issues.append(f"vsig_gap event for block seq {seq} that is not in signals.csv")

    # summary.json vehicle_signal_* keys.
    expected_summary = {
        "vehicle_signal_block_count": block_count,
        "vehicle_signal_record_count": record_count,
        "vehicle_signal_lost_blocks": lost_total,
    }
    for key, expected in expected_summary.items():
        if key in summary and summary[key] != expected:
            issues.append(f"summary {key} is {summary[key]!r} but signals.csv gives {expected}")
    reported = summary.get("vehicle_signal_lost_percent")
    if reported is not None and (
        not isinstance(reported, int | float) or abs(reported - lost_percent) > 1e-3
    ):
        issues.append(
            f"summary vehicle_signal_lost_percent is {reported!r} but signals.csv gives "
            f"{lost_percent:.4f}"
        )
    return issues


def _vehicle_signals_summary(
    signals_csv: Path,
    schema: dict[str, Any],
    meta: dict[str, Any],
    events_csv: Path | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Vehicle-signal block statistics (D-066 item 4), all re-derived from did + raw.

    Blocks lost come from `block_seq` gaps (vehicleSignalBlock.sequenceRule), one block
    being the consecutive rows that share their block header. `signals` has one entry
    per DID seen, keyed by the DID in hex; a DID missing from the pinned catalog is
    listed in `unknown_dids` and kept with its states only.
    """
    rows = decode_signals_csv(signals_csv, schema)
    blocks = group_signal_blocks(rows)

    lost_total = 0
    recomputed_lost: dict[int, int] = {}  # block_seq -> blocks lost just before it
    prev_seq = None
    for block in blocks:
        seq = next((r.block_seq for r in block if r.block_seq is not None), None)
        if seq is None:
            continue
        if prev_seq is not None:
            lost = (seq - prev_seq - 1) % 256
            lost_total += lost
            recomputed_lost[seq] = lost
        prev_seq = seq

    per_did: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.decode_error or row.did is None:
            continue
        item = per_did.setdefault(
            row.did,
            {
                "did": row.did,
                "name": row.signal,
                "unit": row.unit,
                "records": 0,
                "states": {},
                "valid_count": 0,
                "min": None,
                "max": None,
                "last_value": None,
                "nrcs": set(),
            },
        )
        item["records"] += 1
        item["states"][row.state_name] = item["states"].get(row.state_name, 0) + 1
        if row.state == _VSIG_NEGATIVE_RESPONSE:  # raw holds the NRC
            item["nrcs"].add(row.raw)
        if row.state == _VSIG_VALID and row.value is not None:
            item["valid_count"] += 1
            item["min"] = row.value if item["min"] is None else min(item["min"], row.value)
            item["max"] = row.value if item["max"] is None else max(item["max"], row.value)
            item["last_value"] = row.value

    signals_by_did: dict[str, dict[str, Any]] = {}
    for did in sorted(per_did):
        item = per_did[did]
        item["states"] = dict(sorted(item["states"].items()))
        item["nrcs"] = [f"0x{nrc:02X}" for nrc in sorted(item["nrcs"])]
        signals_by_did[f"0x{did:04X}"] = item

    block_count = len(blocks)
    total_expected = lost_total + block_count
    lost_percent = (lost_total / total_expected * 100) if total_expected else 0.0
    issues = _vsig_consistency_issues(
        blocks,
        recomputed_lost,
        lost_total,
        block_count,
        len(rows),
        lost_percent,
        events_csv or Path("events.csv"),
        summary or {},
    )
    return {
        "block_count": block_count,
        "record_count": len(rows),
        "lost_blocks": lost_total,
        "lost_percent": lost_percent,
        "consistency_issues": issues,
        "total_records": max((r.total_records or 0 for r in rows), default=0),
        "decode_error_rows": [r.row_index for r in rows if r.decode_error],
        "mismatch_rows": [r.row_index for r in rows if r.mismatches],
        "malformed_rows": [r.row_index for r in rows if r.malformed],
        "signals": signals_by_did,
        "unknown_dids": [k for k, v in signals_by_did.items() if v["name"] is None],
        "meta_block_version": meta.get("vehicle_signal_block_version"),
    }


def _block_version_problem(meta: dict[str, Any], key: str, pinned: int) -> str | None:
    """Reason text when meta.json declares a block version the pinned defs do not decode."""
    if key not in meta or meta[key] is None:
        return None  # a session recorded before the block existed
    value = meta[key]
    if isinstance(value, bool) or not isinstance(value, int):
        return f"meta {key} {value!r} is not an integer"
    if value != pinned:
        return (
            f"meta {key} {value} is not supported: the pinned defs "
            f"{defs_version()} decode version {pinned} only"
        )
    return None


def build_report(
    session_dir: Path,
    session_id: str,
    meta: dict[str, Any],
    summary: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema = schema or ble_schema.load_default_schema()
    gap_threshold_ms = schema["notifyPeriodMs"] * GAP_MULTIPLIER

    layout, rows = decode_telemetry_csv(session_dir / "telemetry.csv", schema)

    packet_versions: dict[int, int] = {}
    decode_error_rows: list[int] = []
    mismatch_rows: list[int] = []
    schema_versions_seen: set[int] = set()
    for row in rows:
        packet_versions[row.app_packet_version] = packet_versions.get(row.app_packet_version, 0) + 1
        if row.decoded is not None:
            schema_versions_seen.add(row.decoded["version"])
        if row.decode_error:
            decode_error_rows.append(row.row_index)
        if row.mismatches:
            mismatch_rows.append(row.row_index)

    packet_count = len(rows)
    lost_total = 0
    ordering_violations: list[int] = []
    device_time_resets: list[dict[str, Any]] = []
    rx_gaps: list[dict[str, Any]] = []
    device_time_gaps: list[dict[str, Any]] = []

    prev = None
    for row in rows:
        if prev is not None:
            lost_total += (row.seq - prev.seq - 1) % 256
            if row.rx_mono_ms < prev.rx_mono_ms:
                ordering_violations.append(row.row_index)
            elif row.rx_mono_ms - prev.rx_mono_ms > gap_threshold_ms:
                rx_gaps.append({"row": row.row_index, "gap_ms": row.rx_mono_ms - prev.rx_mono_ms})

            curr_dt = row.decoded["device_time_ms"] if row.decoded else None
            prev_dt = prev.decoded["device_time_ms"] if prev.decoded else None
            if curr_dt is not None and prev_dt is not None:
                if curr_dt < prev_dt:
                    wrap_delta = (curr_dt - prev_dt) % _UINT32_MAX
                    is_plausible_wrap = (
                        prev_dt > _UINT32_MAX - _WRAP_NEAR_MAX_MARGIN
                        and wrap_delta < gap_threshold_ms * 10
                    )
                    if not is_plausible_wrap:
                        device_time_resets.append(
                            {"row": row.row_index, "prev": prev_dt, "curr": curr_dt}
                        )
                elif curr_dt - prev_dt > gap_threshold_ms:
                    device_time_gaps.append({"row": row.row_index, "gap_ms": curr_dt - prev_dt})
        prev = row

    loss_percent = (
        lost_total / (lost_total + packet_count) * 100 if (lost_total + packet_count) else 0.0
    )

    range_summary: dict[str, dict[str, Any]] = {}
    valid_summary: dict[str, dict[str, Any]] = {}
    for defs_signal in signals.known_defs_signals(schema):
        lo, hi = signals.signal_range(defs_signal)
        flag_name = signals.flag_name_for(defs_signal)
        key = signals.signal_key(defs_signal)
        total_count = 0
        valid_count = 0
        out_of_range = 0
        for row in rows:
            if row.decoded is None:
                continue
            value = row.decoded["physical"].get(defs_signal)
            if value is None:
                continue
            total_count += 1
            is_valid = bool(row.decoded["flags"].get(flag_name)) if flag_name else True
            if is_valid:
                valid_count += 1
                if not (lo <= value <= hi):
                    out_of_range += 1
        range_summary[key] = {
            "checked_count": valid_count,
            "out_of_range_count": out_of_range,
            "out_of_range_percent": (out_of_range / valid_count * 100) if valid_count else 0.0,
        }
        valid_summary[key] = {
            "total_count": total_count,
            "valid_count": valid_count,
            "valid_percent": (valid_count / total_count * 100) if total_count else 0.0,
        }

    can_rows = [row for row in rows if row.decoded and row.decoded["can_health"]]
    can_health_summary = None
    if can_rows:
        bus_states_seen = sorted(
            {
                row.decoded["can_health"]["bus_state"]
                for row in can_rows
                if row.decoded["can_health"]["bus_state"]
            }
        )
        ecu_present = [row.decoded["flags"].get("ecuPresent") for row in can_rows]
        can_health_summary = {
            "max_tec": max(row.decoded["can_health"]["tec"] for row in can_rows),
            "max_rec": max(row.decoded["can_health"]["rec"] for row in can_rows),
            "max_bus_off_count": max(
                row.decoded["can_health"]["bus_off_count"] for row in can_rows
            ),
            "max_unanswered_did_count": max(
                row.decoded["can_health"]["unanswered_did_count"] for row in can_rows
            ),
            "latched_foreign_tester": any(
                row.decoded["can_health"]["flags"].get("latchedForeignTester") for row in can_rows
            ),
            "latched_bus_off": any(
                row.decoded["can_health"]["flags"].get("latchedBusOff") for row in can_rows
            ),
            "ecu_present_rows": sum(flag is True for flag in ecu_present),
            "ecu_absent_rows": sum(flag is False for flag in ecu_present),
            "ecu_present_last": ecu_present[-1],
            "bus_states_seen": bus_states_seen,
        }

    tester_stats_summary = _tester_stats_summary(rows)

    imu_summary = None
    imu_csv = session_dir / "imu.csv"
    if imu_csv.exists():
        imu_rows = decode_imu_csv(imu_csv, schema)
        missing_total = 0
        prev_sample_index = None
        for imu_row in imu_rows:
            if prev_sample_index is not None:
                missing_total += (imu_row.sample_index - prev_sample_index - 1) % 65536
            prev_sample_index = imu_row.sample_index
        scale_mismatch_rows = [r.row_index for r in imu_rows if r.mismatches]
        duration_s = None
        if len(imu_rows) >= 2:
            duration_s = (
                (imu_rows[-1].device_time_ms - imu_rows[0].device_time_ms) % _UINT32_MAX
            ) / 1000.0
        effective_rate_hz = (len(imu_rows) / duration_s) if duration_s else None
        total_expected = missing_total + len(imu_rows)

        # block_flags is repeated across every sample row of the same block
        # (one BLE notification per block); count each block once, by its
        # block_seq transition, per imuBlock.flags in the schema.
        imu_flag_bits = schema["imuBlock"]["flags"]["bits"]
        block_count = 0
        device_overflow_blocks = 0
        read_error_blocks = 0
        sensor_reconfigured_blocks = 0
        prev_block_seq = None
        for imu_row in imu_rows:
            if imu_row.block_seq == prev_block_seq:
                continue
            prev_block_seq = imu_row.block_seq
            block_count += 1
            block_flags = ble_schema.decode_flags(imu_row.block_flags, imu_flag_bits)
            device_overflow_blocks += bool(block_flags.get("deviceOverflow"))
            read_error_blocks += bool(block_flags.get("readError"))
            sensor_reconfigured_blocks += bool(block_flags.get("sensorReconfigured"))

        imu_summary = {
            "sample_count": len(imu_rows),
            "missing_samples": missing_total,
            "loss_percent": (missing_total / total_expected * 100) if total_expected else 0.0,
            "effective_rate_hz": effective_rate_hz,
            "scale_mismatch_rows": scale_mismatch_rows,
            "block_count": block_count,
            "blocks_with_device_overflow": device_overflow_blocks,
            "blocks_with_read_error": read_error_blocks,
            "blocks_with_sensor_reconfigured": sensor_reconfigured_blocks,
        }

    gps_csv = session_dir / "gps.csv"
    gps_summary = _gps_summary(gps_csv, schema) if gps_csv.exists() else None

    signals_csv = session_dir / "signals.csv"
    vsig_summary = (
        _vehicle_signals_summary(signals_csv, schema, meta, session_dir / "events.csv", summary)
        if signals_csv.exists()
        else None
    )

    status = "ok"
    reasons: list[str] = []

    def warn(reason: str) -> None:
        nonlocal status
        if status == "ok":
            status = "warn"
        reasons.append(reason)

    if decode_error_rows:
        status = "fail"
        reasons.append(f"{len(decode_error_rows)} telemetry row(s) failed to decode from raw_hex")
    if gps_summary and gps_summary["decode_error_rows"]:
        status = "fail"
        reasons.append(
            f"{len(gps_summary['decode_error_rows'])} GPS row(s) failed to decode from raw_hex"
        )
    if vsig_summary and vsig_summary["decode_error_rows"]:
        status = "fail"
        reasons.append(
            f"{len(vsig_summary['decode_error_rows'])} vehicle-signal row(s) failed validation"
        )
    if any(item["out_of_range_count"] for item in range_summary.values()):
        warn("one or more signals had out-of-range values")
    if loss_percent > 1.0:
        warn(f"packet loss {loss_percent:.2f}% > 1%")
    if mismatch_rows:
        warn(f"{len(mismatch_rows)} row(s) disagree with the app's own decoded columns")
    if ordering_violations:
        warn(f"{len(ordering_violations)} rx_mono_ms ordering violation(s)")
    if device_time_resets:
        warn(f"{len(device_time_resets)} device_time_ms reset/reboot event(s)")
    if imu_summary and imu_summary["missing_samples"]:
        warn(f"{imu_summary['missing_samples']} missing IMU sample(s)")
    if imu_summary and imu_summary["blocks_with_sensor_reconfigured"]:
        warn(
            f"{imu_summary['blocks_with_sensor_reconfigured']} IMU block(s) after a sensor "
            "reconfiguration (samples before may be stale or wrong-scale)"
        )
    if gps_summary:
        if gps_summary["mismatch_rows"]:
            warn(
                f"{len(gps_summary['mismatch_rows'])} GPS row(s) disagree with the app's own "
                "decoded columns"
            )
        if gps_summary["lost_or_skipped_blocks"]:
            warn(f"{gps_summary['lost_or_skipped_blocks']} GPS block(s) lost or MTU-skipped")
        if gps_summary["blocks_with_parse_error"] or gps_summary["blocks_with_uart_overflow"]:
            warn(
                f"GPS node errors: parse error in {gps_summary['blocks_with_parse_error']} "
                f"block(s), UART overflow in {gps_summary['blocks_with_uart_overflow']} block(s)"
            )
        if gps_summary["block_count"] and not gps_summary["speed_check_usable_blocks"]:
            warn("no GPS block is usable for the speed check (no 3-D fix with gnssFixOk)")
    if vsig_summary:
        if vsig_summary["mismatch_rows"]:
            warn(
                f"{len(vsig_summary['mismatch_rows'])} vehicle-signal row(s) disagree with the "
                "app's own decoded columns"
            )
        if vsig_summary["malformed_rows"]:
            warn(
                f"{len(vsig_summary['malformed_rows'])} vehicle-signal record(s) with a raw "
                "value that does not fit the DID's length"
            )
        if vsig_summary["lost_blocks"]:
            warn(f"{vsig_summary['lost_blocks']} vehicle-signal block(s) lost")
        if vsig_summary["consistency_issues"]:
            warn(
                f"{len(vsig_summary['consistency_issues'])} vehicle-signal loss figure(s) "
                f"disagree: {'; '.join(vsig_summary['consistency_issues'][:3])}"
            )
    elif summary.get("vehicle_signal_block_count"):
        warn("summary.json counts vehicle-signal blocks but the session has no signals.csv")
    for key, pinned in (
        ("gps_block_version", defs_ble.GPS_BLOCK_VERSION),
        ("vehicle_signal_block_version", defs_ble.VSIG_BLOCK_VERSION),
    ):
        problem = _block_version_problem(meta, key, pinned)
        if problem:
            status = "fail"
            reasons.append(problem)
    if not reasons:
        reasons.append("no issues found")

    return {
        "session_id": session_id,
        "defs_version": defs_version(),
        "app_version": meta.get("app_version"),
        "ble_schema_version_meta": meta.get("ble_schema_version"),
        "telemetry_layout": layout,
        "schema_versions_seen": sorted(schema_versions_seen),
        "packet_versions": {str(k): v for k, v in sorted(packet_versions.items())},
        "packet_count": packet_count,
        "status": status,
        "reasons": reasons,
        "loss": {"lost_count": lost_total, "loss_percent": loss_percent},
        "consistency": {"decode_error_rows": decode_error_rows, "mismatch_rows": mismatch_rows},
        "ordering": {
            "rx_mono_ms_violations": ordering_violations,
            "device_time_ms_resets": device_time_resets,
        },
        "gaps": {
            "rx_mono_ms_gaps": rx_gaps,
            "device_time_ms_gaps": device_time_gaps,
            "threshold_ms": gap_threshold_ms,
        },
        "range_checks": range_summary,
        "valid_ratios": valid_summary,
        "can_health": can_health_summary,
        "tester_stats": tester_stats_summary,
        "imu": imu_summary,
        "gps": gps_summary,
        "vehicle_signals": vsig_summary,
        "summary_json": summary,
    }


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f} ms" if isinstance(value, float) else f"{value} ms"


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"Session {report['session_id']} -- status: {report['status'].upper()}",
        f"  defs version: {report['defs_version']}  app version: {report['app_version']}",
        f"  telemetry layout: {report['telemetry_layout']}  "
        f"packet versions seen: {report['packet_versions']}",
        f"  packets: {report['packet_count']}  lost: {report['loss']['lost_count']} "
        f"({report['loss']['loss_percent']:.2f}%)",
        "  reasons:",
    ]
    lines += [f"    - {reason}" for reason in report["reasons"]]

    if report["consistency"]["mismatch_rows"]:
        lines.append(
            f"  app/server decode mismatches: {len(report['consistency']['mismatch_rows'])} row(s)"
        )
    if report["ordering"]["rx_mono_ms_violations"]:
        lines.append(
            f"  rx_mono_ms ordering violations: {len(report['ordering']['rx_mono_ms_violations'])}"
        )
    if report["ordering"]["device_time_ms_resets"]:
        lines.append(f"  device_time_ms resets: {len(report['ordering']['device_time_ms_resets'])}")
    if report["gaps"]["rx_mono_ms_gaps"]:
        lines.append(
            f"  rx_mono_ms gaps (> {report['gaps']['threshold_ms']} ms): "
            f"{len(report['gaps']['rx_mono_ms_gaps'])}"
        )

    lines.append("  signal ranges:")
    for key, item in report["range_checks"].items():
        valid = report["valid_ratios"][key]
        lines.append(
            f"    {key}: valid {valid['valid_percent']:.1f}% of {valid['total_count']}, "
            f"out-of-range {item['out_of_range_count']} ({item['out_of_range_percent']:.2f}%)"
        )

    if report["can_health"]:
        can = report["can_health"]
        if "ecu_present_rows" in can:
            present = can["ecu_present_rows"]
            absent = can["ecu_absent_rows"]
            known = present + absent
            ecu = f"ecu present={present}/{known} absent={absent} last={can['ecu_present_last']}"
        else:
            ecu = "ecu present=n/a (old report)"
        lines.append(
            f"  CAN health: max TEC={can['max_tec']} max REC={can['max_rec']} "
            f"bus-off={can['max_bus_off_count']} unanswered DIDs={can['max_unanswered_did_count']} "
            f"latched(foreign_tester={can['latched_foreign_tester']}, "
            f"bus_off={can['latched_bus_off']}) states={can['bus_states_seen']} {ecu}"
        )

    tester = report.get("tester_stats")
    if tester:
        lines.append(
            f"  tester stats: step gap max {tester['step_gap_max_ms']} ms, "
            f"over client_step_max_ms {tester['step_gap_over_count']}"
        )
        for did_hex, item in tester["rtt"].items():
            label = f"{did_hex} ({item['name']})" if item["name"] else did_hex
            lines.append(
                f"    {label}: rtt min {_fmt_ms(item['min_ms'])} max {_fmt_ms(item['max_ms'])} "
                f"avg {_fmt_ms(item['avg_ms'])}, {item['count']} sample(s), "
                f"NRC 0x78 {item['nrc78_count']}"
            )

    if report["imu"]:
        imu = report["imu"]
        rate = f"{imu['effective_rate_hz']:.1f} Hz" if imu["effective_rate_hz"] else "n/a"
        lines.append(
            f"  IMU: {imu['sample_count']} samples in {imu['block_count']} block(s), "
            f"{imu['missing_samples']} missing ({imu['loss_percent']:.2f}%), "
            f"effective rate {rate}, scale mismatches {len(imu['scale_mismatch_rows'])}"
        )
        if imu["blocks_with_sensor_reconfigured"]:
            lines.append(
                f"    sensor reconfigured: {imu['blocks_with_sensor_reconfigured']} block(s)"
            )
        if imu["blocks_with_device_overflow"] or imu["blocks_with_read_error"]:
            lines.append(
                f"    device overflow: {imu['blocks_with_device_overflow']} block(s), "
                f"read error: {imu['blocks_with_read_error']} block(s)"
            )

    gps = report.get("gps")
    if gps:
        rate = f"{gps['effective_rate_hz']:.1f} Hz" if gps["effective_rate_hz"] else "n/a"
        lines.append(
            f"  GPS: {gps['block_count']} block(s), {gps['lost_or_skipped_blocks']} lost or "
            f"MTU-skipped ({gps['lost_or_skipped_percent']:.2f}%), effective rate {rate}, "
            f"usable for the speed check {gps['speed_check_usable_percent']:.1f}%"
        )
        lines.append(f"    fix types: {gps['fix_types']}")
        if gps["blocks_with_parse_error"] or gps["blocks_with_uart_overflow"]:
            lines.append(
                f"    parse error: {gps['blocks_with_parse_error']} block(s), "
                f"UART overflow: {gps['blocks_with_uart_overflow']} block(s)"
            )

    vsig = report.get("vehicle_signals")
    if vsig:
        lines.append(
            f"  vehicle signals: {vsig['record_count']} record(s) in {vsig['block_count']} "
            f"block(s), {vsig['lost_blocks']} lost ({vsig['lost_percent']:.2f}%), "
            f"{len(vsig['signals'])} of {vsig['total_records']} table DID(s) seen"
        )
        for did_hex, item in vsig["signals"].items():
            label = f"{did_hex} ({item['name'] or 'unknown DID'})"
            detail = f"states {item['states']}"
            if item["valid_count"]:
                detail += f", valid {item['min']} .. {item['max']} {item['unit']}"
            if item["nrcs"]:
                detail += f", NRC {item['nrcs']}"
            lines.append(f"    {label}: {detail}")

    derived = report.get("derived")
    if derived and "error" in derived:
        lines.append(f"  derived signals: error: {derived['error']}")
    elif derived:
        summary = ", ".join(f"{n} {status}" for status, n in sorted(derived["summary"].items()))
        lines.append(
            f"  derived signals: {derived['rows']} row(s), {derived['valid_rows']} valid "
            f"({summary})"
        )

    ride = report.get("derived_summary")
    if ride and "error" in ride:
        lines.append(f"  ride summaries: error: {ride['error']}")
    elif ride:
        counts = ", ".join(f"{n} {status}" for status, n in sorted(ride["summary"].items()))
        lines.append(f"  ride summaries: one {ride['ride_definition']} = one ride ({counts})")

    return "\n".join(lines) + "\n"
