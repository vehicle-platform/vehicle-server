"""Read-only exposure of the stored class-B per-ride summaries (D-070, D-087).

The summaries are computed once by ``derived.summary`` and stored in
``parquet/derived_summary.parquet`` plus the ``derived_summary`` block of
``report.json``. This module only re-shapes those stored values for the read API:
it never evaluates a summary, changes a formula or reads a raw series. Names,
units, ranges and bin edges come from the pinned defs.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from vehicle_server.defs import vehicle_catalog, vehicle_derived
from vehicle_server.derived import SUMMARY_FILE
from vehicle_server.derived.summary import bin_edges, cell_axes, cell_count


class NoRideSummary(Exception):
    """A session has no usable stored ride summary."""


class BadRideQuery(Exception):
    """A query parameter of GET /rides is invalid."""


def _entry_signals(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = block.get("records")
    if not isinstance(records, dict):
        raise NoRideSummary("session has no ride summary")
    signals: dict[str, dict[str, Any]] = {}
    for record in records.values():
        if not isinstance(record, dict):
            continue
        entries = record.get("signals")
        if not isinstance(entries, dict):
            continue
        for name, entry in entries.items():
            if isinstance(name, str) and isinstance(entry, dict):
                signals[name] = entry
    return signals


def _read_report(session_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((session_dir / "report.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise NoRideSummary("session has no ride summary") from exc


def _summary_block(report: dict[str, Any]) -> dict[str, Any]:
    """The stored ``derived_summary`` block; NoRideSummary when it is missing or an error."""
    block = report.get("derived_summary")
    if not isinstance(block, dict):
        raise NoRideSummary("session has no ride summary")
    if "error" in block:
        raise NoRideSummary(str(block["error"]))
    return block


def read_summary_block(session_dir: Path) -> dict[str, Any] | None:
    """The stored ``derived_summary`` block, or None when the report/file is missing or
    the block is an error. Used by /rides, which must keep walking past such sessions."""
    try:
        block = _summary_block(_read_report(session_dir))
    except NoRideSummary:
        return None
    return block if isinstance(block.get("records"), dict) else None


def _read_summary_rows(session_dir: Path) -> list[dict[str, Any]]:
    path = session_dir / SUMMARY_FILE
    if not path.is_file():
        raise NoRideSummary("session has no ride summary")
    try:
        return pq.read_table(path).to_pylist()
    except Exception as exc:  # a damaged file must not become a 500
        raise NoRideSummary(f"ride summary data is unreadable: {exc}") from exc


def _input_unit(name: str) -> str:
    kind, _, key = name.partition(".")
    if kind == "sensor":
        return vehicle_derived.SENSORS[key]["unit"]
    if kind == "derived":
        return vehicle_derived.SIGNALS[key]["unit"]
    return vehicle_catalog.SIGNALS[key]["unit"]


def _binned_entry(
    name: str,
    entry: dict[str, Any],
    bins: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_cell: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("signal") != name:
            continue
        cell = row.get("cell")
        if isinstance(cell, int):
            by_cell[cell] = row

    def_edges = [[list(bin_edges(axis, index)) for index in range(axis["count"])] for axis in bins]
    stored_edges: list[dict[int, list[float]]] = [{} for _ in bins]
    for cell, row in by_cell.items():
        indices = cell_axes(bins, cell)
        for axis_index, bin_index in enumerate(indices):
            stored_edges[axis_index][bin_index] = [
                row[f"bin{axis_index}_low"],
                row[f"bin{axis_index}_high"],
            ]

    mismatch = len(by_cell) != cell_count(bins)
    axes: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(bins):
        edges: list[list[float]] = []
        for bin_index in range(axis["count"]):
            edge = stored_edges[axis_index].get(bin_index)
            if edge is None or edge != def_edges[axis_index][bin_index]:
                mismatch = True
            edges.append(edge or def_edges[axis_index][bin_index])
        axes.append(
            {
                "input": axis["input"],
                "unit": _input_unit(axis["input"]),
                "edges": edges,
            }
        )

    cells = [
        {
            "index": cell,
            "axes": cell_axes(bins, cell),
            "value": row.get("value"),
            "flag": row.get("flag"),
        }
        for cell, row in sorted(by_cell.items())
    ]
    response: dict[str, Any] = {
        **{k: entry[k] for k in ("name", "status", "unit", "range") if k in entry},
        "kind": "binned",
        "axes": axes,
        "cells": cells,
        "total_s": entry.get("total_s", 0.0),
        "outside_s": entry.get("outside_s", 0.0),
    }
    if mismatch:
        response["defs_mismatch"] = True
    return response


def _signal_entry(
    name: str,
    entry: dict[str, Any],
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    status = entry.get("status")
    if status != "computed":
        response = {"name": name, "status": status}
        for key in ("reason", "pending_inputs", "missing", "message"):
            if key in entry:
                response[key] = entry[key]
        return response

    base = {
        "name": name,
        "status": "computed",
        "unit": spec["unit"],
        "range": list(spec["range"]),
    }
    bins = list(spec.get("bins") or [])
    if bins:
        return {
            **base,
            **_binned_entry(name, entry, bins, rows),
        }
    return {
        **base,
        "kind": "scalar",
        "value": entry.get("value"),
        "flag": entry.get("flag"),
    }


def build_ride_summary(
    session_id: str,
    report: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the /summary response from one stored report and summary rows."""
    block = _summary_block(report)
    stored = _entry_signals(block)
    rows = rows or []
    signals: list[dict[str, Any]] = []
    for name, spec in vehicle_derived.SIGNALS.items():
        if name not in stored:
            continue
        signals.append(_signal_entry(name, stored[name], spec, rows))
    return {
        "session_id": session_id,
        "ride_definition": block.get("ride_definition"),
        "defs_version": report.get("defs_version"),
        "signals": signals,
    }


def load_ride_summary(session_dir: Path, session_id: str) -> dict[str, Any]:
    """Read the stored files of one session and build the /summary response."""
    report = _read_report(session_dir)
    stored = _entry_signals(_summary_block(report))
    needs_rows = any(
        stored.get(name, {}).get("status") == "computed" and spec.get("bins")
        for name, spec in vehicle_derived.SIGNALS.items()
        if name in stored
    )
    rows = _read_summary_rows(session_dir) if needs_rows else []
    return build_ride_summary(session_id, report, rows)


def scalar_names() -> list[str]:
    """Class-B scalar names in defs order (binned summaries are not trend columns)."""
    return [
        name
        for name, spec in vehicle_derived.SIGNALS.items()
        if spec["class"] == "B" and not spec.get("bins")
    ]


def select_scalar_names(names: str | None) -> list[str]:
    """Validate the /rides `names` query parameter; the names keep their query order."""
    available = scalar_names()
    if names is None:
        return available
    requested = list(dict.fromkeys(part.strip() for part in names.split(",") if part.strip()))
    if not requested:
        raise BadRideQuery("unknown class-B scalar names: (empty list)")
    binned = [
        name
        for name in requested
        if name in vehicle_derived.SIGNALS
        and vehicle_derived.SIGNALS[name]["class"] == "B"
        and vehicle_derived.SIGNALS[name].get("bins")
    ]
    unknown = [name for name in requested if name not in available and name not in binned]
    if unknown:
        raise BadRideQuery(f"unknown class-B scalar names: {', '.join(unknown)}")
    if binned:
        raise BadRideQuery(
            "binned summaries belong to /sessions/{id}/summary: " + ", ".join(binned)
        )
    return requested


def parse_utc_bound(value: str, *, end: bool) -> dt.datetime:
    """Parse an ISO 8601 date or timestamp as an inclusive UTC bound.

    Date-only values cover the whole UTC day: 00:00:00 for `from`, 23:59:59.999999 for
    `to`. Timestamps without an offset are UTC; offsets are converted to UTC.
    """
    text = value.strip()
    try:
        if len(text) == 10 and text[4] == "-" and text[7] == "-":
            parsed = dt.datetime.combine(
                dt.date.fromisoformat(text),
                dt.time.max if end else dt.time.min,
                tzinfo=dt.UTC,
            )
        else:
            parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BadRideQuery(f"invalid ISO 8601 date or timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def created_in_range(
    created_utc: str | None, start: dt.datetime | None, end: dt.datetime | None
) -> bool:
    if start is None and end is None:
        return True
    if created_utc is None:
        return False
    try:
        created = parse_utc_bound(created_utc, end=False)
    except BadRideQuery:
        return False
    return (start is None or created >= start) and (end is None or created <= end)


def build_rides(
    sessions: list[tuple[str, str | None, dict[str, Any] | None]],
    names: list[str],
    start: dt.datetime | None,
    end: dt.datetime | None,
) -> dict[str, Any]:
    """Build the /rides table from (session_id, created_utc, summary_block) tuples."""
    columns = [{"name": name, "unit": vehicle_derived.SIGNALS[name]["unit"]} for name in names]
    rows: list[dict[str, Any]] = []
    skipped = 0
    for session_id, created_utc, block in sessions:
        if not created_in_range(created_utc, start, end):
            continue
        if block is None:  # counted only inside the requested range
            skipped += 1
            continue
        stored = _entry_signals(block)
        values: dict[str, float | None] = {}
        flags: dict[str, str | None] = {}
        for name in names:
            entry = stored.get(name)
            if entry is None:
                values[name] = None
                flags[name] = "missing"
            elif entry.get("status") == "computed":
                values[name] = entry.get("value")
                flags[name] = entry.get("flag")
            else:
                values[name] = None
                flags[name] = entry.get("status")
        rows.append(
            {
                "session_id": session_id,
                "created_utc": created_utc,
                "values": values,
                "flags": flags,
            }
        )
    rows.sort(key=lambda row: (row["created_utc"] is None, row["created_utc"] or ""))
    return {"columns": columns, "rows": rows, "skipped": skipped}
