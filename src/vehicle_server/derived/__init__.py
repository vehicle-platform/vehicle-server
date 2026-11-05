"""Derived signals of a stored session: the Python reference of D-070 item 2.

derive_session() evaluates vehicle_defs.vehicle_derived over a session's parquet files and
returns the long-format table plus the coverage block; store() writes
<session_dir>/derived.parquet (session_id, device_time_ms, signal, value, unit, valid,
flag) and returns the `derived` block of report.json. Only class A signals with a formula in
formulas.py are computed; every other signal is listed in the coverage with its status.

store_summary() does the same for the class-B per-ride summaries (summary.py): it writes
<session_dir>/parquet/derived_summary.parquet (long format, one row per scalar or per cell:
session_id, record, did, signal, cell, bin0_low, bin0_high, bin1_low, bin1_high, value, unit,
valid, flag) and returns the `derived_summary` block of report.json. In Phase 0 one stored
session is one ride.

store_all() runs both and evaluates the class-A signals once: the summaries reuse that
evaluation instead of computing it again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from vehicle_server.defs import defs_version, vehicle_derived
from vehicle_server.derived import engine, summary
from vehicle_server.derived.inputs import SessionInputs

DERIVED_FILE = "parquet/derived.parquet"
SUMMARY_FILE = "parquet/derived_summary.parquet"
RIDE_DEFINITION = "session"  # Phase 0: no ride segmentation, one session = one ride

TABLE_SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),
        ("device_time_ms", pa.int64()),
        ("signal", pa.string()),
        ("value", pa.float64()),
        ("unit", pa.string()),
        ("valid", pa.bool_()),
        ("flag", pa.string()),
    ]
)

SUMMARY_SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),
        ("record", pa.string()),
        ("did", pa.int64()),
        ("signal", pa.string()),
        ("cell", pa.int64()),
        ("bin0_low", pa.float64()),
        ("bin0_high", pa.float64()),
        ("bin1_low", pa.float64()),
        ("bin1_high", pa.float64()),
        ("value", pa.float64()),
        ("unit", pa.string()),
        ("valid", pa.bool_()),
        ("flag", pa.string()),
    ]
)


@dataclass
class DeriveResult:
    table: pa.Table
    coverage: dict[str, Any]  # the `derived` block of report.json


def derived_path(session_dir: Path) -> Path:
    return session_dir / DERIVED_FILE


def _table(session_id: str, outputs: list[engine.SignalOutput]) -> pa.Table:
    times: list[int] = []
    signals: list[str] = []
    values: list[float | None] = []
    units: list[str] = []
    flags: list[str | None] = []
    for out in outputs:
        n = len(out.times)
        times += out.times
        signals += [out.name] * n
        values += out.values
        units += [out.unit] * n
        flags += out.flags
    return pa.table(
        {
            "session_id": pa.array([session_id] * len(times), type=pa.string()),
            "device_time_ms": pa.array(times, type=pa.int64()),
            "signal": pa.array(signals, type=pa.string()),
            "value": pa.array(values, type=pa.float64()),
            "unit": pa.array(units, type=pa.string()),
            "valid": pa.array([v is not None for v in values], type=pa.bool_()),
            "flag": pa.array(flags, type=pa.string()),
        },
        schema=TABLE_SCHEMA,
    )


def derive_session(
    session_dir: Path, session_id: str, evaluation: engine.Evaluation | None = None
) -> DeriveResult:
    """Evaluates all derived signals of the session stored under `session_dir`, or uses
    `evaluation` when the caller already has it."""
    if evaluation is None:
        evaluation = engine.evaluate(SessionInputs(session_dir))
    statuses: dict[str, int] = {}
    for entry in evaluation.coverage.values():
        statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
    coverage = {
        "defs_version": defs_version(),
        "summary": statuses,
        "rows": sum(len(o.times) for o in evaluation.outputs),
        "valid_rows": sum(o.valid_count for o in evaluation.outputs),
        "signals": evaluation.coverage,
    }
    return DeriveResult(_table(session_id, evaluation.outputs), coverage)


def store(
    session_dir: Path, session_id: str, evaluation: engine.Evaluation | None = None
) -> tuple[dict[str, Any], bool]:
    """Derives and writes derived.parquet (removed when there is no row). Returns the
    `derived` report block and whether the file exists. Any exception becomes
    {"error": ...} (and no file): derived signals never fail an ingest."""
    path = derived_path(session_dir)
    try:
        result = derive_session(session_dir, session_id, evaluation)
        if result.table.num_rows:
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(result.table, path)
        else:
            path.unlink(missing_ok=True)
        return result.coverage, result.table.num_rows > 0
    except Exception as exc:
        path.unlink(missing_ok=True)
        return {"error": f"{type(exc).__name__}: {exc}"}, False


@dataclass
class SummaryResult:
    table: pa.Table
    coverage: dict[str, Any]  # the `derived_summary` block of report.json


def summary_path(session_dir: Path) -> Path:
    return session_dir / SUMMARY_FILE


def _summary_table(session_id: str, rows: list[summary.SummaryRow]) -> pa.Table:
    records = vehicle_derived.RIDE_SUMMARY_RECORDS
    specs = vehicle_derived.SIGNALS
    edges: list[list[float | None]] = [[], [], [], []]
    for row in rows:
        for axis in range(2):
            low, high = row.edges[axis] if axis < len(row.edges) else (None, None)
            edges[2 * axis].append(low)
            edges[2 * axis + 1].append(high)
    names = [specs[row.signal]["record"] for row in rows]
    return pa.table(
        {
            "session_id": pa.array([session_id] * len(rows), type=pa.string()),
            "record": pa.array(names, type=pa.string()),
            "did": pa.array([records[n][0] for n in names], type=pa.int64()),
            "signal": pa.array([row.signal for row in rows], type=pa.string()),
            "cell": pa.array([row.cell for row in rows], type=pa.int64()),
            "bin0_low": pa.array(edges[0], type=pa.float64()),
            "bin0_high": pa.array(edges[1], type=pa.float64()),
            "bin1_low": pa.array(edges[2], type=pa.float64()),
            "bin1_high": pa.array(edges[3], type=pa.float64()),
            "value": pa.array([row.value for row in rows], type=pa.float64()),
            "unit": pa.array([row.unit for row in rows], type=pa.string()),
            "valid": pa.array([row.value is not None for row in rows], type=pa.bool_()),
            "flag": pa.array([row.flag for row in rows], type=pa.string()),
        },
        schema=SUMMARY_SCHEMA,
    )


def _health_windows_block(
    evaluation: summary.SummaryEvaluation, session_dir: Path
) -> dict[str, Any]:
    """The D-079 window instances on the same t0 axis as GET /sessions/{id}/series. A failure
    here becomes {"error": ...} in this key only, so it never costs the class-B scalars."""
    try:
        return _health_windows_instances(evaluation, session_dir)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _health_windows_instances(
    evaluation: summary.SummaryEvaluation, session_dir: Path
) -> dict[str, Any]:
    from vehicle_server import series

    t0 = series.t0_of(series.load_all(session_dir))
    windows: dict[str, Any] = {}
    for key in vehicle_derived.WINDOWS:
        result = evaluation.windows.get(key)
        if result is None:
            continue
        windows[key] = {
            "qualifying_count": result.qualifying_count,
            "instances": [
                {
                    "start_ms": instance.open_ms - t0,
                    "end_ms": instance.close_ms - t0,
                    "duration_ms": instance.duration_ms,
                    "close_reason": instance.close_reason,
                    "qualifies": instance.qualifies,
                    "first_qualifying": instance == result.first_qualifying,
                }
                for instance in result.instances
            ],
        }
    return {
        "rule": "provisional: Claude's reading of D-079 (vehicle-server README)",
        "t0_device_ms": t0,
        "windows": windows,
    }


def summarize_session(
    session_dir: Path,
    session_id: str,
    inputs: SessionInputs | None = None,
    class_a: engine.Evaluation | None = None,
) -> SummaryResult:
    """Evaluates the class-B summaries of the session stored under `session_dir`. The
    class-A evaluation is computed first, in memory, unless the caller passes it."""
    if inputs is None:
        inputs = SessionInputs(session_dir)
    if class_a is None:
        class_a = engine.evaluate(inputs)
    evaluation = summary.evaluate(inputs, class_a)
    statuses: dict[str, int] = {}
    records: dict[str, dict[str, Any]] = {
        name: {"did": did, "signals": {}}
        for name, (did, _description) in vehicle_derived.RIDE_SUMMARY_RECORDS.items()
    }
    for name, entry in evaluation.coverage.items():
        statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
        records[vehicle_derived.SIGNALS[name]["record"]]["signals"][name] = entry
    coverage = {
        "defs_version": defs_version(),
        "ride_definition": RIDE_DEFINITION,
        "summary": statuses,
        "records": records,
        "health_windows": _health_windows_block(evaluation, session_dir),
    }
    return SummaryResult(_summary_table(session_id, evaluation.rows), coverage)


def store_summary(
    session_dir: Path,
    session_id: str,
    inputs: SessionInputs | None = None,
    class_a: engine.Evaluation | None = None,
) -> tuple[dict[str, Any], bool]:
    """Writes parquet/derived_summary.parquet (removed when there is no row). Returns the
    `derived_summary` report block and whether the file exists. Any exception becomes
    {"error": ...} (and no file): summaries never fail an ingest."""
    path = summary_path(session_dir)
    try:
        result = summarize_session(session_dir, session_id, inputs, class_a)
        if result.table.num_rows:
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(result.table, path)
        else:
            path.unlink(missing_ok=True)
        return result.coverage, result.table.num_rows > 0
    except Exception as exc:
        path.unlink(missing_ok=True)
        return {"error": f"{type(exc).__name__}: {exc}"}, False


def store_all(session_dir: Path, session_id: str) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    """store() and store_summary() over one class-A evaluation. Returns the `derived` block,
    whether derived.parquet exists and the `derived_summary` block. A failed shared
    evaluation becomes the same {"error": ...} in both blocks (and neither file), as when
    each one evaluated on its own."""
    try:
        inputs = SessionInputs(session_dir)
        class_a = engine.evaluate(inputs)
    except Exception as exc:
        derived_path(session_dir).unlink(missing_ok=True)
        summary_path(session_dir).unlink(missing_ok=True)
        error = {"error": f"{type(exc).__name__}: {exc}"}
        return error, False, dict(error)
    block, has_derived = store(session_dir, session_id, class_a)
    summary_block, _ = store_summary(session_dir, session_id, inputs, class_a)
    return block, has_derived, summary_block
