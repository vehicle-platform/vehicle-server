"""Golden ride-summary records for the rt-core host tests (D-098 item 5, C3 part 2b).

One more generated header next to the class-A and class-B vectors (committed into
vehicle-rt-core under tests/host/vectors/, banner and content-sha256 as the others):

    derived_ride_records.h

It holds three tables, all encoded by vehicle_server.ride_record (the Python side of the
generated D-098 layout, `vehicle_defs.uds.RIDE_RECORDS`):

1. Field vectors: per row of `platform_uds_ride_fields[]` (same order, checked against the
   generated platform_uds.c) the raw value of range_min, of the largest valid value, of one
   resolution above and below the range, of a missing value, and the half-to-even ties
   2.5 -> 2 and 3.5 -> 4 where the float32 quotient is an exact tie. They test the generic
   writer alone (value, valid in; raw out).
2. No-ride records: every served record before any ride closed (counter 0, RIDE_CLOSED 0,
   every field NOT_AVAILABLE).
3. Ride records per class-B scenario and ride: the encoding of the expected closed values of
   derived_<scenario>_b.h (the class-B signals rt-core computes; every other field
   NOT_AVAILABLE), RIDE_CLOSED 1, counter = ride + 1 (the epoch counter the ride test drives).
   Fed the expected values, the writer must give these bytes exactly; end to end (rt-core's
   own float32 values, within resolution / 2) a field may differ by one raw step.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

from vehicle_server import ride_record
from vehicle_server.derived import golden
from vehicle_server.derived.golden_b import ClassBGeneration, RideRow, b_scenarios

RECORDS_HEADER = "derived_ride_records.h"
TIE_CASES = ((2, 2), (3, 4))  # quotient k + 0.5 -> raw (half to even)

_C_ROW_RE = re.compile(r"PLATFORM_UDS_([A-Z0-9_]+?)_BYTE, ")


def field_rows() -> list[tuple[str, str, dict]]:
    """(record, field, layout) in the order of platform_uds_ride_fields[]."""
    rows = [
        (rec, fname, f)
        for rec, layout in ride_record.RECORDS.items()
        for fname, f in layout["fields"].items()
    ]
    c_text = (golden.DEFS_C_RT_CORE / "platform_uds.c").read_text()
    start = c_text.index("platform_uds_ride_fields[")
    c_order = _C_ROW_RE.findall(c_text[start:])
    py_order = [f"{rec}_{fname}" for rec, fname, _ in rows]
    if c_order != py_order:
        raise RuntimeError("vehicle_defs.uds.RIDE_RECORDS and platform_uds_ride_fields[] differ")
    return rows


def field_cases(field: Mapping) -> list[tuple[str, bool, float, int]]:
    """(case, has_value, value, expected raw) cases of one field."""
    lo, res, top = field["range_min"], field["resolution"], field["raw_max"]
    f32 = ride_record.f32
    cases: list[tuple[str, bool, float, int]] = [
        ("min", True, f32(lo), 0),
        ("max", True, f32(lo + top * res), top),
        ("above", True, f32(lo + (top + 1) * res), field["na"]),
        ("below", True, f32(lo - res), field["na"]),
        ("missing", False, 0.0, field["na"]),
    ]
    for k, raw in TIE_CASES:
        value = f32(lo + (k + 0.5) * res)
        if top >= raw and ride_record.quotient(field, value) == k + 0.5:
            cases.append((f"tie {k}.5", True, value, raw))
    for _, has, value, raw in cases:
        got = ride_record.field_raw(field, value if has else None)
        if got != raw:
            raise RuntimeError(f"{field['signal']}: {value!r} encodes to {got}, expected {raw}")
    return cases


def _bytes(data: bytes) -> str:
    return ", ".join(f"0x{b:02X}u" for b in data)


def _record_row(prefix: str, rec: str, data: bytes) -> str:
    return f"    {{ {prefix}PLATFORM_UDS_DID_{rec}, {len(data)}u, {{ {_bytes(data)} }} }},"


def records_header(per_scenario: Mapping[str, list[list[RideRow]]]) -> str:
    rows = field_rows()
    max_len = max(r["length"] for r in ride_record.RECORDS.values())
    lines = [
        "/* Golden ride-summary records (D-098) for the rt-core host tests: the encoding of the",
        " * generated layout (platform_uds.h: PLATFORM_UDS_DID_RIDE_*, platform_uds_ride_fields[])",
        " * by vehicle-server's Python encoder. raw = round half to even of",
        " * (value - range_min) / resolution in float32 (lrintf, never roundf); all ones =",
        " * NOT_AVAILABLE (missing, invalid, or a raw value outside 0..raw_max: never clamped).",
        " * Header: byte 0 layout version, bytes 1-2 ride counter (big-endian), byte 3 bit 0",
        " * RIDE_CLOSED. bytes[] holds the data record only (no 0x62 / DID echo), zero-padded",
        f" * to {max_len} bytes; length is the record length.",
        " *",
        " * derived_ride_field_vectors: one generic-writer case per row; field is the row of",
        " *   platform_uds_ride_fields[]; has_value = false means no valid value.",
        " * derived_no_ride_records: every served record before any ride closed.",
        " * derived_<scenario>_ride_records: per ride of the class-B scenario, the encoding of",
        " *   the expected closed values of derived_<scenario>_b.h (other fields NOT_AVAILABLE),",
        " *   counter = ride + 1. Exact when the writer is fed those values; end to end a field",
        " *   may differ by one raw step (value tolerance resolution / 2). */",
        "",
        "#ifndef DERIVED_RIDE_RECORDS_H",
        "#define DERIVED_RIDE_RECORDS_H",
        "",
        "#include <stdbool.h>",
        "#include <stdint.h>",
        "",
        '#include "platform_uds.h"',
        "",
        f"#define DERIVED_RIDE_RECORD_LAYOUT_VERSION ({ride_record.LAYOUT_VERSION}u)",
        f"#define DERIVED_RIDE_RECORD_COUNT ({len(ride_record.RECORDS)}u)",
        f"#define DERIVED_RIDE_RECORD_MAX_LENGTH ({max_len}u)",
        f"#define DERIVED_RIDE_FIELD_COUNT ({len(rows)}u) /* = PLATFORM_UDS_RIDE_FIELD_COUNT */",
        "",
        "typedef struct {",
        "    uint8_t field;     /* row of platform_uds_ride_fields[] */",
        "    bool has_value;",
        "    float value;",
        "    uint32_t raw;      /* expected raw, or the field's NOT_AVAILABLE */",
        "} derived_ride_field_vector_t;",
        "",
        "typedef struct {",
        "    uint16_t did;",
        "    uint8_t length;",
        "    uint8_t bytes[DERIVED_RIDE_RECORD_MAX_LENGTH];",
        "} derived_ride_record_vector_t;",
        "",
        "typedef struct {",
        "    uint16_t ride;     /* row of derived_<scenario>_rides[]; counter = ride + 1 */",
        "    uint16_t did;",
        "    uint8_t length;",
        "    uint8_t bytes[DERIVED_RIDE_RECORD_MAX_LENGTH];",
        "} derived_ride_record_expected_t;",
        "",
    ]
    vectors: list[str] = []
    ties = set()
    for i, (rec, fname, f) in enumerate(rows):
        for case, has, value, raw in field_cases(f):
            if case.startswith("tie"):
                ties.add(raw)
            has_c = "true" if has else "false"
            vectors.append(
                f"    {{ {i}u, {has_c}, {golden.c_float(value)}, {raw}u }}, "
                f"/* {rec}.{fname} {case} */"
            )
    if ties != {2, 4}:
        raise RuntimeError(f"no exact half-to-even tie case for raw {sorted({2, 4} - ties)}")
    lines.append(f"#define DERIVED_RIDE_FIELD_VECTOR_COUNT ({len(vectors)}u)")
    lines += ["", "static const derived_ride_field_vector_t derived_ride_field_vectors[] = {"]
    lines += vectors
    lines += ["};", "", "static const derived_ride_record_vector_t derived_no_ride_records[] = {"]
    for rec in ride_record.RECORDS:
        lines.append(_record_row("", rec, ride_record.encode(rec, {}, counter=0, closed=False)))
    lines += ["};", ""]
    for name, rides in per_scenario.items():
        upper = name.upper()
        body: list[str] = []
        for ride, ride_rows in enumerate(rides):
            values = {r.signal: (r.value if r.valid else None) for r in ride_rows}
            for rec in ride_record.RECORDS:
                carried = set(ride_record.signals(rec))
                vals = {s: v for s, v in values.items() if s in carried}
                data = ride_record.encode(rec, vals, counter=ride + 1, closed=True)
                body.append(_record_row(f"{ride}u, ", rec, data))
        lines.append(f"#define DERIVED_{upper}_RIDE_RECORD_COUNT ({len(body)}u)")
        lines.append(
            f"static const derived_ride_record_expected_t derived_{name}_ride_records[] = {{"
        )
        lines += [*body, "};", ""]
    lines += ["#endif /* DERIVED_RIDE_RECORDS_H */", ""]
    return "\n".join(lines)


def generate_records(class_b: ClassBGeneration) -> golden.GeneratedFile:
    order = [sc.name for sc in b_scenarios()]
    per_scenario = {name: class_b.ride_values[name] for name in order}
    body = records_header(per_scenario)
    return golden.GeneratedFile(
        RECORDS_HEADER, golden.with_banner(body), hashlib.sha256(body.encode()).hexdigest()
    )
