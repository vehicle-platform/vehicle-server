"""Per-input time series of a stored session, the raw material of the derived signals.

Every input name of vehicle_defs.vehicle_derived.SIGNALS[...]["inputs"] is one of
    did.<NAME>      a vehicle-catalog DID (signals.parquet; the telemetry columns as fallback)
    sensor.<name>   a derived-signal sensor (imu.parquet / gps.parquet / telemetry.parquet)
    const.<name>    a vehicle constant (vehicle_defs.vehicle_constants), not a series
    derived.<name>  another derived signal (produced by the engine, not read here)
The time axis is the node's device_time_ms (integer ms). A series only holds samples
with a value: a record whose DID was not valid is absent, so its age grows until the
engine marks the consumer invalid (stale input), never a held value presented as fresh.

Sensors map through SENSORS[name]["phase0"] ("ble.imuBlock.ax", "ble.gpsBlock.groundSpeed",
"ble.telemetry.rttMaxMs"): the schema field name is converted to the snake_case column of
the matching parquet file (decode.csv_column_for_field) and the scaled column (suffix
_g, _dps, _mps, _deg) is preferred over the raw one. GPS: only speed, speed accuracy,
heading of motion and fix type are ever read, never a position (D-060, invariant 7).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from vehicle_server import signals
from vehicle_server.decode import GPS_RAW_COLUMNS, GPS_SCALED_COLUMNS, csv_column_for_field
from vehicle_server.defs import vehicle_catalog, vehicle_cl250, vehicle_constants, vehicle_derived

# Standard gravity (ISO 80000-3): the IMU parquet stores acceleration in g, the derived
# sensors are defined in m/s2.
STANDARD_GRAVITY_MPS2 = 9.80665

_UINT32 = 2**32

# Scaled-column suffix + SENSORS unit -> factor into the SENSORS unit.
_UNIT_FACTORS = {
    ("g", "m/s2"): STANDARD_GRAVITY_MPS2,
    ("dps", "deg/s"): 1.0,
    ("mps", "m/s"): 1.0,
    ("deg", "deg"): 1.0,
}

# The IMU parquet columns that hold scaled values (parquet.build_imu_table).
_IMU_COLUMNS = ("ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps")

# GPS fields that may be read (never a position). Everything else is refused.
GPS_ALLOWED_FIELDS = frozenset({"groundSpeed", "speedAccuracy", "headingOfMotion", "fixType"})

# The telemetry.parquet columns that a sensor can map to: the v4 tester columns and the
# v3 CAN health counters. unanswered_did_count is the BLE field (timed-out requests since
# boot); D-103 defines ride_unanswered_did_count on that field (last minus first in the ride).
_TELEMETRY_SENSOR_COLUMNS = (
    "unanswered_did_count",
    "step_gap_max_ms",
    "rtt_max_ms",
    "rtt_nrc78_count",
    "can_tx_error_count",
    "can_rx_error_count",
    "can_bus_off_count",
)


@dataclass(frozen=True)
class Series:
    """Samples (device_time_ms ascending, strictly increasing) with a float value each."""

    times: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.times)


@dataclass(frozen=True)
class SensorSource:
    """Where a sensor lives: parquet file, column and the factor into the SENSORS unit."""

    file: str  # "imu" | "gps" | "telemetry"
    column: str
    factor: float


def parse_phase0(phase0: str) -> tuple[str, str]:
    """ "ble.imuBlock.ax" -> ("imuBlock", "ax"); the block of a telemetry field is "telemetry"."""
    parts = phase0.split(".")
    if len(parts) != 3 or parts[0] != "ble":
        raise ValueError(f"unsupported phase0 field: {phase0!r}")
    return parts[1], parts[2]


def resolve_sensor(sensor: str) -> SensorSource | None:
    """The parquet source of a sensor name, or None when this server does not store it."""
    entry = vehicle_derived.SENSORS[sensor]
    phase0 = entry.get("phase0")
    if not phase0:
        return None
    block, field = parse_phase0(phase0)
    base = csv_column_for_field(field)
    unit = entry["unit"]

    if block == "imuBlock":
        candidates, file = _IMU_COLUMNS, "imu"
    elif block == "gpsBlock":
        if field not in GPS_ALLOWED_FIELDS:
            return None
        candidates, file = (*GPS_SCALED_COLUMNS, *GPS_RAW_COLUMNS), "gps"
    elif block == "telemetry":
        candidates, file = _TELEMETRY_SENSOR_COLUMNS, "telemetry"
    else:
        return None

    for column in candidates:
        if column == base:
            return SensorSource(file, column, 1.0)
        if column.startswith(base + "_"):
            factor = _UNIT_FACTORS.get((column[len(base) + 1 :], unit))
            if factor is not None:
                return SensorSource(file, column, factor)
    return None


def _unwrap(times: list[int]) -> list[int]:
    """Removes the uint32 device_time_ms wraparound (~49.7 days) in file order."""
    out: list[int] = []
    offset = 0
    prev: int | None = None
    for t in times:
        if prev is not None and t < prev - _UINT32 // 2:
            offset += _UINT32
        out.append(t + offset)
        prev = t
    return out


def _to_series(times: list[int | None], values: list[float | None]) -> Series:
    """Drops samples without a time or value, sorts by time, keeps the last of equal times."""
    pairs: dict[int, float] = {}
    clean = [(t, v) for t, v in zip(times, values, strict=True) if t is not None and v is not None]
    for t, v in sorted(clean, key=lambda p: p[0]):
        pairs[t] = float(v)
    return Series(list(pairs), list(pairs.values()))


class SessionInputs:
    """Lazy reader of the parquet files of one session (<session_dir>/parquet/)."""

    def __init__(self, session_dir: Path):
        self._parquet_dir = session_dir / "parquet"
        self._tables: dict[str, object] = {}
        self._cache: dict[str, Series | None] = {}

    def _table(self, name: str):
        if name not in self._tables:
            path = self._parquet_dir / f"{name}.parquet"
            self._tables[name] = pq.read_table(path) if path.is_file() else None
        return self._tables[name]

    def series(self, input_name: str) -> Series | None:
        """The series of a did.* / sensor.* input; None when absent or empty here."""
        if input_name not in self._cache:
            kind, _, name = input_name.partition(".")
            if kind == "did":
                series = self._did_series(name)
            elif kind == "sensor":
                series = self._sensor_series(name)
            else:
                raise ValueError(f"not a series input: {input_name!r}")
            self._cache[input_name] = series if series is not None and len(series) else None
        return self._cache[input_name]

    def _did_series(self, name: str) -> Series | None:
        if name not in vehicle_catalog.SIGNALS:
            return None
        table = self._table("signals")
        if table is not None:
            rows = table.filter(
                pc.and_(
                    pc.equal(table["signal"], name),
                    pc.and_(
                        pc.is_valid(table["value"]), pc.is_valid(table["sample_device_time_ms"])
                    ),
                )
            )
            if rows.num_rows:
                times = rows["sample_device_time_ms"].to_pylist()
                return _to_series(_unwrap(times), rows["value"].to_pylist())
        return self._telemetry_did_series(name)

    def _telemetry_did_series(self, name: str) -> Series | None:
        """The five DIDs of the telemetry packet (valid flag set), sampled at
        device_time_ms - age_ms; the fallback for sessions without signals.csv."""
        table = self._table("telemetry")
        if table is None or name not in vehicle_cl250.DIDS:
            return None
        value, age, valid = (
            signals.physical_column(name),
            signals.age_column(name),
            signals.valid_column(name),
        )
        if not {value, age, valid} <= set(table.column_names):
            return None
        device = _unwrap([t or 0 for t in table["device_time_ms"].to_pylist()])
        times: list[int | None] = []
        values: list[float | None] = []
        for t, v, a, ok in zip(
            device,
            table[value].to_pylist(),
            table[age].to_pylist(),
            table[valid].to_pylist(),
            strict=True,
        ):
            usable = ok is True and v is not None and a is not None
            times.append(t - a if usable else None)
            values.append(v if usable else None)
        return _to_series(times, values)

    def _sensor_series(self, name: str) -> Series | None:
        source = resolve_sensor(name)
        if source is None:
            return None
        table = self._table(source.file)
        if table is None or source.column not in table.column_names:
            return None
        if source.file == "gps":
            # Speed and heading are only meaningful with a fix (NAV-PVT gnssFixOk).
            if "fix_type" not in source.column:
                table = table.filter(pc.equal(table["gnss_fix_ok"], True))
        table = table.filter(pc.is_valid(table["device_time_ms"]))
        times = _unwrap(table["device_time_ms"].to_pylist())
        values = [
            None if v is None else v * source.factor for v in table[source.column].to_pylist()
        ]
        return _to_series(times, values)


def constant(name: str) -> float:
    """Value of a vehicle constant by its const.<name> input name (without the prefix)."""
    return float(vehicle_constants.CONSTANTS[name][0])
