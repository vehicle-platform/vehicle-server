# vehicle-server

The server side of the vehicle-platform SDV stack (D-010): ingests ride sessions
recorded by the phone app (`vehicle-mobile`), validates and decodes them against
the BLE telemetry schema, and stores them as Parquet + a queryable index for
`vehicle-ml` and analysis. Part of the [vehicle-platform](https://github.com/vehicle-platform)
organization; see `vehicle-defs/docs/ARCHITECTURE.md` for the platform picture.

## What this repo is not

- No model training (`vehicle-ml` reads data from here).
- Never sends commands to the vehicle; it only stores and serves data.
- MDF4 is an archive/sharing export; Parquet remains the source for ML.

## Setup

```bash
git submodule update --init --recursive   # external/vehicle-defs, pinned to v0.24.0
uv sync                                    # installs runtime + dev dependencies
```

The `export` command (MDF4) needs the optional `mdf4` extra: `uv sync --extra mdf4`
(the dev group already includes it). It pulls in asammdf, which is LGPL-3.0+, so the
default install and the Docker image leave it out (D-091).

## Run

```bash
export VEHICLE_API_TOKEN=some-long-random-token   # required; the API refuses (503) without it
uv run vehicle-server serve                       # http://0.0.0.0:8000
```

Or with Docker Compose (builds the image, mounts a named volume for `/data`):

```bash
VEHICLE_API_TOKEN=some-long-random-token docker compose up --build
```

A deployment exposed outside a trusted network needs TLS in front of it (e.g.
a reverse proxy) -- the API itself only does bearer-token auth, not TLS.

## CLI

```bash
uv run vehicle-server import <folder-or-zip>      # ingest a session, print its report
uv run vehicle-server report <session_id>         # re-print a stored session's report
uv run vehicle-server report <session_id> --format json
uv run vehicle-server bringup <session_id> --gate V0 [--json]  # bring-up gate check (B0/V0/V1)
uv run vehicle-server list                        # one line per ingested session
uv run vehicle-server derive <session_id>         # recompute the derived signals and ride summaries of a stored session
uv run vehicle-server derive --all                # ... of every stored session
uv run vehicle-server export <session_id>         # write <session>/parquet/<session_id>.mf4
uv run vehicle-server export <session_id> --include-position  # include GPS position columns (off by default, D-076)
uv run vehicle-server serve [--host] [--port] [--reload]
```

`import` and `report` exit 0 for status `ok`/`warn`, 1 for `fail` or an
invalid session, 2 for a session-id conflict (same id, different content).
`bringup` exits 0 for `pass`, 1 for `fail` or an unknown session, 2 for `incomplete`.

## HTTP API

All endpoints except `GET /health` require `Authorization: Bearer $VEHICLE_API_TOKEN`.

| Method & path | Description |
|---|---|
| `GET /health` | Liveness check, no auth. |
| `POST /sessions` | Multipart field `archive` (a session `.zip`). `201` created, `200` identical re-upload (idempotent), `409` same id/different content, `422` invalid session, `413` over `VEHICLE_MAX_UPLOAD_MB`. |
| `GET /sessions` | List ingested sessions from the index. |
| `GET /sessions/{id}/report` | The session's `report.json`. Add `?format=text` for a human-readable summary. `404` for an unknown id. |
| `GET /sessions/{id}/series` | Downsampled time series for the analyzer oscilloscope/replay. Query: `signals` (comma-separated `did.<NAME>`, `sensor.<name>`, `derived.<signal>`; default all available), `max_points` (10..20000, default 2000; min/max buckets so spikes survive), `from_ms`/`to_ms` (inclusive window relative to `t0_device_ms`, applied before downsampling). Returns `{session_id, t0_device_ms, max_points, series: [{name, kind, unit, points_total, downsampled, t_ms, v}]}`; `t0_device_ms` is the earliest sample of any series of the session, so calls share one time axis. `404` unknown session, `422` unknown series name or `max_points` out of range. Never contains a GPS position. |
| `GET /sessions/{id}/track` | Relative dead-reckoned ride track from GNSS speed and heading (D-060, D-076). Query: `max_points` (10..20000, default 2000, RDP simplification), `min_heading_speed_mps` (0..10, default 1.0: below it the last trusted heading is held), `color` (optional one `did.<NAME>`, `sensor.<name>` or `derived.<signal>`; position-looking and unknown names are `404` like `/stats`), `from_ms`/`to_ms` (inclusive window on the `/series` t0 axis, applied before integration). Returns `{session_id, t0_device_ms, method: "dead_reckoning", frame: "relative_enu_from_first_fix", max_points, points_total, downsampled, path_length_m, segments_total, excluded: {no_fix, stale_fix, untrusted_heading}, color: {name, unit} | null, segments: [{t_ms, x_m, y_m, speed_mps, heading_deg, color_v?, gap_before_ms}]}`. The shape is dead-reckoned from speed and heading in metres relative to the first usable sample (east = +x, north = +y); error grows with distance, it is approximate and **not a map**. It never contains an absolute position. `404` unknown session or `session has no GPS`; `422` too many segments for `max_points` or an out-of-range query. `color_v` is the latest colour sample at or before each kept point, null after staleness; derived series use their defs `stale_after_ms`, else three `period_ms`; a catalog DID without a poll-plan `stale_after_ms` and a derived signal with neither are held. Simplification is exact RDP: the tolerance is the smallest one whose result fits `max_points`. |
| `GET /sessions/{id}/summary` | Stored class-B per-ride summary of one session (D-070, D-087). Returns `{session_id, ride_definition, defs_version, signals: [...]}` in defs order. A computed scalar has `{name, status: "computed", unit, range, kind: "scalar", value, flag}`; a computed binned signal has `{name, status: "computed", unit, range, kind: "binned", axes: [{input, unit, edges: [[lo, hi], ...]}], cells: [{index, axes: [bin, ...], value, flag}], total_s, outside_s}`. Stored cells and their edges come from `parquet/derived_summary.parquet`; axes and units from the pinned defs. If the stored edges disagree with the defs (a session decoded with another defs version), the stored edges are returned and the signal has `defs_mismatch: true`. Non-computed entries keep `status` plus `reason`/`pending_inputs`/`missing`/`message`. `404` unknown session or `session has no ride summary`; an error block is `404` with its stored text. Never contains a position. |
| `GET /rides` | Cross-session class-B scalar trend table. Query: `from`/`to` (optional ISO 8601 date or timestamp, inclusive, UTC; a date-only value covers the whole UTC day, a timestamp without an offset is UTC), `names` (optional comma-separated class-B scalars; default all, response order follows the query). Returns `{columns: [{name, unit}], rows: [{session_id, created_utc, values: {name: value|null}, flags: {name: flag|status|null}}], skipped}` sorted by `created_utc` ascending. Sessions inside the range without a stored summary block are skipped (`skipped`); a non-computed value is null with its status in `flags`. `422` unknown or non-class-B names; binned summaries are refused here and belong to `/summary`. Reads report blocks only, never every parquet file. Never contains a position. |
| `GET /sessions/{id}/bringup` | Bring-up gate check of a stored session (see *Bring-up gate check*). Query: `gate` (`B0`, `V0` or `V1`, case-insensitive). Returns the same JSON as `vehicle-server bringup --json`. `404` unknown session, `422` missing or unknown gate. |
| `GET /sessions/{id}/stats` | Exact statistics over full-resolution points. Query: `signals` (comma-separated `did.<NAME>`, `sensor.<name>`, `derived.<signal>`; default all available), `start_ms`/`end_ms` (inclusive window relative to `t0_device_ms`, the same axis as `/series`). Returns `{session_id, t0_device_ms, start_ms, end_ms, series: [{name, kind, unit, count, mean, std, min, max, p05, p50, p95, t_first_ms, t_last_ms}]}`. `std` is the population standard deviation and percentiles use linear interpolation, computed before any downsampling; an empty window has `count: 0` and null statistics/times. `404` unknown session or unknown/position-looking series name (listed). Never contains a GPS position. |
| `GET /ws/live` (WebSocket) | Live stream for vehicle-analyzer (D-081). First message `{"type": "hello", "token": "<VEHICLE_API_TOKEN>"}` within 5 s, optional frame filters `"channels": [1, 2]` and `"ids": [...]`; the token is never put in the URL. Then `{"type": "batch", "payload": StreamBatchDto}` messages and every 5 s `{"type": "heartbeat", "payload": {"source", "dropped", "timestamp_us"}}`. Close `1008` bad hello or token, `1011` no `VEHICLE_API_TOKEN`. One bounded queue per client: a slow client loses its oldest batches (`dropped`), never blocks the others. |

A session upload is a zip with `meta.json`, `telemetry.csv`, `events.csv`,
`summary.json` (and optional `imu.csv`, `gps.csv` and `signals.csv`) either at the zip root or inside one
top-level folder -- see the session contract the phone app writes to.

Example `/sessions/{id}/summary` response (one scalar, one binned signal; other signals elided):

```json
{"session_id": "20260115-120000-abcd", "ride_definition": "session", "defs_version": "v0.24.0",
 "signals": [
   {"name": "ride_distance", "status": "computed", "unit": "km", "range": [0.0, 2000.0],
    "kind": "scalar", "value": 12.3, "flag": null},
   {"name": "ride_coolant_time_at_temp", "status": "computed", "unit": "s",
    "range": [0.0, 86400.0], "kind": "binned",
    "axes": [{"input": "did.COOLANT_TEMP", "unit": "degC",
              "edges": [[30.0, 40.0], [40.0, 50.0]]}],
    "cells": [{"index": 0, "axes": [0], "value": 41.0, "flag": null}],
    "total_s": 900.0, "outside_s": 3.0}]}
```

Example `/rides?names=ride_distance,ride_time` response:

```json
{"columns": [{"name": "ride_distance", "unit": "km"}, {"name": "ride_time", "unit": "s"}],
 "rows": [{"session_id": "20260115-120000-abcd", "created_utc": "2026-01-15T12:00:00+00:00",
           "values": {"ride_distance": 12.3, "ride_time": 900.0},
           "flags": {"ride_distance": null, "ride_time": null}}],
 "skipped": 0}
```

## Storage layout

```
$VEHICLE_DATA_DIR/
  index.sqlite                     # one row per ingested session (has_imu, has_gps, has_vehicle_signals, has_derived)
  sessions/<session_id>/
    raw/                           # the uploaded files, verbatim
    parquet/telemetry.parquet      # decoded telemetry, one row per BLE packet (incl. CAN health, tester stats)
    parquet/imu.parquet            # decoded IMU samples (if imu.csv was present)
    parquet/gps.parquet            # decoded GPS blocks (if gps.csv was present)
    parquet/signals.parquet        # vehicle-signal records (if signals.csv was present)
    parquet/<session_id>.mf4       # archive/sharing export written on demand by `export`
    parquet/derived.parquet        # derived signals, long format (if any could be computed)
    parquet/derived_summary.parquet  # class-B ride summaries, long format (if any)
    report.json                    # validation report (see below), incl. the `derived` and `derived_summary` blocks
```

Parquet column names follow `<defs signal>_<unit>` (e.g. `engine_speed_rpm`,
`vehicle_speed_kmh`, `coolant_temp_degc`, `throttle_pos_pct`,
`battery_voltage_v`), with `<signal>_age_ms` and `<signal>_valid` alongside
each. Units come from `vehicle_defs.vehicle_cl250.DIDS`, never hand-written
(platform rule: signals only come from vehicle-defs).

Telemetry version 4 (D-058) adds eight tester-statistics columns at the end of
`telemetry.csv` (`step_gap_max_ms`, `step_gap_over_count`, `rtt_did`, `rtt_min_ms`,
`rtt_max_ms`, `rtt_sum_ms`, `rtt_count`, `rtt_nrc78_count`; 43 columns in total). They
are stored under the same names as nullable integer Parquet columns, decoded from
`raw_hex` (null for version 2/3 rows). A `telemetry.csv` recorded before them (21 or
35 columns) is still accepted and gets null columns.

The version 3 CAN health fields are stored too, also decoded from `raw_hex`, as nullable
integers under the schema field names in snake_case: `can_bus_state`, `can_tx_error_count`,
`can_rx_error_count`, `can_bus_off_count`, `unanswered_did_count`, `can_flags` (the csv calls
the error counters `can_tec`/`can_rec`). They are null for version 2 rows.

### Validation report

Every ingested session gets a `report.json` (and `render_text()` summary)
covering: which telemetry layout (v2/v3/v4) and packet versions were seen, the
defs version used to decode it, a raw_hex re-decode vs. the app's own decoded
columns (consistency check), packet loss recomputed from `seq`, rx/device
time ordering and gaps, per-signal range and validity checks against
`vehicle_defs.vehicle_cl250.DIDS`, CAN health (v3; including `ecu_present_rows` /
`ecu_absent_rows` / `ecu_present_last` from the `ecuPresent` flag), tester stats (v4), IMU
sample-loss/scale checks (recomputed from `sample_index` and `*_raw`, including
`sensorReconfigured` blocks), a GPS summary and a vehicle-signal summary (see below). Overall status is
`ok` / `warn` / `fail`.

The `tester_stats` section (absent when the session has no version 4 packet) holds the
last `step_gap_max_ms` / `step_gap_over_count` and, per `rtt_did` (keyed in hex, e.g.
`0xF40C`), its last record: min, max, avg (`sum / count`, null when the count is 0 or the
sum or count is saturated at `0xFFFFFFFF`), count and NRC 0x78 count. These are
measurements of the temporary vehicle-bus tester (D-058), not vehicle signals; they do
not change the session status.

### Bring-up gate check

`vehicle-server bringup <session_id> --gate B0|V0|V1` and `GET /sessions/{id}/bringup?gate=`
(`src/vehicle_server/bringup.py`) answer whether a stored session passes a gate of the
bring-up sequence in `vehicle-defs/docs/hardware-integration.md` §6, one line per "Pass when"
criterion:

- **B0** (desk, mock firmware): report status `ok`, IMU gap count 0, and "at rest" (manual).
- **V0** (bike, ignition on, engine off): ECU present on every CAN health row, every polled DID
  (the BLE telemetry fields with a `defsSignal`) has a valid sample and none out of range, no
  latch flag, no bus-off.
- **V1** (engine idling, closed area): the V0 CAN health checks over the whole session, CAN
  health rows spanning at least 5 min of node time ("clean" = those session-wide checks, not a
  sliding window), and "RPM changes with a throttle blip" (manual: the min / p50 / max of
  `ENGINE_SPEED` is shown).

V0 and V1 also show the §7.2 tester statistics as warnings: a step gap above
`CLIENT_STEP_MAX_MS` or a round trip above `ASSUMED_ROUND_TRIP_MS` (both from `vehicle_defs`). A session without
version 4 tester data (a v3 recording) therefore reaches `incomplete` at best on V0 and V1.

Each check has `id`, `status`, `measured`, `expected`, `source` (doc section or defs constant)
and `note`. Statuses: `pass`, `fail`, `warn` (never fails the gate), `manual` (the doc gives no
number, so **you** judge it from `measured`; it never changes the verdict) and `unknown` (the
session or an older report lacks the data, e.g. no IMU, v2 telemetry without CAN health, no v4
tester data, a report from before the ECU presence counts, or a node reboot during the session).
Verdict: `fail` if any check fails, else `incomplete` if any is unknown, else `pass`
(the text form adds "pending N manual check(s)"). The check only reads `report.json`,
`telemetry.parquet` and `signals.parquet`; it never reads GPS data and never rewrites a report.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `VEHICLE_DATA_DIR` | `./data` | Root of all stored session data and the SQLite index. |
| `VEHICLE_API_TOKEN` | *(none)* | Bearer token for every endpoint except `/health`. Protected endpoints return `503` if unset. |
| `VEHICLE_MAX_UPLOAD_MB` | `100` | Maximum accepted `POST /sessions` upload size. |
| `VEHICLE_CORS_ORIGINS` | *(empty)* | Comma-separated browser origins allowed to call the API (e.g. `http://localhost:5173` for vehicle-analyzer). Empty: no CORS middleware. Methods GET/POST, headers Authorization + Content-Type, no credentials. |
| `VEHICLE_LIVE_SOURCE` | `none` | Producer of `/ws/live`: `none` (heartbeats only) or `mock` (synthetic frames built from defs by `mock_stream.py`; the heartbeat says `"source": "mock"`). The real source (rt-core via the Raspi, D-077) is not built yet. |

## BLE schema

The BLE packet layouts (telemetry versions 2, 3 and 4, the IMU, GPS and vehicle-signal blocks) come from
vehicle-defs: `vehicle_defs.ble.SCHEMA`, generated from its `ble/ble_schema.json`
(D-061). This repo keeps no copy and no drift test; `ble_schema.load_default_schema()`
returns a deep copy of it and the decoder dispatches on the packet's `version` byte. A
layout change is a defs release; bump the `external/vehicle-defs` pin to take it.

## GPS (speed and heading only)

The phone records the GPS block of connectivity-node (D-060, BLE schema
`gpsBlock`) to an optional `gps.csv`, one row per block: receive time, `seq`,
`lost_since_prev`, `raw_hex` (26 bytes, authoritative), the raw fields
(`device_time_ms`, `ground_speed`, `heading_of_motion`, `speed_accuracy`,
`heading_accuracy`, `fix_type`, `num_sv`, `flags`), their scaled values (`*_mps`,
`*_deg`) and the flag bits (`gnss_fix_ok`, `parse_error`, `uart_overflow`). The
server re-decodes `raw_hex` and stores `gps.parquet` from it, with the same
columns plus `fix_type_name`.

The report's `gps` section (null without `gps.csv`) gives the block count, the
blocks **lost or MTU-skipped** from `seq` gaps (the node also advances `seq` for a
block it skips because the MTU is too small, D-062, so this is not a radio-loss
figure), the effective rate from `device_time_ms`, the fix types seen, the share
usable for the CAN speed check (fix3d or gnssDeadReckoning with `gnssFixOk`,
schema `fixType.rule`) and the blocks flagged with a node parse error or UART
overflow (each flag covers the interval since the previous block). A GPS row
whose `raw_hex` does not decode fails the session; GPS gaps, node errors, app/
server mismatches or no usable block make it `warn`.

**No position, ever.** Latitude, longitude and height never leave
connectivity-node (D-060 item 3, invariant 7). The `gps.csv` header must match
exactly, so a session with any extra column (a position included) is rejected.
A speed + heading series can still rebuild the route's shape by dead reckoning
(D-060 item 5, accepted residual risk): in Phase 0 this server runs on the
user's own machine, sessions with `gps.csv` must not be uploaded to a server
that is not local, and real sessions are never committed (D-033).

## Vehicle signals (D-066 item 4)

Every DID of the vehicle-bus tester's table travels in the generic BLE vehicle-signal
block (schema `vehicleSignalBlock`, characteristic `vehicleSignals`; records keyed by DID,
at most 3 blocks per 200 ms). The phone writes them to an optional `signals.csv`, one row
per **record**; the block header (`block_seq`, `device_time_ms`, `total_records`,
`first_record`, and the phone's receive time) is repeated on every row of a block. The
wire values `did`, `state` (`BleVehicleSignalState`: 0 neverReceived, 1 valid, 2 stale,
3 skipped, 4 negativeResponse), `age_ms` (sentinels `65535` = no positive answer since
boot, `65534` = saturated) and `raw` (the data bytes as an unsigned big-endian number for
state 1 and 2, the NRC for state 4) are authoritative. `signal` and `value` are the app's
catalog decode. The contract source is vehicle-mobile's `docs/session-format.md`; the header
must match exactly (any other column is rejected).

The server validates the wire values against the schema field sizes (`did` and `age_ms`
u16, `raw` u32, `state` a known value) and re-derives `signal` and `value` from `did` +
`raw` with the generated vehicle catalog (`vehicle_defs.vehicle_catalog`: `BY_DID`, `SIGNALS`,
`decode`), `raw * factor_num / factor_den + offset`, only for state 1 or 2 and a raw that
fits the DID's `length`. The app's columns are a consistency check (a mismatch makes the
session `warn`). A DID missing from the pinned catalog (a sender built from a newer defs)
is kept with its raw value and an empty signal/value. A DID promoted from the candidate
catalog (D-066 item 3) is handled by the catalog alone: bump the defs pin, no code change.

`signals.parquet` has one row per record: `rx_utc`, `rx_mono_ms`, the block header,
`did`, `state`, `state_name`, `age_ms`, `raw`, `signal`, `unit`, `value` and
`sample_device_time_ms` (`device_time_ms - age_ms`: the node time of the DID's last
positive answer, null for the sentinel ages and for states without one, so a record
repeated by several blocks can be deduplicated). `value` is set for state `valid` only
(a stale value is shown as unavailable; `raw` is always kept). A row whose wire values
fail validation keeps its receive time and a `decode_error`, and fails the session.

The report's `vehicle_signals` section (null without `signals.csv`) gives the block and
record counts, the blocks lost from `block_seq` gaps (`sequenceRule`), and per DID (keyed
in hex): the catalog name and unit, the states seen, the valid-value range, the last
valid value and the NRCs. Lost blocks, app/server mismatches, a raw that does not fit its
DID or a `vehicle_signal_block_version` in `meta.json` that differs from the pinned defs
make the session `warn` (the key is absent in sessions from older app builds). Negative
responses, stale, skipped and unknown DIDs are normal and do not. `signals.csv` holds
engine data only (no position), so the D-063 local-only rule for `gps.csv` does not apply.

## Derived signals (D-070 item 2)

`vehicle_defs.vehicle_derived` defines the derived signals (class A live on rt-core, B per ride,
C across rides, D server only). Until rt-core exists the Python reference in
`src/vehicle_server/derived/` computes them offline from a stored session:

- `inputs.py`: one series per input on the node's `device_time_ms`. `did.<NAME>` rows come
  from `signals.parquet` (`signal == NAME`, `value` set, at `sample_device_time_ms`; the five
  telemetry DIDs are the fallback when there is no `signals.csv`); `sensor.<name>` maps
  through `SENSORS[name]["phase0"]` (`ble.imuBlock.ax` -> `imu.parquet` `ax_g`, converted to
  m/s2 with standard gravity; `ble.gpsBlock.*` -> `gps.parquet`, speed, speed accuracy,
  heading of motion and fix type only, never a position); `const.<name>` is
  `vehicle_defs.vehicle_constants`. Sensors the server does not store (e.g. `lean_angle`) are
  `missing_inputs`; the tester and CAN health counters come from `telemetry.parquet`.
- `engine.py`: topological order over `derived.*` inputs; the output grid is the signal's
  `period_ms` from the span where all inputs exist. An input counts while its age is within
  its own `stale_after_ms` (the DID table's `DID_STALE_AFTER_MS` for `did.*`, `SENSORS` for sensors, the producer's for
  `derived.*`); otherwise the row is null with flag `stale_input`. A null derived input gives
  `invalid_input`, too few samples in a window `insufficient_samples`, a result outside the
  defs `range` null with `out_of_range`. `window_ms` is the least-squares slope window
  (`vehicle_accel`, `vehicle_jerk`) or the moving-average window (`yaw_rate`,
  `accel_imu_residual`).
- `formulas.py`: 34 class-A signals (and 21 D-103 signals in `d103_*.py`, below), from vehicle-data-catalog section 5 (and each signal's
  defs description): part 1 (19, below) and part 2 (fuel economy, trims and throttle/pedal
  residuals, `altitude_change`, `road_grade`, `coolant_rate`, `oil_temp_rate`,
  `lean_crosscheck_residual`, `stop_event`). Assumptions: the ECU speed is km/h and integer; the
  IMU z axis is the yaw axis and x the longitudinal axis (no mounting or lean correction,
  gravity not removed from `accel_imu_residual`); `trip_distance`, `trip_time` and `idle_time`
  integrate from the first common sample and hold a sample for at most its stale limit, so a
  gap adds nothing; `gear_estimate` matches within 5 % of a gear ratio (else 0);
  `turn_radius` is `v / |omega|` and null below 0.5 deg/s; `air_density` is the dry ideal-gas
  density (R = 287.058 J/(kg K)) with the intake temperature; `power_correction_factor` is
  rho_ref / rho with rho_ref at the SAE J1349 reference (99 kPa, 25 degC), the density-ratio
  form rather than the full standard; `air_fuel_ratio` and `fuel_rate_maf` use the commanded
  lambda and `stoichiometric_afr` / `fuel_density_kg_per_l`. Provisional vehicle constants
  used are listed in the report block.
- Part 2 assumptions: `fuel_per_stroke` is one injection per two revolutions;
  `altitude_change` is the hypsometric equation against the first valid barometric sample of
  the session (g0 = 9.80665); `road_grade` uses the first and last samples of altitude and
  trip distance in the window (`sensor.imu_accel_x` is declared as context only, unused);
  `lean_crosscheck_residual` takes both signs as positive = right (to be confirmed on the
  bike) and needs `lean_angle`, which this server does not store (`missing_inputs`).
  New flags: `no_consumption`, `no_power`, `engine_off`.
- Event kind: a formula of kind `event` has no grid; its rows sit at the event times and its
  coverage entry has no `period_ms`. Only `stop_event` exists: a stop starts at the first speed
  0 with RPM above 0 and ends at the first later speed above 0 (value = seconds, row time =
  the end); a gap in the speed samples above the speed DID's own `stale_after_ms` (D-087) or a stop still open at
  the end gives no event. The range check applies to events too.
- D-103 (vehicle-defs v0.23.0, wiring fixes in v0.24.0): 21 more class-A signals in `d103_states.py` (`clutch_state`,
  `clutch_slip_ratio`, `neutral_coasting_state`, `engine_braking_phase`, `over_rev`, `lugging`,
  `cold_engine_abuse`), `d103_windows.py` (`idle_rpm_std`, `idle_rpm_drift`,
  `charging_plateau_voltage`, `fuel_level_filtered`, `road_roughness`, `heat_soak_delta`,
  `idle_map_residual`) and `d103_events.py` (`shift_event`, `launch_event`, `braking_event`,
  `pitch_event`, `refuel_event`, `cranking_dip`, `battery_rest_voltage`), and the class-B
  summaries in `d103_summaries.py`. The rule of each is the "Rule (D-103 ...)" sentence of its
  defs description; **every threshold, band, window and reference is read from its declared
  `const.*` input** (`vehicle/cl250_constants.yaml`, all `provisional`), so a later defs release
  changes the results without a code change. The values are provisional. Where a sentence is
  ambiguous the function docstring names the reading taken. New flags: `not_moving`, `no_gear`,
  `not_applicable`, `no_data`, `no_step`, `starts_warm`, `never_warm`, `no_steady_samples`.
  `road_roughness` band-passes the IMU vertical acceleration with a first-order high-pass and
  low-pass (bilinear sections, pure Python). defs v0.24.0 completed the inputs and rules of
  `wheel_slip`, `launch_event` (measured ratio against `gear_ratio_1`), `volumetric_efficiency`
  (R_air = 287.05 J/(kg K), a physical constant), `intake_flow_residual` (new constant
  `intake_flow_residual_ve_reference_pct`) and the class-B summaries below. Only
  `data_quality_mask` stays `not_implemented` among class A (`NOT_IMPLEMENTED_REASONS_A` in
  `d103_common.py`: bit layout undefined).
- Not computed (`not_implemented`): the class-A signals without a formula (missing thresholds
  or constants or defs gaps), and class C/D; signals with `ready_phase0:
  false` or `pending_inputs` (`not_ready`, never evaluated).

`parquet/derived.parquet` is long format: `session_id` (every row),
`device_time_ms`, `signal`, `value` (float64, null when invalid), `unit`, `valid` (bool) and
`flag` (null, or why the row is invalid). The report's `derived` block holds the defs version
used, `summary` (count per status), `rows` / `valid_rows` and, per signal of the defs, its
`status`: `computed` (with `rows`, `valid_rows`, `flag_counts`), `not_implemented`,
`not_ready` (`reason` `ready_phase0` / `pending_inputs`), `missing_inputs` (`missing`) or
`error` (`message`). It is computed during ingest after the parquet step; an exception is
recorded as `derived: {"error": ...}`, leaves the ingest status unchanged and writes no
`derived.parquet`. `vehicle-server derive` recomputes it after a defs bump or a formula change.

### Ride summaries (class B, part 3)

`src/vehicle_server/derived/summary.py` evaluates the class-B signals after the class-A pass.
Phase 0 has no ride segmentation: **one stored session is one ride** (the report says
`ride_definition: "session"`). Every summary reads every input it declares in the defs (`inputs:`),
so a declared input always gates or changes the result (D-095). A summary reads the valid rows of
its class-A inputs, the scalar results of its class-B inputs (`ride_fuel_consumption` follows
`ride_fuel_used` and `ride_distance`, dependency order) and the raw `did.*` / `sensor.*` series.
Time integrals and
time-in-cell use the class-A hold rule (`formulas.hold_segments`): a value is held between
samples but never beyond the input's freshness (`did.*` of a class-B signal: the DID's own `stale_after_ms`,
D-087; sensors: `SENSORS` `stale_after_ms`), so a gap adds nothing. A value outside the defs
`range` is null with `out_of_range` (each cell too); statuses are the class-A ones.

- Implemented (57 of the 75 class-B signals): `ride_distance`, `ride_time`, `ride_idle_time`
  (last valid `trip_distance` / `trip_time` / `idle_time`), `ride_stop_count` (valid
  `stop_event` rows), `ride_fuel_used` (integral of the fuel rate, L), `ride_fuel_consumption`
  (`ride_fuel_used / ride_distance * 100`; distance <= 0 is `standstill`), `ride_ltft_mean`,
  `ride_stft_std` (population standard deviation, `insufficient_samples` below 2),
  `operating_point_time` ((ENGINE_SPEED, ENGINE_LOAD) cells), `ride_coolant_time_at_temp`,
  `ride_oil_time_at_temp`, `ride_coolant_rate_max`, `ride_start_coolant_temp`,
  `ride_end_coolant_temp`, `ride_start_intake_temp`, `ride_distance_gps` (integral of the GNSS
  ground speed, km; speed only, never a position), `ride_bus_off_count`, `ride_tester_rtt_max`,
  `ride_tester_step_gap_max`.
- Bins: uniform `[min, max)` per axis from the defs `bins`; the cell index is row-major with
  the first `bins` entry as the major axis (`cell = bin0 * count1 + bin1`). The time of a held
  value goes to its cell; a held value outside an axis range is in no cell and counted in the
  report's `outside_s`.
- `ride_bus_off_count` sums the increments of the cumulative counter between consecutive
  samples (the first sample is the baseline); a decrease means the node restarted and the new
  value is the increment. It reads `can_bus_off_count` of `telemetry.parquet`.
  `ride_distance_gps` integrates the GNSS ground speed only over samples where the latest
  `sensor.gps_fix_type` sample is usable (fix3d or gnssDeadReckoning, D-099: no 2D fix) and fresh
  by the same rule as `/track` (`gps_fix.py`);
  a gap without a usable fix adds nothing, and a ride with no usable fix is `missing_inputs`.
- Choices the defs leave open (assumptions of this reference): `ride_bus_off_count` counts
  only bus-offs seen within the session and treats a decrease as a restart; `ride_ltft_mean` is
  a sample mean and `ride_stft_std` a population standard deviation (each DID has a fixed poll
  period, so a sample mean is a time average). A session that contains a key-off and key-on is
  still one "ride" here, so its class-B values mix both; `ride_stop_count` ignores a stop
  still open at the end of the data (the class-A `stop_event` rule).
- D-103 summaries (`d103_summaries.py`): `ride_warmup_time`, `ride_thermal_cycle_count`,
  `throttle_torque_lag`, `speedometer_error`, `effective_tyre_radius`,
  `ride_aggressiveness_score`, `ride_cornering_aggressiveness`, `ride_can_error_count`,
  `ride_unanswered_did_count` and the usage of the new class-A signals (`ride_heat_soak_max`,
  `ride_cranking_dip`, `ride_rest_voltage`, `ride_cold_abuse_time`, `ride_over_rev_time`,
  `ride_lugging_time`, `ride_idle_rpm_std`, `ride_idle_rpm_drift`, `ride_shift_count`,
  `ride_launch_count`), plus (v0.24.0) `ride_fan_on_temperature` and `ride_final_drive_ratio`
  (`per_gear`: one row per gear in `parquet/derived_summary.parquet`, `cell` = gear - 1, and
  `kind: "per_gear"` in `/sessions/{id}/summary`; not a `/rides` scalar). Thresholds come from
  the constants file, as for class A.
- `not_implemented` with a `reason`: `ride_tester_nrc78_count` (needs a per-rttDid sensor),
  `coast_down_cda` and `coast_down_rolling_resistance` (no coast-down speed band), and the
  two D-079 cruise residuals `window_cruise_final_drive_residual` and
  `window_cruise_ve_residual` (see above). `ready_phase0: false` / `pending_inputs`
  signals are `not_ready`; absent inputs are `missing_inputs` (with the list).

#### D-079 health windows (provisional)

The reference (`health_windows.py`, `health_window_summaries.py`) is **provisional (Claude's
reading of D-079, LR17)**. One session is one ride. A window is evaluated on the
merged sample times of its `condition` / `stable` inputs, plus the instant each sample goes
stale; an input counts only while its latest sample is within its own `stale_after_ms`, and an
unknown input makes a condition **not met** (nothing is held), so a lost input closes the
instance at that instant and the gap never counts as window time. A `condition` instance opens
when all conditions hold and closes on the first failing condition, a `stable` spread above its
`max_spread`, or `max_duration_ms` (closed at open + `max_duration_ms`);
`key_on` opens at the first session sample only if its conditions hold, at most once;
`engine_start` needs a valid 0 -> valid > 0 transition on consecutive `ENGINE_SPEED` samples;
`engine_stop` needs a valid > 0 -> valid 0 transition, then holds like `condition`. An
instance qualifies when its duration is at least `min_duration_ms` (a `max_duration_ms` close
also qualifies); condition windows re-arm only after their conditions were not met at least
once. Values use the valid samples inside the first qualifying instance, and
`<window>_count` is the number of qualifying instances (0 when none). No held value and no
GPS input is used. Every evaluated window's instances are also stored in
`derived_summary.health_windows` (shape below) and vehicle-analyzer view E renders them
directly from `report.json`; the analyzer never recomputes a window. Computed (17): W1 `window_key_on_rest_voltage`,
`window_key_on_sensor_spread`, `window_key_on_map_baro_delta`,
`window_key_on_warm_restart`, `window_key_on_count`; W2 `window_crank_count`; W3
`window_warm_idle_rpm_spread`, `window_warm_idle_rpm_drift`,
`window_warm_idle_map_spread`, `window_warm_idle_stft_mean`,
`window_warm_idle_stft_std`, `window_warm_idle_count`; W4 `window_cruise_count`; W5
`window_post_stop_coolant_start`, `window_post_stop_coolant_end`,
`window_post_stop_coolant_tau`, `window_post_stop_count`. Not computed: fuel-pump current /
prime time, crank voltage / current / resistance / duration and W3/W4 vibration (their
inputs are not recorded yet), plus the two W4 cruise residuals above (`not_implemented` with
a defs-specific reason). A window signal with no qualifying instance is null with
`no_window`, no valid sample in its instance is `no_data`, and the STFT / tau fits with too
few points are `insufficient_samples`. Golden B excludes every signal with `window:`
(`SKIP_HEALTH_WINDOW`) until rt-core implements the windows, so the golden manifest is
unchanged. If D-079 or the defs description differs from this reading, they win.

#### Synthetic demo session

`uv run python -m tests.make_demo_session --out <dir>` writes a deterministic 20-minute
session named `synthetic-demo-20260101` (add `--zip` for a zip, `--date YYYY-MM-DD` to change
the day). It is synthetic, never a real ride, has no GPS block or position column, and its
values come from the pinned defs catalog. It exercises all five D-079 windows (including one
`engine_zero` and one `unknown` close), a qualifying and a short warm idle, cruises in gears
4-6, two D-075 stops and a final engine stop with a post-stop cool-down. Import it with
`uv run vehicle-server import <dir-or-zip>`, then view the ride in vehicle-analyzer view E
(summary and Health windows card) and on the phone under Server rides.

`parquet/derived_summary.parquet` is long format, one row per scalar or per cell: `session_id`,
`record` (e.g. `RIDE_USAGE`), `did` (int64, from `RIDE_SUMMARY_RECORDS`), `signal`, `cell`
(int64, null for scalars), `bin0_low`, `bin0_high`, `bin1_low`, `bin1_high` (float64, null when
absent), `value` (float64, null when invalid), `unit`, `valid`, `flag`. The report's
`derived_summary` block: `{defs_version, ride_definition, summary: {status: count}, records:
{RECORD_NAME: {did, signals: {name: entry}}}, health_windows: {rule, t0_device_ms, windows:
{WINDOW_KEY: {qualifying_count, instances: [{start_ms, end_ms, duration_ms, close_reason,
qualifies, first_qualifying}]}}}}`; a scalar entry has `status` / `value` / `unit` /
`flag`, a binned one `status` / `unit` / `cells` / `nonzero_cells` / `total_s` / `outside_s`
(no per-cell values), the others `reason` / `missing` / `message`. `health_windows` has one
entry per evaluated window in `vehicle_derived.WINDOWS` order (currently W1..W5);
`start_ms`/`end_ms`/`duration_ms` are on the `/series` `t0_device_ms` axis, so the first
instance of W3 may show `3.0-30.0 s` in the text report. `instances` is `[]` when a window
never opened; `first_qualifying` is true on exactly one instance when `qualifying_count > 0`
and false on every instance otherwise. A failed summary evaluation stores no
`health_windows` block. It is computed right after
the class-A step; an exception is recorded as `derived_summary: {"error": ...}`, writes no file
and never fails the ingest. The file is removed when there is no row. `vehicle-server derive`
recomputes both files and prints one more line per session. There is no index column.

## Golden vectors for rt-core (D-070 item 2)

`src/vehicle_server/derived/golden.py` runs the Python reference over five synthetic
scenarios (`idle_warmup`, `accel_gears`, `stale_gap`, `start_idle_refuel`, `edge_values`; no GPS, no real ride,
no randomness) and writes the inputs and the expected outputs as C99 headers for the
vehicle-rt-core host tests:

```bash
uv run vehicle-server golden --out <vehicle-rt-core>/tests/host/vectors   # also python -m vehicle_server.derived.golden
```

- Defs version: the headers record the defs version they were generated from (v0.24.0 now).
  The expected outputs use that version's derived `stale_after_ms` (the age limit of a
  derived input; a DID input uses the DID's own limit, D-070), which changed in v0.14.0, so rt-core's derived-module test needs a
  defs pin with the same derived table (v0.14.0 or later); against v0.13.0 the torque and
  power rows would differ. The indices are the same in v0.13.0.
- Location in rt-core: `tests/host/vectors/` (`derived_vectors.h` types + one
  `derived_<scenario>.h` per scenario). The headers are committed there; rt-core never
  depends on this repo, the host tests read no files and parse no strings. The headers use
  `VEHICLE_CL250_IDX_<NAME>` / `VEHICLE_CL250_DERIVED_<NAME>` of the generated defs code.
- Selection rule: class A, a formula in `FORMULAS`, `ready_rt_core` true, no `sensor.*`
  input (a GPS name is asserted never to be an input). Point and series signals have every
  output row; `stop_event` has one row per event at the event time (value = duration, s).
  Inputs are only the DIDs of those signals (plus `BATTERY_VOLTAGE`) at their own
  `poll_period_ms`, quantised to the DID's raw grid and float32. Invalid rows carry `valid =
  false`, a flag and value `0.0f`. The command fails if a selected signal is never valid.
- Tolerance: `abs_tol` = the defs `resolution` / 2 per signal (`derived_<scenario>_tol[]`).
- Drift check: `tests/golden/MANIFEST.sha256` lists the content-sha256 of every file (the
  hash in the banner covers the body after it, so it does not change with the server commit)
  and the defs version. `tests/test_golden.py` regenerates and compares; after a formula or
  defs-pin change, rerun the command above, commit the headers in rt-core and the manifest here.

### Class-B (per-ride) golden vectors (C3 part 2a)

The same command also writes `derived_vectors_b.h` (types) and `derived_<scenario>_b.h` (one per
scenario, plus `derived_two_rides_b.h`) with the expected per-ride summaries of
`derived/summary.py`, `golden_b.py` holds the generator. Same banner / content-sha256 / manifest
as class A; tolerance = resolution / 2.

- Rows: `derived_vector_b_output_t { ride, signal (VEHICLE_CL250_DERIVED_*), element, valid, flag,
  value }`, `derived_vector_tol_t` per signal. `element` is always 0 (reserved).
- Ride = one D-075 stop epoch (D-087 item 2): the vector start opens the ride, its end closes it, so
  single-ride scenarios have no epoch input. Their inputs are those of `derived_<scenario>.h`.
- Rides table `derived_vector_ride_t { ride, start_ms, end_ms }` is the epoch-marker table: at
  `start_ms` the rt-core test opens a new epoch, at `end_ms` it ends the ride (ECU absent) and
  compares that ride's closed values. `two_rides` (accel_gears, then edge_values after an 8 s
  stretch without any input, longer than `ecu_absent_timeout_ms` = 3000 ms, asserted) carries its own
  inputs and each ride is evaluated from a fresh state.
- Selected (12): ride_distance, ride_time, ride_idle_time, ride_stop_count, ride_fuel_used,
  ride_fuel_consumption, ride_ltft_mean, ride_stft_std, ride_coolant_rate_max,
  ride_start_coolant_temp, ride_start_intake_temp, ride_end_coolant_temp.
- Skipped: binned multi-element signals (operating_point_time, ride_coolant/oil_time_at_temp: out of
  scope of 2a), signals with a `sensor.*` input (ride_distance_gps, ride_bus_off_count,
  ride_tester_rtt_max, ride_tester_step_gap_max) and every class-B signal without a summary in the
  reference. No GPS name appears in any file (asserted).

### Ride-summary records (D-098, C3 part 2b)

`src/vehicle_server/ride_record.py` is the raw encoder (and decoder) of the ride-summary record
DIDs 0xFD40-0xFD4E (0xFD42 not served in v1). The layout comes from the defs codegen
(`vehicle_defs.uds.RIDE_RECORDS`, the same run that writes rt-core's `platform_uds_ride_fields[]`):
4-byte header (layout version, big-endian stop-epoch counter of the closed ride, bit 0
`RIDE_CLOSED`), then big-endian unsigned fields `raw = round_half_even((value - range_min) /
resolution)`, computed in float32 like rt-core's writer (`lrintf`, never `roundf`). The all-ones
value of a field is NOT_AVAILABLE: missing, not finite, or a rounded raw value outside
`0..raw_max` (never clamped). With no ride closed every field is NOT_AVAILABLE.

The golden command also writes `derived_ride_records.h` (`derived/golden_records.py`):

- `derived_ride_field_vectors[]`: per row of `platform_uds_ride_fields[]` (order checked against
  the generated `platform_uds.c`) the raw value of range_min, the largest valid value, one
  resolution above and below the range, a missing value and the exact float32 ties 2.5 -> 2 and
  3.5 -> 4 where they exist. Tests the generic writer alone.
- `derived_no_ride_records[]`: every served record before any ride closed (counter 0).
- `derived_<scenario>_ride_records[]`: per ride of each class-B scenario, the encoding of the
  expected closed values of `derived_<scenario>_b.h` (other fields NOT_AVAILABLE), counter =
  ride + 1 (the epoch counter the ride test drives). Exact when the writer is fed those values;
  end to end a field may differ by one raw step (value tolerance resolution / 2).

## Development

```bash
uv run pytest              # tests, including tests/fixtures/{v2,v3,v4,v4_gps,v4_vsig}_session
uv run ruff check .
uv run ruff format --check .
uv run python tests/make_fixtures.py   # regenerate the committed fixtures
```

Fixtures are generated from the schema itself (`ble_schema.pack_fields`), not
hand-written byte offsets; regenerate and commit them if the fixture
scenarios need to change.

## License

MIT, see `LICENSE` (D-036).
