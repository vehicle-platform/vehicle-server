# CLAUDE.md — vehicle-server

@.claude/PLATFORM-RULES.md

## What this repo is

The server side (Python, D-010): receives, stores, and serves ride data coming from the vehicle.

### v0 as built

- **Ingestion:** `POST /sessions` (FastAPI, `src/vehicle_server/api.py`). Multipart field `archive` = a session `.zip` written by the phone app (`vehicle-mobile`, BLE path of D-032). The zip holds `meta.json`, `telemetry.csv`, `events.csv`, `summary.json` and optional `imu.csv`, `gps.csv` (D-060) and `signals.csv` (the D-066 item 4 vehicle-signal block), at the zip root or in one top-level folder. Responses: `201` created, `200` identical re-upload (idempotent), `409` same session id with different content, `422` invalid session, `413` over `VEHICLE_MAX_UPLOAD_MB`. The same ingest runs from the CLI (`vehicle-server import <folder-or-zip>`).
- **Read API:** `GET /sessions` (list), `GET /sessions/{id}/report` (`?format=text` for a summary), `GET /sessions/{id}/series` (`src/vehicle_server/series.py`: `did.*` / `sensor.*` via `SessionInputs`, `derived.*` from `derived.parquet`; shared t0, min/max downsampling, `max_points` 10..20000; never a GPS position), `GET /sessions/{id}/stats` (exact count/mean/std/min/max/p05/p50/p95 over the full-resolution points, `start_ms`/`end_ms` on the `/series` t0, 404 for an unknown name), `GET /sessions/{id}/track` (`src/vehicle_server/gps_track.py`, pure: a relative dead-reckoned track from GNSS speed + heading in metres from the first usable fix, never an absolute position; fix types by defs enum name, staleness from defs, segments at gaps, exact RDP to `max_points`, optional `color=` series), `GET /sessions/{id}/summary` and `GET /rides` (`src/vehicle_server/ride_summary.py`, read-only: the stored class-B ride summaries in defs order, binned cells from `derived_summary.parquet` with `defs_mismatch` when stored and defs bin edges differ; `/rides` = one row per session from the `report.json` blocks only, class-B scalars, `from`/`to` inclusive UTC), `GET /health` (no auth). Position-looking names are refused by `src/vehicle_server/position.py`, the one rule shared by `/series`, `/stats`, `/track` and the MDF4 export (D-076). Everything else needs a bearer token in the Authorization header (`VEHICLE_API_TOKEN`; 503 if unset). `VEHICLE_CORS_ORIGINS` (comma-separated, default empty = no CORS) lets vehicle-analyzer call it from a browser.
- **Live stream:** `GET /ws/live` (WebSocket, `src/vehicle_server/stream.py`) for vehicle-analyzer: token in the first `hello` message, per-client bounded queue (drop oldest), heartbeats with the source name. `VEHICLE_LIVE_SOURCE=mock` feeds it from `mock_stream.py`; the default `none` sends heartbeats only.
- **Validation and gap report (D-032, D-045):** every session gets a `report.json` with the telemetry layout/packet versions, the defs version used to decode, a raw_hex re-decode check, packet loss recomputed from `seq`, rx/device-time ordering and gaps, per-signal range/validity checks against `vehicle_defs`, CAN health (v3), tester stats (v4: last step gap, last per-DID round-trip record, D-058), IMU sample-loss/scale checks and a GPS block summary (D-060: blocks lost or MTU-skipped from `seq`, fix types, share usable for the speed check, node parse/UART errors) and a vehicle-signal summary (D-066: blocks lost from `block_seq`, per DID the states seen, valid value range, NRCs, unknown DIDs). Status is `ok` / `warn` / `fail`.
- **Bring-up gate check:** `GET /sessions/{id}/bringup?gate=B0|V0|V1` and `vehicle-server bringup <id> --gate ... [--json]` (`src/vehicle_server/bringup.py`, pure checks + `load_inputs`): one check per `hardware-integration.md` §6 "Pass when" criterion, §7.2 tester stats as `warn` only, `manual` where the doc gives no number, `unknown` (never `pass`) when the report lacks the data; verdict `fail` > `incomplete` > `pass` (exit 1 / 2 / 0). Thresholds from `vehicle_cl250` or quoted from §6; reads `report.json` + telemetry/signals parquet, never GPS. The report's `can_health` counts `ecuPresent` (`ecu_present_rows`/`ecu_absent_rows`/`ecu_present_last`).
- **Storage:** files + SQLite index under `VEHICLE_DATA_DIR`: raw upload verbatim, decoded **Parquet** (`telemetry.parquet`, `imu.parquet`, `gps.parquet`, `signals.parquet`, `derived.parquet` under `parquet/`), `report.json`, `index.sqlite` (`has_imu`, `has_gps`, `has_vehicle_signals`, `has_derived`; new columns are added to an existing index on open). The decoded telemetry Parquet also carries the 8 version 4 tester columns (nullable integers, same names as in `telemetry.csv`) and the 6 version 3 CAN health fields (nullable integers, schema field names in snake_case, e.g. `can_bus_off_count`; derived-signal sensors read them). Decoding uses the BLE schema from the defs submodule (`vehicle_defs.ble.SCHEMA`, D-061; no copy in this repo, no drift test) and signals from `vehicle_defs`, never hand-written. `signals.csv` rows are re-decoded from `did` + `raw` through the generated vehicle catalog (`vehicle_defs.vehicle_catalog`, D-066): a DID promoted from the candidate catalog needs no change here, only a defs bump.
- **Derived signals (D-070 item 2, parts 1-3):** `src/vehicle_server/derived/` is the Python reference of `vehicle_defs.vehicle_derived`: `inputs.py` builds per-input series from the session's parquet files (device_time_ms axis; DIDs from `signals.parquet`, the five telemetry DIDs as fallback; sensors by field-name conversion of `SENSORS[...]["phase0"]`; GPS speed/heading only, never a position), `engine.py` evaluates in dependency order on the defs' `period_ms` grid (stale input -> null, result outside `range` -> null + `out_of_range`, never a held value), `formulas.py` holds the arithmetic. 34 class-A signals (+21 D-103): part 1 = 19 (gearbox ratio, gear estimate, vehicle accel/jerk, IMU residual, trip distance/time, idle time, yaw rate, lateral accel, turn radius, engine/demand torque, torque gap, engine power, air density, power correction, AFR, MAF fuel rate), part 2 = fuel economy, trims, throttle/pedal residuals, altitude change, road grade, coolant/oil rates, lean cross-check and the `stop_event` event (formula kind `event`, no grid); D-103 (defs v0.24.0) adds 21 class-A signals (`d103_states/windows/events.py`) and class-B summaries (`d103_summaries.py`): the rule is the "Rule (D-103 ...)" sentence of each description and every threshold is read from its declared `const.*` input, never a literal (values are provisional); `data_quality_mask`, `ride_tester_nrc78_count`, the coast-down pair and the two D-079 cruise residuals stay `not_implemented` with a reason; the 17 ready D-079 health-window signals (W1-W5, provisional rules in the README) come from the pure state machine `health_windows.py` + `health_window_summaries.py` and golden B skips every `window:` signal until rt-core implements the windows; `per_gear` summaries (`ride_final_drive_ratio`) store one row per gear (cell = gear - 1); class C/D are `not_implemented`, `ready_phase0: false` / `pending_inputs` signals are `not_ready`. Output `derived.parquet` (long format) + the `derived` block of `report.json`; computed after the parquet step, an exception is recorded as `derived.error` and never fails the ingest. `vehicle-server derive <session-id>|--all` recomputes stored sessions. Part 3: class-B per-ride summaries (`derived/summary.py`, one session = one ride) → `derived_summary.parquet` + the `derived_summary` report block, including the D-079 `health_windows` instances on the `/series` t0 axis (a failure there stays `health_windows.error` and never costs the scalars). `vehicle-server golden --out <vehicle-rt-core>/tests/host/vectors` writes the synthetic golden vectors for rt-core (C headers; drift check `tests/golden/MANIFEST.sha256`; need the same defs derived table, v0.14.0+). Part 2b (D-098): `ride_record.py` encodes/decodes the generated ride-summary record DIDs 0xFD40-0xFD4E (float32 + half-even like rt-core's `lrintf` writer, all ones = NOT_AVAILABLE, never clamped); the golden command adds `derived_ride_records.h` (field vectors, no-ride records, per-ride expected records).

### Planned (not built yet)

- Ingestion of rt-core's binary log files and a real live source for `/ws/live` (Kuksa over MQTT/Zenoh, D-077). (Persistent on-vehicle logging is rt-core's job, D-045.)
- **MDF4** export for archive/sharing (`mdf4.py` is a skeleton that raises `NotImplementedError`).
- Moving storage to InfluxDB + Grafana if it grows (the Eclipse Fleet Management pattern is a reference).
- API: `query_ride_history`, ride summaries, firmware packages (OTA source).

## What this repo is NOT

- No model training (`vehicle-ml` reads data from here).
- Never sends commands to the vehicle. It only **serves** the OTA package; distribution is done by the Raspi.

## Rules

- The data schema and metadata match `../vehicle-defs/docs/phase0-data-collection-plan.md` §3 exactly. A session ID is present in every record (ML's per-session split relies on this).
- Signal names/scales come from `vehicle_defs` (the pinned defs submodule), never hand-written. The report records the defs version used to decode. (Planned with the rt-core binary log: pick the decoder by the defs version in the log header.)
- Location history is stored encrypted at the disk/DB level. Externally exposed endpoints require TLS + authentication.

## Dependencies

`external/vehicle-defs` (tagged).

## Build

`uv` + `ruff` + `pytest` (D-007), FastAPI. Local services run via `docker compose`.

```bash
git submodule update --init --recursive   # external/vehicle-defs, pinned to v0.24.0
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run vehicle-server serve      # needs VEHICLE_API_TOKEN
VEHICLE_API_TOKEN=... docker compose up --build
```

See README.md for the CLI, HTTP API, storage layout and environment variables.

## Context

ARCHITECTURE §7 · `../vehicle-defs/docs/phase0-data-collection-plan.md` §3, §8 · `../vehicle-defs/docs/hardware-architecture.md` §5b.8.
