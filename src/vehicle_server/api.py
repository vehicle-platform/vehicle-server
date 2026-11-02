"""FastAPI application: session ingestion + read API.

Bearer token auth (VEHICLE_API_TOKEN) guards every endpoint except GET /health.
GET /ws/live (WebSocket) checks the same token in its first "hello" message
(vehicle_server.stream); VEHICLE_LIVE_SOURCE picks its producer ("none" or "mock").
If VEHICLE_API_TOKEN is not set, protected endpoints return 503 (the API does
not silently run open).
"""

from __future__ import annotations

import contextlib
import hmac
import json
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from vehicle_server import __version__, gps_track, index, report, ride_summary, series, stream
from vehicle_server.config import Settings, load_settings
from vehicle_server.ingest import (
    SESSION_ID_RE,
    InvalidSessionError,
    SessionConflictError,
    ingest_path,
)
from vehicle_server.position import is_position_name

_UPLOAD_CHUNK_BYTES = 1024 * 1024


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


SettingsDep = Annotated[Settings, Depends(get_settings)]


def require_token(request: Request, settings: SettingsDep) -> None:
    if settings.api_token is None:
        raise HTTPException(status_code=503, detail="VEHICLE_API_TOKEN is not configured")
    auth = request.headers.get("authorization", "")
    # Constant-time comparison so response timing does not leak the token.
    if not hmac.compare_digest(auth.encode(), f"Bearer {settings.api_token}".encode()):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


AuthDep = Depends(require_token)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    hub = stream.LiveHub(settings.live_source)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        async with stream.running_source(hub):
            yield

    app = FastAPI(title="vehicle-server", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.live_hub = hub

    if settings.cors_origins:
        # Opt-in (VEHICLE_CORS_ORIGINS) for the analyzer in a browser on another origin. The
        # token travels in the Authorization header, so credentials (cookies) stay off.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
            allow_credentials=False,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/sessions", dependencies=[AuthDep], status_code=201)
    async def create_session(archive: UploadFile, settings: SettingsDep) -> JSONResponse:
        with tempfile.TemporaryDirectory(prefix="vehicle-server-upload-") as tmp:
            tmp_zip = Path(tmp) / "upload.zip"
            total = 0
            with tmp_zip.open("wb") as out:
                while chunk := await archive.read(_UPLOAD_CHUNK_BYTES):
                    total += len(chunk)
                    if total > settings.max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"upload exceeds VEHICLE_MAX_UPLOAD_MB={settings.max_upload_mb}",
                        )
                    out.write(chunk)

            try:
                result = ingest_path(tmp_zip, settings)
            except InvalidSessionError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except SessionConflictError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

            status_code = 201 if result.created else 200
            return JSONResponse(status_code=status_code, content=result.report)

    @app.get("/sessions", dependencies=[AuthDep])
    def list_sessions(settings: SettingsDep) -> list[dict]:
        rows = index.list_sessions(settings.index_db_path)
        return [index.as_dict(row) for row in rows]

    @app.get("/sessions/{session_id}/report", dependencies=[AuthDep])
    def get_session_report(session_id: str, settings: SettingsDep, format: str = "json"):
        report_path = settings.sessions_dir / session_id / "report.json"
        if not report_path.is_file():
            raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")

        report_dict = json.loads(report_path.read_text())
        if format == "text":
            return PlainTextResponse(report.render_text(report_dict))
        return JSONResponse(content=report_dict)

    @app.get("/sessions/{session_id}/series", dependencies=[AuthDep])
    def get_session_series(
        session_id: str,
        settings: SettingsDep,
        signals: str | None = None,
        max_points: Annotated[
            int, Query(ge=series.MIN_MAX_POINTS, le=series.MAX_MAX_POINTS)
        ] = series.DEFAULT_MAX_POINTS,
        from_ms: int | None = None,
        to_ms: int | None = None,
    ) -> dict:
        session_dir = settings.sessions_dir / session_id
        if not SESSION_ID_RE.match(session_id) or not session_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")

        available = series.load_all(session_dir)
        names: list[str] | None = None
        if signals is not None:
            names = list(dict.fromkeys(n.strip() for n in signals.split(",") if n.strip()))
            bad = [n for n in names if n not in available]
            if bad or not names:
                raise HTTPException(
                    status_code=422,
                    detail=f"unknown or unavailable series: {', '.join(bad) or '(empty list)'}",
                )
        return series.build_response(session_id, available, names, max_points, from_ms, to_ms)

    @app.get("/sessions/{session_id}/track", dependencies=[AuthDep])
    def get_session_track(
        session_id: str,
        settings: SettingsDep,
        max_points: Annotated[
            int, Query(ge=gps_track.MIN_MAX_POINTS, le=gps_track.MAX_MAX_POINTS)
        ] = gps_track.DEFAULT_MAX_POINTS,
        min_heading_speed_mps: Annotated[
            float, Query(ge=gps_track.MIN_HEADING_SPEED_MPS, le=gps_track.MAX_HEADING_SPEED_MPS)
        ] = gps_track.DEFAULT_MIN_HEADING_SPEED_MPS,
        color: str | None = None,
        from_ms: int | None = None,
        to_ms: int | None = None,
    ) -> dict:
        session_dir = settings.sessions_dir / session_id
        if not SESSION_ID_RE.match(session_id) or not session_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")

        available = series.load_all(session_dir)
        speed = available.get("sensor.gps_ground_speed")
        heading = available.get("sensor.gps_heading")
        fix_type = available.get("sensor.gps_fix_type")
        if speed is None or heading is None or fix_type is None:
            raise HTTPException(status_code=404, detail="session has no GPS")

        if color is not None and (is_position_name(color) or color not in available):
            # Same status and wording as /stats for an unknown or position-looking name.
            raise HTTPException(status_code=404, detail=f"unknown or unavailable series: {color}")

        t0 = series.t0_of(available)
        lo = None if from_ms is None else t0 + from_ms
        hi = None if to_ms is None else t0 + to_ms
        speed_times, speed_values = series.window(speed.times, speed.values, lo, hi)
        try:
            track = gps_track.build_track(
                gps_track.TrackSeries(speed_times, speed_values),
                gps_track.TrackSeries(heading.times, heading.values),
                gps_track.TrackSeries(fix_type.times, fix_type.values),
                max_points=max_points,
                min_heading_speed_mps=min_heading_speed_mps,
            )
        except gps_track.TrackTooManyPoints as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if not track.segments and from_ms is None and to_ms is None:
            raise HTTPException(status_code=404, detail="session has no GPS")

        color_series = available.get(color) if color is not None else None
        color_stale = gps_track.series_stale_after_ms(color) if color is not None else None
        segments: list[dict] = []
        for segment in track.segments:
            entry: dict[str, object] = {
                "t_ms": [t - t0 for t in segment.t_ms],
                "x_m": segment.x_m,
                "y_m": segment.y_m,
                "speed_mps": segment.speed_mps,
                "heading_deg": segment.heading_deg,
                "gap_before_ms": segment.gap_before_ms,
            }
            if color_series is not None:
                entry["color_v"] = gps_track.sample_series(
                    gps_track.TrackSeries(color_series.times, color_series.values),
                    segment.t_ms,
                    color_stale,
                )
            segments.append(entry)

        return {
            "session_id": session_id,
            "t0_device_ms": t0,
            "method": "dead_reckoning",
            "frame": "relative_enu_from_first_fix",
            "max_points": max_points,
            "points_total": track.points_total,
            "downsampled": track.downsampled,
            "path_length_m": track.path_length_m,
            "segments_total": track.segments_total,
            "excluded": track.excluded,
            "color": (
                {"name": color, "unit": color_series.unit}
                if color is not None and color_series is not None
                else None
            ),
            "segments": segments,
        }

    @app.get("/sessions/{session_id}/summary", dependencies=[AuthDep])
    def get_session_summary(session_id: str, settings: SettingsDep) -> dict:
        session_dir = settings.sessions_dir / session_id
        if not SESSION_ID_RE.match(session_id) or not session_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
        try:
            return ride_summary.load_ride_summary(session_dir, session_id)
        except ride_summary.NoRideSummary as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/rides", dependencies=[AuthDep])
    def list_rides(
        settings: SettingsDep,
        from_: str | None = Query(None, alias="from"),
        to: str | None = None,
        names: str | None = None,
    ) -> dict:
        try:
            selected = ride_summary.select_scalar_names(names)
            start = ride_summary.parse_utc_bound(from_, end=False) if from_ else None
            end = ride_summary.parse_utc_bound(to, end=True) if to else None
        except ride_summary.BadRideQuery as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        sessions: list[tuple[str, str | None, dict | None]] = []
        for row in index.list_sessions(settings.index_db_path):
            session_dir = settings.sessions_dir / row.session_id
            sessions.append(
                (
                    row.session_id,
                    row.created_utc,
                    ride_summary.read_summary_block(session_dir),
                )
            )
        return ride_summary.build_rides(sessions, selected, start, end)

    @app.get("/sessions/{session_id}/stats", dependencies=[AuthDep])
    def get_session_stats(
        session_id: str,
        settings: SettingsDep,
        signals: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> dict:
        session_dir = settings.sessions_dir / session_id
        if not SESSION_ID_RE.match(session_id) or not session_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")

        available = series.load_all(session_dir)
        names: list[str] | None = None
        if signals is not None:
            names = list(dict.fromkeys(n.strip() for n in signals.split(",") if n.strip()))
            bad = [n for n in names if n not in available]
            if bad or not names:
                raise HTTPException(
                    status_code=404,
                    detail=f"unknown or unavailable series: {', '.join(bad) or '(empty list)'}",
                )
        return series.build_stats_response(session_id, available, names, start_ms, end_ms)

    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket) -> None:
        await stream.serve_client(websocket, hub, settings.api_token)

    return app


app = create_app()
