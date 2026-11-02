"""Dead-reckoned relative ride track from GNSS speed and heading (D-060, D-076).

This module is deliberately independent of FastAPI and of stored sessions: it takes three
full-resolution series (ground speed, heading of motion and fix type) and returns segments
in metres relative to the first usable sample. It never reads or produces a position.

The rules (the same ones the /track endpoint and its README row document):

* a sample is usable only with a fix type whose defs enum name is fix2d, fix3d or
  gnssDeadReckoning, and only while the latest fix-type sample is fresh;
* below ``min_heading_speed_mps`` the heading of motion is noise, so the last trusted
  heading is held; a segment does not start before the first trusted heading;
* between two usable samples the distance is the speed trapezoid and the direction is the
  circular mean of the two headings (359 deg and 1 deg average to 0 deg, not 180 deg);
* a gap longer than the speed sensor's ``stale_after_ms``, or any excluded sample, starts a
  new segment at the previous position; the unknown movement is never invented;
* RDP simplification is per segment and keeps both endpoints; if the endpoints alone do
  not fit ``max_points`` the caller gets ``TrackTooManyPoints`` (HTTP 422).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from vehicle_server.defs import ble as defs_ble
from vehicle_server.defs import vehicle_cl250, vehicle_derived

DEFAULT_MAX_POINTS = 2000
MIN_MAX_POINTS = 10
MAX_MAX_POINTS = 20000
DEFAULT_MIN_HEADING_SPEED_MPS = 1.0
MIN_HEADING_SPEED_MPS = 0.0
MAX_HEADING_SPEED_MPS = 10.0

USABLE_FIX_NAMES = ("fix2d", "fix3d", "gnssDeadReckoning")


class TrackTooManyPoints(ValueError):
    """The segment endpoints alone exceed max_points; the API maps this to HTTP 422."""


@dataclass(frozen=True)
class TrackSeries:
    """A full-resolution series on the node's device_time_ms axis (ascending)."""

    times: list[int]
    values: list[float]


@dataclass(frozen=True)
class TrackSegment:
    """One continuous piece of the track; all lists have the same length."""

    t_ms: list[int]
    x_m: list[float]
    y_m: list[float]
    speed_mps: list[float]
    heading_deg: list[float]
    gap_before_ms: int | None


@dataclass(frozen=True)
class TrackResult:
    segments: list[TrackSegment]
    points_total: int
    path_length_m: float
    segments_total: int
    downsampled: bool
    excluded: dict[str, int]


@dataclass(frozen=True)
class _Sample:
    t: int
    speed: float
    heading: float
    break_before: bool = False


@dataclass(frozen=True)
class _Segment:
    samples: list[_Sample]
    gap_before_ms: int | None


def _fix_values_by_name() -> dict[str, float]:
    gps = defs_ble.SCHEMA["gpsBlock"]
    return {entry["name"]: float(entry["value"]) for entry in gps["fixType"]["values"]}


def _usable_fix_values() -> frozenset[float]:
    by_name = _fix_values_by_name()
    missing = [name for name in USABLE_FIX_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"gps fixType enum has no {', '.join(missing)}")
    return frozenset(by_name[name] for name in USABLE_FIX_NAMES)


def _sensor_stale_after_ms(name: str) -> int:
    return int(vehicle_derived.SENSORS[name]["stale_after_ms"])


def series_stale_after_ms(name: str) -> int | None:
    """Freshness limit of a /series name, from defs; None when the defs give none.

    DIDs use their own stale_after_ms, sensors use SENSORS, derived signals use the
    derived table's stale_after_ms or (when absent) three sample periods. A catalog DID the
    vehicle's poll plan does not serve (a candidate) has no limit, so it is held.
    """
    kind, _, key = name.partition(".")
    if kind == "did":
        limit = vehicle_cl250.DID_STALE_AFTER_MS.get(key)
        return None if limit is None else int(limit)
    if kind == "sensor":
        return int(vehicle_derived.SENSORS[key]["stale_after_ms"])
    if kind == "derived":
        spec = vehicle_derived.SIGNALS.get(key, {})
        if "stale_after_ms" in spec:
            return int(spec["stale_after_ms"])
        if "period_ms" in spec:
            return int(spec["period_ms"]) * 3
    return None


def _latest(times: list[int], t: int) -> int:
    return bisect.bisect_right(times, t) - 1


def _circular_mean_deg(first: float, second: float) -> float:
    a = math.radians(first)
    b = math.radians(second)
    x = math.sin(a) + math.sin(b)
    y = math.cos(a) + math.cos(b)
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        # Exactly opposite headings have no circular mean; hold the first one.
        return first % 360.0
    return math.degrees(math.atan2(x, y)) % 360.0


def _usable_samples(
    speed: TrackSeries,
    heading: TrackSeries,
    fix_type: TrackSeries,
    min_heading_speed_mps: float,
) -> tuple[list[_Sample], dict[str, int]]:
    """Filters speed samples and returns (usable samples, excluded counts)."""
    usable_values = _usable_fix_values()
    heading_stale = _sensor_stale_after_ms("gps_heading")
    fix_stale = _sensor_stale_after_ms("gps_fix_type")
    speed_stale = _sensor_stale_after_ms("gps_ground_speed")

    excluded = {"no_fix": 0, "stale_fix": 0, "untrusted_heading": 0}
    samples: list[_Sample] = []
    last_heading: float | None = None
    pending_break = False

    def exclude(kind: str) -> None:
        nonlocal pending_break, last_heading
        excluded[kind] += 1
        if samples:
            pending_break = True
            # A gap starts a new segment; its first trusted heading must be fresh.
            last_heading = None

    for t, speed_mps in zip(speed.times, speed.values, strict=True):
        if not math.isfinite(speed_mps):
            exclude("no_fix")
            continue

        fix_i = _latest(fix_type.times, t)
        if fix_i < 0 or t - fix_type.times[fix_i] > fix_stale:
            exclude("stale_fix")
            continue
        if fix_type.values[fix_i] not in usable_values:
            exclude("no_fix")
            continue

        heading_i = _latest(heading.times, t)
        if heading_i < 0 or t - heading.times[heading_i] > heading_stale:
            exclude("untrusted_heading")
            continue
        heading_deg = heading.values[heading_i]
        if not math.isfinite(heading_deg):
            exclude("untrusted_heading")
            continue

        if samples and t - samples[-1].t > speed_stale:
            # The next sample starts a new segment after a gap; do not carry the
            # previous segment's heading into it.
            last_heading = None

        if speed_mps < min_heading_speed_mps:
            if last_heading is None:
                exclude("untrusted_heading")
                continue
            heading_deg = last_heading
        else:
            last_heading = heading_deg

        samples.append(_Sample(t, float(speed_mps), heading_deg, pending_break))
        pending_break = False

    return samples, excluded


def _rdp_significance(xs: list[float], ys: list[float]) -> list[float]:
    """Squared RDP significance of every point (endpoints: infinity).

    One full Ramer-Douglas-Peucker pass with tolerance 0 (deterministic; the first farthest
    point wins). A point's significance is the smallest squared distance on its split chain,
    so RDP at tolerance ``t`` keeps exactly the points whose significance exceeds ``t**2``:
    the split order does not depend on the tolerance, only where the recursion stops. That
    turns the max_points search into one pass and a sort instead of RDP per bisection step.
    """
    n = len(xs)
    significance = [0.0] * n
    if n == 0:
        return significance
    significance[0] = significance[-1] = math.inf
    stack = [(0, n - 1, math.inf)]
    while stack:
        start, end, parent = stack.pop()
        ax, ay = xs[start], ys[start]
        bx, by = xs[end], ys[end]
        dx, dy = bx - ax, by - ay
        denominator = dx * dx + dy * dy
        farthest = -1.0
        farthest_index = -1
        for i in range(start + 1, end):
            px, py = xs[i], ys[i]
            if denominator == 0.0:
                distance_sq = (px - ax) ** 2 + (py - ay) ** 2
            else:
                cross = (px - ax) * dy - (py - ay) * dx
                distance_sq = cross * cross / denominator
            if distance_sq > farthest:
                farthest = distance_sq
                farthest_index = i
        if farthest > 0.0:
            chain = min(parent, farthest)
            significance[farthest_index] = chain
            stack.append((start, farthest_index, chain))
            stack.append((farthest_index, end, chain))
    return significance


def _build_segments_with_positions(
    samples: list[_Sample],
) -> tuple[list[_Segment], float, list[tuple[list[float], list[float]]]]:
    """Splits usable samples at gaps and integrates each continuous run."""
    speed_stale = _sensor_stale_after_ms("gps_ground_speed")
    segments: list[_Segment] = []
    positions: list[tuple[list[float], list[float]]] = []
    current: list[_Sample] = []
    xs: list[float] = []
    ys: list[float] = []
    previous: _Sample | None = None
    gap_before: int | None = None
    path_length_m = 0.0

    def finish() -> None:
        nonlocal current, xs, ys
        if current:
            segments.append(_Segment(current, gap_before))
            positions.append((xs, ys))
        current, xs, ys = [], [], []

    for sample in samples:
        if previous is None:
            current = [sample]
            xs, ys = [0.0], [0.0]
            previous = sample
            continue

        gap_ms = sample.t - previous.t
        if sample.break_before or gap_ms > speed_stale:
            finish()
            current = [_Sample(sample.t, sample.speed, sample.heading)]
            xs, ys = [positions[-1][0][-1]], [positions[-1][1][-1]]
            gap_before = gap_ms
            previous = sample
            continue

        dt_s = gap_ms / 1000.0
        distance = (previous.speed + sample.speed) / 2.0 * dt_s
        direction = _circular_mean_deg(previous.heading, sample.heading)
        radians = math.radians(direction)
        xs.append(xs[-1] + distance * math.sin(radians))
        ys.append(ys[-1] + distance * math.cos(radians))
        current.append(sample)
        path_length_m += distance
        previous = sample

    finish()
    return segments, path_length_m, positions


def simplify_segments(
    segments: list[_Segment],
    positions: list[tuple[list[float], list[float]]],
    max_points: int,
) -> tuple[list[list[int]], bool]:
    """RDP indices per segment, and whether any point was dropped."""
    total = sum(len(segment.samples) for segment in segments)
    if total <= max_points:
        return [list(range(len(segment.samples))) for segment in segments], False

    endpoint_count = sum(min(2, len(segment.samples)) for segment in segments)
    if endpoint_count > max_points:
        raise TrackTooManyPoints(
            f"track needs at least {endpoint_count} points for its {len(segments)} segments, "
            f"more than max_points={max_points}"
        )

    significance = [_rdp_significance(xs, ys) for xs, ys in positions]
    # The smallest tolerance whose RDP result fits: the (max_points + 1)-th largest
    # significance. Ties at that value are all dropped, so the count stays <= max_points.
    ranked = sorted((s for values in significance for s in values), reverse=True)
    threshold = ranked[max_points] if len(ranked) > max_points else 0.0
    kept = [[i for i, s in enumerate(values) if s > threshold] for values in significance]
    return kept, any(
        len(indices) < len(segment.samples) for segment, indices in zip(segments, kept, strict=True)
    )


def build_track(
    speed: TrackSeries,
    heading: TrackSeries,
    fix_type: TrackSeries,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    min_heading_speed_mps: float = DEFAULT_MIN_HEADING_SPEED_MPS,
) -> TrackResult:
    """Builds the relative dead-reckoned track from three full-resolution series."""
    if not MIN_MAX_POINTS <= max_points <= MAX_MAX_POINTS:
        raise ValueError(f"max_points must be {MIN_MAX_POINTS}..{MAX_MAX_POINTS}")
    if not MIN_HEADING_SPEED_MPS <= min_heading_speed_mps <= MAX_HEADING_SPEED_MPS:
        raise ValueError(
            f"min_heading_speed_mps must be {MIN_HEADING_SPEED_MPS}..{MAX_HEADING_SPEED_MPS}"
        )

    samples, excluded = _usable_samples(speed, heading, fix_type, min_heading_speed_mps)
    segments, path_length_m, positions = _build_segments_with_positions(samples)
    kept, downsampled = simplify_segments(segments, positions, max_points)

    out: list[TrackSegment] = []
    for segment, indices, (xs, ys) in zip(segments, kept, positions, strict=True):
        out.append(
            TrackSegment(
                t_ms=[segment.samples[i].t for i in indices],
                x_m=[xs[i] for i in indices],
                y_m=[ys[i] for i in indices],
                speed_mps=[segment.samples[i].speed for i in indices],
                heading_deg=[segment.samples[i].heading for i in indices],
                gap_before_ms=segment.gap_before_ms,
            )
        )
    return TrackResult(
        segments=out,
        points_total=sum(len(segment.samples) for segment in segments),
        path_length_m=path_length_m,
        segments_total=len(segments),
        downsampled=downsampled,
        excluded=excluded,
    )


def sample_series(
    series: TrackSeries, times: list[int], stale_after_ms: int | None
) -> list[float | None]:
    """Latest sample at or before each time, or None when it is older than the limit."""
    out: list[float | None] = []
    for t in times:
        i = _latest(series.times, t)
        if i < 0 or (stale_after_ms is not None and t - series.times[i] > stale_after_ms):
            out.append(None)
        else:
            out.append(series.values[i])
    return out
