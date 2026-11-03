"""Pure dead-reckoning rules of gps_track.py (TR-1)."""

from __future__ import annotations

import copy
import math
import random
import time

import pytest

from vehicle_server import gps_track
from vehicle_server.defs import ble as defs_ble
from vehicle_server.defs import vehicle_catalog, vehicle_cl250

FIX = {
    entry["name"]: float(entry["value"])
    for entry in defs_ble.SCHEMA["gpsBlock"]["fixType"]["values"]
}


def fix(name: str) -> float:
    return FIX[name]


def series(times: list[int], values: list[float]) -> gps_track.TrackSeries:
    return gps_track.TrackSeries(times, values)


def straight(
    *, speed: float = 10.0, heading: float = 0.0, steps: int = 10, step_ms: int = 100
) -> tuple[gps_track.TrackSeries, gps_track.TrackSeries, gps_track.TrackSeries]:
    times = [i * step_ms for i in range(steps + 1)]
    return (
        series(times, [speed] * len(times)),
        series(times, [heading] * len(times)),
        series(times, [fix("fix3d")] * len(times)),
    )


def test_straight_line_north():
    result = gps_track.build_track(*straight(), max_points=2000)
    assert result.segments_total == 1
    assert result.excluded == {"no_fix": 0, "stale_fix": 0, "untrusted_heading": 0}
    segment = result.segments[0]
    assert all(abs(x) < 1e-9 for x in segment.x_m)
    assert segment.y_m[-1] == pytest.approx(10.0)
    assert result.path_length_m == pytest.approx(10.0)


def test_circle_closes_and_keeps_its_length():
    times = [i * 100 for i in range(101)]
    speed = [10.0] * len(times)
    heading = [(360.0 * t / times[-1]) % 360.0 for t in times]
    track = gps_track.build_track(
        series(times, speed),
        series(times, heading),
        series(times, [fix("fix3d")] * len(times)),
        max_points=2000,
    )
    assert track.segments_total == 1
    segment = track.segments[0]
    radius = 10.0 / (2.0 * math.pi / 10.0)
    assert track.path_length_m == pytest.approx(2.0 * math.pi * radius, rel=0.01)
    closure = math.hypot(segment.x_m[-1] - segment.x_m[0], segment.y_m[-1] - segment.y_m[0])
    assert closure < 0.05 * radius


def test_heading_wrap_uses_a_circular_mean():
    times = [i * 100 for i in range(11)]
    headings = [359.5 if i % 2 == 0 else 0.5 for i in range(len(times))]
    track = gps_track.build_track(
        series(times, [10.0] * len(times)),
        series(times, headings),
        series(times, [fix("fix3d")] * len(times)),
        max_points=2000,
    )
    segment = track.segments[0]
    assert all(abs(x) < 1e-9 for x in segment.x_m)
    assert segment.y_m[-1] == pytest.approx(10.0)
    assert track.path_length_m == pytest.approx(10.0)


def test_gap_splits_and_the_second_segment_continues_from_the_last_point():
    times = [0, 100, 1000, 1100]
    track = gps_track.build_track(
        series(times, [10.0] * 4),
        series(times, [0.0] * 4),
        series(times, [fix("fix3d")] * 4),
        max_points=2000,
    )
    assert track.segments_total == 2
    first, second = track.segments
    assert first.gap_before_ms is None
    assert second.gap_before_ms == 900
    assert (second.x_m[0], second.y_m[0]) == (first.x_m[-1], first.y_m[-1])
    assert track.path_length_m == pytest.approx(2.0)


def test_no_fix_and_stale_fix_are_excluded_and_counted():
    speed_times = [0, 100, 200, 300, 700]
    fix_times = [0, 100, 200, 300]
    track = gps_track.build_track(
        series(speed_times, [10.0] * len(speed_times)),
        series(speed_times, [0.0] * len(speed_times)),
        series(fix_times, [fix("noFix"), fix("fix3d"), fix("fix3d"), fix("fix3d")]),
        max_points=2000,
    )
    assert track.excluded == {"no_fix": 1, "stale_fix": 1, "untrusted_heading": 0}
    assert track.segments_total == 1
    assert track.points_total == 3
    assert track.path_length_m == pytest.approx(2.0)


def test_fix_type_values_are_looked_up_by_name_not_as_literals(monkeypatch: pytest.MonkeyPatch):
    schema = copy.deepcopy(defs_ble.SCHEMA)
    for entry in schema["gpsBlock"]["fixType"]["values"]:
        entry["value"] = float(entry["value"]) + 100.0
    monkeypatch.setattr(defs_ble, "SCHEMA", schema)
    assert gps_track._usable_fix_values() == {
        fix("fix3d") + 100.0,
        fix("gnssDeadReckoning") + 100.0,
    }


def test_fix_type_values_come_from_the_defs_enum_names():
    # The module must look the names up; a literal 2/3/4 would not follow a defs change.
    assert gps_track._usable_fix_values() == {
        fix("fix3d"),
        fix("gnssDeadReckoning"),
    }
    assert fix("noFix") not in gps_track._usable_fix_values()
    # D-099: a 2D fix does not count, like the D-060 item 4 speed check.
    assert fix("fix2d") not in gps_track._usable_fix_values()


def test_low_speed_holds_the_last_trusted_heading():
    times = [0, 100, 200, 300]
    track = gps_track.build_track(
        series(times, [10.0, 0.5, 10.0, 10.0]),
        series(times, [90.0, 0.0, 90.0, 90.0]),
        series(times, [fix("fix3d")] * 4),
        max_points=2000,
        min_heading_speed_mps=1.0,
    )
    segment = track.segments[0]
    assert segment.x_m[-1] > 2.0
    assert all(abs(y) < 1e-9 for y in segment.y_m)


def test_heading_trust_restarts_after_a_gap():
    times = [0, 100, 2000, 2100]
    track = gps_track.build_track(
        series(times, [10.0, 10.0, 0.5, 10.0]),
        series(times, [90.0, 90.0, 0.0, 90.0]),
        series(times, [fix("fix3d")] * 4),
        min_heading_speed_mps=1.0,
    )
    assert track.excluded["untrusted_heading"] == 1
    assert track.segments_total == 2
    first, second = track.segments
    assert second.gap_before_ms == 2000
    assert (second.x_m[0], second.y_m[0]) == (first.x_m[-1], first.y_m[-1])
    assert second.x_m == [first.x_m[-1]]
    assert second.y_m == [first.y_m[-1]]


def test_no_segment_starts_before_the_first_trusted_heading():
    times = [0, 100, 200]
    track = gps_track.build_track(
        series(times, [0.5, 10.0, 10.0]),
        series(times, [0.0, 90.0, 90.0]),
        series(times, [fix("fix3d")] * 3),
        max_points=2000,
        min_heading_speed_mps=1.0,
    )
    assert track.excluded["untrusted_heading"] == 1
    assert track.segments_total == 1
    assert track.points_total == 2
    assert track.segments[0].x_m[-1] == pytest.approx(1.0)


def _corner() -> tuple[gps_track.TrackSeries, gps_track.TrackSeries, gps_track.TrackSeries]:
    times = [i * 100 for i in range(400)]
    headings = [90.0 if i < 200 else 0.0 for i in range(400)]
    return (
        series(times, [10.0] * 400),
        series(times, headings),
        series(times, [fix("fix3d")] * 400),
    )


def test_simplification_is_bounded_deterministic_and_keeps_endpoints_and_corner():
    first = gps_track.build_track(*_corner(), max_points=10)
    second = gps_track.build_track(*_corner(), max_points=10)
    assert first == second
    assert first.downsampled is True
    segment = first.segments[0]
    assert len(segment.t_ms) <= 10
    assert (segment.x_m[0], segment.y_m[0]) == (0.0, 0.0)
    assert segment.x_m[-1] == pytest.approx(199.70710678118655)
    assert segment.y_m[-1] == pytest.approx(199.70710678118655)
    assert any(x > 190.0 and y < 1.0 for x, y in zip(segment.x_m, segment.y_m, strict=True))


def _reference_rdp(xs: list[float], ys: list[float], tolerance_m: float) -> list[int]:
    """Textbook recursive RDP, kept here only to check the one-pass significance."""

    def keep(start: int, end: int) -> list[int]:
        ax, ay, bx, by = xs[start], ys[start], xs[end], ys[end]
        dx, dy = bx - ax, by - ay
        denominator = dx * dx + dy * dy
        best, best_i = -1.0, -1
        for i in range(start + 1, end):
            if denominator == 0.0:
                d = (xs[i] - ax) ** 2 + (ys[i] - ay) ** 2
            else:
                d = ((xs[i] - ax) * dy - (ys[i] - ay) * dx) ** 2 / denominator
            if d > best:
                best, best_i = d, i
        if best > tolerance_m * tolerance_m:
            return keep(start, best_i)[:-1] + keep(best_i, end)
        return [start, end]

    return keep(0, len(xs) - 1) if len(xs) > 1 else list(range(len(xs)))


@pytest.mark.parametrize("tolerance_m", [0.0, 0.05, 0.5, 2.0, 20.0])
def test_significance_threshold_equals_rdp_at_that_tolerance(tolerance_m: float):
    # A wobbly synthetic path: random walk of headings, fixed seed.
    rng = random.Random(7)
    xs, ys, heading = [0.0], [0.0], 0.0
    for _ in range(300):
        heading += rng.gauss(0.0, 0.3)
        xs.append(xs[-1] + math.sin(heading))
        ys.append(ys[-1] + math.cos(heading))
    significance = gps_track._rdp_significance(xs, ys)
    kept = [i for i, s in enumerate(significance) if s > tolerance_m * tolerance_m]
    assert kept == _reference_rdp(xs, ys, tolerance_m)


def test_simplification_of_a_long_ride_is_fast():
    # One hour at 10 Hz: the max_points search must not rerun RDP per bisection step.
    rng = random.Random(1)
    steps = 36000
    times = [i * 100 for i in range(steps)]
    headings, heading = [], 0.0
    for _ in range(steps):
        heading = (heading + rng.gauss(0.0, 2.0)) % 360.0
        headings.append(heading)
    start = time.perf_counter()
    track = gps_track.build_track(
        series(times, [10.0] * steps),
        series(times, headings),
        series(times, [fix("fix3d")] * steps),
        max_points=2000,
    )
    assert time.perf_counter() - start < 2.0
    assert sum(len(segment.t_ms) for segment in track.segments) <= 2000


def test_a_candidate_did_without_a_poll_limit_is_held():
    candidate = next(
        name for name in vehicle_catalog.SIGNALS if name not in vehicle_cl250.DID_STALE_AFTER_MS
    )
    assert gps_track.series_stale_after_ms(f"did.{candidate}") is None
    served = next(iter(vehicle_cl250.DID_STALE_AFTER_MS))
    assert gps_track.series_stale_after_ms(f"did.{served}") == int(
        vehicle_cl250.DID_STALE_AFTER_MS[served]
    )


def test_too_many_segments_for_max_points_raises():
    times: list[int] = []
    for i in range(6):
        times.extend([i * 1000, i * 1000 + 100])
    with pytest.raises(gps_track.TrackTooManyPoints):
        gps_track.build_track(
            series(times, [10.0] * len(times)),
            series(times, [0.0] * len(times)),
            series(times, [fix("fix3d")] * len(times)),
            max_points=10,
        )


def test_color_sampling_holds_until_staleness():
    color = series([0, 1000], [1.0, 2.0])
    assert gps_track.sample_series(color, [0, 300, 301, 1000, 1300, 1301], 300) == [
        1.0,
        1.0,
        None,
        2.0,
        2.0,
        None,
    ]


def test_single_sample_is_one_point_with_zero_length():
    track = gps_track.build_track(
        series([100], [10.0]),
        series([100], [90.0]),
        series([100], [fix("fix3d")]),
    )
    assert track.segments_total == 1
    assert track.points_total == 1
    assert track.path_length_m == 0.0
    assert track.segments[0].gap_before_ms is None
    assert track.segments[0].t_ms == [100]


def test_all_no_fix_is_excluded_without_inventing_a_segment():
    times = [0, 100, 200]
    track = gps_track.build_track(
        series(times, [10.0] * 3),
        series(times, [0.0] * 3),
        series(times, [fix("noFix")] * 3),
    )
    assert track.segments == []
    assert track.segments_total == 0
    assert track.points_total == 0
    assert track.path_length_m == 0.0
    assert track.excluded == {"no_fix": 3, "stale_fix": 0, "untrusted_heading": 0}


def test_non_finite_input_is_excluded_and_never_reaches_the_output():
    times = [0, 100, 200, 300, 400]
    track = gps_track.build_track(
        series(times, [10.0, math.nan, math.inf, 10.0, 10.0]),
        series(times, [0.0, 0.0, 0.0, math.nan, math.inf]),
        series(times, [fix("fix3d")] * 5),
    )
    assert track.excluded == {"no_fix": 2, "stale_fix": 0, "untrusted_heading": 2}
    assert track.segments_total == 1
    assert track.points_total == 1
    assert all(
        math.isfinite(value)
        for segment in track.segments
        for values in (segment.x_m, segment.y_m, segment.speed_mps, segment.heading_deg)
        for value in values
    )


def test_a_session_shorter_than_the_stale_limit_is_one_segment():
    times = [0, 100]
    track = gps_track.build_track(
        series(times, [10.0, 10.0]),
        series(times, [0.0, 0.0]),
        series(times, [fix("fix3d"), fix("fix3d")]),
    )
    assert track.segments_total == 1
    assert track.segments[0].gap_before_ms is None
    assert track.path_length_m == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_min_heading_speed_is_rejected(bad: float):
    with pytest.raises(ValueError):
        gps_track.build_track(*straight(), min_heading_speed_mps=bad)
