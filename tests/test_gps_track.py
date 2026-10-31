"""Pure dead-reckoning rules of gps_track.py (TR-1)."""

from __future__ import annotations

import math

import pytest

from vehicle_server import gps_track
from vehicle_server.defs import ble as defs_ble

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


def test_fix_type_values_come_from_the_defs_enum_names():
    # The module must look the names up; a literal 2/3/4 would not follow a defs change.
    assert gps_track._usable_fix_values() == {
        fix("fix2d"),
        fix("fix3d"),
        fix("gnssDeadReckoning"),
    }
    assert fix("noFix") not in gps_track._usable_fix_values()


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
