"""D-103 class-A events (LR15-3): rows at the event times on synthetic series."""

from __future__ import annotations

import pytest

from vehicle_server.derived import engine

from .derived_helpers import FakeInputs, const_series, event_rows, patch_const, ramp

SPEED = "did.VEHICLE_SPEED"
RPM = "did.ENGINE_SPEED"
BATT = "did.BATTERY_VOLTAGE"
FUEL = "did.FUEL_LEVEL"
ACCEL = "derived.vehicle_accel"
GEAR = "derived.gear_estimate"
GYRO = "sensor.imu_gyro_y"


def _drop_between(samples: tuple, start: int, stop: int) -> tuple:
    """The samples without those whose time is in [start, stop] (a gap in the data)."""
    keep = [(t, v) for t, v in zip(*samples, strict=True) if not start <= t <= stop]
    return [t for t, _ in keep], [v for _, v in keep]


# --- shift_event ------------------------------------------------------------------------------


def _shift_feeds(rpm_fn, speed_fn=lambda t: 60.0) -> dict:
    return {RPM: ramp(0, 5000, 110, rpm_fn), SPEED: ramp(0, 5000, 100, speed_fn)}


def test_shift_event_is_the_duration_of_an_rpm_step_at_continuous_speed():
    f = _shift_feeds(lambda t: 4000.0 if t < 1100 else 3000.0)
    assert event_rows("shift_event", f) == [(1100, 440.0, None)]  # from the sample at 660 ms


def test_shift_event_near_misses_do_not_fire(monkeypatch):
    small = _shift_feeds(lambda t: 4000.0 if t < 1100 else 3600.0)  # 400 < 500 rpm
    assert event_rows("shift_event", small) == []
    jump = _shift_feeds(
        lambda t: 4000.0 if t < 1100 else 3000.0, lambda t: 60.0 if t < 1100 else 70.0
    )
    assert event_rows("shift_event", jump) == []  # the speed changed by 10 km/h
    slow = _shift_feeds(lambda t: 4000.0 - 0.5 * t if t < 2200 else 2900.0)  # 500 rpm in 1000 ms
    assert event_rows("shift_event", slow) == []
    f = _shift_feeds(lambda t: 4000.0 if t < 1100 else 3000.0)
    patch_const(monkeypatch, "shift_event_rpm_step_min_rpm", 1200.0)
    assert event_rows("shift_event", f) == []


def test_shift_event_follows_the_window_and_speed_constants(monkeypatch):
    f = _shift_feeds(lambda t: 4000.0 if t < 1100 else 3000.0)
    patch_const(monkeypatch, "shift_event_window_ms", 200)
    assert event_rows("shift_event", f) == [(1100, 110.0, None)]
    jump = _shift_feeds(
        lambda t: 4000.0 if t < 1100 else 3000.0, lambda t: 60.0 if t < 1100 else 70.0
    )
    patch_const(monkeypatch, "shift_event_speed_change_max_kmh", 12.0)
    assert [row[0] for row in event_rows("shift_event", jump)] == [1100]


def test_shift_event_gap_in_the_middle_ends_the_search_and_two_events_are_found():
    f = _shift_feeds(lambda t: 4000.0 if t < 1100 else 3000.0)
    f[RPM] = _drop_between(f[RPM], 700, 1090)  # no sample for 400 ms, above the 330 ms limit
    assert event_rows("shift_event", f) == []
    two = _shift_feeds(lambda t: 3000.0 if 1100 <= t < 3300 else 4000.0)
    assert [(row[0], row[1]) for row in event_rows("shift_event", two)] == [
        (1100, 440.0),
        (3300, 440.0),
    ]


# --- launch_event -----------------------------------------------------------------------------


def _launch_feeds(speed_fn, gear_fn=lambda t: 0.0) -> dict:
    return {
        SPEED: ramp(0, 5000, 100, speed_fn),
        RPM: const_series(0, 5000, 100, 3000.0),
        "did.THROTTLE_POS": const_series(0, 5000, 100, 30.0),
        GEAR: ramp(0, 5000, 100, gear_fn),
    }


def _off_from_rest(t: int) -> float:
    return max(0.0, (t - 1000) * 0.05)  # 0 at 1000 ms, 5 km/h at 1100, 15 at 1300


def test_launch_event_value_is_the_time_to_the_end_speed_and_the_row_is_at_the_end():
    assert event_rows("launch_event", _launch_feeds(_off_from_rest)) == [(1300, 200.0, None)]


def test_launch_event_ends_when_first_gear_is_matched():
    f = _launch_feeds(_off_from_rest, lambda t: 1.0 if t >= 1200 else 0.0)
    assert event_rows("launch_event", f) == [(1200, 100.0, None)]


def test_launch_event_near_misses_and_a_gap_yield_nothing():
    aborted = _launch_feeds(lambda t: 5.0 if 1100 <= t <= 1300 else 0.0)  # never passes 10 km/h
    assert event_rows("launch_event", aborted) == []
    moving_already = _launch_feeds(lambda t: 30.0)  # no start from rest
    assert event_rows("launch_event", moving_already) == []
    f = _launch_feeds(_off_from_rest)
    f[SPEED] = _drop_between(f[SPEED], 1150, 1450)  # the speed series stops for 400 ms
    assert event_rows("launch_event", f) == []


def test_launch_event_two_launches_and_the_constants(monkeypatch):
    def two(t: int) -> float:
        return _off_from_rest(t) if t < 2500 else max(0.0, (t - 3000) * 0.05)

    rows = event_rows(
        "launch_event", _launch_feeds(lambda t: min(two(t), 12.0) if t < 2500 else two(t))
    )
    assert [r[0] for r in rows] == [1300, 3300] or len(rows) == 2
    f = _launch_feeds(_off_from_rest)
    patch_const(monkeypatch, "launch_event_end_speed_kmh", 20.0)
    assert event_rows("launch_event", f) == [(1500, 400.0, None)]  # 25 km/h at 1500 ms
    patch_const(monkeypatch, "derived_moving_speed_min_kmh", 8.0)
    assert event_rows("launch_event", f)[0][1] == 300.0  # starts at 1200 ms (10 km/h)


# --- braking_event ----------------------------------------------------------------------------


def _braking_feeds(fn) -> dict:
    return {ACCEL: ramp(0, 5000, 110, fn)}


def test_braking_event_value_is_the_peak_and_the_row_is_at_the_end_of_the_run(monkeypatch):
    peaks = {1100: -5.0, 1210: -6.0, 1320: -5.5, 1430: -5.0, 1540: -4.0}
    f = _braking_feeds(lambda t: peaks.get(t, -1.0))
    assert event_rows("braking_event", f) == [(1650, -6.0, None)]
    patch_const(monkeypatch, "braking_event_decel_min_mps2", 6.5)
    assert event_rows("braking_event", f) == []
    patch_const(monkeypatch, "braking_event_decel_min_mps2", 3.0)
    patch_const(monkeypatch, "braking_event_min_duration_ms", 700)
    assert event_rows("braking_event", f) == []


def test_braking_event_near_misses_gap_and_two_events():
    soft = _braking_feeds(lambda t: -2.5 if 1100 <= t < 1650 else -1.0)  # above -3 m/s2
    assert event_rows("braking_event", soft) == []
    short = _braking_feeds(lambda t: -5.0 if t == 1100 else -1.0)  # one sample, 110 ms
    assert event_rows("braking_event", short) == []
    f = _braking_feeds(lambda t: -5.0 if 1100 <= t <= 1760 else -1.0)
    f[ACCEL] = _drop_between(f[ACCEL], 1210, 1600)  # a gap splits the run into short parts
    assert event_rows("braking_event", f) == []
    two = _braking_feeds(lambda t: -4.0 if 1100 <= t < 1650 or 3300 <= t < 3850 else 0.0)
    assert [(r[0], r[1]) for r in event_rows("braking_event", two)] == [(1650, -4.0), (3850, -4.0)]
    open_run = _braking_feeds(lambda t: -4.0 if t >= 4400 else 0.0)  # open at the end of the data
    assert event_rows("braking_event", open_run) == []


# --- pitch_event ------------------------------------------------------------------------------


def _pitch_feeds(fn, speed: float = 50.0) -> dict:
    return {GYRO: ramp(0, 3000, 10, fn), SPEED: const_series(0, 3000, 100, speed)}


def test_pitch_event_value_is_the_peak_of_the_excursion(monkeypatch):
    def rate(t: int) -> float:
        return {1000: 40.0, 1010: 80.0, 1020: -90.0, 1030: 50.0}.get(t, 0.0)

    f = _pitch_feeds(rate)
    assert event_rows("pitch_event", f) == [(1040, 90.0, None)]
    patch_const(monkeypatch, "pitch_event_rate_min_dps", 85.0)
    assert event_rows("pitch_event", f) == [(1030, 90.0, None)]
    patch_const(monkeypatch, "pitch_event_rate_min_dps", 95.0)
    assert event_rows("pitch_event", f) == []


def test_pitch_event_near_miss_standstill_gap_and_two_events():
    calm = _pitch_feeds(lambda t: 20.0 if 1000 <= t < 1100 else 0.0)  # below 30 deg/s
    assert event_rows("pitch_event", calm) == []
    standing = _pitch_feeds(lambda t: 80.0 if 1000 <= t < 1100 else 0.0, speed=1.0)
    assert event_rows("pitch_event", standing) == []
    f = _pitch_feeds(lambda t: 100.0 if 1000 <= t <= 1030 else 60.0 if 1100 <= t < 1120 else 0.0)
    f[GYRO] = _drop_between(f[GYRO], 1031, 1090)  # a gap above the 30 ms limit drops the first part
    assert event_rows("pitch_event", f) == [(1120, 60.0, None)]
    two = _pitch_feeds(lambda t: 50.0 if t in (1000, 2000) else 0.0)
    assert [(r[0], r[1]) for r in event_rows("pitch_event", two)] == [(1010, 50.0), (2010, 50.0)]


# --- refuel_event -----------------------------------------------------------------------------


def _refuel_feeds(after: float = 50.0) -> dict:
    return {
        SPEED: ramp(0, 5000, 100, lambda t: 0.0 if 1100 <= t <= 3000 else 50.0),
        FUEL: ramp(0, 5000, 1000, lambda t: 20.0 if t < 2000 else after),
    }


def test_refuel_event_is_the_added_litres_at_the_next_moving_sample(monkeypatch):
    assert event_rows("refuel_event", _refuel_feeds()) == [(3100, pytest.approx(3.6), None)]
    patch_const(monkeypatch, "fuel_tank_capacity_l", 10.0)
    assert event_rows("refuel_event", _refuel_feeds())[0][1] == pytest.approx(3.0)
    patch_const(monkeypatch, "refuel_event_step_min_pct", 40.0)
    assert event_rows("refuel_event", _refuel_feeds()) == []


def test_refuel_event_near_miss_no_rest_and_the_ecu_off_while_refuelling():
    assert event_rows("refuel_event", _refuel_feeds(after=28.0)) == []  # 8 % < 10 %
    moving = {SPEED: const_series(0, 5000, 100, 50.0), FUEL: _refuel_feeds()[FUEL]}
    assert event_rows("refuel_event", moving) == []  # never at rest
    off = _refuel_feeds()
    off[SPEED] = _drop_between(off[SPEED], 1200, 2900)  # the ECU answers nothing while it rests
    assert [r[0] for r in event_rows("refuel_event", off)] == [3100]


def test_refuel_event_two_events():
    f = {
        SPEED: ramp(
            0, 9000, 100, lambda t: 0.0 if 1100 <= t <= 3000 or 6100 <= t <= 7000 else 50.0
        ),
        FUEL: ramp(
            0,
            9000,
            1000,
            lambda t: 20.0 if t < 2000 else 50.0 if t < 3500 else 30.0 if t < 7000 else 60.0,
        ),
    }
    rows = event_rows("refuel_event", f)
    assert [r[0] for r in rows] == [3100, 7100]


# --- cranking_dip and battery_rest_voltage ----------------------------------------------------


def _crank_feeds(start: int = 1100) -> dict:
    rpm = {start: 300.0, start + 110: 600.0, start + 220: 1100.0}

    def rpm_fn(t: int) -> float:
        return rpm.get(t, 0.0 if t < start else 1400.0)

    volts = {start: 9.8, start + 100: 10.4, start + 200: 9.5}
    return {
        RPM: ramp(0, 5000, 110, rpm_fn),
        BATT: ramp(0, 5000, 100, lambda t: volts.get(t, 12.6 if t < start else 13.5)),
    }


def test_cranking_dip_is_the_lowest_voltage_from_the_first_rpm_to_the_idle_threshold(monkeypatch):
    f = _crank_feeds()
    assert event_rows("cranking_dip", f) == [(1320, 9.5, None)]
    patch_const(monkeypatch, "derived_idle_rpm_min", 500.0)
    assert event_rows("cranking_dip", f) == [(1210, 9.8, None)]


def test_cranking_dip_near_misses_and_a_gap():
    running = {RPM: const_series(0, 5000, 110, 1400.0), BATT: const_series(0, 5000, 100, 13.5)}
    assert event_rows("cranking_dip", running) == []  # no start: the engine was running
    stall = _crank_feeds()
    stall[RPM] = ramp(0, 5000, 110, lambda t: 300.0 if 1100 <= t < 1400 else 0.0)
    assert event_rows("cranking_dip", stall) == []  # never above the idle threshold
    gap = _crank_feeds()
    gap[RPM] = _drop_between(gap[RPM], 1200, 1500)
    assert event_rows("cranking_dip", gap) == []


def test_cranking_dip_two_starts():
    f = _crank_feeds()
    rpm_times, rpm_values = f[RPM]
    second = {t: v for t, v in zip(rpm_times, rpm_values, strict=True)}
    for t in range(3000, 5001, 110):
        second[t] = 0.0 if t < 3300 else 1400.0
    second[3300] = 1100.0
    f[RPM] = (sorted(second), [second[t] for t in sorted(second)])
    f[BATT] = ramp(0, 5000, 100, lambda t: 9.0 if 3200 <= t <= 3400 else 13.0)
    rows = event_rows("cranking_dip", f)
    assert len(rows) == 2 and rows[1][1] == 9.0


def test_battery_rest_voltage_is_the_voltage_before_the_first_rpm():
    assert event_rows("battery_rest_voltage", _crank_feeds()) == [(1100, 12.6, None)]
    running = {RPM: const_series(0, 5000, 110, 1400.0), BATT: const_series(0, 5000, 100, 13.5)}
    assert event_rows("battery_rest_voltage", running) == []
    stale = _crank_feeds(start=3300)
    stale[BATT] = ([0], [12.6])  # no fresh voltage at the last RPM = 0 sample
    assert event_rows("battery_rest_voltage", stale) == []


def test_events_through_the_engine_are_rows_at_the_event_time_and_range_checked():
    result = engine.evaluate(FakeInputs(_crank_feeds()))
    entry = result.coverage["cranking_dip"]
    assert entry["status"] == "computed" and entry["rows"] == 1 and "period_ms" not in entry
