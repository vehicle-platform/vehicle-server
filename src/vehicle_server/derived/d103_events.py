"""D-103 class-A events (LR15-3): shift, launch, braking, pitch, refuel, cranking dip and
battery rest voltage. Rows are at the event times (see stop_event); thresholds come from the
declared const.* inputs."""

from __future__ import annotations

from bisect import bisect_left, bisect_right

from vehicle_server.derived.context import At, Env, InputInvalid
from vehicle_server.derived.d103_common import (
    _ACCEL,
    _BATT,
    _FUEL_LEVEL,
    _GEAR,
    _GYRO_Y,
    _IDLE_MIN,
    _MOVING_MIN,
    _RPM,
    _SPEED,
    _speed_at,
)
from vehicle_server.derived.formulas import EventRow, _event

# --- LR15-3: events ------------------------------------------------------------------------------


@_event("shift_event")
def shift_event(env: Env) -> list[EventRow]:
    """An ENGINE_SPEED change (either direction) of at least shift_event_rpm_step_min_rpm
    between two samples at most shift_event_window_ms apart while the VEHICLE_SPEED (fresh at
    both times) changed by at most shift_event_speed_change_max_kmh.

    Reading: the row is at the sample that completes the step, the value is the time from the
    EARLIEST sample of the window that forms such a step to it (the duration of the step, ms).
    After an event the search restarts from the completing sample (no overlapping events). A
    gap in the ENGINE_SPEED samples above its stale_after_ms ends the search."""
    step = env.const("const.shift_event_rpm_step_min_rpm")
    window = env.const("const.shift_event_window_ms")
    max_change = env.const("const.shift_event_speed_change_max_kmh")
    rpm = env.feed(_RPM)
    events: list[EventRow] = []
    recent: list[tuple[int, float, float | None]] = []
    previous: int | None = None
    for t, value in zip(rpm.times, rpm.values, strict=True):
        if value is None:
            continue
        if previous is not None and t - previous > rpm.stale_after_ms:
            recent = []
        previous = t
        recent = [s for s in recent if t - s[0] <= window]
        speed = _speed_at(env, t)
        if speed is not None:
            for start_t, start_rpm, start_speed in recent:
                if (
                    start_speed is not None
                    and abs(value - start_rpm) >= step
                    and abs(speed - start_speed) <= max_change
                ):
                    events.append((t, float(t - start_t), None))
                    recent = []
                    break
        recent.append((t, value, speed))
    return events


@_event("launch_event")
def launch_event(env: Env) -> list[EventRow]:
    """Moving off from rest. It starts at the speed sample that is above
    derived_moving_speed_min_kmh when the previous sample was not; it ends at the first later
    sample where the speed is above launch_event_end_speed_kmh or gear_estimate is 1. The
    value is the time from start to end (ms), the row is at the end.

    Reading: "the measured ratio is within derived_gear_ratio_tolerance of gear 1" is read as
    gear_estimate == 1 (the declared input that carries the measured ratio's match); that
    estimate uses its own fixed match tolerance, derived_gear_ratio_tolerance cannot be
    applied (the measured ratio is not a declared input). A launch that falls back to rest
    before it ends, or a gap in the speed samples above the speed's stale_after_ms, yields
    nothing. THROTTLE_POS and ENGINE_SPEED are declared, not used."""
    moving = env.const(_MOVING_MIN)
    end_speed = env.const("const.launch_event_end_speed_kmh")
    speed = env.feed(_SPEED)
    events: list[EventRow] = []
    start: int | None = None
    previous_t: int | None = None
    previous_v: float | None = None
    for t, v in zip(speed.times, speed.values, strict=True):
        if v is None:
            continue
        if previous_t is not None and t - previous_t > speed.stale_after_ms:
            start, previous_v = None, None
        if start is not None:
            try:
                first_gear = At(env, t).value(_GEAR) == 1
            except InputInvalid:
                first_gear = False
            if v > end_speed or first_gear:
                events.append((t, float(t - start), None))
                start = None
            elif v <= moving:
                start = None
        elif previous_v is not None and previous_v <= moving < v:
            start = t
        previous_t, previous_v = t, v
    return events


@_event("braking_event")
def braking_event(env: Env) -> list[EventRow]:
    """vehicle_accel below -braking_event_decel_min_mps2 for at least
    braking_event_min_duration_ms. The run lasts from its first sample below the threshold to
    the first sample that is not (the value is held until the next sample); the row is at that
    end sample, the value is the lowest acceleration (the peak deceleration, negative). An
    invalid sample, a gap above the accel's stale_after_ms, or a run still open at the end of
    the data yields nothing."""
    threshold = -env.const("const.braking_event_decel_min_mps2")
    min_duration = env.const("const.braking_event_min_duration_ms")
    feed = env.feed(_ACCEL)
    events: list[EventRow] = []
    run_start: int | None = None
    peak = 0.0
    previous: int | None = None
    for t, v in zip(feed.times, feed.values, strict=True):
        if v is None:
            run_start, previous = None, None
            continue
        if previous is not None and t - previous > feed.stale_after_ms:
            run_start = None
        previous = t
        if v < threshold:
            peak = v if run_start is None else min(peak, v)
            run_start = t if run_start is None else run_start
        else:
            if run_start is not None and t - run_start >= min_duration:
                events.append((t, peak, None))
            run_start = None
    return events


@_event("pitch_event")
def pitch_event(env: Env) -> list[EventRow]:
    """An excursion of |imu_gyro_y| above pitch_event_rate_min_dps while moving (speed above
    derived_moving_speed_min_kmh, fresh at the sample). The excursion ends at the first sample
    that is not above the threshold or not moving; the row is at that sample, the value is the
    peak |imu_gyro_y| of the excursion. A gap above the gyro's stale_after_ms or an excursion
    still open at the end of the data yields nothing."""
    threshold = env.const("const.pitch_event_rate_min_dps")
    moving_min = env.const(_MOVING_MIN)
    feed = env.feed(_GYRO_Y)
    events: list[EventRow] = []
    peak: float | None = None
    previous: int | None = None
    for t, v in zip(feed.times, feed.values, strict=True):
        if v is None:
            peak, previous = None, None
            continue
        if previous is not None and t - previous > feed.stale_after_ms:
            peak = None
        previous = t
        speed = _speed_at(env, t)
        if speed is not None and speed > moving_min and abs(v) > threshold:
            peak = abs(v) if peak is None else max(peak, abs(v))
        else:
            if peak is not None:
                events.append((t, peak, None))
            peak = None
    return events


@_event("refuel_event")
def refuel_event(env: Env) -> list[EventRow]:
    """FUEL_LEVEL rise at rest. Between a moving sample (speed above
    derived_moving_speed_min_kmh) and the next moving sample, with at least one sample at rest
    in between, the fuel level (fresh at both moving samples) rose by at least
    refuel_event_step_min_pct. The row is at the next moving sample, the value is the added
    litres, step x fuel_tank_capacity_l / 100.

    Reading: the ECU may be off while refuelling, so a gap in the speed samples does not end
    this detection (the level is taken at the two moving samples and must be fresh there)."""
    moving_min = env.const(_MOVING_MIN)
    step_min = env.const("const.refuel_event_step_min_pct")
    tank = env.const("const.fuel_tank_capacity_l")
    speed = env.feed(_SPEED)
    events: list[EventRow] = []
    last_moving: int | None = None
    rested = False
    for t, v in zip(speed.times, speed.values, strict=True):
        if v is None:
            continue
        if v > moving_min:
            if last_moving is not None and rested:
                try:
                    rise = At(env, t).value(_FUEL_LEVEL) - At(env, last_moving).value(_FUEL_LEVEL)
                except InputInvalid:
                    rise = 0.0
                if rise >= step_min:
                    events.append((t, rise * tank / 100.0, None))
            last_moving, rested = t, False
        elif last_moving is not None:
            rested = True
    return events


def _engine_starts(env: Env) -> list[tuple[int, int, float]]:
    """(last zero-RPM sample, first RPM > 0 sample, that sample's rpm) per start: RPM goes
    from 0 to above 0. A gap in the ENGINE_SPEED samples above its stale_after_ms forgets the
    zero. An engine already running at the start of the data has no start."""
    rpm = env.feed(_RPM)
    starts: list[tuple[int, int, float]] = []
    zero_t: int | None = None
    previous: int | None = None
    for t, v in zip(rpm.times, rpm.values, strict=True):
        if v is None:
            continue
        if previous is not None and t - previous > rpm.stale_after_ms:
            zero_t = None
        previous = t
        if v <= 0:
            zero_t = t
        elif zero_t is not None:
            starts.append((zero_t, t, v))
            zero_t = None
    return starts


@_event("cranking_dip")
def cranking_dip(env: Env) -> list[EventRow]:
    """The lowest BATTERY_VOLTAGE sample during a start. A start begins at the first
    ENGINE_SPEED sample above 0 after a sample at 0 and ends when ENGINE_SPEED first exceeds
    derived_idle_rpm_min; the voltage samples with a time in [start, end] count. The row is at
    the end, the value the lowest voltage; a start without a voltage sample in it, or still
    open at the end of the data (also when the RPM falls back to 0), yields nothing."""
    idle_min = env.const(_IDLE_MIN)
    rpm, batt = env.feed(_RPM), env.feed(_BATT)
    events: list[EventRow] = []
    zero_t: int | None = None
    start: int | None = None
    previous: int | None = None
    for t, v in zip(rpm.times, rpm.values, strict=True):
        if v is None:
            continue
        if previous is not None and t - previous > rpm.stale_after_ms:
            zero_t = start = None
        previous = t
        if v <= 0:
            zero_t, start = t, None
            continue
        if start is None and zero_t is not None:
            start, zero_t = t, None
        if start is not None and v > idle_min:
            lo, hi = bisect_left(batt.times, start), bisect_right(batt.times, t)
            lows = [b for b in batt.values[lo:hi] if b is not None]
            if lows:
                events.append((t, min(lows), None))
            start = None
    return events


@_event("battery_rest_voltage")
def battery_rest_voltage(env: Env) -> list[EventRow]:
    """The BATTERY_VOLTAGE before the crank. The description has no rule sentence; the
    reading: at each start (ENGINE_SPEED from 0 to above 0) the value is the battery voltage
    fresh at the last sample with ENGINE_SPEED 0 before it, the row is at the first sample
    above 0. A start without a fresh voltage at that moment yields nothing."""
    events: list[EventRow] = []
    for zero_t, start_t, _ in _engine_starts(env):
        try:
            events.append((start_t, At(env, zero_t).value(_BATT), None))
        except InputInvalid:
            continue
    return events
