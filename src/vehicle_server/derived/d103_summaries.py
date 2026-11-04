"""D-103 class-B ride summaries (LR15-4).

The rule of each is the "Rule (D-103 ...)" sentence of its defs description; every threshold,
reference or window comes from a declared const.* input through SummaryEnv.const. Where a
sentence is ambiguous the simplest literal reading is taken and written in the docstring. One
session is one ride (D-087), so every summary covers the whole session.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from statistics import median

from vehicle_server.defs import vehicle_derived
from vehicle_server.derived.context import At, InputInvalid
from vehicle_server.derived.formulas import Outcome
from vehicle_server.derived.summary import SummaryEnv, _summary
from vehicle_server.gps_fix import usable_fix_values

FLAG_NO_DATA = "no_data"
FLAG_NO_STEP = "no_step"
FLAG_STARTS_WARM = "starts_warm"
FLAG_NEVER_WARM = "never_warm"
FLAG_NO_STEADY_SAMPLES = "no_steady_samples"
FLAG_INVALID_INPUT = "invalid_input"

_COOLANT = "did.COOLANT_TEMP"
_RUN_TIME = "did.RUN_TIME_SINCE_START"
_THROTTLE = "did.THROTTLE_POS"
_TORQUE_PCT = "did.ACTUAL_TORQUE_PCT"
_RPM = "did.ENGINE_SPEED"
_SPEED = "did.VEHICLE_SPEED"
_GPS_SPEED = "sensor.gps_ground_speed"
_GPS_ACCURACY = "sensor.gps_speed_accuracy"
_GPS_FIX = "sensor.gps_fix_type"
_LEAN = "sensor.lean_angle"
_TX_ERRORS = "sensor.can_tx_error_count"
_RX_ERRORS = "sensor.can_rx_error_count"
_UNANSWERED = "sensor.unanswered_did_count"

# Unit conversion: GNSS speed is in m/s, the ECU speed in km/h.
_MPS_TO_KMH = 3.6


def _valid(env: SummaryEnv, name: str) -> list[float]:
    values = env.valid(name)
    if not values:
        raise InputInvalid(FLAG_INVALID_INPUT)
    return values


def _percentile_95(values: list[float]) -> float:
    """Linear-interpolation 95th percentile (numpy's default method); values not empty."""
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _clamp_unit(x: float) -> float:
    """clamp(x, 0, 1): the clamp bounds of a score."""
    return min(1.0, max(0.0, x))


def _flag_seconds(env: SummaryEnv, name: str) -> float:
    """Seconds a class-A flag output is 1: each valid row with value 1 counts until the next
    row, but never longer than the signal's stale_after_ms (a gap adds nothing); the last
    row counts nothing (as hold_segments)."""
    times, values = env.rows(name)
    if not any(v is not None for v in values):
        raise InputInvalid(FLAG_INVALID_INPUT)
    limit = vehicle_derived.SIGNALS[name.removeprefix("derived.")]["stale_after_ms"]
    total = 0.0
    for i in range(len(times) - 1):
        if values[i] == 1.0:
            total += min(times[i + 1] - times[i], limit) / 1000.0
    return total


# --- usage of the class-A flags and events -----------------------------------------------------


@_summary("ride_heat_soak_max")
def ride_heat_soak_max(env: SummaryEnv) -> Outcome:
    """Largest valid derived.heat_soak_delta, degC."""
    return max(_valid(env, "derived.heat_soak_delta")), None


@_summary("ride_cranking_dip")
def ride_cranking_dip(env: SummaryEnv) -> Outcome:
    """The cranking dip of the ride's start, read as the first valid derived.cranking_dip row
    (one session = one ride; a restart later in it is not "the ride's start"), V."""
    return _valid(env, "derived.cranking_dip")[0], None


@_summary("ride_rest_voltage")
def ride_rest_voltage(env: SummaryEnv) -> Outcome:
    """The first valid derived.battery_rest_voltage row (the key-on of the ride), V."""
    return _valid(env, "derived.battery_rest_voltage")[0], None


@_summary("ride_cold_abuse_time")
def ride_cold_abuse_time(env: SummaryEnv) -> Outcome:
    """Seconds with derived.cold_engine_abuse set."""
    return _flag_seconds(env, "derived.cold_engine_abuse"), None


@_summary("ride_over_rev_time")
def ride_over_rev_time(env: SummaryEnv) -> Outcome:
    """Seconds with derived.over_rev set."""
    return _flag_seconds(env, "derived.over_rev"), None


@_summary("ride_lugging_time")
def ride_lugging_time(env: SummaryEnv) -> Outcome:
    """Seconds with derived.lugging set."""
    return _flag_seconds(env, "derived.lugging"), None


@_summary("ride_idle_rpm_std")
def ride_idle_rpm_std(env: SummaryEnv) -> Outcome:
    """Largest valid derived.idle_rpm_std, rpm."""
    return max(_valid(env, "derived.idle_rpm_std")), None


@_summary("ride_idle_rpm_drift")
def ride_idle_rpm_drift(env: SummaryEnv) -> Outcome:
    """ "Drift over the ride" is read as the last valid derived.idle_rpm_drift row (the drift
    against the first warm-idle level at the end of the ride), rpm."""
    return _valid(env, "derived.idle_rpm_drift")[-1], None


@_summary("ride_shift_count")
def ride_shift_count(env: SummaryEnv) -> Outcome:
    """Number of valid derived.shift_event rows."""
    return float(len(env.valid("derived.shift_event"))), None


@_summary("ride_launch_count")
def ride_launch_count(env: SummaryEnv) -> Outcome:
    """Number of valid derived.launch_event rows."""
    return float(len(env.valid("derived.launch_event"))), None


# --- thermal ------------------------------------------------------------------------------------


@_summary("ride_warmup_time")
def ride_warmup_time(env: SummaryEnv) -> Outcome:
    """Seconds from the engine start to the first COOLANT_TEMP >= derived_operating_coolant_degc.

    ENGINE_SPEED is not a declared input; the engine start ("the first ENGINE_SPEED above 0")
    is read as RUN_TIME_SINCE_START = 0, so the value is the RUN_TIME_SINCE_START (fresh) at
    the first warm coolant sample. Invalid (flag starts_warm) when the first coolant sample is
    already warm, (never_warm) when none is."""
    operating = env.const("const.derived_operating_coolant_degc")
    coolant = env.feed(_COOLANT)
    env.feed(_RUN_TIME)  # read at the first warm sample through At below
    samples = [(t, v) for t, v in zip(coolant.times, coolant.values, strict=True) if v is not None]
    if not samples:
        raise InputInvalid(FLAG_INVALID_INPUT)
    if samples[0][1] >= operating:
        raise InputInvalid(FLAG_STARTS_WARM)
    for t, v in samples:
        if v >= operating:
            return At(env, t).value(_RUN_TIME), None
    raise InputInvalid(FLAG_NEVER_WARM)


@_summary("ride_thermal_cycle_count")
def ride_thermal_cycle_count(env: SummaryEnv) -> Outcome:
    """Cold-to-hot cycles of the coolant: a cycle is a COOLANT_TEMP below
    derived_cold_coolant_degc followed by a sample at or above derived_operating_coolant_degc
    (between the two thresholds the state is kept). A ride that starts warm counts no cycle
    until it has been cold."""
    cold, operating = (
        env.const("const.derived_cold_coolant_degc"),
        env.const("const.derived_operating_coolant_degc"),
    )
    was_cold = False
    cycles = 0
    for value in env.samples(_COOLANT):
        if value < cold:
            was_cold = True
        elif value >= operating and was_cold:
            cycles += 1
            was_cold = False
    return float(cycles), None


# --- engine stress ------------------------------------------------------------------------------


@_summary("throttle_torque_lag")
def throttle_torque_lag(env: SummaryEnv) -> Outcome:
    """Median lag (ms) over the throttle steps of the ride.

    A step is a THROTTLE_POS rise of at least throttle_torque_lag_step_min_pct between two
    valid samples at most throttle_torque_lag_step_window_ms apart (the earliest such sample
    starts it; after a step the search restarts after its end). For each step the lag is the
    tau in 0..throttle_torque_lag_search_max_ms (in steps of the signal's defs resolution)
    that maximises the mean-removed cross-correlation of the throttle samples in
    [start, start + search_max] with the ACTUAL_TORQUE_PCT held value (fresh) tau later; a
    tau with fewer than 3 pairs is skipped and a tie keeps the smallest tau. The ride value is
    the median of the lags (the mean of the two middle ones for an even count). ENGINE_SPEED is
    declared, not used. No step: invalid, flag no_step."""
    step_min = env.const("const.throttle_torque_lag_step_min_pct")
    window = env.const("const.throttle_torque_lag_step_window_ms")
    search = env.const("const.throttle_torque_lag_search_max_ms")
    resolution = vehicle_derived.SIGNALS["throttle_torque_lag"]["resolution"]
    throttle = env.feed(_THROTTLE)
    env.feed(_TORQUE_PCT)  # read per lag candidate through At below
    env.feed(_RPM)  # declared by the defs; the rule does not use it
    points = [(t, v) for t, v in zip(throttle.times, throttle.values, strict=True) if v is not None]

    steps: list[int] = []
    recent: list[tuple[int, float]] = []
    previous: int | None = None
    for t, v in points:
        if previous is not None and t - previous > throttle.stale_after_ms:
            recent = []
        previous = t
        recent = [s for s in recent if t - s[0] <= window]
        for start_t, start_v in recent:
            if v - start_v >= step_min:
                steps.append(start_t)
                recent = []
                break
        recent.append((t, v))

    lags: list[float] = []
    for start in steps:
        segment = [(t, v) for t, v in points if start <= t <= start + search]
        best: tuple[float, float] | None = None
        tau = 0.0
        while tau <= search:
            pairs = []
            for t, x in segment:
                at = At(env, int(t + tau))
                try:
                    pairs.append((x, at.value(_TORQUE_PCT)))
                except InputInvalid:
                    continue
            if len(pairs) >= 3:
                mean_x = sum(x for x, _ in pairs) / len(pairs)
                mean_y = sum(y for _, y in pairs) / len(pairs)
                score = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
                if best is None or score > best[0]:
                    best = (score, tau)
            tau += resolution
        if best is not None:
            lags.append(best[1])
    if not lags:
        raise InputInvalid(FLAG_NO_STEP)
    return float(median(lags)), None


# --- calibration at steady speed ----------------------------------------------------------------


def _steady_pairs(env: SummaryEnv) -> list[tuple[float, float]]:
    """(ECU speed, GNSS speed), both km/h, at the GNSS samples inside a steady-speed segment.

    Steady speed = a run of consecutive VEHICLE_SPEED samples (no gap above the speed's
    stale_after_ms) whose spread (max - min) stays within derived_steady_speed_spread_kmh, lasts
    at least derived_steady_min_duration_ms (first to last sample) and whose samples are all above
    derived_calibration_speed_min_kmh. A run is extended greedily; the sample that breaks the
    spread closes it and starts the next one. A GNSS sample counts when its fix type is usable
    (3D or dead reckoning, a 2D fix does not count, D-099), its speed accuracy is at most
    derived_gnss_speed_accuracy_max_mps and the ECU speed is fresh at its time. Speed only."""
    spread = env.const("const.derived_steady_speed_spread_kmh")
    duration = env.const("const.derived_steady_min_duration_ms")
    minimum = env.const("const.derived_calibration_speed_min_kmh")
    accuracy_max = env.const("const.derived_gnss_speed_accuracy_max_mps")
    speed, gps = env.feed(_SPEED), env.feed(_GPS_SPEED)
    env.feed(_GPS_FIX), env.feed(_GPS_ACCURACY)  # read per GNSS sample through At below

    segments: list[tuple[int, int]] = []
    run: list[tuple[int, float]] = []

    def close() -> None:
        if run and run[-1][0] - run[0][0] >= duration and min(v for _, v in run) > minimum:
            segments.append((run[0][0], run[-1][0]))

    previous: int | None = None
    for t, v in zip(speed.times, speed.values, strict=True):
        if v is None:
            continue
        if previous is not None and t - previous > speed.stale_after_ms:
            close()
            run = []
        previous = t
        if run and max(max(x for _, x in run), v) - min(min(x for _, x in run), v) > spread:
            close()
            run = []
        run.append((t, v))
    close()

    usable = usable_fix_values()
    starts = [s for s, _ in segments]
    pairs: list[tuple[float, float]] = []
    for t, v in zip(gps.times, gps.values, strict=True):
        i = bisect_right(starts, t) - 1
        if v is None or i < 0 or t > segments[i][1]:
            continue
        at = At(env, t)
        try:
            good = at.value(_GPS_FIX) in usable and at.value(_GPS_ACCURACY) <= accuracy_max
            ecu = at.value(_SPEED)
        except InputInvalid:
            continue
        if good and v > 0:
            pairs.append((ecu, v * _MPS_TO_KMH))
    return pairs


@_summary("speedometer_error")
def speedometer_error(env: SummaryEnv) -> Outcome:
    """Mean of (ECU speed - GNSS speed) / GNSS speed x 100 over the steady-speed samples with
    a good fix (see _steady_pairs); positive = the ECU speed is above the GNSS speed, %."""
    pairs = _steady_pairs(env)
    if not pairs:
        raise InputInvalid(FLAG_NO_STEADY_SAMPLES)
    return sum((ecu - gnss) / gnss for ecu, gnss in pairs) / len(pairs) * 100.0, None


@_summary("effective_tyre_radius")
def effective_tyre_radius(env: SummaryEnv) -> Outcome:
    """Effective rear tyre radius, m, on the samples of speedometer_error.

    The "engine-side ratio" of the description is not a declared input. The ECU speed is
    taken as the wheel speed times the dynamic radius rear_wheel_radius_dynamic_m, so the
    effective radius is that constant times the mean of GNSS speed / ECU speed."""
    radius = env.const("const.rear_wheel_radius_dynamic_m")
    pairs = _steady_pairs(env)
    if not pairs:
        raise InputInvalid(FLAG_NO_STEADY_SAMPLES)
    return radius * sum(gnss / ecu for ecu, gnss in pairs) / len(pairs), None


# --- rider style --------------------------------------------------------------------------------


@_summary("ride_aggressiveness_score")
def ride_aggressiveness_score(env: SummaryEnv) -> Outcome:
    """100 x the mean over the features with data of clamp(feature / reference, 0, 1), equal
    weights: the p95 throttle opening rate (positive THROTTLE_POS slopes between consecutive
    samples without a gap above the DID's stale_after_ms, %/s), the p95 |vehicle_jerk|, the
    largest braking-event peak (|value| of derived.braking_event) and the p95 RPM before an
    upshift. The last feature has no data here: ENGINE_SPEED is not a declared input and the
    value of shift_event is the step duration, not an RPM. No feature with data: invalid, flag
    no_data. derived.launch_event is declared, not used."""
    rate_ref = env.const("const.ride_aggressiveness_score_throttle_rate_ref_pct_s")
    jerk_ref = env.const("const.ride_aggressiveness_score_jerk_ref_mps3")
    decel_ref = env.const("const.ride_aggressiveness_score_decel_ref_mps2")
    env.const("const.ride_aggressiveness_score_shift_rpm_ref")  # the shift feature has no data
    env.valid("derived.shift_event")  # durations, not RPM: no data for the shift feature
    env.valid("derived.launch_event")  # declared by the defs; the rule does not use it
    scores: list[float] = []
    feed = env.feed(_THROTTLE)
    valid = [(t, v) for t, v in zip(feed.times, feed.values, strict=True) if v is not None]
    rates = [
        (v2 - v1) / ((t2 - t1) / 1000.0)
        for (t1, v1), (t2, v2) in zip(valid, valid[1:], strict=False)
        if 0 < t2 - t1 <= feed.stale_after_ms and v2 > v1
    ]
    if rates:
        scores.append(_clamp_unit(_percentile_95(rates) / rate_ref))
    jerk = [abs(v) for v in env.valid("derived.vehicle_jerk")]
    if jerk:
        scores.append(_clamp_unit(_percentile_95(jerk) / jerk_ref))
    braking = [abs(v) for v in env.valid("derived.braking_event")]
    if braking:
        scores.append(_clamp_unit(max(braking) / decel_ref))
    if not scores:
        raise InputInvalid(FLAG_NO_DATA)
    return 100.0 * sum(scores) / len(scores), None


@_summary("ride_cornering_aggressiveness")
def ride_cornering_aggressiveness(env: SummaryEnv) -> Outcome:
    """100 x the mean of clamp(p95 |lateral accel| / lat ref, 0, 1) and clamp(p95 |lean| /
    lean ref, 0, 1). Both features are required (no data in either: invalid, flag no_data)."""
    lateral = [abs(v) for v in env.valid("derived.lateral_accel")]
    lean = [abs(v) for v in env.samples(_LEAN)]
    if not lateral or not lean:
        raise InputInvalid(FLAG_NO_DATA)
    lat_score = _clamp_unit(
        _percentile_95(lateral)
        / env.const("const.ride_cornering_aggressiveness_lat_accel_ref_mps2")
    )
    lean_score = _clamp_unit(
        _percentile_95(lean) / env.const("const.ride_cornering_aggressiveness_lean_ref_deg")
    )
    return 100.0 * (lat_score + lean_score) / 2.0, None


# --- ECU and bus health -------------------------------------------------------------------------


@_summary("ride_can_error_count")
def ride_can_error_count(env: SummaryEnv) -> Outcome:
    """Sum of the positive increments between consecutive samples of the TEC and the REC (both
    are levels; a lower bound, a saturated 255 adds nothing; a decrease adds nothing)."""
    total = 0.0
    for name in (_TX_ERRORS, _RX_ERRORS):
        samples = env.samples(name)
        total += sum(max(0.0, b - a) for a, b in zip(samples, samples[1:], strict=False))
    return total, None


@_summary("ride_unanswered_did_count")
def ride_unanswered_did_count(env: SummaryEnv) -> Outcome:
    """Last minus first value of the since-boot count of timed-out requests; a decrease is a
    reboot and restarts the baseline from the new value (what was counted before it is kept,
    what the new boot counted up to its first sample is not). Saturates at the defs range
    maximum."""
    samples = env.samples(_UNANSWERED)
    total = sum(b - a for a, b in zip(samples, samples[1:], strict=False) if b >= a)
    return min(total, vehicle_derived.SIGNALS["ride_unanswered_did_count"]["range"][1]), None
