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
from vehicle_server.derived.context import FLAG_INVALID_INPUT, Feed, InputInvalid
from vehicle_server.derived.formulas import Outcome
from vehicle_server.derived.summary import PerGear, SummaryEnv, _summary
from vehicle_server.gps_fix import usable_fix_values

FLAG_NO_DATA = "no_data"
FLAG_NO_STEP = "no_step"
FLAG_STARTS_WARM = "starts_warm"
FLAG_NEVER_WARM = "never_warm"
FLAG_NO_STEADY_SAMPLES = "no_steady_samples"
FLAG_NO_FAN_STEP = "no_fan_step"

_COOLANT = "did.COOLANT_TEMP"
_RPM = "did.ENGINE_SPEED"
_BATTERY = "did.BATTERY_VOLTAGE"
_GEAR = "derived.gear_estimate"
_RATIO = "derived.gearbox_ratio_measured"
_THROTTLE = "did.THROTTLE_POS"
_TORQUE_PCT = "did.ACTUAL_TORQUE_PCT"
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
_SECONDS_PER_MINUTE = 60.0


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


# --- helpers for feeds that are not in the At accessor ------------------------------------------

# A lag candidate needs at least this many (throttle, torque) pairs to be correlated.
MIN_CORRELATION_PAIRS = 3


def _fresh(feed: Feed, t: int) -> float | None:
    """The latest sample at or before t if it is within the feed's stale_after_ms, else None."""
    i = feed.latest(t)
    if i < 0 or t - feed.times[i] > feed.stale_after_ms or feed.values[i] is None:
        return None
    return feed.values[i]


class _Derived:
    """Lookup of a declared class-A input by time: the latest valid row at or before t within
    that signal's own stale_after_ms (None when absent or stale)."""

    def __init__(self, env: SummaryEnv, name: str):
        self.times, self.values = env.rows(name)
        self.stale_after_ms = vehicle_derived.SIGNALS[name.removeprefix("derived.")][
            "stale_after_ms"
        ]

    def at(self, t: int) -> float | None:
        i = bisect_right(self.times, t) - 1
        if i < 0 or t - self.times[i] > self.stale_after_ms:
            return None
        return self.values[i]


# --- thermal ------------------------------------------------------------------------------------


@_summary("ride_warmup_time")
def ride_warmup_time(env: SummaryEnv) -> Outcome:
    """Seconds from the first ENGINE_SPEED sample above 0 to the first COOLANT_TEMP sample >=
    derived_operating_coolant_degc (D-103 rule). Invalid, flag starts_warm, when the first
    coolant sample is already warm (or precedes the engine start); never_warm when none is.
    RUN_TIME_SINCE_START is declared by the defs but the rule does not use it."""
    operating = env.const("const.derived_operating_coolant_degc")
    coolant, rpm = env.feed(_COOLANT), env.feed(_RPM)
    samples = [(t, v) for t, v in zip(coolant.times, coolant.values, strict=True) if v is not None]
    start = next((t for t, v in zip(rpm.times, rpm.values, strict=True) if v and v > 0), None)
    if not samples or start is None:
        raise InputInvalid(FLAG_INVALID_INPUT)
    if samples[0][1] >= operating:
        raise InputInvalid(FLAG_STARTS_WARM)
    for t, v in samples:
        if v >= operating:
            if t < start:
                raise InputInvalid(FLAG_STARTS_WARM)
            return (t - start) / 1000.0, None
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


@_summary("ride_fan_on_temperature")
def ride_fan_on_temperature(env: SummaryEnv) -> Outcome:
    """The COOLANT_TEMP at the first BATTERY_VOLTAGE drop of at least
    ride_fan_on_temperature_voltage_step_v between two consecutive samples (no gap above the
    battery DID's stale_after_ms) while at rest (VEHICLE_SPEED <= derived_moving_speed_min_kmh),
    ENGINE_SPEED in [derived_idle_rpm_min, derived_idle_rpm_max] and COOLANT_TEMP above
    ride_fan_on_temperature_coolant_min_degc (D-103 rule). The conditions are checked at both
    samples of the pair; the coolant is the one fresh at the second. None: invalid, flag
    no_fan_step."""
    step = env.const("const.ride_fan_on_temperature_voltage_step_v")
    coolant_min = env.const("const.ride_fan_on_temperature_coolant_min_degc")
    moving = env.const("const.derived_moving_speed_min_kmh")
    idle_low, idle_high = (
        env.const("const.derived_idle_rpm_min"),
        env.const("const.derived_idle_rpm_max"),
    )
    battery, coolant = env.feed(_BATTERY), env.feed(_COOLANT)
    speed, rpm = env.feed(_SPEED), env.feed(_RPM)

    def resting_idle(t: int) -> float | None:
        """The fresh coolant at t if at rest, in the idle band and above the minimum."""
        v, n, c = _fresh(speed, t), _fresh(rpm, t), _fresh(coolant, t)
        if v is None or n is None or c is None:
            return None
        return c if v <= moving and idle_low <= n <= idle_high and c > coolant_min else None

    samples = [(t, v) for t, v in zip(battery.times, battery.values, strict=True) if v is not None]
    for (t1, v1), (t2, v2) in zip(samples, samples[1:], strict=False):
        if t2 - t1 <= battery.stale_after_ms and v1 - v2 >= step:
            if resting_idle(t1) is not None and (c := resting_idle(t2)) is not None:
                return c, None
    raise InputInvalid(FLAG_NO_FAN_STEP)


# --- engine stress ------------------------------------------------------------------------------


@_summary("throttle_torque_lag")
def throttle_torque_lag(env: SummaryEnv) -> Outcome:
    """Median lag (ms) over the throttle steps of the ride.

    A step is a THROTTLE_POS rise of at least throttle_torque_lag_step_min_pct between two
    valid samples at most throttle_torque_lag_step_window_ms apart (the earliest such sample
    starts it; after a step the search restarts after its end). For each step the lag is the
    tau in 0..throttle_torque_lag_search_max_ms that maximises the mean-removed
    cross-correlation of the throttle samples in [start, start + search_max] with the
    ACTUAL_TORQUE_PCT held value (fresh) tau later; a tau with fewer than
    MIN_CORRELATION_PAIRS pairs is skipped and a tie keeps the smallest tau. The ride value is
    the median of the lags (the mean of the two middle ones for an even count). No step:
    invalid, flag no_step.

    The tau grid uses the signal's defs resolution (10 ms), not the torque DID's poll period:
    the poll period (2 s on the CL250) is longer than the whole search range, so a grid at that
    period would only ever test tau = 0. With a held torque value the lag is therefore only as
    fine as the torque data allows; a real ride at the CL250 poll rate cannot resolve a lag
    below the poll period."""
    step_min = env.const("const.throttle_torque_lag_step_min_pct")
    window = env.const("const.throttle_torque_lag_step_window_ms")
    search = env.const("const.throttle_torque_lag_search_max_ms")
    resolution = vehicle_derived.SIGNALS["throttle_torque_lag"]["resolution"]
    throttle, torque = env.feed(_THROTTLE), env.feed(_TORQUE_PCT)
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
            pairs = [(x, y) for t, x in segment if (y := _fresh(torque, int(t + tau))) is not None]
            if len(pairs) >= MIN_CORRELATION_PAIRS:
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


# --- steady speed: final drive, speedometer, tyre radius ----------------------------------------


def _steady_runs(env: SummaryEnv, gear_of=None) -> list[tuple[list[tuple[int, float]], int]]:
    """Steady-speed runs of consecutive VEHICLE_SPEED samples: the spread (max - min) stays
    within derived_steady_speed_spread_kmh, the run lasts at least
    derived_steady_min_duration_ms (first to last sample), every sample is above
    derived_calibration_speed_min_kmh and there is no gap above the speed's stale_after_ms.
    Runs are closed greedily: a run is extended sample by sample, the sample that breaks the
    spread (or the gear, with gear_of) closes it and starts the next run, so a steady stretch
    can be split by a single outlier. With gear_of(t) -> gear | None a run also stays in ONE
    valid gear (a sample without a gear closes it). Returns ([(t, speed), ...], gear) per run;
    the gear is 0 without gear_of."""
    spread = env.const("const.derived_steady_speed_spread_kmh")
    duration = env.const("const.derived_steady_min_duration_ms")
    minimum = env.const("const.derived_calibration_speed_min_kmh")
    speed = env.feed(_SPEED)
    runs: list[tuple[list[tuple[int, float]], int]] = []
    run: list[tuple[int, float]] = []
    run_gear = 0

    def close() -> None:
        if run and run[-1][0] - run[0][0] >= duration and min(v for _, v in run) > minimum:
            runs.append((list(run), run_gear))

    previous: int | None = None
    for t, v in zip(speed.times, speed.values, strict=True):
        if v is None:
            continue
        if previous is not None and t - previous > speed.stale_after_ms:
            close()
            run = []
        previous = t
        gear = gear_of(t) if gear_of else 0
        if gear is None:
            close()
            run = []
            continue
        if run and (
            gear != run_gear
            or max(max(x for _, x in run), v) - min(min(x for _, x in run), v) > spread
        ):
            close()
            run = []
        if not run:
            run_gear = gear
        run.append((t, v))
    close()
    return runs


def _steady_gnss_samples(env: SummaryEnv, gear_of=None) -> list[tuple[int, float, float, int]]:
    """(time, ECU speed km/h, GNSS speed m/s, gear) at the GNSS samples inside a steady run
    (see _steady_runs) with a good fix: fix type usable (3D or dead reckoning, a 2D fix does
    not count, D-099), speed accuracy at most derived_gnss_speed_accuracy_max_mps, the ECU speed
    fresh at its time and a GNSS speed above 0. Speed only, never a position."""
    accuracy_max = env.const("const.derived_gnss_speed_accuracy_max_mps")
    speed, gps = env.feed(_SPEED), env.feed(_GPS_SPEED)
    fix, accuracy = env.feed(_GPS_FIX), env.feed(_GPS_ACCURACY)
    runs = _steady_runs(env, gear_of)
    starts = [run[0][0] for run, _ in runs]
    usable = usable_fix_values()
    samples: list[tuple[int, float, float, int]] = []
    for t, v in zip(gps.times, gps.values, strict=True):
        i = bisect_right(starts, t) - 1
        if v is None or i < 0 or t > runs[i][0][-1][0]:
            continue
        kind, sigma, ecu = _fresh(fix, t), _fresh(accuracy, t), _fresh(speed, t)
        if kind in usable and sigma is not None and sigma <= accuracy_max and ecu and v > 0:
            samples.append((t, ecu, v, runs[i][1]))
    return samples


@_summary("speedometer_error")
def speedometer_error(env: SummaryEnv) -> Outcome:
    """Mean of (ECU speed - GNSS speed) / GNSS speed x 100 over the steady-speed samples with
    a good fix (D-103 rule: no gear condition; steady runs are closed greedily, see
    _steady_runs); positive = the ECU speed is above the GNSS speed, %."""
    samples = _steady_gnss_samples(env)
    if not samples:
        raise InputInvalid(FLAG_NO_STEADY_SAMPLES)
    return sum((e - g * _MPS_TO_KMH) / (g * _MPS_TO_KMH) for _, e, g, _ in samples) / len(
        samples
    ) * 100.0, None


@_summary("effective_tyre_radius")
def effective_tyre_radius(env: SummaryEnv) -> Outcome:
    """Median, over the samples of speedometer_error that also have a valid gear_estimate n
    (>= 1), of GNSS speed / (2 pi x ENGINE_SPEED / 60 / (primary_reduction_ratio x gear_ratio_n x
    secondary_reduction_ratio)), m (D-103 rule). A sample with ENGINE_SPEED not above 0 is
    left out."""
    gear = _Derived(env, _GEAR)
    primary = env.const("const.primary_reduction_ratio")
    secondary = env.const("const.secondary_reduction_ratio")
    gear_ratios = _gear_ratios(env)
    rpm = env.feed(_RPM)

    def valid_gear(t: int) -> int | None:
        g = gear.at(t)
        return int(g) if g is not None and g >= 1 else None

    radii = []
    for t, _, gnss, n in _steady_gnss_samples(env, valid_gear):
        engine = _fresh(rpm, t)
        if engine and engine > 0:
            wheel_rad_s = (
                math.tau * engine / _SECONDS_PER_MINUTE / (primary * gear_ratios[n] * secondary)
            )
            radii.append(gnss / wheel_rad_s)
    if not radii:
        raise InputInvalid(FLAG_NO_STEADY_SAMPLES)
    return float(median(radii)), None


def _gear_ratios(env: SummaryEnv) -> dict[int, float]:
    """{gear: ratio} from the declared const.gear_ratio_N inputs."""
    prefix = "const.gear_ratio_"
    return {
        int(name.removeprefix(prefix)): env.const(name)
        for name in env.inputs
        if name.startswith(prefix)
    }


@_summary("ride_final_drive_ratio")
def ride_final_drive_ratio(env: SummaryEnv) -> PerGear:
    """Per gear n, the median over the steady-cruise samples in that gear of
    gearbox_ratio_measured x secondary_reduction_ratio / gear_ratio_n (D-103 rule). Steady
    cruise = a steady run (see _steady_runs) in one valid gear_estimate; the samples are the
    VEHICLE_SPEED samples of the run that have a measured ratio. A gear without samples is
    invalid, flag no_steady_samples. THROTTLE_POS and primary_reduction_ratio are declared by
    the defs but the rule does not use them."""
    gear, ratio = _Derived(env, _GEAR), _Derived(env, _RATIO)
    secondary = env.const("const.secondary_reduction_ratio")
    gear_ratios = _gear_ratios(env)

    def valid_gear(t: int) -> int | None:
        g = gear.at(t)
        return int(g) if g is not None and g >= 1 else None

    per_gear: dict[int, list[float]] = {n: [] for n in gear_ratios}
    for run, n in _steady_runs(env, valid_gear):
        for t, _ in run:
            measured = ratio.at(t)
            if measured is not None and n in per_gear:
                per_gear[n].append(measured * secondary / gear_ratios[n])
    return PerGear(
        [
            (float(median(per_gear[n])), None) if per_gear[n] else (None, FLAG_NO_STEADY_SAMPLES)
            for n in sorted(gear_ratios)
        ]
    )


# --- rider style --------------------------------------------------------------------------------


@_summary("ride_aggressiveness_score")
def ride_aggressiveness_score(env: SummaryEnv) -> Outcome:
    """100 x the mean over the features with data of clamp(feature / reference, 0, 1), equal
    weights (D-103 rule): the p95 throttle opening rate (positive THROTTLE_POS slopes between
    consecutive samples without a gap above the DID's stale_after_ms, %/s), the p95
    |vehicle_jerk|, the largest braking-event peak (|value| of derived.braking_event) and the
    p95 ENGINE_SPEED at the start of the upshifts: the shift_event rows whose ENGINE_SPEED
    falls (fresh ENGINE_SPEED at the start, row time minus the event duration, and at the row
    time). No feature with data: invalid, flag no_data."""
    rate_ref = env.const("const.ride_aggressiveness_score_throttle_rate_ref_pct_s")
    jerk_ref = env.const("const.ride_aggressiveness_score_jerk_ref_mps3")
    decel_ref = env.const("const.ride_aggressiveness_score_decel_ref_mps2")
    shift_ref = env.const("const.ride_aggressiveness_score_shift_rpm_ref")
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
    rpm = env.feed(_RPM)
    shift_times, shift_durations = env.rows("derived.shift_event")
    upshift_rpm = []
    for t, duration in zip(shift_times, shift_durations, strict=True):
        if duration is None:
            continue
        before, after = _fresh(rpm, int(t - duration)), _fresh(rpm, t)
        if before is not None and after is not None and after < before:
            upshift_rpm.append(before)
    if upshift_rpm:
        scores.append(_clamp_unit(_percentile_95(upshift_rpm) / shift_ref))
    if not scores:
        raise InputInvalid(FLAG_NO_DATA)
    return 100.0 * sum(scores) / len(scores), None


@_summary("ride_cornering_aggressiveness")
def ride_cornering_aggressiveness(env: SummaryEnv) -> Outcome:
    """100 x the mean of clamp(p95 |lateral accel| / lat ref, 0, 1) and clamp(p95 |lean| /
    lean ref, 0, 1). Both features are required (no data in either: invalid, flag no_data)."""
    lateral = [abs(v) for v in env.valid("derived.lateral_accel")]
    lean = [abs(v) for v in env.samples(_LEAN)]
    lat_ref = env.const("const.ride_cornering_aggressiveness_lat_accel_ref_mps2")
    lean_ref = env.const("const.ride_cornering_aggressiveness_lean_ref_deg")
    if not lateral or not lean:
        raise InputInvalid(FLAG_NO_DATA)
    lat_score = _clamp_unit(_percentile_95(lateral) / lat_ref)
    lean_score = _clamp_unit(_percentile_95(lean) / lean_ref)
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
