"""D-103 class-A windows and residuals (LR15-2): warm-idle statistics, charging plateau,
filtered fuel level, road roughness, heat soak, idle MAP residual. Thresholds come from the
declared const.* inputs."""

from __future__ import annotations

import math
from bisect import bisect_right

from vehicle_server.derived.context import FLAG_INSUFFICIENT_SAMPLES, At, Env, InputInvalid
from vehicle_server.derived.d103_common import (
    _ACCEL,
    _ACCEL_Z,
    _AMBIENT,
    _BATT,
    _COOLANT,
    _FUEL_LEVEL,
    _IAT,
    _IDLE_MAX,
    _IDLE_MIN,
    _MAP,
    _MOVING_MIN,
    _OPERATING,
    _RPM,
    _SPEED,
    _THROTTLE,
    _THROTTLE_CLOSED,
    FLAG_NOT_APPLICABLE,
    _require_moving,
)
from vehicle_server.derived.formulas import Outcome, _point, _series

# --- LR15-2: windows and residuals -------------------------------------------------------------


def _warm_idle_points(env: Env) -> list[tuple[int, float]]:
    """ENGINE_SPEED samples at warm idle: COOLANT_TEMP >= derived_operating_coolant_degc, at
    rest (VEHICLE_SPEED not above derived_moving_speed_min_kmh) and the RPM inside
    [derived_idle_rpm_min, derived_idle_rpm_max]. The coolant and the speed are those fresh at
    the sample time; a sample without a fresh one is left out."""
    rpm = env.feed(_RPM)
    operating, moving = env.const(_OPERATING), env.const(_MOVING_MIN)
    low, high = env.const(_IDLE_MIN), env.const(_IDLE_MAX)
    points: list[tuple[int, float]] = []
    for t, value in zip(rpm.times, rpm.values, strict=True):
        if value is None or not low <= value <= high:
            continue
        at = At(env, t)
        try:
            warm = at.value(_COOLANT) >= operating
            at_rest = at.value(_SPEED) <= moving
        except InputInvalid:
            continue
        if warm and at_rest:
            points.append((t, value))
    return points


def _window_slice(times: list[int], t: int, window_ms: int) -> tuple[int, int]:
    return bisect_right(times, t - window_ms), bisect_right(times, t)


@_series("idle_rpm_std")
def idle_rpm_std(env: Env, grid: list[int]) -> list[Outcome]:
    """Population standard deviation of the warm-idle RPM samples in (t - window_ms, t];
    fewer than two such samples is insufficient_samples. The newest ENGINE_SPEED sample must
    be fresh."""
    points = _warm_idle_points(env)
    times = [t for t, _ in points]
    out: list[Outcome] = []
    for t in grid:
        try:
            At(env, t).value(_RPM)
        except InputInvalid as exc:
            out.append((None, exc.flag))
            continue
        lo, hi = _window_slice(times, t, env.window_ms or 0)
        values = [v for _, v in points[lo:hi]]
        if len(values) < 2:
            out.append((None, FLAG_INSUFFICIENT_SAMPLES))
            continue
        mean = sum(values) / len(values)
        out.append((math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)), None))
    return out


@_series("idle_rpm_drift")
def idle_rpm_drift(env: Env, grid: list[int]) -> list[Outcome]:
    """Mean warm-idle RPM of (t - window_ms, t] minus the earlier level.

    The earlier level is read as the mean of the warm-idle samples in the first full window
    of the ride: [t0, t0 + window_ms) with t0 the first warm-idle sample (warm idle as in
    idle_rpm_std; the samples need not be contiguous). The output is insufficient_samples
    until t0 + window_ms has been reached and while the current window holds no warm-idle
    sample."""
    window = env.window_ms or 0
    points = _warm_idle_points(env)
    times = [t for t, _ in points]
    baseline: float | None = None
    baseline_end: int | None = None
    if points:
        first_end = points[0][0] + window
        first = [v for t, v in points if t < first_end]
        baseline, baseline_end = sum(first) / len(first), first_end
    out: list[Outcome] = []
    for t in grid:
        try:
            At(env, t).value(_RPM)
        except InputInvalid as exc:
            out.append((None, exc.flag))
            continue
        lo, hi = _window_slice(times, t, window)
        values = [v for _, v in points[lo:hi]]
        if baseline is None or baseline_end is None or t < baseline_end or not values:
            out.append((None, FLAG_INSUFFICIENT_SAMPLES))
            continue
        out.append((sum(values) / len(values) - baseline, None))
    return out


@_point("charging_plateau_voltage")
def charging_plateau_voltage(at: At) -> float:
    """Mean of the BATTERY_VOLTAGE samples of the window whose ENGINE_SPEED (fresh at the
    sample time) is above charging_plateau_rpm_min; none is insufficient_samples."""
    minimum = at.const("const.charging_plateau_rpm_min")
    values: list[float] = []
    for t, voltage in at.window(_BATT):
        try:
            rpm = At(at.env, t).value(_RPM)
        except InputInvalid:
            continue
        if rpm > minimum:
            values.append(voltage)
    if not values:
        raise InputInvalid(FLAG_INSUFFICIENT_SAMPLES)
    return sum(values) / len(values)


@_point("fuel_level_filtered")
def fuel_level_filtered(at: At) -> float:
    """Mean FUEL_LEVEL of the window without the samples where |vehicle_accel| (at the sample
    time) is above fuel_level_filtered_accel_max_mps2. A sample whose acceleration is not
    available (stale or invalid) cannot be shown to be calm and is left out too. VEHICLE_SPEED
    is declared, not used."""
    limit = at.const("const.fuel_level_filtered_accel_max_mps2")
    values: list[float] = []
    for t, level in at.window(_FUEL_LEVEL):
        try:
            acceleration = At(at.env, t).value(_ACCEL)
        except InputInvalid:
            continue
        if abs(acceleration) <= limit:
            values.append(level)
    if not values:
        raise InputInvalid(FLAG_INSUFFICIENT_SAMPLES)
    return sum(values) / len(values)


def _bilinear_k(corner_hz: float, dt_s: float) -> float:
    """tan(pi * corner * dt): the pre-warped bilinear-transform constant of a first-order
    section, so its -3 dB corner stays exact at the actual sample spacing. A corner at or
    above the Nyquist frequency is held just below it (the section is then nearly flat)."""
    angle = math.pi * corner_hz * dt_s
    return math.tan(min(angle, math.pi / 2 - 1e-3))


@_series("road_roughness")
def road_roughness(env: Env, grid: list[int]) -> list[Outcome]:
    """RMS of the band-passed vertical acceleration over window_ms, only while moving.

    Filter: a first-order high-pass at road_roughness_band_low_hz followed by a first-order
    low-pass at road_roughness_band_high_hz, each a bilinear (Tustin) section with the corner
    pre-warped at the actual sample spacing (pure Python, no new dependency). The filter state
    restarts after a gap longer than the input's stale_after_ms. It is a gentle band-pass (the
    band edges are the -3 dB corners of the two sections), enough for an index. The RMS is that
    of the filtered samples in (t - window_ms, t]; the speed must exceed
    derived_moving_speed_min_kmh at t (else null, not_moving)."""
    feed = env.feed(_ACCEL_Z)
    low_hz = env.const("const.road_roughness_band_low_hz")
    high_hz = env.const("const.road_roughness_band_high_hz")
    times: list[int] = []
    squares = [0.0]
    previous_t: int | None = None
    previous_x = previous_high = high_pass = low_pass = 0.0
    for t, x in zip(feed.times, feed.values, strict=True):
        if x is None:
            previous_t = None
            continue
        if previous_t is None or t - previous_t > feed.stale_after_ms:
            high_pass = low_pass = 0.0
        else:
            dt = (t - previous_t) / 1000.0
            k_high, k_low = _bilinear_k(low_hz, dt), _bilinear_k(high_hz, dt)
            high_pass = ((x - previous_x) + (1.0 - k_high) * high_pass) / (1.0 + k_high)
            low_pass = (k_low * (high_pass + previous_high) + (1.0 - k_low) * low_pass) / (
                1.0 + k_low
            )
        previous_t, previous_x, previous_high = t, x, high_pass
        times.append(t)
        squares.append(squares[-1] + low_pass * low_pass)
    out: list[Outcome] = []
    for t in grid:
        try:
            at = At(env, t)
            _require_moving(at)
            at.value(_ACCEL_Z)
        except InputInvalid as exc:
            out.append((None, exc.flag))
            continue
        lo, hi = _window_slice(times, t, env.window_ms or 0)
        if hi <= lo:
            out.append((None, FLAG_INSUFFICIENT_SAMPLES))
            continue
        out.append((math.sqrt((squares[hi] - squares[lo]) / (hi - lo)), None))
    return out


@_point("heat_soak_delta")
def heat_soak_delta(at: At) -> float:
    """INTAKE_AIR_TEMP minus AMBIENT_AIR_TEMP, only while VEHICLE_SPEED <=
    heat_soak_delta_speed_max_kmh (else null, flag not_applicable)."""
    speed = at.value(_SPEED)
    intake = at.value(_IAT)
    ambient = at.value(_AMBIENT)
    if speed > at.const("const.heat_soak_delta_speed_max_kmh"):
        raise InputInvalid(FLAG_NOT_APPLICABLE)
    return intake - ambient


@_point("idle_map_residual")
def idle_map_residual(at: At) -> float:
    """INTAKE_MAP minus idle_map_residual_expected_kpa while THROTTLE_POS <= closed max and
    ENGINE_SPEED is in the idle band [derived_idle_rpm_min, derived_idle_rpm_max] (else null,
    flag not_applicable)."""
    pressure = at.value(_MAP)
    throttle = at.value(_THROTTLE)
    rpm = at.value(_RPM)
    if throttle > at.const(_THROTTLE_CLOSED) or not at.const(_IDLE_MIN) <= rpm <= at.const(
        _IDLE_MAX
    ):
        raise InputInvalid(FLAG_NOT_APPLICABLE)
    return pressure - at.const("const.idle_map_residual_expected_kpa")
