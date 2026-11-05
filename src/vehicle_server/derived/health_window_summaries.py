"""D-079 health-window class-B summaries (reference only).

Each summary reads the first qualifying instance of its window through SummaryEnv.window and
applies the rule in D-079 section 2.3. Thresholds and durations live in the window definition;
the values below are only the arithmetic over the samples inside that instance. The state
machine itself is in health_windows.py.
"""

from __future__ import annotations

import math

from vehicle_server.derived.context import FLAG_INSUFFICIENT_SAMPLES, InputInvalid
from vehicle_server.derived.d103_summaries import FLAG_NO_DATA
from vehicle_server.derived.formulas import Outcome
from vehicle_server.derived.health_windows import (
    CLOSE_MAX_DURATION,
    CLOSE_SESSION_END,
    WindowInstance,
)
from vehicle_server.derived.summary import SummaryEnv, _summary

FLAG_NO_WINDOW = "no_window"

_COOLANT = "did.COOLANT_TEMP"
_AMBIENT = "did.AMBIENT_AIR_TEMP"
_INTAKE = "did.INTAKE_AIR_TEMP"
_MAP = "did.INTAKE_MAP"
_BARO = "did.BAROMETRIC_PRESSURE"
_BATTERY = "did.BATTERY_VOLTAGE"
_RPM = "did.ENGINE_SPEED"
_STFT = "did.SHORT_TERM_FUEL_TRIM_B1"

_COLD_SOAK = "const.window_cold_soak_max_delta_degc"

NOT_IMPLEMENTED_WINDOW_REASONS = {
    "window_cruise_final_drive_residual": (
        "definition: declared inputs have no const.gear_ratio_<n>, so the measured final drive "
        "cannot be separated from the gearbox ratio (compare ride_final_drive_ratio)"
    ),
    "window_cruise_ve_residual": (
        "definition: defs has no expected-VE model at that gear and speed"
    ),
}


def _instance(env: SummaryEnv, window: str) -> WindowInstance:
    instance = env.window(window).first_qualifying
    if instance is None:
        raise InputInvalid(FLAG_NO_WINDOW)
    return instance


def _inside(instance: WindowInstance, t: int) -> bool:
    """A sample is inside [open, close); a close by max duration or session end keeps its
    boundary sample, a condition/stable/transition close excludes the sample that closed it."""
    if t < instance.open_ms or t > instance.close_ms:
        return False
    if t < instance.close_ms:
        return True
    return instance.close_reason in (CLOSE_MAX_DURATION, CLOSE_SESSION_END)


def _points(env: SummaryEnv, name: str, instance: WindowInstance) -> list[tuple[int, float]]:
    """Valid samples of a declared did.* input inside the instance, time order."""
    feed = env.feed(name)
    return [
        (t, v)
        for t, v in zip(feed.times, feed.values, strict=True)
        if _inside(instance, t) and v is not None
    ]


def _mean(values: list[float]) -> float:
    if not values:
        raise InputInvalid(FLAG_NO_DATA)
    return sum(values) / len(values)


# --- W1: key on, engine off --------------------------------------------------------------------


@_summary("window_key_on_rest_voltage")
def window_key_on_rest_voltage(env: SummaryEnv) -> Outcome:
    """Mean battery voltage in the first qualifying key-on window, V."""
    instance = _instance(env, "W1_KEY_ON")
    return _mean([v for _, v in _points(env, _BATTERY, instance)]), None


@_summary("window_key_on_sensor_spread")
def window_key_on_sensor_spread(env: SummaryEnv) -> Outcome:
    """Largest minus smallest mean of coolant, intake-air and ambient temperature, degC."""
    instance = _instance(env, "W1_KEY_ON")
    means = [
        _mean([v for _, v in _points(env, name, instance)])
        for name in (_COOLANT, _INTAKE, _AMBIENT)
    ]
    return max(means) - min(means), None


@_summary("window_key_on_map_baro_delta")
def window_key_on_map_baro_delta(env: SummaryEnv) -> Outcome:
    """Mean intake MAP minus mean barometric pressure in the key-on window, kPa."""
    instance = _instance(env, "W1_KEY_ON")
    map_mean = _mean([v for _, v in _points(env, _MAP, instance)])
    baro_mean = _mean([v for _, v in _points(env, _BARO, instance)])
    return map_mean - baro_mean, None


@_summary("window_key_on_warm_restart")
def window_key_on_warm_restart(env: SummaryEnv) -> Outcome:
    """1 when the key-on coolant differs from ambient by more than the defs cold-soak delta."""
    instance = _instance(env, "W1_KEY_ON")
    coolant = _mean([v for _, v in _points(env, _COOLANT, instance)])
    ambient = _mean([v for _, v in _points(env, _AMBIENT, instance)])
    return (1.0 if abs(coolant - ambient) > env.const(_COLD_SOAK) else 0.0), None


@_summary("window_key_on_count")
def window_key_on_count(env: SummaryEnv) -> Outcome:
    """Number of qualifying key-on windows in the ride (0 when none)."""
    return float(env.window("W1_KEY_ON").qualifying_count), None


# --- W2: crank ---------------------------------------------------------------------------------


@_summary("window_crank_count")
def window_crank_count(env: SummaryEnv) -> Outcome:
    """Number of qualifying crank windows in the ride (0 when none)."""
    return float(env.window("W2_CRANK").qualifying_count), None


# --- W3: warm idle -----------------------------------------------------------------------------


@_summary("window_warm_idle_rpm_spread")
def window_warm_idle_rpm_spread(env: SummaryEnv) -> Outcome:
    """Largest minus smallest engine speed in the first qualifying warm-idle window, rpm."""
    instance = _instance(env, "W3_WARM_IDLE")
    values = [v for _, v in _points(env, _RPM, instance)]
    if not values:
        raise InputInvalid(FLAG_NO_DATA)
    return max(values) - min(values), None


@_summary("window_warm_idle_rpm_drift")
def window_warm_idle_rpm_drift(env: SummaryEnv) -> Outcome:
    """Mean engine speed of the second half of the instance minus the first half, rpm."""
    instance = _instance(env, "W3_WARM_IDLE")
    middle = instance.open_ms + (instance.close_ms - instance.open_ms) / 2.0
    points = _points(env, _RPM, instance)
    first = _mean([v for t, v in points if t < middle])
    second = _mean([v for t, v in points if t >= middle])
    return second - first, None


@_summary("window_warm_idle_map_spread")
def window_warm_idle_map_spread(env: SummaryEnv) -> Outcome:
    """Largest minus smallest intake MAP in the first qualifying warm-idle window, kPa."""
    instance = _instance(env, "W3_WARM_IDLE")
    values = [v for _, v in _points(env, _MAP, instance)]
    if not values:
        raise InputInvalid(FLAG_NO_DATA)
    return max(values) - min(values), None


@_summary("window_warm_idle_stft_mean")
def window_warm_idle_stft_mean(env: SummaryEnv) -> Outcome:
    """Mean short-term fuel trim in the first qualifying warm-idle window, %."""
    instance = _instance(env, "W3_WARM_IDLE")
    return _mean([v for _, v in _points(env, _STFT, instance)]), None


@_summary("window_warm_idle_stft_std")
def window_warm_idle_stft_std(env: SummaryEnv) -> Outcome:
    """Sample standard deviation (n - 1) of the short-term fuel trim in the window, %."""
    instance = _instance(env, "W3_WARM_IDLE")
    values = [v for _, v in _points(env, _STFT, instance)]
    if len(values) < 2:
        raise InputInvalid(FLAG_INSUFFICIENT_SAMPLES)
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)), None


@_summary("window_warm_idle_count")
def window_warm_idle_count(env: SummaryEnv) -> Outcome:
    """Number of qualifying warm-idle windows in the ride (0 when none)."""
    return float(env.window("W3_WARM_IDLE").qualifying_count), None


# --- W4: cruise --------------------------------------------------------------------------------


@_summary("window_cruise_count")
def window_cruise_count(env: SummaryEnv) -> Outcome:
    """Number of qualifying cruise windows in the ride (0 when none)."""
    return float(env.window("W4_CRUISE").qualifying_count), None


# --- W5: post stop -----------------------------------------------------------------------------


@_summary("window_post_stop_coolant_start")
def window_post_stop_coolant_start(env: SummaryEnv) -> Outcome:
    """First valid coolant temperature in the first qualifying post-stop window, degC."""
    instance = _instance(env, "W5_POST_STOP")
    points = _points(env, _COOLANT, instance)
    if not points:
        raise InputInvalid(FLAG_NO_DATA)
    return points[0][1], None


@_summary("window_post_stop_coolant_end")
def window_post_stop_coolant_end(env: SummaryEnv) -> Outcome:
    """Last valid coolant temperature in the first qualifying post-stop window, degC."""
    instance = _instance(env, "W5_POST_STOP")
    points = _points(env, _COOLANT, instance)
    if not points:
        raise InputInvalid(FLAG_NO_DATA)
    return points[-1][1], None


@_summary("window_post_stop_coolant_tau")
def window_post_stop_coolant_tau(env: SummaryEnv) -> Outcome:
    """Newton-cooling time constant of the coolant in the first qualifying post-stop window, s.

    Least-squares slope of ln(T - T_amb) over the elapsed seconds, tau = -1 / slope; fewer than
    three positive differences, a degenerate time axis or a slope >= 0 is invalid.
    """
    instance = _instance(env, "W5_POST_STOP")
    coolant = _points(env, _COOLANT, instance)
    ambient = _mean([v for _, v in _points(env, _AMBIENT, instance)])
    fit = [
        ((t - instance.open_ms) / 1000.0, math.log(v - ambient))
        for t, v in coolant
        if v - ambient > 0.0
    ]
    if len(fit) < 3:
        raise InputInvalid(FLAG_INSUFFICIENT_SAMPLES)
    mean_x = sum(x for x, _ in fit) / len(fit)
    mean_y = sum(y for _, y in fit) / len(fit)
    variance = sum((x - mean_x) ** 2 for x, _ in fit)
    if variance == 0.0:
        raise InputInvalid(FLAG_INSUFFICIENT_SAMPLES)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in fit) / variance
    if slope >= 0.0:
        raise InputInvalid(FLAG_NO_DATA)
    return -1.0 / slope, None


@_summary("window_post_stop_count")
def window_post_stop_count(env: SummaryEnv) -> Outcome:
    """Number of qualifying post-stop windows in the ride (0 when none)."""
    return float(env.window("W5_POST_STOP").qualifying_count), None
