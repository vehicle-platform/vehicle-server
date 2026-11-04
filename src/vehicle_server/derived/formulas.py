"""The formula of each implemented derived signal (D-070 item 2, class A, parts 1 and 2).

Names, units, ranges, periods, windows and stale limits come from
vehicle_defs.vehicle_derived.SIGNALS; vehicle constants from vehicle_defs.vehicle_constants.
This module only holds the arithmetic, which follows vehicle-defs
docs/vehicle-data-catalog.md section 5 and each signal's description in the defs. The input
names used below are checked against the defs by tests (every one must be a declared input
of its signal and must resolve).

Three kinds of formula:
  point   fn(at) -> float, evaluated on the signal's output grid; raises InputInvalid
          (flag) when an input is unusable.
  series  fn(env, grid) -> [(value | None, flag | None)] per grid time, for the
          integrals (distance, trip time, idle time) and the altitude reference that carry
          state along the time axis.
  event   fn(env) -> [(time_ms, value | None, flag | None)]; no output grid: the engine
          evaluates it over the input feeds and the rows are at the event times.
Unit conversions and physics constants are named below; they are not vehicle data.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass

from vehicle_server.derived.context import (
    FLAG_INSUFFICIENT_SAMPLES,
    FLAG_INVALID_INPUT,
    At,
    Env,
    InputInvalid,
)
from vehicle_server.derived.inputs import STANDARD_GRAVITY_MPS2

# --- unit conversions and physics (not vehicle constants) -----------------------------
KMH_TO_MPS = 1.0 / 3.6
RPM_TO_RAD_PER_S = 2.0 * math.pi / 60.0
DEG_TO_RAD = math.pi / 180.0
# Specific gas constant of dry air, J/(kg*K) (ideal-gas law rho = p / (R * T)).
SPECIFIC_GAS_CONSTANT_DRY_AIR = 287.058
KELVIN_OFFSET = 273.15
# Standard gravity g0, m/s2 (hypsometric equation, lean cross-check); same value as the
# g-to-m/s2 factor of the IMU columns.
STANDARD_GRAVITY = STANDARD_GRAVITY_MPS2
# Unit conversions of the fuel formulas.
MG_PER_KG = 1.0e6
G_PER_KG = 1000.0
SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3600.0
# One injection per two crankshaft revolutions (four-stroke, one injector; defs description).
REVOLUTIONS_PER_INJECTION = 2.0
# SAE J1349 reference conditions of the power correction: 99 kPa dry-air pressure, 25 degC.
J1349_REFERENCE_PRESSURE_KPA = 99.0
J1349_REFERENCE_TEMPERATURE_C = 25.0

# --- tuning of the estimators (assumptions of this reference, documented in the README) -
# A measured gearbox ratio matches a gear when it is within this relative distance of the
# gear's ratio; the closest gear ratios (5th/6th) are 12 % apart, so 5 % cannot overlap.
GEAR_MATCH_TOLERANCE = 0.05
# Below this yaw rate the turn radius is not defined (going straight; gyro noise floor).
TURN_RADIUS_MIN_YAW_RATE_DPS = 0.5

FLAG_STANDSTILL = "standstill"
FLAG_STRAIGHT_LINE = "straight_line"
FLAG_NO_CONSUMPTION = "no_consumption"
FLAG_NO_POWER = "no_power"
FLAG_ENGINE_OFF = "engine_off"

# --- input names (all of them declared by the defs; tests check this) -------------------
_RPM = "did.ENGINE_SPEED"
_SPEED = "did.VEHICLE_SPEED"
_TORQUE_ACTUAL_PCT = "did.ACTUAL_TORQUE_PCT"
_TORQUE_DEMAND_PCT = "did.DEMAND_TORQUE_PCT"
_TORQUE_REFERENCE = "did.REFERENCE_TORQUE"
_BARO = "did.BAROMETRIC_PRESSURE"
_IAT = "did.INTAKE_AIR_TEMP"
_LAMBDA = "did.COMMANDED_EQUIV_RATIO"
_MAF = "did.MAF_RATE"
_IMU_AX = "sensor.imu_accel_x"
_IMU_GZ = "sensor.imu_gyro_z"
_RATIO = "derived.gearbox_ratio_measured"
_ACCEL = "derived.vehicle_accel"
_YAW = "derived.yaw_rate"
_TORQUE = "derived.engine_torque"
_TORQUE_DEMAND = "derived.demand_torque"
_DENSITY = "derived.air_density"
_GEAR_PREFIX = "const.gear_ratio_"
_FUEL_RATE = "did.ENGINE_FUEL_RATE"
_FUEL_LEVEL = "did.FUEL_LEVEL"
_STFT = "did.SHORT_TERM_FUEL_TRIM_B1"
_LTFT = "did.LONG_TERM_FUEL_TRIM_B1"
_THROTTLE_A = "did.THROTTLE_POS"
_THROTTLE_B = "did.ABSOLUTE_THROTTLE_POS_B"
_THROTTLE_R = "did.RELATIVE_THROTTLE_POS"
_THROTTLE_CMD = "did.COMMANDED_THROTTLE"
_PEDAL_D = "did.ACCEL_PEDAL_POS_D"
_PEDAL_REL = "did.RELATIVE_ACCEL_PEDAL_POS"
_COOLANT = "did.COOLANT_TEMP"
_OIL = "did.ENGINE_OIL_TEMP"
_LEAN = "sensor.lean_angle"
_ALTITUDE = "derived.altitude_change"
_TRIP_DISTANCE = "derived.trip_distance"
_LATERAL = "derived.lateral_accel"
_FUEL_AVG = "derived.fuel_consumption_avg"
_ENGINE_POWER = "derived.engine_power"

PointFn = Callable[[At], float]
Outcome = tuple[float | None, str | None]
SeriesFn = Callable[[Env, list[int]], list[Outcome]]
EventRow = tuple[int, float | None, str | None]
EventFn = Callable[[Env], list[EventRow]]


@dataclass(frozen=True)
class Formula:
    kind: str  # "point" | "series" | "event"
    fn: Callable


FORMULAS: dict[str, Formula] = {}


def _point(name: str) -> Callable[[PointFn], PointFn]:
    def register(fn: PointFn) -> PointFn:
        FORMULAS[name] = Formula("point", fn)
        return fn

    return register


def _series(name: str) -> Callable[[SeriesFn], SeriesFn]:
    def register(fn: SeriesFn) -> SeriesFn:
        FORMULAS[name] = Formula("series", fn)
        return fn

    return register


def _event(name: str) -> Callable[[EventFn], EventFn]:
    def register(fn: EventFn) -> EventFn:
        FORMULAS[name] = Formula("event", fn)
        return fn

    return register


# --- helpers -----------------------------------------------------------------------------


def air_density_kg_m3(pressure_kpa: float, temperature_c: float) -> float:
    """Ideal-gas density of dry air, rho = p / (R * T)."""
    return pressure_kpa * 1000.0 / (SPECIFIC_GAS_CONSTANT_DRY_AIR * (temperature_c + KELVIN_OFFSET))


def slope_per_s(points: list[tuple[int, float]]) -> float:
    """Least-squares slope (unit per second) of (time_ms, value) samples."""
    if len(points) < 2:
        raise InputInvalid(FLAG_INSUFFICIENT_SAMPLES)
    n = len(points)
    mean_t = sum(t for t, _ in points) / n
    mean_v = sum(v for _, v in points) / n
    var = sum((t - mean_t) ** 2 for t, _ in points)
    if var == 0:
        raise InputInvalid(FLAG_INSUFFICIENT_SAMPLES)
    return sum((t - mean_t) * (v - mean_v) for t, v in points) / var * 1000.0


def mean(points: list[tuple[int, float]]) -> float:
    if not points:
        raise InputInvalid(FLAG_INSUFFICIENT_SAMPLES)
    return sum(v for _, v in points) / len(points)


# --- class A: speed derivatives ------------------------------------------------------------


@_point("vehicle_accel")
def vehicle_accel(at: At) -> float:
    """d(speed)/dt: least-squares slope of the ECU speed over window_ms (filtered)."""
    return slope_per_s([(t, v * KMH_TO_MPS) for t, v in at.window(_SPEED)])


@_point("vehicle_jerk")
def vehicle_jerk(at: At) -> float:
    """Slope of the filtered acceleration over window_ms."""
    return slope_per_s(at.window(_ACCEL))


@_point("accel_imu_residual")
def accel_imu_residual(at: At) -> float:
    """IMU longitudinal acceleration (mean over the same window) minus the speed-derived
    acceleration. Gravity along the pitch axis is not removed (cross-check only)."""
    return mean(at.window(_IMU_AX)) - at.value(_ACCEL)


# --- class A: gearbox ------------------------------------------------------------------------


@_point("gearbox_ratio_measured")
def gearbox_ratio_measured(at: At) -> float:
    """Engine revolutions per gearbox output revolution:
    rpm / (wheel rpm * primary * secondary), wheel rpm from the speed and the dynamic radius."""
    speed = at.value(_SPEED) * KMH_TO_MPS
    if speed <= 0:
        raise InputInvalid(FLAG_STANDSTILL)
    radius = at.const("const.rear_wheel_radius_dynamic_m")
    wheel_rpm = speed / (2.0 * math.pi * radius) * 60.0
    reduction = at.const("const.primary_reduction_ratio") * at.const(
        "const.secondary_reduction_ratio"
    )
    return at.value(_RPM) / (wheel_rpm * reduction)


@_point("gear_estimate")
def gear_estimate(at: At) -> float:
    """The gear (1..n) whose ratio is nearest to the measured one, 0 when none is within
    GEAR_MATCH_TOLERANCE (relative)."""
    measured = at.value(_RATIO)
    gears = {
        int(name.removeprefix(_GEAR_PREFIX)): at.const(name)
        for name in at.env.inputs
        if name.startswith(_GEAR_PREFIX)
    }
    gear, ratio = min(gears.items(), key=lambda item: abs(measured / item[1] - 1.0))
    return float(gear) if abs(measured / ratio - 1.0) <= GEAR_MATCH_TOLERANCE else 0.0


# --- class A: distance and time (integrals along the time axis) -----------------------------


def hold_segments(env: Env, names: list[str]) -> tuple[list[int], list[list[float]], list[float]]:
    """The time axis of an integral: per usable breakpoint (the union of the input sample
    times where every input is fresh) its time, the input values held from there and the
    seconds they are held.

    A value is held until the next breakpoint, but never longer than the shortest remaining
    freshness of the inputs, so a gap in the data adds nothing. Shared by the class-A
    integrals and the class-B summaries (derived/summary.py)."""
    feeds = [env.feed(n) for n in names]
    breakpoints = sorted({t for f in feeds for t in f.times})
    starts: list[int] = []
    held: list[list[float]] = []
    spans: list[float] = []
    for j, b in enumerate(breakpoints):
        at = At(env, b)
        try:
            values = [at.value(n) for n in names]
        except InputInvalid:
            continue
        remaining = min(f.times[f.latest(b)] + f.stale_after_ms - b for f in feeds)
        until_next = breakpoints[j + 1] - b if j + 1 < len(breakpoints) else 0
        starts.append(b)
        held.append(values)
        spans.append(max(0, min(until_next, remaining)) / 1000.0)
    return starts, held, spans


def _integrate(
    env: Env, names: list[str], rate: Callable[[list[float]], float], grid: list[int]
) -> list[Outcome]:
    """Cumulative integral of rate(inputs) per second over the whole session.

    The rate is held between breakpoints (the union of the input sample times), but never
    longer than the shortest remaining freshness of the inputs, so a gap in the data adds
    nothing. The output at a grid time is null while any input is stale there. The
    integral starts when every input first exists (session start = key-on)."""
    starts, held, spans = hold_segments(env, names)
    rates = [rate(values) for values in held]
    prefix = [0.0]
    for r, s in zip(rates, spans, strict=True):
        prefix.append(prefix[-1] + r * s)

    out: list[Outcome] = []
    for t in grid:
        try:
            at = At(env, t)
            for n in names:
                at.value(n)
        except InputInvalid as exc:
            out.append((None, exc.flag))
            continue
        j = bisect_right(starts, t) - 1
        if j < 0:
            out.append((0.0, None))
            continue
        out.append((prefix[j] + rates[j] * min((t - starts[j]) / 1000.0, spans[j]), None))
    return out


@_series("trip_distance")
def trip_distance(env: Env, grid: list[int]) -> list[Outcome]:
    """Integral of the ECU speed, km (km/h divided by 3600 s/h per second)."""
    return _integrate(env, [_SPEED], lambda v: v[0] / 3600.0, grid)


@_series("trip_time")
def trip_time(env: Env, grid: list[int]) -> list[Outcome]:
    """Seconds with the engine running (RPM above 0)."""
    return _integrate(env, [_RPM], lambda v: 1.0 if v[0] > 0 else 0.0, grid)


@_series("idle_time")
def idle_time(env: Env, grid: list[int]) -> list[Outcome]:
    """Seconds with the engine running and the vehicle standing (speed 0, RPM above 0)."""
    return _integrate(env, [_SPEED, _RPM], lambda v: 1.0 if v[0] == 0 and v[1] > 0 else 0.0, grid)


# --- class A: yaw rate, lateral acceleration, turn radius ------------------------------------


@_point("yaw_rate")
def yaw_rate(at: At) -> float:
    """Mean of the IMU z gyro (deg/s) over window_ms. The IMU z axis is taken as the
    vehicle yaw axis (no mounting or lean correction)."""
    return mean(at.window(_IMU_GZ))


@_point("lateral_accel")
def lateral_accel(at: At) -> float:
    """a_lat = v * omega (v in m/s, omega in rad/s)."""
    return at.value(_SPEED) * KMH_TO_MPS * at.value(_YAW) * DEG_TO_RAD


@_point("turn_radius")
def turn_radius(at: At) -> float:
    """r = v / |omega|, m; the direction of the turn is the sign of the yaw rate."""
    yaw = at.value(_YAW)
    if abs(yaw) < TURN_RADIUS_MIN_YAW_RATE_DPS:
        raise InputInvalid(FLAG_STRAIGHT_LINE)
    return at.value(_SPEED) * KMH_TO_MPS / (abs(yaw) * DEG_TO_RAD)


# --- class A: torque and power ----------------------------------------------------------------


@_point("engine_torque")
def engine_torque(at: At) -> float:
    """Nm = actual torque % * reference torque / 100."""
    return at.value(_TORQUE_ACTUAL_PCT) * at.value(_TORQUE_REFERENCE) / 100.0


@_point("demand_torque")
def demand_torque(at: At) -> float:
    """Nm = demand torque % * reference torque / 100."""
    return at.value(_TORQUE_DEMAND_PCT) * at.value(_TORQUE_REFERENCE) / 100.0


@_point("torque_gap")
def torque_gap(at: At) -> float:
    """Demanded minus actual torque, Nm."""
    return at.value(_TORQUE_DEMAND) - at.value(_TORQUE)


@_point("engine_power")
def engine_power(at: At) -> float:
    """P = T * omega, kW."""
    return at.value(_TORQUE) * at.value(_RPM) * RPM_TO_RAD_PER_S / 1000.0


# --- class A: air and fuel ----------------------------------------------------------------------


@_point("air_density")
def air_density(at: At) -> float:
    """rho = p / (R * T) from the barometric pressure and the intake air temperature
    (dry air; the intake temperature stands in for the ambient one, as the defs list)."""
    return air_density_kg_m3(at.value(_BARO), at.value(_IAT))


@_point("power_correction_factor")
def power_correction_factor(at: At) -> float:
    """Density-ratio form of the SAE J1349 correction: rho_ref / rho, with rho_ref the dry-air
    density at the J1349 reference conditions (99 kPa, 25 degC). A "J1349-style"
    approximation (catalog): humidity is ignored and the barometric reading (total pressure)
    stands in for the dry-air pressure."""
    reference = air_density_kg_m3(J1349_REFERENCE_PRESSURE_KPA, J1349_REFERENCE_TEMPERATURE_C)
    return reference / at.value(_DENSITY)


@_point("air_fuel_ratio")
def air_fuel_ratio(at: At) -> float:
    """AFR = lambda * stoichiometric AFR."""
    return at.value(_LAMBDA) * at.const("const.stoichiometric_afr")


@_point("fuel_rate_maf")
def fuel_rate_maf(at: At) -> float:
    """L/h = MAF / (AFR * fuel density); MAF in g/s is 3.6 kg/h per g/s."""
    afr = at.value(_LAMBDA) * at.const("const.stoichiometric_afr")
    if afr <= 0:
        raise InputInvalid(FLAG_INVALID_INPUT)
    return at.value(_MAF) * 3.6 / (afr * at.const("const.fuel_density_kg_per_l"))


# --- class A, part 2: fuel economy ----------------------------------------------------------


@_point("fuel_consumption_instant")
def fuel_consumption_instant(at: At) -> float:
    """L/100km = fuel rate [L/h] / speed [km/h] * 100."""
    speed = at.value(_SPEED)
    if speed <= 0:
        raise InputInvalid(FLAG_STANDSTILL)
    return at.value(_FUEL_RATE) / speed * 100.0


@_point("fuel_consumption_avg")
def fuel_consumption_avg(at: At) -> float:
    """L/100km = 100 * mean(fuel rate) / mean(speed) over window_ms: the ratio of the window
    integrals, since each DID is sampled at its own fixed period, so its sample mean is its
    time average over the window."""
    rate = at.window(_FUEL_RATE)
    speed = at.window(_SPEED)
    if len(rate) < 2 or len(speed) < 2:
        raise InputInvalid(FLAG_INSUFFICIENT_SAMPLES)
    mean_speed = mean(speed)
    if mean_speed <= 0:
        raise InputInvalid(FLAG_STANDSTILL)
    return 100.0 * mean(rate) / mean_speed


@_point("fuel_range")
def fuel_range(at: At) -> float:
    """km = fuel level [%] / 100 * tank capacity [L] / average consumption [L/100km] * 100."""
    average = at.value(_FUEL_AVG)
    if average <= 0:
        raise InputInvalid(FLAG_NO_CONSUMPTION)
    litres = at.value(_FUEL_LEVEL) / 100.0 * at.const("const.fuel_tank_capacity_l")
    return litres / average * 100.0


@_point("bsfc")
def bsfc(at: At) -> float:
    """Brake-specific fuel consumption, g/kWh = fuel rate [L/h] * density [kg/L] * 1000 g/kg
    / engine power [kW]."""
    power = at.value(_ENGINE_POWER)
    if power <= 0:
        raise InputInvalid(FLAG_NO_POWER)
    mass_rate = at.value(_FUEL_RATE) * at.const("const.fuel_density_kg_per_l")
    return mass_rate * G_PER_KG / power


@_point("fuel_per_stroke")
def fuel_per_stroke(at: At) -> float:
    """mg per injection = fuel rate [L/h] * density [kg/L] * 1e6 mg/kg / injections per hour,
    with one injection per two revolutions: RPM * 60 / 2 injections/h."""
    rpm = at.value(_RPM)
    if rpm <= 0:
        raise InputInvalid(FLAG_ENGINE_OFF)
    mg_per_hour = at.value(_FUEL_RATE) * at.const("const.fuel_density_kg_per_l") * MG_PER_KG
    return mg_per_hour / (rpm * SECONDS_PER_MINUTE / REVOLUTIONS_PER_INJECTION)


@_point("total_fuel_trim")
def total_fuel_trim(at: At) -> float:
    """Short-term plus long-term fuel trim, bank 1, %."""
    return at.value(_STFT) + at.value(_LTFT)


# --- class A, part 2: redundant channels ----------------------------------------------------


@_point("throttle_channel_residual")
def throttle_channel_residual(at: At) -> float:
    """max(|A-B|, |A-R|, |B-R|) over the throttle position A, the absolute throttle B and
    the relative throttle R, %."""
    a, b, r = at.value(_THROTTLE_A), at.value(_THROTTLE_B), at.value(_THROTTLE_R)
    return max(abs(a - b), abs(a - r), abs(b - r))


@_point("throttle_command_residual")
def throttle_command_residual(at: At) -> float:
    """Commanded throttle minus throttle position, %."""
    return at.value(_THROTTLE_CMD) - at.value(_THROTTLE_A)


@_point("grip_channel_residual")
def grip_channel_residual(at: At) -> float:
    """Accelerator pedal position D minus the relative accelerator pedal position, %."""
    return at.value(_PEDAL_D) - at.value(_PEDAL_REL)


# --- class A, part 2: altitude and grade ---------------------------------------------------


@_series("altitude_change")
def altitude_change(env: Env, grid: list[int]) -> list[Outcome]:
    """Hypsometric equation dh = (R_dry * T / g0) * ln(p0 / p), m, with T the intake air
    temperature in K at t, p the barometric pressure at t and p0 the first valid barometric
    sample of the session (the key-on reference, "since key-on" in the catalog)."""
    feed = env.feed(_BARO)
    reference = next((v for v in feed.values if v is not None and v > 0), None)
    out: list[Outcome] = []
    for t in grid:
        try:
            at = At(env, t)
            pressure = at.value(_BARO)
            temperature = at.value(_IAT) + KELVIN_OFFSET
            if reference is None or pressure <= 0 or temperature <= 0:
                raise InputInvalid(FLAG_INVALID_INPUT)
        except InputInvalid as exc:
            out.append((None, exc.flag))
            continue
        scale_height = SPECIFIC_GAS_CONSTANT_DRY_AIR * temperature / STANDARD_GRAVITY
        out.append((scale_height * math.log(reference / pressure), None))
    return out


@_point("road_grade")
def road_grade(at: At) -> float:
    """% = 100 * (alt_last - alt_first) / ((dist_last - dist_first) * 1000), using the first
    and last valid samples of the altitude change [m] and the trip distance [km] in the
    window. sensor.imu_accel_x is declared as context only and is not used numerically."""
    altitude = at.window(_ALTITUDE)
    distance = at.window(_TRIP_DISTANCE)
    if len(altitude) < 2 or len(distance) < 2:
        raise InputInvalid(FLAG_INSUFFICIENT_SAMPLES)
    run_m = (distance[-1][1] - distance[0][1]) * 1000.0
    if run_m <= 0:
        raise InputInvalid(FLAG_STANDSTILL)
    return 100.0 * (altitude[-1][1] - altitude[0][1]) / run_m


# --- class A, part 2: thermal rates and lean cross-check -----------------------------------


@_point("coolant_rate")
def coolant_rate(at: At) -> float:
    """degC/min = least-squares slope of the coolant temperature over window_ms * 60."""
    return slope_per_s(at.window(_COOLANT)) * SECONDS_PER_MINUTE


@_point("oil_temp_rate")
def oil_temp_rate(at: At) -> float:
    """degC/min = least-squares slope of the engine oil temperature over window_ms * 60."""
    return slope_per_s(at.window(_OIL)) * SECONDS_PER_MINUTE


@_point("lean_crosscheck_residual")
def lean_crosscheck_residual(at: At) -> float:
    """deg = lean angle - atan2(lateral acceleration, g0) in degrees. Both are taken as
    positive = right (lean_angle per the defs; the sign of lateral_accel is the IMU z gyro
    sign as mounted): to be confirmed on the bike. In phase 0 the BLE leanAngle field is
    deprecated, so this input is normally missing."""
    expected = math.degrees(math.atan2(at.value(_LATERAL), STANDARD_GRAVITY))
    return at.value(_LEAN) - expected


# --- class A, part 2: events ----------------------------------------------------------------


@_event("stop_event")
def stop_event(env: Env) -> list[EventRow]:
    """The vehicle came to a stop with the engine running (catalog: speed = 0 and RPM > 0).
    A stop starts at a speed sample equal to 0 that follows a sample above 0, while the
    latest engine speed sample (fresh) is above 0; it ends at the first later speed sample
    above 0. The value is the duration in s and the row time the end time. Standing at the
    start of the data is no stop (the vehicle never came to it). A gap in the speed samples
    above the speed DID's own stale_after_ms ends both the stop and the "moving" state, so neither
    is followed across it; a stop still open at the end of the data yields no event."""
    speed = env.feed(_SPEED)
    limit = speed.stale_after_ms  # the speed DID's own stale_after_ms (D-087)
    events: list[EventRow] = []
    start: int | None = None
    previous: int | None = None
    moving = False
    for t, v in zip(speed.times, speed.values, strict=True):
        if v is None:
            continue
        if previous is not None and t - previous > limit:
            start, moving = None, False  # nothing is followed across a gap
        previous = t
        if start is not None:
            if v > 0:
                events.append((t, (t - start) / 1000.0, None))
                start = None
        elif v == 0 and moving:
            try:
                engine_on = At(env, t).value(_RPM) > 0
            except InputInvalid:
                engine_on = False
            if engine_on:
                start = t
        moving = v > 0
    return events


# The D-103 signals register themselves in FORMULAS; the modules need the registries above.
from vehicle_server.derived import d103_states, d103_windows  # noqa: E402, F401
