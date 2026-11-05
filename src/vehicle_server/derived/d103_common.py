"""Shared pieces of the D-103 class-A signals (d103_states, d103_windows, d103_events).

vehicle-defs v0.23.0 (D-103; wiring fixes in v0.24.0) gave each of these signals a
"Rule (D-103, provisional)" sentence in its description and declared every threshold as a
`const.*` input. These modules implement the sentence as written. Every threshold, band,
window or reference value is read from its declared constant (`at.const(...)` /
`env.const(...)`), so a later defs release changes the results without a code change. The
only literals are unit conversions and 0/1.

Where a sentence is ambiguous the simplest literal reading is taken and written in the
docstring of the function. The three modules register their formulas in formulas.FORMULAS
(they are imported at the end of formulas.py).

Signals of D-103 that are NOT implemented, with the reason, are in NOT_IMPLEMENTED_REASONS_A
(the engine reports them as not_implemented).
"""

from __future__ import annotations

from vehicle_server.derived.context import (
    At,
    Env,
    InputInvalid,
)

FLAG_NOT_MOVING = "not_moving"
FLAG_NO_GEAR = "no_gear"
FLAG_NOT_APPLICABLE = "not_applicable"

# Class-A signals of the D-103 batch that stay not_implemented, with the reason.
NOT_IMPLEMENTED_REASONS_A: dict[str, str] = {
    "data_quality_mask": (
        "definition: bit layout of the mask is not defined and the DID ages/states are not inputs"
    ),
}

_RPM = "did.ENGINE_SPEED"
_SPEED = "did.VEHICLE_SPEED"
_THROTTLE = "did.THROTTLE_POS"
_LOAD = "did.ENGINE_LOAD"
_COOLANT = "did.COOLANT_TEMP"
_BATT = "did.BATTERY_VOLTAGE"
_FUEL_RATE = "did.ENGINE_FUEL_RATE"
_FUEL_LEVEL = "did.FUEL_LEVEL"
_MAP = "did.INTAKE_MAP"
_IAT = "did.INTAKE_AIR_TEMP"
_AMBIENT = "did.AMBIENT_AIR_TEMP"
_RATIO = "derived.gearbox_ratio_measured"
_GEAR = "derived.gear_estimate"
_ACCEL = "derived.vehicle_accel"
_MAF = "did.MAF_RATE"
_GPS_SPEED = "sensor.gps_ground_speed"
_GPS_ACCURACY = "sensor.gps_speed_accuracy"
_GPS_FIX = "sensor.gps_fix_type"
_GYRO_Y = "sensor.imu_gyro_y"
_ACCEL_Z = "sensor.imu_accel_z"

_GEAR_PREFIX = "const.gear_ratio_"
_MOVING_MIN = "const.derived_moving_speed_min_kmh"
_THROTTLE_CLOSED = "const.derived_throttle_closed_max_pct"
_IDLE_MIN = "const.derived_idle_rpm_min"
_IDLE_MAX = "const.derived_idle_rpm_max"
_OPERATING = "const.derived_operating_coolant_degc"


# --- helpers -----------------------------------------------------------------------------


def _gear_ratios(at: At) -> dict[int, float]:
    """{gear: ratio} from the declared const.gear_ratio_N inputs."""
    return {
        int(name.removeprefix(_GEAR_PREFIX)): at.const(name)
        for name in at.env.inputs
        if name.startswith(_GEAR_PREFIX)
    }


def _require_moving(at: At) -> float:
    """The (fresh) ECU speed, km/h; raises not_moving at or below the moving threshold."""
    speed = at.value(_SPEED)
    if speed <= at.const(_MOVING_MIN):
        raise InputInvalid(FLAG_NOT_MOVING)
    return speed


def _speed_at(env: Env, t: int) -> float | None:
    try:
        return At(env, t).value(_SPEED)
    except InputInvalid:
        return None


def _flag(condition: bool) -> float:
    return 1.0 if condition else 0.0
