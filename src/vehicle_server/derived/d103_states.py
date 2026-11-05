"""D-103 class-A states and flags (LR15-1): clutch, neutral/coasting, engine braking,
over-rev, lugging, cold-engine abuse. The rule of each is the "Rule (D-103 ...)" sentence of
its defs description; thresholds come from the declared const.* inputs."""

from __future__ import annotations

from vehicle_server.derived.context import FLAG_INVALID_INPUT, At, InputInvalid
from vehicle_server.derived.d103_common import (
    _COOLANT,
    _FUEL_RATE,
    _GEAR,
    _IDLE_MAX,
    _LOAD,
    _MOVING_MIN,
    _OPERATING,
    _RATIO,
    _RPM,
    _SPEED,
    _THROTTLE,
    _THROTTLE_CLOSED,
    FLAG_NO_GEAR,
    _flag,
    _gear_ratios,
    _require_moving,
)
from vehicle_server.derived.formulas import _point

# --- LR15-1: states and flags ------------------------------------------------------------------


@_point("clutch_state")
def clutch_state(at: At) -> float:
    """0 engaged, 1 disengaged, 2 slipping, only while moving (speed above the moving
    threshold; otherwise null, flag not_moving).

    engaged: the measured ratio is within derived_gear_ratio_tolerance of some gear ratio.
    "Within" is read as a relative distance |measured / gear - 1| <= tolerance (the defs
    describe the tolerance as half the relative step of the two closest gears).
    slipping: the measured ratio is above its nearest gear (nearest by the same relative
    distance) by more than the tolerance and THROTTLE_POS > derived_throttle_load_min_pct.
    disengaged: farther than the tolerance from every gear otherwise.
    ENGINE_SPEED is read (it must be fresh); derived_throttle_closed_max_pct is declared
    but the rule does not use it."""
    _require_moving(at)
    at.value(_RPM)
    measured = at.value(_RATIO)
    tolerance = at.const("const.derived_gear_ratio_tolerance")
    nearest = min(_gear_ratios(at).values(), key=lambda ratio: abs(measured / ratio - 1.0))
    distance = measured / nearest - 1.0
    if abs(distance) <= tolerance:
        return 0.0
    if distance > tolerance and at.value(_THROTTLE) > at.const(
        "const.derived_throttle_load_min_pct"
    ):
        return 2.0
    return 1.0


@_point("clutch_slip_ratio")
def clutch_slip_ratio(at: At) -> float:
    """Measured gearbox ratio over the nominal ratio of gear_estimate's gear (the current
    sample of the measured ratio; window_ms is not used). Invalid while gear_estimate is
    invalid or 0 (no gear matches, flag no_gear). THROTTLE_POS is declared, not used."""
    gear = int(at.value(_GEAR))
    if gear < 1:
        raise InputInvalid(FLAG_NO_GEAR)
    return at.value(_RATIO) / _gear_ratios(at)[gear]


@_point("neutral_coasting_state")
def neutral_coasting_state(at: At) -> float:
    """0 geared and driven, 1 neutral, 2 coasting; only while moving (else null, not_moving).

    neutral: no valid gear_estimate. Read as gear_estimate == 0 (no gear matches) or an
    invalid gear_estimate sample (e.g. the measured ratio is out of range); a STALE
    gear_estimate stays null. coasting: a gear and THROTTLE_POS <= derived_throttle_closed_max_pct.
    ENGINE_SPEED is read (it must be fresh)."""
    _require_moving(at)
    at.value(_RPM)
    try:
        gear = at.value(_GEAR)
    except InputInvalid as exc:
        if exc.flag != FLAG_INVALID_INPUT:
            raise
        return 1.0
    if gear < 1:
        return 1.0
    return 2.0 if at.value(_THROTTLE) <= at.const(_THROTTLE_CLOSED) else 0.0


@_point("engine_braking_phase")
def engine_braking_phase(at: At) -> float:
    """1 when moving, THROTTLE_POS <= closed max, ENGINE_SPEED > derived_idle_rpm_max and
    ENGINE_FUEL_RATE <= engine_braking_phase_fuel_rate_max_lph, else 0 (all inputs fresh)."""
    speed = at.value(_SPEED)
    throttle = at.value(_THROTTLE)
    rpm = at.value(_RPM)
    fuel = at.value(_FUEL_RATE)
    return _flag(
        speed > at.const(_MOVING_MIN)
        and throttle <= at.const(_THROTTLE_CLOSED)
        and rpm > at.const(_IDLE_MAX)
        and fuel <= at.const("const.engine_braking_phase_fuel_rate_max_lph")
    )


@_point("over_rev")
def over_rev(at: At) -> float:
    """1 while ENGINE_SPEED >= over_rev_rpm_min."""
    return _flag(at.value(_RPM) >= at.const("const.over_rev_rpm_min"))


@_point("lugging")
def lugging(at: At) -> float:
    """1 when gear_estimate >= 2, ENGINE_SPEED < lugging_rpm_max and ENGINE_LOAD >
    lugging_load_min_pct. An invalid gear_estimate makes the output null."""
    gear = at.value(_GEAR)
    rpm = at.value(_RPM)
    load = at.value(_LOAD)
    return _flag(
        gear >= 2
        and rpm < at.const("const.lugging_rpm_max")
        and load > at.const("const.lugging_load_min_pct")
    )


@_point("cold_engine_abuse")
def cold_engine_abuse(at: At) -> float:
    """1 when COOLANT_TEMP < derived_operating_coolant_degc and (ENGINE_SPEED >
    cold_engine_abuse_rpm_max or ENGINE_LOAD > cold_engine_abuse_load_max_pct). The oil
    temperature is declared but the rule does not use it."""
    coolant = at.value(_COOLANT)
    rpm = at.value(_RPM)
    load = at.value(_LOAD)
    return _flag(
        coolant < at.const(_OPERATING)
        and (
            rpm > at.const("const.cold_engine_abuse_rpm_max")
            or load > at.const("const.cold_engine_abuse_load_max_pct")
        )
    )
