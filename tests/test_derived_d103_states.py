"""D-103 class-A states and flags (LR15-1) on synthetic series; thresholds follow the defs."""

from __future__ import annotations

import pytest

from vehicle_server.defs import vehicle_constants

from .derived_helpers import const_series, outcome, patch_const

SPEED = "did.VEHICLE_SPEED"
RPM = "did.ENGINE_SPEED"
THROTTLE = "did.THROTTLE_POS"
LOAD = "did.ENGINE_LOAD"
COOLANT = "did.COOLANT_TEMP"
FUEL = "did.ENGINE_FUEL_RATE"
RATIO = "derived.gearbox_ratio_measured"
GEAR = "derived.gear_estimate"
T = 500


def gear(n: int) -> float:
    return vehicle_constants.CONSTANTS[f"gear_ratio_{n}"][0]


def const(name: str) -> float:
    return vehicle_constants.CONSTANTS[name][0]


def feeds(**values: float) -> dict:
    names = {
        "speed": SPEED,
        "rpm": RPM,
        "throttle": THROTTLE,
        "load": LOAD,
        "coolant": COOLANT,
        "fuel": FUEL,
        "ratio": RATIO,
        "gear": GEAR,
    }
    return {names[k]: const_series(0, 1000, 10, v) for k, v in values.items()}


# --- clutch_state -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ratio_factor", "throttle", "expected"),
    [
        (1.0, 50.0, 0.0),  # on the gear ratio: engaged
        (1.04, 50.0, 0.0),  # inside the tolerance: engaged
        (1.2, 50.0, 2.0),  # above the nearest gear, throttle open: slipping
        (1.2, 5.0, 1.0),  # above the gear, throttle not open: disengaged
        (0.8, 50.0, 1.0),  # below every gear: disengaged
    ],
)
def test_clutch_state_engaged_slipping_disengaged(ratio_factor, throttle, expected):
    f = feeds(speed=50.0, rpm=5000.0, throttle=throttle, ratio=gear(6) * ratio_factor)
    assert outcome("clutch_state", f, T) == (expected, None)


def test_clutch_state_only_while_moving_and_stale_is_null():
    f = feeds(speed=2.0, rpm=5000.0, throttle=50.0, ratio=gear(3))
    assert outcome("clutch_state", f, T) == (None, "not_moving")
    f = feeds(speed=50.0, rpm=5000.0, throttle=50.0, ratio=gear(3))
    f[RATIO] = ([0], [gear(3)])  # newest ratio older than its stale limit
    assert outcome("clutch_state", f, T) == (None, "stale_input")


def test_clutch_state_follows_the_tolerance_and_the_load_threshold(monkeypatch):
    f = feeds(speed=50.0, rpm=5000.0, throttle=15.0, ratio=gear(3) * 1.1)
    assert outcome("clutch_state", f, T)[0] == 2.0  # 15 % > the load minimum (10 %)
    patch_const(monkeypatch, "derived_throttle_load_min_pct", 20.0)
    assert outcome("clutch_state", f, T)[0] == 1.0
    patch_const(monkeypatch, "derived_gear_ratio_tolerance", 0.15)
    assert outcome("clutch_state", f, T)[0] == 0.0


def test_clutch_state_follows_the_moving_threshold(monkeypatch):
    f = feeds(speed=5.0, rpm=5000.0, throttle=50.0, ratio=gear(3))
    assert outcome("clutch_state", f, T)[0] == 0.0
    patch_const(monkeypatch, "derived_moving_speed_min_kmh", 6.0)
    assert outcome("clutch_state", f, T) == (None, "not_moving")


# --- clutch_slip_ratio ------------------------------------------------------------------------


def test_clutch_slip_ratio_is_measured_over_the_nominal_ratio_of_the_estimated_gear():
    f = feeds(ratio=gear(4) * 1.25, gear=4.0, throttle=50.0)
    value, flag = outcome("clutch_slip_ratio", f, T)
    assert flag is None and value == pytest.approx(1.25)


def test_clutch_slip_ratio_is_invalid_without_a_gear():
    f = feeds(ratio=gear(4) * 1.25, gear=0.0, throttle=50.0)
    assert outcome("clutch_slip_ratio", f, T) == (None, "no_gear")
    f[GEAR] = ([0], [4.0])
    assert outcome("clutch_slip_ratio", f, T) == (None, "stale_input")


def test_clutch_slip_ratio_follows_the_gear_ratio_constant(monkeypatch):
    f = feeds(ratio=3.0, gear=2.0, throttle=50.0)
    assert outcome("clutch_slip_ratio", f, T)[0] == pytest.approx(3.0 / gear(2))
    patch_const(monkeypatch, "gear_ratio_2", 3.0)
    assert outcome("clutch_slip_ratio", f, T)[0] == pytest.approx(1.0)


# --- neutral_coasting_state -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gear_value", "throttle", "expected"),
    [(3.0, 50.0, 0.0), (3.0, 1.0, 2.0), (0.0, 50.0, 1.0), (0.0, 1.0, 1.0)],
)
def test_neutral_coasting_state_driven_coasting_neutral(gear_value, throttle, expected):
    f = feeds(speed=50.0, rpm=4000.0, throttle=throttle, gear=gear_value)
    assert outcome("neutral_coasting_state", f, T) == (expected, None)


def test_neutral_coasting_state_invalid_gear_is_neutral_but_stale_is_null():
    f = feeds(speed=50.0, rpm=4000.0, throttle=50.0)
    f[GEAR] = ([0, 10, 490, 500], [3.0, 3.0, None, None])  # the newest estimate is invalid
    assert outcome("neutral_coasting_state", f, T) == (1.0, None)
    f[GEAR] = ([0], [3.0])  # stale
    assert outcome("neutral_coasting_state", f, T) == (None, "stale_input")
    f = feeds(speed=1.0, rpm=4000.0, throttle=50.0, gear=3.0)
    assert outcome("neutral_coasting_state", f, T) == (None, "not_moving")


def test_neutral_coasting_state_follows_the_closed_throttle_threshold(monkeypatch):
    f = feeds(speed=50.0, rpm=4000.0, throttle=5.0, gear=3.0)
    assert outcome("neutral_coasting_state", f, T)[0] == 0.0
    patch_const(monkeypatch, "derived_throttle_closed_max_pct", 6.0)
    assert outcome("neutral_coasting_state", f, T)[0] == 2.0


# --- engine_braking_phase ---------------------------------------------------------------------


def _braking(**overrides: float) -> dict:
    base = {"throttle": 0.0, "rpm": 4000.0, "speed": 60.0, "fuel": 0.0}
    return feeds(**{**base, **overrides})


def test_engine_braking_phase_flag_edges():
    assert outcome("engine_braking_phase", _braking(), T) == (1.0, None)
    assert outcome("engine_braking_phase", _braking(throttle=5.0), T)[0] == 0.0
    assert outcome("engine_braking_phase", _braking(rpm=1500.0), T)[0] == 0.0  # idle band
    assert outcome("engine_braking_phase", _braking(fuel=0.5), T)[0] == 0.0
    assert outcome("engine_braking_phase", _braking(speed=1.0), T)[0] == 0.0
    # the thresholds are inclusive where the rule says <=
    assert (
        outcome(
            "engine_braking_phase",
            _braking(fuel=const("engine_braking_phase_fuel_rate_max_lph")),
            T,
        )[0]
        == 1.0
    )


def test_engine_braking_phase_stale_input_is_null():
    f = _braking()
    f[FUEL] = ([0], [0.0])
    assert outcome("engine_braking_phase", f, 10000) == (None, "stale_input")


def test_engine_braking_phase_follows_the_fuel_constant(monkeypatch):
    f = _braking(fuel=0.5)
    assert outcome("engine_braking_phase", f, T)[0] == 0.0
    patch_const(monkeypatch, "engine_braking_phase_fuel_rate_max_lph", 1.0)
    assert outcome("engine_braking_phase", f, T)[0] == 1.0


# --- over_rev ---------------------------------------------------------------------------------


def test_over_rev_edge_and_constant(monkeypatch):
    limit = const("over_rev_rpm_min")
    assert outcome("over_rev", feeds(rpm=limit), T) == (1.0, None)
    assert outcome("over_rev", feeds(rpm=limit - 1), T) == (0.0, None)
    patch_const(monkeypatch, "over_rev_rpm_min", limit + 500)
    assert outcome("over_rev", feeds(rpm=limit), T) == (0.0, None)
    f = feeds(rpm=limit)
    f[RPM] = ([0], [limit])
    assert outcome("over_rev", f, T) == (None, "stale_input")


# --- lugging ----------------------------------------------------------------------------------


def test_lugging_needs_gear_two_low_rpm_and_high_load(monkeypatch):
    base = {"gear": 3.0, "rpm": 2000.0, "load": 90.0}
    assert outcome("lugging", feeds(**base), T) == (1.0, None)
    assert outcome("lugging", feeds(**{**base, "gear": 1.0}), T)[0] == 0.0
    assert outcome("lugging", feeds(**{**base, "gear": 2.0}), T)[0] == 1.0  # >= 2
    assert outcome("lugging", feeds(**{**base, "rpm": 3500.0}), T)[0] == 0.0
    assert outcome("lugging", feeds(**{**base, "load": 50.0}), T)[0] == 0.0
    patch_const(monkeypatch, "lugging_load_min_pct", 95.0)
    assert outcome("lugging", feeds(**base), T)[0] == 0.0


def test_lugging_with_an_invalid_gear_estimate_is_null():
    f = feeds(rpm=2000.0, load=90.0)
    f[GEAR] = ([0, 500], [3.0, None])
    assert outcome("lugging", f, T) == (None, "invalid_input")


# --- cold_engine_abuse ------------------------------------------------------------------------


def test_cold_engine_abuse_cold_with_revs_or_load(monkeypatch):
    base = {"coolant": 40.0, "rpm": 3000.0, "load": 30.0}
    assert outcome("cold_engine_abuse", feeds(**base), T)[0] == 0.0
    assert outcome("cold_engine_abuse", feeds(**{**base, "rpm": 7000.0}), T)[0] == 1.0
    assert outcome("cold_engine_abuse", feeds(**{**base, "load": 80.0}), T)[0] == 1.0
    warm = {**base, "coolant": 90.0, "rpm": 7000.0}
    assert outcome("cold_engine_abuse", feeds(**warm), T)[0] == 0.0
    patch_const(monkeypatch, "derived_operating_coolant_degc", 95.0)
    assert outcome("cold_engine_abuse", feeds(**warm), T)[0] == 1.0
    patch_const(monkeypatch, "cold_engine_abuse_rpm_max", 8000.0)
    assert outcome("cold_engine_abuse", feeds(**warm), T)[0] == 0.0


def test_cold_engine_abuse_stale_coolant_is_null():
    f = feeds(coolant=40.0, rpm=7000.0, load=30.0)
    f[COOLANT] = ([0], [40.0])
    assert outcome("cold_engine_abuse", f, 10000) == (None, "stale_input")
