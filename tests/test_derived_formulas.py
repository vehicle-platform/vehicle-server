"""Formulas of the derived signals (parts 1 and 2), against hand-computed values."""

from __future__ import annotations

import math

import pytest

from vehicle_server.defs import vehicle_catalog, vehicle_cl250, vehicle_constants, vehicle_derived
from vehicle_server.derived import engine
from vehicle_server.derived.context import InputInvalid
from vehicle_server.derived.formulas import FORMULAS
from vehicle_server.derived.inputs import STANDARD_GRAVITY_MPS2, resolve_sensor

from .derived_helpers import FakeInputs, const_series, env_for, point_value, ramp

SPEED = "did.VEHICLE_SPEED"
RPM = "did.ENGINE_SPEED"
KMH_36 = 36.0  # 10 m/s
# The server stores no parquet column for these sensors (missing_inputs in practice).
UNSTORED_SENSORS = {"lean_angle"}
PART2_SIGNALS = [
    "fuel_consumption_instant",
    "fuel_consumption_avg",
    "fuel_range",
    "bsfc",
    "fuel_per_stroke",
    "total_fuel_trim",
    "throttle_channel_residual",
    "throttle_command_residual",
    "grip_channel_residual",
    "altitude_change",
    "road_grade",
    "coolant_rate",
    "oil_temp_rate",
    "lean_crosscheck_residual",
    "stop_event",
]
D103_SIGNALS = [
    "clutch_state",
    "clutch_slip_ratio",
    "neutral_coasting_state",
    "engine_braking_phase",
    "over_rev",
    "lugging",
    "cold_engine_abuse",
    "idle_rpm_std",
    "idle_rpm_drift",
    "charging_plateau_voltage",
    "fuel_level_filtered",
    "road_roughness",
    "heat_soak_delta",
    "idle_map_residual",
    "wheel_slip",
    "volumetric_efficiency",
    "intake_flow_residual",
    "shift_event",
    "launch_event",
    "braking_event",
    "pitch_event",
    "refuel_event",
    "cranking_dip",
    "battery_rest_voltage",
]
FUEL_DENSITY = vehicle_constants.CONSTANTS["fuel_density_kg_per_l"][0]
TANK_L = vehicle_constants.CONSTANTS["fuel_tank_capacity_l"][0]


def _outputs(inputs: dict, only: str):
    evaluation = engine.evaluate(FakeInputs(inputs))
    return {o.name: o for o in evaluation.outputs}[only], evaluation


# --- the registry matches the defs ---------------------------------------------------------


def test_every_formula_is_a_class_a_signal_with_resolvable_inputs():
    names = set()
    for name, formula in FORMULAS.items():
        spec = vehicle_derived.SIGNALS[name]
        assert spec["class"] == "A"
        assert bool(spec.get("event")) == (formula.kind == "event")
        assert spec["ready_phase0"] and not spec.get("pending_inputs")
        for inp in spec["inputs"]:
            kind, _, key = inp.partition(".")
            names.add(inp)
            if kind == "did":
                assert key in vehicle_catalog.SIGNALS, inp
            elif kind == "const":
                assert key in vehicle_constants.CONSTANTS, inp
            elif kind == "derived":
                assert key in vehicle_derived.SIGNALS, inp
            else:
                assert kind == "sensor" and key in vehicle_derived.SENSORS, inp
                if key in UNSTORED_SENSORS:
                    assert resolve_sensor(key) is None
                else:
                    assert resolve_sensor(key) is not None, f"{inp} has no parquet column"
    assert len(FORMULAS) == 19 + len(PART2_SIGNALS) + len(D103_SIGNALS)
    assert set(D103_SIGNALS) <= set(FORMULAS)
    assert set(PART2_SIGNALS) <= set(FORMULAS)
    assert "did.VEHICLE_SPEED" in names


def test_formulas_only_read_declared_inputs():
    """Env.feed raises KeyError for an undeclared input: run every formula once over a
    feed for each declared input and require that nothing but a data problem is raised."""
    for name, formula in FORMULAS.items():
        spec = vehicle_derived.SIGNALS[name]
        feeds = {
            inp: const_series(0, 5000, 10, 1.0)
            for inp in spec["inputs"]
            if not inp.startswith("const.")
        }
        try:
            if formula.kind == "point":
                point_value(name, feeds, 5000)
            elif formula.kind == "event":
                formula.fn(env_for(name, feeds))
            else:
                formula.fn(env_for(name, feeds), [5000])
        except (InputInvalid, ArithmeticError):
            pass


# --- speed derivatives ---------------------------------------------------------------------


def test_vehicle_accel_is_the_speed_slope_over_the_window():
    # +1 m/s every 110 ms = 3.6 km/h steps -> 1 / 0.11 s = 9.0909 m/s2
    speed = ramp(0, 880, 110, lambda t: 3.6 * t / 110)
    assert point_value("vehicle_accel", {SPEED: speed}, 880) == pytest.approx(10 / 1.1)


def test_vehicle_accel_needs_two_samples_in_the_window():
    with pytest.raises(InputInvalid) as exc:
        point_value("vehicle_accel", {SPEED: ([0, 100], [10.0, 12.0])}, 500)
    assert exc.value.flag == "stale_input"  # newest sample 400 ms old (limit 330)
    with pytest.raises(InputInvalid) as exc:
        point_value("vehicle_accel", {SPEED: ([100], [10.0])}, 100)
    assert exc.value.flag == "insufficient_samples"


def test_vehicle_jerk_is_the_slope_of_the_acceleration():
    accel = ramp(0, 880, 110, lambda t: t / 110)  # 1 m/s2 per 110 ms
    assert point_value("vehicle_jerk", {"derived.vehicle_accel": accel}, 880) == pytest.approx(
        1 / 0.11
    )


def test_accel_imu_residual_is_window_mean_minus_speed_acceleration():
    imu = ramp(0, 1000, 10, lambda _t: 2.0)
    value = point_value(
        "accel_imu_residual",
        {"sensor.imu_accel_x": imu, "derived.vehicle_accel": ([900], [1.5])},
        1000,
    )
    assert value == pytest.approx(0.5)


# --- gearbox ---------------------------------------------------------------------------------


def test_gearbox_ratio_from_rpm_speed_and_the_defs_constants():
    # 36 km/h = 10 m/s, r = 0.311 m, primary 2.807, secondary 2.642, 3000 rpm
    value = point_value("gearbox_ratio_measured", {RPM: ([0], [3000.0]), SPEED: ([0], [KMH_36])}, 0)
    assert value == pytest.approx(1.31745, abs=1e-4)
    r = vehicle_constants.CONSTANTS["rear_wheel_radius_dynamic_m"][0]
    wheel_rpm = 10 / (2 * math.pi * r) * 60
    reduction = (
        vehicle_constants.CONSTANTS["primary_reduction_ratio"][0]
        * vehicle_constants.CONSTANTS["secondary_reduction_ratio"][0]
    )
    assert value == pytest.approx(3000 / (wheel_rpm * reduction))


def test_gearbox_ratio_is_invalid_when_standing():
    with pytest.raises(InputInvalid) as exc:
        point_value("gearbox_ratio_measured", {RPM: ([0], [1500.0]), SPEED: ([0], [0.0])}, 0)
    assert exc.value.flag == "standstill"


@pytest.mark.parametrize(
    ("ratio", "gear"),
    [
        (3.416, 1.0),
        (1.3174, 4.0),  # 2.4 % below the 4th ratio 1.35
        (1.20, 5.0),  # 2.9 % above the 5th ratio 1.166
        (1.25, 0.0),  # 7.2 % from 5th and 7.4 % from 4th: no gear
        (6.0, 0.0),
    ],
)
def test_gear_estimate_is_the_nearest_gear_within_the_tolerance(ratio: float, gear: float):
    feed = {"derived.gearbox_ratio_measured": ([0], [ratio])}
    assert point_value("gear_estimate", feed, 0) == gear


# --- distance and time -------------------------------------------------------------------------


def test_trip_distance_integrates_the_speed():
    # 10 m/s for 3 s, a sample every 100 ms: 20 m at 2000 ms, 30 m at 3000 ms
    out, _ = _outputs({SPEED: const_series(0, 3000, 100, KMH_36)}, "trip_distance")
    assert out.times == [0, 1000, 2000, 3000]
    assert out.values == pytest.approx([0.0, 0.01, 0.02, 0.03])


def test_trip_distance_does_not_integrate_a_gap_longer_than_the_stale_limit():
    # speed samples 0..1000 and 9000..10000: the pre-gap sample is held for stale_after_ms
    # (VEHICLE_SPEED's own 300 ms, D-070) only, so 1 s + 0.3 s + 1 s at 10 m/s = 23 m;
    # 1300..9000 adds nothing.
    times = [*range(0, 1001, 100), *range(9000, 10001, 100)]
    out, _ = _outputs({SPEED: (times, [KMH_36] * len(times))}, "trip_distance")
    by_time = dict(zip(out.times, out.values, strict=True))
    assert by_time[10000] == pytest.approx(0.023)
    assert by_time[5000] is None  # stale while the gap lasts, never a held value
    assert out.flags[out.times.index(5000)] == "stale_input"


def test_trip_time_counts_the_engine_running():
    times = list(range(0, 2001, 100))
    rpm = [1000.0 if t < 1000 else 0.0 for t in times]
    out, _ = _outputs({RPM: (times, rpm)}, "trip_time")
    assert dict(zip(out.times, out.values, strict=True))[2000] == pytest.approx(1.0)


def test_idle_time_counts_standing_with_the_engine_running():
    times = list(range(0, 2001, 100))
    speed = [0.0 if t < 1000 else KMH_36 for t in times]
    out, _ = _outputs({SPEED: (times, speed), RPM: (times, [1000.0] * len(times))}, "idle_time")
    assert dict(zip(out.times, out.values, strict=True))[2000] == pytest.approx(1.0)


# --- yaw rate, lateral acceleration, turn radius ------------------------------------------------


def test_yaw_rate_is_the_gz_mean_over_the_window():
    gz = ramp(0, 200, 10, lambda t: t / 10)  # window (0, 100]: gz = 1..10
    assert point_value("yaw_rate", {"sensor.imu_gyro_z": gz}, 100) == pytest.approx(5.5)


def test_lateral_acceleration_is_speed_times_yaw_rate():
    feeds = {"derived.yaw_rate": ([0], [18.0]), SPEED: ([0], [KMH_36])}  # 10 m/s, 18 deg/s
    assert point_value("lateral_accel", feeds, 0) == pytest.approx(math.pi)  # 10 * 0.31416
    feeds = {"derived.yaw_rate": ([0], [-18.0]), SPEED: ([0], [KMH_36])}
    assert point_value("lateral_accel", feeds, 0) == pytest.approx(-math.pi)


def test_turn_radius_is_speed_over_yaw_rate_and_invalid_when_straight():
    feeds = {"derived.yaw_rate": ([0], [-18.0]), SPEED: ([0], [KMH_36])}
    assert point_value("turn_radius", feeds, 0) == pytest.approx(31.831, abs=1e-3)
    feeds = {"derived.yaw_rate": ([0], [0.2]), SPEED: ([0], [KMH_36])}
    with pytest.raises(InputInvalid) as exc:
        point_value("turn_radius", feeds, 0)
    assert exc.value.flag == "straight_line"


def test_turn_radius_beyond_the_range_is_null_out_of_range():
    # 0.6 deg/s at 10 m/s -> 955 m is fine; standing still gives 0 m, below the 3 m minimum
    inputs = {
        "sensor.imu_gyro_z": const_series(0, 400, 10, 0.6),
        SPEED: const_series(0, 400, 10, 0.0),
    }
    out, _ = _outputs(inputs, "turn_radius")
    assert set(out.flags) == {"out_of_range"} and set(out.values) == {None}


# --- torque and power ---------------------------------------------------------------------------


def test_torques_are_percent_times_reference():
    ref = ([0], [23.0])
    actual = point_value(
        "engine_torque", {"did.ACTUAL_TORQUE_PCT": ([0], [50.0]), "did.REFERENCE_TORQUE": ref}, 0
    )
    assert actual == pytest.approx(11.5)
    demand = point_value(
        "demand_torque", {"did.DEMAND_TORQUE_PCT": ([0], [60.0]), "did.REFERENCE_TORQUE": ref}, 0
    )
    assert demand == pytest.approx(13.8)
    negative = point_value(
        "engine_torque", {"did.ACTUAL_TORQUE_PCT": ([0], [-10.0]), "did.REFERENCE_TORQUE": ref}, 0
    )
    assert negative == pytest.approx(-2.3)


def test_torque_gap_is_demand_minus_actual():
    feeds = {"derived.demand_torque": ([0], [13.8]), "derived.engine_torque": ([0], [11.5])}
    assert point_value("torque_gap", feeds, 0) == pytest.approx(2.3)


def test_engine_power_is_torque_times_angular_speed():
    feeds = {"derived.engine_torque": ([0], [10.0]), RPM: ([0], [6000.0])}
    assert point_value("engine_power", feeds, 0) == pytest.approx(6.2832, abs=1e-4)  # 10 * 628.3 W


# --- air and fuel -------------------------------------------------------------------------------


def test_air_density_is_the_ideal_gas_density():
    feeds = {"did.BAROMETRIC_PRESSURE": ([0], [101.0]), "did.INTAKE_AIR_TEMP": ([0], [20.0])}
    assert point_value("air_density", feeds, 0) == pytest.approx(1.20022, abs=1e-5)


def test_power_correction_factor_is_the_density_ratio_to_the_j1349_reference():
    # rho_ref = 99 kPa, 25 degC = 1.15673 kg/m3; at 1.20022 kg/m3 -> 0.96376
    value = point_value("power_correction_factor", {"derived.air_density": ([0], [1.20022])}, 0)
    assert value == pytest.approx(0.96376, abs=1e-4)
    standard = point_value("power_correction_factor", {"derived.air_density": ([0], [1.15673])}, 0)
    assert standard == pytest.approx(1.0, abs=1e-4)


def test_air_fuel_ratio_is_lambda_times_the_stoichiometric_ratio():
    assert point_value("air_fuel_ratio", {"did.COMMANDED_EQUIV_RATIO": ([0], [1.0])}, 0) == (
        pytest.approx(14.7)
    )
    assert point_value("air_fuel_ratio", {"did.COMMANDED_EQUIV_RATIO": ([0], [0.9])}, 0) == (
        pytest.approx(13.23)
    )


def test_fuel_rate_from_air_mass_flow_and_afr():
    # 2 g/s = 7.2 kg/h of air; AFR 14.7; fuel density 0.745 kg/L -> 0.65744 L/h
    feeds = {
        "did.MAF_RATE": ([0], [2.0]),
        "did.COMMANDED_EQUIV_RATIO": ([0], [1.0]),
    }
    assert point_value("fuel_rate_maf", feeds, 0) == pytest.approx(0.65744, abs=1e-5)
    with pytest.raises(InputInvalid):
        point_value("fuel_rate_maf", {**feeds, "did.COMMANDED_EQUIV_RATIO": ([0], [0.0])}, 0)


# --- sensors ------------------------------------------------------------------------------------


def test_imu_acceleration_is_converted_from_g_to_mps2():
    source = resolve_sensor("imu_accel_x")
    assert (source.file, source.column) == ("imu", "ax_g")
    assert source.factor == STANDARD_GRAVITY_MPS2
    assert resolve_sensor("imu_gyro_z").column == "gz_dps"
    assert resolve_sensor("imu_gyro_z").factor == 1.0


# --- part 2: fuel economy -------------------------------------------------------------------

RATE = "did.ENGINE_FUEL_RATE"


def _flag(signal: str, feeds: dict, t: int) -> str:
    with pytest.raises(InputInvalid) as exc:
        point_value(signal, feeds, t)
    return exc.value.flag


def test_fuel_consumption_instant_is_rate_over_speed():
    feeds = {RATE: ([0], [2.0]), SPEED: ([0], [50.0])}  # 2 L/h at 50 km/h
    assert point_value("fuel_consumption_instant", feeds, 0) == pytest.approx(4.0)
    assert _flag("fuel_consumption_instant", {RATE: ([0], [2.0]), SPEED: ([0], [0.0])}, 0) == (
        "standstill"
    )
    assert _flag("fuel_consumption_instant", {RATE: ([0], [2.0]), SPEED: ([0], [None])}, 0) == (
        "invalid_input"
    )
    assert _flag("fuel_consumption_instant", {RATE: ([0], [2.0]), SPEED: ([0], [50.0])}, 9000) == (
        "stale_input"
    )


def test_fuel_consumption_avg_is_the_ratio_of_the_window_means():
    times = [0, 1000, 2000]
    feeds = {RATE: (times, [1.0, 2.0, 3.0]), SPEED: (times, [40.0, 50.0, 60.0])}
    assert point_value("fuel_consumption_avg", feeds, 2000) == pytest.approx(4.0)  # 100 * 2 / 50
    # samples older than window_ms (60 s) are left out
    old = {RATE: ([0, 61000, 62000], [9.0, 2.0, 2.0]), SPEED: ([0, 61000, 62000], [9, 50.0, 50.0])}
    assert point_value("fuel_consumption_avg", old, 62000) == pytest.approx(4.0)


def test_fuel_consumption_avg_flags():
    one = {RATE: ([0, 1000], [1.0, 2.0]), SPEED: ([1000], [50.0])}
    assert _flag("fuel_consumption_avg", one, 1000) == "insufficient_samples"
    one = {RATE: ([1000], [2.0]), SPEED: ([0, 1000], [50.0, 50.0])}
    assert _flag("fuel_consumption_avg", one, 1000) == "insufficient_samples"
    still = {RATE: ([0, 1000], [1.0, 2.0]), SPEED: ([0, 1000], [0.0, 0.0])}
    assert _flag("fuel_consumption_avg", still, 1000) == "standstill"


def test_fuel_range_from_level_tank_and_average_consumption():
    feeds = {"did.FUEL_LEVEL": ([0], [50.0]), "derived.fuel_consumption_avg": ([0], [4.0])}
    # 50 % of the tank / 4 L per 100 km
    assert point_value("fuel_range", feeds, 0) == pytest.approx(0.5 * TANK_L / 4.0 * 100.0)
    assert point_value("fuel_range", feeds, 0) == pytest.approx(150.0)  # 12 L tank
    zero = {"did.FUEL_LEVEL": ([0], [50.0]), "derived.fuel_consumption_avg": ([0], [0.0])}
    assert _flag("fuel_range", zero, 0) == "no_consumption"


def test_bsfc_is_fuel_mass_rate_over_power():
    feeds = {RATE: ([0], [2.0]), "derived.engine_power": ([0], [5.0])}
    # 2 L/h * 0.745 kg/L = 1.49 kg/h = 1490 g/h over 5 kW
    assert point_value("bsfc", feeds, 0) == pytest.approx(2.0 * FUEL_DENSITY * 1000.0 / 5.0)
    assert point_value("bsfc", feeds, 0) == pytest.approx(298.0)
    assert _flag("bsfc", {RATE: ([0], [2.0]), "derived.engine_power": ([0], [0.0])}, 0) == (
        "no_power"
    )
    assert _flag("bsfc", {RATE: ([0], [2.0]), "derived.engine_power": ([0], [-1.0])}, 0) == (
        "no_power"
    )


def test_fuel_per_stroke_is_one_injection_per_two_revolutions():
    feeds = {RATE: ([0], [2.0]), RPM: ([0], [3000.0])}
    # 1.49e6 mg/h over 3000 * 60 / 2 = 90000 injections/h
    assert point_value("fuel_per_stroke", feeds, 0) == pytest.approx(16.5556, abs=1e-4)
    assert _flag("fuel_per_stroke", {RATE: ([0], [2.0]), RPM: ([0], [0.0])}, 0) == "engine_off"


def test_total_fuel_trim_is_the_sum_of_both_trims():
    feeds = {
        "did.SHORT_TERM_FUEL_TRIM_B1": ([0], [2.5]),
        "did.LONG_TERM_FUEL_TRIM_B1": ([0], [-4.0]),
    }
    assert point_value("total_fuel_trim", feeds, 0) == pytest.approx(-1.5)
    assert _flag(
        "total_fuel_trim", {**feeds, "did.LONG_TERM_FUEL_TRIM_B1": ([0], [1.0])}, 9000
    ) == ("stale_input")


# --- part 2: redundant channels -----------------------------------------------------------------


def test_throttle_channel_residual_is_the_largest_pairwise_difference():
    feeds = {
        "did.THROTTLE_POS": ([0], [10.0]),
        "did.ABSOLUTE_THROTTLE_POS_B": ([0], [12.0]),
        "did.RELATIVE_THROTTLE_POS": ([0], [15.0]),
    }
    assert point_value("throttle_channel_residual", feeds, 0) == pytest.approx(5.0)
    feeds["did.ABSOLUTE_THROTTLE_POS_B"] = ([0], [None])
    assert _flag("throttle_channel_residual", feeds, 0) == "invalid_input"


def test_throttle_command_and_grip_residuals():
    cmd = {"did.COMMANDED_THROTTLE": ([0], [20.0]), "did.THROTTLE_POS": ([0], [15.0])}
    assert point_value("throttle_command_residual", cmd, 0) == pytest.approx(5.0)
    grip = {"did.ACCEL_PEDAL_POS_D": ([0], [30.0]), "did.RELATIVE_ACCEL_PEDAL_POS": ([0], [28.0])}
    assert point_value("grip_channel_residual", grip, 0) == pytest.approx(2.0)
    grip["did.RELATIVE_ACCEL_PEDAL_POS"] = ([0], [31.0])
    assert point_value("grip_channel_residual", grip, 0) == pytest.approx(-1.0)


# --- part 2: altitude and grade -----------------------------------------------------------------

BARO = "did.BAROMETRIC_PRESSURE"
IAT = "did.INTAKE_AIR_TEMP"


def _altitude(baro: tuple, iat: tuple, grid: list[int]):
    return FORMULAS["altitude_change"].fn(env_for("altitude_change", {BARO: baro, IAT: iat}), grid)


def test_altitude_change_is_the_hypsometric_height_over_the_first_pressure():
    baro = ([0, 5000, 10000], [101.325, 101.325, 100.0])
    iat = ([0, 5000, 10000], [15.0, 15.0, 15.0])
    out = _altitude(baro, iat, [0, 5000, 10000])
    # 287.058 * 288.15 / 9.80665 * ln(101.325 / 100) = 111.025 m
    assert out[0] == (pytest.approx(0.0), None) and out[1] == (pytest.approx(0.0), None)
    assert out[2][0] == pytest.approx(111.025, abs=1e-3) and out[2][1] is None
    # a lower pressure than the reference means a climb, a higher one a descent
    down = _altitude(([0, 5000], [100.0, 101.325]), ([0, 5000], [15.0, 15.0]), [5000])
    assert down[0][0] == pytest.approx(-111.025, abs=1e-3)  # reference p0 = 100.0 kPa


def test_altitude_change_flags_stale_and_invalid_inputs():
    baro = ([0, 5000, 10000], [101.325, None, 100.0])
    iat = ([0, 5000], [15.0, 15.0])
    out = _altitude(baro, iat, [5000, 30000, 40000])
    assert out[0] == (None, "invalid_input")  # barometric sample at 5000 is invalid
    assert out[1] == (None, "stale_input")  # both inputs older than stale_after_ms (15000)
    assert out[2] == (None, "stale_input")
    # the reference is the first valid sample, not the first sample
    skip = _altitude(([0, 5000], [None, 100.0]), ([0, 5000], [15.0, 15.0]), [5000])
    assert skip[0] == (pytest.approx(0.0), None)


def test_road_grade_is_altitude_over_distance_in_the_window():
    feeds = {
        "derived.altitude_change": ([5000, 10000, 25000], [0.0, 1.0, 4.0]),
        "derived.trip_distance": ([5000, 25000], [0.0, 0.2]),
    }
    # 4 m climbed over 0.2 km = 200 m of road
    assert point_value("road_grade", feeds, 26000) == pytest.approx(2.0)
    down = {**feeds, "derived.altitude_change": ([5000, 25000], [0.0, -3.0])}
    assert point_value("road_grade", down, 26000) == pytest.approx(-1.5)


def test_road_grade_flags_and_leaves_the_imu_input_unused():
    one = {
        "derived.altitude_change": ([25000], [4.0]),
        "derived.trip_distance": ([5000, 25000], [0.0, 0.2]),
    }
    assert _flag("road_grade", one, 26000) == "insufficient_samples"
    one = {
        "derived.altitude_change": ([5000, 25000], [0.0, 4.0]),
        "derived.trip_distance": ([25000], [0.2]),
    }
    assert _flag("road_grade", one, 26000) == "insufficient_samples"
    still = {
        "derived.altitude_change": ([5000, 25000], [0.0, 0.0]),
        "derived.trip_distance": ([5000, 25000], [0.2, 0.2]),
    }
    assert _flag("road_grade", still, 26000) == "standstill"
    # no IMU feed is needed by the formula itself (declared as context only)
    assert "sensor.imu_accel_x" in vehicle_derived.SIGNALS["road_grade"]["inputs"]


# --- part 2: thermal rates and lean cross-check -------------------------------------------------


def test_coolant_rate_is_the_slope_in_degrees_per_minute():
    times = [0, 800, 1600, 2400]
    feeds = {"did.COOLANT_TEMP": (times, [80.0 + 0.1 * t / 1000 for t in times])}  # 0.1 degC/s
    assert point_value("coolant_rate", feeds, 2400) == pytest.approx(6.0)
    assert _flag("coolant_rate", {"did.COOLANT_TEMP": ([2400], [90.0])}, 2400) == (
        "insufficient_samples"
    )


def test_oil_temp_rate_is_the_slope_in_degrees_per_minute():
    times = [0, 5000, 10000]
    feeds = {"did.ENGINE_OIL_TEMP": (times, [90.0 - 0.05 * t / 1000 for t in times])}
    assert point_value("oil_temp_rate", feeds, 10000) == pytest.approx(-3.0)
    assert _flag("oil_temp_rate", {"did.ENGINE_OIL_TEMP": ([10000], [90.0])}, 10000) == (
        "insufficient_samples"
    )


def test_lean_crosscheck_residual_is_the_lean_angle_minus_the_accel_angle():
    feeds = {"sensor.lean_angle": ([0], [20.0]), "derived.lateral_accel": ([0], [3.5])}
    # atan2(3.5, 9.80665) = 19.6415 deg
    assert point_value("lean_crosscheck_residual", feeds, 0) == pytest.approx(0.35848, abs=1e-5)
    left = {"sensor.lean_angle": ([0], [-20.0]), "derived.lateral_accel": ([0], [-3.5])}
    assert point_value("lean_crosscheck_residual", left, 0) == pytest.approx(-0.35848, abs=1e-5)
    stale = {"sensor.lean_angle": ([0], [20.0]), "derived.lateral_accel": ([0], [3.5])}
    assert _flag("lean_crosscheck_residual", stale, 5000) == "stale_input"
    assert resolve_sensor("lean_angle") is None  # not stored: missing_inputs in a real session


# --- part 2: stop_event -------------------------------------------------------------------------


def _stops(speed: tuple, rpm: tuple):
    return FORMULAS["stop_event"].fn(env_for("stop_event", {SPEED: speed, RPM: rpm}))


def _speed_profile(times: list[int], moving: set[range]) -> tuple:
    return times, [KMH_36 if any(t in r for r in moving) else 0.0 for t in times]


def test_stop_event_duration_is_start_to_the_first_moving_sample():
    times = list(range(0, 3001, 100))
    speed = _speed_profile(times, {range(0, 1100), range(2100, 3001)})  # stopped 1100..2000
    assert _stops(speed, const_series(0, 3000, 100, 1000.0)) == [(2100, 1.0, None)]
    two = _speed_profile(times, {range(0, 500), range(800, 1500), range(2500, 3001)})
    assert _stops(two, const_series(0, 3000, 100, 1000.0)) == [(800, 0.3, None), (2500, 1.0, None)]


def test_stop_event_needs_the_engine_running_at_the_start():
    times = list(range(0, 3001, 100))
    speed = _speed_profile(times, {range(0, 1100), range(2100, 3001)})
    assert _stops(speed, const_series(0, 3000, 100, 0.0)) == []
    # the engine starts only during the stop: it did not come to a stop running
    rpm = ramp(0, 3000, 100, lambda t: 0.0 if t < 1500 else 900.0)
    assert _stops(speed, rpm) == []


def test_stop_event_needs_a_moving_sample_before_the_stop():
    # standing with the engine running at the start of the data is no stop
    times = list(range(0, 3001, 100))
    standing_first = _speed_profile(times, {range(1500, 2000), range(2500, 3001)})
    assert _stops(standing_first, const_series(0, 3000, 100, 1000.0)) == [(2500, 0.5, None)]


def test_stop_event_is_dropped_across_a_gap_in_the_speed_series():
    limit = vehicle_cl250.DID_STALE_AFTER_MS["VEHICLE_SPEED"]  # 300 ms (D-087)
    rpm = const_series(0, 4000, 100, 1000.0)
    gap = ([900, 1000, 1100, 2000, 2100], [KMH_36, 0.0, 0.0, KMH_36, KMH_36])
    assert _stops(gap, rpm) == []
    edge = ([900, 1000, 1000 + limit, 1000 + 2 * limit], [KMH_36, 0.0, 0.0, KMH_36])
    assert _stops(edge, rpm) == [(1000 + 2 * limit, 2 * limit / 1000, None)]
    # a gap just before the stop: the moving state is not followed across it either
    before = ([900, 901 + limit, 1000 + limit, 1100 + limit], [KMH_36, 0.0, 0.0, KMH_36])
    assert _stops(before, rpm) == []


def test_stop_event_judges_the_speed_by_its_own_limit_not_three_polls():
    # D-087: 3 x poll (330 ms) is within the old rule but beyond the DID's own 300 ms
    own = vehicle_cl250.DID_STALE_AFTER_MS["VEHICLE_SPEED"]
    three_polls = 3 * vehicle_catalog.SIGNALS["VEHICLE_SPEED"]["poll_period_ms"]
    assert own < three_polls
    rpm = const_series(0, 4000, 100, 1000.0)
    gap = three_polls
    stop = ([900, 1000, 1000 + gap, 1000 + 2 * gap], [KMH_36, 0.0, 0.0, KMH_36])
    assert _stops(stop, rpm) == []


def test_stop_event_still_open_at_the_end_yields_nothing():
    times = list(range(0, 2001, 100))
    speed = _speed_profile(times, {range(0, 1100)})
    assert _stops(speed, const_series(0, 2000, 100, 1000.0)) == []


def test_stop_event_through_the_engine_has_no_grid_and_checks_the_range():
    times = list(range(0, 3001, 100))
    inputs = {
        SPEED: _speed_profile(times, {range(0, 1100), range(2100, 3001)}),
        RPM: const_series(0, 3000, 100, 1000.0),
    }
    out, evaluation = _outputs(inputs, "stop_event")
    assert (out.times, out.values, out.flags) == ([2100], [1.0], [None])
    entry = evaluation.coverage["stop_event"]
    assert entry["status"] == "computed" and entry["rows"] == 1 and "period_ms" not in entry
    # the engine applies the defs range to events too: a stop longer than 3600 s
    long_times = [0, *range(100, 3700_000, 300)]
    n = len(long_times)
    end = long_times[-1] + 300
    long_speed = (long_times + [end], [KMH_36] + [0.0] * (n - 1) + [KMH_36])
    long_rpm = (long_times + [end], [1000.0] * (n + 1))
    out, evaluation = _outputs({SPEED: long_speed, RPM: long_rpm}, "stop_event")
    assert out.values == [None] and out.flags == ["out_of_range"]
    assert evaluation.coverage["stop_event"]["flag_counts"] == {"out_of_range": 1}
