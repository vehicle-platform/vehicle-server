"""Class-B (per-ride) golden vectors for rt-core (C3 part 2a)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from vehicle_server.defs import vehicle_cl250
from vehicle_server.derived import golden, golden_b


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> tuple[golden.Generation, Path]:
    out = tmp_path_factory.mktemp("vectors_b")
    return golden.write(out, None), out


@pytest.fixture(scope="module")
def class_b() -> golden_b.ClassBGeneration:
    return golden_b.generate_b()


def test_files_are_in_the_manifest_set(generated):
    gen, _ = generated
    names = {f.name for f in gen.files}
    assert golden_b.TYPES_HEADER_B in names
    assert {f"derived_{s.name}_b.h" for s in golden_b.b_scenarios()} <= names


def test_selection_and_skips_are_complete(class_b):
    from vehicle_server.defs import vehicle_derived

    classes_b = {n for n, s in vehicle_derived.SIGNALS.items() if s["class"] == "B"}
    assert set(class_b.selected) | set(class_b.skipped) == classes_b
    assert not set(class_b.selected) & set(class_b.skipped)
    assert class_b.selected
    for name in class_b.selected:
        spec = vehicle_derived.SIGNALS[name]
        assert not spec.get("bins")
        assert not any(i.startswith("sensor.") or "gps" in i.lower() for i in spec["inputs"])


def test_health_window_signals_are_never_selected(class_b):
    from vehicle_server.defs import vehicle_derived

    for name, spec in vehicle_derived.SIGNALS.items():
        if spec.get("window"):
            assert name not in class_b.selected
            assert class_b.skipped[name] == golden_b.SKIP_HEALTH_WINDOW


def test_no_gps_in_class_b_files(class_b):
    for f in class_b.files:
        code = re.sub(r"/\*.*?\*/", "", f.text, flags=re.S)
        assert "gps" not in code.lower()


def test_count_macros_match_the_arrays(generated):
    _, out = generated
    for sc in golden_b.b_scenarios():
        n, upper = sc.name, sc.name.upper()
        text = (out / f"derived_{n}_b.h").read_text()
        kinds = [("RIDE", "rides"), ("B_OUTPUT", "b_outputs"), ("B_TOL", "b_tol")]
        if sc.own_inputs:
            kinds.append(("B_INPUT", "b_inputs"))
        for kind, array in kinds:
            count = int(re.search(rf"DERIVED_{upper}_{kind}_COUNT \((\d+)u\)", text)[1])
            block = re.search(rf"derived_{n}_{array}\[\] = \{{\n(.*?)\n\}};", text, re.S)[1]
            assert count == len(block.splitlines()) > 0, (n, array)


def test_two_ride_gap_exceeds_the_ecu_absent_timeout(generated):
    _, out = generated
    text = (out / "derived_two_rides_b.h").read_text()
    rides = [
        tuple(map(int, m))
        for m in re.findall(r"\{ (\d+)u, (\d+)u, (\d+)u \},", text.split("_b_")[0])
    ]
    assert [r[0] for r in rides] == [0, 1]
    assert rides[1][1] - rides[0][2] > vehicle_cl250.ECU_ABSENT_TIMEOUT_MS
    inputs = re.search(r"derived_two_rides_b_inputs\[\] = \{\n(.*?)\n\};", text, re.S)[1]
    times = [int(t) for t in re.findall(r"\{ (\d+)u,", inputs)]
    assert times == sorted(times)
    assert not [t for t in times if rides[0][2] <= t < rides[1][1]]  # ECU-absent stretch
    assert max(times) < rides[1][2]


def test_two_rides_are_evaluated_independently(class_b):
    first, second = class_b.ride_values["two_rides"]

    def values(rows):
        return {r.signal: (r.valid, r.flag, r.value) for r in rows}

    # Each ride equals a standalone run of the same profile (fresh state at the epoch).
    assert values(first) == values(class_b.ride_values["accel_gears"][0])
    assert values(second) == values(class_b.ride_values["edge_values"][0])
    assert values(first) != values(second)
    # Not accumulated: ride 1 distance is far below the sum, and ride 1 starts from its own
    # first coolant sample.
    by1, by2 = values(first), values(second)
    assert by2["ride_distance"][2] < by1["ride_distance"][2] + by2["ride_distance"][2]
    assert by2["ride_time"][2] <= 30.0
    assert by2["ride_stop_count"][2] == 1.0 and by1["ride_stop_count"][2] == 0.0


def test_flags_are_in_the_enum(generated):
    _, out = generated
    types = (out / golden.TYPES_HEADER).read_text()
    enum = set(re.findall(r"(DERIVED_VECTOR_FLAG_\w+) = \d+", types))
    for sc in golden_b.b_scenarios():
        used = set(
            re.findall(r"DERIVED_VECTOR_FLAG_\w+", (out / f"derived_{sc.name}_b.h").read_text())
        )
        assert used <= enum


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_class_b_headers_compile_as_c99(generated, tmp_path):
    _, out = generated
    lines = [f'#include "derived_{s.name}_b.h"' for s in golden_b.b_scenarios()]
    for s in golden_b.b_scenarios():
        n, u = s.name, s.name.upper()
        for kind, array in (("B_OUTPUT", "b_outputs"), ("B_TOL", "b_tol"), ("RIDE", "rides")):
            lines.append(
                f"typedef char check_{n}_{array}"
                f"[(sizeof(derived_{n}_{array}) / sizeof(derived_{n}_{array}[0]) "
                f"== DERIVED_{u}_{kind}_COUNT) ? 1 : -1];"
            )
    lines.append("int main(void) { return 0; }")
    source = tmp_path / "check.c"
    source.write_text("\n".join(lines) + "\n")
    result = subprocess.run(
        [
            "cc", "-std=c99", "-pedantic", "-Wall", "-Wextra", "-Werror",
            "-I", str(golden.DEFS_C_RT_CORE), "-I", str(out),
            "-c", str(source), "-o", str(tmp_path / "check.o"),
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
