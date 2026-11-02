"""Audit that every implemented class-B summary reads every input it declares (E-26)."""

from __future__ import annotations

import pytest

from vehicle_server.defs import vehicle_derived
from vehicle_server.derived import engine, summary
from vehicle_server.derived.context import Feed
from vehicle_server.gps_fix import usable_fix_values

SPECS = vehicle_derived.SIGNALS

IMPLEMENTED = [
    name
    for name, spec in SPECS.items()
    if spec["class"] == "B"
    and name in summary.SUMMARIES
    and spec["ready_phase0"]
    and not spec.get("pending_inputs")
]


class RecordingEnv(summary.SummaryEnv):
    """SummaryEnv that records every declared input a summary asks for."""

    def __init__(self, spec, feeds, outputs, results):
        super().__init__(spec, feeds, outputs, results)
        self.accessed: set[str] = set()

    def valid(self, name: str) -> list[float]:
        self.accessed.add(name)
        return super().valid(name)

    def result(self, name: str) -> float | None:
        self.accessed.add(name)
        return super().result(name)

    def samples(self, name: str) -> list[float]:
        self.accessed.add(name)
        return super().samples(name)

    def feed(self, name: str) -> Feed:
        self.accessed.add(name)
        return super().feed(name)

    def const(self, name: str) -> float:
        self.accessed.add(name)
        return super().const(name)


def _env(name: str) -> RecordingEnv:
    spec = SPECS[name]
    feeds: dict[str, Feed] = {}
    outputs: dict[str, engine.SignalOutput] = {}
    results: dict[str, float | None] = {}
    times = list(range(0, 5001, 100))
    for inp in spec["inputs"]:
        kind, _, key = inp.partition(".")
        if kind == "derived" and SPECS[key]["class"] == "A":
            outputs[key] = engine.SignalOutput(
                key, SPECS[key]["unit"], [0, 1000, 2000], [1.0, 2.0, 3.0], [None, None, None]
            )
        elif kind == "derived":
            results[key] = 1.0
        else:
            value = next(iter(usable_fix_values())) if inp == "sensor.gps_fix_type" else 1.0
            feeds[inp] = Feed(times, [value] * len(times), 10_000)
    return RecordingEnv(spec, feeds, outputs, results)


def declared_but_unread(name: str) -> set[str]:
    env = _env(name)
    summary.SUMMARIES[name].fn(env)
    return set(env.inputs) - env.accessed


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_every_implemented_summary_reads_its_declared_inputs(name: str) -> None:
    unread = declared_but_unread(name)
    assert not unread, f"{name} declares but never reads {sorted(unread)}"
