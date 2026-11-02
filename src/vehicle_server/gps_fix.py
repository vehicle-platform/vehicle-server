"""The GNSS fix usability rule shared by /track and the class-B GNSS distance (D-060).

The fix-type enum names and values come from the BLE schema; this module holds no vehicle
facts of its own. A fix is usable when its defs enum name is fix2d, fix3d or
gnssDeadReckoning, and only while the latest fix-type sample is fresh by its
``stale_after_ms``.
"""

from __future__ import annotations

from vehicle_server.defs import ble as defs_ble

USABLE_FIX_NAMES = ("fix2d", "fix3d", "gnssDeadReckoning")


def fix_values_by_name() -> dict[str, float]:
    gps = defs_ble.SCHEMA["gpsBlock"]
    return {entry["name"]: float(entry["value"]) for entry in gps["fixType"]["values"]}


def usable_fix_values() -> frozenset[float]:
    by_name = fix_values_by_name()
    missing = [name for name in USABLE_FIX_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"gps fixType enum has no {', '.join(missing)}")
    return frozenset(by_name[name] for name in USABLE_FIX_NAMES)
