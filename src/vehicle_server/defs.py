"""Bridge to the vehicle-defs submodule (external/vehicle-defs).

vehicle-defs is the single source of truth for signals (CAN IDs, DIDs,
scales, VSS paths). This repo never copies or vendors its generated code
(platform rule #2); instead it adds the submodule's ``gen/python`` directory
to ``sys.path`` and imports ``vehicle_defs`` from there. ``gen/python`` ships as
a plain package with no ``pyproject.toml`` of its own, so a path injection is
the simplest approach that also works unmodified in Docker and CI, where the
submodule is checked out at the same relative location.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFS_ROOT = _REPO_ROOT / "external" / "vehicle-defs"
_DEFS_PYTHON = _DEFS_ROOT / "gen" / "python"
_FALLBACK_VERSION_FILE = Path(__file__).resolve().parent / "defs_pin.txt"

if _DEFS_PYTHON.is_dir() and str(_DEFS_PYTHON) not in sys.path:
    sys.path.insert(0, str(_DEFS_PYTHON))

try:
    import vehicle_defs  # noqa: E402
    from vehicle_defs import (  # noqa: E402
        ble,
        uds,
        vehicle_catalog,
        vehicle_cl250,
        vehicle_constants,
        vehicle_derived,
    )
except ImportError as exc:  # pragma: no cover - fails fast at startup
    raise ImportError(
        "vehicle_defs could not be imported from "
        f"{_DEFS_PYTHON}. Did you run "
        "`git submodule update --init --recursive`?"
    ) from exc


def defs_version() -> str:
    """Version of the pinned vehicle-defs submodule.

    Tries ``git describe --tags`` on the submodule checkout first (accurate
    for any local checkout, including a moved tag); falls back to the
    constant recorded in ``defs_pin.txt`` (this repo's own file, updated
    whenever the submodule pin changes) when git metadata is unavailable,
    e.g. in a Docker image built without the ``.git`` directory.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(_DEFS_ROOT), "describe", "--tags", "--always"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        version = result.stdout.strip()
        if version:
            return version
    except (OSError, subprocess.CalledProcessError):
        pass

    try:
        return _FALLBACK_VERSION_FILE.read_text().strip() or "unknown"
    except OSError:
        return "unknown"


__all__ = [
    "vehicle_defs",
    "ble",
    "uds",
    "vehicle_catalog",
    "vehicle_cl250",
    "vehicle_constants",
    "vehicle_derived",
    "defs_version",
]
