"""Walker spawn presets for interactive simulation (trail24 map).

Default for `main.py` when SIM_SPAWN_PRESETS=eval.spawn_presets (the default).
For thesis batch scenarios S1–S6, use eval_sim/spawn_presets.py instead.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Hardcoded trail24 crossing polygons (world XY, meters). Shared with eval_sim/spawn_presets.py.
CROSSING_ZONE_1 = [
    (-115.0, -159.9),
    (-109.4, -159.9),
    (-109.19, -155.15),
    (-114.49, -155.15),
]
CROSSING_ZONE_2 = [
    (-106.4, -159.9),
    (-101.6, -159.9),
    (-102.29, -155.15),
    (-106.99, -155.15),
]

SPAWN_PRESETS: Dict[str, Dict[str, Any]] = {
    "on_zone": {
        "x": -112.0,
        "y": -157.5,
        "yaw": 90.0,
        "speed_ms": 0.0,
        "disable_spawn": False,
        "zone": 1,
    },
    "zone_2": {
        "x": -104.0,
        "y": -157.5,
        "yaw": 90.0,
        "speed_ms": 0.0,
        "disable_spawn": False,
        "zone": 2,
    },
    "farside": {
        "x": -90.42,
        "y": -158.08,
        "yaw": 180.0,
        "speed_ms": 1.2,
        "disable_spawn": False,
        "zone": None,
    },
    "nearside_off": {
        "x": -100.26,
        "y": -156.91,
        "yaw": 180.0,
        "speed_ms": 1.2,
        "disable_spawn": False,
        "zone": None,
    },
    "wait_curb": {
        "x": -100.26,
        "y": -156.91,
        "yaw": 180.0,
        "speed_ms": 0.0,
        "disable_spawn": False,
        "zone": None,
    },
    "group_walking": {
        "x": -105.5,
        "y": -156.3,
        "yaw": 90.0,
        "speed_ms": 1.2,
        "disable_spawn": False,
        "zone": None,
        "extra_walkers": [
            {"x": -103.0, "y": -156.5, "yaw": 90.0, "speed_ms": 1.2},
            {"x": -101.0, "y": -156.7, "yaw": 90.0, "speed_ms": 1.2},
            {"x": -98.5, "y": -156.4, "yaw": 90.0, "speed_ms": 1.2},
        ],
    },
    "both_sides": {
        "x": -100.26,
        "y": -156.91,
        "yaw": 180.0,
        "speed_ms": 1.2,
        "disable_spawn": False,
        "zone": None,
        "extra_walkers": [
            {"x": -90.42, "y": -158.08, "yaw": 0.0, "speed_ms": 1.2},
        ],
    },
    "none": {
        "x": None,
        "y": None,
        "yaw": None,
        "speed_ms": 0.0,
        "disable_spawn": True,
        "zone": None,
    },
}


def _point_in_polygon(x: float, y: float, polygon) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def point_in_zone(x: float, y: float, zone_index: int) -> bool:
    poly = CROSSING_ZONE_1 if zone_index == 1 else CROSSING_ZONE_2
    return _point_in_polygon(x, y, poly)


def resolve_spawn_config(
    preset_name: Optional[str],
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return walker spawn dict consumed by main.py (primary + optional extra_walkers).

    Explicit override keys: x, y, yaw, speed_ms, disable_spawn.
    Env-style keys also accepted: WALKER_SPAWN_X, WALKER_SPAWN_Y, etc.
    """
    overrides = dict(overrides or {})
    name = (preset_name or "nearside_off").strip().lower()
    if name not in SPAWN_PRESETS:
        raise ValueError(f"Unknown spawn preset: {preset_name!r}; choose from {sorted(SPAWN_PRESETS)}")

    base = dict(SPAWN_PRESETS[name])
    base["preset"] = name

    env_map = {
        "WALKER_SPAWN_X": "x",
        "WALKER_SPAWN_Y": "y",
        "WALKER_SPAWN_YAW": "yaw",
        "WALKER_SPEED_MS": "speed_ms",
        "DISABLE_WALKER_SPAWN": "disable_spawn",
    }
    for ek, bk in env_map.items():
        if ek in overrides and overrides[ek] is not None:
            val = overrides[ek]
            if bk == "disable_spawn":
                base[bk] = str(val).strip().lower() in ("1", "true", "yes")
            elif bk in ("x", "y", "yaw", "speed_ms"):
                base[bk] = float(val)
        elif bk in overrides and overrides[bk] is not None:
            base[bk] = overrides[bk]

    if overrides.get("speed_ms") is not None:
        base["speed_ms"] = float(overrides["speed_ms"])

    if base.get("disable_spawn"):
        return {
            "preset": name,
            "x": None,
            "y": None,
            "yaw": None,
            "speed_ms": 0.0,
            "disable_spawn": True,
            "zone": None,
            "extra_walkers": [],
        }

    extra = []
    for w in base.get("extra_walkers") or []:
        extra.append({
            "x": float(w["x"]),
            "y": float(w["y"]),
            "yaw": float(w["yaw"]),
            "speed_ms": float(w.get("speed_ms", base.get("speed_ms", 1.2))),
        })

    return {
        "preset": name,
        "x": float(base["x"]),
        "y": float(base["y"]),
        "yaw": float(base["yaw"]),
        "speed_ms": float(base.get("speed_ms", 1.2)),
        "disable_spawn": False,
        "zone": base.get("zone"),
        "extra_walkers": extra,
    }
