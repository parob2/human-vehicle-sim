"""Walker spawn presets for thesis eval_sim scenarios (trail24 map).

Used when SIM_SPAWN_PRESETS=eval_sim.spawn_presets (set by eval_sim/run_scenarios.py).
Scenario IDs S1–S6 map to presets via eval_sim/scenarios.yaml.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

# RSU RGB camera anchor (main.py) — stationary peds face this for reliable pose.
RSU_X = -83.58
RSU_Y = -155.22
# RSU_X = -107.90
# RSU_Y = -185.20

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

# nearside_off geometry (S2 reference crossing path)
_NEAR_SIDE_X = -100.26
_NEAR_SIDE_Y = -156.91
_CROSSING_YAW = 180.0  # walk west into crossing zone 1


def yaw_toward_rsu(x: float, y: float) -> float:
    """CARLA yaw (deg): face the RSU camera."""
    return round(math.degrees(math.atan2(RSU_Y - y, RSU_X - x)), 1)


SPAWN_PRESETS: Dict[str, Dict[str, Any]] = {
    # shared base presets (main.py default crossing geometry).
    "nearside_off": {
        "x": _NEAR_SIDE_X,
        "y": _NEAR_SIDE_Y,
        "yaw": _CROSSING_YAW,
        "speed_ms": 1.2,
        "disable_spawn": False,
        "zone": None,
    },
    "farside": {
        "x": -90.42,
        "y": -158.08,
        "yaw": _CROSSING_YAW,
        "speed_ms": 1.2,
        "disable_spawn": False,
        "zone": None,
    },
    "none": {
        "x": None,
        "y": None,
        "yaw": None,
        "speed_ms": 0.0,
        "disable_spawn": True,
        "zone": None,
    },
    # thesis scenarios (eval_scenarios.odt).
    "sidewalk_walk": {
        # Walk east along sidewalk (away from crossing polygon), roughly toward RSU/camera.
        "x": -100.59,
        "y": -154.79,
        "yaw": 0.0,
        "speed_ms": 1.2,
        "disable_spawn": False,
        "zone": None,
    },
    "talking_pair": {
        "x": -100.59,
        "y": -154.79,
        "yaw": yaw_toward_rsu(-100.59, -154.79),
        "speed_ms": 0.0,
        "disable_spawn": False,
        "zone": None,
        "extra_walkers": [
            {
                "x": -100.20,
                "y": -155.29,
                "yaw": yaw_toward_rsu(-100.20, -155.29),
                "speed_ms": 0.0,
            },
        ],
    },
    "slow_crossing": {
        "x": -107.99,
        "y": -155.69,
        "yaw": 180.0,
        "speed_ms": 0.35,
        "disable_spawn": False,
        "zone": None,
    },
    "multi_cross": {
        # Primary nearside + farside + second nearside lane (all westbound into zone 1).
        "x": _NEAR_SIDE_X,
        "y": _NEAR_SIDE_Y,
        "yaw": _CROSSING_YAW,
        "speed_ms": 1.2,
        "disable_spawn": False,
        "zone": None,
        "extra_walkers": [
            {"x": -90.42, "y": -158.08, "yaw": _CROSSING_YAW, "speed_ms": 1.2},
            {"x": -100.59, "y": -155.19, "yaw": _CROSSING_YAW, "speed_ms": 1.2},
        ],
    },
    "running_cross": {
        # East-adjacent spawn closer to zone 1 than S2 (nearside_off); higher speed.
        "x": -90.64,
        "y": -154.58,
        "yaw": 180.0,
        "speed_ms": 2.8,
        "disable_spawn": False,
        "zone": None,
    },
}


def _point_in_polygon(x: float, y: float, polygon) -> bool:
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
    """Merge preset defaults with WALKER_SPAWN_* env overrides from main.py."""
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
