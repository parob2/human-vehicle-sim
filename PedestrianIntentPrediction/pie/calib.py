"""Monocular ego-relative Δx/Δy from PIE bbox + dashcam calibration."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from pie.config_pie import DEFAULT_PED_HEIGHT_M, PIE_ROOT

_CALIB_CACHE: Dict[str, float] | None = None


def load_calibration(calib_path: Path | None = None) -> Dict[str, float]:
    global _CALIB_CACHE
    if _CALIB_CACHE is not None:
        return _CALIB_CACHE

    path = calib_path or (PIE_ROOT / "camera_params" / "calibration_data.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    k = raw["K"]
    _CALIB_CACHE = {
        "fx": float(k[0][0]),
        "fy": float(k[1][1]),
        "cx": float(k[0][2]),
        "cy": float(k[1][2]),
        "cam_height_m": float(raw["cam_height_mm"]) / 1000.0,
        "cam_pitch_deg": float(raw["cam_pitch_deg"]),
        "img_w": float(raw["dim"][0]),
        "img_h": float(raw["dim"][1]),
    }
    return _CALIB_CACHE


def _bbox_bottom_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, float(y2)


def bbox_to_ego_delta_xy(
    bbox: Tuple[float, float, float, float],
    *,
    calib: Dict[str, float] | None = None,
    ped_height_m: float = DEFAULT_PED_HEIGHT_M,
) -> Tuple[float, float]:
    """
    Estimate ego-frame (Δx forward, Δy lateral) in meters.

    Uses pinhole depth from bbox height (typical when Mobileye ranging is unavailable)
    and refines lateral offset with the bottom-center pixel ray.
    """
    cal = calib or load_calibration()
    x1, y1, x2, y2 = bbox
    bbox_h = max(abs(y2 - y1), 1.0)
    u, v = _bbox_bottom_center(bbox)

    # Depth along optical axis from apparent pedestrian height.
    depth = (ped_height_m * cal["fy"]) / bbox_h

    # Lateral offset at that depth (camera x-right, vehicle y-left).
    lateral = (u - cal["cx"]) * depth / cal["fx"]

    pitch = math.radians(cal["cam_pitch_deg"])
    # Project optical-axis depth to ground-plane forward distance.
    delta_x = depth * math.cos(pitch) + cal["cam_height_m"] * math.sin(-pitch)
    delta_x = max(delta_x, 0.5)

    # Sign: positive lateral = pedestrian left of image center (ego-left).
    delta_y = -lateral

    return float(delta_x), float(delta_y)


def bbox_height_depth_m(
    bbox: Tuple[float, float, float, float],
    *,
    calib: Dict[str, float] | None = None,
    ped_height_m: float = DEFAULT_PED_HEIGHT_M,
) -> float:
    cal = calib or load_calibration()
    bbox_h = max(abs(bbox[3] - bbox[1]), 1.0)
    return float((ped_height_m * cal["fy"]) / bbox_h)
