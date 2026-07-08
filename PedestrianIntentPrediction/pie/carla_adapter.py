"""Runtime Ma-Rong feature vectors for CARLA RSU (PIE-trained RF)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from pie.calib import bbox_to_ego_delta_xy
from pie.features_pie import (
    build_pie_feature_vector,
    build_pie_heading_feature_vector,
    feature_dim,
    feature_dim_heading,
    feature_names,
    feature_names_heading,
)

# YOLO COCO pose indices (same as sim_config.COCO_MAPPING subset)
_HIP_L, _HIP_R = 11, 12
_KNEE_L, _KNEE_R = 13, 14
_ANKLE_L, _ANKLE_R = 15, 16


def rsu_camera_calib(
    image_w: float,
    image_h: float,
    fov_deg: float,
    *,
    pitch_deg: float = -20.0,
    cam_height_m: float = 8.0,
) -> Dict[str, float]:
    """Approximate pinhole intrinsics for the fixed RSU RGB camera in main.py."""
    fov_rad = math.radians(float(fov_deg))
    fy = (image_h / 2.0) / math.tan(fov_rad / 2.0)
    fx = fy
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(image_w / 2.0),
        "cy": float(image_h / 2.0),
        "cam_height_m": float(cam_height_m),
        "cam_pitch_deg": float(pitch_deg),
        "img_w": float(image_w),
        "img_h": float(image_h),
    }


def joints_from_kpts(kpts_data, min_conf: float = 0.15) -> Optional[Dict[str, Tuple[float, float]]]:
    return _joints_from_kpts(kpts_data, min_conf=min_conf)


def _joints_from_kpts(kpts_data, min_conf: float = 0.15) -> Optional[Dict[str, Tuple[float, float]]]:
    mapping = {
        "hip_l": _HIP_L,
        "hip_r": _HIP_R,
        "knee_l": _KNEE_L,
        "knee_r": _KNEE_R,
        "ankle_l": _ANKLE_L,
        "ankle_r": _ANKLE_R,
    }
    out: Dict[str, Tuple[float, float]] = {}
    for name, idx in mapping.items():
        x, y, c = kpts_data[idx]
        if float(c) < min_conf:
            return None
        out[name] = (float(x), float(y))
    return out


def build_pie_carla_features(
    box,
    kpts_data,
    *,
    v_ego_kmh: float,
    image_w: float,
    image_h: float,
    fov_deg: float,
    cam_pitch_deg: float = -20.0,
    cam_height_m: float = 8.0,
) -> Optional[List[float]]:
    """Build 7-dim PIE Ma-Rong vector from a live RSU detection."""
    bbox = tuple(float(v) for v in box)
    joints = _joints_from_kpts(kpts_data)
    if joints is None:
        return None
    calib = rsu_camera_calib(
        image_w,
        image_h,
        fov_deg,
        pitch_deg=cam_pitch_deg,
        cam_height_m=cam_height_m,
    )
    return build_pie_feature_vector(
        bbox=bbox,
        obd_speed_kmh=float(max(v_ego_kmh, 0.0)),
        joints=joints,
        impute_invalid_angles=True,
    )


def build_pie_carla_heading_features(
    box,
    kpts_data,
    *,
    v_ego_kmh: float,
    image_w: float,
    image_h: float,
    fov_deg: float,
    cam_pitch_deg: float = -20.0,
    cam_height_m: float = 8.0,
    prev_joints: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[List[float]]:
    """Build 9-dim PIE Ma-Rong vector (7 paper features + trajectory heading).

    prev_joints: foot/keypoint positions from the previous analysis step; required
    for sin/cos trajectory heading when the track persists across frames.
    """
    bbox = tuple(float(v) for v in box)
    joints = _joints_from_kpts(kpts_data)
    if joints is None:
        return None
    calib = rsu_camera_calib(
        image_w,
        image_h,
        fov_deg,
        pitch_deg=cam_pitch_deg,
        cam_height_m=cam_height_m,
    )
    _ = calib  # calib kept for API parity with 7-feature builder
    return build_pie_heading_feature_vector(
        bbox=bbox,
        obd_speed_kmh=float(max(v_ego_kmh, 0.0)),
        joints=joints,
        prev_joints=prev_joints,
        impute_invalid_angles=True,
    )


def expected_feature_dim() -> int:
    return feature_dim()


def expected_feature_dim_heading() -> int:
    return feature_dim_heading()


def expected_feature_names() -> List[str]:
    return feature_names()


def expected_feature_names_heading() -> List[str]:
    return feature_names_heading()
