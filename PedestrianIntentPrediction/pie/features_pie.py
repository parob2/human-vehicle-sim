"""Build the paper's 7-dimensional Ma-Rong feature vector on PIE samples."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from config import ALPHA_MAX_RAD, ALPHA_MIN_RAD, ANGLE_BIN_DEG, THETA_MAX_RAD, THETA_MIN_RAD
from features_ma_rong import bin_angle_deg, calf_ground_angle_rad, thigh_calf_angle_rad

from pie.calib import bbox_to_ego_delta_xy
from pie.config_pie import FEATURE_DIM_PIE, FEATURE_DIM_PIE_HEADING, FEATURE_NAMES_PIE, FEATURE_NAMES_PIE_HEADING


def compute_pose_angles_deg_from_joints(joints: Dict[str, Tuple[float, float]]) -> Dict[str, Optional[float]]:
    hip_l = joints["hip_l"]
    hip_r = joints["hip_r"]
    knee_l = joints["knee_l"]
    knee_r = joints["knee_r"]
    ankle_l = joints["ankle_l"]
    ankle_r = joints["ankle_r"]

    return {
        "alpha_L_deg": bin_angle_deg(thigh_calf_angle_rad(hip_l, knee_l, ankle_l)),
        "alpha_R_deg": bin_angle_deg(thigh_calf_angle_rad(hip_r, knee_r, ankle_r)),
        "theta_L_deg": bin_angle_deg(calf_ground_angle_rad(knee_l, ankle_l)),
        "theta_R_deg": bin_angle_deg(calf_ground_angle_rad(knee_r, ankle_r)),
    }


def build_pie_feature_vector(
    *,
    bbox: Tuple[float, float, float, float],
    obd_speed_kmh: float,
    joints: Optional[Dict[str, Tuple[float, float]]] = None,
    impute_invalid_angles: bool = True,
    default_angle_deg: float = 120.0,
) -> Optional[List[float]]:
    delta_x, delta_y = bbox_to_ego_delta_xy(bbox)

    if joints is None:
        return None

    angles = compute_pose_angles_deg_from_joints(joints)
    angle_vals = [angles[k] for k in ("alpha_L_deg", "alpha_R_deg", "theta_L_deg", "theta_R_deg")]

    if impute_invalid_angles:
        angle_vals = [v if v is not None else default_angle_deg for v in angle_vals]
    elif any(v is None for v in angle_vals):
        return None

    # Paper order: Δy lateral, Δx longitudinal, v_ego, α_L, α_R, θ_L, θ_R
    return [
        float(delta_y),
        float(delta_x),
        float(max(obd_speed_kmh, 0.0)),
        float(angle_vals[0]),
        float(angle_vals[1]),
        float(angle_vals[2]),
        float(angle_vals[3]),
    ]


def _feet_midpoint_from_joints(joints: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    al = joints["ankle_l"]
    ar = joints["ankle_r"]
    return (al[0] + ar[0]) / 2.0, (al[1] + ar[1]) / 2.0


def trajectory_heading_sin_cos_from_joints(
    joints: Dict[str, Tuple[float, float]],
    prev_joints: Optional[Dict[str, Tuple[float, float]]],
) -> Tuple[float, float]:
    """Image-space heading from ankle-midpoint displacement between frames."""
    mx_t, my_t = _feet_midpoint_from_joints(joints)
    if prev_joints is None:
        return 0.0, 0.0
    mx_p, my_p = _feet_midpoint_from_joints(prev_joints)
    dx, dy = mx_t - mx_p, my_t - my_p
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return 0.0, 0.0
    ang = math.atan2(dy, dx)
    return math.sin(ang), math.cos(ang)


def trajectory_heading_sin_cos_from_bbox(
    bbox: Tuple[float, float, float, float],
    prev_bbox: Optional[Tuple[float, float, float, float]],
) -> Tuple[float, float]:
    """BBox foot-point proxy when pose joints are unavailable in cache."""
    x1, y1, x2, y2 = bbox
    feet_x, feet_y = (x1 + x2) / 2.0, y2
    if prev_bbox is None:
        return 0.0, 0.0
    px1, py1, px2, py2 = prev_bbox
    prev_feet_x, prev_feet_y = (px1 + px2) / 2.0, py2
    dx, dy = feet_x - prev_feet_x, feet_y - prev_feet_y
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return 0.0, 0.0
    ang = math.atan2(dy, dx)
    return math.sin(ang), math.cos(ang)


def build_pie_heading_feature_vector(
    *,
    bbox: Tuple[float, float, float, float],
    obd_speed_kmh: float,
    joints: Optional[Dict[str, Tuple[float, float]]] = None,
    prev_joints: Optional[Dict[str, Tuple[float, float]]] = None,
    prev_bbox: Optional[Tuple[float, float, float, float]] = None,
    impute_invalid_angles: bool = True,
    default_angle_deg: float = 120.0,
) -> Optional[List[float]]:
    """Paper 7 features plus trajectory heading sin/cos after v_ego."""
    base = build_pie_feature_vector(
        bbox=bbox,
        obd_speed_kmh=obd_speed_kmh,
        joints=joints,
        impute_invalid_angles=impute_invalid_angles,
        default_angle_deg=default_angle_deg,
    )
    if base is None:
        return None

    if joints is not None:
        h_sin, h_cos = trajectory_heading_sin_cos_from_joints(joints, prev_joints)
    else:
        h_sin, h_cos = trajectory_heading_sin_cos_from_bbox(bbox, prev_bbox)

    return [
        base[0],
        base[1],
        base[2],
        float(h_sin),
        float(h_cos),
        base[3],
        base[4],
        base[5],
        base[6],
    ]


def feature_dim() -> int:
    return FEATURE_DIM_PIE


def feature_dim_heading() -> int:
    return FEATURE_DIM_PIE_HEADING


def feature_names() -> List[str]:
    return list(FEATURE_NAMES_PIE)


def feature_names_heading() -> List[str]:
    return list(FEATURE_NAMES_PIE_HEADING)
