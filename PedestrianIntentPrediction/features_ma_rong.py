"""
Ma & Rong (2022) multi-feature fusion — pose angles + RSU proxy distances.

Paper: θ = arctan((y_knee - y_ankle) / (x_knee - x_ankle))
       α = arccos((a² + b² - c²) / (2ab))
Angles are binned at 30° intervals per paper §2.3.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from config import (
    ALPHA_MAX_RAD,
    ALPHA_MIN_RAD,
    ANGLE_BIN_DEG,
    FEATURE_DIM,
    FEATURE_DIM_HEADING,
    FEATURE_NAMES,
    FEATURE_NAMES_HEADING,
    JOINT_ANKLE_L,
    JOINT_ANKLE_R,
    JOINT_HIP_L,
    JOINT_HIP_R,
    JOINT_KNEE_L,
    JOINT_KNEE_R,
    THETA_MAX_RAD,
    THETA_MIN_RAD,
)


def _joint_xy(sk: dict, name: str) -> Tuple[float, float]:
    xy = sk.get(name) or {}
    return float(xy.get("x", 0.0)), float(xy.get("y", 0.0))


def _sanitize_scalar(x: Any) -> float:
    if x is None:
        return 0.0
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(xf):
        return 0.0
    return xf


def calf_ground_angle_rad(knee: Tuple[float, float], ankle: Tuple[float, float]) -> Optional[float]:
    """θ: angle between lower leg and ground (paper Eq. 1)."""
    dx = knee[0] - ankle[0]
    dy = knee[1] - ankle[1]
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return None
    # Paper uses arctan of vertical over horizontal in image coords.
    theta = math.atan2(abs(dy), abs(dx))
    if theta <= THETA_MIN_RAD or theta > THETA_MAX_RAD:
        return None
    return theta


def thigh_calf_angle_rad(
    hip: Tuple[float, float],
    knee: Tuple[float, float],
    ankle: Tuple[float, float],
) -> Optional[float]:
    """α: angle between thigh and calf (paper Eq. 2–5)."""
    a = math.hypot(hip[0] - knee[0], hip[1] - knee[1])
    b = math.hypot(ankle[0] - knee[0], ankle[1] - knee[1])
    c = math.hypot(hip[0] - ankle[0], hip[1] - ankle[1])
    if a < 1e-8 or b < 1e-8:
        return None
    cos_v = (a * a + b * b - c * c) / (2.0 * a * b)
    cos_v = max(-1.0, min(1.0, cos_v))
    alpha = math.acos(cos_v)
    if alpha <= ALPHA_MIN_RAD or alpha > ALPHA_MAX_RAD:
        return None
    return alpha


def bin_angle_deg(angle_rad: Optional[float]) -> Optional[float]:
    """Group angles at 30° intervals (paper §2.3)."""
    if angle_rad is None:
        return None
    deg = math.degrees(angle_rad)
    binned = math.floor(deg / ANGLE_BIN_DEG) * ANGLE_BIN_DEG
    return float(binned)


def compute_pose_angles_deg(sk: dict) -> Dict[str, Optional[float]]:
    """Return binned α_L, α_R, θ_L, θ_R in degrees."""
    hip_l = _joint_xy(sk, JOINT_HIP_L)
    hip_r = _joint_xy(sk, JOINT_HIP_R)
    knee_l = _joint_xy(sk, JOINT_KNEE_L)
    knee_r = _joint_xy(sk, JOINT_KNEE_R)
    ankle_l = _joint_xy(sk, JOINT_ANKLE_L)
    ankle_r = _joint_xy(sk, JOINT_ANKLE_R)

    return {
        "alpha_L_deg": bin_angle_deg(thigh_calf_angle_rad(hip_l, knee_l, ankle_l)),
        "alpha_R_deg": bin_angle_deg(thigh_calf_angle_rad(hip_r, knee_r, ankle_r)),
        "theta_L_deg": bin_angle_deg(calf_ground_angle_rad(knee_l, ankle_l)),
        "theta_R_deg": bin_angle_deg(calf_ground_angle_rad(knee_r, ankle_r)),
    }


def _feet_midpoint(sk: dict) -> Tuple[float, float]:
    fl = _joint_xy(sk, JOINT_ANKLE_L)
    fr = _joint_xy(sk, JOINT_ANKLE_R)
    return (fl[0] + fr[0]) / 2.0, (fl[1] + fr[1]) / 2.0


def trajectory_heading_sin_cos(
    sk_t: dict,
    sk_prev: Optional[dict],
) -> Tuple[float, float]:
    """Image-space trajectory heading from foot-midpoint displacement between frames."""
    if not sk_t:
        return 0.0, 0.0
    mx_t, my_t = _feet_midpoint(sk_t)
    if sk_prev:
        mx_p, my_p = _feet_midpoint(sk_prev)
        dx, dy = mx_t - mx_p, my_t - my_p
    else:
        return 0.0, 0.0

    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return 0.0, 0.0
    ang = math.atan2(dy, dx)
    return math.sin(ang), math.cos(ang)


def _pose_angle_features(
    sk: dict,
    *,
    impute_invalid_angles: bool,
    default_angle_deg: float,
) -> Optional[List[float]]:
    angles = compute_pose_angles_deg(sk)
    angle_vals = [angles[k] for k in ("alpha_L_deg", "alpha_R_deg", "theta_L_deg", "theta_R_deg")]
    if impute_invalid_angles:
        angle_vals = [v if v is not None else default_angle_deg for v in angle_vals]
    elif any(v is None for v in angle_vals):
        return None
    return [float(v) for v in angle_vals]


def build_ma_rong_feature_vector(
    ped: dict,
    *,
    impute_invalid_angles: bool = True,
    default_angle_deg: float = 120.0,
) -> Optional[List[float]]:
    """
    Build 6-dim feature vector from a pedestrian dict.

    Returns None if skeleton is missing or too many angles are invalid.
    """
    sk = ped.get("skeleton_normalized") or {}
    if not sk:
        return None

    angle_vals = _pose_angle_features(
        sk,
        impute_invalid_angles=impute_invalid_angles,
        default_angle_deg=default_angle_deg,
    )
    if angle_vals is None:
        return None

    return [
        _sanitize_scalar(ped.get("dist_to_curb_norm", 0.0)),
        _sanitize_scalar(ped.get("movement_norm", 0.0)),
        *angle_vals,
    ]


def build_ma_rong_heading_feature_vector(
    ped: dict,
    prev_ped: Optional[dict] = None,
    *,
    impute_invalid_angles: bool = True,
    default_angle_deg: float = 120.0,
) -> Optional[List[float]]:
    """
    Build 8-dim feature vector: Ma-Rong pose + RSU proxies + trajectory heading.

    Heading uses foot-midpoint displacement in normalized skeleton space vs the
    previous frame for the same pedestrian track.
    """
    sk = ped.get("skeleton_normalized") or {}
    if not sk:
        return None

    angle_vals = _pose_angle_features(
        sk,
        impute_invalid_angles=impute_invalid_angles,
        default_angle_deg=default_angle_deg,
    )
    if angle_vals is None:
        return None

    prev_sk = (prev_ped or {}).get("skeleton_normalized") if prev_ped else None
    h_sin, h_cos = trajectory_heading_sin_cos(sk, prev_sk)

    return [
        _sanitize_scalar(ped.get("dist_to_curb_norm", 0.0)),
        _sanitize_scalar(ped.get("movement_norm", 0.0)),
        float(h_sin),
        float(h_cos),
        *angle_vals,
    ]


def ego_relative_delta_xy(
    ped_x: float,
    ped_y: float,
    ego_x: float,
    ego_y: float,
    ego_yaw_deg: float,
) -> Tuple[float, float]:
    """
    Project pedestrian position into ego vehicle frame (future full 7-feature mode).

    Returns (delta_x_longitudinal, delta_y_lateral) in meters.
    """
    dx = ped_x - ego_x
    dy = ped_y - ego_y
    yaw = math.radians(ego_yaw_deg)
    fwd_x, fwd_y = math.cos(yaw), math.sin(yaw)
    right_x, right_y = -math.sin(yaw), math.cos(yaw)
    delta_x = dx * fwd_x + dy * fwd_y
    delta_y = dx * right_x + dy * right_y
    return delta_x, delta_y


def feature_dim() -> int:
    return FEATURE_DIM


def feature_dim_heading() -> int:
    return FEATURE_DIM_HEADING


def feature_names() -> List[str]:
    return list(FEATURE_NAMES)


def feature_names_heading() -> List[str]:
    return list(FEATURE_NAMES_HEADING)
