"""Shared Ma & Rong RF hyperparameters and angle-bin settings for the PIE pipeline.

PIE-specific feature names and paths live in pie/config_pie.py.
Simulation world coordinates live in CarlaSim/sim_config.py.
"""
from __future__ import annotations

import math

# Feature schema (6-dim RSU proxy; paper §2.4.1 had 7 with v_ego).
FEATURE_NAMES = [
    "dist_to_curb_norm",  # RSU proxy for Δy (lateral)
    "movement_norm",      # RSU proxy for Δx (longitudinal approach cue)
    "alpha_L_deg",        # left thigh–calf angle (binned degrees)
    "alpha_R_deg",        # right thigh–calf angle
    "theta_L_deg",        # left calf–ground angle
    "theta_R_deg",        # right calf–ground angle
]

FEATURE_DIM = len(FEATURE_NAMES)
FEATURES_KEY = "features_ma_rong"

# 8-dim variant: adds image-space trajectory heading (foot-midpoint displacement).
FEATURE_NAMES_HEADING = [
    "dist_to_curb_norm",
    "movement_norm",
    "trajectory_heading_sin",
    "trajectory_heading_cos",
    "alpha_L_deg",
    "alpha_R_deg",
    "theta_L_deg",
    "theta_R_deg",
]
FEATURE_DIM_HEADING = len(FEATURE_NAMES_HEADING)
FEATURES_KEY_HEADING = "features_ma_rong_heading"

TARGET_KEY = "intent_crossing_within_T_s"

# Random Forest (paper §3.2).
RF_N_ESTIMATORS = 320
RF_RANDOM_STATE = 42
RF_CLASS_WEIGHT = "balanced"
TRAIN_TEST_SPLIT = 0.30  # 30% test, 70% train
DECISION_THRESHOLD = 0.5

# Angle preprocessing (paper §2.3).
ANGLE_BIN_DEG = 30.0
# θ ∈ (0, π/2], α ∈ (π/2, π] in radians
THETA_MIN_RAD = 0.0
THETA_MAX_RAD = math.pi / 2.0
ALPHA_MIN_RAD = math.pi / 2.0
ALPHA_MAX_RAD = math.pi

# Skeleton joint mapping (YOLO crl_* → paper hip/knee/ankle).
JOINT_HIP_L = "crl_thigh_l"
JOINT_HIP_R = "crl_thigh_r"
JOINT_KNEE_L = "crl_leg_l"
JOINT_KNEE_R = "crl_leg_r"
JOINT_ANKLE_L = "crl_foot_l"
JOINT_ANKLE_R = "crl_foot_r"
