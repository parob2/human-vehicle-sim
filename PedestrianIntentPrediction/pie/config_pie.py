"""Ma & Rong (2022) seven-feature schema on the PIE ego-view dataset."""
from __future__ import annotations

from pathlib import Path

# Paper §2.4.1 — seven ego-view features (Mobileye/CAN + OpenPose on BPI; PIE proxies below).
FEATURE_NAMES_PIE = [
    "delta_y_m",       # lateral distance pedestrian ↔ ego (m)
    "delta_x_m",       # longitudinal distance along ego heading (m)
    "v_ego_kmh",       # ego speed from PIE OBD (km/h)
    "alpha_L_deg",     # left thigh–calf, 30° binned
    "alpha_R_deg",     # right thigh–calf
    "theta_L_deg",     # left calf–ground
    "theta_R_deg",     # right calf–ground
]

FEATURE_DIM_PIE = len(FEATURE_NAMES_PIE)
FEATURES_KEY_PIE = "features_ma_rong_pie"

# 9-dim variant: paper 7 features + image-space trajectory heading (foot displacement).
FEATURE_NAMES_PIE_HEADING = [
    "delta_y_m",
    "delta_x_m",
    "v_ego_kmh",
    "trajectory_heading_sin",
    "trajectory_heading_cos",
    "alpha_L_deg",
    "alpha_R_deg",
    "theta_L_deg",
    "theta_R_deg",
]
FEATURE_DIM_PIE_HEADING = len(FEATURE_NAMES_PIE_HEADING)
FEATURES_KEY_PIE_HEADING = "features_ma_rong_pie_heading"

TARGET_KEY_PIE = "intention_binary"

MODEL_VERSION_PIE = "ma_rong_pie_v1"
MODEL_VERSION_PIE_HEADING = "ma_rong_pie_heading_v1"
DEFAULT_PED_HEIGHT_M = 1.70

PIE_ROOT = Path(__file__).resolve().parents[2] / "PIE_dataset"
PIE_CLIPS_DIR = PIE_ROOT / "PIE_clips"
PIE_FEATURES_DIR = Path(__file__).resolve().parent / "features_cache"
BY_VIDEO_DIR = PIE_FEATURES_DIR / "by_video"
LEGACY_BULK_JSONL = PIE_FEATURES_DIR / "pie_ma_rong_features.jsonl"

# Full-dataset caches (preferred for production training).
DEFAULT_FEATURES_JSONL = PIE_FEATURES_DIR / "pie_ma_rong_features_full.jsonl"
DEFAULT_FEATURES_JSONL_HEADING = PIE_FEATURES_DIR / "pie_ma_rong_features_full_heading.jsonl"
DEFAULT_FEATURES_JSONL_FULL = DEFAULT_FEATURES_JSONL

# Minimal experiment caches (set02/video_0001 + set04/video_0001 only).
MINIMAL_FEATURES_JSONL = PIE_FEATURES_DIR / "pie_ma_rong_features_minimal.jsonl"
MINIMAL_FEATURES_JSONL_HEADING = PIE_FEATURES_DIR / "pie_ma_rong_features_minimal_heading.jsonl"

EXTRACTION_SCHEMA_VERSION = 2
DEFAULT_YOLO_POSE = Path(__file__).resolve().parents[2] / "CarlaSim" / "yolov8m-pose.pt"
DEFAULT_MODEL_PATH_PIE = Path(__file__).resolve().parent / "models" / "rf_ma_rong_pie.joblib"
DEFAULT_MODEL_PATH_PIE_HEADING = Path(__file__).resolve().parent / "models" / "rf_ma_rong_pie_heading.joblib"
DEFAULT_PIE_HEADING_EVAL_OUT = Path(__file__).resolve().parent / "models" / "eval_out_pie_heading"

# Re-use parent RF hyperparameters (paper §3.2).
from config import (  # noqa: E402
    ANGLE_BIN_DEG,
    DECISION_THRESHOLD,
    RF_CLASS_WEIGHT,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE,
    TRAIN_TEST_SPLIT,
)
