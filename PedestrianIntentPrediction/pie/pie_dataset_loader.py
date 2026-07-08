"""Load Ma-Rong PIE feature rows from JSONL cache."""
from __future__ import annotations

import glob
import json
from typing import Any, Dict, List, Tuple

import numpy as np

from pie.config_pie import (
    FEATURE_DIM_PIE,
    FEATURE_DIM_PIE_HEADING,
    FEATURE_NAMES_PIE,
    FEATURE_NAMES_PIE_HEADING,
    TARGET_KEY_PIE,
)


def load_pie_features_from_jsonl(
    paths: List[str],
    *,
    require_valid_pose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    X: List[List[float]] = []
    y: List[int] = []
    groups: List[str] = []
    meta: List[Dict[str, Any]] = []

    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if require_valid_pose and not row.get("pose_ok", False):
                        continue
                    feats = row.get("features")
                    if not feats or len(feats) != len(FEATURE_NAMES_PIE):
                        continue
                    label = row.get(TARGET_KEY_PIE)
                    if label is None:
                        continue

                    group_id = f"{row['set_id']}::{row['video_id']}::{row['ped_id']}"
                    X.append([float(v) for v in feats])
                    y.append(int(label))
                    groups.append(group_id)
                    meta.append({
                        "source": path,
                        "set_id": row.get("set_id"),
                        "video_id": row.get("video_id"),
                        "ped_id": row.get("ped_id"),
                        "frame_id": row.get("frame_id"),
                        "group_id": group_id,
                        "intention_prob": row.get("intention_prob"),
                    })

    if not X:
        raise ValueError("No PIE feature rows found. Run extract_pie_features.py first.")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int), np.asarray(groups), meta


def load_pie_heading_features_from_jsonl(
    paths: List[str],
    *,
    require_valid_pose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    X: List[List[float]] = []
    y: List[int] = []
    groups: List[str] = []
    meta: List[Dict[str, Any]] = []

    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if require_valid_pose and not row.get("pose_ok", False):
                        continue
                    feats = row.get("features")
                    if not feats or len(feats) != len(FEATURE_NAMES_PIE_HEADING):
                        continue
                    label = row.get(TARGET_KEY_PIE)
                    if label is None:
                        continue

                    group_id = f"{row['set_id']}::{row['video_id']}::{row['ped_id']}"
                    X.append([float(v) for v in feats])
                    y.append(int(label))
                    groups.append(group_id)
                    meta.append({
                        "source": path,
                        "set_id": row.get("set_id"),
                        "video_id": row.get("video_id"),
                        "ped_id": row.get("ped_id"),
                        "frame_id": row.get("frame_id"),
                        "group_id": group_id,
                        "intention_prob": row.get("intention_prob"),
                        "heading_source": row.get("heading_source"),
                    })

    if not X:
        raise ValueError("No PIE heading feature rows found. Run augment_pie_heading_jsonl.py first.")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int), np.asarray(groups), meta
