#!/usr/bin/env python3
"""Augment cached PIE 7-feature JSONL rows with trajectory heading (9 features)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pie.config_pie import (
    DEFAULT_FEATURES_JSONL,
    DEFAULT_FEATURES_JSONL_HEADING,
    FEATURE_DIM_PIE,
    FEATURE_DIM_PIE_HEADING,
)
from pie.features_pie import (
    feature_names_heading,
    trajectory_heading_sin_cos_from_bbox,
    trajectory_heading_sin_cos_from_joints,
)


def _joints_from_row(row: dict) -> dict | None:
    raw = row.get("joints")
    if not raw:
        return None
    try:
        return {name: (float(xy[0]), float(xy[1])) for name, xy in raw.items()}
    except (TypeError, ValueError, IndexError):
        return None


def augment_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["set_id"], row["video_id"], row["ped_id"])
        grouped[key].append(row)

    out: list[dict] = []
    stats = {
        "tracks": len(grouped),
        "written": 0,
        "skipped": 0,
        "heading_from_joints": 0,
        "heading_from_bbox": 0,
    }

    for _key, track_rows in grouped.items():
        track_rows.sort(key=lambda r: int(r["frame_id"]))
        prev_bbox = None
        prev_joints = None
        for row in track_rows:
            feats = row.get("features") or []
            if len(feats) != FEATURE_DIM_PIE:
                stats["skipped"] += 1
                continue

            bbox = tuple(float(v) for v in row["bbox"])
            joints = _joints_from_row(row)
            if joints is not None:
                h_sin, h_cos = trajectory_heading_sin_cos_from_joints(joints, prev_joints)
                heading_source = "ankle_midpoint"
                stats["heading_from_joints"] += 1
                prev_joints = joints
            else:
                h_sin, h_cos = trajectory_heading_sin_cos_from_bbox(bbox, prev_bbox)
                heading_source = "bbox_footpoint"
                stats["heading_from_bbox"] += 1

            heading_feats = [
                float(feats[0]),
                float(feats[1]),
                float(feats[2]),
                float(h_sin),
                float(h_cos),
                float(feats[3]),
                float(feats[4]),
                float(feats[5]),
                float(feats[6]),
            ]

            prev_bbox = bbox
            new_row = dict(row)
            new_row["features"] = heading_feats
            new_row["feature_names"] = feature_names_heading()
            new_row["heading_source"] = heading_source
            out.append(new_row)
            stats["written"] += 1

    out.sort(key=lambda r: (r["set_id"], r["video_id"], r["ped_id"], int(r["frame_id"])))
    if out and len(out[0]["features"]) != FEATURE_DIM_PIE_HEADING:
        raise RuntimeError("Augmented feature dimension mismatch")
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Add trajectory heading to cached PIE JSONL")
    parser.add_argument("--input", default=str(DEFAULT_FEATURES_JSONL))
    parser.add_argument("--out", default=str(DEFAULT_FEATURES_JSONL_HEADING))
    args = parser.parse_args()

    rows: list[dict] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    augmented, stats = augment_rows(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in augmented:
            f.write(json.dumps(row) + "\n")

    print(json.dumps({"output": str(out_path), **stats}, indent=2))


if __name__ == "__main__":
    main()
