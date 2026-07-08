#!/usr/bin/env python3
"""
Extract Ma-Rong 7-feature vectors from PIE intention frames.

Requires PIE video clips under PIE_dataset/PIE_clips/{setXX}/{video_XXXX}.mp4.
Runs YOLOv8-pose (OpenPose-equivalent hip/knee/ankle) on each annotated frame.

Per-video output (default): features_cache/by_video/{set}_{video}.jsonl + .manifest.json
Use run_extraction_pipeline.py to process all clips with resume/skip.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pie.config_pie import (
    BY_VIDEO_DIR,
    DEFAULT_FEATURES_JSONL,
    DEFAULT_YOLO_POSE,
    EXTRACTION_SCHEMA_VERSION,
)
from pie.feature_io import (
    by_video_jsonl_path,
    by_video_manifest_path,
    should_skip_video_extraction,
    write_manifest,
    write_rows_jsonl,
)
from pie.features_pie import build_pie_feature_vector, feature_names
from pie.pie_intention_samples import PIESample, iter_intention_samples
from pie.pose_yolo import extract_joints_from_frame


def _open_video_cache():
    return {}


def _read_frame(video_cache: dict, sample: PIESample):
    key = str(sample.video_path)
    if key not in video_cache:
        cap = cv2.VideoCapture(str(sample.video_path))
        if not cap.isOpened():
            video_cache[key] = None
            return None
        video_cache[key] = cap
    cap = video_cache[key]
    if cap is None:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_id)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _joints_to_record(joints: dict) -> dict:
    return {name: [float(xy[0]), float(xy[1])] for name, xy in joints.items()}


def _record_from_sample(sample: PIESample, joints: dict, feats: list[float]) -> dict:
    return {
        "set_id": sample.set_id,
        "video_id": sample.video_id,
        "ped_id": sample.ped_id,
        "frame_id": sample.frame_id,
        "bbox": list(sample.bbox),
        "joints": _joints_to_record(joints),
        "occlusion": sample.occlusion,
        "intention_prob": sample.intention_prob,
        "intention_binary": sample.intention_binary,
        "obd_speed_kmh": sample.obd_speed_kmh,
        "features": feats,
        "feature_names": feature_names(),
        "pose_ok": True,
    }


def extract_features(
    *,
    set_ids: list[str] | None,
    video_ids: list[str] | None,
    yolo_weights: Path,
    out_path: Path | None,
    per_video: bool,
    max_samples: int | None,
    device: str,
    conf: float,
    force: bool,
    upgrade_joints: bool = False,
) -> dict:
    from ultralytics import YOLO

    yolo_weights = yolo_weights.resolve()
    model = YOLO(str(yolo_weights))

    overall = defaultdict(int)
    video_cache = _open_video_cache()

    # Determine which (set, video) pairs to process.
    targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sample in iter_intention_samples(set_ids=set_ids, video_ids=video_ids, require_video=True):
        key = (sample.set_id, sample.video_id)
        if key in seen:
            continue
        seen.add(key)
        targets.append(key)

    for set_id, video_id in sorted(targets):
        if per_video:
            video_out = by_video_jsonl_path(set_id, video_id)
            manifest_path = by_video_manifest_path(set_id, video_id)
            skip, reason = should_skip_video_extraction(
                set_id,
                video_id,
                yolo_weights=yolo_weights,
                conf=conf,
                force=force,
                upgrade_joints=upgrade_joints,
            )
            if skip:
                manifest = json.loads(
                    by_video_manifest_path(set_id, video_id).read_text(encoding="utf-8")
                )
                overall["skipped_videos"] += 1
                overall["skipped_rows"] += int(manifest.get("rows_written", 0))
                tag = "legacy, no joints" if reason == "legacy" else "cached"
                print(
                    f"  skip {set_id}/{video_id} ({manifest.get('rows_written', 0)} rows, {tag})"
                )
                continue
            write_path = video_out
        else:
            if out_path is None:
                raise ValueError("out_path required when per_video=False")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_path = out_path

        stats = defaultdict(int)
        rows: list[dict] = []
        video_ids_filter = [video_id]
        set_ids_filter = [set_id]

        for sample in iter_intention_samples(
            set_ids=set_ids_filter,
            video_ids=video_ids_filter,
            require_video=True,
        ):
            stats["candidates"] += 1
            if max_samples is not None and stats["written"] >= max_samples:
                break

            frame = _read_frame(video_cache, sample)
            if frame is None:
                stats["missing_frame"] += 1
                continue

            joints = extract_joints_from_frame(
                model,
                frame,
                sample.bbox,
                device=device,
                conf=conf,
            )
            if joints is None:
                stats["no_pose_match"] += 1
                continue

            feats = build_pie_feature_vector(
                bbox=sample.bbox,
                obd_speed_kmh=sample.obd_speed_kmh,
                joints=joints,
                impute_invalid_angles=False,
            )
            if feats is None:
                stats["invalid_angles"] += 1
                continue

            rows.append(_record_from_sample(sample, joints, feats))
            stats["written"] += 1

            if stats["written"] % 100 == 0:
                print(f"  {set_id}/{video_id}: extracted {stats['written']} rows …")

        if per_video:
            write_rows_jsonl(write_path, rows)
            write_manifest(
                manifest_path,
                {
                    "set_id": set_id,
                    "video_id": video_id,
                    "schema_version": EXTRACTION_SCHEMA_VERSION,
                    "has_joints": True,
                    "yolo_weights": str(yolo_weights),
                    "conf": float(conf),
                    "device": str(device),
                    "rows_written": int(stats["written"]),
                    "candidates": int(stats["candidates"]),
                    "missing_frame": int(stats["missing_frame"]),
                    "no_pose_match": int(stats["no_pose_match"]),
                    "invalid_angles": int(stats["invalid_angles"]),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(
                f"  done {set_id}/{video_id}: {stats['written']}/{stats['candidates']} rows → {write_path.name}"
            )
        else:
            with open(write_path, "w", encoding="utf-8") as out_f:
                for row in rows:
                    out_f.write(json.dumps(row) + "\n")

        for k, v in stats.items():
            overall[k] += v
        overall["processed_videos"] += 1
        if per_video:
            overall["output_dir"] = str(BY_VIDEO_DIR)
        else:
            overall["output_path"] = str(write_path)

    for cap in video_cache.values():
        if cap is not None:
            cap.release()

    return dict(overall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Ma-Rong 7-feature PIE dataset")
    parser.add_argument("--sets", nargs="*", default=None, help="PIE sets, e.g. set01 set02")
    parser.add_argument("--videos", nargs="*", default=None, help="Limit to video ids, e.g. video_0001")
    parser.add_argument("--yolo", default=str(DEFAULT_YOLO_POSE), help="YOLOv8-pose weights")
    parser.add_argument(
        "--out",
        default=None,
        help="Monolithic output JSONL (only used with --no-per-video)",
    )
    parser.add_argument(
        "--per-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one JSONL + manifest per video under features_cache/by_video/ (default: on)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max rows per video")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--force", action="store_true", help="Re-extract even if per-video cache exists")
    parser.add_argument(
        "--upgrade-joints",
        action="store_true",
        help="Re-extract legacy per-video caches that lack stored joints",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else None
    if not args.per_video and out_path is None:
        out_path = Path(DEFAULT_FEATURES_JSONL)

    if args.per_video:
        print(f"Per-video cache → {BY_VIDEO_DIR}")
    else:
        print(f"Writing features to {out_path}")

    stats = extract_features(
        set_ids=args.sets,
        video_ids=args.videos,
        yolo_weights=Path(args.yolo),
        out_path=out_path,
        per_video=args.per_video,
        max_samples=args.max_samples,
        device=args.device,
        conf=args.conf,
        force=args.force,
        upgrade_joints=args.upgrade_joints,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
