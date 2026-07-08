#!/usr/bin/env python3
"""
Orchestrate PIE feature extraction with resume, legacy migration, merge, and time estimates.

Typical one-shot workflow:
  python pie/run_extraction_pipeline.py --migrate-legacy --extract --merge --augment

Status only:
  python pie/run_extraction_pipeline.py --status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pie.config_pie import (
    BY_VIDEO_DIR,
    DEFAULT_FEATURES_JSONL,
    DEFAULT_FEATURES_JSONL_HEADING,
    DEFAULT_YOLO_POSE,
    LEGACY_BULK_JSONL,
)
from pie.extract_pie_features import extract_features
from pie.feature_io import (
    has_legacy_video_cache,
    is_video_cache_complete,
    list_by_video_status,
    merge_by_video,
    migrate_legacy_bulk,
    should_skip_video_extraction,
)
from pie.pie_intention_samples import iter_intention_samples


# Empirical throughput from prior runs on this project (GPU, yolov8x-pose).
ROWS_PER_MINUTE_ESTIMATE = 280.0
POSE_SUCCESS_RATE_ESTIMATE = 0.62


def _videos_with_clips() -> list[tuple[str, str, int]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for sample in iter_intention_samples(require_video=True):
        counts[(sample.set_id, sample.video_id)] += 1
    return sorted((s, v, counts[(s, v)]) for (s, v) in counts)


def estimate_remaining(
    *,
    yolo_weights: Path,
    conf: float,
    force: bool,
    upgrade_joints: bool,
) -> dict:
    videos = _videos_with_clips()
    remaining_candidates = 0
    remaining_videos = 0
    done_videos = 0
    legacy_videos = 0

    for set_id, video_id, n_cands in videos:
        skip, _ = should_skip_video_extraction(
            set_id,
            video_id,
            yolo_weights=yolo_weights,
            conf=conf,
            force=force,
            upgrade_joints=upgrade_joints,
        )
        if skip:
            if has_legacy_video_cache(set_id, video_id) and not is_video_cache_complete(
                set_id, video_id, yolo_weights=yolo_weights, conf=conf, require_joints=True
            ):
                legacy_videos += 1
            else:
                done_videos += 1
            continue

        remaining_videos += 1
        remaining_candidates += n_cands

    est_rows = int(remaining_candidates * POSE_SUCCESS_RATE_ESTIMATE)
    est_minutes = est_rows / ROWS_PER_MINUTE_ESTIMATE if est_rows else 0.0

    return {
        "total_videos_with_clips": len(videos),
        "done_videos_with_joints": done_videos,
        "legacy_videos_no_joints": legacy_videos,
        "remaining_videos_to_extract": remaining_videos,
        "remaining_intention_candidates": remaining_candidates,
        "estimated_usable_rows": est_rows,
        "estimated_minutes": round(est_minutes, 1),
        "estimated_hours": round(est_minutes / 60.0, 2),
        "throughput_assumption_rows_per_min": ROWS_PER_MINUTE_ESTIMATE,
        "pose_success_rate_assumption": POSE_SUCCESS_RATE_ESTIMATE,
    }


def print_status(*, yolo_weights: Path, conf: float, force: bool, upgrade_joints: bool) -> dict:
    videos = _videos_with_clips()
    by_video = list_by_video_status(yolo_weights=yolo_weights, conf=conf)
    estimate = estimate_remaining(
        yolo_weights=yolo_weights, conf=conf, force=force, upgrade_joints=upgrade_joints
    )

    print("=== PIE extraction status ===")
    print(f"Clips with intention frames: {len(videos)}")
    print(f"Per-video cache dir: {BY_VIDEO_DIR}")
    print(f"Complete (joints, matching YOLO/conf): {by_video['complete_with_joints']}")
    print(f"Legacy cached (no joints): {by_video['complete_legacy_no_joints']}")
    print(f"Rows in by_video/: {by_video['rows_in_by_video']}")
    print()
    print("=== Remaining work estimate ===")
    print(f"Videos left to extract: {estimate['remaining_videos_to_extract']}")
    print(f"Intention-frame candidates left: {estimate['remaining_intention_candidates']:,}")
    print(f"Estimated usable rows: ~{estimate['estimated_usable_rows']:,}")
    print(f"Estimated time: ~{estimate['estimated_hours']} h ({estimate['estimated_minutes']} min)")
    print(f"  (assumes {ROWS_PER_MINUTE_ESTIMATE:.0f} rows/min, {POSE_SUCCESS_RATE_ESTIMATE:.0%} pose success)")
    print()

    pending = []
    for set_id, video_id, n in videos:
        skip, _ = should_skip_video_extraction(
            set_id,
            video_id,
            yolo_weights=yolo_weights,
            conf=conf,
            force=force,
            upgrade_joints=upgrade_joints,
        )
        if skip:
            continue
        pending.append((set_id, video_id, n))
    if pending:
        print("Next videos to process:")
        for set_id, video_id, n in pending[:8]:
            print(f"  {set_id}/{video_id} ({n} candidates)")
        if len(pending) > 8:
            print(f"  … and {len(pending) - 8} more")
    else:
        print("All videos cached with joints.")

    return {"by_video": by_video, "estimate": estimate, "pending_videos": len(pending)}


def run_augment(input_path: Path, out_path: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "pie" / "augment_pie_heading_jsonl.py"),
        "--input",
        str(input_path),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return {"augmented": str(out_path)}


def pending_video_count(
    *,
    yolo_weights: Path,
    conf: float,
    force: bool = False,
    upgrade_joints: bool = False,
) -> int:
    videos = _videos_with_clips()
    n = 0
    for set_id, video_id, _ in videos:
        skip, _ = should_skip_video_extraction(
            set_id,
            video_id,
            yolo_weights=yolo_weights,
            conf=conf,
            force=force,
            upgrade_joints=upgrade_joints,
        )
        if not skip:
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="PIE feature extraction pipeline (resume-safe)")
    parser.add_argument("--yolo", default=str(DEFAULT_YOLO_POSE))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--status", action="store_true", help="Print progress + time estimate only")
    parser.add_argument("--migrate-legacy", action="store_true", help="Split pie_ma_rong_features.jsonl into by_video/")
    parser.add_argument("--extract", action="store_true", help="Extract missing per-video caches")
    parser.add_argument("--merge", action="store_true", help="Merge by_video/ → pie_ma_rong_features_full.jsonl")
    parser.add_argument("--augment", action="store_true", help="Build 9-feature heading JSONL")
    parser.add_argument("--force", action="store_true", help="Re-extract all videos")
    parser.add_argument(
        "--upgrade-joints",
        action="store_true",
        help="Re-extract legacy per-video caches that lack stored joints (5 videos today)",
    )
    parser.add_argument("--pending-count", action="store_true", help="Print pending video count and exit")
    parser.add_argument("--sets", nargs="*", default=None)
    parser.add_argument("--videos", nargs="*", default=None)
    args = parser.parse_args()

    yolo_weights = Path(args.yolo).resolve()
    if not yolo_weights.is_file():
        print(f"WARNING: YOLO weights not found at {yolo_weights}", file=sys.stderr)

    if args.pending_count:
        print(
            pending_video_count(
                yolo_weights=yolo_weights,
                conf=args.conf,
                force=args.force,
                upgrade_joints=args.upgrade_joints,
            )
        )
        return

    results: dict = {}

    if args.migrate_legacy:
        results["migrate"] = migrate_legacy_bulk(LEGACY_BULK_JSONL)
        print(json.dumps(results["migrate"], indent=2))

    if args.status and not any([args.extract, args.merge, args.augment]):
        print_status(
            yolo_weights=yolo_weights,
            conf=args.conf,
            force=args.force,
            upgrade_joints=args.upgrade_joints,
        )
        return

    if args.status:
        pass  # print at end after work
    elif not any([args.migrate_legacy, args.extract, args.merge, args.augment]):
        report = print_status(
            yolo_weights=yolo_weights,
            conf=args.conf,
            force=args.force,
            upgrade_joints=args.upgrade_joints,
        )
        if report["pending_videos"] == 0:
            print("Nothing pending. Use --merge --augment to finalize, or --force to re-extract.")
        return

    if args.extract:
        t0 = time.time()
        # upgrade-joints only affects skip logic inside extract_features (re-do legacy
        # caches lacking joints); do NOT restrict set_ids/video_ids here.
        results["extract"] = extract_features(
            set_ids=args.sets,
            video_ids=args.videos,
            yolo_weights=yolo_weights,
            out_path=None,
            per_video=True,
            max_samples=None,
            device=args.device,
            conf=args.conf,
            force=args.force,
            upgrade_joints=args.upgrade_joints,
        )
        results["extract"]["elapsed_s"] = round(time.time() - t0, 1)
        print(json.dumps(results["extract"], indent=2))

    if args.merge:
        results["merge"] = merge_by_video(Path(DEFAULT_FEATURES_JSONL))
        print(json.dumps(results["merge"], indent=2))

    if args.augment:
        results["augment"] = run_augment(Path(DEFAULT_FEATURES_JSONL), Path(DEFAULT_FEATURES_JSONL_HEADING))

    print_status(
        yolo_weights=yolo_weights,
        conf=args.conf,
        force=False,
        upgrade_joints=False,
    )


if __name__ == "__main__":
    main()
