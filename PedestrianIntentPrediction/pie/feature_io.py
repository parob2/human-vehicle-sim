"""Paths, manifests, merge, and legacy migration for PIE feature JSONL caches."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from pie.config_pie import (
    BY_VIDEO_DIR,
    EXTRACTION_SCHEMA_VERSION,
    PIE_FEATURES_DIR,
)

RowKey = Tuple[str, str, str, int]


def by_video_jsonl_path(set_id: str, video_id: str) -> Path:
    return BY_VIDEO_DIR / f"{set_id}_{video_id}.jsonl"


def by_video_manifest_path(set_id: str, video_id: str) -> Path:
    return BY_VIDEO_DIR / f"{set_id}_{video_id}.manifest.json"


def row_key(row: dict) -> RowKey:
    return (
        str(row["set_id"]),
        str(row["video_id"]),
        str(row["ped_id"]),
        int(row["frame_id"]),
    )


def load_jsonl_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl_rows(paths: Iterable[Path]) -> Iterator[dict]:
    for path in sorted(paths):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_manifest(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def manifest_matches_settings(
    manifest: dict,
    *,
    yolo_weights: Path,
    conf: float,
    require_joints: bool,
) -> bool:
    if int(manifest.get("schema_version", 0)) < EXTRACTION_SCHEMA_VERSION:
        return False
    if manifest.get("yolo_weights") != str(yolo_weights.resolve()):
        return False
    if float(manifest.get("conf", -1.0)) != float(conf):
        return False
    if require_joints and not manifest.get("has_joints", False):
        return False
    if int(manifest.get("rows_written", 0)) <= 0:
        return False
    return True


def is_video_cache_complete(
    set_id: str,
    video_id: str,
    *,
    yolo_weights: Path,
    conf: float,
    require_joints: bool = True,
) -> bool:
    jsonl = by_video_jsonl_path(set_id, video_id)
    manifest = load_manifest(by_video_manifest_path(set_id, video_id))
    if manifest is None or not jsonl.is_file():
        return False
    return manifest_matches_settings(
        manifest,
        yolo_weights=yolo_weights,
        conf=conf,
        require_joints=require_joints,
    )


def has_legacy_video_cache(set_id: str, video_id: str) -> bool:
    """True when a per-video JSONL exists (possibly without stored joints)."""
    jsonl = by_video_jsonl_path(set_id, video_id)
    manifest = load_manifest(by_video_manifest_path(set_id, video_id))
    if not jsonl.is_file() or manifest is None:
        return False
    return int(manifest.get("rows_written", 0)) > 0


def should_skip_video_extraction(
    set_id: str,
    video_id: str,
    *,
    yolo_weights: Path,
    conf: float,
    force: bool,
    upgrade_joints: bool,
) -> tuple[bool, str]:
    if force:
        return False, ""
    if is_video_cache_complete(
        set_id, video_id, yolo_weights=yolo_weights, conf=conf, require_joints=True
    ):
        return True, "joints"
    manifest = load_manifest(by_video_manifest_path(set_id, video_id))
    if has_legacy_video_cache(set_id, video_id):
        if upgrade_joints and not manifest.get("has_joints", False):
            return False, ""
        if not upgrade_joints:
            return True, "legacy"
    return False, ""


def write_rows_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def migrate_legacy_bulk(
    legacy_path: Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Split a monolithic JSONL into per-video caches (legacy rows lack joints)."""
    if not legacy_path.is_file():
        return {"migrated_videos": 0, "rows": 0, "skipped_videos": 0}

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in load_jsonl_rows(legacy_path):
        grouped[(row["set_id"], row["video_id"])].append(row)

    migrated = 0
    skipped = 0
    total_rows = 0
    for (set_id, video_id), rows in sorted(grouped.items()):
        out = by_video_jsonl_path(set_id, video_id)
        manifest_path = by_video_manifest_path(set_id, video_id)
        if out.is_file() and manifest_path.is_file() and not overwrite:
            skipped += 1
            continue

        rows.sort(key=lambda r: (r["ped_id"], int(r["frame_id"])))
        write_rows_jsonl(out, rows)
        write_manifest(
            manifest_path,
            {
                "set_id": set_id,
                "video_id": video_id,
                "schema_version": 1,
                "has_joints": False,
                "heading_source_legacy": "bbox_footpoint",
                "rows_written": len(rows),
                "candidates": None,
                "migrated_from": str(legacy_path),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        migrated += 1
        total_rows += len(rows)

    return {
        "migrated_videos": migrated,
        "skipped_videos": skipped,
        "rows": total_rows,
        "legacy_path": str(legacy_path),
    }


def merge_by_video(
    out_path: Path,
    *,
    by_video_dir: Path | None = None,
    dedupe: bool = True,
) -> dict:
    """Merge per-video JSONL files into one training cache."""
    src_dir = by_video_dir or BY_VIDEO_DIR
    paths = sorted(src_dir.glob("*.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[RowKey] = set()
    written = 0
    skipped_dupes = 0
    by_video: Counter[tuple[str, str]] = Counter()

    with open(out_path, "w", encoding="utf-8") as out_f:
        for path in paths:
            for row in load_jsonl_rows(path):
                key = row_key(row)
                if dedupe:
                    if key in seen:
                        skipped_dupes += 1
                        continue
                    seen.add(key)
                out_f.write(json.dumps(row) + "\n")
                written += 1
                by_video[(row["set_id"], row["video_id"])] += 1

    return {
        "output": str(out_path),
        "source_files": len(paths),
        "rows_written": written,
        "skipped_dupes": skipped_dupes,
        "videos": len(by_video),
    }


def list_by_video_status(
    *,
    yolo_weights: Path,
    conf: float,
) -> dict:
    manifests = sorted(BY_VIDEO_DIR.glob("*.manifest.json"))
    complete_joint = 0
    complete_legacy = 0
    rows = 0
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path) or {}
        set_id = manifest.get("set_id", "")
        video_id = manifest.get("video_id", "")
        jsonl = by_video_jsonl_path(set_id, video_id)
        n = sum(1 for _ in open(jsonl)) if jsonl.is_file() else 0
        rows += n
        if manifest_matches_settings(
            manifest, yolo_weights=yolo_weights, conf=conf, require_joints=True
        ):
            complete_joint += 1
        elif int(manifest.get("rows_written", 0)) > 0 and jsonl.is_file():
            complete_legacy += 1

    return {
        "by_video_manifests": len(manifests),
        "complete_with_joints": complete_joint,
        "complete_legacy_no_joints": complete_legacy,
        "rows_in_by_video": rows,
    }
