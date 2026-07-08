"""Iterate PIE intention-estimation frames (exp_start … critical_point)."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

PIE_UTIL = Path(__file__).resolve().parents[2] / "PIE_dataset" / "utilities"
if str(PIE_UTIL) not in sys.path:
    sys.path.insert(0, str(PIE_UTIL))

from pie_data import PIE  # noqa: E402

from pie.config_pie import PIE_CLIPS_DIR, PIE_ROOT


@dataclass(frozen=True)
class PIESample:
    set_id: str
    video_id: str
    ped_id: str
    frame_id: int
    bbox: Tuple[float, float, float, float]
    occlusion: int
    intention_prob: float
    intention_binary: int
    obd_speed_kmh: float
    video_path: Path


def _video_path(set_id: str, video_id: str) -> Path:
    return PIE_CLIPS_DIR / set_id / f"{video_id}.mp4"


def _video_ready(path: Path, min_bytes: int = 400_000_000) -> bool:
    """Skip missing clips and partial wget downloads (complete PIE clips are ~440 MB–1.3 GB)."""
    return path.is_file() and path.stat().st_size >= min_bytes


def _resolve_sets(set_ids: Optional[Sequence[str]]) -> List[str]:
    if set_ids:
        return list(set_ids)
    ann_dir = PIE_ROOT / "annotations"
    return sorted(
        p.name
        for p in ann_dir.iterdir()
        if p.is_dir() and p.name.startswith("set") and not p.name.endswith(".zip")
    )


def iter_intention_samples(
    *,
    pie_root: Path | None = None,
    set_ids: Optional[Sequence[str]] = None,
    video_ids: Optional[Sequence[str]] = None,
    require_video: bool = True,
) -> Iterator[PIESample]:
    """
    Yield one row per annotated intention frame.

    Matches PIE ``_get_intention`` window and skips irrelevant pedestrians (crossing == -1).
    """
    root = pie_root or PIE_ROOT
    pie = PIE(regen_database=False, data_path=str(root))
    db = pie.generate_database()

    for set_id in _resolve_sets(set_ids):
        if set_id not in db:
            continue
        for video_id in sorted(db[set_id].keys()):
            if video_ids and video_id not in video_ids:
                continue
            vid_path = _video_path(set_id, video_id)
            if require_video and not _video_ready(vid_path):
                continue

            ped_annots = db[set_id][video_id]["ped_annotations"]
            veh = db[set_id][video_id]["vehicle_annotations"]

            for ped_id, ped in sorted(ped_annots.items()):
                attrs = ped["attributes"]
                if int(attrs.get("crossing", 0)) == -1:
                    continue

                frames = ped["frames"]
                boxes = ped["bbox"]
                occlusions = ped["occlusion"]
                exp_start = int(attrs["exp_start_point"])
                critical = int(attrs["critical_point"])

                if exp_start not in frames or critical not in frames:
                    continue

                start_idx = frames.index(exp_start)
                end_idx = frames.index(critical)
                intention_prob = float(attrs["intention_prob"])
                intention_binary = int(intention_prob > 0.5)

                for idx in range(start_idx, end_idx + 1):
                    frame_id = int(frames[idx])
                    bbox = tuple(float(v) for v in boxes[idx])
                    obd = veh.get(frame_id, {})
                    speed = float(obd.get("OBD_speed", obd.get("GPS_speed", 0.0)))

                    yield PIESample(
                        set_id=set_id,
                        video_id=video_id,
                        ped_id=str(ped_id),
                        frame_id=frame_id,
                        bbox=bbox,
                        occlusion=int(occlusions[idx]),
                        intention_prob=intention_prob,
                        intention_binary=intention_binary,
                        obd_speed_kmh=speed,
                        video_path=vid_path,
                    )


def count_samples(**kwargs) -> int:
    return sum(1 for _ in iter_intention_samples(**kwargs))
