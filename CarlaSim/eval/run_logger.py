"""Structured JSONL run logger for main.py validation."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

_SIM_TIME_PREC = 8


class RunLogger:
    """Append one JSON object per simulation tick to a JSONL file."""

    def __init__(
        self,
        capture_dir: str,
        *,
        scenario_id: str = "default",
        run_subdir: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.scenario_id = scenario_id
        self.run_subdir = run_subdir or f"{scenario_id}/run_{ts}"
        self.run_dir = Path(capture_dir) / self.run_subdir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "run.jsonl"
        self.metadata_path = self.run_dir / "run_metadata.json"
        self._metadata = dict(metadata or {})
        self._metadata.setdefault("scenario_id", scenario_id)
        self._metadata.setdefault("run_subdir", self.run_subdir)
        self._metadata.setdefault("jsonl_path", str(self.path))
        self._fh = open(self.path, "a", encoding="utf-8")
        self._metadata_written = False
        self.tick_idx = 0
        self.analysis_step_idx = 0
        self.last_analysis_sim_time_s: Optional[float] = None
        self.min_centroid_distance_m = float("inf")
        self.collision_events: list = []
        self._last_perception: Dict[str, Any] = {}
        self._last_hazard_decision: Dict[str, Any] = {}
        self._last_v2x: Dict[str, Any] = {}
        self._last_robot: Dict[str, Any] = {}
        self._last_denm: Dict[str, Any] = {}
        self._last_cpm: Dict[str, Any] = {}
        self._last_cam: Dict[str, Any] = {}

    @property
    def run_dir_path(self) -> Path:
        return self.run_dir

    def write_metadata(self, extra: Optional[Dict[str, Any]] = None):
        if extra:
            self._metadata.update(extra)
        if not self._metadata_written:
            meta_line = {"record_type": "run_metadata", **self._metadata}
            self._fh.write(json.dumps(meta_line, default=str) + "\n")
            self._fh.flush()
            self._metadata_written = True
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def close(self):
        if self._fh and not self._fh.closed:
            self.write_metadata()
            self._fh.close()

    def record_analysis(self, *, sim_time_s: float, perception: Dict[str, Any]):
        self.analysis_step_idx += 1
        self.last_analysis_sim_time_s = float(sim_time_s)
        self._last_perception = dict(perception)

    def record_hazard_decision(self, hazard_decision: Dict[str, Any]):
        self._last_hazard_decision = dict(hazard_decision)

    def record_v2x(self, v2x: Dict[str, Any]):
        self._last_v2x = dict(v2x)

    def record_robot(self, robot: Dict[str, Any]):
        self._last_robot = dict(robot)

    def record_denm(self, denm: Dict[str, Any]):
        self._last_denm = dict(denm)

    def record_cpm(self, cpm: Dict[str, Any]):
        self._last_cpm = dict(cpm)

    def record_cam(self, cam: Dict[str, Any]):
        self._last_cam = dict(cam)

    def update_min_distance(self, dist_m: float):
        if math.isfinite(dist_m):
            self.min_centroid_distance_m = min(self.min_centroid_distance_m, dist_m)

    def record_collision(self, *, sim_time_s: float, other_actor_id: int):
        self.collision_events.append(
            {"sim_time_s": sim_time_s, "other_actor_id": other_actor_id}
        )

    def log_tick(
        self,
        *,
        sim_time_s: float,
        ego: Optional[Dict[str, Any]],
        ped_gt: Optional[Dict[str, Any]],
        ego_brake_level: float,
        collision_this_tick: bool = False,
    ):
        perception_age_s = None
        if self.last_analysis_sim_time_s is not None:
            perception_age_s = round(
                float(sim_time_s) - self.last_analysis_sim_time_s, _SIM_TIME_PREC
            )

        record = {
            "scenario_id": self.scenario_id,
            "sim_time_s": round(float(sim_time_s), _SIM_TIME_PREC),
            "tick_idx": self.tick_idx,
            "clock": {
                "sim_time_s": round(float(sim_time_s), _SIM_TIME_PREC),
                "analysis_step_idx": self.analysis_step_idx,
                "last_analysis_sim_time_s": self.last_analysis_sim_time_s,
                "perception_age_s": perception_age_s,
            },
            "ego": ego,
            "ped_gt": ped_gt,
            "ped_perception": dict(self._last_perception),
            "hazard_decision": dict(self._last_hazard_decision),
            "v2x": dict(self._last_v2x),
            "denm": dict(self._last_denm),
            "cpm": dict(self._last_cpm),
            "cam": dict(self._last_cam),
            "robot": dict(self._last_robot),
            "ego_brake_level": round(float(ego_brake_level), 4),
            "safety": {
                "min_centroid_distance_m": (
                    None
                    if not math.isfinite(self.min_centroid_distance_m)
                    else round(self.min_centroid_distance_m, 4)
                ),
                "collision_this_tick": collision_this_tick,
                "collision_count": len(self.collision_events),
            },
        }
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()
        self.tick_idx += 1
