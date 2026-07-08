"""CARLA crosswalk simulation runtime (orchestrator).

Thesis-aligned modules:
  rsu_perception.py       — YOLOv8-pose + ByteTrack, zone projection
  intent_prediction.py    — PIE RF early crossing intent
  cooperative_awareness.py — ped phase context + ego CAM parsing
  hazard_decision_policy.py — hazard decision policy / risk scoring (decision layer)
  ped_state_machine.py    — off-zone / on-zone phase classification
  denm_dissemination.py   — DENM transmission policy
  actuation_v2x.py        — DENM/CPM/CAM broadcast, ego brake, robot indicator

Setup: see ../README.md and ../.env.example.
"""

import importlib
import json
import math
import os
import queue
import random
import sys
import time
from pathlib import Path

import carla
import cv2
import numpy as np
import pygame

import sim_config  # noqa: F401 — triggers CARLA PythonAPI path bootstrap
from actuation_v2x import (
    _configure_v2x_custom_blueprint,
    _object_id_as_int,
    _v2x_custom_send,
    build_cam_payload,
    build_cpm_payload,
    build_denm_payload,
    compute_robot_light_state,
    cpm_select_objects,
    draw_robot_light_block,
    ego_brake_level_from_ttc_s,
    ego_ttc_from_cpm_objects,
    ego_vru_brake_control,
    match_cpm_pedestrian_for_denm,
    parse_cam_payload_from_v2x,
    parse_cpm_payload_from_v2x,
    parse_denm_payload_from_v2x,
    serialize_cam_v2x_message,
    serialize_cpm_v2x_message,
    serialize_denm_v2x_message,
)
from cooperative_awareness import (
    any_pedestrian_in_crossing_zone,
    assess_ego_path_conflict,
    max_pedestrian_speed_planar_ms,
    min_pedestrian_distance_to_crossings,
)
from ego_navigation import greedy_lane_follow_control, simple_pursuit_control
from intent_prediction import load_rf_payload
from rsu_perception import (
    bbox_color_bgr_for_id,
    crossing_world_polys_to_zones_scaled,
    crosswalk_radius_m,
    detection_to_ped_dict,
    draw_pose_skeleton_bgr,
    extract_tracked_pose_detections,
    min_distance_to_any_crossing_zone,
    new_yolo_track_state,
    pixel_foot_to_world_xy,
    point_in_any_crossing_zone,
    prune_yolo_track_state,
    rsu_camera_rotation_toward,
    yolo_track_pose,
)
from sim_config import *
from sim_spawn import pedestrian_blueprints

_V2X_TIME_PREC = 8


def _v2x_link_log(
    *,
    sent_this_tick: bool,
    rx_this_tick: bool,
    sent_wall_s: float | None,
    sent_sim_s: float | None,
    rx_wall_s: float | None,
    rx_sim_s: float | None,
    rx_latency_s: float | None,
    msg_seq: int | None,
    last_sent_wall_s: float | None = None,
    last_received_wall_s: float | None = None,
) -> dict:
    return {
        "sent_this_tick": bool(sent_this_tick),
        "received_this_tick": bool(rx_this_tick),
        "sent_wall_s": (
            round(float(sent_wall_s), _V2X_TIME_PREC)
            if sent_this_tick and sent_wall_s is not None
            else None
        ),
        "sent_sim_s": (
            round(float(sent_sim_s), _V2X_TIME_PREC)
            if sent_this_tick and sent_sim_s is not None
            else None
        ),
        "received_wall_s": (
            round(float(rx_wall_s), _V2X_TIME_PREC)
            if rx_this_tick and rx_wall_s is not None
            else None
        ),
        "received_sim_s": (
            round(float(rx_sim_s), _V2X_TIME_PREC)
            if rx_this_tick and rx_sim_s is not None
            else None
        ),
        "latency_wall_s": (
            round(float(rx_latency_s), _V2X_TIME_PREC)
            if rx_this_tick and rx_latency_s is not None
            else None
        ),
        "msg_seq": (
            int(msg_seq)
            if sent_this_tick and msg_seq is not None
            else (int(msg_seq) if rx_this_tick and msg_seq is not None else None)
        ),
        "last_sent_wall_s": (
            None
            if last_sent_wall_s is None
            else round(float(last_sent_wall_s), _V2X_TIME_PREC)
        ),
        "last_received_wall_s": (
            None
            if last_received_wall_s is None
            else round(float(last_received_wall_s), _V2X_TIME_PREC)
        ),
    }


def vision_detection_is_ari_robot(world_pos, ari_robot_actor=None, *, margin_m=None) -> bool:
    """
    True when a YOLO foot projection lies on the ARI robot.
    Excludes the robot from RSU pedestrian perception and downstream hazard logic.
    """
    if world_pos is None:
        return False
    margin_m = ARI_ROBOT_VISION_EXCLUDE_M if margin_m is None else float(margin_m)
    wx, wy = float(world_pos[0]), float(world_pos[1])
    if math.hypot(wx - (-115.72), wy - (-159.81)) <= margin_m:
        return True
    if ari_robot_actor is not None and ari_robot_actor.is_alive:
        loc = ari_robot_actor.get_transform().location
        if math.hypot(wx - loc.x, wy - loc.y) <= margin_m:
            return True
    return False


def main():
    from ultralytics import YOLO

    # hazard_decision_policy.py lives next to this file; ensure it is importable.
    _this_dir = str(Path(__file__).resolve().parent)
    _crosswalk_dir = str(Path(__file__).resolve().parent.parent)
    _pip_dir = str(Path(__file__).resolve().parent.parent / "PedestrianIntentPrediction")
    for p in (_this_dir, _crosswalk_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    build_pie_carla_features = None
    build_pie_carla_heading_features = None
    pip_rf_mode = bool(PIP_RF_MODEL)
    if pip_rf_mode and _pip_dir not in sys.path:
        sys.path.append(_pip_dir)  # append, not insert(0): carla/ modules must not be shadowed
    from hazard_decision_policy import compute_hazard_decision
    from denm_dissemination import hazard_kind, should_transmit_denm
    from ped_state_machine import (
        PHASE_CROSSING,
        PHASE_NOT_CROSSING,
        classify_phase,
        aggregate_ped_phase,
        predict_intent,
    )
    from eval.run_logger import RunLogger
    import importlib
    # Spawn preset module: eval.spawn_presets (interactive) or eval_sim.spawn_presets (thesis batch).
    _spawn_presets_mod = os.environ.get("SIM_SPAWN_PRESETS", "eval.spawn_presets")
    resolve_spawn_config = importlib.import_module(_spawn_presets_mod).resolve_spawn_config

    random.seed(SIM_RANDOM_SEED)
    np.random.seed(SIM_RANDOM_SEED % (2**32))

    spawn_overrides: dict = {}
    if WALKER_SPAWN_X is not None:
        spawn_overrides["WALKER_SPAWN_X"] = WALKER_SPAWN_X
    if WALKER_SPAWN_Y is not None:
        spawn_overrides["WALKER_SPAWN_Y"] = WALKER_SPAWN_Y
    if WALKER_SPAWN_YAW is not None:
        spawn_overrides["WALKER_SPAWN_YAW"] = WALKER_SPAWN_YAW
    if WALKER_SPEED_MS is not None:
        spawn_overrides["WALKER_SPEED_MS"] = WALKER_SPEED_MS
        spawn_overrides["speed_ms"] = WALKER_SPEED_MS
    if DISABLE_WALKER_SPAWN:
        spawn_overrides["DISABLE_WALKER_SPAWN"] = "1"
    spawn_cfg = resolve_spawn_config(WALKER_SPAWN_PRESET or "nearside_off", overrides=spawn_overrides)
    resolved_walker_speed_ms = float(spawn_cfg["speed_ms"])
    disable_walker_spawn = bool(spawn_cfg["disable_spawn"])

    base_scenario_id = SIM_SCENARIO_ID.split("_run")[0] if "_run" in SIM_SCENARIO_ID else SIM_SCENARIO_ID

    actor_list = []
    active_walkers = []
    spawned_walkers = []

    os.makedirs(CAPTURE_DIR, exist_ok=True)
    print(f"[simulation] RSU annotated frames saved every {ANALYSIS_INTERVAL_S}s to: {CAPTURE_DIR}")
    print(f"[simulation] Open latest run folder under simulation_captures/ or eval rsu_sequences/")
    print(
        f"[simulation] Validation mode={SIM_VALIDATION_MODE} seed={SIM_RANDOM_SEED} "
        f"spawn_preset={spawn_cfg.get('preset')}"
    )
    print(
        "[simulation] Ego CAM → RSU/robot via sensor.other.v2x_custom "
        "(HF speed + basic-container ref position; RSU derives crossing distance locally). "
        "RF v_ego_kmh uses received CAM speed, not ego ground truth."
    )

    pygame.init()
    pygame.display.set_mode((400, 300))
    pygame.display.set_caption("CARLA - YOLO Pose + RF Intent + V2X")

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    print(
        f"[simulation] YOLO: model={YOLO_MODEL_NAME} imgsz={DET_IMGSZ} conf={DET_CONF} nms_iou={YOLO_NMS_IOU} "
        f"new_track_min={CONF_NEW_TRACK_MIN} min_bbox_h={MIN_BBOX_H_PX} track_age={MIN_TRACK_AGE_FRAMES} "
        f"debug_raw={YOLO_DEBUG_RAW} gate_debug={YOLO_GATE_DEBUG} "
        f"tracker={YOLO_TRACKER} "
        f"device={YOLO_DEVICE or 'auto'} half={YOLO_HALF}"
    )
    yolo_model = YOLO(YOLO_MODEL_NAME)
    if pip_rf_mode:
        _rf_load_path = PIP_RF_MODEL
    elif RF_MODEL_PATH:
        _rf_load_path = RF_MODEL_PATH
    else:
        _rf_load_path = ""
    rf_payload = load_rf_payload(_rf_load_path) if _rf_load_path else None
    rf_model = rf_payload["model"] if rf_payload else None
    rf_dim = int(rf_payload["feature_dim"]) if rf_payload else None
    _env_threshold = os.environ.get("PIP_RF_THRESHOLD", "").strip()
    rf_threshold = float(_env_threshold) if _env_threshold else (
        float(rf_payload.get("threshold", 0.5)) if rf_payload else 0.5
    )
    rf_model_type = str(rf_payload.get("model_type", "legacy")) if rf_payload else "legacy"
    if rf_payload and pip_rf_mode:
        if rf_model_type in ("ma_rong_pie_7", "ma_rong_pie_9_heading"):
            if rf_model_type == "ma_rong_pie_9_heading":
                from pie.carla_adapter import (
                    build_pie_carla_heading_features as _build_pie_carla_heading_features,
                )

                build_pie_carla_heading_features = _build_pie_carla_heading_features
                print(
                    f"[simulation] PIE Ma-Rong 9-feature heading RF: path={_rf_load_path} "
                    f"dim={rf_dim} threshold={rf_threshold}",
                    flush=True,
                )
            else:
                from pie.carla_adapter import build_pie_carla_features as _build_pie_carla_features

                build_pie_carla_features = _build_pie_carla_features
                print(
                    f"[simulation] PIE Ma-Rong 7-feature RF: path={_rf_load_path} dim={rf_dim} "
                    f"threshold={rf_threshold}",
                    flush=True,
                )
    elif rf_model is not None:
        print(
            f"[simulation] Legacy RF enabled: path={_rf_load_path} dim={rf_dim}",
            flush=True,
        )
    else:
        print(
            "[simulation] RF intent disabled (set PIP_RF_MODEL or RF_INTENT_MODEL).",
            flush=True,
        )

    ped_history_feat = {}
    ped_hist_prev_joints = {}
    yolo_track_state = new_yolo_track_state()
    capture_idx = 0
    rsu_capture_dir = Path(CAPTURE_DIR)
    rsu_frames_manifest: list = []

    img_queue = queue.Queue(maxsize=4)
    world = None
    run_logger = None
    rgb_camera = None
    robot_v2x = None
    ego_v2x = None
    rsu_v2x = None
    v2x_counter = 0

    class _ConsoleHud:
        def notification(self, text, seconds=4.0):
            print(text, flush=True)

    _hud = _ConsoleHud()

    # DENM rate-limit: repetition interval; hazard edge triggers DENM(new) immediately.
    _denm_last_sent_time: float = 0.0
    _denm_hazard_was_active: bool = False
    _denm_last_hazard_kind: str = ""
    _DENM_SEND_INTERVAL_S: float = 1.0
    _robot_cam_speed_ms = None
    _robot_cam_x = None
    _robot_cam_y = None
    _rsu_cam_speed_ms = None
    _rsu_cam_x = None
    _rsu_cam_y = None
    _rsu_cam_dist_m = None
    _cam_sent_seq: int = 0
    _cam_sent_wall_s: float | None = None
    _cam_sent_sim_s: float | None = None
    _rsu_cam_rx_wall_s: float | None = None
    _rsu_cam_rx_sim_s: float | None = None
    _rsu_cam_rx_latency_s: float | None = None
    _rsu_cam_rx_seq: int | None = None
    _robot_cam_rx_wall_s: float | None = None
    _robot_cam_rx_sim_s: float | None = None
    _robot_cam_rx_latency_s: float | None = None
    _robot_cam_rx_seq: int | None = None
    _cam_sent_this_tick = False
    _rsu_cam_rx_this_tick = False
    _robot_cam_rx_this_tick = False

    _denm_sent_seq: int = 0
    _denm_sent_wall_s: float | None = None
    _denm_sent_sim_s: float | None = None
    _denm_last_log_meta: dict = {}
    _ego_denm_rx_wall_s: float | None = None
    _ego_denm_rx_sim_s: float | None = None
    _ego_denm_rx_latency_s: float | None = None
    _ego_denm_rx_seq: int | None = None
    _ego_denm_rx_this_tick = False
    _robot_denm_rx_wall_s: float | None = None
    _robot_denm_rx_sim_s: float | None = None
    _robot_denm_rx_latency_s: float | None = None
    _robot_denm_rx_seq: int | None = None
    _robot_denm_rx_this_tick = False

    _cpm_sent_seq: int = 0
    _cpm_sent_wall_s: float | None = None
    _cpm_sent_sim_s: float | None = None
    _cpm_last_log_meta: dict = {}
    _ego_cpm_rx_wall_s: float | None = None
    _ego_cpm_rx_sim_s: float | None = None
    _ego_cpm_rx_latency_s: float | None = None
    _ego_cpm_rx_seq: int | None = None
    _ego_cpm_rx_this_tick = False

    # CPM generation state (ETSI TS 103 324 object inclusion).
    _cam_last_gen_time: float = 0.0
    _cpm_last_gen_time: float = 0.0
    _cpm_obj_last_included: dict = {}

    # Ego OBU V2X RX state (received DENM + CPM from RSU).
    _ego_cpm_objects: list = []
    _ego_denm_human_presence_until: float = 0.0
    _ego_denm_object_id: int | None = None
    _ego_denm_event_pos: dict = {}

    # Robot V2X RX state (DENM + ego CAM on v2x_custom).
    _robot_denm_human_presence_until: float = 0.0
    _robot_denm_cause_code: int | None = None
    _robot_denm_sub_cause_code: int | None = None
    # Previous robot light level — carries over across ticks for hysteresis.
    _prev_light_level: str = "NONE"

    def on_camera_image(image):
        try:
            image.convert(carla.ColorConverter.Raw)
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = np.reshape(arr, (image.height, image.width, 4))
            bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            if img_queue.full():
                try:
                    img_queue.get_nowait()
                except queue.Empty:
                    pass
            img_queue.put_nowait(bgr.copy())
        except Exception:
            pass

    try:
        collision_sensor = None
        print("Loading map 'trail24'...")
        world = client.load_world("trail24")

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        bp_library = world.get_blueprint_library()
        v2x_bp = bp_library.find("sensor.other.v2x_custom")
        _configure_v2x_custom_blueprint(v2x_bp)

        print("Generating background traffic...")
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        vehicle_bps = bp_library.filter("vehicle.*")
        vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute("number_of_wheels")) == 4]

        num_traffic_vehicles = 30
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= num_traffic_vehicles:
                break
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            bp.set_attribute("role_name", "autopilot")

            batch.append(
                SpawnActor(bp, transform).then(
                    SetAutopilot(FutureActor, True, traffic_manager.get_port())
                )
            )

        results = client.apply_batch_sync(batch, True)
        for res in results:
            if not res.error:
                actor = world.get_actor(res.actor_id)
                actor_list.append(actor)
        print(f"Spawned {len([r for r in results if not r.error])} background vehicles.", flush=True)

        print(
            f"[simulation] Sync bootstrap: {CARLA_BOOTSTRAP_TICKS} x world.tick() "
            "(lets the map render / avoids sync deadlock before spawns continue)...",
            flush=True,
        )
        for _ in range(CARLA_BOOTSTRAP_TICKS):
            world.tick()

        print("[simulation] Spawning ego vehicle...", flush=True)
        car_loc = carla.Location(x=-142.0, y=-175.9, z=364.90 + 0.5)
        car_rot = carla.Rotation(yaw=0.35)
        car_bp = bp_library.find("vehicle.tesla.model3")
        vehicle = world.try_spawn_actor(car_bp, carla.Transform(car_loc, car_rot))
        nav_agent = None
        ego_python_mode = None  # None | "lane" | "simple" — when BasicAgent not used
        if vehicle:
            actor_list.append(vehicle)
            print("[simulation] Ego vehicle spawned.", flush=True)
        else:
            print(
                "[simulation] WARNING: Ego vehicle failed to spawn (collision?). "
                "Navigation disabled.",
                flush=True,
            )

        # ARI robot: vehicle.pal.ari_robot
        ari_vehicle = None
        ari_bp = bp_library.find("vehicle.pal.ari_robot")
        ari_vehicle = world.try_spawn_actor(
            ari_bp,
            carla.Transform(
                carla.Location(x=-115.72, y=-159.81, z=364.9),
                carla.Rotation(yaw=0),
            ),
        )
        if ari_vehicle:
            actor_list.append(ari_vehicle)
            active_walkers.append(ari_vehicle)
            print("[INFO] ARI spawned", flush=True)
        else:
            print("[WARN] ARI missing (ignored)", flush=True)
        world.tick()

        crossing_zone_polygons = []  # populated below; CAM RX may fire before that

        # Spawn initial walker(s) (pedestrian).
        walker_bp = bp_library.find(WALKER_BP_DEFAULT)
        walker_specs = []
        if not disable_walker_spawn:
            spawn_specs = [
                {
                    "x": float(spawn_cfg["x"]),
                    "y": float(spawn_cfg["y"]),
                    "yaw": float(spawn_cfg["yaw"]),
                    "speed_ms": float(spawn_cfg.get("speed_ms", resolved_walker_speed_ms)),
                }
            ]
            spawn_specs.extend(spawn_cfg.get("extra_walkers") or [])

            ped_bps = pedestrian_blueprints(bp_library, len(spawn_specs))
            for idx, spec in enumerate(spawn_specs):
                walker_loc = carla.Location(
                    x=float(spec["x"]),
                    y=float(spec["y"]),
                    z=365.34 + 0.5,
                )
                walker_rot = carla.Rotation(yaw=float(spec["yaw"]))
                bp = ped_bps[idx % len(ped_bps)]
                spawned = world.try_spawn_actor(bp, carla.Transform(walker_loc, walker_rot))
                if spawned:
                    actor_list.append(spawned)
                    active_walkers.append(spawned)
                    spawned_walkers.append(spawned)
                    walker_specs.append({
                        "actor": spawned,
                        "rot": walker_rot,
                        "speed_ms": float(spec.get("speed_ms", resolved_walker_speed_ms)),
                    })
                    print(
                        f"[simulation] Walker spawned preset={spawn_cfg.get('preset')} "
                        f"bp={bp.id} at ({spec['x']}, {spec['y']}) yaw={spec['yaw']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[simulation] WARNING: Walker {idx + 1} failed to spawn "
                        f"at ({spec['x']}, {spec['y']}).",
                        flush=True,
                    )
            if not spawned_walkers:
                print("[simulation] WARNING: No pedestrian walkers spawned.", flush=True)
        else:
            walker_loc = carla.Location(x=-100.26, y=-156.91, z=365.34 + 0.5)
            walker_rot = carla.Rotation(yaw=180.0)
            print("[simulation] Walker spawn disabled (preset=none or DISABLE_WALKER_SPAWN).", flush=True)

        # V2X sensors.

        def _cam_rx_sim_time_s() -> float | None:
            try:
                return float(world.get_snapshot().timestamp.elapsed_seconds)
            except Exception:
                return None

        def robot_v2x_callback(event):
            nonlocal _robot_denm_human_presence_until
            nonlocal _robot_denm_cause_code, _robot_denm_sub_cause_code
            nonlocal _robot_cam_speed_ms, _robot_cam_x, _robot_cam_y
            nonlocal _robot_cam_rx_wall_s, _robot_cam_rx_sim_s
            nonlocal _robot_cam_rx_latency_s, _robot_cam_rx_seq
            nonlocal _robot_cam_rx_this_tick
            nonlocal _robot_denm_rx_wall_s, _robot_denm_rx_sim_s
            nonlocal _robot_denm_rx_latency_s, _robot_denm_rx_seq
            nonlocal _robot_denm_rx_this_tick
            try:
                for msg in event:
                    text = msg.get()
                    power = msg.power
                    cam = parse_cam_payload_from_v2x(text)
                    if cam is not None:
                        _rx_wall = time.time()
                        if cam.get("speed_ms") is not None:
                            _robot_cam_speed_ms = cam["speed_ms"]
                        if cam.get("x") is not None:
                            _robot_cam_x = cam["x"]
                        if cam.get("y") is not None:
                            _robot_cam_y = cam["y"]
                        _robot_cam_rx_wall_s = _rx_wall
                        _robot_cam_rx_sim_s = _cam_rx_sim_time_s()
                        _robot_cam_rx_seq = _cam_sent_seq if _cam_sent_seq > 0 else None
                        _robot_cam_rx_latency_s = (
                            (_rx_wall - _cam_sent_wall_s)
                            if _cam_sent_wall_s is not None
                            else None
                        )
                        _robot_cam_rx_this_tick = True
                        continue
                    denm = parse_denm_payload_from_v2x(text)
                    if denm is None:
                        continue
                    if denm.get("termination"):
                        _robot_denm_human_presence_until = 0.0
                        _robot_denm_cause_code = None
                        _robot_denm_sub_cause_code = None
                        print(
                            f"[V2X ROBOT RX DENM] cancellation | Power: {power:.2f} dBm",
                            flush=True,
                        )
                        continue
                    if int(denm.get("causeCode", 0)) in (
                        ETSI_CAUSE_HUMAN_PRESENCE,
                        ETSI_CAUSE_COLLISION_RISK,
                    ):
                        _rx_wall = time.time()
                        validity = float(denm.get("validityDuration", DENM_VALIDITY_DURATION_S))
                        _robot_denm_human_presence_until = _rx_wall + validity
                        _robot_denm_cause_code = int(denm.get("causeCode", 0))
                        sub = denm.get("subCauseCode")
                        _robot_denm_sub_cause_code = int(sub) if sub is not None else None
                        _robot_denm_rx_wall_s = _rx_wall
                        _robot_denm_rx_sim_s = _cam_rx_sim_time_s()
                        _robot_denm_rx_seq = _denm_sent_seq if _denm_sent_seq > 0 else None
                        _robot_denm_rx_latency_s = (
                            (_rx_wall - _denm_sent_wall_s)
                            if _denm_sent_wall_s is not None
                            else None
                        )
                        _robot_denm_rx_this_tick = True
                        print(
                            f"[V2X ROBOT RX DENM] new/repetition cause={denm.get('causeCode')} "
                            f"sub={denm.get('subCauseCode')} validity={validity}s | Power: {power:.2f} dBm",
                            flush=True,
                        )
            except Exception as e:
                print(f"[V2X] Robot callback error: {e}", flush=True)

        def ego_v2x_callback(event):
            nonlocal _ego_denm_human_presence_until, _ego_denm_object_id, _ego_denm_event_pos
            nonlocal _ego_cpm_objects
            nonlocal _ego_cpm_rx_wall_s, _ego_cpm_rx_sim_s
            nonlocal _ego_cpm_rx_latency_s, _ego_cpm_rx_seq, _ego_cpm_rx_this_tick
            nonlocal _ego_denm_rx_wall_s, _ego_denm_rx_sim_s
            nonlocal _ego_denm_rx_latency_s, _ego_denm_rx_seq, _ego_denm_rx_this_tick
            try:
                for msg in event:
                    text = msg.get()
                    power = msg.power
                    cpm = parse_cpm_payload_from_v2x(text)
                    if cpm is not None:
                        _rx_wall = time.time()
                        _ego_cpm_objects = cpm["perceivedObjects"]
                        _ego_cpm_rx_wall_s = _rx_wall
                        _ego_cpm_rx_sim_s = _cam_rx_sim_time_s()
                        _ego_cpm_rx_seq = _cpm_sent_seq if _cpm_sent_seq > 0 else None
                        _ego_cpm_rx_latency_s = (
                            (_rx_wall - _cpm_sent_wall_s)
                            if _cpm_sent_wall_s is not None
                            else None
                        )
                        _ego_cpm_rx_this_tick = True
                        print(
                            f"[V2X EGO RX CPM] objects={len(_ego_cpm_objects)} | Power: {power:.2f} dBm",
                            flush=True,
                        )
                        continue
                    denm = parse_denm_payload_from_v2x(text)
                    if denm is None:
                        continue
                    if denm.get("termination"):
                        _ego_denm_human_presence_until = 0.0
                        _ego_denm_object_id = None
                        _ego_denm_event_pos = {}
                        print(
                            f"[V2X EGO RX DENM] cancellation | Power: {power:.2f} dBm",
                            flush=True,
                        )
                        continue
                    if int(denm.get("causeCode", 0)) in (
                        ETSI_CAUSE_HUMAN_PRESENCE,
                        ETSI_CAUSE_COLLISION_RISK,
                    ):
                        _rx_wall = time.time()
                        validity = float(denm.get("validityDuration", DENM_VALIDITY_DURATION_S))
                        _ego_denm_human_presence_until = _rx_wall + validity
                        _ego_denm_object_id = denm.get("objectID")
                        _ego_denm_event_pos = denm.get("eventPosition") or {}
                        _ego_denm_rx_wall_s = _rx_wall
                        _ego_denm_rx_sim_s = _cam_rx_sim_time_s()
                        _ego_denm_rx_seq = _denm_sent_seq if _denm_sent_seq > 0 else None
                        _ego_denm_rx_latency_s = (
                            (_rx_wall - _denm_sent_wall_s)
                            if _denm_sent_wall_s is not None
                            else None
                        )
                        _ego_denm_rx_this_tick = True
                        print(
                            f"[V2X EGO RX DENM] cause={denm.get('causeCode')} "
                            f"sub={denm.get('subCauseCode')} objectID={_ego_denm_object_id} "
                            f"validity={validity}s | Power: {power:.2f} dBm",
                            flush=True,
                        )
            except Exception as e:
                print(f"[V2X] Ego callback error: {e}", flush=True)

        def rsu_v2x_callback(event):
            nonlocal _rsu_cam_speed_ms, _rsu_cam_x, _rsu_cam_y, _rsu_cam_dist_m
            nonlocal _rsu_cam_rx_wall_s, _rsu_cam_rx_sim_s
            nonlocal _rsu_cam_rx_latency_s, _rsu_cam_rx_seq
            nonlocal _rsu_cam_rx_this_tick
            try:
                for msg in event:
                    text = msg.get()
                    cam = parse_cam_payload_from_v2x(text)
                    if cam is None:
                        continue
                    _rx_wall = time.time()
                    if cam.get("speed_ms") is not None:
                        _rsu_cam_speed_ms = cam["speed_ms"]
                    if cam.get("x") is not None:
                        _rsu_cam_x = cam["x"]
                    if cam.get("y") is not None:
                        _rsu_cam_y = cam["y"]
                    if (
                        _rsu_cam_x is not None
                        and _rsu_cam_y is not None
                        and crossing_zone_polygons
                    ):
                        _rsu_cam_dist_m = min_distance_to_any_crossing_zone(
                            _rsu_cam_x, _rsu_cam_y, crossing_zone_polygons,
                        )
                    _rsu_cam_rx_wall_s = _rx_wall
                    _rsu_cam_rx_sim_s = _cam_rx_sim_time_s()
                    _rsu_cam_rx_seq = _cam_sent_seq if _cam_sent_seq > 0 else None
                    _rsu_cam_rx_latency_s = (
                        (_rx_wall - _cam_sent_wall_s)
                        if _cam_sent_wall_s is not None
                        else None
                    )
                    _rsu_cam_rx_this_tick = True
            except Exception as e:
                print(f"[V2X] RSU callback error: {e}", flush=True)

        if ari_vehicle is not None:
            try:
                robot_v2x = world.spawn_actor(
                    v2x_bp, carla.Transform(), attach_to=ari_vehicle
                )
                robot_v2x.listen(robot_v2x_callback)
                actor_list.append(robot_v2x)
                print("[V2X] Robot V2X sensor attached and listening.", flush=True)
            except Exception as ex:
                robot_v2x = None
                print(f"[V2X] Robot V2X attach failed: {ex}", flush=True)

        if vehicle is not None:
            try:
                ego_v2x = world.spawn_actor(
                    v2x_bp, carla.Transform(), attach_to=vehicle
                )
                ego_v2x.listen(ego_v2x_callback)
                actor_list.append(ego_v2x)
                print("[V2X] Ego V2X attached (DENM+CPM RX, CAM TX via v2x_custom).", flush=True)
            except Exception as ex:
                ego_v2x = None
                print(f"[V2X] Ego V2X attach failed: {ex}", flush=True)

        # RSU + robot (ARI) world coords — RSU base matches mainv2.py rsu_transform
        rsu_x, rsu_y, rsu_z = RSU_X, RSU_Y, RSU_Z
        rsu_sensor_ident = carla.Location(x=rsu_x, y=rsu_y, z=rsu_z + 7.0)

        ari_x, ari_y, ari_z = -115.72, -159.81, 364.5
        ari_light_loc = carla.Location(x=ari_x, y=ari_y, z=ari_z + 2.2)

        rsu_prop = None
        try:
            rsu_prop_bp = bp_library.find("static.prop.electricpole")
            rsu_base_tf = carla.Transform(
                carla.Location(x=rsu_x, y=rsu_y, z=rsu_z),
                # carla.Rotation(yaw=RSU_YAW),
            )
            rsu_prop = world.try_spawn_actor(rsu_prop_bp, rsu_base_tf)
            if rsu_prop:
                print("[simulation] RSU spawned (static.prop.electricpole).", flush=True)
        except IndexError:
            print(
                "[simulation] Blueprint static.prop.electricpole missing — RSU prop skipped.",
                flush=True,
            )

        cam_bp = bp_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAM_IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(CAM_IMAGE_H))
        cam_bp.set_attribute("fov", str(CAM_FOV))
        cam_loc = carla.Location(x=rsu_x, y=rsu_y, z=rsu_z + CAM_REL_Z)
        if RSU_MANUAL_CAM:
            cam_pitch, cam_yaw = CAM_PITCH, CAM_YAW
        else:
            cam_pitch, cam_yaw = rsu_camera_rotation_toward(
                rsu_x,
                rsu_y,
                rsu_z + CAM_REL_Z,
                RSU_LOOK_AT_X,
                RSU_LOOK_AT_Y,
                RSU_LOOK_AT_Z,
                yaw_offset_deg=CAM_YAW_OFFSET,
                pitch_offset_deg=CAM_PITCH_OFFSET,
            )
        cam_tf = carla.Transform(
            cam_loc,
            carla.Rotation(pitch=cam_pitch, yaw=cam_yaw),
        )
        rgb_camera = world.try_spawn_actor(cam_bp, cam_tf)
        if rgb_camera:
            print(
                f"[simulation] RSU camera pitch={cam_pitch:.1f}° yaw={cam_yaw:.1f}° "
                f"fov={CAM_FOV:.0f}° look_at=({RSU_LOOK_AT_X},{RSU_LOOK_AT_Y})",
                flush=True,
            )
        if rgb_camera:
            rgb_camera.listen(on_camera_image)
            actor_list.append(rgb_camera)

        # Destroy RSU prop after attached camera (CARLA parent-child cleanup order)
        if rsu_prop:
            actor_list.append(rsu_prop)

        # Attach RSU V2X sensor to the RSU prop (or place at world coords if prop is missing)
        try:
            if rsu_prop is not None:
                rsu_v2x = world.spawn_actor(
                    v2x_bp, carla.Transform(), attach_to=rsu_prop
                )
            else:
                rsu_v2x = world.spawn_actor(
                    v2x_bp,
                    carla.Transform(carla.Location(x=rsu_x, y=rsu_y, z=rsu_z + CAM_REL_Z)),
                )
            rsu_v2x.listen(rsu_v2x_callback)
            actor_list.append(rsu_v2x)
            print("[V2X] RSU V2X sensor attached (ego CAM RX via v2x_custom).", flush=True)
        except Exception as ex:
            print(f"[V2X] RSU V2X attach failed: {ex}", flush=True)

        spectator = world.get_spectator()
        warm_hold = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True,
            reverse=False,
        )

        print(
            f"[simulation] View warm-up: {CARLA_WARMUP_TICKS} x world.tick() "
            f"(sync mode — map should stream in; ego held)...",
            flush=True,
        )
        for wi in range(CARLA_WARMUP_TICKS):
            if ari_vehicle is not None and ari_vehicle.is_alive:
                ari_vehicle.apply_control(warm_hold)
            if vehicle is not None and vehicle.is_alive:
                vehicle.apply_control(warm_hold)
                v_tf = vehicle.get_transform()
                spectator.set_transform(
                    carla.Transform(
                        v_tf.location
                        + carla.Location(z=12.0)
                        + v_tf.get_forward_vector() * -15.0,
                        carla.Rotation(pitch=-20.0, yaw=v_tf.rotation.yaw),
                    )
                )
            world.tick()
            if wi == 25 or wi == CARLA_WARMUP_TICKS // 2:
                print(f"[simulation]   warm-up progress: {wi}/{CARLA_WARMUP_TICKS} ticks...", flush=True)

        if vehicle is None:
            print("[simulation] Skipping ego navigation (no ego vehicle).", flush=True)
        elif EGO_NAV_MODE in ("lane", "road", "greedy"):
            ego_python_mode = "lane"
            print(
                f"[simulation] EGO_NAV_MODE={EGO_NAV_MODE}: greedy lane follow (OpenDRIVE waypoints), "
                f"lookahead={EGO_LANE_LOOKAHEAD} m, arrival={EGO_SIMPLE_ARRIVE_DIST} m.",
                flush=True,
            )
        elif EGO_NAV_MODE in ("simple", "pursuit", "direct"):
            ego_python_mode = "simple"
            print(
                f"[simulation] EGO_NAV_MODE={EGO_NAV_MODE}: straight-line pursuit (ignores road centerlines). "
                f"Arrival radius={EGO_SIMPLE_ARRIVE_DIST} m.",
                flush=True,
            )
        elif EGO_NAV_MODE == "basic":
            try:
                from agents.navigation.basic_agent import BasicAgent
                from agents.navigation.global_route_planner import GlobalRoutePlanner
            except ImportError as ex:
                print(
                    "[simulation] EGO_NAV_MODE=basic requires agents.navigation. "
                    "Set CARLA_ROOT or CARLA_PYTHONAPI_CARLA. Falling back to lane follow.",
                    flush=True,
                )
                ego_python_mode = "lane"
            else:
                print(
                    "[simulation] Building navigation (GlobalRoutePlanner + BasicAgent). "
                    "Temporarily disabling sync mode so the viewport can update during the "
                    f"road-graph build (sampling={CARLA_GRP_SAMPLING} m)...",
                    flush=True,
                )
                carla_map = world.get_map()
                _nav_settings = world.get_settings()
                try:
                    _nav_settings.synchronous_mode = False
                    world.apply_settings(_nav_settings)
                    traffic_manager.set_synchronous_mode(False)

                    t0 = time.time()
                    grp = GlobalRoutePlanner(carla_map, CARLA_GRP_SAMPLING)
                    print(f"[simulation]   topology OK in {time.time() - t0:.1f} s", flush=True)

                    t0 = time.time()
                    nav_agent = BasicAgent(
                        vehicle,
                        target_speed=NAV_TARGET_SPEED_KMH,
                        map_inst=carla_map,
                        grp_inst=grp,
                    )
                    print(f"[simulation]   BasicAgent OK in {time.time() - t0:.1f} s", flush=True)

                    t0 = time.time()
                    nav_agent.set_destination(EGO_DESTINATION)
                    print(f"[simulation]   set_destination OK in {time.time() - t0:.1f} s", flush=True)
                    print(
                        f"[simulation] Autonomous navigation ready (basic) -> "
                        f"({EGO_DESTINATION.x:.2f}, {EGO_DESTINATION.y:.2f}, {EGO_DESTINATION.z:.2f}) m "
                        f"@ {NAV_TARGET_SPEED_KMH:.0f} km/h",
                        flush=True,
                    )
                except Exception as ex:
                    print(
                        f"[simulation] BasicAgent / planner failed ({ex}); "
                        "falling back to lane follow.",
                        flush=True,
                    )
                    nav_agent = None
                    ego_python_mode = "lane"
                finally:
                    _nav_settings = world.get_settings()
                    _nav_settings.synchronous_mode = True
                    _nav_settings.fixed_delta_seconds = 0.05
                    world.apply_settings(_nav_settings)
                    traffic_manager.set_synchronous_mode(True)
                    print("[simulation] Synchronous mode restored (fixed_delta_seconds=0.05).", flush=True)
        else:
            print(
                f"[simulation] Unknown EGO_NAV_MODE={EGO_NAV_MODE!r}; using lane follow.",
                flush=True,
            )
            ego_python_mode = "lane"

        for _ in range(max(5, CARLA_BOOTSTRAP_TICKS // 5)):
            world.tick()

        '''
        # World crossing zones (trail24): drawn in yellow for visibility.
        '''
        crossing_zone_1 = [
            (-115.0, -159.9),
            (-109.4, -159.9),
            (-109.19, -155.15),
            (-114.49, -155.15)
        ]
        # Temporarily disabled: zone closer to RSU (-96.20, -98.00).
        '''
        crossing_zone_2 = [
            (-106.4, -159.9),
            (-101.6, -159.9),
            (-102.29, -155.15),
            (-106.99, -155.15)
        ]
        '''

        crossing_zone_polygons = [crossing_zone_1]
        crossing_zone_z = 364.5

        rsu_stop_control = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=False,
            reverse=False,
        )
        clock = pygame.time.Clock()

        sim_start_time = time.time()
        last_spawn_time = sim_start_time
        walker_started = False

        walker_controls = []
        for spec in walker_specs:
            ctrl = carla.WalkerControl()
            ctrl.direction = spec["rot"].get_forward_vector()
            ctrl.speed = 0.0
            walker_controls.append((spec["actor"], ctrl, spec["speed_ms"]))

        run_logger = None
        if SIM_RUN_LOGGER:
            run_logger = RunLogger(
                CAPTURE_DIR,
                scenario_id=base_scenario_id,
                metadata={
                    "random_seed": SIM_RANDOM_SEED,
                    "spawn_preset": spawn_cfg.get("preset"),
                    "spawn_x": spawn_cfg.get("x"),
                    "spawn_y": spawn_cfg.get("y"),
                    "spawn_yaw": spawn_cfg.get("yaw"),
                    "walker_speed_ms": resolved_walker_speed_ms,
                    "sim_scenario_id": SIM_SCENARIO_ID,
                    "rf_model_id": os.environ.get("SIM_RF_MODEL_ID", ""),
                    "pip_rf_model": PIP_RF_MODEL or None,
                    "rf_intent_model": RF_MODEL_PATH or None,
                    "rf_threshold": rf_threshold,
                    "rf_model_type": rf_model_type,
                },
            )
            if SIM_VALIDATION_MODE:
                rsu_capture_dir = run_logger.run_dir / "rsu"
                rsu_capture_dir.mkdir(parents=True, exist_ok=True)
                run_logger._metadata["image_dir"] = str(rsu_capture_dir)
            print(f"[simulation] JSONL run log: {run_logger.path}", flush=True)
            run_logger.write_metadata()

        collision_this_tick = False

        def _on_collision(event):
            nonlocal collision_this_tick
            collision_this_tick = True
            if run_logger is not None:
                try:
                    run_logger.record_collision(
                        sim_time_s=float(event.timestamp),
                        other_actor_id=int(event.other_actor.id),
                    )
                except Exception:
                    pass

        if vehicle and vehicle.is_alive:
            try:
                col_bp = bp_library.find("sensor.other.collision")
                collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
                collision_sensor.listen(_on_collision)
                actor_list.append(collision_sensor)
                print("[simulation] Collision sensor attached to ego.", flush=True)
            except Exception as ex:
                print(f"[simulation] Collision sensor attach failed: {ex}", flush=True)

        _fault_drop_remaining = 0
        _fault_intent_spike_used = False
        _cam_speed_frozen_until = 0.0
        _cam_speed_frozen_value = None
        _target_walker = spawned_walkers[0] if spawned_walkers else None

        print(
            f"AUTONOMOUS EGO — nav={EGO_NAV_MODE} (lane=waypoints / simple=beeline / basic=agent) -> BP_PowerPole2; "
            "RSU master (range+TTC+intent) -> RED / YELLOW+yield / GREEN | V2X: robot+ego sensors active",
            flush=True,
        )
        debug = world.debug

        last_analysis_time = 0.0

        pedestrian_detected = False
        intent_predicted = False
        intent_proba = None
        max_crossing_intent_proba = None
        ped_on_crossing_vision = False
        ped_phase = PHASE_NOT_CROSSING
        tracked_pedestrians_report = []
        _hazard_decision_debug_last_print = 0.0

        # Main loop: sync tick → optional RSU analysis → hazard policy → V2X TX/RX → ego/robot actuation.
        while True:
            clock.tick(60)
            _cam_sent_this_tick = False
            _rsu_cam_rx_this_tick = False
            _robot_cam_rx_this_tick = False
            _ego_denm_rx_this_tick = False
            _robot_denm_rx_this_tick = False
            _ego_cpm_rx_this_tick = False
            world.tick()
            snap = world.get_snapshot()
            sim_time_s = snap.timestamp.elapsed_seconds

            if SIM_MAX_DURATION_S > 0 and sim_time_s >= SIM_MAX_DURATION_S:
                print(
                    f"[simulation] SIM_MAX_DURATION_S={SIM_MAX_DURATION_S} reached — stopping.",
                    flush=True,
                )
                break

            current_time = time.time()
            elapsed_time = current_time - sim_start_time

            if not walker_started and sim_time_s >= WALKER_START_DELAY_S:
                for _actor, ctrl, speed_ms in walker_controls:
                    ctrl.speed = speed_ms
                walker_started = True

            if (
                not SIM_VALIDATION_MODE
                and (current_time - last_spawn_time) >= 20.0
            ):
                variant_bps = pedestrian_blueprints(bp_library, 10)
                new_walker_bp = random.choice(variant_bps) if variant_bps else walker_bp
                new_walker = world.try_spawn_actor(
                    new_walker_bp, carla.Transform(walker_loc, walker_rot)
                )
                if new_walker:
                    actor_list.append(new_walker)
                    active_walkers.append(new_walker)
                    spawned_walkers.append(new_walker)
                last_spawn_time = current_time

            for w, ctrl, _speed_ms in walker_controls:
                if w.is_alive:
                    w.apply_control(ctrl)

            if vehicle and vehicle.is_alive:
                v_loc = vehicle.get_transform().location
                v_velocity = vehicle.get_velocity()
                v_speed = math.hypot(v_velocity.x, v_velocity.y)
                d_ego_to_crossing_m = min_distance_to_any_crossing_zone(
                    v_loc.x, v_loc.y, crossing_zone_polygons
                )
            else:
                v_speed = 0.0
                d_ego_to_crossing_m = float("inf")

            # Zone metrics: human walkers only — ARI robot is parked inside zone 1 and
            # must not set ped_in_crossing_zone / d_ped (false CRITICAL braking).
            d_ped_to_crossing_m = min_pedestrian_distance_to_crossings(
                spawned_walkers, crossing_zone_polygons
            )
            ped_in_crossing_zone = (
                ped_on_crossing_vision
                or any_pedestrian_in_crossing_zone(
                    spawned_walkers, crossing_zone_polygons
                )
            )

            # RSU perception + RF intent (rate-limited by ANALYSIS_INTERVAL_S, default 0.25 s).
            if current_time - last_analysis_time >= ANALYSIS_INTERVAL_S:
                last_analysis_time = current_time
                pedestrian_detected = False
                intent_predicted = False
                intent_proba = None
                max_crossing_intent_proba = None
                ped_on_crossing_vision = False
                tracked_pedestrians_report = []

                _skip_analysis_yolo = False
                if FAULT_DROP_DETECTIONS_N > 0:
                    if _fault_drop_remaining == 0:
                        _fault_drop_remaining = FAULT_DROP_DETECTIONS_N
                    if _fault_drop_remaining > 0:
                        _skip_analysis_yolo = True
                        _fault_drop_remaining -= 1

                bgr = None
                try:
                    bgr = img_queue.get_nowait()
                except queue.Empty:
                    pass

                if bgr is not None and not _skip_analysis_yolo:
                    h, w = bgr.shape[:2]
                    zones_scaled = crossing_world_polys_to_zones_scaled(
                        crossing_zone_polygons,
                        crossing_zone_z,
                        rgb_camera,
                        w,
                        h,
                        CAM_FOV,
                    )
                    yolo_track_state["analysis_seq"] = int(yolo_track_state["analysis_seq"]) + 1
                    stale_ids = prune_yolo_track_state(
                        yolo_track_state,
                        max_age_steps=FALLBACK_MAX_AGE_ANALYSIS_STEPS,
                    )
                    for _pid in stale_ids:
                        ped_history_feat.pop(_pid, None)

                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    res = yolo_track_pose(
                        yolo_model,
                        bgr,
                        imgsz=DET_IMGSZ,
                        tracker=YOLO_TRACKER,
                        device=YOLO_DEVICE,
                        half=YOLO_HALF,
                    )
                    if YOLO_DEBUG_RAW and res and res[0].boxes is not None:
                        rb = res[0].boxes
                        n = len(rb)
                        cc = rb.conf.cpu().numpy().tolist() if rb.conf is not None else []
                        xy = rb.xyxy.cpu().numpy().tolist() if rb.xyxy is not None else []
                        tid = rb.id.int().cpu().tolist() if rb.id is not None else None
                        print(f"[YOLO raw] boxes={n} id={tid} conf={cc} xyxy={xy}")
                    dets = extract_tracked_pose_detections(res[0], yolo_track_state) if res else []
                    if YOLO_DEBUG_RAW or YOLO_GATE_DEBUG:
                        n_raw = len(res[0].boxes) if res and res[0].boxes is not None else 0
                        print(f"[YOLO gate] raw_boxes={n_raw} passed_gates={len(dets)}")
                    vis = bgr.copy()

                    for _zn, poly_px in zones_scaled.items():
                        for i in range(len(poly_px)):
                            p1 = tuple(map(int, poly_px[i]))
                            p2 = tuple(map(int, poly_px[(i + 1) % len(poly_px)]))
                            cv2.line(vis, p1, p2, (0, 255, 255), 2)

                    max_intent_p = None
                    det_rf_results = []

                    if not dets:
                        pedestrian_detected = False
                        intent_predicted = False
                        intent_proba = None
                        max_crossing_intent_proba = None
                        ped_phase = PHASE_NOT_CROSSING
                    else:
                        pedestrian_detected = False
                        active_pids = set()
                        for det in dets:
                            pid = det["person_id"]
                            box = det["box"]
                            kpts_data = det["kpts"]

                            # Skip ARI robot footprint; YOLO can classify it as a pedestrian.
                            feet_u = (box[0] + box[2]) / 2
                            feet_v = float(box[3])
                            world_pos = pixel_foot_to_world_xy(
                                feet_u, feet_v, rgb_camera, crossing_zone_z, w, h, CAM_FOV
                            )
                            if vision_detection_is_ari_robot(world_pos, ari_vehicle):
                                continue

                            pedestrian_detected = True
                            active_pids.add(pid)
                            ped_dict = detection_to_ped_dict(
                                box, kpts_data, pid, ped_history_feat, zones_scaled
                            )
                            if ped_dict.get("is_on_crossing"):
                                ped_on_crossing_vision = True
                            _ego_kmh_rf = None
                            if _rsu_cam_speed_ms is not None:
                                _ego_kmh_rf = float(_rsu_cam_speed_ms) * 3.6
                            feat_vec = None
                            if (
                                rf_model is not None
                                and rf_dim is not None
                                and not ped_dict.get("is_on_crossing")
                                and _ego_kmh_rf is not None
                            ):
                                if (
                                    rf_model_type == "ma_rong_pie_9_heading"
                                    and build_pie_carla_heading_features is not None
                                ):
                                    from pie.carla_adapter import joints_from_kpts

                                    _joints_now = joints_from_kpts(kpts_data)
                                    feat_vec = build_pie_carla_heading_features(
                                        box,
                                        kpts_data,
                                        v_ego_kmh=_ego_kmh_rf,
                                        image_w=float(w),
                                        image_h=float(h),
                                        fov_deg=float(CAM_FOV),
                                        cam_pitch_deg=float(CAM_PITCH),
                                        cam_height_m=float(CAM_REL_Z),
                                        prev_joints=ped_hist_prev_joints.get(pid),
                                    )
                                    if _joints_now is not None:
                                        ped_hist_prev_joints[pid] = _joints_now
                                elif rf_model_type == "ma_rong_pie_7" and build_pie_carla_features is not None:
                                    feat_vec = build_pie_carla_features(
                                        box,
                                        kpts_data,
                                        v_ego_kmh=_ego_kmh_rf,
                                        image_w=float(w),
                                        image_h=float(h),
                                        fov_deg=float(CAM_FOV),
                                        cam_pitch_deg=float(CAM_PITCH),
                                        cam_height_m=float(CAM_REL_Z),
                                    )
                                else:
                                    feat_vec = None

                            ip = None
                            on_zone = bool(ped_dict.get("is_on_crossing"))
                            if (
                                not on_zone
                                and rf_model is not None
                                and rf_dim is not None
                                and feat_vec is not None
                                and len(feat_vec) == rf_dim
                            ):
                                x_in = np.asarray([feat_vec], dtype=np.float32)
                                ip = float(rf_model.predict_proba(x_in)[0, 1])
                            if ip is not None:
                                max_intent_p = ip if max_intent_p is None else max(max_intent_p, ip)

                            det_rf_results.append(
                                {"pid": pid, "ped_dict": ped_dict, "ip": ip, "on_zone": on_zone}
                            )

                            # CPM: world projection (foot pixel → world XY).
                            if world_pos is not None and FAULT_PROJECTION_NOISE_M > 0:
                                world_pos = (
                                    world_pos[0] + random.gauss(0, FAULT_PROJECTION_NOISE_M),
                                    world_pos[1] + random.gauss(0, FAULT_PROJECTION_NOISE_M),
                                )

                            # CPM: trajectory heading from world position history.
                            prev_wp = yolo_track_state["track_world_positions"].get(pid)
                            if world_pos is not None and prev_wp is not None:
                                dx_w = world_pos[0] - prev_wp[0]
                                dy_w = world_pos[1] - prev_wp[1]
                                if math.hypot(dx_w, dy_w) > 0.05:
                                    yolo_track_state["track_heading_deg"][pid] = math.degrees(
                                        math.atan2(dy_w, dx_w)
                                    )
                            if world_pos is not None:
                                yolo_track_state["track_world_positions"][pid] = world_pos
                            heading_deg = yolo_track_state["track_heading_deg"].get(pid)

                            # CPM: vision-to-CARLA actor ID match (nearest walker ≤ 3 m).
                            carla_obj_id = None
                            matched_actor = None
                            if world_pos is not None:
                                _ari_id = ari_vehicle.id if ari_vehicle is not None else None
                                for w_actor in spawned_walkers:
                                    if not w_actor.is_alive:
                                        continue
                                    if _ari_id is not None and w_actor.id == _ari_id:
                                        continue
                                    aloc = w_actor.get_transform().location
                                    if math.hypot(world_pos[0] - aloc.x, world_pos[1] - aloc.y) < 3.0:
                                        carla_obj_id = w_actor.id
                                        matched_actor = w_actor
                                        break

                            # CPM: tracking confidence (saturates at 1.0 after 20 frames).
                            seen = yolo_track_state["track_seen_count"].get(pid, 1)
                            tracking_conf = min(1.0, seen / 20.0)

                            # CPM: fused confidence.
                            intent_c = ip if ip is not None else 0.0
                            fused_conf = round(
                                CPM_CONF_W_DET * float(det["conf"])
                                + CPM_CONF_W_TRACK * tracking_conf
                                + CPM_CONF_W_INTENT * intent_c,
                                4,
                            )

                            # CPM: velocity components from matched CARLA actor.
                            av = None
                            if matched_actor is not None and matched_actor.is_alive:
                                av = matched_actor.get_velocity()

                            tracked_pedestrians_report.append(
                                {
                                    "objectID": carla_obj_id if carla_obj_id is not None else pid,
                                    "objectClass": ETSI_CLASS_MAP.get(0, "pedestrian"),
                                    "xDistance": round(world_pos[0], 3) if world_pos is not None else None,
                                    "yDistance": round(world_pos[1], 3) if world_pos is not None else None,
                                    "zDistance": crossing_zone_z if world_pos is not None else None,
                                    "xSpeed": round(float(av.x), 3) if av is not None else None,
                                    "ySpeed": round(float(av.y), 3) if av is not None else None,
                                    "heading": round(heading_deg, 2) if heading_deg is not None else None,
                                    "positionConfidence": fused_conf if world_pos is not None else None,
                                    "speedConfidence": round(tracking_conf, 3) if av is not None else None,
                                    "headingConfidence": round(tracking_conf, 3) if heading_deg is not None else None,
                                }
                            )

                            x1, y1, x2, y2 = map(int, box)
                            col = bbox_color_bgr_for_id(pid)
                            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                            draw_pose_skeleton_bgr(vis, kpts_data, col)

                            label = f"id={pid} det={float(det['conf']):.2f}"
                            if on_zone:
                                label += " zone=CROSSING"
                            elif ip is not None:
                                label += f" rf={ip:.2f}"
                            else:
                                label += " rf=n/a"
                            cv2.putText(
                                vis,
                                label,
                                (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )

                        # Phase update after CPM vision positions are available
                        phase_map = {
                            str(item["pid"]): classify_phase(is_on_crossing=item["on_zone"])
                            for item in det_rf_results
                        }
                        ped_phase = aggregate_ped_phase(phase_map)
                        max_crossing_intent_proba = (
                            round(float(max_intent_p), 4) if max_intent_p is not None else None
                        )
                        if ped_phase == PHASE_CROSSING:
                            intent_predicted = False
                            intent_proba = None
                        else:
                            intent_predicted = predict_intent(max_intent_p, rf_threshold)
                            intent_proba = max_crossing_intent_proba

                    # Overlay for saved RSU frames (bbox labels only; no model tag / phase / save path).
                    cv2.putText(
                        vis,
                        f"intent_p={max_crossing_intent_proba} t={sim_time_s:.2f}s",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        vis,
                        f"ego={v_speed * 3.6:.1f} km/h",
                        (12, 54),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )

                    if tracked_pedestrians_report:
                        print(
                            f"[CPM] sim_t={sim_time_s:.3f}s objects={len(tracked_pedestrians_report)}",
                            flush=True,
                        )
                        print(json.dumps(tracked_pedestrians_report, indent=2), flush=True)

                    fn = os.path.join(
                        str(rsu_capture_dir),
                        f"frame_{capture_idx:06d}_t{sim_time_s:.3f}s.jpg",
                    )
                    cv2.imwrite(fn, vis)
                    if SIM_VALIDATION_MODE:
                        rsu_frames_manifest.append(
                            {
                                "path": fn,
                                "sim_time_s": round(float(sim_time_s), 4),
                                "index": capture_idx,
                            }
                        )
                    capture_idx += 1

                if run_logger is not None:
                    run_logger.record_analysis(
                        sim_time_s=sim_time_s,
                        perception={
                            "detected": pedestrian_detected,
                            "intent_pred": intent_predicted,
                            "intent_proba": intent_proba,
                            "rf_proba_raw": (
                                round(float(max_crossing_intent_proba), 4)
                                if max_crossing_intent_proba is not None
                                else None
                            ),
                            "rf_threshold": rf_threshold,
                            "ped_phase": ped_phase,
                            "in_zone_vis": ped_on_crossing_vision,
                            "analysis_skipped": _skip_analysis_yolo,
                        },
                    )

            # ETSI hazard decision policy.
            _ped_speed_ms = max_pedestrian_speed_planar_ms(spawned_walkers)
            _pos_conf = (
                tracked_pedestrians_report[0]["positionConfidence"]
                if tracked_pedestrians_report
                and tracked_pedestrians_report[0]["positionConfidence"] is not None
                else 0.5
            )
            _vision_ped_xy = [
                (
                    r.get("xDistance"),
                    r.get("yDistance"),
                    r.get("objectID"),
                )
                for r in tracked_pedestrians_report
            ]
            _exclude_ids = {ari_vehicle.id} if ari_vehicle is not None else set()
            path_conflict = False
            if vehicle is not None and vehicle.is_alive:
                path_conflict = assess_ego_path_conflict(
                    vehicle,
                    crossing_zone_polygons,
                    walker_actors=spawned_walkers,
                    vision_ped_xy=_vision_ped_xy,
                    exclude_actor_ids=_exclude_ids,
                )
            if FAULT_INTENT_SPIKE and not _fault_intent_spike_used:
                intent_predicted = True
                intent_proba = 0.95
                _fault_intent_spike_used = True

            if ped_in_crossing_zone:
                ped_phase = PHASE_CROSSING
                intent_predicted = False
                intent_proba = None

            ped_state = {
                "ped_phase": ped_phase,
                "intent_proba": intent_proba if ped_phase == PHASE_NOT_CROSSING else None,
                "intent_confirmed": (
                    (ped_phase == PHASE_NOT_CROSSING and intent_predicted)
                    or (ped_phase == PHASE_CROSSING and path_conflict)
                ),
                "path_conflict": path_conflict,
                "is_on_crossing": ped_on_crossing_vision or ped_phase == PHASE_CROSSING,
                "dist_to_crossing_m": d_ped_to_crossing_m,
                "speed_ms": _ped_speed_ms,
                "position_confidence": _pos_conf,
            }
            ego_state = {
                "speed_ms": (
                    _rsu_cam_speed_ms if _rsu_cam_speed_ms is not None else v_speed
                ),
                "dist_to_crossing_m": (
                    _rsu_cam_dist_m
                    if _rsu_cam_dist_m is not None and math.isfinite(_rsu_cam_dist_m)
                    else d_ego_to_crossing_m
                ),
            }
            map_context = {
                "master_range_m": RSU_MASTER_RANGE_M,
                "unsafe_ttc_s": TTC_AEB_S,
                "min_threat_speed_ms": RSU_MIN_THREAT_SPEED_MS,
                "ttc_aeb_s": TTC_AEB_S,
                "ttc_high_s": TTC_HIGH_S,
                "ttc_t0_s": TTC_T0_S,
            }
            hazard_decision = compute_hazard_decision(ped_state, ego_state, map_context)
            priority_level = hazard_decision["priority_level"]

            # Ego OBU — braking driven solely by received V2X (DENM gate + linked CPM TTC).
            ego_denm_active = current_time <= _ego_denm_human_presence_until
            ego_ttc_cpm = float("inf")
            ego_brake_level = 0.0
            _matched_cpm_ped = None
            if ego_denm_active and vehicle is not None and vehicle.is_alive and _ego_cpm_objects:
                _matched_cpm_ped = match_cpm_pedestrian_for_denm(
                    _ego_denm_object_id,
                    _ego_denm_event_pos,
                    _ego_cpm_objects,
                )
                if _matched_cpm_ped is not None:
                    _ego_tf = vehicle.get_transform()
                    _ego_fwd = _ego_tf.get_forward_vector()
                    ego_ttc_cpm = ego_ttc_from_cpm_objects(
                        _ego_tf.location.x,
                        _ego_tf.location.y,
                        _ego_fwd.x,
                        _ego_fwd.y,
                        v_speed,
                        [_matched_cpm_ped],
                        half_width_m=EGO_PATH_HALF_WIDTH_M + 1.5,
                        ahead_max_m=EGO_PATH_AHEAD_MAX_M,
                        behind_m=EGO_PASS_BEHIND_M,
                        crossing_zone_polygons=crossing_zone_polygons,
                    )
                    ego_brake_level = ego_brake_level_from_ttc_s(
                        ego_ttc_cpm,
                        ttc_aeb_s=TTC_AEB_S,
                        ttc_high_s=TTC_HIGH_S,
                        ttc_t0_s=TTC_T0_S,
                    )

            _denm_sent_this_tick = False
            _cpm_sent_this_tick = False
            if run_logger is not None:
                run_logger.record_hazard_decision({
                    "level": priority_level,
                    "signal_priority": hazard_decision.get("signal_priority"),
                    "brake_priority": hazard_decision.get("brake_priority"),
                    "ttc_zone_s": hazard_decision.get("ttc_s"),
                    "path_conflict": path_conflict,
                    "vehicle_threat": hazard_decision.get("vehicle_threat"),
                    "risk_score": hazard_decision.get("risk_score"),
                    "hazard_active": hazard_decision.get("hazard_active"),
                    "hazard_in_zone": hazard_decision.get("hazard_in_zone"),
                    "send_denm": hazard_decision.get("hazard_active"),
                })

            if RSU_HAZARD_DECISION_DEBUG and (current_time - _hazard_decision_debug_last_print) >= 0.5:
                _hazard_decision_debug_last_print = current_time
                _ttc_dbg = hazard_decision["ttc_s"]
                _ttc_str = f"{_ttc_dbg:.2f}" if _ttc_dbg is not None else "inf"
                _cpm_ttc_str = f"{ego_ttc_cpm:.2f}" if math.isfinite(ego_ttc_cpm) else "inf"
                print(
                    f"[RSU hazard decision] sim_t={sim_time_s:.2f}s "
                    f"phase={ped_phase} "
                    f"intent_pred={int(intent_predicted)} "
                    f"path_conflict={int(path_conflict)} "
                    f"intent_p={max_crossing_intent_proba} "
                    f"ped_zone_vis={int(ped_on_crossing_vision)} "
                    f"d_ego={d_ego_to_crossing_m:.1f}m d_ped={d_ped_to_crossing_m:.1f}m "
                    f"v_ego={v_speed:.2f}ms ttc={_ttc_str}s "
                    f"risk={hazard_decision['risk_score']:.3f} "
                    f"signal={hazard_decision.get('signal_priority', '?')} "
                    f"brake_pri={hazard_decision.get('brake_priority', '?')} "
                    f"prio={priority_level} ego_denm_rx={int(ego_denm_active)} "
                    f"cpm_match={int(_matched_cpm_ped is not None)} "
                    f"cpm_ttc={_cpm_ttc_str}s brake={ego_brake_level:.2f} "
                    f"threat={int(hazard_decision['vehicle_threat'])}",
                    flush=True,
                )

            _robot_cam_dist = None
            if _robot_cam_x is not None and _robot_cam_y is not None:
                _robot_cam_dist = min_distance_to_any_crossing_zone(
                    _robot_cam_x, _robot_cam_y, crossing_zone_polygons,
                )

            _robot_denm_active = current_time <= _robot_denm_human_presence_until
            _ped_on_crossing_v2x = bool(
                _robot_denm_active
                and _robot_denm_cause_code == ETSI_CAUSE_COLLISION_RISK
                and _robot_denm_sub_cause_code == ETSI_SUB_COLLISION_PED
            )

            # Robot indicator: received DENM + ego CAM only (no RSU vision / GT fallbacks).
            light_level, ari_light_color, denm_phase = compute_robot_light_state(
                denm_human_presence_until=_robot_denm_human_presence_until,
                current_time=current_time,
                cam_speed_ms=_robot_cam_speed_ms,
                cam_dist_to_crossing_m=_robot_cam_dist,
                unsafe_ttc_s=TTC_HIGH_S,
                min_threat_speed_ms=RSU_MIN_THREAT_SPEED_MS,
                ped_on_crossing=_ped_on_crossing_v2x,
                prev_light_level=_prev_light_level,
            )
            _prev_light_level = light_level
            current_light_state = f"{light_level}_{denm_phase}"
            _robot_hold = light_level == "RED"

            _robot_cam_ttc = None
            if (
                _robot_cam_speed_ms is not None
                and _robot_cam_dist is not None
                and math.isfinite(_robot_cam_dist)
                and _robot_cam_speed_ms >= RSU_MIN_THREAT_SPEED_MS
            ):
                _robot_cam_ttc = _robot_cam_dist / max(_robot_cam_speed_ms, 0.5)
            if run_logger is not None:
                run_logger.record_robot({
                    "hold": bool(_robot_hold),
                    "spatem": denm_phase,
                    "light_level": light_level,
                    "denm_active": bool(_robot_denm_active),
                    "denm_rx_until_s": round(_robot_denm_human_presence_until, 2),
                    "cam_ttc_s": (None if _robot_cam_ttc is None else round(_robot_cam_ttc, 3)),
                    "cam_dist_m": (None if _robot_cam_dist is None else round(_robot_cam_dist, 3)),
                })

            # ARI robot motion: hold when V2X light state is RED.
            if ari_vehicle is not None and ari_vehicle.is_alive:
                if _robot_hold:
                    ari_vehicle.apply_control(
                        carla.VehicleControl(
                            throttle=0.0,
                            steer=0.0,
                            brake=1.0,
                            hand_brake=False,
                            reverse=False,
                        )
                    )
                else:
                    ari_vehicle.apply_control(
                        carla.VehicleControl(
                            throttle=ROBOT_CROSSING_THROTTLE,
                            steer=0.0,
                            brake=0.0,
                            hand_brake=False,
                            reverse=False,
                        )
                    )

            if (
                ari_vehicle is not None
                and ari_vehicle.is_alive
                and ari_light_color is not None
            ):
                draw_robot_light_block(
                    debug,
                    ari_light_loc,
                    ari_light_color,
                    half_extent_m=ROBOT_LIGHT_BLOCK_HALF_M,
                    life_time_s=ROBOT_LIGHT_BLOCK_LIFE_S,
                )
            debug.draw_point(rsu_sensor_ident, size=0.25, color=carla.Color(r=0, g=0, b=255))

            # V2X send.

            if (
                ego_v2x is not None
                and ego_v2x.is_alive
                and vehicle is not None
                and vehicle.is_alive
                and (current_time - _cam_last_gen_time) >= CAM_T_GEN_MIN_S
            ):
                cam_payload = build_cam_payload(v_loc.x, v_loc.y, v_speed)
                cam_str = serialize_cam_v2x_message(cam_payload)
                _v2x_custom_send(ego_v2x, cam_str)
                _cam_last_gen_time = current_time
                _cam_sent_seq += 1
                _cam_sent_wall_s = current_time
                _cam_sent_sim_s = sim_time_s
                _cam_sent_this_tick = True

            if rsu_v2x is not None and rsu_v2x.is_alive:
                rsu_msg = (
                    f"RSU#{v2x_counter} priority={priority_level} light={light_level} "
                    f"intent={int(intent_predicted)} "
                    f"d_ped={d_ped_to_crossing_m:.1f}m "
                    f"d_ego={d_ego_to_crossing_m:.1f}m "
                    f"brake={ego_brake_level:.1f} robot_hold={int(_robot_hold)}"
                )
                if V2X_SEND_PLAIN_RSU_DEBUG:
                    _v2x_custom_send(rsu_v2x, rsu_msg)

                # DENM: hazard-event dissemination (independent of ego range / TTC).
                # ETSI EN 302 637-3 cause codes (from hazard kind, not brake priority):
                #   zone occupancy → 97 / 4  collisionRisk / VRU collision risk
                #   intent only    → 12 / 0  humanPresenceOnTheRoad / unavailable
                _hazard_active = bool(hazard_decision.get("hazard_active"))
                _hazard_in_zone = bool(hazard_decision.get("hazard_in_zone"))
                _hazard_activated = _hazard_active and not _denm_hazard_was_active
                _denm_due = (current_time - _denm_last_sent_time) >= _DENM_SEND_INTERVAL_S
                _hazard_kind = hazard_kind(in_zone_vis=_hazard_in_zone)
                _hazard_kind_changed = (
                    _hazard_active
                    and _denm_last_hazard_kind
                    and _hazard_kind != _denm_last_hazard_kind
                )
                _tx_denm, _denm_kind = should_transmit_denm(
                    hazard_active=_hazard_active,
                    hazard_activated=_hazard_activated,
                    denm_due=_denm_due,
                    hazard_kind_changed=_hazard_kind_changed,
                )
                if _tx_denm:
                    # Event position + objectID: first tracked pedestrian, else crossing centroid.
                    _denm_object_id = None
                    if tracked_pedestrians_report:
                        _first = tracked_pedestrians_report[0]
                        _ex = _first["xDistance"] if _first["xDistance"] is not None else -109.8
                        _ey = _first["yDistance"] if _first["yDistance"] is not None else -156.7
                        if _first.get("objectID") is not None:
                            _denm_object_id = _object_id_as_int(_first["objectID"])
                    else:
                        _all_verts = [v for poly in crossing_zone_polygons for v in poly]
                        _ex = sum(v[0] for v in _all_verts) / len(_all_verts)
                        _ey = sum(v[1] for v in _all_verts) / len(_all_verts)

                    # informationQuality from hazard kind (not TTC-derived priority).
                    _denm_iq = 7 if _hazard_in_zone else 5

                    _relevance_distance = crosswalk_radius_m(crossing_zone_polygons)

                    if _hazard_in_zone:
                        _cause, _sub = ETSI_CAUSE_COLLISION_RISK, ETSI_SUB_COLLISION_PED
                    else:
                        _cause, _sub = ETSI_CAUSE_HUMAN_PRESENCE, ETSI_SUB_PEDESTRIAN

                    denm_payload = build_denm_payload(
                        _ex, _ey,
                        information_quality=_denm_iq,
                        relevance_distance=round(_relevance_distance, 2),
                        validity_duration=5.0,
                        cause_code=_cause,
                        sub_cause_code=_sub,
                        object_id=_denm_object_id,
                    )
                    denm_str = serialize_denm_v2x_message(denm_payload)
                    _v2x_custom_send(rsu_v2x, denm_str)
                    _denm_last_sent_time = current_time
                    _denm_last_hazard_kind = _hazard_kind
                    print(
                        f"[DENM] {_denm_kind} | hazard_zone={int(_hazard_in_zone)} "
                        f"cause={_cause}/{_sub} iq={_denm_iq} "
                        f"relDist={round(_relevance_distance, 2)} "
                        f"pos=({_ex:.1f},{_ey:.1f}) bytes={len(denm_str)}",
                        flush=True,
                    )
                    print(denm_str, flush=True)
                    _denm_sent_this_tick = True
                    _denm_sent_seq += 1
                    _denm_sent_wall_s = current_time
                    _denm_sent_sim_s = sim_time_s
                    _denm_last_log_meta = {
                        "sent": True,
                        "kind": _denm_kind,
                        "sim_time_s": sim_time_s,
                        "cause_code": _cause,
                        "sub_cause_code": _sub,
                        "information_quality": _denm_iq,
                        "relevance_m": round(_relevance_distance, 2),
                        "hazard_in_zone": _hazard_in_zone,
                    }

                _denm_hazard_was_active = _hazard_active
                if not _hazard_active:
                    _denm_last_hazard_kind = ""

                # CPM: broadcast perceived pedestrians when detected (ETSI TS 103 324).
                if (
                    pedestrian_detected
                    and tracked_pedestrians_report
                    and (current_time - _cpm_last_gen_time) >= CPM_T_GEN_MIN_S
                ):
                    _cpm_objects = cpm_select_objects(
                        tracked_pedestrians_report,
                        _cpm_obj_last_included,
                        current_time,
                    )
                    if _cpm_objects:
                        cpm_payload = build_cpm_payload(
                            _cpm_objects,
                            station_id=RSU_STATION_ID,
                            ref_x=rsu_x,
                            ref_y=rsu_y,
                        )
                        cpm_str = serialize_cpm_v2x_message(cpm_payload)
                        _v2x_custom_send(rsu_v2x, cpm_str)
                        _cpm_last_gen_time = current_time
                        _cpm_sent_this_tick = True
                        _cpm_sent_seq += 1
                        _cpm_sent_wall_s = current_time
                        _cpm_sent_sim_s = sim_time_s
                        _cpm_last_log_meta = {
                            "sent": True,
                            "sim_time_s": sim_time_s,
                            "num_objects": len(_cpm_objects),
                            "object_ids": [o.get("objectID") for o in _cpm_objects],
                        }
                        print(
                            f"[CPM] sent | bytes={len(cpm_str)} objects={len(_cpm_objects)} "
                            f"payload={cpm_str}",
                            flush=True,
                        )

            if run_logger is not None:
                def _cam_link_log(
                    *,
                    sent_this_tick: bool,
                    rx_this_tick: bool,
                    rx_wall_s: float | None,
                    rx_sim_s: float | None,
                    rx_latency_s: float | None,
                    rx_seq: int | None,
                    speed_ms: float | None,
                    ref_x: float | None,
                    ref_y: float | None,
                    dist_to_crossing_m: float | None,
                ) -> dict:
                    link = _v2x_link_log(
                        sent_this_tick=sent_this_tick,
                        rx_this_tick=rx_this_tick,
                        sent_wall_s=_cam_sent_wall_s,
                        sent_sim_s=_cam_sent_sim_s,
                        rx_wall_s=rx_wall_s,
                        rx_sim_s=rx_sim_s,
                        rx_latency_s=rx_latency_s,
                        msg_seq=_cam_sent_seq if sent_this_tick else rx_seq,
                        last_sent_wall_s=_cam_sent_wall_s,
                        last_received_wall_s=rx_wall_s,
                    )
                    link.update({
                        "speed_ms": (None if speed_ms is None else round(float(speed_ms), 3)),
                        "ref_x": (None if ref_x is None else round(float(ref_x), 3)),
                        "ref_y": (None if ref_y is None else round(float(ref_y), 3)),
                        "dist_to_crossing_m": (
                            None
                            if dist_to_crossing_m is None or not math.isfinite(dist_to_crossing_m)
                            else round(float(dist_to_crossing_m), 3)
                        ),
                    })
                    return link

                run_logger.record_denm({
                    **_denm_last_log_meta,
                    "rsu_ego": _v2x_link_log(
                        sent_this_tick=_denm_sent_this_tick,
                        rx_this_tick=_ego_denm_rx_this_tick,
                        sent_wall_s=_denm_sent_wall_s,
                        sent_sim_s=_denm_sent_sim_s,
                        rx_wall_s=_ego_denm_rx_wall_s,
                        rx_sim_s=_ego_denm_rx_sim_s,
                        rx_latency_s=_ego_denm_rx_latency_s,
                        msg_seq=_denm_sent_seq if _denm_sent_this_tick else _ego_denm_rx_seq,
                        last_sent_wall_s=_denm_sent_wall_s,
                        last_received_wall_s=_ego_denm_rx_wall_s,
                    ),
                    "rsu_robot": _v2x_link_log(
                        sent_this_tick=_denm_sent_this_tick,
                        rx_this_tick=_robot_denm_rx_this_tick,
                        sent_wall_s=_denm_sent_wall_s,
                        sent_sim_s=_denm_sent_sim_s,
                        rx_wall_s=_robot_denm_rx_wall_s,
                        rx_sim_s=_robot_denm_rx_sim_s,
                        rx_latency_s=_robot_denm_rx_latency_s,
                        msg_seq=_denm_sent_seq if _denm_sent_this_tick else _robot_denm_rx_seq,
                        last_sent_wall_s=_denm_sent_wall_s,
                        last_received_wall_s=_robot_denm_rx_wall_s,
                    ),
                })
                run_logger.record_cpm({
                    **_cpm_last_log_meta,
                    "rsu_ego": _v2x_link_log(
                        sent_this_tick=_cpm_sent_this_tick,
                        rx_this_tick=_ego_cpm_rx_this_tick,
                        sent_wall_s=_cpm_sent_wall_s,
                        sent_sim_s=_cpm_sent_sim_s,
                        rx_wall_s=_ego_cpm_rx_wall_s,
                        rx_sim_s=_ego_cpm_rx_sim_s,
                        rx_latency_s=_ego_cpm_rx_latency_s,
                        msg_seq=_cpm_sent_seq if _cpm_sent_this_tick else _ego_cpm_rx_seq,
                        last_sent_wall_s=_cpm_sent_wall_s,
                        last_received_wall_s=_ego_cpm_rx_wall_s,
                    ),
                })
                run_logger.record_cam({
                    "ego_rsu": _cam_link_log(
                        sent_this_tick=_cam_sent_this_tick,
                        rx_this_tick=_rsu_cam_rx_this_tick,
                        rx_wall_s=_rsu_cam_rx_wall_s,
                        rx_sim_s=_rsu_cam_rx_sim_s,
                        rx_latency_s=_rsu_cam_rx_latency_s,
                        rx_seq=_rsu_cam_rx_seq,
                        speed_ms=_rsu_cam_speed_ms,
                        ref_x=_rsu_cam_x,
                        ref_y=_rsu_cam_y,
                        dist_to_crossing_m=_rsu_cam_dist_m,
                    ),
                    "ego_robot": _cam_link_log(
                        sent_this_tick=_cam_sent_this_tick,
                        rx_this_tick=_robot_cam_rx_this_tick,
                        rx_wall_s=_robot_cam_rx_wall_s,
                        rx_sim_s=_robot_cam_rx_sim_s,
                        rx_latency_s=_robot_cam_rx_latency_s,
                        rx_seq=_robot_cam_rx_seq,
                        speed_ms=_robot_cam_speed_ms,
                        ref_x=_robot_cam_x,
                        ref_y=_robot_cam_y,
                        dist_to_crossing_m=_robot_cam_dist,
                    ),
                })
                run_logger.record_v2x({
                    "denm_sent": _denm_sent_this_tick,
                    "cpm_sent": _cpm_sent_this_tick,
                    "cam_sent": _cam_sent_this_tick,
                    "cam_rsu_rx": _rsu_cam_rx_this_tick,
                    "cam_robot_rx": _robot_cam_rx_this_tick,
                    "denm_ego_rx": _ego_denm_rx_this_tick,
                    "denm_robot_rx": _robot_denm_rx_this_tick,
                    "cpm_ego_rx": _ego_cpm_rx_this_tick,
                    "ego_denm_active_rx": ego_denm_active,
                    "ego_cpm_ttc_s": (None if not math.isfinite(ego_ttc_cpm) else round(ego_ttc_cpm, 3)),
                    "ego_cpm_objects_rx": len(_ego_cpm_objects),
                    "ego_cpm_matched_id": (
                        None if _matched_cpm_ped is None else _matched_cpm_ped.get("objectID")
                    ),
                    "plain_rsu_sent": rsu_v2x is not None and rsu_v2x.is_alive,
                    "robot_denm_active_rx": _robot_denm_active,
                    "cam_speed_ms": _rsu_cam_speed_ms,
                    "cam_dist_m": (
                        None
                        if _rsu_cam_dist_m is None or not math.isfinite(_rsu_cam_dist_m)
                        else round(_rsu_cam_dist_m, 3)
                    ),
                })

            v2x_counter += 1

            crossing_zone_line_color = carla.Color(r=255, g=255, b=0)
            for cz_poly in crossing_zone_polygons:
                for i in range(len(cz_poly)):
                    p1 = cz_poly[i]
                    p2 = cz_poly[(i + 1) % len(cz_poly)]
                    loc1 = carla.Location(x=p1[0], y=p1[1], z=crossing_zone_z)
                    loc2 = carla.Location(x=p2[0], y=p2[1], z=crossing_zone_z)
                    debug.draw_line(
                        loc1,
                        loc2,
                        thickness=0.1,
                        color=crossing_zone_line_color,
                        life_time=0.1,
                    )

            if run_logger is not None:
                ego_log = None
                if vehicle and vehicle.is_alive:
                    ego_log = {
                        "x": round(v_loc.x, 3),
                        "y": round(v_loc.y, 3),
                        "speed_ms": round(v_speed, 3),
                        "brake": round(ego_brake_level, 3),
                        "d_cross_m": round(d_ego_to_crossing_m, 3),
                    }
                ped_gt = None
                if _target_walker is not None and _target_walker.is_alive:
                    wloc = _target_walker.get_transform().location
                    wvel = _target_walker.get_velocity()
                    in_zone = point_in_any_crossing_zone(
                        wloc.x, wloc.y, crossing_zone_polygons
                    )
                    ped_gt = {
                        "walker_id": int(_target_walker.id),
                        "x": round(wloc.x, 3),
                        "y": round(wloc.y, 3),
                        "speed_ms": round(math.hypot(wvel.x, wvel.y), 3),
                        "in_zone": bool(in_zone),
                    }
                    if vehicle and vehicle.is_alive:
                        run_logger.update_min_distance(
                            math.hypot(wloc.x - v_loc.x, wloc.y - v_loc.y)
                        )
                pp = dict(run_logger._last_perception)
                pp["intent_confirmed"] = bool(ped_state.get("intent_confirmed"))
                pp["ped_phase"] = ped_phase
                run_logger._last_perception = pp
                run_logger.log_tick(
                    sim_time_s=sim_time_s,
                    ego=ego_log,
                    ped_gt=ped_gt,
                    ego_brake_level=ego_brake_level,
                    collision_this_tick=collision_this_tick,
                )
                collision_this_tick = False

            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            if vehicle and vehicle.is_alive:
                # AEB VRU graded braking (logged ego_brake_level):
                #   CRITICAL (1.0) — TTC ≤ 1.5 s   HIGH (0.6) — 1.5–2.5 s
                #   MEDIUM (0.2) — 2.5–4.0 s     LOW (0.0) — navigation
                # Actuator mapping escalates to full brake + hand-brake hold in CARLA.
                _vru_ctrl = ego_vru_brake_control(vehicle, ego_brake_level, speed_ms=v_speed)
                if _vru_ctrl is not None:
                    vehicle.apply_control(_vru_ctrl)
                elif ego_python_mode == "lane":
                    vehicle.apply_control(
                        greedy_lane_follow_control(vehicle, EGO_DESTINATION, NAV_TARGET_SPEED_KMH)
                    )
                elif ego_python_mode == "simple":
                    vehicle.apply_control(
                        simple_pursuit_control(vehicle, EGO_DESTINATION, NAV_TARGET_SPEED_KMH)
                    )
                elif nav_agent is not None:
                    vehicle.apply_control(nav_agent.run_step())
                else:
                    vehicle.apply_control(carla.VehicleControl())

                v_trans = vehicle.get_transform()
                cam_loc = v_trans.location + carla.Location(z=8) + v_trans.get_forward_vector() * -12
                spectator.set_transform(
                    carla.Transform(cam_loc, carla.Rotation(pitch=-25, yaw=v_trans.rotation.yaw))
                )

    except KeyboardInterrupt:
        pass
    finally:
        print("\nCleaning up actors...")
        if run_logger is not None:
            try:
                if SIM_VALIDATION_MODE and rsu_frames_manifest:
                    manifest_path = run_logger.run_dir / "frames_manifest.json"
                    with open(manifest_path, "w", encoding="utf-8") as mf:
                        json.dump(
                            {
                                "scenario_id": base_scenario_id,
                                "random_seed": SIM_RANDOM_SEED,
                                "frame_count": len(rsu_frames_manifest),
                                "frames": rsu_frames_manifest,
                            },
                            mf,
                            indent=2,
                        )
                run_logger.close()
                print(f"[simulation] Run log closed: {run_logger.path}", flush=True)
            except Exception:
                pass
        if rgb_camera is not None:
            try:
                rgb_camera.stop()
            except Exception:
                pass
        for _v2x in (
            rsu_v2x, robot_v2x, ego_v2x,
            collision_sensor if collision_sensor is not None else None,
        ):
            if _v2x is not None:
                try:
                    _v2x.stop()
                except Exception:
                    pass
        if world is not None:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)

        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        pygame.quit()


if __name__ == "__main__":
    main()
