"""RSU perception (thesis §4.2.1): fixed RGB camera analysis, YOLOv8-pose + ByteTrack,
bounding boxes / skeleton keypoints, world projection, crossing-zone polygon matching."""

from __future__ import annotations

import math

import cv2
import numpy as np

from sim_config import (
    COCO17_LIMBS,
    COCO_MAPPING,
    CONF_NEW_TRACK_MIN,
    DET_CLASSES,
    DET_CONF,
    FALLBACK_ID_MATCH_RATIO,
    KP_CONF_THRESH,
    MIN_ASPECT_RATIO,
    MIN_BBOX_H_PX,
    MIN_HIP_CONF,
    MIN_KPT_MAX_CONF,
    MIN_TRACK_AGE_FRAMES,
    MIN_VISIBLE_KPTS,
    POSE_DRAW_CONF,
    YOLO_DEBUG_RAW,
    YOLO_GATE_DEBUG,
    YOLO_NMS_IOU,
)

def is_point_in_polygon(x, y, poly):
    """Ray-casting: point inside polygon."""
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def point_in_any_crossing_zone(x, y, crossing_zone_polygons):
    """True if (x, y) lies inside any trail24 crossing zone (world XY)."""
    for poly in crossing_zone_polygons:
        if is_point_in_polygon(x, y, poly):
            return True
    return False


def dist_point_to_segment(px, py, x1, y1, x2, y2):
    sl = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if sl == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / sl))
    qx = x1 + t * (x2 - x1)
    qy = y1 + t * (y2 - y1)
    return math.hypot(px - qx, py - qy)


def point_to_polygon_distance(px, py, poly):
    if is_point_in_polygon(px, py, poly):
        return 0.0
    dmin = float("inf")
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        dmin = min(dmin, dist_point_to_segment(px, py, x1, y1, x2, y2))
    return dmin


def min_distance_to_any_crossing_zone(px, py, crossing_zone_polygons):
    """Shortest distance from a world XY point to the union of crossing polygons (m)."""
    if not crossing_zone_polygons:
        return float("inf")
    return min(point_to_polygon_distance(px, py, poly) for poly in crossing_zone_polygons)
def crosswalk_radius_m(crossing_zone_polygons):
    """Characteristic radius (m) from hardcoded crossing zone polygons (max centroid-to-vertex)."""
    radius = 0.0
    for poly in crossing_zone_polygons:
        if not poly:
            continue
        cx = sum(v[0] for v in poly) / len(poly)
        cy = sum(v[1] for v in poly) / len(poly)
        for vx, vy in poly:
            radius = max(radius, math.hypot(vx - cx, vy - cy))
    return radius
def get_centroid(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def rsu_camera_rotation_toward(
    cam_x: float,
    cam_y: float,
    cam_z: float,
    look_x: float,
    look_y: float,
    look_z: float,
    *,
    yaw_offset_deg: float = 0.0,
    pitch_offset_deg: float = 0.0,
) -> tuple[float, float]:
    """CARLA camera pitch/yaw (deg) to look from cam position toward a world point."""
    dx = look_x - cam_x
    dy = look_y - cam_y
    dz = look_z - cam_z
    horiz = math.hypot(dx, dy)
    yaw = math.degrees(math.atan2(dy, dx)) + yaw_offset_deg
    pitch = (
        -math.degrees(math.atan2(dz, horiz)) + pitch_offset_deg
        if horiz > 1e-6
        else -20.0 + pitch_offset_deg
    )
    return pitch, yaw


def world_polygon_xy_to_image_pixels(poly_xy, ground_z, sensor, img_w, img_h, fov_deg):
    """
    Project world (x, y) vertices at fixed Z to RSU image pixels.
    Axis remap matches CARLA PythonAPI/examples/client_bounding_boxes.py.
    Vertices behind the camera are skipped; returns pixel list if at least 3 corners are visible.
    """
    if sensor is None or not sensor.is_alive:
        return []
    try:
        M_inv = np.array(sensor.get_transform().get_inverse_matrix(), dtype=np.float64).reshape(4, 4)
    except Exception:
        return []
    fx = img_w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fy = fx
    cx = img_w / 2.0
    cy = img_h / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    out = []
    for xw, yw in poly_xy:
        pw = np.array([xw, yw, ground_z, 1.0], dtype=np.float64)
        ps = M_inv @ pw
        xs, ys, zs = float(ps[0]), float(ps[1]), float(ps[2])
        xc, yc, zc = ys, -zs, xs
        if zc <= 0.05:
            continue
        xyz = K @ np.array([xc, yc, zc])
        out.append((float(xyz[0] / xyz[2]), float(xyz[1] / xyz[2])))
    return out if len(out) >= 3 else []


def pixel_foot_to_world_xy(u, v, sensor, ground_z, img_w, img_h, fov_deg):
    """
    Back-project a foot pixel (u, v) to world XY using ground-plane assumption (world z = ground_z).
    Inverse of world_polygon_xy_to_image_pixels; uses the same CARLA axis remap (xc=ys, yc=-zs, zc=xs).
    Returns (world_x, world_y) or None if the ray misses the ground plane or sensor is unavailable.
    """
    if sensor is None or not sensor.is_alive:
        return None
    fx = img_w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx_i = img_w / 2.0
    cy_i = img_h / 2.0
    # Camera-space ray direction (normalized to zc=1)
    xc = (u - cx_i) / fx
    yc = (v - cy_i) / fx
    # Reverse axis remap: xc=ys, yc=-zs, zc=xs → sensor-space [xs, ys, zs] = [1, xc, -yc]
    xs, ys, zs = 1.0, xc, -yc
    try:
        M = np.array(sensor.get_transform().get_matrix(), dtype=np.float64).reshape(4, 4)
    except Exception:
        return None
    origin = M @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    direction = M @ np.array([xs, ys, zs, 0.0], dtype=np.float64)
    dz = float(direction[2])
    if abs(dz) < 1e-6:
        return None
    t = (ground_z - float(origin[2])) / dz
    if t < 0:
        return None
    return float(origin[0] + t * direction[0]), float(origin[1] + t * direction[1])


def crossing_world_polys_to_zones_scaled(polygons_world_xy, ground_z, sensor, img_w, img_h, fov_deg):
    """Build zones_scaled dict for detection_to_ped_dict + cv2 overlay from hardcoded world quads."""
    names = [f"crossing_zone_{i + 1}" for i in range(len(polygons_world_xy))]
    scaled = {}
    for name, poly in zip(names, polygons_world_xy):
        pix = world_polygon_xy_to_image_pixels(poly, ground_z, sensor, img_w, img_h, fov_deg)
        if len(pix) >= 3:
            scaled[name] = pix
    return scaled


def detection_to_ped_dict(box, kpts_data, ped_id, ped_history, zones_scaled):
    """
    Same feature logic as generate_carla_dataset_v8.py: feet vs scaled Shapely-style zones —
    on_crossing if distance to any zone <= bbox_h * 0.35; dist_to_curb_norm is min distance
    to any zone divided by bbox_h.
    """
    bbox_h = max(1.0, float(box[3] - box[1]))
    feet_x = (box[0] + box[2]) / 2
    feet_y = box[3]
    tolerance = bbox_h * 0.35

    on_crossing = 0
    crosswalk_zone_names = []
    dist_to_curb_raw = float("inf")

    for zone_name, poly_px in zones_scaled.items():
        dist = point_to_polygon_distance(feet_x, feet_y, poly_px)
        dist_to_curb_raw = min(dist_to_curb_raw, dist)
        if dist <= tolerance:
            on_crossing = 1
            crosswalk_zone_names.append(zone_name)

    if not math.isfinite(dist_to_curb_raw):
        dist_to_curb_raw = 0.0
    dist_to_curb_norm = float(dist_to_curb_raw / bbox_h)

    movement_norm = 0.0
    if ped_id in ped_history:
        px, py = ped_history[ped_id]
        movement_norm = float(math.hypot(feet_x - px, feet_y - py) / bbox_h)
    ped_history[ped_id] = (feet_x, feet_y)

    root_x = (kpts_data[11][0] + kpts_data[12][0]) / 2
    root_y = (kpts_data[11][1] + kpts_data[12][1]) / 2
    skeleton_norm = {}
    for idx, name in COCO_MAPPING.items():
        x, y, conf = kpts_data[idx]
        if conf > 0.25:
            skeleton_norm[name] = {
                "x": float((x - root_x) / bbox_h),
                "y": float((y - root_y) / bbox_h),
            }

    return {
        "person_id": ped_id,
        "is_on_crossing": on_crossing,
        "crosswalk_zone_names": crosswalk_zone_names,
        "movement_norm": movement_norm,
        "dist_to_curb_norm": dist_to_curb_norm,
        "skeleton_normalized": skeleton_norm,
    }


def new_yolo_track_state():
    return {
        "analysis_seq": 0,
        "track_centers": {},
        "track_last_seen": {},
        "track_seen_count": {},
        "tracker_to_stable": {},
        "fallback_next_id": 0,
        "track_world_positions": {},  # person_id → (wx, wy) last projected world position
        "track_heading_deg": {},      # person_id → heading in degrees (world XY frame)
    }


def prune_yolo_track_state(state, *, max_age_steps):
    """Drop stale person_ids; returns list of pruned ids for clearing RF history."""
    seq = int(state["analysis_seq"])
    tls = state["track_last_seen"]
    stale = [pid for pid, last_idx in tls.items() if (seq - last_idx) > max_age_steps]
    for pid in stale:
        state["track_centers"].pop(pid, None)
        tls.pop(pid, None)
        state["track_seen_count"].pop(pid, None)
        state["track_world_positions"].pop(pid, None)
        state["track_heading_deg"].pop(pid, None)
    tts = state["tracker_to_stable"]
    rm_raw = [rt for rt, sp in list(tts.items()) if sp in stale]
    for rt in rm_raw:
        tts.pop(rt, None)
    return stale


def extract_tracked_pose_detections(result, state):
    """
    All people passing pose/geometry gates, each with a stable string person_id
    (ByteTrack id preferred; else nearest-centroid fallback / fb_N).
    Mutates state (centers, seen counts, tracker map).
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    if result.keypoints is None or result.keypoints.data is None:
        return []

    seq = int(state["analysis_seq"])
    track_centers = state["track_centers"]
    track_last_seen = state["track_last_seen"]
    track_seen_count = state["track_seen_count"]
    tracker_to_stable = state["tracker_to_stable"]

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    ids = None
    if result.boxes.id is not None:
        ids = result.boxes.id.int().cpu().numpy()
    kpts = result.keypoints.data.cpu().numpy()

    out = []
    assigned_ids_in_frame = set()

    def _gate_log(msg):
        if YOLO_GATE_DEBUG:
            print(f"[YOLO gate] {msg}")

    for i in range(len(boxes)):
        box = boxes[i]
        bbox_h = max(1.0, float(box[3] - box[1]))
        bbox_w = max(1.0, float(box[2] - box[0]))
        det_conf_early = float(confs[i])
        if bbox_h < MIN_BBOX_H_PX:
            _gate_log(
                f"box[{i}] reject=bbox_h<{MIN_BBOX_H_PX} h={bbox_h:.1f} w={bbox_w:.1f} conf={det_conf_early:.3f}"
            )
            continue
        if bbox_h / bbox_w < MIN_ASPECT_RATIO:
            _gate_log(
                f"box[{i}] reject=aspect h/w={bbox_h / bbox_w:.2f}<{MIN_ASPECT_RATIO} "
                f"h={bbox_h:.1f} conf={det_conf_early:.3f}"
            )
            continue

        cx, cy = get_centroid(box)
        raw_tid = None
        if ids is not None and i < len(ids):
            tid_val = ids[i]
            try:
                tv = float(tid_val)
                if not np.isnan(tv):
                    raw_tid = str(int(tv))
            except (TypeError, ValueError):
                raw_tid = None

        best_pid = None
        best_dist = float("inf")
        for pid, (pcx, pcy) in track_centers.items():
            dist = float(math.hypot(cx - pcx, cy - pcy))
            if dist < best_dist:
                best_dist = dist
                best_pid = pid
        match_thresh = bbox_h * FALLBACK_ID_MATCH_RATIO

        if raw_tid is not None:
            if raw_tid in tracker_to_stable:
                person_id = tracker_to_stable[raw_tid]
            elif best_pid is not None and best_dist <= match_thresh and best_pid not in assigned_ids_in_frame:
                person_id = best_pid
                tracker_to_stable[raw_tid] = person_id
            else:
                person_id = raw_tid
                tracker_to_stable[raw_tid] = person_id
        else:
            if best_pid is not None and best_dist <= match_thresh and best_pid not in assigned_ids_in_frame:
                person_id = best_pid
            else:
                person_id = f"fb_{state['fallback_next_id']}"
                state["fallback_next_id"] = int(state["fallback_next_id"]) + 1

        kpts_i = kpts[i]
        kc = kpts_i[:, 2]
        n_vis = int(np.sum(kc > KP_CONF_THRESH))
        if n_vis < MIN_VISIBLE_KPTS:
            _gate_log(
                f"box[{i}] id={person_id} reject=visible_kpts {n_vis}<{MIN_VISIBLE_KPTS} conf={det_conf_early:.3f}"
            )
            continue
        kmax = float(np.max(kc))
        if kmax < MIN_KPT_MAX_CONF:
            _gate_log(
                f"box[{i}] id={person_id} reject=kpt_max {kmax:.3f}<{MIN_KPT_MAX_CONF} conf={det_conf_early:.3f}"
            )
            continue
        hip_m = float(max(kc[11], kc[12]))
        if hip_m < MIN_HIP_CONF:
            _gate_log(
                f"box[{i}] id={person_id} reject=hip {hip_m:.3f}<{MIN_HIP_CONF} conf={det_conf_early:.3f}"
            )
            continue

        det_conf = float(confs[i])
        seen_frames = int(track_seen_count.get(person_id, 0))
        if seen_frames < MIN_TRACK_AGE_FRAMES and det_conf < CONF_NEW_TRACK_MIN:
            _gate_log(
                f"box[{i}] id={person_id} reject=new_track seen={seen_frames}<{MIN_TRACK_AGE_FRAMES} "
                f"conf={det_conf:.3f}<{CONF_NEW_TRACK_MIN:.3f}"
            )
            continue

        track_seen_count[person_id] = seen_frames + 1
        track_centers[person_id] = (float(cx), float(cy))
        track_last_seen[person_id] = seq
        assigned_ids_in_frame.add(person_id)

        out.append(
            {
                "person_id": person_id,
                "box": box,
                "kpts": kpts_i.copy(),
                "conf": det_conf,
            }
        )

    return out


def bbox_color_bgr_for_id(person_id: str):
    """Distinct BGR color per id for overlays."""
    h = hash(person_id) & 0xFFFFFFFF
    r = 80 + (h & 0x7F)
    g = 80 + ((h >> 8) & 0x7F)
    b = 80 + ((h >> 16) & 0x7F)
    return int(b), int(g), int(r)


def draw_pose_skeleton_bgr(vis, kpts_17x3, color_bgr, conf_thresh=POSE_DRAW_CONF):
    """Draw COCO17 limbs and joint markers on BGR image."""
    for i, j in COCO17_LIMBS:
        if i >= len(kpts_17x3) or j >= len(kpts_17x3):
            continue
        xi, yi, ci = kpts_17x3[i]
        xj, yj, cj = kpts_17x3[j]
        if ci < conf_thresh or cj < conf_thresh:
            continue
        p1 = (int(round(xi)), int(round(yi)))
        p2 = (int(round(xj)), int(round(yj)))
        cv2.line(vis, p1, p2, color_bgr, 2, cv2.LINE_AA)
    for idx in range(len(kpts_17x3)):
        x, y, c = kpts_17x3[idx]
        if c < conf_thresh:
            continue
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(vis, (int(round(x)), int(round(y))), 5, (255, 255, 255), 1, cv2.LINE_AA)


def yolo_track_pose(model, bgr, *, imgsz, tracker, device=None, half=False):
    """YOLO pose + ByteTrack; persist=True keeps ids across analysis frames."""
    kw = dict(
        imgsz=imgsz,
        conf=DET_CONF,
        iou=YOLO_NMS_IOU,
        classes=DET_CLASSES,
        verbose=False,
        half=half,
        persist=True,
        tracker=tracker,
    )
    if device:
        kw["device"] = device
    try:
        return model.track(bgr, **kw)
    except Exception as ex:
        msg = str(ex).lower()
        oom = "out of memory" in msg
        cuda_oom = False
        try:
            import torch

            cuda_oom = isinstance(ex, torch.cuda.OutOfMemoryError)
        except Exception:
            pass
        if not (oom or cuda_oom):
            raise
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        fallback_sz = max(640, min(960, int(imgsz) * 3 // 4))
        print(f"[simulation] YOLO track CUDA OOM — retry on CPU imgsz={fallback_sz} ({str(ex)[:120]})")
        return model.track(
            bgr,
            imgsz=fallback_sz,
            device="cpu",
            conf=DET_CONF,
            iou=YOLO_NMS_IOU,
            classes=DET_CLASSES,
            verbose=False,
            half=False,
            persist=True,
            tracker=tracker,
        )
