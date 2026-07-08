"""YOLOv8 pose → Ma-Rong OpenPose-style hip/knee/ankle keypoints."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

# COCO / YOLOv8-pose indices
YOLO_LEFT_HIP = 11
YOLO_RIGHT_HIP = 12
YOLO_LEFT_KNEE = 13
YOLO_RIGHT_KNEE = 14
YOLO_LEFT_ANKLE = 15
YOLO_RIGHT_ANKLE = 16

YOLO_IDX = {
    "hip_l": YOLO_LEFT_HIP,
    "hip_r": YOLO_RIGHT_HIP,
    "knee_l": YOLO_LEFT_KNEE,
    "knee_r": YOLO_RIGHT_KNEE,
    "ankle_l": YOLO_LEFT_ANKLE,
    "ankle_r": YOLO_RIGHT_ANKLE,
}


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(area_a + area_b - inter, 1e-8))


def match_pose_to_bbox(result, gt_bbox: Tuple[float, float, float, float], min_iou: float = 0.15):
    if result.boxes is None or len(result.boxes) == 0:
        return None
    if result.keypoints is None:
        return None

    best_iou = 0.0
    best_idx = None
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    for i, box in enumerate(boxes_xyxy):
        iou = bbox_iou(tuple(box.tolist()), gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    if best_idx is None or best_iou < min_iou:
        return None
    return best_idx


def keypoints_from_detection(
    result,
    det_idx: int,
    min_conf: float = 0.15,
) -> Optional[Dict[str, Tuple[float, float]]]:
    kpts = result.keypoints.xy[det_idx].cpu().numpy()
    conf = result.keypoints.conf[det_idx].cpu().numpy()

    out: Dict[str, Tuple[float, float]] = {}
    for name, idx in YOLO_IDX.items():
        if conf[idx] < min_conf:
            return None
        out[name] = (float(kpts[idx][0]), float(kpts[idx][1]))
    return out


def crop_with_margin(
    frame,
    bbox: Tuple[float, float, float, float],
    margin: float = 0.15,
):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - margin * bw))
    y1 = max(0, int(y1 - margin * bh))
    x2 = min(w, int(x2 + margin * bw))
    y2 = min(h, int(y2 + margin * bh))
    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1)


def extract_joints_from_frame(
    model,
    frame,
    gt_bbox: Tuple[float, float, float, float],
    *,
    device: str,
    conf: float,
    min_iou: float = 0.15,
):
    """Run pose on a padded pedestrian crop (more reliable than full-frame matching)."""
    crop, (ox, oy) = crop_with_margin(frame, gt_bbox)
    if crop.size == 0:
        return None

    results = model.predict(crop, conf=conf, verbose=False, device=device)
    if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
        return None

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        det_idx = int(result.keypoints.conf.sum(dim=1).argmax())
    else:
        local_bbox = (gt_bbox[0] - ox, gt_bbox[1] - oy, gt_bbox[2] - ox, gt_bbox[3] - oy)
        det_idx = match_pose_to_bbox(result, local_bbox, min_iou=min_iou)
        if det_idx is None:
            det_idx = int(result.boxes.conf.argmax())

    joints = keypoints_from_detection(result, det_idx, min_conf=0.15)
    if joints is None:
        return None

    return {name: (xy[0] + ox, xy[1] + oy) for name, xy in joints.items()}
