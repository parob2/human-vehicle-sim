"""Simulation configuration: env vars, paths, and CARLA PythonAPI bootstrap."""

import os
import sys
from pathlib import Path

import carla

def _prepend_carla_pythonapi_carla():
    extra = os.environ.get("CARLA_PYTHONAPI_CARLA", "").strip()
    if extra and os.path.isdir(extra) and extra not in sys.path:
        sys.path.insert(0, extra)
        return
    root = os.environ.get("CARLA_ROOT", "").strip()
    if root:
        p = os.path.join(root, "PythonAPI", "carla")
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


_prepend_carla_pythonapi_carla()

RF_MODEL_PATH = os.environ.get("RF_INTENT_MODEL", "").strip()

# Ma & Rong PIE 9-feature heading model (PedestrianIntentPrediction/pie/). Default for simulation.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PIE_RF_DEFAULT = str(
    _REPO_ROOT / "PedestrianIntentPrediction/pie/models/rf_ma_rong_pie_heading.joblib"
)
PIP_RF_MODEL = os.environ.get("PIP_RF_MODEL", _PIE_RF_DEFAULT).strip()
_CARLA_DIR = Path(__file__).resolve().parent

# Active test model (medium — faster than x, less VRAM).
YOLO_POSE_MODEL_DEFAULT = "yolov8m-pose.pt"


def resolve_yolo_model_path(name: str) -> str:
    """Prefer local .pt weights; only fall back to Ultralytics download if missing."""
    raw = (name or YOLO_POSE_MODEL_DEFAULT).strip()
    p = Path(raw).expanduser()
    if p.is_file():
        return str(p.resolve())
    for candidate in (
        _CARLA_DIR / raw,
        _CARLA_DIR.parent / raw,
        _REPO_ROOT / raw,
        _REPO_ROOT / "PedestrianIntentPrediction" / raw,
    ):
        if candidate.is_file():
            return str(candidate.resolve())
    return raw


YOLO_MODEL_NAME = resolve_yolo_model_path(
    os.environ.get("YOLO_POSE_MODEL", YOLO_POSE_MODEL_DEFAULT)
)

# Output images (annotated camera frames on each analysis step)
CAPTURE_DIR = os.environ.get(
    "SIM_CAPTURE_DIR",
    str(Path(__file__).resolve().parent / "simulation_captures"),
)

# Validation / scenario harness (see eval_sim/scenarios.yaml)
SIM_SCENARIO_ID = os.environ.get("SIM_SCENARIO_ID", "default")
SIM_RUN_LOGGER = os.environ.get("SIM_RUN_LOGGER", "1").strip().lower() not in ("0", "false", "no")
SIM_VALIDATION_MODE = (
    os.environ.get("SIM_VALIDATION_MODE", "").strip().lower() in ("1", "true", "yes")
    or SIM_RUN_LOGGER
)
SIM_MAX_DURATION_S = float(os.environ.get("SIM_MAX_DURATION_S", "0"))  # 0 = run until Ctrl+C
SIM_RANDOM_SEED = int(os.environ.get("SIM_RANDOM_SEED", "42"))
# Walker geometry: preset name + module (eval.spawn_presets for main.py, eval_sim.spawn_presets for batch eval).
WALKER_SPAWN_PRESET = os.environ.get("WALKER_SPAWN_PRESET", "").strip() or None
DISABLE_WALKER_SPAWN = os.environ.get("DISABLE_WALKER_SPAWN", "").strip().lower() in ("1", "true", "yes")
WALKER_START_DELAY_S = float(os.environ.get("WALKER_START_DELAY_S", "3.0"))
WALKER_SPAWN_X = os.environ.get("WALKER_SPAWN_X")
WALKER_SPAWN_Y = os.environ.get("WALKER_SPAWN_Y")
WALKER_SPAWN_YAW = os.environ.get("WALKER_SPAWN_YAW")
WALKER_SPEED_MS = os.environ.get("WALKER_SPEED_MS")
WALKER_BP_DEFAULT = os.environ.get("WALKER_BP_DEFAULT", "walker.pedestrian.0004").strip()
_WALKER_BP_VARIANT_RAW = os.environ.get(
    "WALKER_BP_VARIANTS",
    "0004,0003,0005,0006,0007,0008,0009,0010",
).strip()
# Fault injection (eval/fault_injection.py live counterparts)
FAULT_DROP_DETECTIONS_N = int(os.environ.get("FAULT_DROP_DETECTIONS_N", "0"))
FAULT_PROJECTION_NOISE_M = float(os.environ.get("FAULT_PROJECTION_NOISE_M", "0"))
FAULT_INTENT_SPIKE = os.environ.get("FAULT_INTENT_SPIKE", "").strip().lower() in ("1", "true", "yes")

ANALYSIS_INTERVAL_S = 0.25

# RSU master arbitration (distance to crossing + TTC + intent). Tune via env.
RSU_MASTER_RANGE_M = float(os.environ.get("RSU_MASTER_RANGE_M", "50"))
RSU_MIN_THREAT_SPEED_MS = float(os.environ.get("RSU_MIN_THREAT_SPEED_MS", "1.5"))

# AEB VRU TTC thresholds for graded braking and robot indicator
# T0 = 4.0 s — monitoring window opens (test validity boundary)
# AEB = 1.5 s — emergency braking activation threshold (≥ -1 m/s²)
TTC_T0_S  = float(os.environ.get("TTC_T0_S", "4.0"))
TTC_AEB_S = float(os.environ.get("TTC_AEB_S", "1.5"))
TTC_HIGH_S = float(os.environ.get("TTC_HIGH_S", "2.5"))

# ETSI EN 302 637-3 DENM cause / sub-cause codes
ETSI_CAUSE_HUMAN_PRESENCE = 12   # humanPresenceOnTheRoad
ETSI_SUB_PEDESTRIAN       = 0    # unavailable (sub-cause of humanPresenceOnTheRoad)
ETSI_CAUSE_COLLISION_RISK = 97   # collisionRisk
ETSI_SUB_COLLISION_PED    = 4    # VRU collision risk
DENM_VALIDITY_DURATION_S = float(os.environ.get("DENM_VALIDITY_DURATION_S", "600"))

RSU_STATION_ID = int(os.environ.get("RSU_STATION_ID", "2001"))

# ETSI TS 102 637-2 CAM generation interval (ego OBU → RSU via v2x_custom).
CAM_T_GEN_MIN_S = float(os.environ.get("CAM_T_GEN_MIN_S", "0.1"))

# ETSI TS 103 324 CPM generation and per-object inclusion rules.
CPM_T_GEN_MIN_S = float(os.environ.get("CPM_T_GEN_MIN_S", "0.1"))
CPM_T_GEN_MAX_S = float(os.environ.get("CPM_T_GEN_MAX_S", "1.0"))
CPM_MIN_POS_CHANGE_M = float(os.environ.get("CPM_MIN_POS_CHANGE_M", "4.0"))
CPM_MIN_SPEED_CHANGE_MS = float(os.environ.get("CPM_MIN_SPEED_CHANGE_MS", "0.5"))
CPM_MIN_HEADING_CHANGE_DEG = float(os.environ.get("CPM_MIN_HEADING_CHANGE_DEG", "4.0"))

# CARLA sensor.other.v2x_custom: CustomV2XM_t.message is char message[100] (LibITS.h).
V2X_CUSTOM_MAX_MSG_BYTES = int(os.environ.get("V2X_CUSTOM_MAX_MSG_BYTES", "99"))
# Optional plain-text RSU debug broadcast on v2x_custom.
# Keep disabled by default so DENM/CPM are not competing with extra traffic.
V2X_SEND_PLAIN_RSU_DEBUG = os.environ.get("V2X_SEND_PLAIN_RSU_DEBUG", "").strip().lower() in (
    "1", "true", "yes",
)
# Symmetric ITS-G5-style radio parameters for sensor.other.v2x_custom (CARLA attribute names).
V2X_TRANSMIT_POWER = os.environ.get("V2X_TRANSMIT_POWER", "20")
V2X_RECEIVER_SENSITIVITY = os.environ.get("V2X_RECEIVER_SENSITIVITY", "-100")
V2X_FREQUENCY_GHZ = os.environ.get("V2X_FREQUENCY_GHZ", "5.9")
# Base64-wrap wire payloads to survive CARLA C++ std::string null-byte truncation (optional).
V2X_CUSTOM_B64 = os.environ.get("V2X_CUSTOM_B64", "").strip().lower() in ("1", "true", "yes")
V2X_CUSTOM_B64_PREFIX = "B64:"

# Terminal debug for RSU hazard decision policy / ego brake (every ~0.5 s when active)
RSU_HAZARD_DECISION_DEBUG = os.environ.get(
    "RSU_HAZARD_DECISION_DEBUG",
    os.environ.get("RSU_PRIORITY_DEBUG", ""),
).strip().lower() in (
    "1", "true", "yes",
)

# Ego path clearance: release brake when no pedestrian blocks the driving corridor.
EGO_PATH_HALF_WIDTH_M = float(os.environ.get("EGO_PATH_HALF_WIDTH_M", "2.5"))
EGO_PATH_AHEAD_MAX_M = float(os.environ.get("EGO_PATH_AHEAD_MAX_M", "45.0"))
EGO_PASS_BEHIND_M = float(os.environ.get("EGO_PASS_BEHIND_M", "3.0"))
# When ego is nearly stopped, resume if path is clear (avoids stuck at brake=0.2).
EGO_STUCK_SPEED_MS = float(os.environ.get("EGO_STUCK_SPEED_MS", "0.35"))
# Hand-brake hold once speed drops below this while a V2X brake command is active.
EGO_STOP_HOLD_SPEED_MS = float(os.environ.get("EGO_STOP_HOLD_SPEED_MS", "0.5"))

# ARI robot (`vehicle.pal.ari_robot`): crossing motion replaces legacy walker actors.
ROBOT_CROSSING_THROTTLE = float(os.environ.get("ROBOT_CROSSING_THROTTLE", "0.18"))
ROBOT_LIGHT_BLOCK_HALF_M = float(os.environ.get("ROBOT_LIGHT_BLOCK_HALF_M", "0.85"))
ROBOT_LIGHT_BLOCK_LIFE_S = float(os.environ.get("ROBOT_LIGHT_BLOCK_LIFE_S", "0.15"))
# Ignore RSU vision detections projected around ARI robot pose (meters).
ARI_ROBOT_VISION_EXCLUDE_M = float(os.environ.get("ARI_ROBOT_VISION_EXCLUDE_M", "2.5"))

CAM_IMAGE_W = 1280
CAM_IMAGE_H = 720
# Horizontal FOV (deg): lower = more zoom. Default wide; reduce once aim is correct (e.g. CAM_FOV=45).
CAM_FOV = float(os.environ.get("CAM_FOV", "75.0"))
# RSU pole base (Unreal Editor cm / 100 -> CARLA API meters)
RSU_X = -83.58
RSU_Y = -155.22
RSU_Z = 363.52
# RSU_X = -107.90
# RSU_Y = -185.20
# RSU_Z = 364.60
# RSU_YAW = float(os.environ.get("RSU_YAW", "-90.0"))

CAM_REL_Z = 8.0
# Auto-aim: camera looks at this world point (crossing centroid). Override when crossing moves.
RSU_LOOK_AT_X = float(os.environ.get("RSU_LOOK_AT_X", "-112.0"))
RSU_LOOK_AT_Y = float(os.environ.get("RSU_LOOK_AT_Y", "-157.5"))
RSU_LOOK_AT_Z = float(os.environ.get("RSU_LOOK_AT_Z", "364.5"))
CAM_YAW_OFFSET = float(os.environ.get("CAM_YAW_OFFSET", "0.0"))
CAM_PITCH_OFFSET = float(os.environ.get("CAM_PITCH_OFFSET", "-18.0"))
# Set RSU_MANUAL_CAM=1 to use fixed CAM_PITCH / CAM_YAW instead of look-at.
RSU_MANUAL_CAM = os.environ.get("RSU_MANUAL_CAM", "").strip().lower() in ("1", "true", "yes")
CAM_PITCH = float(os.environ.get("CAM_PITCH", "-20.0"))
CAM_YAW = float(os.environ.get("CAM_YAW", "245.0"))
# RSU_MANUAL_CAM = os.environ.get("RSU_MANUAL_CAM", "1").strip().lower() in ("1", "true", "yes")
# CAM_PITCH = float(os.environ.get("CAM_PITCH", "0.0"))
# CAM_YAW = float(os.environ.get("CAM_YAW", "90.0"))

# Pose / detection gates (relaxed defaults for thesis eval; override via YOLO_* env vars)
# Smaller imgsz reduces VRAM during warmup/inference (critical with CARLA on same GPU).
DET_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "1280"))
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "").strip() or None
YOLO_HALF = os.environ.get("YOLO_HALF", "0").lower() in ("1", "true", "yes")
DET_CONF = float(os.environ.get("YOLO_CONF", "0.10"))
DET_CLASSES = [0]
YOLO_NMS_IOU = float(os.environ.get("YOLO_IOU", "0.72"))
KP_CONF_THRESH = 0.30
MIN_VISIBLE_KPTS = 2
MIN_KPT_MAX_CONF = 0.30
MIN_HIP_CONF = 0.08
MIN_BBOX_H_PX = int(os.environ.get("YOLO_MIN_BBOX_H_PX", "10"))
MIN_ASPECT_RATIO = float(os.environ.get("YOLO_MIN_ASPECT_RATIO", "0.65"))  # bbox_h / bbox_w
MIN_TRACK_AGE_FRAMES = 1
_DEFAULT_NEW_TRACK = str(max(0.05, DET_CONF - 0.07))
CONF_NEW_TRACK_MIN = float(os.environ.get("YOLO_CONF_NEW_TRACK_MIN", _DEFAULT_NEW_TRACK))
YOLO_DEBUG_RAW = os.environ.get("YOLO_DEBUG_RAW", "").strip().lower() in ("1", "true", "yes")
YOLO_GATE_DEBUG = os.environ.get("YOLO_GATE_DEBUG", "").strip().lower() in ("1", "true", "yes")

_TRACKER_DEFAULT = Path(__file__).resolve().parent.parent / "bytetrack_low_thresh.yaml"
YOLO_TRACKER = os.environ.get(
    "YOLO_TRACKER",
    str(_TRACKER_DEFAULT) if _TRACKER_DEFAULT.is_file() else "bytetrack.yaml",
)
FALLBACK_ID_MATCH_RATIO = 0.5
FALLBACK_MAX_AGE_ANALYSIS_STEPS = 80

# ETSI CPM object class mapping (YOLO class index → ETSI CPM semantic class string)
ETSI_CLASS_MAP = {0: "pedestrian", 1: "cyclist", 2: "passengerVehicle", 3: "motorcycle"}

# Fused confidence weights: det_conf * W_DET + tracking_conf * W_TRACK + intent_proba * W_INTENT
CPM_CONF_W_DET = 0.5
CPM_CONF_W_TRACK = 0.2
CPM_CONF_W_INTENT = 0.3

# COCO17 pose limbs for visualization (indices match ultralytics/YOLO pose)
COCO17_LIMBS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]
POSE_DRAW_CONF = 0.25

COCO_MAPPING = {
    0: "crl_Head",
    5: "crl_arm_l",
    6: "crl_arm_r",
    7: "crl_forearm_l",
    8: "crl_forearm_r",
    9: "crl_hand_l",
    10: "crl_hand_r",
    11: "crl_thigh_l",
    12: "crl_thigh_r",
    13: "crl_leg_l",
    14: "crl_leg_r",
    15: "crl_foot_l",
    16: "crl_foot_r",
}

# BP_PowerPole2 (Unreal Editor cm) -> CARLA Python API meters (divide by 100)
EGO_DESTINATION = carla.Location(x=-111.7, y=-146.6, z=363.7)
# BasicAgent target cruise speed (km/h); local planner PID follows route / hazards.
NAV_TARGET_SPEED_KMH = float(os.environ.get("EGO_NAV_TARGET_SPEED_KMH", "20"))
# In synchronous mode the server only advances when the client calls world.tick().
# A few ticks after heavy spawn / before navigation routing avoids a frozen viewport and stuck RPCs.
CARLA_BOOTSTRAP_TICKS = int(os.environ.get("CARLA_BOOTSTRAP_TICKS", "25"))
# Extra ticks after spawns: map/camera update while ego is held (sync mode only advances on tick).
CARLA_WARMUP_TICKS = int(os.environ.get("CARLA_WARMUP_TICKS", "150"))
# Coarser graph = faster GlobalRoutePlanner build (default in BasicAgent is 2.0 m).
CARLA_GRP_SAMPLING = float(os.environ.get("CARLA_GRP_SAMPLING", "8.0"))
# "lane" = greedy OpenDRIVE waypoint following (roads, no full GlobalRoutePlanner graph).
# "simple" = steer toward destination XY (shortcut across terrain).
# "basic" = CARLA BasicAgent + GlobalRoutePlanner (full agent; topology build can take minutes).
EGO_NAV_MODE = os.environ.get("EGO_NAV_MODE", "lane").strip().lower()
# Shorter lookahead tracks tight curves (e.g. roundabouts) instead of aiming past the bend.
EGO_LANE_LOOKAHEAD = float(os.environ.get("EGO_LANE_LOOKAHEAD", "6.5"))
EGO_SIMPLE_ARRIVE_DIST = float(os.environ.get("EGO_SIMPLE_ARRIVE_DIST", "5.0"))
# Multiplies signed heading error (rad) into [-1,1] steer; atan2 path gives full lock when needed.
EGO_SIMPLE_STEER_GAIN = float(os.environ.get("EGO_SIMPLE_STEER_GAIN", "2.75"))

