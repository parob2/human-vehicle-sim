"""Actuation and V2X broadcast (thesis §4.2.5): CPM object lists, DENM hazard events,
ego CAM cooperative awareness, ego graded braking from received DENM+CPM, robot indicator from DENM+CAM."""

from __future__ import annotations

import base64
import json
import math
import os
import weakref

import carla

from cooperative_awareness import point_in_any_crossing_zone
from sim_config import (
    CPM_MIN_HEADING_CHANGE_DEG,
    CPM_MIN_POS_CHANGE_M,
    CPM_MIN_SPEED_CHANGE_MS,
    CPM_T_GEN_MAX_S,
    DENM_VALIDITY_DURATION_S,
    EGO_STOP_HOLD_SPEED_MS,
    EGO_STUCK_SPEED_MS,
    ETSI_CAUSE_HUMAN_PRESENCE,
    ETSI_SUB_PEDESTRIAN,
    TTC_AEB_S,
    TTC_HIGH_S,
    TTC_T0_S,
    RSU_MIN_THREAT_SPEED_MS,
    V2X_CUSTOM_B64,
    V2X_CUSTOM_B64_PREFIX,
    V2X_CUSTOM_MAX_MSG_BYTES,
    V2X_FREQUENCY_GHZ,
    V2X_RECEIVER_SENSITIVITY,
    V2X_TRANSMIT_POWER,
)

def _configure_v2x_custom_blueprint(bp):
    """Apply symmetric TX/RX radio parameters before spawning v2x_custom sensors."""
    bp.set_attribute("transmit_power", V2X_TRANSMIT_POWER)
    bp.set_attribute("receiver_sensitivity", V2X_RECEIVER_SENSITIVITY)
    bp.set_attribute("frequency_ghz", V2X_FREQUENCY_GHZ)
    path_loss = os.environ.get("V2X_CUSTOM_PATH_LOSS", "geometric").strip()
    bp.set_attribute("path_loss_model", path_loss)
    return bp


def _v2x_custom_wire_encode(payload_str: str) -> str:
    """Encode payload for CARLA v2x_custom send (optional Base64 to avoid null-byte truncation)."""
    if not payload_str:
        return payload_str
    use_b64 = V2X_CUSTOM_B64 or ("\x00" in payload_str)
    if not use_b64:
        return payload_str
    wire = V2X_CUSTOM_B64_PREFIX + base64.b64encode(payload_str.encode("utf-8")).decode("ascii")
    if len(wire) <= V2X_CUSTOM_MAX_MSG_BYTES:
        return wire
    if use_b64:
        print(
            f"[V2X] B64 wire {len(wire)}B exceeds {V2X_CUSTOM_MAX_MSG_BYTES}B limit — sending raw",
            flush=True,
        )
    return payload_str


def _v2x_custom_wire_decode(text: str) -> str | None:
    """Decode a v2x_custom wire string (plain JSON or B64:-prefixed)."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith(V2X_CUSTOM_B64_PREFIX):
        try:
            return base64.b64decode(text[len(V2X_CUSTOM_B64_PREFIX):], validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None
    return text


def _v2x_custom_payload_text(raw) -> str | None:
    """Extract JSON payload text from CARLA CustomV2XData.get() dict or a wire string."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return _v2x_custom_wire_decode(raw)
    if isinstance(raw, dict):
        msg_block = raw.get("Message") or raw.get("message")
        if isinstance(msg_block, str):
            return _v2x_custom_wire_decode(msg_block)
        if isinstance(msg_block, dict):
            inner = msg_block.get("Message") or msg_block.get("message")
            if isinstance(inner, str):
                return _v2x_custom_wire_decode(inner)
    return None


def _v2x_custom_send(sensor, payload_str: str) -> str:
    """Send on sensor.other.v2x_custom; return the exact wire bytes/string transmitted."""
    wire = _v2x_custom_wire_encode(payload_str)
    if len(wire) > V2X_CUSTOM_MAX_MSG_BYTES:
        print(
            f"[V2X] wire payload {len(wire)}B exceeds {V2X_CUSTOM_MAX_MSG_BYTES}B CARLA buffer",
            flush=True,
        )
    sensor.send(wire)
    return wire


def _parse_v2x_json(raw):
    """Decode a JSON V2X payload from sensor.other.v2x_custom (wire string or CARLA dict)."""
    text = _v2x_custom_payload_text(raw) if not isinstance(raw, str) else _v2x_custom_wire_decode(raw)
    if not text:
        return None
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_denm_payload_from_v2x(raw):
    """Parse compact or verbose DENM JSON from sensor.other.v2x_custom."""
    data = _parse_v2x_json(raw)
    if data is None:
        return None
    return _normalize_denm_payload(data)


def parse_cpm_payload_from_v2x(raw):
    """Parse compact or full ETSI TS 103 324 CPM JSON from sensor.other.v2x_custom."""
    data = _parse_v2x_json(raw)
    if data is None:
        return None
    msg_type = data.get("messageType") or data.get("mT")
    if msg_type != "CPM":
        return None

    objects = _cpm_objects_from_payload(data)
    if not objects:
        return None
    return {
        "stationID": data.get("stationID") or data.get("sid"),
        "generationDeltaTime": data.get("generationDeltaTime"),
        "perceivedObjects": objects,
    }


def _cpm_normalize_object_entry(obj: dict) -> dict | None:
    """Map compact wire keys to ETSI perceived-object field names."""
    oid = obj.get("objectID") or obj.get("oID") or obj.get("id")
    x = obj.get("xDistance") or obj.get("x")
    y = obj.get("yDistance") or obj.get("y")
    if oid is None or x is None or y is None:
        return None
    cls = obj.get("objectClass")
    if cls is None:
        classification = obj.get("classification") or {}
        cls = classification.get("objectClass")
    out = {
        "objectID": _object_id_as_int(oid),
        "objectClass": cls or "pedestrian",
        "xDistance": float(x),
        "yDistance": float(y),
    }
    xs = obj.get("xSpeed")
    ys = obj.get("ySpeed")
    if xs is None and "sx" in obj:
        xs = float(obj["sx"]) / 100.0
    if ys is None and "sy" in obj:
        ys = float(obj["sy"]) / 100.0
    if xs is None:
        vel = obj.get("velocity") or {}
        xs = vel.get("vx")
    if ys is None:
        vel = obj.get("velocity") or {}
        ys = vel.get("vy")
    if xs is not None:
        out["xSpeed"] = float(xs)
    if ys is not None:
        out["ySpeed"] = float(ys)
    return out


def _cpm_objects_from_payload(data: dict) -> list:
    """Extract perceived objects from compact or full CPM payload shapes."""
    objects: list = []

    single = data.get("obj")
    if isinstance(single, dict):
        norm = _cpm_normalize_object_entry(single)
        if norm is not None:
            objects.append(norm)

    perc_obj = data.get("percObj") or {}
    for raw in perc_obj.get("objs") or []:
        if isinstance(raw, dict):
            norm = _cpm_normalize_object_entry(raw)
            if norm is not None:
                objects.append(norm)

    poc = data.get("perceivedObjectContainer") or {}
    for raw in poc.get("perceivedObjects") or []:
        if isinstance(raw, dict):
            norm = _cpm_normalize_object_entry(raw)
            if norm is not None:
                objects.append(norm)

    return objects


def cpm_pedestrian_world_xy(obj: dict):
    """Pedestrian world XY in metres (CARLA API frame)."""
    x = obj.get("xDistance")
    y = obj.get("yDistance")
    if x is None or y is None:
        return None
    return float(x), float(y)


def match_cpm_pedestrian_for_denm(
    denm_object_id,
    denm_event_pos,
    cpm_objects,
    *,
    match_radius_m=5.0,
):
    """Resolve the CPM pedestrian linked to a DENM (objectID first, then eventPosition)."""
    cpm_objects = cpm_objects or []
    if denm_object_id is not None:
        target_id = _object_id_as_int(denm_object_id)
        for obj in cpm_objects:
            if _object_id_as_int(obj.get("objectID")) == target_id:
                if obj.get("objectClass") in (None, "pedestrian"):
                    return obj
    ep = denm_event_pos or {}
    ex = ep.get("x") if isinstance(ep, dict) else None
    ey = ep.get("y") if isinstance(ep, dict) else None
    if ex is None or ey is None:
        return None
    best = None
    best_d = float("inf")
    for obj in cpm_objects:
        if obj.get("objectClass") not in (None, "pedestrian"):
            continue
        xy = cpm_pedestrian_world_xy(obj)
        if xy is None:
            continue
        d = math.hypot(xy[0] - float(ex), xy[1] - float(ey))
        if d < best_d:
            best_d = d
            best = obj
    if best is not None and best_d <= float(match_radius_m):
        return best
    return None


def ego_brake_level_from_ttc_s(ttc_s, *, ttc_aeb_s, ttc_high_s, ttc_t0_s):
    """Graded ego brake [0, 1] from TTC (AEB VRU bands)."""
    if not math.isfinite(ttc_s):
        return 0.0
    if ttc_s <= ttc_aeb_s:
        return 1.0
    if ttc_s <= ttc_high_s:
        return 0.6
    if ttc_s <= ttc_t0_s:
        return 0.2
    return 0.0


def ego_vru_actuator_brake(brake_level, speed_ms, *, stop_hold_speed_ms=None):
    """
    Map graded brake level to CARLA actuator inputs.

    Logged brake_level stays on the TTC bands; CARLA undershoots at 0.2/0.6 so HIGH/AEB
    use full service brake and MEDIUM scales with speed for a reliable stop.
    """
    if brake_level <= 0.0:
        return 0.0, False
    stop_hold = EGO_STOP_HOLD_SPEED_MS if stop_hold_speed_ms is None else stop_hold_speed_ms
    spd = max(0.0, float(speed_ms))
    if brake_level >= 0.6:
        actuator = 1.0
    elif brake_level >= 0.2:
        actuator = min(1.0, 0.55 + 0.09 * spd)
    else:
        actuator = float(brake_level)
    hand = spd < stop_hold
    return actuator, hand


def ego_vru_brake_control(vehicle, brake_level, *, speed_ms=None):
    """Apply closed-loop V2X braking; returns None when navigation should resume."""
    if brake_level <= 0.0:
        return None
    if speed_ms is None:
        vel = vehicle.get_velocity()
        speed_ms = math.hypot(vel.x, vel.y, vel.z)
    actuator, hand = ego_vru_actuator_brake(brake_level, speed_ms)
    return carla.VehicleControl(
        throttle=0.0,
        brake=actuator,
        steer=0.0,
        hand_brake=hand,
        reverse=False,
    )


def ego_ttc_from_cpm_objects(
    ego_x,
    ego_y,
    fwd_x,
    fwd_y,
    ego_speed_ms,
    cpm_objects,
    *,
    half_width_m,
    ahead_max_m,
    behind_m,
    crossing_zone_polygons=None,
):
    """Minimum closing TTC (s) from received CPM pedestrians (world XY, metres)."""
    best = float("inf")
    fn = math.hypot(fwd_x, fwd_y)
    if fn < 1e-6:
        return best
    fx, fy = fwd_x / fn, fwd_y / fn
    ego_vx, ego_vy = fx * float(ego_speed_ms), fy * float(ego_speed_ms)
    for obj in cpm_objects or []:
        if obj.get("objectClass") not in (None, "pedestrian"):
            continue
        xy = cpm_pedestrian_world_xy(obj)
        if xy is None:
            continue
        ox, oy = xy
        in_zone = (
            crossing_zone_polygons is not None
            and point_in_any_crossing_zone(ox, oy, crossing_zone_polygons)
        )
        dx = ox - float(ego_x)
        dy = oy - float(ego_y)
        long_ahead = dx * fx + dy * fy
        lat = abs(dx * (-fy) + dy * fx)
        if not in_zone:
            if long_ahead < -float(behind_m) or long_ahead > float(ahead_max_m):
                continue
            if lat > float(half_width_m):
                continue
        ovx = obj.get("xSpeed") or 0.0
        ovy = obj.get("ySpeed") or 0.0
        ped_spd = math.hypot(float(ovx), float(ovy))
        rvx = float(ovx) - ego_vx
        rvy = float(ovy) - ego_vy
        r = math.hypot(dx, dy)
        if r < 1e-3:
            return 0.0
        closing = -(dx * rvx + dy * rvy) / r
        if closing > 0.1:
            best = min(best, r / closing)
        elif (
            in_zone
            and ped_spd < 0.3
            and long_ahead > 0.0
            and float(ego_speed_ms) > 0.05
        ):
            # Stationary pedestrian in zone: closing rate from ego speed alone.
            best = min(best, long_ahead / float(ego_speed_ms))
    return best


def _object_id_as_int(oid) -> int:
    """Stable integer objectID for CPM/DENM (ByteTrack id, fb_N fallback, numeric str)."""
    if isinstance(oid, int):
        return oid
    s = str(oid).strip()
    if s.isdigit():
        return int(s)
    if s.startswith("fb_"):
        suffix = s.split("_", 1)[1]
        if suffix.isdigit():
            return 900_000 + int(suffix)
    return 800_000 + (abs(hash(s)) % 100_000)


def _cpm_wire_object(obj: dict, *, include_speed: bool = True) -> dict | None:
    """Compact perceived-object entry: objectID + x/y (int metres) + optional sx/sy (cm/s)."""
    oid = obj.get("objectID")
    x = obj.get("xDistance")
    y = obj.get("yDistance")
    if oid is None or x is None or y is None:
        return None
    wire = {
        "id": _object_id_as_int(oid),
        "x": int(round(float(x))),
        "y": int(round(float(y))),
    }
    if include_speed:
        xs = obj.get("xSpeed")
        ys = obj.get("ySpeed")
        if xs is not None or ys is not None:
            wire["sx"] = int(round(float(xs or 0.0) * 100.0))
            wire["sy"] = int(round(float(ys or 0.0) * 100.0))
    return wire


def build_cpm_payload(
    perceived_objects: list,
    *,
    station_id: int,
    ref_x: float,
    ref_y: float,
) -> dict:
    """
    Compact ETSI TS 103 324 logical profile for CARLA v2x_custom (≤99-byte wire limit).

    Abbreviated keys map to ETSI containers:
      mT               → messageType
      sid              → stationID
      mgmt.ref         → managementContainer.referencePosition
      obj              → perceivedObjectContainer.perceivedObjects[0]
      percObj.objs     → perceivedObjectContainer.perceivedObjects
      id / x / y       → objectID / xDistance / yDistance
      sx / sy          → xSpeed / ySpeed (cm/s)
    """
    wire_objs = []
    for obj in perceived_objects:
        wire = _cpm_wire_object(obj, include_speed=True)
        if wire is not None:
            wire_objs.append(wire)

    payload = {
        "mT": "CPM",  # messageType (ETSI ITS PDU header)
        "sid": int(station_id),
        "mgmt": {
            "ref": {
                "x": int(round(float(ref_x))),
                "y": int(round(float(ref_y))),
            },
        },
    }
    if len(wire_objs) == 1:
        payload["obj"] = wire_objs[0]
    elif wire_objs:
        payload["percObj"] = {"objs": wire_objs}
    return payload


def serialize_cpm_v2x_message(payload: dict, *, max_bytes: int = V2X_CUSTOM_MAX_MSG_BYTES) -> str:
    """Serialize CPM JSON; trim speed fields, then objects, until it fits the CARLA buffer."""
    limit = max(1, int(max_bytes))
    working = dict(payload)
    wire_objs = []
    if "obj" in working:
        wire_objs = [dict(working["obj"])]
        working.pop("obj", None)
        working.pop("percObj", None)
    elif isinstance(working.get("percObj"), dict):
        wire_objs = [dict(o) for o in working["percObj"].get("objs") or []]
        working.pop("percObj", None)

    def _pack(objs):
        trial = dict(working)
        if len(objs) == 1:
            trial["obj"] = objs[0]
        elif objs:
            trial["percObj"] = {"objs": objs}
        return json.dumps(trial, separators=(",", ":"), ensure_ascii=True)

    while True:
        text = _pack(wire_objs)
        if len(text) <= limit:
            return text
        for obj in wire_objs:
            if "sx" in obj or "sy" in obj:
                obj.pop("sx", None)
                obj.pop("sy", None)
                text = _pack(wire_objs)
                if len(text) <= limit:
                    return text
        if not wire_objs:
            return text
        wire_objs.pop()


class V2XSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.v2x')
        bp.set_attribute("path_loss_model", "geometric")
        self.hud = hud
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: V2XSensor._V2X_callback(weak_self, sensor_data))

    @staticmethod
    def _V2X_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        print(f"[V2X CAM] message count: {sensor_data.get_message_count()}", flush=True)
        for data in sensor_data:
            msg = data.get()
            power = data.power
            print(f"[V2X CAM] power={power:.1f} dBm | raw={msg}", flush=True)
            self.hud.notification('Cam message received with power %f ' % power)

def _angle_diff_deg(a, b):
    """Smallest absolute difference between two headings in degrees [0, 180]."""
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(d)


def cpm_select_objects(perceived_objects, last_included, now_s):
    """ETSI TS 103 324 object inclusion rules."""
    selected = []
    for obj in perceived_objects:
        oid = obj.get("objectID")
        x = obj.get("xDistance")
        y = obj.get("yDistance")
        speed = math.hypot(obj.get("xSpeed") or 0.0, obj.get("ySpeed") or 0.0)
        heading = obj.get("heading")
        prev = last_included.get(oid)
        include = False
        if prev is None:
            include = True
        elif (now_s - prev["t"]) >= CPM_T_GEN_MAX_S:
            include = True
        elif (
            x is not None and prev["x"] is not None
            and math.hypot(x - prev["x"], y - prev["y"]) >= CPM_MIN_POS_CHANGE_M
        ):
            include = True
        elif abs(speed - prev["speed"]) >= CPM_MIN_SPEED_CHANGE_MS:
            include = True
        elif (
            heading is not None and prev["heading"] is not None
            and _angle_diff_deg(heading, prev["heading"]) >= CPM_MIN_HEADING_CHANGE_DEG
        ):
            include = True
        if include:
            selected.append(obj)
            last_included[oid] = {
                "t": now_s, "x": x, "y": y, "speed": speed, "heading": heading,
            }
    return selected


def build_denm_payload(
    event_pos_x: float,
    event_pos_y: float,
    *,
    information_quality: int = 7,
    relevance_distance: float = 100.0,
    validity_duration: float = 5.0,
    cause_code: int = ETSI_CAUSE_HUMAN_PRESENCE,
    sub_cause_code: int = ETSI_SUB_PEDESTRIAN,
    object_id: int | None = None,
) -> dict:
    """Build a compact DENM logical profile for CARLA v2x_custom (≤99-byte wire limit).

    Abbreviated keys map to ETSI DENM fields consumed by ego braking and robot logic:
      mT  → messageType
      cc  → causeCode (12 humanPresenceOnTheRoad, 97 collisionRisk)
      sc  → subCauseCode (0 unavailable, 4 VRU collision risk)
      vd  → validityDuration (s)
      oid → linkedObjectID / objectID (CPM pedestrian match)
      x/y → eventPosition (world metres, int)

    information_quality and relevance_distance are RSU-side only (not transmitted).
    """
    _ = information_quality, relevance_distance  # kept for RSU logging / API compatibility
    payload = {
        "mT": "DENM",
        "cc": int(cause_code),
        "sc": int(sub_cause_code),
        "vd": int(round(float(validity_duration))),
        "x": int(round(float(event_pos_x))),
        "y": int(round(float(event_pos_y))),
    }
    if object_id is not None:
        payload["oid"] = _object_id_as_int(object_id)
    return payload


def serialize_denm_v2x_message(payload: dict, *, max_bytes: int = V2X_CUSTOM_MAX_MSG_BYTES) -> str:
    """Serialize compact DENM JSON for sensor.other.v2x_custom."""
    limit = max(1, int(max_bytes))
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    if len(text) <= limit:
        return text
    # Last resort: drop objectID (CPM match falls back to eventPosition proximity).
    trimmed = dict(payload)
    trimmed.pop("oid", None)
    text = json.dumps(trimmed, separators=(",", ":"), ensure_ascii=True)
    return text


def build_cam_payload(
    ref_x: float,
    ref_y: float,
    speed_ms: float,
) -> dict:
    """Build a compact ETSI CAM logical profile for sensor.other.v2x_custom.

    Transmitted fields (receiver derives distance-to-crossing locally):
      mT  → messageType ("CAM")
      x/y → basicContainer.referencePosition (planar CARLA world metres)
      sv  → highFrequencyContainer speedValue (0.01 m/s, ETSI integer)
    """
    return {
        "mT": "CAM",
        "x": round(float(ref_x), 2),
        "y": round(float(ref_y), 2),
        "sv": int(round(max(0.0, float(speed_ms)) * 100.0)),
    }


def serialize_cam_v2x_message(payload: dict, *, max_bytes: int = V2X_CUSTOM_MAX_MSG_BYTES) -> str:
    """Serialize compact CAM JSON for sensor.other.v2x_custom."""
    limit = max(1, int(max_bytes))
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    if len(text) <= limit:
        return text
    trimmed = dict(payload)
    for key in ("x", "y"):
        if key in trimmed:
            trimmed[key] = int(round(float(trimmed[key])))
    return json.dumps(trimmed, separators=(",", ":"), ensure_ascii=True)


def _cam_speed_ms_from_payload(data: dict) -> float | None:
    """Decode planar speed (m/s) from CAM high-frequency container fields."""
    if data.get("sv") is not None:
        try:
            return float(data["sv"]) / 100.0
        except (TypeError, ValueError):
            pass
    if data.get("speed_ms") is not None:
        try:
            return float(data["speed_ms"])
        except (TypeError, ValueError):
            pass

    hf = (
        data.get("highFrequencyContainer")
        or data.get("HighFrequencyContainer")
        or data.get("hf")
        or {}
    )
    if not isinstance(hf, dict):
        hf = {}
    bvc = (
        hf.get("basicVehicleContainerHighFrequency")
        or hf.get("Basic Vehicle Container High Frequency")
        or hf
    )
    if not isinstance(bvc, dict):
        bvc = {}
    speed_block = bvc.get("speed") or bvc.get("Speed") or {}
    if isinstance(speed_block, dict):
        raw = speed_block.get("speedValue") or speed_block.get("Value")
        if raw is not None:
            try:
                return float(raw) / 100.0
            except (TypeError, ValueError):
                pass
    for key in ("speedValue", "SpeedValue"):
        if key in bvc:
            try:
                return float(bvc[key]) / 100.0
            except (TypeError, ValueError):
                pass
    return None


def _cam_reference_xy_from_payload(data: dict) -> tuple[float | None, float | None]:
    """Decode planar reference position (m) from CAM basic-container fields."""
    for xk, yk in (("x", "y"), ("X", "Y")):
        if data.get(xk) is not None and data.get(yk) is not None:
            try:
                x = float(data[xk])
                y = float(data[yk])
                if math.isfinite(x) and math.isfinite(y):
                    return x, y
            except (TypeError, ValueError):
                pass

    basic = (
        data.get("basicContainer")
        or data.get("BasicContainer")
        or data.get("Basic Container")
        or data.get("bc")
        or {}
    )
    if not isinstance(basic, dict):
        basic = {}
    ref = (
        basic.get("referencePosition")
        or basic.get("ReferencePosition")
        or basic.get("Reference Position")
        or basic.get("rp")
        or {}
    )
    if not isinstance(ref, dict):
        ref = {}

    carla_pos = ref.get("carlaWorldPosition") or ref.get("CarlaWorldPosition") or {}
    if isinstance(carla_pos, dict) and carla_pos:
        try:
            return float(carla_pos.get("x")), float(carla_pos.get("y"))
        except (TypeError, ValueError):
            pass

    for xk, yk in (("x", "y"), ("X", "Y")):
        if ref.get(xk) is not None and ref.get(yk) is not None:
            try:
                x = float(ref[xk])
                y = float(ref[yk])
                if math.isfinite(x) and math.isfinite(y):
                    return x, y
            except (TypeError, ValueError):
                pass
    return None, None


def _normalize_cam_payload(data: dict) -> dict | None:
    """Map compact or verbose CAM JSON to normalized kinematics used by RSU/robot RX."""
    msg_type = data.get("messageType") or data.get("mT")
    if msg_type != "CAM":
        return None
    speed_ms = _cam_speed_ms_from_payload(data)
    x, y = _cam_reference_xy_from_payload(data)
    if speed_ms is None and x is None and y is None:
        return None
    return {
        "messageType": "CAM",
        "speed_ms": speed_ms,
        "x": x,
        "y": y,
    }


def parse_cam_payload_from_v2x(raw):
    """Parse compact or verbose CAM JSON from sensor.other.v2x_custom."""
    data = _parse_v2x_json(raw)
    if data is None:
        return None
    return _normalize_cam_payload(data)


def _normalize_denm_payload(data: dict) -> dict | None:
    """Map compact or verbose DENM JSON to the normalized flat dict used by RX handlers."""
    msg_type = data.get("messageType") or data.get("mT")

    mgmt = data.get("management")
    if isinstance(mgmt, dict):
        situation = data.get("situation") or {}
        event_type = situation.get("eventType") or {}
        cause = event_type.get("causeCode", situation.get("causeCode"))
        if cause is None:
            return None
        term_raw = mgmt.get("termination")
        is_termination = term_raw is not None
        return {
            "messageType": "DENM",
            "causeCode": int(cause),
            "subCauseCode": event_type.get("subCauseCode", situation.get("subCauseCode")),
            "validityDuration": float(mgmt.get("validityDuration", DENM_VALIDITY_DURATION_S)),
            "termination": bool(is_termination),
            "terminationType": term_raw,
            "actionID": mgmt.get("actionID", {}),
            "referenceTime": mgmt.get("referenceTime"),
            "eventPosition": mgmt.get("eventPosition", {}),
            "informationQuality": situation.get("informationQuality"),
            "objectID": data.get("objectID") or data.get("linkedObjectID"),
        }

    cc = data.get("cc")
    if cc is None:
        cc = data.get("causeCode")
    if cc is None:
        if msg_type not in (None, "DENM"):
            return None
        return None

    sc = data.get("sc")
    if sc is None:
        sc = data.get("subCauseCode")
    vd = data.get("vd")
    if vd is None:
        vd = data.get("validityDuration", DENM_VALIDITY_DURATION_S)
    oid = data.get("oid")
    if oid is None:
        oid = data.get("objectID") or data.get("linkedObjectID")

    event_pos = {}
    if data.get("x") is not None and data.get("y") is not None:
        event_pos = {"x": float(data["x"]), "y": float(data["y"])}
    elif isinstance(data.get("eventPosition"), dict):
        event_pos = dict(data["eventPosition"])

    return {
        "messageType": "DENM",
        "causeCode": int(cc),
        "subCauseCode": int(sc) if sc is not None else None,
        "validityDuration": float(vd),
        "termination": bool(data.get("termination")),
        "terminationType": data.get("termination"),
        "actionID": data.get("actionID", {}),
        "referenceTime": data.get("referenceTime"),
        "eventPosition": event_pos,
        "informationQuality": data.get("informationQuality") or data.get("iq"),
        "objectID": int(oid) if oid is not None else None,
    }


def compute_robot_light_state(
    *,
    denm_human_presence_until,
    current_time,
    cam_speed_ms=None,
    cam_dist_to_crossing_m=None,
    unsafe_ttc_s=TTC_HIGH_S,
    emergency_ttc_s=TTC_AEB_S,
    release_ttc_s=TTC_T0_S,
    min_threat_speed_ms=RSU_MIN_THREAT_SPEED_MS,
    ped_on_crossing: bool = False,
    prev_light_level: str = "NONE",
):
    """
    Robot indicator from received V2X only: DENM gates evaluation; CAM supplies ego kinematics.

    Two-phase VRU approach model:

    Pre-crossing (ped approaching, not yet on crosswalk):
      RED    — ego TTC ≤ unsafe_ttc_s (TTC_HIGH_S = 2.5 s, HIGH/AEB braking band)
      RED    — hysteresis hold: TTC 2.5–4.0 s while recovering from RED, prevents flicker
      GREEN  — TTC > release_ttc_s (T0 = 4.0 s) or ego clearly yielding/slow

    Active crossing (pedestrian confirmed on crosswalk):
      RED    — emergency_ttc_s only (TTC_AEB_S = 1.5 s); vehicle clearly not stopping
      GREEN  — all other cases; pedestrian has right of way, vehicle must yield

    NONE   — no active DENM (no pedestrian presence event active)
    """
    if current_time > denm_human_presence_until:
        return "NONE", None, "off"

    v_ego = cam_speed_ms
    d_ego = cam_dist_to_crossing_m
    if v_ego is None or d_ego is None or not math.isfinite(d_ego):
        return "GREEN", carla.Color(r=0, g=255, b=0), "denm_active_cam_pending"

    v_ego = float(v_ego)
    d_ego = float(d_ego)
    if v_ego < min_threat_speed_ms or v_ego < EGO_STUCK_SPEED_MS:
        return "GREEN", carla.Color(r=0, g=255, b=0), "ego_yielding_or_slow-CAM"

    ttc = d_ego / max(v_ego, 0.5)

    if ped_on_crossing:
        # Pedestrian has right of way — only a true AEB emergency overrides GREEN.
        if ttc <= float(emergency_ttc_s):
            return "RED", carla.Color(r=255, g=0, b=0), f"aeb_emergency_ttc={ttc:.1f}s"
        return "GREEN", carla.Color(r=0, g=255, b=0), f"ped_crossing_yield_ttc={ttc:.1f}s"

    # Pre-crossing: HIGH-zone threat check.
    if ttc <= float(unsafe_ttc_s):
        return "RED", carla.Color(r=255, g=0, b=0), f"vehicle_threat_ttc={ttc:.1f}s"

    # Hysteresis: once RED, hold until TTC recovers to T0 (4.0 s) to avoid flicker.
    if prev_light_level == "RED" and ttc < float(release_ttc_s):
        return "RED", carla.Color(r=255, g=0, b=0), f"hysteresis_hold_ttc={ttc:.1f}s"

    return "GREEN", carla.Color(r=0, g=255, b=0), f"safe_to_cross_ttc={ttc:.1f}s"


def draw_robot_light_block(debug, center, color, *, half_extent_m, life_time_s):
    """Large flat square above the robot (reference screenshot style)."""
    bb = carla.BoundingBox(
        carla.Vector3D(float(center.x), float(center.y), float(center.z)),
        carla.Vector3D(float(half_extent_m), float(half_extent_m), 0.04),
    )
    debug.draw_box(
        bb,
        carla.Rotation(pitch=0, yaw=0, roll=0),
        thickness=0.12,
        color=color,
        life_time=float(life_time_s),
    )
