"""Cooperative awareness (thesis §4.2.3): off-zone / on-zone context, ego CAM kinematics,
path-conflict geometry for decision inputs."""

from __future__ import annotations

import math

import carla

from rsu_perception import (
    is_point_in_polygon,
    min_distance_to_any_crossing_zone,
    point_in_any_crossing_zone,
    point_to_polygon_distance,
)
from sim_config import (
    EGO_PASS_BEHIND_M,
    EGO_PATH_AHEAD_MAX_M,
    EGO_PATH_HALF_WIDTH_M,
)

def parse_cam_speed_ms(cam_msg):
    """Extract speed (m/s) from a CARLA sensor.other.v2x CAM payload (ETSI units when applicable)."""
    try:
        data = cam_msg.get()
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    def _speed_from_raw(speed_raw):
        if speed_raw is None:
            return None
        try:
            return float(speed_raw) / 100.0
        except (TypeError, ValueError):
            try:
                return float(speed_raw)
            except (TypeError, ValueError):
                return None

    cam_params = data.get("cam", {}).get("camParameters", {})
    if cam_params:
        hf = cam_params.get("highFrequencyContainer", {}).get(
            "basicVehicleContainerHighFrequency", {}
        )
        speed_block = hf.get("speed")
        if isinstance(speed_block, dict):
            parsed = _speed_from_raw(speed_block.get("speedValue"))
            if parsed is not None:
                return parsed

    def _speed_from_hf(hf_block):
        if not isinstance(hf_block, dict):
            return None
        speed_block = hf_block.get("speed") or hf_block.get("Speed")
        if isinstance(speed_block, dict):
            return _speed_from_raw(
                speed_block.get("speedValue") or speed_block.get("Value")
            )
        for key in ("speedValue", "SpeedValue"):
            if key in hf_block:
                return _speed_from_raw(hf_block[key])
        return None

    msg = data.get("Message") or data.get("message") or data
    # CARLA nested CAM: Message → Message → CAM Parameters → High Frequency Container
    inner = msg.get("Message") if isinstance(msg, dict) else None
    if isinstance(inner, dict):
        cam_params = inner.get("CAM Parameters") or inner.get("camParameters") or {}
        hf_wrap = cam_params.get("High Frequency Container") or cam_params.get(
            "highFrequencyContainer", {}
        )
        if isinstance(hf_wrap, dict):
            bvc = hf_wrap.get("Basic Vehicle Container High Frequency") or hf_wrap.get(
                "basicVehicleContainerHighFrequency", {}
            )
            parsed = _speed_from_hf(bvc if isinstance(bvc, dict) else hf_wrap)
            if parsed is not None:
                return parsed

    hf = msg.get("highFrequencyContainer") or msg.get("HighFrequencyContainer") or {}
    if isinstance(hf, dict):
        bvc = hf.get("basicVehicleContainerHighFrequency") or hf.get(
            "Basic Vehicle Container High Frequency", hf
        )
        parsed = _speed_from_hf(bvc if isinstance(bvc, dict) else hf)
        if parsed is not None:
            return parsed
    for key in ("speedValue", "SpeedValue"):
        if key in hf:
            parsed = _speed_from_raw(hf[key])
            if parsed is not None:
                return parsed
    if "speed" in msg:
        try:
            return float(msg["speed"])
        except (TypeError, ValueError):
            pass
    return None


# CARLA CaService encodes actor world X/Y in ETSI latitude/longitude integer fields
# (see Unreal/.../CaService.cpp: round(RefPos.X * 1e6) * oneMicroDegreeNorth).
CARLA_CAM_LAT_UNAVAILABLE = 900_000_001
CARLA_CAM_LON_UNAVAILABLE = 1_800_000_001
CARLA_CAM_WORLD_XY_SCALE = 1e7
ETSI_CAM_ALTITUDE_UNAVAILABLE = 800_000
ETSI_CAM_ALTITUDE_MIN = -100_000
ETSI_CAM_ALTITUDE_SCALE_M = 0.01

# Cached WGS84→CARLA linearisation from Map.transform_to_geolocation (fallback).
_CAM_GEO_PROJECTION = None

def _cam_parameters_from_data(data):
    """Locate the CAM Parameters block across CARLA / compact payload layouts."""
    if not isinstance(data, dict):
        return {}

    cam = data.get("cam")
    if isinstance(cam, dict):
        params = cam.get("camParameters") or cam.get("cam_parameters")
        if isinstance(params, dict):
            return params

    msg = data.get("Message") or data.get("message") or data
    if not isinstance(msg, dict):
        return {}

    candidates = [msg]
    inner = msg.get("Message") or msg.get("message")
    if isinstance(inner, dict):
        candidates.append(inner)

    for block in candidates:
        params = (
            block.get("CAM Parameters")
            or block.get("camParameters")
            or block.get("cam_parameters")
        )
        if isinstance(params, dict):
            return params
    return {}


def _cam_reference_position_block(cam_params):
    """Return the basic-container referencePosition dict from CAM parameters."""
    if not isinstance(cam_params, dict):
        return {}

    basic = (
        cam_params.get("Basic Container")
        or cam_params.get("BasicContainer")
        or cam_params.get("basicContainer")
        or {}
    )
    if not isinstance(basic, dict):
        basic = {}

    ref = (
        basic.get("Reference Position")
        or basic.get("ReferencePosition")
        or basic.get("referencePosition")
        or cam_params.get("referencePosition")
        or cam_params.get("ReferencePosition")
        or {}
    )
    return ref if isinstance(ref, dict) else {}


def _etsi_cam_coord_scaled(raw):
    """Decode one ETSI referencePosition lat/lon/alt integer (1e-7 units) to float."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw / CARLA_CAM_WORLD_XY_SCALE
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    # Large magnitudes are still ASN.1 integers that arrived as float (keep full scale).
    if abs(val) >= 1_000_000.0:
        return val / CARLA_CAM_WORLD_XY_SCALE
    return val


def _cam_reference_altitude_m(ref_pos):
    """Decode CAM referencePosition altitude to metres (ETSI 0.01 m units)."""
    if not isinstance(ref_pos, dict):
        return 0.0
    for key in ("Altitude", "altitude"):
        alt_block = ref_pos.get(key)
        if alt_block is None:
            continue
        if isinstance(alt_block, dict):
            raw = alt_block.get("altitudeValue") or alt_block.get("value") or alt_block.get("Value")
        else:
            raw = alt_block
        try:
            raw_int = int(raw)
        except (TypeError, ValueError):
            continue
        if raw_int >= ETSI_CAM_ALTITUDE_UNAVAILABLE or raw_int < ETSI_CAM_ALTITUDE_MIN:
            return 0.0
        return raw_int * ETSI_CAM_ALTITUDE_SCALE_M
    return 0.0


def init_cam_geo_projection(carla_map):
    """
    Sample Map.transform_to_geolocation to build a local WGS84→CARLA XY fallback.

    Used when geolocation_to_transform fails (e.g. bad altitude in CAM payload or
    a broken/absent map GeoReference that makes geolocation_to_transform return
    an identity-like result).

    Sampling strategy: try multiple anchor points because world (0, 0) may lie
    outside the map's loaded terrain tiles (trail24's working area is around
    x ≈ −120, y ≈ −165).  The first anchor that yields a non-degenerate
    lat/lon differential wins.

    If every CARLA API attempt fails (map has no GeoReference at all), fall back
    to the hardcoded trail24 linearisation derived from the known empirical
    correspondence world (−117.07, −170.36) ↔ WGS84 (48.75 °N, 11.48 °E).
    """
    global _CAM_GEO_PROJECTION
    _CAM_GEO_PROJECTION = None
    if carla_map is None:
        return None

    # Try progressively: world origin first (general maps), then trail24 area.
    candidate_origins = [
        (0.0, 0.0),
        (-117.07, -170.36),  # trail24 crossing zone
        (-50.0, -100.0),
    ]
    for ref_ox, ref_oy in candidate_origins:
        try:
            samples = []
            for dx, dy in ((0.0, 0.0), (100.0, 0.0), (0.0, 100.0)):
                geo = carla_map.transform_to_geolocation(
                    carla.Location(x=ref_ox + dx, y=ref_oy + dy, z=0.0)
                )
                samples.append((ref_ox + dx, ref_oy + dy, geo.latitude, geo.longitude))
            ref_x, ref_y, ref_lat, ref_lon = samples[0]
            _, _, _, lon_x = samples[1]
            _, _, lat_y, _ = samples[2]
            dlon_dx = lon_x - ref_lon
            dlat_dy = lat_y - ref_lat
            if abs(dlon_dx) < 1e-12 or abs(dlat_dy) < 1e-12:
                raise ValueError("degenerate geo calibration")
            _CAM_GEO_PROJECTION = {
                "ref_x": ref_x,
                "ref_y": ref_y,
                "ref_lat": ref_lat,
                "ref_lon": ref_lon,
                "m_per_deg_lon": 100.0 / dlon_dx,
                "m_per_deg_lat": 100.0 / dlat_dy,
            }
            print(
                "[V2X CAM] geo calibration ok "
                f"anchor=({ref_ox:.0f},{ref_oy:.0f}) "
                f"ref=({ref_lat:.6f},{ref_lon:.6f}) "
                f"m/deg=({_CAM_GEO_PROJECTION['m_per_deg_lat']:.1f},"
                f"{_CAM_GEO_PROJECTION['m_per_deg_lon']:.1f})",
                flush=True,
            )
            return _CAM_GEO_PROJECTION
        except Exception as ex:
            print(
                f"[V2X CAM] geo calibration failed at ({ref_ox:.0f},{ref_oy:.0f}): {ex}",
                flush=True,
            )

    # Absolute last resort: hardcoded trail24 linearisation.
    # Empirical: world (−117.07, −170.36) ↔ WGS84 (48.75 °N, 11.48 °E).
    # m_per_deg at 48.75 °N — same values as test_geometry_and_v2x mock.
    _CAM_GEO_PROJECTION = {
        "ref_x": -117.07,
        "ref_y": -170.36,
        "ref_lat": 48.75,
        "ref_lon": 11.48,
        "m_per_deg_lat": 111_000.0,
        "m_per_deg_lon": 69_000.0,
    }
    print(
        "[V2X CAM] geo calibration: using hardcoded trail24 fallback "
        "ref=(48.75,11.48) m/deg=(111000,69000)",
        flush=True,
    )
    return _CAM_GEO_PROJECTION


def _cam_latlon_is_wgs84(lat_val, lon_val):
    """True when decoded lat/lon look like geographic degrees, not CARLA world metres."""
    if lat_val is None or lon_val is None:
        return False
    if not (math.isfinite(lat_val) and math.isfinite(lon_val)):
        return False
    if abs(lat_val) > 90.0 or abs(lon_val) > 180.0:
        return False
    # trail24 and similar maps use negative world-frame coordinates (hundreds of metres).
    if lat_val < 0.0 and lon_val < 0.0:
        return False
    if abs(lat_val) > 500.0 or abs(lon_val) > 500.0:
        return False
    return True


def _decode_carla_cam_world_xy(lat_raw, lon_raw):
    """Decode CARLA CaService hack: world X/Y stored in ETSI lat/lon integer fields."""
    try:
        lat = int(lat_raw)
        lon = int(lon_raw)
    except (TypeError, ValueError):
        return None, None
    if lat in (CARLA_CAM_LAT_UNAVAILABLE, -CARLA_CAM_LAT_UNAVAILABLE):
        return None, None
    if lon in (CARLA_CAM_LON_UNAVAILABLE, -CARLA_CAM_LON_UNAVAILABLE):
        return None, None
    return lat / CARLA_CAM_WORLD_XY_SCALE, lon / CARLA_CAM_WORLD_XY_SCALE


_CAM_GEO_PROJ_IDENTITY_WARNED = False


def _cam_wgs84_to_carla_xy(lat_deg, lon_deg, alt_m, carla_map, geo_proj=None):
    """Project WGS84 CAM referencePosition to CARLA map XY (metres).

    On maps whose OpenDRIVE GeoReference is absent or mis-configured, CARLA's
    geolocation_to_transform() returns an identity-like result where
    location.x ≈ latitude and location.y ≈ longitude (geographic degrees
    passed through as world metres).  The check below catches this and falls
    through to the linear geo_proj fallback, which is always reliable because
    it is built from the forward-direction transform_to_geolocation samples.
    """
    global _CAM_GEO_PROJ_IDENTITY_WARNED
    alt_use = 0.0
    if alt_m is not None and math.isfinite(alt_m) and abs(alt_m) <= 10_000.0:
        alt_use = float(alt_m)

    if carla_map is not None:
        try:
            geo = carla.GeoLocation(
                latitude=float(lat_deg),
                longitude=float(lon_deg),
                altitude=alt_use,
            )
            tf = carla_map.geolocation_to_transform(geo)
            if tf is not None:
                cx, cy = float(tf.location.x), float(tf.location.y)
                # If the returned world coords are within 1 m of the input
                # lat/lon values the map GeoReference is likely missing and
                # CARLA passed the degrees through as metres.  Fall through to
                # the linear fallback in that case.
                if not (abs(cx - lat_deg) < 1.0 and abs(cy - lon_deg) < 1.0):
                    return cx, cy
                if not _CAM_GEO_PROJ_IDENTITY_WARNED:
                    _CAM_GEO_PROJ_IDENTITY_WARNED = True
                    print(
                        f"[V2X CAM] geolocation_to_transform returned identity-like "
                        f"result ({cx:.4f}, {cy:.4f}) for lat={lat_deg:.4f}, "
                        f"lon={lon_deg:.4f}; map GeoReference may be absent. "
                        f"Falling back to linear geo_proj.",
                        flush=True,
                    )
        except Exception:
            pass

    proj = geo_proj if geo_proj is not None else _CAM_GEO_PROJECTION
    if proj is not None:
        x = proj["ref_x"] + (float(lon_deg) - proj["ref_lon"]) * proj["m_per_deg_lon"]
        y = proj["ref_y"] + (float(lat_deg) - proj["ref_lat"]) * proj["m_per_deg_lat"]
        return x, y
    return None, None


def _cam_latlon_to_carla_xy(lat_raw, lon_raw, ref_pos, carla_map=None, geo_proj=None):
    """
    Convert CAM referencePosition lat/lon to CARLA world XY.

    CARLA may emit either:
    - WGS84 geographic coordinates (standard ETSI, requires map projection), or
    - world X/Y directly in the lat/lon integer fields (legacy CaService encoding).
    """
    lat_dec = _etsi_cam_coord_scaled(lat_raw)
    lon_dec = _etsi_cam_coord_scaled(lon_raw)
    if lat_dec is None or lon_dec is None:
        return None, None

    if _cam_latlon_is_wgs84(lat_dec, lon_dec):
        alt_m = _cam_reference_altitude_m(ref_pos)
        return _cam_wgs84_to_carla_xy(
            lat_dec, lon_dec, alt_m, carla_map, geo_proj=geo_proj,
        )

    # Legacy/native: integer fields carry world metres (values often exceed ±90° as degrees).
    if isinstance(lat_raw, int) and isinstance(lon_raw, int):
        return _decode_carla_cam_world_xy(lat_raw, lon_raw)
    return lat_dec, lon_dec


def _extract_cam_reference_xy(data, carla_map=None, geo_proj=None):
    """Best-effort world XY from a CARLA/ETSI CAM referencePosition block."""
    if not isinstance(data, dict):
        return None, None

    ref_pos = _cam_reference_position_block(_cam_parameters_from_data(data))

    carla_pos = ref_pos.get("carlaWorldPosition") or ref_pos.get("CarlaWorldPosition") or {}
    if isinstance(carla_pos, dict) and carla_pos:
        try:
            return float(carla_pos.get("x")), float(carla_pos.get("y"))
        except (TypeError, ValueError):
            pass

    for lat_key, lon_key in (
        ("Latitude", "Longitude"),
        ("latitude", "longitude"),
    ):
        if lat_key in ref_pos and lon_key in ref_pos:
            x, y = _cam_latlon_to_carla_xy(
                ref_pos.get(lat_key),
                ref_pos.get(lon_key),
                ref_pos,
                carla_map=carla_map,
                geo_proj=geo_proj,
            )
            if x is not None and y is not None:
                return x, y

    for xk, yk in (("x", "y"), ("X", "Y")):
        try:
            x = float(ref_pos.get(xk))
            y = float(ref_pos.get(yk))
            if math.isfinite(x) and math.isfinite(y):
                return x, y
        except (TypeError, ValueError):
            pass

    return None, None


def parse_cam_data(cam_msg, crossing_zone_polygons=None, carla_map=None, geo_proj=None):
    """Extract ego speed and optional distance-to-crossing from a received CAM."""
    try:
        data = cam_msg.get()
    except Exception:
        return None
    speed_ms = parse_cam_speed_ms(cam_msg)
    x, y = _extract_cam_reference_xy(data, carla_map=carla_map, geo_proj=geo_proj)
    dist = None
    if x is not None and y is not None and crossing_zone_polygons:
        dist = min_distance_to_any_crossing_zone(x, y, crossing_zone_polygons)
    return {
        "speed_ms": speed_ms,
        "x": x,
        "y": y,
        "dist_to_crossing_m": dist,
    }
def min_pedestrian_distance_to_crossings(active_walkers, crossing_zone_polygons):
    """Min distance from any tracked actor (e.g. ARI robot vehicle) to the nearest crossing zone edge (m)."""
    best = float("inf")
    for w in active_walkers:
        if not w.is_alive:
            continue
        loc = w.get_transform().location
        d = min_distance_to_any_crossing_zone(loc.x, loc.y, crossing_zone_polygons)
        best = min(best, d)
    return best if math.isfinite(best) else float("inf")


def max_pedestrian_speed_planar_ms(active_walkers):
    """Largest horizontal speed among tracked actors (m/s)."""
    vmax = 0.0
    for w in active_walkers:
        if not w.is_alive:
            continue
        v = w.get_velocity()
        vmax = max(vmax, math.hypot(v.x, v.y))
    return vmax


def any_pedestrian_in_crossing_zone(active_walkers, crossing_zone_polygons):
    for w in active_walkers:
        if not w.is_alive:
            continue
        loc = w.get_transform().location
        if point_in_any_crossing_zone(loc.x, loc.y, crossing_zone_polygons):
            return True
    return False


def crossing_zone_index_for_xy(x, y, crossing_zone_polygons):
    """Index of the crossing polygon containing (x, y), or None."""
    for i, poly in enumerate(crossing_zone_polygons):
        if is_point_in_polygon(x, y, poly):
            return i
    return None


def pedestrian_blocks_ego_path(
    ego_transform,
    ped_x,
    ped_y,
    *,
    half_width_m=None,
    ahead_max_m=None,
    behind_clear_m=None,
):
    """
    True if (ped_x, ped_y) lies in the ego forward driving corridor (CARLA XY).
    Pedestrians clearly behind the ego or far to the side do not block.
    """
    half_width_m = EGO_PATH_HALF_WIDTH_M if half_width_m is None else half_width_m
    ahead_max_m = EGO_PATH_AHEAD_MAX_M if ahead_max_m is None else ahead_max_m
    behind_clear_m = EGO_PASS_BEHIND_M if behind_clear_m is None else behind_clear_m

    loc = ego_transform.location
    fwd = ego_transform.get_forward_vector()
    fx, fy = fwd.x, fwd.y
    fn = math.hypot(fx, fy)
    if fn < 1e-6:
        return False
    fx, fy = fx / fn, fy / fn
    dx = float(ped_x) - loc.x
    dy = float(ped_y) - loc.y
    long_ahead = dx * fx + dy * fy
    lat = abs(dx * (-fy) + dy * fx)
    if long_ahead < -behind_clear_m:
        return False
    if long_ahead > ahead_max_m:
        return False
    if lat > half_width_m:
        return False
    return True


def assess_ego_path_conflict(
    vehicle,
    crossing_zone_polygons,
    *,
    walker_actors,
    vision_ped_xy,
    exclude_actor_ids=None,
):
    """
    True if any human pedestrian (not ARI) blocks the ego path or occupies the
    same crossing strip while in the forward corridor.

    Pedestrians on the other crossing zone (island strip) are ignored so the ego
    can proceed when someone crosses on the far side.
    """
    exclude_actor_ids = set(exclude_actor_ids or ())
    if vehicle is None or not vehicle.is_alive:
        return False

    ego_tf = vehicle.get_transform()
    ego_x = ego_tf.location.x
    ego_y = ego_tf.location.y
    ego_zone = crossing_zone_index_for_xy(ego_x, ego_y, crossing_zone_polygons)

    candidates = []

    for w in walker_actors:
        if not w.is_alive or w.id in exclude_actor_ids:
            continue
        loc = w.get_transform().location
        candidates.append((loc.x, loc.y, "walker", w.id))

    for item in vision_ped_xy:
        px, py = item[0], item[1]
        if px is None or py is None:
            continue
        candidates.append((float(px), float(py), "vision", item[2] if len(item) > 2 else None))

    for px, py, _src, _aid in candidates:
        ped_zone = crossing_zone_index_for_xy(px, py, crossing_zone_polygons)
        if (
            ego_zone is not None
            and ped_zone is not None
            and ped_zone != ego_zone
        ):
            continue
        if pedestrian_blocks_ego_path(ego_tf, px, py):
            return True
    return False
