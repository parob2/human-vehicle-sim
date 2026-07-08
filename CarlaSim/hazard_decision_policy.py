"""
Hazard decision policy (thesis §4.2.4): weighted risk score, priority bands,
TTC-based graded braking (geometry-only), separate DENM hazard trigger.

Pedestrian phases:
    NOT_CROSSING  — off-zone; intent_predicted flag carries RF prediction
    CROSSING      — geometry-confirmed zone occupancy (TTC brake; DENM via hazard_event_active)
"""
import math

from ped_state_machine import (
    PHASE_CROSSING,
    PHASE_NOT_CROSSING,
    hazard_event_active,
)

PHASE_PREPARING = PHASE_NOT_CROSSING

# Internal helpers.

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


# Step 1 — Risk score [0, 1].

def compute_risk_score(ped_state: dict, ego_state: dict) -> float:
    """
    Weighted combination of sub-scores.

    * crossing_score — only when ped_phase == CROSSING (geometry)
    * intent_score   — when RF intent is predicted (off-zone, proba >= threshold)
    """
    d_ego = float(ego_state.get("dist_to_crossing_m", 0.0))
    v_ego = float(ego_state.get("speed_ms", 0.0))
    ped_phase = ped_state.get("ped_phase", PHASE_NOT_CROSSING)
    intent_proba = ped_state.get("intent_proba")
    intent_predicted = bool(ped_state.get("intent_confirmed", False))
    is_on_crossing = bool(ped_state.get("is_on_crossing", False))

    if ped_phase != PHASE_CROSSING and not is_on_crossing and not intent_predicted:
        if intent_proba is None:
            ttc = d_ego / max(v_ego, 0.5)
            ttc_score = _clamp(1.0 - ttc / 6.0, 0.0, 1.0)
            return round(_clamp(0.10 * ttc_score, 0.0, 0.10), 4)

    ttc = d_ego / max(v_ego, 0.5)
    ttc_score = _clamp(1.0 - ttc / 6.0, 0.0, 1.0)

    if ped_phase == PHASE_CROSSING or is_on_crossing:
        crossing_score = 1.0
        intent_score = 0.0
    elif intent_predicted or intent_proba is not None:
        crossing_score = 0.0
        intent_score = float(intent_proba or 0.0)
    else:
        crossing_score = 0.0
        intent_score = 0.0

    proximity_score = _clamp(1.0 - d_ego / 40.0, 0.0, 1.0)
    conf_penalty = 1.0 - _clamp(float(ped_state.get("position_confidence") or 0.5), 0.0, 1.0)

    score = (
        0.35 * ttc_score
        + 0.25 * crossing_score
        + 0.20 * intent_score
        + 0.10 * proximity_score
        + 0.10 * conf_penalty
    )
    return round(_clamp(score, 0.0, 1.0), 4)


# Step 2 — Score → priority level.

_THRESHOLDS = (
    (0.75, "CRITICAL"),
    (0.50, "HIGH"),
    (0.30, "MEDIUM"),
    (0.00, "LOW"),
)


def score_to_priority_level(score: float) -> str:
    """Map a risk score in [0, 1] to a priority level string."""
    for threshold, level in _THRESHOLDS:
        if score >= threshold:
            return level
    return "LOW"


# Step 3 — Priority level → graded action policy.

_BRAKE_LEVELS = {
    "CRITICAL": 1.0,
    "HIGH":     0.6,
    "MEDIUM":   0.2,
    "LOW":      0.0,
}


def priority_level_from_ttc(
    ttc_s: float,
    *,
    ttc_aeb_s: float = 1.5,
    ttc_high_s: float = 2.5,
    ttc_t0_s: float = 4.0,
) -> str:
    """AEB VRU TTC bands → priority level."""
    t = float(ttc_s)
    if t <= ttc_aeb_s:
        return "CRITICAL"
    if t <= ttc_high_s:
        return "HIGH"
    if t <= ttc_t0_s:
        return "MEDIUM"
    return "LOW"


def ego_brake_level_from_ttc(
    ttc_s: float,
    *,
    ttc_aeb_s: float = 1.5,
    ttc_high_s: float = 2.5,
    ttc_t0_s: float = 4.0,
) -> float:
    """Graded brake [0, 1] from TTC only (AEB VRU phases)."""
    return _BRAKE_LEVELS[priority_level_from_ttc(
        ttc_s, ttc_aeb_s=ttc_aeb_s, ttc_high_s=ttc_high_s, ttc_t0_s=ttc_t0_s,
    )]


def map_priority_to_actions(priority_level: str, *, vehicle_threat: bool) -> dict:
    brake = _BRAKE_LEVELS.get(priority_level, 0.0)
    hold = vehicle_threat

    actors = []
    if brake > 0.0:
        actors.append("ego_vehicle")
    if hold:
        actors.append("robot")

    rationale_map = {
        "CRITICAL": "critical_risk_emergency_brake",
        "HIGH":     "high_risk_strong_deceleration",
        "MEDIUM":   "medium_risk_light_deceleration",
        "LOW":      "low_risk_normal_operation",
    }

    return {
        "ego_brake_level":      brake,
        "suggested_hold_robot": hold,
        "affected_actors":      actors,
        "rationale":            rationale_map.get(priority_level, "unknown"),
    }


def compute_hazard_decision(ped_state: dict, ego_state: dict, map_context: dict) -> dict:
    """Single public entry point: build a complete hazard_decision dict."""
    d_ego = float(ego_state.get("dist_to_crossing_m", 0.0))
    v_ego = float(ego_state.get("speed_ms", 0.0))
    d_ped = float(ped_state.get("dist_to_crossing_m", 0.0))
    ped_phase = ped_state.get("ped_phase", PHASE_NOT_CROSSING)
    ped_in_zone = bool(ped_state.get("is_on_crossing", False))
    path_conflict = bool(ped_state.get("path_conflict", True))
    intent_proba = ped_state.get("intent_proba")

    unsafe_ttc_s = float(map_context.get("unsafe_ttc_s", 1.5))
    min_threat_spd = float(map_context.get("min_threat_speed_ms", 1.5))
    master_range_m = float(map_context.get("master_range_m", 50.0))
    ttc_aeb_s = float(map_context.get("ttc_aeb_s", 1.5))
    ttc_high_s = float(map_context.get("ttc_high_s", 2.5))
    ttc_t0_s = float(map_context.get("ttc_t0_s", 4.0))

    if v_ego < 0.5:
        ttc = float("inf")
    else:
        ttc = d_ego / v_ego

    intent_confirmed = bool(ped_state.get("intent_confirmed", False))
    in_master_range = d_ego <= master_range_m

    vehicle_threat = bool(
        intent_confirmed
        and path_conflict
        and in_master_range
        and v_ego >= min_threat_spd
        and math.isfinite(ttc)
        and ttc <= unsafe_ttc_s
    )

    risk_score = compute_risk_score(ped_state, ego_state)
    signal_priority = score_to_priority_level(risk_score)

    # Ego TTC brake only on confirmed zone occupancy (geometry), not RF alone.
    crossing_confirmed = ped_phase == PHASE_CROSSING or ped_in_zone
    if crossing_confirmed and in_master_range and path_conflict and math.isfinite(ttc):
        brake_priority = priority_level_from_ttc(
            ttc, ttc_aeb_s=ttc_aeb_s, ttc_high_s=ttc_high_s, ttc_t0_s=ttc_t0_s,
        )
        ego_brake = ego_brake_level_from_ttc(
            ttc, ttc_aeb_s=ttc_aeb_s, ttc_high_s=ttc_high_s, ttc_t0_s=ttc_t0_s,
        )
    else:
        brake_priority = "LOW"
        ego_brake = 0.0

    priority_level = brake_priority if ego_brake > 0.0 else signal_priority

    actions = map_priority_to_actions(priority_level, vehicle_threat=vehicle_threat)

    # DENM dissemination: hazard events only (no ego range / TTC / path-conflict gating).
    hazard_active = hazard_event_active(
        intent_confirmed=intent_confirmed,
        in_zone_vis=ped_in_zone,
    )

    actions["ego_brake_level"] = ego_brake

    return {
        "risk_score":                     risk_score,
        "priority_level":                 priority_level,
        "signal_priority":                signal_priority,
        "brake_priority":                 brake_priority,
        "ego_brake_level":                ego_brake,
        "suggested_hold_robot":           actions["suggested_hold_robot"],
        "affected_actors":                actions["affected_actors"],
        "hazard_active":                  hazard_active,
        "send_denm":                      hazard_active,
        "hazard_in_zone":                 ped_in_zone,
        "vehicle_threat":                 vehicle_threat,
        "rationale":                      actions["rationale"],
        "ttc_s":                          round(ttc, 2) if math.isfinite(ttc) else None,
        "distanceEgoToCrossing_m":        round(d_ego, 2),
        "distancePedestrianToCrossing_m": round(d_ped, 2),
        "vEgo_ms":                        round(v_ego, 2),
        "pedestrianInCrossingZone":       ped_in_zone,
        "ped_phase":                      ped_phase,
        "path_conflict":                  path_conflict,
    }
