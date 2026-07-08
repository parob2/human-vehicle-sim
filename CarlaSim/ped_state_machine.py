"""
Pedestrian phase: CROSSING (geometry-confirmed zone occupancy) or NOT_CROSSING.

Intent prediction is a direct RF probability threshold comparison — no sustain
counter, no path-conflict gate, no intermediate states.
"""
from __future__ import annotations

from typing import Optional

PHASE_CROSSING = "CROSSING"
PHASE_NOT_CROSSING = "NOT_CROSSING"

# Legacy aliases so any remaining references resolve without error.
PHASE_WAITING = PHASE_NOT_CROSSING
PHASE_PREPARING = PHASE_NOT_CROSSING
PHASE_EXITING = PHASE_NOT_CROSSING

DENM_MEDIUM_INTENT_PROBA = float(__import__("os").environ.get("DENM_MEDIUM_INTENT_PROBA", "0.8"))


def classify_phase(*, is_on_crossing: bool) -> str:
    """Return CROSSING when pedestrian is inside a crossing zone polygon."""
    return PHASE_CROSSING if is_on_crossing else PHASE_NOT_CROSSING


def predict_intent(rf_proba: Optional[float], rf_threshold: float) -> bool:
    """True when RF probability meets the threshold (pedestrian must be off-zone)."""
    if rf_proba is None:
        return False
    return float(rf_proba) >= rf_threshold


def aggregate_ped_phase(phases: dict) -> str:
    """Return CROSSING if any tracked pedestrian is on zone, else NOT_CROSSING."""
    if PHASE_CROSSING in phases.values():
        return PHASE_CROSSING
    return PHASE_NOT_CROSSING


def hazard_event_active(*, intent_confirmed: bool, in_zone_vis: bool) -> bool:
    """
    RSU hazard activation for DENM dissemination (ETSI event trigger).

    Independent of ego distance, TTC, path conflict, and priority level.
    """
    return bool(intent_confirmed or in_zone_vis)


def denm_medium_from_preparing(ped_phase: str, intent_proba: Optional[float], path_conflict: bool) -> bool:
    """Deprecated: use hazard_event_active(); kept for legacy tests."""
    return (
        ped_phase == PHASE_NOT_CROSSING
        and path_conflict
        and intent_proba is not None
        and float(intent_proba) >= DENM_MEDIUM_INTENT_PROBA
    )
