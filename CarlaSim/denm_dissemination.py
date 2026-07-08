"""DENM transmission policy — hazard events only, separate from ego braking."""
from __future__ import annotations


def should_transmit_denm(
    *,
    hazard_active: bool,
    hazard_activated: bool,
    denm_due: bool,
    hazard_kind_changed: bool,
) -> tuple[bool, str]:
    """
    Return (transmit, denm_kind) where denm_kind is 'new' or 'repetition'.

    DENM(new): first tick hazard becomes active, or hazard kind changes.
    DENM(repetition): hazard still active and send interval elapsed.
    """
    if not hazard_active:
        return False, ""
    if hazard_activated or hazard_kind_changed:
        return True, "new"
    if denm_due:
        return True, "repetition"
    return False, ""


def hazard_kind(*, in_zone_vis: bool) -> str:
    """Stable signature for cause-code selection (zone occupancy vs intent-only)."""
    return "zone" if in_zone_vis else "intent"
