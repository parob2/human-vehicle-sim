"""CARLA pedestrian blueprint selection for scenario spawns."""

from sim_config import WALKER_BP_DEFAULT, _WALKER_BP_VARIANT_RAW

def _walker_bp_variant_ids():
    ids = [v.strip() for v in _WALKER_BP_VARIANT_RAW.split(",") if v.strip()]
    default_suffix = WALKER_BP_DEFAULT.rsplit(".", 1)[-1]
    if default_suffix not in ids:
        ids.insert(0, default_suffix)
    elif ids[0] != default_suffix:
        ids = [default_suffix] + [i for i in ids if i != default_suffix]
    return ids


def pedestrian_blueprints(bp_library, count: int):
    """Return up to `count` distinct walker.pedestrian.* blueprints (0004 first by default)."""
    available = {bp.id: bp for bp in (bp_library.filter("walker.pedestrian.*") or [])}
    out = []
    seen = set()
    for suffix in _walker_bp_variant_ids():
        bp_id = f"walker.pedestrian.{suffix}"
        bp = available.get(bp_id) or bp_library.find(bp_id)
        if bp is None or bp.id in seen:
            continue
        out.append(bp)
        seen.add(bp.id)
        if len(out) >= count:
            return out
    for bp_id in sorted(available):
        if len(out) >= count:
            break
        if bp_id not in seen:
            out.append(available[bp_id])
            seen.add(bp_id)
    if not out:
        fallback = bp_library.find(WALKER_BP_DEFAULT)
        if fallback is not None:
            out = [fallback]
