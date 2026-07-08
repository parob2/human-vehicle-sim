"""Ego vehicle low-level navigation (lane follow / pursuit)."""

import math

import carla

from sim_config import (
    EGO_LANE_LOOKAHEAD,
    EGO_SIMPLE_ARRIVE_DIST,
    EGO_SIMPLE_STEER_GAIN,
)

def pursuit_toward_point(vehicle, target, target_speed_kmh):
    """Low-level steer/throttle toward an arbitrary world point (XY geometry)."""
    t = vehicle.get_transform()
    loc = t.location
    v = vehicle.get_velocity()
    speed_ms = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    cap_ms = max(0.1, target_speed_kmh / 3.6)

    dx = target.x - loc.x
    dy = target.y - loc.y
    dist = math.hypot(dx, dy)
    if dist < EGO_SIMPLE_ARRIVE_DIST:
        return carla.VehicleControl(throttle=0.0, brake=0.85, steer=0.0)

    inv = 1.0 / max(dist, 1e-3)
    ux, uy = dx * inv, dy * inv
    fwd = t.get_forward_vector()
    cross = fwd.x * uy - fwd.y * ux
    dot = fwd.x * ux + fwd.y * uy
    # Heading error in radians (not sin(error)); tracks sharp turns without under-steering.
    heading_err = math.atan2(cross, dot)
    steer = max(-1.0, min(1.0, heading_err * EGO_SIMPLE_STEER_GAIN))

    # Gentler acceleration; cap throttle so the ego is less aggressive in CARLA.
    if speed_ms < cap_ms * 0.9:
        th = min(0.42, 0.14 + 0.32 * (cap_ms - speed_ms) / max(cap_ms, 0.1))
    else:
        th = 0.07 if speed_ms < cap_ms * 1.05 else 0.0
    br = 0.22 if speed_ms > cap_ms * 1.15 else 0.0
    return carla.VehicleControl(throttle=th, steer=steer, brake=br)


def simple_pursuit_control(vehicle, destination, target_speed_kmh):
    """Straight-line pursuit to the final destination (ignores lanes)."""
    return pursuit_toward_point(vehicle, destination, target_speed_kmh)


def greedy_lane_follow_control(vehicle, destination, target_speed_kmh):
    """
    Follow drivable lanes toward the goal using waypoint.next() (OpenDRIVE).
    At forks, pick the outgoing waypoint whose XY is closest to the goal waypoint.
    No GlobalRoutePlanner graph — fast on large maps; not identical to Unreal RoutePlanner actors.
    """
    carla_map = vehicle.get_world().get_map()
    loc = vehicle.get_transform().location

    if loc.distance(destination) < EGO_SIMPLE_ARRIVE_DIST:
        return carla.VehicleControl(throttle=0.0, brake=0.85, steer=0.0)

    ego_wp = carla_map.get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_wp is None:
        ego_wp = carla_map.get_waypoint(
            loc, project_to_road=True, lane_type=carla.LaneType.Any
        )

    dest_wp = carla_map.get_waypoint(
        destination,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if dest_wp is None:
        dest_wp = carla_map.get_waypoint(
            destination, project_to_road=True, lane_type=carla.LaneType.Any
        )

    if ego_wp is None or dest_wp is None:
        return simple_pursuit_control(vehicle, destination, target_speed_kmh)

    la = max(2.0, EGO_LANE_LOOKAHEAD)
    nxt = ego_wp.next(la)
    if not nxt:
        nxt = ego_wp.next(la * 0.5)
    if not nxt:
        nxt = ego_wp.next(3.0)
    if not nxt:
        return simple_pursuit_control(vehicle, destination, target_speed_kmh)

    if len(nxt) == 1:
        target_pt = nxt[0].transform.location
    else:
        best = min(
            nxt,
            key=lambda nw: nw.transform.location.distance(dest_wp.transform.location),
        )
        target_pt = best.transform.location

    return pursuit_toward_point(vehicle, target_pt, target_speed_kmh)
