"""Project semantic route objects to bounded map-frame standoff goals."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _obstacle_points(cloud: np.ndarray, robot_z: float) -> np.ndarray:
    if cloud.ndim != 2 or cloud.shape[1] < 3 or not len(cloud):
        return np.zeros((0, 3), dtype=np.float32)
    finite = np.isfinite(cloud[:, :3]).all(axis=1)
    obstacle = cloud[finite]
    # /state_estimation is the lidar sensor pose (about 0.75 m above the
    # floor), not a ground contact pose.  Keep obstacles above the floor while
    # rejecting the dense floor sheet itself.
    obstacle = obstacle[(obstacle[:, 2] >= robot_z - 0.60) & (obstacle[:, 2] <= robot_z + 1.00)]
    return obstacle


def _clearance(point: tuple[float, float], obstacle: np.ndarray) -> float:
    """Return endpoint clearance for ranking, not a navigation hard gate.

    The challenge navigation stack owns path finding and dynamic collision
    avoidance after a waypoint is published.  Requiring a clear straight line
    here incorrectly rejects goals whenever furniture lies between the robot
    and an otherwise reachable observation point.
    """
    if not len(obstacle):
        return 0.0
    squared = (obstacle[:, 0] - point[0]) ** 2 + (obstacle[:, 1] - point[1]) ** 2
    return float(math.sqrt(max(0.0, float(np.min(squared)))))


def _goal_for_object(
    obj: Mapping[str, Any],
    start: Sequence[float],
    cloud: np.ndarray,
    *,
    independent_probe: bool,
    terrain_hull: np.ndarray | None = None,
) -> tuple[float, float, float] | None:
    center = [float(value) for value in obj["center_3d"]]
    bbox = [float(value) for value in obj["bbox_3d"]]
    base_radius = min(2.0, max(0.85, 0.5 * math.hypot(bbox[0], bbox[1]) + 0.65))
    obstacle = _obstacle_points(cloud, float(start[2]))
    preferred = math.atan2(float(start[1]) - center[1], float(start[0]) - center[0])
    offsets = (
        (0.0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2, math.pi)
        if independent_probe
        else (0.0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2, math.pi)
    )
    candidates: list[tuple[float, float, float, float]] = []
    radii = tuple(dict.fromkeys((base_radius, max(0.85, base_radius - 0.25), min(2.0, base_radius + 0.25))))
    for radius in radii:
        for offset_index, offset in enumerate(offsets):
            angle = preferred + offset
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            displacement = math.dist((x, y), (float(start[0]), float(start[1])))
            if independent_probe and displacement < 0.35:
                continue
            # Registered lidar ranks endpoint clearance.  It does not predict
            # the route; ROS is free to go around sofas and other obstacles.
            clearance = min(1.0, _clearance((x, y), obstacle))
            # Bonus for waypoints inside traversable terrain
            terrain_bonus = 0.3 if _point_in_hull((x, y), terrain_hull) else 0.0
            score = clearance + terrain_bonus - 0.08 * offset_index - 0.03 * displacement
            heading = math.atan2(center[1] - y, center[0] - x)
            candidates.append((score, x, y, heading))
    if not candidates:
        return None
    _score, x, y, heading = max(candidates, key=lambda value: value[0])
    return (x, y, heading)


def _terrain_hull(terrain: np.ndarray) -> np.ndarray | None:
    """Compute a 2D convex hull of the traversable terrain for fast containment checks."""
    if len(terrain) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(terrain[:, :2])
        return terrain[:, :2][hull.vertices]
    except Exception:
        return None


def _point_in_hull(point: tuple[float, float], hull_vertices: np.ndarray) -> bool:
    """Check if a 2D point is inside a convex polygon using cross-product signs."""
    if hull_vertices is None or len(hull_vertices) < 3:
        return True  # no hull → assume traversable
    px, py = point
    signs = []
    n = len(hull_vertices)
    for i in range(n):
        x1, y1 = hull_vertices[i]
        x2, y2 = hull_vertices[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        signs.append(cross > 0)
    # All same sign → inside
    return all(signs) or not any(signs)


def plan_route(
    route_objects: Sequence[Mapping[str, Any]],
    *,
    registered_scan_path: str,
    robot_pose_xyz: Sequence[float],
    global_geometry_manifest_path: str | None = None,
    terrain_map_path: str | None = None,
) -> dict[str, Any]:
    path = Path(registered_scan_path)
    cloud = np.load(path) if path.is_file() else np.zeros((0, 3), dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        cloud = np.zeros((0, 3), dtype=np.float32)
    registered_scan_point_count = int(len(cloud))
    # Load terrain map for traversability validation
    terrain_hull = None
    terrain_point_count = 0
    if terrain_map_path:
        tpath = Path(terrain_map_path)
        if tpath.is_file():
            terrain = np.load(tpath)
            if terrain.ndim == 2 and terrain.shape[1] >= 3 and len(terrain) >= 3:
                terrain_point_count = int(len(terrain))
                terrain_hull = _terrain_hull(terrain)
    sga_point_count = 0
    if global_geometry_manifest_path:
        import json

        manifest = json.loads(Path(global_geometry_manifest_path).read_text(encoding="utf-8"))
        sga_clouds = []
        for view in manifest["views"]:
            with np.load(view["pointmap_path"]) as pointmap:
                points = pointmap["points_map"].reshape(-1, 3)
                confidence = pointmap["confidence"].reshape(-1)
            keep = np.isfinite(points).all(axis=1) & np.isfinite(confidence) & (confidence >= 1.5)
            points = points[keep]
            if len(points) > 4000:
                points = points[np.linspace(0, len(points) - 1, 4000, dtype=np.int64)]
            sga_clouds.append(points)
        if sga_clouds:
            global_cloud = np.concatenate(sga_clouds).astype(np.float32)
            sga_point_count = int(len(global_cloud))
            # Dense MASt3R geometry is reported downstream but is not an
            # occupancy grid.  Treating every reconstructed surface as a
            # collision obstacle made target furniture block its own viewing
            # waypoint.
    start = [float(value) for value in robot_pose_xyz[:3]]
    if len(start) != 3 or not all(math.isfinite(value) for value in start):
        return {"status": "blocked", "reason": "robot_pose_invalid", "waypoints": []}
    waypoints = []
    for step in route_objects:
        goal = _goal_for_object(
            step["object"],
            start,
            cloud,
            independent_probe=str(step.get("action", "")) == "probe",
            terrain_hull=terrain_hull,
        )
        if goal is None:
            return {
                "status": "blocked",
                "reason": f"no_collision_free_standoff:{step['order']}",
                "waypoints": waypoints,
            }
        waypoints.append(list(goal))
        start = [goal[0], goal[1], start[2]]
    return {
        "status": "completed",
        "frame": "map",
        "planner": "registered_scan_ranked_standoff_v3",
        "waypoints": waypoints,
        "registered_scan_point_count": registered_scan_point_count,
        "terrain_map_point_count": terrain_point_count,
        "terrain_map_used": terrain_hull is not None,
        "sga_dense_point_count": sga_point_count,
        "global_geometry_manifest_path": global_geometry_manifest_path,
    }
