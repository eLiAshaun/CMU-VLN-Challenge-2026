"""Sparse semantic waypoint selection.

This module does not implement a waypoint converter or a global planner. It
keeps semantic target coordinates intact, uses terrain only as a soft feature,
and emits one point for a goal or two points for an ordered corridor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from integrations.execution.trajectory_geometry import (
    compile_directive_geometry,
    point_in_region,
    segment_intersects_region,
)


def _voxel_downsample_xyz(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)[:, :3]
    if not math.isfinite(float(voxel_size_m)) or float(voxel_size_m) <= 0.0:
        return points
    keys = np.floor(points / float(voxel_size_m)).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    np.add.at(sums, inverse, points)
    return sums / counts[:, None]


def _load_official_terrain(
    *,
    terrain_paths: list[str],
    obstacle_height_threshold: float,
    obstacle_clearance_m: float,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, object]] | None:
    """Load official terrain features without selecting a replacement goal."""
    terrain = None
    source_path = ""
    for raw_path in terrain_paths:
        path = Path(str(raw_path))
        if not str(raw_path) or not path.is_file():
            continue
        try:
            loaded = np.load(path)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                try:
                    points = np.asarray(loaded["points"])
                finally:
                    loaded.close()
            else:
                points = np.asarray(loaded)
        except (OSError, ValueError, KeyError):
            continue
        if points.ndim == 2 and points.shape[1] >= 4 and len(points):
            terrain = points
            source_path = str(path)
            break
    if terrain is None:
        return None
    finite = np.isfinite(terrain[:, :4]).all(axis=1)
    traversable_raw = terrain[finite & (terrain[:, 3] < float(obstacle_height_threshold)), :3]
    obstacle_raw = terrain[finite & (terrain[:, 3] >= float(obstacle_height_threshold)), :3]
    traversable = _voxel_downsample_xyz(traversable_raw, voxel_size_m)
    obstacles = _voxel_downsample_xyz(obstacle_raw, voxel_size_m)
    safe = traversable
    clearance = float(obstacle_clearance_m)
    if len(obstacles) and clearance > 0.0 and len(traversable):
        safe_mask = np.ones(len(traversable), dtype=bool)
        for index, point in enumerate(traversable):
            distance = np.hypot(obstacles[:, 0] - point[0], obstacles[:, 1] - point[1])
            safe_mask[index] = not bool(np.any(distance < clearance))
        safe = traversable[safe_mask]
    diagnostics = {
        "terrain_source_path": source_path,
        "terrain_source_authority": "official_terrain_map",
        "obstacle_height_threshold_m": float(obstacle_height_threshold),
        "obstacle_clearance_m": clearance,
        "terrain_voxel_size_m": float(voxel_size_m),
        "raw_point_count": int(len(terrain)),
        "raw_traversable_point_count": int(len(traversable_raw)),
        "raw_obstacle_point_count": int(len(obstacle_raw)),
        "voxel_traversable_point_count": int(len(traversable)),
        "voxel_obstacle_point_count": int(len(obstacles)),
        "safe_traversable_point_count": int(len(safe)),
    }
    return safe, obstacles, source_path, diagnostics


def _current_map_pose(path_value: object) -> tuple[float, float, float] | None:
    path = Path(str(path_value or ""))
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        position = state.get("position_xyz")
        orientation = state.get("orientation_xyzw")
        if not isinstance(position, Sequence) or len(position) < 2:
            return None
        if not isinstance(orientation, Sequence) or len(orientation) < 4:
            return (float(position[0]), float(position[1]), 0.0)
        x, y, z, w = (float(value) for value in orientation[:4])
        heading = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        result = (float(position[0]), float(position[1]), heading)
        return result if all(math.isfinite(value) for value in result) else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _target_xy(directive: Mapping[str, Any]) -> tuple[float, float]:
    raw = directive.get("semantic_target_xy", directive.get("navigation_target_xy"))
    obj = directive.get("object")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 2:
        raw = obj.get("center_3d") if isinstance(obj, Mapping) else None
    if not isinstance(raw, Sequence) or len(raw) < 2:
        raise ValueError("waypoint_coordinate_missing")
    result = (float(raw[0]), float(raw[1]))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("waypoint_coordinate_nonfinite")
    return result


def _region_samples(region: Mapping[str, Any]) -> list[tuple[float, float]]:
    kind = str(region.get("kind", ""))
    if kind == "approach_band" or kind == "pass_corridor":
        polygon = region.get("footprint_polygon_xy", ())
        center = region.get("center_xy", (0.0, 0.0))
        minimum = float(region.get("surface_distance_min_m", 0.8))
        maximum = float(region.get("surface_distance_max_m", minimum))
        if isinstance(polygon, Sequence) and len(polygon) >= 3 and isinstance(center, Sequence):
            distances = tuple(dict.fromkeys((
                min(maximum, minimum + 0.05),
                0.5 * (minimum + maximum),
            )))
            result: list[tuple[float, float]] = []
            for index, first in enumerate(polygon):
                second = polygon[(index + 1) % len(polygon)]
                edge_center = (
                    0.5 * (float(first[0]) + float(second[0])),
                    0.5 * (float(first[1]) + float(second[1])),
                )
                dx = edge_center[0] - float(center[0])
                dy = edge_center[1] - float(center[1])
                norm = math.hypot(dx, dy) or 1.0
                for fraction in (0.2, 0.5, 0.8):
                    boundary = (
                        float(first[0]) + fraction * (float(second[0]) - float(first[0])),
                        float(first[1]) + fraction * (float(second[1]) - float(first[1])),
                    )
                    for distance in distances:
                        candidate = (
                            boundary[0] + distance * dx / norm,
                            boundary[1] + distance * dy / norm,
                        )
                        if point_in_region(candidate, region):
                            result.append(candidate)
            return list(dict.fromkeys(
                (round(value[0], 6), round(value[1], 6)) for value in result
            ))
    polygon = region.get("polygon_xy")
    if isinstance(polygon, Sequence) and len(polygon) >= 3:
        return [
            (float(point[0]), float(point[1])) for point in polygon
        ] + [tuple(float(value) for value in region.get("center_xy", (0.0, 0.0))[:2])]
    center = region.get("center_xy", region.get("semantic_target_xy", (0.0, 0.0)))
    if not isinstance(center, Sequence) or len(center) < 2:
        return []
    return [(float(center[0]), float(center[1]))]


def _distance_to_region(point: Sequence[float], region: Mapping[str, Any], semantic_target: Sequence[float]) -> float:
    if point_in_region(point, region):
        return 0.0
    return min(
        math.dist(point[:2], semantic_target[:2]),
        math.dist(point[:2], region.get("center_xy", semantic_target)[:2]),
    )


def _nearest_terrain_distance(point: Sequence[float], terrain: np.ndarray) -> float:
    if not len(terrain):
        return 1.0
    return float(np.min(np.hypot(terrain[:, 0] - float(point[0]), terrain[:, 1] - float(point[1]))))


def _uncertainty(directive: Mapping[str, Any]) -> float:
    obj = directive.get("object", {})
    covariance = obj.get("center_cov") if isinstance(obj, Mapping) else None
    try:
        value = math.sqrt(max(0.0, float(covariance[0][0]) + float(covariance[1][1])))
    except (TypeError, ValueError, IndexError):
        value = 0.25
    return max(0.0, min(1.0, value / 1.5))


def _alignment_cost(start: Sequence[float], candidate: Sequence[float], next_target: Sequence[float] | None) -> float:
    if next_target is None:
        return 0.5
    first = (float(candidate[0]) - float(start[0]), float(candidate[1]) - float(start[1]))
    second = (float(next_target[0]) - float(candidate[0]), float(next_target[1]) - float(candidate[1]))
    first_norm, second_norm = math.hypot(*first), math.hypot(*second)
    if first_norm <= 1e-6 or second_norm <= 1e-6:
        return 0.5
    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_norm * second_norm)
    return 0.5 * (1.0 - max(-1.0, min(1.0, cosine)))


def _approach_direction_cost(
    start: Sequence[float],
    candidate: Sequence[float],
    center: Sequence[float],
) -> float:
    """Prefer the visible, start-facing side of an approach band.

    The square-root chord cost gives useful separation between a radial
    approach and a tangential point on the same band.  This is a local
    semantic-side preference; FAR remains responsible for global routing.
    """
    outward = (
        float(candidate[0]) - float(center[0]),
        float(candidate[1]) - float(center[1]),
    )
    visible_side = (
        float(start[0]) - float(center[0]),
        float(start[1]) - float(center[1]),
    )
    route = (
        float(candidate[0]) - float(start[0]),
        float(candidate[1]) - float(start[1]),
    )
    target_route = (
        float(center[0]) - float(start[0]),
        float(center[1]) - float(start[1]),
    )

    def chord(first: Sequence[float], second: Sequence[float]) -> float:
        first_norm = math.hypot(float(first[0]), float(first[1]))
        second_norm = math.hypot(float(second[0]), float(second[1]))
        if first_norm <= 1e-6 or second_norm <= 1e-6:
            return 0.5
        cosine = (
            float(first[0]) * float(second[0])
            + float(first[1]) * float(second[1])
        ) / (first_norm * second_norm)
        return math.sqrt(
            0.5 * (1.0 - max(-1.0, min(1.0, cosine)))
        )

    return max(chord(outward, visible_side), chord(route, target_route))


def _probe_focus_direction_cost(
    candidate: Sequence[float],
    center: Sequence[float],
    focus: Sequence[float],
) -> float:
    """Continuous cost for separating subject and anchor in the next view.

    Moving directly toward a subject hidden behind its relation anchor keeps
    both on nearly the same image ray and can repeat the original occlusion.
    The chord between the two candidate-to-object unit rays is a bounded
    parallax signal: zero separation costs one, while wider angular separation
    continuously lowers the cost.  Terrain and travel terms still choose the
    feasible member of the semantic goal set.
    """
    anchor_ray = (
        float(center[0]) - float(candidate[0]),
        float(center[1]) - float(candidate[1]),
    )
    subject_ray = (
        float(focus[0]) - float(candidate[0]),
        float(focus[1]) - float(candidate[1]),
    )
    anchor_norm = math.hypot(*anchor_ray)
    subject_norm = math.hypot(*subject_ray)
    if anchor_norm <= 1e-6 or subject_norm <= 1e-6:
        return 0.5
    cosine = (
        anchor_ray[0] * subject_ray[0] + anchor_ray[1] * subject_ray[1]
    ) / (anchor_norm * subject_norm)
    angular_chord = math.sqrt(
        0.5 * (1.0 - max(-1.0, min(1.0, cosine)))
    )
    return 1.0 - angular_chord


def _forbidden(candidate: Sequence[float], start: Sequence[float], regions: Sequence[Mapping[str, Any]]) -> bool:
    for region in regions:
        if region.get("forbidden") is True and (
            point_in_region(candidate, region) or segment_intersects_region(start, candidate, region)
        ):
            return True
    return False


def _failed_waypoint_points(
    navigation_history: Sequence[Mapping[str, Any]],
    *,
    step_index: int,
    semantic_object_id: object,
    radius_m: float,
) -> list[tuple[float, float]]:
    """Return physical points that failed to advance this semantic step.

    These are feedback from real navigation, not an attempt budget.  A point
    is avoided when the same semantic step either reached it without semantic
    progress or failed while targeting it.  Failed actual poses are not
    excluded: they are the next segment start, while the requested/physical
    endpoints are the unusable evidence.  The next selection remains inside
    the current semantic goal set and can use another legal side or baseline.
    """
    result: list[tuple[float, float]] = []
    for attempt in navigation_history:
        if not isinstance(attempt, Mapping):
            continue
        raw_step = attempt.get("step_index")
        try:
            if int(raw_step) != int(step_index):
                continue
        except (TypeError, ValueError):
            continue
        status = str(attempt.get("status", ""))
        arrived_without_progress = bool(
            attempt.get("semantic_progressed") is False
            and status in {
                "arrived",
                "local_waypoint_arrived",
            }
        )
        navigation_failed = status == "failed"
        if not arrived_without_progress and not navigation_failed:
            continue
        pose_keys = [
            "requested_waypoint_pose",
            "physical_waypoint_pose",
        ]
        if arrived_without_progress:
            pose_keys.append("actual_arrival_pose")
        attempted_object_id = attempt.get("semantic_object_id")
        if semantic_object_id is not None and attempted_object_id is not None:
            try:
                if int(attempted_object_id) != int(semantic_object_id):
                    continue
            except (TypeError, ValueError):
                if str(attempted_object_id) != str(semantic_object_id):
                    continue
        for key in pose_keys:
            pose = attempt.get(key)
            if not isinstance(pose, Sequence) or isinstance(pose, (str, bytes)):
                continue
            if len(pose) < 2:
                continue
            try:
                point = (float(pose[0]), float(pose[1]))
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in point):
                continue
            if not any(math.dist(point, existing) <= radius_m for existing in result):
                result.append(point)
    return result


def _select_semantic_candidate(
    directive: Mapping[str, Any],
    region: Mapping[str, Any],
    start_xy: Sequence[float],
    next_target: Sequence[float] | None,
    terrain: np.ndarray,
    semantic_candidates: Sequence[Sequence[float]],
    forbidden_regions: Sequence[Mapping[str, Any]],
    navigation_config: Mapping[str, Any],
    failed_waypoint_points: Sequence[Sequence[float]] = (),
) -> tuple[float, float] | None:
    target = _target_xy(directive)
    roi = max(1.0, float(navigation_config.get("semantic_goal_roi_m", 2.5)))
    action = str(directive.get("action", region.get("action", "go_to"))).lower()
    explicit_navigation = directive.get("navigation_target_xy")
    if (
        action in {"probe", "explore"}
        and isinstance(explicit_navigation, Sequence)
        and not isinstance(explicit_navigation, (str, bytes))
        and len(explicit_navigation) >= 2
    ):
        try:
            motion_target = (
                float(explicit_navigation[0]),
                float(explicit_navigation[1]),
            )
        except (TypeError, ValueError):
            motion_target = target
        if not all(math.isfinite(value) for value in motion_target):
            motion_target = target
    else:
        motion_target = target
    probe_focus_active = False
    probe_object = directive.get("object", {})
    probe_subject_xy = (
        probe_object.get("probe_relation_subject_xy")
        if isinstance(probe_object, Mapping) else None
    )
    if (
        action in {"probe", "explore"}
        and isinstance(probe_subject_xy, Sequence)
        and not isinstance(probe_subject_xy, (str, bytes))
        and len(probe_subject_xy) >= 2
    ):
        try:
            proposed_motion_target = (
                float(probe_subject_xy[0]),
                float(probe_subject_xy[1]),
            )
        except (TypeError, ValueError):
            proposed_motion_target = target
        if all(math.isfinite(value) for value in proposed_motion_target):
            # The subject is a visibility focus, not the navigation target.
            # Keep the explicit evidence viewpoint as motion authority.
            probe_subject_xy = proposed_motion_target
            probe_focus_active = True
    evidence_viewpoints: list[tuple[float, float]] = []
    raw_evidence_viewpoints = (
        probe_object.get("probe_viewpoint_candidates_xy", ())
        if isinstance(probe_object, Mapping) else ()
    )
    for value in raw_evidence_viewpoints:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) < 2
        ):
            continue
        try:
            candidate_viewpoint = (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(component) for component in candidate_viewpoint):
            evidence_viewpoints.append(candidate_viewpoint)
    candidates: list[tuple[float, float]] = [
        (float(value[0]), float(value[1])) for value in semantic_candidates
    ]
    candidates.extend(evidence_viewpoints)
    target_distance = math.dist(start_xy[:2], motion_target[:2])
    if action in {"probe", "explore"} and target_distance > 1e-6:
        # A probe is an epistemic motion toward the uncertain relation, not a
        # premature demand to reach the object's semantic completion band.
        # Keep each observation leg local; FAR still owns the route.
        probe_travel = min(roi, target_distance)
        candidates.append((
            float(start_xy[0])
            + probe_travel * (float(motion_target[0]) - float(start_xy[0]))
            / target_distance,
            float(start_xy[1])
            + probe_travel * (float(motion_target[1]) - float(start_xy[1]))
            / target_distance,
        ))
    if len(terrain):
        terrain_loci = evidence_viewpoints or [motion_target]
        terrain_radius = roi + float(region.get("surface_distance_max_m", 0.0))
        for point in terrain:
            if any(
                math.dist(point[:2], locus[:2]) <= terrain_radius
                for locus in terrain_loci
            ):
                candidates.append((float(point[0]), float(point[1])))
    unique = list(dict.fromkeys((round(x, 5), round(y, 5)) for x, y in candidates))
    if not unique:
        return None
    travel_scale = max(roi, math.dist(start_xy[:2], target[:2]), 1.0)
    terrain_weight = max(0.0, float(navigation_config.get("terrain_distance_weight", 0.20)))
    alignment_weight = max(0.0, float(navigation_config.get("next_step_alignment_weight", 0.15)))
    uncertainty_weight = max(0.0, float(navigation_config.get("geometry_uncertainty_weight", 0.10)))
    semantic_weight = max(0.0, 1.0 - terrain_weight - alignment_weight - uncertainty_weight - 0.15)
    scored = []
    requires_region_membership = (
        str(region.get("kind", "")) == "approach_band"
        and action not in {"probe", "explore"}
    )
    acceptance_radius = max(
        0.0,
        float(navigation_config.get("terrain_waypoint_acceptance_radius_m", 0.30)),
    )
    independent_separation = max(
        0.0,
        float(
            navigation_config.get(
                "minimum_independent_viewpoint_separation_m",
                acceptance_radius,
            )
        ),
    )
    failure_exclusion_radius = max(0.05, acceptance_radius, independent_separation)
    # Arrival may be accepted this far before the requested point.  A probe
    # therefore needs both distances, plus a numerical margin, between start
    # and goal for an independent optical center to be physically achievable.
    minimum_probe_goal_distance = (
        acceptance_radius + independent_separation + 0.02
    )
    for candidate in unique:
        if _forbidden(candidate, start_xy, forbidden_regions):
            continue
        if any(
            math.dist(candidate[:2], point[:2]) <= failure_exclusion_radius
            for point in failed_waypoint_points
            if isinstance(point, Sequence)
            and not isinstance(point, (str, bytes))
            and len(point) >= 2
        ):
            continue
        if requires_region_membership and not point_in_region(candidate, region):
            continue
        candidate_travel = math.dist(start_xy[:2], candidate)
        if (
            action in {"probe", "explore"}
            and candidate_travel < minimum_probe_goal_distance
        ):
            continue
        probe_footprints = [
            region.get("semantic_footprint_polygon_xy", ()),
            *(
                probe_object.get("probe_participant_footprints_xy", ())
                if isinstance(probe_object, Mapping) else ()
            ),
        ]
        if action in {"probe", "explore"} and any(
            isinstance(footprint, Sequence)
            and not isinstance(footprint, (str, bytes))
            and len(footprint) >= 3
            and point_in_region(
                candidate,
                {"shape": "polygon", "polygon_xy": footprint},
            )
            for footprint in probe_footprints
        ):
            continue
        semantic = (
            min(
                1.0,
                min(
                    math.dist(candidate, locus)
                    for locus in (evidence_viewpoints or [motion_target])
                ) / roi,
            )
            if action in {"probe", "explore"}
            else min(1.0, _distance_to_region(candidate, region, target) / roi)
        )
        terrain_cost = min(1.0, _nearest_terrain_distance(candidate, terrain) / roi)
        travel = min(1.0, candidate_travel / travel_scale)
        probe_anchor_locus = (
            probe_object.get("probe_relation_anchor_locus_xy")
            if isinstance(probe_object, Mapping) else None
        )
        alignment = (
            0.70 * _probe_focus_direction_cost(
                candidate,
                probe_anchor_locus
                if isinstance(probe_anchor_locus, Sequence)
                and len(probe_anchor_locus) >= 2
                else region.get("center_xy", target),
                probe_subject_xy,
            )
            + 0.30 * _alignment_cost(start_xy, candidate, next_target)
            if probe_focus_active
            else _approach_direction_cost(
                start_xy,
                candidate,
                region.get("center_xy", target),
            )
            if next_target is None
            and str(region.get("kind", "")) == "approach_band"
            else _alignment_cost(start_xy, candidate, next_target)
        )
        score = (
            semantic_weight * semantic
            + terrain_weight * terrain_cost
            + 0.15 * travel
            + alignment_weight * alignment
            + uncertainty_weight * _uncertainty(directive)
        )
        scored.append((score, math.dist(candidate, motion_target), candidate))
    if not scored:
        return None
    scored.sort(key=lambda value: (value[0], value[1], value[2]))
    return scored[0][2]


def _corridor_waypoints(
    directive: Mapping[str, Any],
    region: Mapping[str, Any],
    start_xy: Sequence[float],
    next_target: Sequence[float] | None,
    terrain: np.ndarray,
    forbidden_regions: Sequence[Mapping[str, Any]],
) -> list[tuple[float, float]]:
    kind = str(region.get("kind", ""))
    center = region.get("center_xy", _target_xy(directive))
    if kind == "between_corridor":
        polygon = region.get("polygon_xy", ())
        if isinstance(polygon, Sequence) and len(polygon) >= 4:
            ordered = [(float(point[0]), float(point[1])) for point in polygon]
            first = ((ordered[0][0] + ordered[3][0]) * 0.5, (ordered[0][1] + ordered[3][1]) * 0.5)
            second = ((ordered[1][0] + ordered[2][0]) * 0.5, (ordered[1][1] + ordered[2][1]) * 0.5)
            if math.dist(start_xy[:2], second) < math.dist(start_xy[:2], first):
                first, second = second, first
            if (
                point_in_region(first, region)
                and point_in_region(second, region)
                and segment_intersects_region(first, second, region)
                and not _forbidden(first, start_xy, forbidden_regions)
                and not _forbidden(second, first, forbidden_regions)
            ):
                return [first, second]
        return []
    polygon = region.get("footprint_polygon_xy", ())
    if (
        kind != "pass_corridor"
        or not isinstance(center, Sequence)
        or len(center) < 2
        or not isinstance(polygon, Sequence)
        or len(polygon) < 3
    ):
        return []
    candidates = list(_region_samples(region))
    candidates.extend(
        (float(point[0]), float(point[1]))
        for point in terrain
        if point_in_region(point[:2], region)
    )
    candidates = list(dict.fromkeys(
        (round(value[0], 5), round(value[1], 5)) for value in candidates
    ))
    candidates = [
        value for value in candidates
        if point_in_region(value, region)
        and not _forbidden(value, start_xy, forbidden_regions)
    ]
    footprint_region = {"shape": "polygon", "polygon_xy": polygon}
    start_vector = (
        float(start_xy[0]) - float(center[0]),
        float(start_xy[1]) - float(center[1]),
    )
    start_norm = math.hypot(*start_vector) or 1.0
    scored: list[tuple[float, tuple[float, float], tuple[float, float]]] = []
    for index, raw_first in enumerate(candidates):
        for raw_second in candidates[index + 1:]:
            separation = math.dist(raw_first, raw_second)
            if not 0.55 <= separation <= 2.50:
                continue
            first_offset = (
                raw_first[0] - float(center[0]),
                raw_first[1] - float(center[1]),
            )
            second_offset = (
                raw_second[0] - float(center[0]),
                raw_second[1] - float(center[1]),
            )
            if first_offset[0] * second_offset[0] + first_offset[1] * second_offset[1] <= 0.0:
                continue
            if segment_intersects_region(raw_first, raw_second, footprint_region):
                continue
            first, second = (
                (raw_first, raw_second)
                if math.dist(start_xy[:2], raw_first) <= math.dist(start_xy[:2], raw_second)
                else (raw_second, raw_first)
            )
            if _forbidden(second, first, forbidden_regions):
                continue
            midpoint_vector = (
                0.5 * (first[0] + second[0]) - float(center[0]),
                0.5 * (first[1] + second[1]) - float(center[1]),
            )
            midpoint_norm = math.hypot(*midpoint_vector) or 1.0
            accessible_side = (
                midpoint_vector[0] * start_vector[0]
                + midpoint_vector[1] * start_vector[1]
            ) / (midpoint_norm * start_norm)
            continuation = (
                math.dist(second, next_target[:2])
                if next_target is not None else 0.25 * math.dist(start_xy[:2], second)
            )
            score = (
                math.dist(start_xy[:2], first)
                + 0.35 * separation
                + 0.65 * continuation
                + 0.75 * (1.0 - accessible_side)
                + 0.10 * (
                    _nearest_terrain_distance(first, terrain)
                    + _nearest_terrain_distance(second, terrain)
                )
            )
            scored.append((score, first, second))
    if not scored:
        return []
    _score, first, second = min(scored, key=lambda value: (value[0], value[1], value[2]))
    return [first, second]


def _avoid_detour_waypoints(
    region: Mapping[str, Any],
    start_xy: Sequence[float],
    next_target: Sequence[float] | None,
    terrain: np.ndarray,
    forbidden_regions: Sequence[Mapping[str, Any]],
    navigation_config: Mapping[str, Any],
) -> list[tuple[float, float]]:
    """Choose one small left/right detour around an inflated forbidden polygon."""
    polygon = region.get("polygon_xy", ())
    if (
        not isinstance(polygon, Sequence)
        or isinstance(polygon, (str, bytes))
        or len(polygon) < 3
        or next_target is None
    ):
        return []
    try:
        vertices = [
            (float(value[0]), float(value[1]))
            for value in polygon
            if isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) >= 2
        ]
        destination = (float(next_target[0]), float(next_target[1]))
    except (TypeError, ValueError):
        return []
    if len(vertices) < 3 or not all(
        math.isfinite(value)
        for point in (*vertices, destination)
        for value in point
    ):
        return []
    center_raw = region.get("center_xy", ())
    try:
        center = (
            float(center_raw[0]),
            float(center_raw[1]),
        )
    except (IndexError, TypeError, ValueError):
        center = (
            sum(value[0] for value in vertices) / len(vertices),
            sum(value[1] for value in vertices) / len(vertices),
        )
    route = (
        destination[0] - float(start_xy[0]),
        destination[1] - float(start_xy[1]),
    )
    route_norm = math.hypot(*route)
    if route_norm <= 1e-6:
        return []
    route_unit = (route[0] / route_norm, route[1] / route_norm)
    normal = (-route_unit[1], route_unit[0])
    outward_margin = max(
        0.08,
        min(
            0.25,
            0.35 * float(
                navigation_config.get(
                    "terrain_waypoint_acceptance_radius_m", 0.30
                )
            ),
        ),
    )
    active_forbidden = dict(region)
    active_forbidden["forbidden"] = True
    blocked_regions = [
        active_forbidden,
        *[
            value for value in forbidden_regions
            if isinstance(value, Mapping) and value is not region
        ],
    ]
    path_candidates: list[
        tuple[float, int, list[tuple[float, float]]]
    ] = []
    for side_order, side in enumerate((1.0, -1.0)):
        lateral = [
            (
                side * (
                    (value[0] - center[0]) * normal[0]
                    + (value[1] - center[1]) * normal[1]
                ),
                value,
            )
            for value in vertices
        ]
        maximum_lateral = max(value[0] for value in lateral)
        side_vertices = [
            value for score, value in lateral
            if score >= maximum_lateral - 1e-6
        ]
        if not side_vertices:
            continue
        side_vertices.sort(key=lambda value: (
            (value[0] - float(start_xy[0])) * route_unit[0]
            + (value[1] - float(start_xy[1])) * route_unit[1]
        ))
        raw_path = [side_vertices[0]]
        if math.dist(side_vertices[0], side_vertices[-1]) > 0.10:
            raw_path.append(side_vertices[-1])
        path = [
            (
                value[0] + side * normal[0] * outward_margin,
                value[1] + side * normal[1] * outward_margin,
            )
            for value in raw_path
        ]
        prior = (float(start_xy[0]), float(start_xy[1]))
        valid = True
        for index, point in enumerate([*path, destination]):
            # This is a local semantic detour, not a global route.  Its own
            # waypoints must avoid every known region.  The hypothetical tail
            # to the semantic target only proves that this side clears the
            # active polygon; a later obstacle is handled after the next real
            # arrival/reacquisition transaction.
            checked_regions = (
                [active_forbidden]
                if index == len(path)
                else blocked_regions
            )
            if _forbidden(point, prior, checked_regions):
                valid = False
                break
            prior = point
        if not valid:
            continue
        travel = 0.0
        prior = (float(start_xy[0]), float(start_xy[1]))
        for point in [*path, destination]:
            travel += math.dist(prior, point)
            prior = point
        terrain_cost = sum(
            min(1.0, _nearest_terrain_distance(point, terrain))
            for point in path
        ) / max(1, len(path))
        alignment_cost = _alignment_cost(start_xy, path[-1], destination)
        score = travel + 0.25 * terrain_cost + 0.35 * alignment_cost
        path_candidates.append((score, side_order, path[:2]))
    if not path_candidates:
        return []
    return min(
        path_candidates,
        key=lambda value: (value[0], value[1], value[2]),
    )[2]


def select_semantic_waypoints(
    directive: Mapping[str, Any],
    current_pose: Sequence[float],
    next_directive: Mapping[str, Any] | None,
    terrain_map: np.ndarray | None,
    terrain_map_ext: np.ndarray | None,
    accumulated_terrain: np.ndarray | None,
    *,
    forbidden_regions: Sequence[Mapping[str, Any]] = (),
    navigation_config: Mapping[str, Any] | None = None,
    navigation_history: Sequence[Mapping[str, Any]] = (),
) -> list[tuple[float, float, float]]:
    """Select a semantic goal set using one continuous candidate cost."""
    config = dict(navigation_config or {})
    region = directive.get("trajectory_region")
    if not isinstance(region, Mapping):
        region = compile_directive_geometry(
            directive,
            clearance_m=float(config.get("terrain_obstacle_clearance_m", 0.75)),
            acceptance_radius_m=float(config.get("terrain_waypoint_acceptance_radius_m", 0.30)),
            stop_surface_distance_m=float(config.get("stop_surface_distance_m", 0.90)),
            stop_band_width_m=float(config.get("stop_band_width_m", 0.45)),
            near_surface_distance_m=float(config.get("near_surface_distance_m", 1.10)),
            near_band_width_m=float(config.get("near_band_width_m", 0.80)),
        ).get("trajectory_region", {})
    terrain_values = [value for value in (terrain_map, terrain_map_ext, accumulated_terrain) if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] >= 3 and len(value)]
    terrain = np.concatenate(terrain_values, axis=0) if terrain_values else np.empty((0, 3), dtype=np.float64)
    semantic_target = _target_xy(directive)
    start_xy = (float(current_pose[0]), float(current_pose[1]))
    next_target = _target_xy(next_directive) if isinstance(next_directive, Mapping) else None
    action = str(directive.get("action", region.get("action", "go_to"))).lower()
    semantic_object = directive.get("object", {})
    semantic_object_id = (
        semantic_object.get("object_id")
        if isinstance(semantic_object, Mapping)
        else None
    )
    try:
        step_index = int(directive.get("order", 0))
    except (TypeError, ValueError):
        step_index = 0
    failure_exclusion_radius = max(
        0.05,
        float(config.get("terrain_waypoint_acceptance_radius_m", 0.30)),
        float(
            config.get(
                "minimum_independent_viewpoint_separation_m",
                config.get("terrain_waypoint_acceptance_radius_m", 0.30),
            )
        ),
    )
    failed_waypoint_points = _failed_waypoint_points(
        navigation_history,
        step_index=step_index,
        semantic_object_id=semantic_object_id,
        radius_m=failure_exclusion_radius,
    )
    blocking_forbidden_regions = [
        value
        for value in forbidden_regions
        if isinstance(value, Mapping)
        and value.get("forbidden") is True
        and segment_intersects_region(start_xy, semantic_target, value)
    ]
    if (
        action not in {"avoid", "avoid_near", "avoid_between"}
        and blocking_forbidden_regions
    ):
        def blocking_distance(value: Mapping[str, Any]) -> float:
            center = value.get("center_xy", ())
            if not isinstance(center, Sequence) or len(center) < 2:
                return float("inf")
            try:
                return math.dist(
                    start_xy,
                    (float(center[0]), float(center[1])),
                )
            except (TypeError, ValueError):
                return float("inf")

        blocking_region = min(
            blocking_forbidden_regions,
            key=blocking_distance,
        )
        local_detour = _avoid_detour_waypoints(
            blocking_region,
            start_xy,
            semantic_target,
            terrain,
            forbidden_regions,
            config,
        )
        if local_detour:
            first = local_detour[0]
            continuation = (
                local_detour[1]
                if len(local_detour) > 1 else semantic_target
            )
            return [(
                first[0],
                first[1],
                math.atan2(
                    continuation[1] - first[1],
                    continuation[0] - first[0],
                ),
            )]
    if action in {"pass_near", "pass_by", "path_near", "pass_between", "between"}:
        route = _corridor_waypoints(
            directive,
            region,
            start_xy,
            next_target,
            terrain,
            forbidden_regions,
        )
    elif action in {"avoid", "avoid_near", "avoid_between"} or region.get("forbidden") is True:
        route = _avoid_detour_waypoints(
            region,
            start_xy,
            next_target,
            terrain,
            forbidden_regions,
            config,
        )
    else:
        route = []
    if action in {"pass_near", "pass_by", "path_near", "pass_between", "between"}:
        if len(route) != 2:
            return []
        valid = [point for point in route if not _forbidden(point, start_xy, forbidden_regions)]
        if len(valid) != 2:
            return []
        return [
            (valid[index][0], valid[index][1], math.atan2(valid[index + 1][1] - valid[index][1], valid[index + 1][0] - valid[index][0]) if index == 0 else math.atan2(semantic_target[1] - valid[index][1], semantic_target[0] - valid[index][0]))
            for index in range(2)
        ]
    if route:
        if action in {"avoid", "avoid_near", "avoid_between"}:
            heading_target = next_target or semantic_target
            return [
                (
                    point[0],
                    point[1],
                    math.atan2(
                        (
                            route[index + 1][1]
                            if index + 1 < len(route)
                            else heading_target[1]
                        ) - point[1],
                        (
                            route[index + 1][0]
                            if index + 1 < len(route)
                            else heading_target[0]
                        ) - point[0],
                    ),
                )
                for index, point in enumerate(route[:2])
            ]
    candidate = _select_semantic_candidate(
        directive,
        region,
        start_xy,
        next_target,
        terrain,
        _region_samples(region),
        forbidden_regions,
        config,
        failed_waypoint_points=failed_waypoint_points,
    )
    if candidate is None:
        return []
    probe_object = directive.get("object", {})
    probe_look_at = (
        directive.get("probe_look_at_xy")
        or (
            probe_object.get("probe_look_at_xy")
            if isinstance(probe_object, Mapping) else None
        )
    )
    heading_target = (
        probe_look_at
        if action in {"probe", "explore"}
        and isinstance(probe_look_at, Sequence)
        and not isinstance(probe_look_at, (str, bytes))
        and len(probe_look_at) >= 2
        else next_target or semantic_target
    )
    return [(candidate[0], candidate[1], math.atan2(heading_target[1] - candidate[1], heading_target[0] - candidate[0]))]


def semantic_waypoint_segment_output(
    trajectory_directives: list[dict],
    *,
    constraint_set_id: str,
    active_directive_index: int = 0,
    competition_geometry: dict | None = None,
    navigation_config: dict | None = None,
    navigation_history: Sequence[Mapping[str, Any]] = (),
) -> dict:
    """Compile one active semantic step to at most two sparse waypoints."""
    if not trajectory_directives or not (0 <= active_directive_index < len(trajectory_directives)):
        return {"schema_version": "semantic_waypoint_segment_v2", "status": "blocked", "reason": "waypoint_coordinate_missing", "waypoints": []}
    config = dict(navigation_config or {})
    compiled: list[dict] = []
    for directive in trajectory_directives:
        value = dict(directive)
        if not isinstance(value.get("trajectory_region"), Mapping):
            value = compile_directive_geometry(
                value,
                clearance_m=float(config.get("terrain_obstacle_clearance_m", 0.75)),
                acceptance_radius_m=float(config.get("terrain_waypoint_acceptance_radius_m", 0.30)),
                stop_surface_distance_m=float(config.get("stop_surface_distance_m", 0.90)),
                stop_band_width_m=float(config.get("stop_band_width_m", 0.45)),
                near_surface_distance_m=float(config.get("near_surface_distance_m", 1.10)),
                near_band_width_m=float(config.get("near_band_width_m", 0.80)),
            )
        compiled.append(value)
    active = compiled[active_directive_index]
    if bool(active.get("forbidden")) and str(active.get("action", "")).lower() not in {"avoid", "avoid_near", "avoid_between"}:
        return {"schema_version": "semantic_waypoint_segment_v2", "status": "blocked", "reason": "active_waypoint_directive_forbidden", "waypoints": []}
    current_pose = _current_map_pose((competition_geometry or {}).get("state_estimation_path", ""))
    if current_pose is None:
        return {"schema_version": "semantic_waypoint_segment_v2", "status": "blocked", "reason": "segment_start_pose_missing", "waypoints": []}
    paths = [str((competition_geometry or {}).get(key, "")) for key in ("terrain_map_path", "terrain_map_ext_path", "accumulated_terrain_path")]
    loaded = _load_official_terrain(
        terrain_paths=paths,
        obstacle_height_threshold=float(config.get("terrain_obstacle_height_threshold", 0.05)),
        obstacle_clearance_m=float(config.get("terrain_obstacle_clearance_m", 0.75)),
        voxel_size_m=float(config.get("terrain_voxel_size_m", 0.05)),
    )
    terrain = loaded[0] if loaded is not None else np.empty((0, 3), dtype=np.float64)
    terrain_diagnostics = loaded[3] if loaded is not None else {"terrain_source_authority": "official_terrain_map", "terrain_source_missing": True}
    empty_terrain = np.empty((0, 3), dtype=np.float64)
    point_arrays = [terrain, empty_terrain, empty_terrain]
    next_directive = next(
        (
            value
            for value in compiled[active_directive_index + 1:]
            if not bool(value.get("forbidden"))
        ),
        None,
    )
    forbidden = [
        value.get("trajectory_region", value)
        for index, value in enumerate(compiled)
        if index != active_directive_index and bool(value.get("forbidden"))
    ]
    waypoints = select_semantic_waypoints(
        active,
        current_pose,
        next_directive,
        *point_arrays,
        forbidden_regions=[value for value in forbidden if isinstance(value, Mapping)],
        navigation_config=config,
        navigation_history=navigation_history,
    )
    if not waypoints:
        return {"schema_version": "semantic_waypoint_segment_v2", "status": "blocked", "reason": "semantic_candidate_set_empty", "waypoints": [], "terrain_projection_diagnostics": terrain_diagnostics}
    active_region = dict(active.get("trajectory_region", {}))
    active_action = str(active.get("action", active_region.get("action", "go_to"))).lower()
    if active_action in {"pass_near", "pass_by", "path_near", "pass_between", "between"} and len(waypoints) >= 2:
        active_region["ingress_xy"] = list(waypoints[0][:2])
        active_region["egress_xy"] = list(waypoints[1][:2])
    context = {
        "constraint_set_id": str(constraint_set_id),
        "step_index": int(active.get("order", active_directive_index)),
        "order": int(active.get("order", active_directive_index)),
        "action": active_action,
        "evidence_viewpoint": bool(active.get("evidence_viewpoint", False)),
        "semantic_object_id": active_region.get("semantic_object_id"),
        "semantic_object_class": active_region.get("semantic_object_class", ""),
        "is_terminal": bool(active.get("terminal", False)),
        "semantic_region": active_region,
        "semantic_target_xy": list(_target_xy(active)),
        "look_at_xy": list(active.get("look_at_xy", _target_xy(active))),
        "local_waypoint_count": len(waypoints),
        "waypoint_selection": "semantic_goal_set_continuous_cost",
    }
    result = {
        "schema_version": "semantic_waypoint_segment_v2",
        "status": "completed",
        "constraint_set_id": str(constraint_set_id),
        "step_index": int(active.get("order", active_directive_index)),
        "semantic_target_object_id": active_region.get("semantic_object_id"),
        "is_terminal": bool(active.get("terminal", False)),
        "evidence_viewpoint": bool(active.get("evidence_viewpoint", False)),
        "frame": "map",
        "semantic_target_xy": list(_target_xy(active)),
        "waypoints": [list(value) for value in waypoints],
        "waypoint_context": context,
        "trajectory_constraints": [
            active_region if index == active_directive_index else dict(value.get("trajectory_region", {}))
            for index, value in enumerate(compiled)
            if isinstance(value.get("trajectory_region"), Mapping)
        ],
        "terrain_projection_diagnostics": terrain_diagnostics,
        "terrain_adjustment": {"mode": "official_converter_near_goal", "semantic_target_preserved": True},
        "local_waypoint_count": len(waypoints),
        "output_policy": "semantic_goal_set_with_soft_terrain_cost_sparse_route",
        "expected_information_gain": context["action"] in {"probe", "explore"},
    }
    return result
