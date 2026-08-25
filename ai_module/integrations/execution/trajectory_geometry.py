"""Geometry for semantic instruction constraints.

The AI module owns semantic regions only. The official converter and FAR
consume the resulting semantic waypoints and remain responsible for local
terrain adjustment and obstacle avoidance.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _xy(value: Mapping[str, Any]) -> list[float]:
    center = value.get("center_3d")
    if not isinstance(center, Sequence) or isinstance(center, (str, bytes)) or len(center) < 2:
        raise ValueError("trajectory_object_center_invalid")
    result = [float(center[0]), float(center[1])]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("trajectory_object_center_invalid")
    return result


def _center_z(value: Mapping[str, Any]) -> float:
    try:
        result = float(value.get("center_3d")[2])
    except (TypeError, ValueError, IndexError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _extent_xy(value: Mapping[str, Any]) -> tuple[float, float]:
    extent = value.get("bbox_3d")
    if not isinstance(extent, Sequence) or isinstance(extent, (str, bytes)) or len(extent) < 2:
        raise ValueError("trajectory_object_bbox_invalid")
    width, depth = float(extent[0]), float(extent[1])
    if not all(math.isfinite(item) and item > 0.0 for item in (width, depth)):
        raise ValueError("trajectory_object_bbox_invalid")
    return width, depth


def _yaw(value: Mapping[str, Any]) -> float:
    for key in ("footprint_yaw_rad", "orientation_yaw_rad", "yaw_rad"):
        raw = value.get(key)
        if raw is not None:
            try:
                result = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(result):
                return result
    obb = value.get("footprint_obb")
    if isinstance(obb, Mapping):
        try:
            result = float(obb.get("yaw_rad", 0.0))
        except (TypeError, ValueError):
            result = 0.0
        if math.isfinite(result):
            return result
    return 0.0


def _obb_polygon(value: Mapping[str, Any]) -> list[list[float]]:
    center = _xy(value)
    width, depth = _extent_xy(value)
    obb = value.get("footprint_obb")
    if isinstance(obb, Mapping):
        raw_center = obb.get("center_xy", center)
        raw_extent = obb.get("extent_xy", obb.get("size_xy", (width, depth)))
        if (
            isinstance(raw_center, Sequence)
            and len(raw_center) >= 2
            and isinstance(raw_extent, Sequence)
            and len(raw_extent) >= 2
        ):
            center = [float(raw_center[0]), float(raw_center[1])]
            width, depth = float(raw_extent[0]), float(raw_extent[1])
    half_x, half_y = 0.5 * width, 0.5 * depth
    yaw = _yaw(value)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    corners = ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y))
    return [
        [center[0] + cosine * x - sine * y, center[1] + sine * x + cosine * y]
        for x, y in corners
    ]


def _polygon_centroid(polygon: Sequence[Sequence[float]]) -> list[float]:
    if not polygon:
        return [0.0, 0.0]
    return [
        sum(float(point[0]) for point in polygon) / len(polygon),
        sum(float(point[1]) for point in polygon) / len(polygon),
    ]


def _point_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.dist(point[:2], start[:2])
    scale = max(0.0, min(1.0, (
        (float(point[0]) - float(start[0])) * dx
        + (float(point[1]) - float(start[1])) * dy
    ) / length_squared))
    nearest = [float(start[0]) + scale * dx, float(start[1]) + scale * dy]
    return math.dist(point[:2], nearest)


def _point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    inside = False
    if len(polygon) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        x1, y1 = float(first[0]), float(first[1])
        x2, y2 = float(second[0]), float(second[1])
        if _point_segment_distance((x, y), first, second) <= 1e-8:
            return True
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / max(1e-12, y2 - y1) + x1
            if x < crossing:
                inside = not inside
    return inside


def _polygon_distance(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> float:
    if _point_in_polygon(point, polygon):
        return 0.0
    return min(
        (_point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
         for index in range(len(polygon))),
        default=float("inf"),
    )


def _segments_intersect(first: Sequence[float], second: Sequence[float], third: Sequence[float], fourth: Sequence[float]) -> bool:
    def orientation(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
        return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (
            float(b[1]) - float(a[1])
        ) * (float(c[0]) - float(a[0]))

    values = (
        orientation(first, second, third),
        orientation(first, second, fourth),
        orientation(third, fourth, first),
        orientation(third, fourth, second),
    )
    if values[0] == 0.0 and _point_segment_distance(third, first, second) <= 1e-8:
        return True
    if values[1] == 0.0 and _point_segment_distance(fourth, first, second) <= 1e-8:
        return True
    if values[2] == 0.0 and _point_segment_distance(first, third, fourth) <= 1e-8:
        return True
    if values[3] == 0.0 and _point_segment_distance(second, third, fourth) <= 1e-8:
        return True
    return (values[0] > 0.0) != (values[1] > 0.0) and (values[2] > 0.0) != (values[3] > 0.0)


def _inflate_polygon(polygon: Sequence[Sequence[float]], amount: float) -> list[list[float]]:
    center = _polygon_centroid(polygon)
    result = []
    for point in polygon:
        dx, dy = float(point[0]) - center[0], float(point[1]) - center[1]
        norm = math.hypot(dx, dy) or 1.0
        result.append([float(point[0]) + amount * dx / norm, float(point[1]) + amount * dy / norm])
    return result


def _support_anchor(obj: Mapping[str, Any], anchors: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Choose a likely support surface without binding ordinary landmarks."""
    obj_center = _xy(obj)
    obj_width, obj_depth = _extent_xy(obj)
    candidates = []
    for anchor in anchors:
        try:
            anchor_center = _xy(anchor)
            anchor_width, anchor_depth = _extent_xy(anchor)
        except ValueError:
            continue
        if _center_z(anchor) > _center_z(obj) + 0.30:
            continue
        if math.dist(obj_center, anchor_center) > 0.5 * math.hypot(anchor_width, anchor_depth) + 0.75:
            continue
        area_ratio = (anchor_width * anchor_depth) / max(1e-6, obj_width * obj_depth)
        candidates.append((area_ratio, -math.dist(obj_center, anchor_center), anchor))
    return max(candidates, key=lambda value: (value[0], value[1]))[2] if candidates else None


def point_in_region(point: Sequence[float], region: Mapping[str, Any]) -> bool:
    """Evaluate semantic region membership in map XY coordinates."""
    kind = str(region.get("kind", ""))
    if kind in {"approach_band", "pass_corridor"}:
        polygon = region.get("footprint_polygon_xy", ())
        if not isinstance(polygon, Sequence) or len(polygon) < 3 or _point_in_polygon(point, polygon):
            return False
        distance = _polygon_distance(point, polygon)
        return float(region.get("surface_distance_min_m", 0.0)) <= distance <= float(region.get("surface_distance_max_m", 0.0))
    polygon = region.get("polygon_xy")
    if isinstance(polygon, Sequence) and len(polygon) >= 3:
        return _point_in_polygon(point, polygon)
    radius = float(region.get("radius_m", -1.0))
    center = region.get("center_xy")
    if str(region.get("shape", "")) == "capsule":
        segment = region.get("segment_xy")
        return bool(isinstance(segment, Sequence) and len(segment) == 2 and _point_segment_distance(point, segment[0], segment[1]) <= radius)
    return bool(isinstance(center, Sequence) and len(center) >= 2 and radius >= 0.0 and math.dist(point[:2], center[:2]) <= radius)


def segment_intersects_region(start: Sequence[float], end: Sequence[float], region: Mapping[str, Any]) -> bool:
    if point_in_region(start, region) or point_in_region(end, region):
        return True
    polygon = region.get("polygon_xy")
    if not isinstance(polygon, Sequence) and str(region.get("kind", "")) in {
        "approach_band", "pass_corridor"
    }:
        polygon = region.get("footprint_polygon_xy")
    if isinstance(polygon, Sequence) and len(polygon) >= 3:
        if any(_segments_intersect(start, end, polygon[index], polygon[(index + 1) % len(polygon)]) for index in range(len(polygon))):
            return True
    samples = max(2, int(math.ceil(math.dist(start[:2], end[:2]) / 0.10)))
    return any(
        point_in_region([
            float(start[0]) + (float(end[0]) - float(start[0])) * index / samples,
            float(start[1]) + (float(end[1]) - float(start[1])) * index / samples,
        ], region)
        for index in range(1, samples)
    )


def compile_directive_geometry(
    directive: Mapping[str, Any],
    *,
    clearance_m: float,
    acceptance_radius_m: float,
    stop_surface_distance_m: float = 0.90,
    stop_band_width_m: float = 0.45,
    near_surface_distance_m: float = 1.10,
    near_band_width_m: float = 0.80,
) -> dict[str, Any]:
    """Compile one directive into support-aware semantic goal geometry."""
    obj = directive.get("object")
    action = str(directive.get("action", "go_to")).strip().lower()
    explicit_navigation = directive.get("navigation_target_xy")
    explicit_xy = None
    if (
        action in {"probe", "explore"}
        and isinstance(explicit_navigation, Sequence)
        and not isinstance(explicit_navigation, (str, bytes))
        and len(explicit_navigation) >= 2
    ):
        try:
            candidate_xy = [
                float(explicit_navigation[0]),
                float(explicit_navigation[1]),
            ]
        except (TypeError, ValueError):
            candidate_xy = None
        if candidate_xy is not None and all(
            math.isfinite(value) for value in candidate_xy
        ):
            explicit_xy = candidate_xy
    if not isinstance(obj, Mapping):
        if explicit_xy is None:
            raise ValueError("trajectory_object_missing")
        obj = {}
    # A probe/explore directive is a viewpoint acquisition, not a physical
    # object hypothesis.  Its navigation target is the geometry authority;
    # requiring a fabricated center/bbox here incorrectly turns an open-world
    # frontier into trajectory_object_center_invalid.
    if explicit_xy is not None:
        semantic_target = list(explicit_xy)
    else:
        semantic_target = _xy(obj)
    anchors = [value for value in directive.get("anchor_objects", ()) if isinstance(value, Mapping)]
    support = (
        _support_anchor(obj, anchors)
        if explicit_xy is None
        and action in {"go_to", "stop_at", "go_near", "stop_near"}
        else None
    )
    footprint_object = support or obj
    try:
        semantic_footprint_polygon = _obb_polygon(footprint_object)
    except ValueError:
        semantic_footprint_polygon = []
    footprint_polygon = (
        semantic_footprint_polygon if explicit_xy is None else []
    )

    if action in {"pass_near", "pass_by", "path_near"}:
        kind = "pass_corridor"
        region = {
            "kind": kind,
            "shape": "approach_band",
            "footprint_polygon_xy": footprint_polygon,
            "center_xy": _polygon_centroid(footprint_polygon),
            "surface_distance_min_m": 0.70,
            "surface_distance_max_m": 1.60,
        }
    elif action in {"pass_between", "between"}:
        kind = "between_corridor"
        if len(anchors) < 2:
            region = {
                "kind": kind,
                "shape": "blocked",
                "polygon_xy": [],
                "center_xy": list(semantic_target),
                "geometry_blocked": True,
                "geometry_blocked_reason": "between_region_anchors_missing",
            }
        else:
            first, second = _xy(anchors[0]), _xy(anchors[1])
            dx, dy = second[0] - first[0], second[1] - first[1]
            length = math.hypot(dx, dy)
            if length <= 1e-6:
                region = {
                    "kind": kind,
                    "shape": "blocked",
                    "polygon_xy": [],
                    "center_xy": list(semantic_target),
                    "geometry_blocked": True,
                    "geometry_blocked_reason": "between_region_anchors_coincident",
                }
            else:
                nx, ny = -dy / length, dx / length
                width = max(0.35, min(1.25, 0.5 * length - 0.5 * max(_extent_xy(anchors[0])) - 0.5 * max(_extent_xy(anchors[1]))))
                corridor = [
                    [first[0] + dx * 0.20 - nx * width, first[1] + dy * 0.20 - ny * width],
                    [second[0] - dx * 0.20 - nx * width, second[1] - dy * 0.20 - ny * width],
                    [second[0] - dx * 0.20 + nx * width, second[1] - dy * 0.20 + ny * width],
                    [first[0] + dx * 0.20 + nx * width, first[1] + dy * 0.20 + ny * width],
                ]
                region = {
                    "kind": kind,
                    "shape": "polygon",
                    "polygon_xy": corridor,
                    "center_xy": [(first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5],
                    "anchor_footprints_xy": [_obb_polygon(anchors[0]), _obb_polygon(anchors[1])],
                }
    elif action in {"avoid_near", "avoid", "avoid_between"} or bool(directive.get("forbidden")):
        kind = "forbidden_polygon"
        polygon = _inflate_polygon(footprint_polygon, float(clearance_m) + float(acceptance_radius_m))
        region = {
            "kind": kind,
            "shape": "polygon",
            "polygon_xy": polygon,
            "center_xy": _polygon_centroid(polygon),
        }
    else:
        kind = "approach_band"
        stop_action = action in {"go_to", "stop_at", "stop_near"}
        surface_distance = (
            float(stop_surface_distance_m)
            if stop_action else float(near_surface_distance_m)
        )
        band_width = (
            float(stop_band_width_m)
            if stop_action else float(near_band_width_m)
        )
        lower_bound = 0.80 if stop_action else 0.70
        upper_bound = 1.10 if stop_action else 1.60
        region = {
            "kind": kind,
            "shape": "approach_band",
            "footprint_polygon_xy": footprint_polygon,
            "center_xy": _polygon_centroid(footprint_polygon),
            "surface_distance_min_m": max(lower_bound, surface_distance - 0.5 * band_width),
            "surface_distance_max_m": min(upper_bound, surface_distance + 0.5 * band_width),
        }

    if explicit_xy is not None:
        # A CountQuery probe is an observation transaction at a
        # query-conditioned viewpoint, not a request to approach the
        # currently selected object's footprint.  Keeping the object's
        # approach band while moving toward a tuple-participant locus
        # creates two incompatible arrival locations.  Preserve the
        # semantic footprint as provenance and make the actual viewpoint
        # the single navigation/arrival geometry used by both planner and
        # trajectory monitor.
        region = {
            "kind": "viewpoint_region",
            "shape": "circle",
            "center_xy": list(explicit_xy),
            "radius_m": max(0.75, float(acceptance_radius_m)),
            "viewpoint_tolerance_m": max(0.75, float(acceptance_radius_m)),
            "semantic_footprint_polygon_xy": semantic_footprint_polygon,
            "semantic_footprint_object_id": (
                int(footprint_object.get("object_id"))
                if str(footprint_object.get("object_id", "")).lstrip("-").isdigit()
                else None
            ),
        }
        region["navigation_target_xy"] = list(explicit_xy)

    raw_look_at = directive.get(
        "probe_look_at_xy",
        obj.get("probe_look_at_xy", semantic_target),
    )
    if (
        not isinstance(raw_look_at, Sequence)
        or isinstance(raw_look_at, (str, bytes))
        or len(raw_look_at) < 2
    ):
        raw_look_at = semantic_target
    try:
        look_at_xy = [float(raw_look_at[0]), float(raw_look_at[1])]
    except (TypeError, ValueError):
        look_at_xy = list(semantic_target)
    if not all(math.isfinite(value) for value in look_at_xy):
        look_at_xy = list(semantic_target)

    region.update({
        "order": int(directive.get("order", 0)),
        "constraint": str(directive.get("trajectory_constraint", "ENTER_REGION")),
        "region_kind": str(
            directive.get(
                "trajectory_region_kind",
                "VIEWPOINT_REGION" if region.get("kind") == "viewpoint_region" else "NEAR_REGION",
            )
        ),
        "action": action,
        "semantic_object_id": int(obj["object_id"]) if str(obj.get("object_id", "")).lstrip("-").isdigit() and int(obj.get("object_id", -1)) >= 0 else None,
        "semantic_object_class": str(obj.get("class_label", "")),
        "terminal": bool(directive.get("terminal", False)),
        "forbidden": bool(directive.get("forbidden", False)) or kind == "forbidden_polygon",
        "support_object_id": int(support.get("object_id")) if support is not None and str(support.get("object_id", "")).lstrip("-").isdigit() else None,
        "navigation_footprint": (
            None
            if region.get("kind") == "viewpoint_region"
            else int(footprint_object.get("object_id", -1))
            if str(footprint_object.get("object_id", "")).lstrip("-").isdigit()
            else None
        ),
        "semantic_target_xy": semantic_target,
        "look_at_xy": look_at_xy,
        "geometry_version": "support_aware_goal_geometry_v1",
        # A metric relation can provide a useful route hypothesis before its
        # identity/geometry evidence has converged.  Preserve that state in
        # the trajectory contract so physical arrival cannot be mistaken for
        # final semantic completion.
        "selector_provisional": bool(directive.get("selector_provisional", False)),
        "selector_state": str(directive.get("selector_state", "")),
        "selector_relation_id": str(directive.get("selector_relation_id", "")),
        "selector_completion_ready": bool(
            directive.get("selector_completion_ready", False)
        ),
    })
    result = dict(directive)
    result["trajectory_region"] = region
    result["navigation_target_xy"] = list(
        region.get("navigation_target_xy", region.get("center_xy", semantic_target))
    )
    result["semantic_target_xy"] = semantic_target
    result["look_at_xy"] = semantic_target
    return result


def compile_directive_geometries(
    directives: Sequence[Mapping[str, Any]],
    *,
    clearance_m: float,
    acceptance_radius_m: float,
) -> list[dict[str, Any]]:
    return [
        compile_directive_geometry(value, clearance_m=clearance_m, acceptance_radius_m=acceptance_radius_m)
        for value in directives
    ]
