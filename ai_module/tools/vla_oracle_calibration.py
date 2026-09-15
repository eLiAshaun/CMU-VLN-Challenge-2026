#!/usr/bin/env python3
"""Development-only calibration of the VLA-3D offline oracle.

This tool is intentionally separate from production and from the balanced VLA
suite.  It executes the relation functions from VLA-3D's
``scene_graph/generate_scene_info.py`` with a small dependency-neutral geometry
shim because the host does not provide the generator's optional pandas,
shapely, scipy, or numba packages.  It never writes to Unity.zip and never
launches Unity/ROS.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import importlib.util
import importlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import types
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = REPO_ROOT / "ai_module"
FAST_TOOL = AI_ROOT / "tools" / "vla_fast_validation.py"
OFFICIAL_ROOT = Path("/home/robot/cmu_vln/VLA-3D/scene_graph")
ARCHIVE = Path("/home/robot/cmu_vln/VLA-3D_dataset/Unity.zip")
OUTPUT = AI_ROOT / "artifacts" / "vla_fast_validation"
RELATIONS = ("on", "above", "below", "near", "in", "between")
RNG_SEED = 20260825


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported_json_value:{type(value).__name__}")


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _load_fast_tool() -> types.ModuleType:
    if str(AI_ROOT) not in sys.path:
        sys.path.insert(0, str(AI_ROOT))
    spec = importlib.util.spec_from_file_location("vla_fast_validation_adapter", FAST_TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"fast_tool_import_spec_missing:{FAST_TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(sum(
        float(points[index][0]) * float(points[(index + 1) % len(points)][1])
        - float(points[(index + 1) % len(points)][0]) * float(points[index][1])
        for index in range(len(points))
    ) / 2.0)


def _cross(first: Sequence[float], second: Sequence[float], point: Sequence[float]) -> float:
    return (
        (float(second[0]) - float(first[0])) * (float(point[1]) - float(first[1]))
        - (float(second[1]) - float(first[1])) * (float(point[0]) - float(first[0]))
    )


def _clip_polygon(subject: Sequence[Sequence[float]], clip: Sequence[Sequence[float]]) -> list[list[float]]:
    """Convex polygon intersection used only to execute VLA's IoM helpers."""
    output = [list(map(float, point)) for point in subject]
    orientation = 1.0 if sum(
        float(clip[index][0]) * float(clip[(index + 1) % len(clip)][1])
        - float(clip[(index + 1) % len(clip)][0]) * float(clip[index][1])
        for index in range(len(clip))
    ) >= 0.0 else -1.0
    for index in range(len(clip)):
        if not output:
            break
        edge_start = clip[index]
        edge_end = clip[(index + 1) % len(clip)]
        previous = output[-1]
        clipped: list[list[float]] = []
        previous_inside = orientation * _cross(edge_start, edge_end, previous) >= -1e-9
        for current in output:
            current_inside = orientation * _cross(edge_start, edge_end, current) >= -1e-9
            if current_inside != previous_inside:
                direction = [current[0] - previous[0], current[1] - previous[1]]
                edge = [edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]]
                denominator = direction[0] * edge[1] - direction[1] * edge[0]
                if abs(denominator) > 1e-12:
                    delta = [edge_start[0] - previous[0], edge_start[1] - previous[1]]
                    t = (delta[0] * edge[1] - delta[1] * edge[0]) / denominator
                    clipped.append([previous[0] + t * direction[0], previous[1] + t * direction[1]])
            if current_inside:
                clipped.append(current)
            previous, previous_inside = current, current_inside
        output = clipped
    return output


class _Polygon:
    def __init__(self, points: Sequence[Sequence[float]]) -> None:
        self.points = [list(map(float, point)) for point in points]
        self.area = _polygon_area(self.points)

    def intersection(self, other: "_Polygon") -> "_Polygon":
        return _Polygon(_clip_polygon(self.points, other.points))

    def union(self, other: "_Polygon") -> "_Polygon":
        # The official relation functions only use union through helpers that
        # are not part of the selected calibration predicates.  Returning a
        # conservative polygon keeps the shim deterministic for import.
        return _Polygon(self.points + other.points)


def _segment_distance(first: Sequence[float], second: Sequence[float], point: Sequence[float]) -> float:
    dx = float(second[0]) - float(first[0])
    dy = float(second[1]) - float(first[1])
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return math.hypot(float(point[0]) - float(first[0]), float(point[1]) - float(first[1]))
    t = max(0.0, min(1.0, ((float(point[0]) - float(first[0])) * dx + (float(point[1]) - float(first[1])) * dy) / denominator))
    return math.hypot(float(point[0]) - (float(first[0]) + t * dx), float(point[1]) - (float(first[1]) + t * dy))


def _polygon_distance(first: _Polygon, second: _Polygon) -> float:
    if first.intersection(second).area > 1e-9:
        return 0.0
    return min(
        _segment_distance(second.points[index], second.points[(index + 1) % len(second.points)], point)
        for point in first.points
        for index in range(len(second.points))
    ) if first.points and second.points else 0.0


def _install_official_dependency_shim() -> dict[str, Any]:
    """Install only import-time substitutes for absent optional packages."""
    bbox = types.ModuleType("bbox_utils")

    def rotz(theta: float) -> np.ndarray:
        return np.array([[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]])

    def get_bbox_coords_heading(prefix: str, row: Mapping[str, Any]) -> list[list[float]]:
        lengths = [float(row[f"{prefix}_bbox_{axis}length"]) for axis in "xyz"]
        center = [float(row[f"{prefix}_bbox_c{axis}"]) for axis in "xyz"]
        heading = float(row[f"{prefix}_bbox_heading"])
        half_x, half_y, half_z = (value / 2.0 for value in lengths)
        x = [-half_x, half_x, half_x, -half_x, -half_x, half_x, half_x, -half_x]
        y = [half_y, half_y, -half_y, -half_y, half_y, half_y, -half_y, -half_y]
        z = [half_z, half_z, half_z, half_z, -half_z, -half_z, -half_z, -half_z]
        corners = rotz(heading) @ np.vstack([x, y, z])
        corners[0, :] += center[0]
        corners[1, :] += center[1]
        corners[2, :] += center[2]
        return [[float(corners[axis, index]) for axis in range(3)] for index in range(8)]

    def calculate_iom_poly(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
        first_polygon = _Polygon(np.asarray(first["bbox"])[[0, 3, 2, 1, 0], :2])
        second_polygon = _Polygon(np.asarray(second["bbox"])[[0, 3, 2, 1, 0], :2])
        minimum = min(first_polygon.area, second_polygon.area)
        return first_polygon.intersection(second_polygon).area / minimum if minimum > 0.0 else 0.0

    def calculate_iom(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        low = np.maximum(first[:, :2], second[:, :2])
        high = np.minimum(first[:, 2:], second[:, 2:])
        intersection = np.maximum(high - low, 0.0).prod(axis=1)
        first_area = np.maximum(first[:, 2] - first[:, 0], 0.0) * np.maximum(first[:, 3] - first[:, 1], 0.0)
        second_area = np.maximum(second[:, 2] - second[:, 0], 0.0) * np.maximum(second[:, 3] - second[:, 1], 0.0)
        return intersection / np.maximum(np.minimum(first_area, second_area), 1e-12)

    def get_2d_bboxes(axes: Sequence[int], first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first_2d, second_2d = first[..., axes], second[..., axes]
        return (
            np.stack([first_2d[..., 0].min(axis=-1), first_2d[..., 1].min(axis=-1), first_2d[..., 0].max(axis=-1), first_2d[..., 1].max(axis=-1)], axis=-1),
            np.stack([second_2d[..., 0].min(axis=-1), second_2d[..., 1].min(axis=-1), second_2d[..., 0].max(axis=-1), second_2d[..., 1].max(axis=-1)], axis=-1),
        )

    def calculate_iom_poly_vectorized(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        result = []
        for index in range(len(second)):
            first_polygon = _Polygon(first[index, :, :2])
            second_polygon = _Polygon(second[index, :, :2])
            minimum = min(first_polygon.area, second_polygon.area)
            result.append(first_polygon.intersection(second_polygon).area / minimum if minimum > 0.0 else 0.0)
        return np.asarray(result, dtype=float)

    def is_inside_bbox(point: np.ndarray, bbox: np.ndarray) -> bool:
        origin = np.asarray(bbox[0], dtype=float)
        basis = np.column_stack([np.asarray(bbox[1]) - origin, np.asarray(bbox[3]) - origin, np.asarray(bbox[4]) - origin])
        try:
            coordinates = np.linalg.solve(basis, np.asarray(point, dtype=float) - origin)
        except np.linalg.LinAlgError:
            return False
        return bool(np.all(coordinates > -1e-8) and np.all(coordinates < 1.0 + 1e-8))

    def get_bbox_horiz_distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
        first_polygon = _Polygon(np.asarray(first["bbox"])[[0, 3, 2, 1, 0], :2])
        second_polygon = _Polygon(np.asarray(second["bbox"])[[0, 3, 2, 1, 0], :2])
        return _polygon_distance(first_polygon, second_polygon)

    bbox.get_bbox_coords_heading = get_bbox_coords_heading
    bbox.get_obj_size = lambda row: tuple(float(row[f"object_bbox_{axis}length"]) for axis in "xyz")
    bbox.get_obj_volume = lambda row: math.prod(bbox.get_obj_size(row))
    bbox.calculate_iom_poly = calculate_iom_poly
    bbox.calculate_iom = calculate_iom
    bbox.calculate_iom_poly_vectorized = calculate_iom_poly_vectorized
    bbox.get_2D_bboxes = get_2d_bboxes
    bbox.is_inside_bbox = is_inside_bbox
    bbox.get_bbox_horiz_distance = get_bbox_horiz_distance
    bbox.Polygon = _Polygon
    sys.modules["bbox_utils"] = bbox

    shapely = types.ModuleType("shapely")
    shapely_geometry = types.ModuleType("shapely.geometry")
    shapely_geometry.Polygon = _Polygon
    shapely.geometry = shapely_geometry
    shapely.distance = _polygon_distance
    shapely.__path__ = []
    sys.modules["shapely"] = shapely
    sys.modules["shapely.geometry"] = shapely_geometry

    sys.modules["pandas"] = types.ModuleType("pandas")
    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda values, *args, **kwargs: values
    sys.modules["tqdm"] = tqdm
    colors = types.ModuleType("colors")
    sys.modules["colors"] = colors
    return {"bbox_helper": "source-compatible dependency shim", "optional_packages_missing": ["pandas", "shapely", "scipy", "numba"]}


def _load_official() -> tuple[types.ModuleType, dict[str, Any]]:
    required_modules = ("pandas", "shapely", "scipy", "numba", "tqdm")
    missing_modules = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            missing_modules.append(f"{module_name}:{type(exc).__name__}")
    if missing_modules:
        raise RuntimeError(
            "official_vla_dependencies_missing:" + ",".join(missing_modules)
        )

    # The official generator imports these modules by their short names.  Drop
    # any prior development-only shim entries before loading the real files.
    for module_name in (
        "bbox_utils",
        "colors",
        "special_relation_classes",
        "polygon_intersection",
    ):
        sys.modules.pop(module_name, None)
    if str(OFFICIAL_ROOT) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_ROOT))
    spec = importlib.util.spec_from_file_location("vla_official_generate_scene_info", OFFICIAL_ROOT / "generate_scene_info.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("official_generator_import_spec_missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {
        "bbox_helper": str((OFFICIAL_ROOT / "bbox_utils.py").resolve()),
        "optional_packages_missing": [],
        "dependency_mode": "isolated pip target with official imports",
        "official_source": str((OFFICIAL_ROOT / "generate_scene_info.py").resolve()),
    }


class _GeneratorArgs:
    between_iom = 0.5
    vertical_iom = 0.5
    near_thres = 0.01
    overlap_thres = 0.3
    symmetry_thres = 0.5
    distance_thres = 1.0
    anchor_size_thres = 1.5
    on_thres = 0.01
    in_thres = 0.1
    under_thres = 0.01
    hanging_thres_h = 0.01
    hanging_thres_v = 0.5
    ordered_thres = 0.2


def _csv(zipped: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    return list(csv.DictReader(zipped.read(member).decode("utf-8-sig").splitlines()))


def _load_raw(archive_path: Path, fast: types.ModuleType) -> dict[str, dict[str, Any]]:
    scenes: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = fast._scene_members(archive)
        for scene_name in sorted(members):
            files = members[scene_name]
            object_rows = _csv(archive, files["object_result.csv"])
            region_rows = _csv(archive, files["region_result.csv"])
            graph = json.loads(archive.read(files["scene_graph.json"]))
            scenes[scene_name] = {
                "name": scene_name,
                "object_rows": object_rows,
                "region_rows": region_rows,
                "rows_by_id": {_int(row.get("object_id")): row for row in object_rows},
                "graph": graph,
                "region_rows_by_id": {str(row.get("region_id")): row for row in region_rows},
                "official_regions": {},
                "fast_objects": {},
            }
            for row in object_rows:
                object_id = _int(row.get("object_id"))
                if object_id >= 0:
                    scenes[scene_name]["fast_objects"][object_id] = fast._gt_object(row, point_count=64)
    return scenes


def _official_object(row: Mapping[str, Any], official: types.ModuleType) -> dict[str, Any]:
    return {
        "object_id": str(row["object_id"]),
        "raw_label": row["raw_label"],
        "nyu_id": row["nyu_id"],
        "nyu40_id": row["nyu40_id"],
        "nyu_label": row["nyu_label"],
        "nyu40_label": row["nyu40_label"],
        "bbox": official.get_bbox_coords_heading("object", row),
        "center": [float(row[f"object_bbox_c{axis}"]) for axis in "xyz"],
        "size": list(official.get_obj_size(row)),
        "volume": float(official.get_obj_volume(row)),
    }


def _official_region(scene: Mapping[str, Any], region_id: str, official: types.ModuleType) -> dict[str, Any]:
    rows = scene["rows_by_id"]
    region_row = scene["region_rows_by_id"][str(region_id)]
    remove = {"void", "otherstructure", "otherprop", "otherfurniture", "nyu40class", "unknown"}
    objects = [
        _official_object(row, official)
        for row in rows.values()
        if str(row.get("region_id")) == str(region_id)
        and str(row.get("nyu_label")) not in remove
    ]
    region = {
        "region_id": str(region_id),
        "region_name": region_row.get("region_label", ""),
        "region_bbox": official.get_bbox_coords_heading("region", region_row),
        "objects": objects,
        "relationships": {},
    }
    official.compute_spatial_relationships(_GeneratorArgs(), region)
    return region


def _get_official_region(scene: dict[str, Any], region_id: str, official: types.ModuleType) -> dict[str, Any]:
    key = str(region_id)
    if key not in scene["official_regions"]:
        scene["official_regions"][key] = _official_region(scene, key, official)
    return scene["official_regions"][key]


def _stored_region(scene: Mapping[str, Any], region_id: str) -> Mapping[str, Any]:
    return (scene["graph"].get("regions", {}) or {}).get(str(region_id), {})


def _semantic_edges(mapping: Mapping[str, Any], relation: str) -> set[tuple[int, ...]]:
    edges: set[tuple[int, ...]] = set()
    for raw_source, raw_targets in (mapping or {}).items():
        source = _int(raw_source)
        if source < 0 or not isinstance(raw_targets, list):
            continue
        for raw_target in raw_targets:
            if relation == "between" and isinstance(raw_target, list) and len(raw_target) == 2:
                first, second = _int(raw_target[0]), _int(raw_target[1])
                if first >= 0 and second >= 0:
                    edges.add((source, *sorted((first, second))))
            else:
                target = _int(raw_target)
                if target < 0:
                    continue
                # Official generator stores binary relations as anchor -> targets.
                # The semantic contract is RELATION(target, anchor).
                edges.add((target, source))
    return edges


def _official_edges(region: Mapping[str, Any], relation: str) -> set[tuple[int, ...]]:
    return _semantic_edges(region.get("relationships", {}).get(relation, {}), relation)


def _stored_edges(scene: Mapping[str, Any], region_id: str, relation: str) -> set[tuple[int, ...]]:
    return _semantic_edges(_stored_region(scene, region_id).get("relationships", {}).get(relation, {}), relation)


def _bbox_trace(row: Mapping[str, Any], official: types.ModuleType) -> dict[str, Any]:
    corners = official.get_bbox_coords_heading("object", row)
    return {
        "object_id": _int(row.get("object_id")),
        "center": [float(row[f"object_bbox_c{axis}"]) for axis in "xyz"],
        "size": [float(row[f"object_bbox_{axis}length"]) for axis in "xyz"],
        "heading_rad": float(row["object_bbox_heading"]),
        "corners": corners,
        "min_z": min(value[2] for value in corners),
        "max_z": max(value[2] for value in corners),
        "xy_envelope": [[min(value[axis] for value in corners) for axis in (0, 1)], [max(value[axis] for value in corners) for axis in (0, 1)]],
    }


def _corner_order_invariant_error(
    first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]
) -> float:
    if len(first) != len(second) or not first:
        return float("inf")
    forward = max(
        min(math.dist(tuple(map(float, left)), tuple(map(float, right))) for right in second)
        for left in first
    )
    reverse = max(
        min(math.dist(tuple(map(float, right)), tuple(map(float, left))) for left in first)
        for right in second
    )
    return max(forward, reverse)


def _archive_obb_audit(
    scenes: Mapping[str, dict[str, Any]], official: types.ModuleType, count: int = 20
) -> dict[str, Any]:
    records = []
    for scene_name, scene in sorted(scenes.items()):
        for region_id, stored_region in sorted(
            (scene["graph"].get("regions", {}) or {}).items(),
            key=lambda value: str(value[0]),
        ):
            for stored_object in stored_region.get("objects", ()):
                object_id = _int(stored_object.get("object_id"))
                row = scene["rows_by_id"].get(object_id)
                if row is None:
                    continue
                trace = _bbox_trace(row, official)
                center = trace["center"]
                size = trace["size"]
                axis_corners = [
                    [
                        center[0] + sx * size[0] / 2.0,
                        center[1] + sy * size[1] / 2.0,
                        center[2] + sz * size[2] / 2.0,
                    ]
                    for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3)
                ]
                stored_corners = stored_object.get("bbox", ())
                corner_error = _corner_order_invariant_error(
                    trace["corners"], stored_corners
                )
                axis_error = _corner_order_invariant_error(axis_corners, stored_corners)
                stored_center = [float(value) for value in stored_object.get("center", ())]
                stored_size = [float(value) for value in stored_object.get("size", ())]
                records.append(
                    {
                        "scene": scene_name,
                        "region": str(region_id),
                        "object_id": object_id,
                        "heading_rad": trace["heading_rad"],
                        "official_center": center,
                        "stored_center": stored_center,
                        "center_error_m": max(
                            (abs(first - second) for first, second in zip(center, stored_center)),
                            default=float("inf"),
                        ),
                        "official_size": size,
                        "stored_size": stored_size,
                        "size_error_m": max(
                            (abs(first - second) for first, second in zip(size, stored_size)),
                            default=float("inf"),
                        ),
                        "official_to_archive_corner_error_m": corner_error,
                        "axis_to_archive_corner_error_m": axis_error,
                        "z_center_reconstruction_error_m": abs(
                            (trace["min_z"] + trace["max_z"]) / 2.0 - center[2]
                        ),
                    }
                )
                if len(records) >= count:
                    break
            if len(records) >= count:
                break
        if len(records) >= count:
            break
    return {
        "objects": len(records),
        "center_exact": sum(value["center_error_m"] <= 1e-7 for value in records),
        "size_exact": sum(value["size_error_m"] <= 1e-7 for value in records),
        "corner_equivalent": sum(
            value["official_to_archive_corner_error_m"] <= 1e-6 for value in records
        ),
        "axis_corner_equivalent": sum(
            value["axis_to_archive_corner_error_m"] <= 1e-6 for value in records
        ),
        "z_center_exact": sum(
            value["z_center_reconstruction_error_m"] <= 1e-7 for value in records
        ),
        "records": records,
        "interpretation": (
            "Order-invariant comparison of official VLA bbox_utils corners against "
            "stored scene_graph.json corners; the axis-aligned production-shaped "
            "adapter is reported separately."
        ),
    }


def _relation_candidates(scene: Mapping[str, Any], relation: str) -> list[dict[str, Any]]:
    result = []
    for region_id, region in sorted(scene["graph"].get("regions", {}).items(), key=lambda value: str(value[0])):
        edges = _semantic_edges(region.get("relationships", {}).get(relation, {}), relation)
        for edge in sorted(edges):
            result.append({"scene": scene["name"], "region": str(region_id), "relation": relation, "edge": list(edge)})
    return result


def _self_check(scenes: Mapping[str, dict[str, Any]], official: types.ModuleType) -> dict[str, Any]:
    checks = []
    selected_regions: dict[str, set[str]] = defaultdict(set)
    for scene_name, scene in sorted(scenes.items()):
        for relation in RELATIONS:
            positives = _relation_candidates(scene, relation)
            for item in positives[:3]:
                selected_regions[scene_name].add(item["region"])
    for scene_name, region_ids in sorted(selected_regions.items()):
        scene = scenes[scene_name]
        for region_id in sorted(region_ids):
            try:
                generated = _get_official_region(scene, region_id, official)
                for relation in RELATIONS:
                    generated_edges = _official_edges(generated, relation)
                    stored_edges = _stored_edges(scene, region_id, relation)
                    checks.append({
                        "scene": scene_name,
                        "region": region_id,
                        "relation": relation,
                        "official_edge_count": len(generated_edges),
                        "stored_edge_count": len(stored_edges),
                        "missing_from_stored": [list(value) for value in sorted(generated_edges - stored_edges)[:20]],
                        "extra_in_stored": [list(value) for value in sorted(stored_edges - generated_edges)[:20]],
                        "status": "PASS" if generated_edges == stored_edges else "FAIL",
                    })
            except Exception as exc:
                for relation in RELATIONS:
                    checks.append({"scene": scene_name, "region": region_id, "relation": relation, "status": "ERROR", "reason": f"{type(exc).__name__}:{exc}"})
    by_relation = {}
    for relation in RELATIONS:
        subset = [value for value in checks if value["relation"] == relation]
        by_relation[relation] = {
            "regions": len(subset),
            "pass": sum(value["status"] == "PASS" for value in subset),
            "fail": sum(value["status"] == "FAIL" for value in subset),
            "error": sum(value["status"] == "ERROR" for value in subset),
        }
    return {"status": "PASS" if all(value["status"] == "PASS" for value in checks) else "FAIL", "scene_count": len(selected_regions), "region_count": len(checks) // len(RELATIONS) if RELATIONS else 0, "by_relation": by_relation, "checks": checks}


def _direction_audit(scenes: Mapping[str, dict[str, Any]], official: types.ModuleType) -> dict[str, Any]:
    traces = []
    for relation in RELATIONS:
        selected = []
        for scene_name, scene in sorted(scenes.items()):
            for candidate in _relation_candidates(scene, relation):
                if candidate["region"] not in scene["official_regions"]:
                    try:
                        _get_official_region(scene, candidate["region"], official)
                    except Exception:
                        continue
                selected.append(candidate)
                if len({value["scene"] for value in selected}) >= 3 and len(selected) >= 6:
                    break
            if len({value["scene"] for value in selected}) >= 3 and len(selected) >= 6:
                break
        for candidate in selected[:6]:
            scene = scenes[candidate["scene"]]
            target_id = int(candidate["edge"][0])
            anchor_ids = [int(value) for value in candidate["edge"][1:]]
            region = _get_official_region(scene, candidate["region"], official)
            objects = region["objects"]
            index_by_id = {int(value["object_id"]): index for index, value in enumerate(objects)}
            if relation == "between":
                raw_call = official.relate_between(_GeneratorArgs().between_iom, index_by_id[target_id], objects, _GeneratorArgs().overlap_thres, _GeneratorArgs().symmetry_thres, _GeneratorArgs().distance_thres, _GeneratorArgs().anchor_size_thres)
                semantic = f"BETWEEN(target={target_id}, anchors={anchor_ids})"
                stored = region["relationships"].get("between", {}).get(str(target_id), [])
                official_call = {"function": "relate_between", "anchor_idx": target_id, "returned": raw_call}
            else:
                function = getattr(official, f"relate_{relation}")
                if relation == "on":
                    raw_call = function(_GeneratorArgs().vertical_iom, _GeneratorArgs().on_thres, _GeneratorArgs().under_thres, _GeneratorArgs().in_thres, index_by_id[anchor_ids[0]], objects)
                elif relation == "above":
                    raw_call = function(_GeneratorArgs().vertical_iom, _GeneratorArgs().on_thres, index_by_id[anchor_ids[0]], objects)
                elif relation == "below":
                    raw_call = function(_GeneratorArgs().vertical_iom, _GeneratorArgs().under_thres, index_by_id[anchor_ids[0]], objects)
                elif relation == "near":
                    raw_call = function(_GeneratorArgs().near_thres, index_by_id[anchor_ids[0]], objects, region["region_bbox"])
                else:
                    raw_call = function(index_by_id[anchor_ids[0]], objects, _GeneratorArgs().in_thres)
                semantic = f"{relation.upper()}(target={target_id}, anchor={anchor_ids[0]})"
                stored = region["relationships"].get(relation, {}).get(str(anchor_ids[0]), [])
                official_call = {"function": f"relate_{relation}", "anchor_idx": anchor_ids[0], "returned": raw_call}
            traces.append({
                "scene": candidate["scene"],
                "region": candidate["region"],
                "relation": relation,
                "semantic_expression": semantic,
                "stored_representation": {"key": target_id if relation == "between" else anchor_ids[0], "value": stored},
                "official_production_function_call": official_call,
                "target": _bbox_trace(scene["rows_by_id"][target_id], official),
                "anchors": [_bbox_trace(scene["rows_by_id"][anchor_id], official) for anchor_id in anchor_ids],
                "orientation_status": "PASS" if target_id in ([_int(value) for value in raw_call] if relation != "between" else [target_id]) else "CHECK_MANUALLY",
            })
    return {"relations": {relation: sum(value["relation"] == relation for value in traces) for relation in RELATIONS}, "traces": traces}


def _legacy_adapter_direction_audit(scenes: Mapping[str, dict[str, Any]], fast: types.ModuleType) -> dict[str, Any]:
    records = []
    for scene_name, scene in sorted(scenes.items()):
        adapter_truth = fast._graph_truth(scene["graph"])
        for relation in RELATIONS:
            semantic = _stored_edges(scene, "0", relation) if "0" in scene["graph"].get("regions", {}) else set()
            # Use all stored regions for the direction comparison.
            semantic = set()
            for region_id in scene["graph"].get("regions", {}):
                semantic.update(_stored_edges(scene, str(region_id), relation))
            legacy = {tuple(int(item) for item in value) for value in adapter_truth.get(relation, set())}
            direct_hits = len(legacy & semantic)
            reversed_hits = len({(value[1], value[0]) for value in legacy if len(value) == 2} & semantic)
            records.append({"scene": scene_name, "relation": relation, "stored_semantic_edges": len(semantic), "legacy_adapter_edges": len(legacy), "direct_hits": direct_hits, "reversed_hits": reversed_hits, "orientation_inversion_detected": reversed_hits > direct_hits and relation != "between"})
    return {"by_relation": {relation: {"scenes": sum(value["relation"] == relation for value in records), "direct_hits": sum(value["direct_hits"] for value in records if value["relation"] == relation), "reversed_hits": sum(value["reversed_hits"] for value in records if value["relation"] == relation), "inversion_detected": sum(bool(value["orientation_inversion_detected"]) for value in records if value["relation"] == relation)} for relation in RELATIONS}, "records": records, "interpretation": "This compares the pre-calibration fast adapter against the stored graph after applying the official anchor->target storage contract; it is not a production result."}


def _obb_audit(scenes: Mapping[str, dict[str, Any]], official: types.ModuleType, count: int = 20) -> dict[str, Any]:
    records = []
    for scene_name, scene in sorted(scenes.items()):
        for object_id, row in sorted(scene["rows_by_id"].items()):
            if str(row.get("region_id")) == "-1":
                continue
            official_trace = _bbox_trace(row, official)
            center = official_trace["center"]
            size = official_trace["size"]
            adapter_axis = [[center[0] + sx * size[0] / 2.0, center[1] + sy * size[1] / 2.0, center[2] + sz * size[2] / 2.0] for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3)]
            official_set = {tuple(round(value, 8) for value in point) for point in official_trace["corners"]}
            axis_set = {tuple(round(value, 8) for value in point) for point in adapter_axis}
            records.append({"scene": scene_name, "object_id": object_id, "region": str(row.get("region_id")), "heading_rad": official_trace["heading_rad"], "center_equal": center == [float(row[f"object_bbox_c{axis}"]) for axis in "xyz"], "size_equal": size == [float(row[f"object_bbox_{axis}length"]) for axis in "xyz"], "official_corners": official_trace["corners"], "adapter_axis_corners": adapter_axis, "corner_equivalent_to_existing_axis_adapter": official_set == axis_set, "min_z": official_trace["min_z"], "max_z": official_trace["max_z"], "z_center_reconstruction_error": abs((official_trace["min_z"] + official_trace["max_z"]) / 2.0 - center[2])})
            if len(records) >= count:
                break
        if len(records) >= count:
            break
    return {"objects": len(records), "center_exact": sum(value["center_equal"] for value in records), "size_exact": sum(value["size_equal"] for value in records), "corner_equivalent": sum(value["corner_equivalent_to_existing_axis_adapter"] for value in records), "z_center_exact": sum(value["z_center_reconstruction_error"] <= 1e-7 for value in records), "records": records, "interpretation": "CSV center/size are exact; existing offline production-shaped bbox_3d is axis-aligned and does not preserve nonzero heading corners."}


def _region_negative_audit(scenes: Mapping[str, dict[str, Any]], official: types.ModuleType) -> dict[str, Any]:
    records = []
    for scene_name, scene in sorted(scenes.items()):
        regions = scene["graph"].get("regions", {}) or {}
        for region_id, stored_region in sorted(regions.items(), key=lambda value: str(value[0])):
            try:
                generated = _get_official_region(scene, str(region_id), official)
            except Exception as exc:
                records.append({"scene": scene_name, "region": str(region_id), "status": "ERROR", "reason": f"{type(exc).__name__}:{exc}"})
                continue
            object_ids = [int(value["object_id"]) for value in generated["objects"]]
            for relation in RELATIONS:
                positive = _official_edges(generated, relation)
                if relation == "between":
                    eligible = {(subject, *sorted((first, second))) for subject in object_ids for first in object_ids if first != subject for second in object_ids if second not in {subject, first}}
                else:
                    eligible = {(target, anchor) for anchor in object_ids for target in object_ids if target != anchor}
                negative = eligible - positive
                records.append({"scene": scene_name, "region": str(region_id), "relation": relation, "eligible": len(eligible), "positive": len(positive), "negative": len(negative), "cross_region_included": False, "status": "PASS"})
    by_relation = {}
    for relation in RELATIONS:
        subset = [value for value in records if value.get("relation") == relation]
        by_relation[relation] = {"regions": len(subset), "eligible": sum(value.get("eligible", 0) for value in subset), "positive": sum(value.get("positive", 0) for value in subset), "negative": sum(value.get("negative", 0) for value in subset), "cross_region_included": sum(bool(value.get("cross_region_included")) for value in subset)}
    return {"by_relation": by_relation, "records": records, "policy": "same official region object set only; absent cross-region edge is not a hard negative"}


def _archive_negative_audit(scenes: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    records = []
    for scene_name, scene in sorted(scenes.items()):
        for region_id, region in sorted(
            (scene["graph"].get("regions", {}) or {}).items(),
            key=lambda value: str(value[0]),
        ):
            object_ids = [
                _int(value.get("object_id"))
                for value in region.get("objects", ())
                if _int(value.get("object_id")) >= 0
            ]
            for relation in RELATIONS:
                positive = _stored_edges(scene, str(region_id), relation)
                if relation == "between":
                    eligible = {
                        (subject, *sorted((first, second)))
                        for subject in object_ids
                        for first in object_ids
                        if first != subject
                        for second in object_ids
                        if second not in {subject, first}
                    }
                else:
                    eligible = {
                        (target, anchor)
                        for anchor in object_ids
                        for target in object_ids
                        if target != anchor
                    }
                records.append(
                    {
                        "scene": scene_name,
                        "region": str(region_id),
                        "relation": relation,
                        "eligible": len(eligible),
                        "positive": len(positive),
                        "negative": len(eligible - positive),
                        "cross_region_included": False,
                    }
                )
    return {
        "by_relation": {
            relation: {
                "regions": sum(value["relation"] == relation for value in records),
                "eligible": sum(value["eligible"] for value in records if value["relation"] == relation),
                "positive": sum(value["positive"] for value in records if value["relation"] == relation),
                "negative": sum(value["negative"] for value in records if value["relation"] == relation),
                "cross_region_included": sum(
                    bool(value["cross_region_included"])
                    for value in records
                    if value["relation"] == relation
                ),
            }
            for relation in RELATIONS
        },
        "records": records,
        "policy": "stored scene_graph.json region object set; absent cross-region edge is not a hard negative",
    }


def _relation_kernel(predicate: str, geometry: Mapping[str, Any]) -> tuple[str, str]:
    """Clean metric component, deliberately independent of runtime evidence fields."""
    if predicate == "near":
        gaps = geometry.get("bounding_sphere_surface_gaps_m", ())
        if isinstance(gaps, Sequence) and gaps and float(min(gaps)) <= 0.0:
            return "YES", "surface_contact_only_kernel"
        return "UNKNOWN", "near_region_scale_not_in_metric_kernel"
    if predicate in {"above", "below", "on"}:
        overlap = geometry.get("horizontal_projection_axis_overlap_m", ())
        delta = _float(geometry.get("vertical_center_delta_m"))
        gap = _float(geometry.get("vertical_gap_m"))
        if not isinstance(overlap, Sequence) or len(overlap) != 2 or not math.isfinite(delta):
            return "UNKNOWN", "vertical_or_projection_geometry_missing"
        horizontal = all(float(value) > 0.0 for value in overlap)
        if predicate == "above":
            return ("YES", "clean_vertical_and_horizontal_overlap") if horizontal and delta > 0.0 else ("NO", "clean_vertical_or_projection_contradiction")
        if predicate == "below":
            return ("YES", "clean_vertical_and_horizontal_overlap") if horizontal and delta < 0.0 else ("NO", "clean_vertical_or_projection_contradiction")
        return ("YES", "clean_support_projection") if horizontal and delta > 0.0 and gap <= 0.02 else ("NO", "clean_support_projection_contradiction")
    if predicate == "in":
        margins = geometry.get("containment_margins_m")
        if not isinstance(margins, Sequence) or len(margins) != 3:
            return "UNKNOWN", "containment_geometry_missing"
        values = [float(value) for pair in margins for value in pair]
        return ("YES", "all_clean_containment_margins_nonnegative") if all(value >= 0.0 for value in values) else ("NO", "clean_containment_margin_negative")
    if predicate == "between":
        interval = geometry.get("segment_position_interval")
        perpendicular = _float(geometry.get("perpendicular_distance_m"))
        corridor = _float(geometry.get("corridor_half_width_m"))
        if not isinstance(interval, Sequence) or len(interval) != 2 or not math.isfinite(perpendicular) or not math.isfinite(corridor):
            return "UNKNOWN", "between_geometry_missing"
        return ("YES", "clean_segment_and_corridor") if 0.0 <= float(interval[0]) and float(interval[1]) <= 1.0 and perpendicular <= corridor else ("NO", "clean_segment_or_corridor_contradiction")
    return "UNKNOWN", "predicate_not_in_kernel"


def _full_contract(predicate: str, subject: Mapping[str, Any], anchors: Sequence[Mapping[str, Any]], fast: types.ModuleType) -> dict[str, Any]:
    node = {"id": "calibration", "predicate": predicate}
    geometry = fast.relation_geometry_diagnostic(node, subject, anchors)
    qwen = {"state": "supported", "jointly_observable": True, "subject_role_state": "YES", "object_role_states": ["YES"] * len(anchors), "confidence": 1.0}
    try:
        state, reason = fast._geometry_consistency(node, geometry, subject, anchors, qwen)
        error = ""
    except Exception as exc:
        state, reason, error = "UNKNOWN", f"owner_exception:{type(exc).__name__}:{exc}", f"{type(exc).__name__}:{exc}"
    return {"geometry": geometry, "state": state, "reason": reason, "error": error}


def _manual_gold(scenes: Mapping[str, dict[str, Any]], official: types.ModuleType, fast: types.ModuleType) -> dict[str, Any]:
    cases = []
    for relation in RELATIONS:
        candidates = []
        for scene_name, scene in sorted(scenes.items()):
            for candidate in _relation_candidates(scene, relation):
                if len({value["scene"] for value in candidates}) >= 3 and len(candidates) >= 8:
                    break
                candidates.append(candidate)
            if len({value["scene"] for value in candidates}) >= 3 and len(candidates) >= 8:
                break
        for candidate in candidates[:6]:
            scene = scenes[candidate["scene"]]
            target_id = int(candidate["edge"][0])
            anchor_ids = [int(value) for value in candidate["edge"][1:]]
            region = _stored_region(scene, candidate["region"])
            target = scene["fast_objects"][target_id]
            anchors = [scene["fast_objects"][value] for value in anchor_ids]
            geometry = fast.relation_geometry_diagnostic({"predicate": relation}, target, anchors)
            kernel_state, kernel_reason = _relation_kernel(relation, geometry)
            full = _full_contract(relation, target, anchors, fast)
            cases.append({"kind": "positive", "scene": candidate["scene"], "region": candidate["region"], "relation": relation, "target_id": target_id, "anchor_ids": anchor_ids, "official_truth": "YES", "geometry_kernel": [kernel_state, kernel_reason], "full_production_contract": {"state": full["state"], "reason": full["reason"], "error": full["error"]}, "geometry": geometry})
            # A matched negative is selected only from the same archived region
            # and the same official eligible object set.
            archive_positive = _stored_edges(scene, candidate["region"], relation)
            object_ids = [int(value["object_id"]) for value in region.get("objects", ())]
            if relation == "between":
                eligible = [(subject, first, second) for subject in object_ids for first in object_ids if first != subject for second in object_ids if second not in {subject, first}]
            else:
                eligible = [(target_value, anchor) for anchor in object_ids for target_value in object_ids if target_value != anchor]
            negative = next((value for value in eligible if tuple([value[0], *sorted(value[1:])]) not in archive_positive), None)
            if negative is not None:
                negative_target = int(negative[0])
                negative_anchors = [int(value) for value in negative[1:]]
                negative_geometry = fast.relation_geometry_diagnostic({"predicate": relation}, scene["fast_objects"][negative_target], [scene["fast_objects"][value] for value in negative_anchors])
                negative_kernel = _relation_kernel(relation, negative_geometry)
                negative_full = _full_contract(relation, scene["fast_objects"][negative_target], [scene["fast_objects"][value] for value in negative_anchors], fast)
                cases.append({"kind": "matched_negative", "scene": candidate["scene"], "region": candidate["region"], "relation": relation, "target_id": negative_target, "anchor_ids": negative_anchors, "official_truth": "NO", "geometry_kernel": list(negative_kernel), "full_production_contract": {"state": negative_full["state"], "reason": negative_full["reason"], "error": negative_full["error"]}, "geometry": negative_geometry})
    by_relation = {}
    for relation in RELATIONS:
        subset = [value for value in cases if value["relation"] == relation]
        by_relation[relation] = {"cases": len(subset), "positive": sum(value["kind"] == "positive" for value in subset), "matched_negative": sum(value["kind"] == "matched_negative" for value in subset), "kernel_correct": sum((value["geometry_kernel"][0] == value["official_truth"]) for value in subset), "full_yes": sum(value["full_production_contract"]["state"] == "YES" for value in subset), "missing_runtime_evidence": sum(value["geometry_kernel"][0] == "YES" and value["full_production_contract"]["state"] == "UNKNOWN" for value in subset)}
    return {"by_relation": by_relation, "cases": cases, "truth_authority": "stored scene_graph.json"}


def _metadata_query(row: Mapping[str, Any]) -> dict[str, Any]:
    from integrations.semantics.task_compiler import GROUNDING_HINTS

    anchors = row.get("anchors", {}) if isinstance(row.get("anchors"), Mapping) else {}
    target_attributes = {}
    if str(row.get("target_color_used", "")).strip():
        target_attributes["color"] = str(row.get("target_color_used", "")).strip()
    if str(row.get("target_size_used", "")).strip():
        target_attributes["size"] = str(row.get("target_size_used", "")).strip()
    target_class = str(row.get("target_class", ""))
    target_aliases = list(GROUNDING_HINTS.get(target_class, {}).get("aliases", ()))
    entities = [{"id": "target_0", "role": "target", "class_name": target_class, "aliases": target_aliases, "attributes": target_attributes}]
    object_entities = []
    for index, value in enumerate(sorted(anchors.values(), key=lambda item: str(item.get("index", "")))):
        entity_id = f"anchor_{index}"
        object_entities.append(entity_id)
        class_name = str(value.get("class", ""))
        aliases = list(GROUNDING_HINTS.get(class_name, {}).get("aliases", ()))
        entities.append({"id": entity_id, "role": "anchor", "class_name": class_name, "aliases": aliases, "attributes": {"color": str(value.get("color_used", "")).strip()} if str(value.get("color_used", "")).strip() else {}})
    return {"schema_version": "task_ir_v2", "task_type": "object_reference", "original_question": str(row.get("language", "")), "entities": entities, "relations": [{"id": "rel_0", "predicate": str(row.get("relation", "")).lower(), "subject_entity": "target_0", "object_entities": object_entities, "depends_on": []}], "ordered_trajectory_constraints": [], "target_entity": "target_0", "required_classes": [value["class_name"] for value in entities], "output_contract": "/selected_object_marker"}


def _archive_truth_feature(
    scene: Mapping[str, Any],
    fast: types.ModuleType,
    predicate: str,
    subject: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
) -> bool:
    truth = fast._graph_truth(scene["graph"])
    subject_id = _int(subject.get("object_id"))
    anchor_ids = tuple(_int(value.get("object_id")) for value in anchors)
    normalized = _norm(predicate)
    if subject_id < 0 or any(value < 0 for value in anchor_ids):
        return False
    if normalized == "between":
        return (subject_id, *sorted(anchor_ids)) in truth.get(normalized, set())
    return (subject_id, *anchor_ids) in truth.get(normalized, set())


def _archive_ordered_selector_closures(
    task: Mapping[str, Any], scene: Mapping[str, Any], fast: types.ModuleType
) -> dict[str, dict[str, int]]:
    entities = {str(value.get("id")): value for value in task.get("entities", ())}

    def base_candidates(entity_id: str) -> list[dict[str, Any]]:
        entity = entities.get(str(entity_id), {})
        class_names = {
            str(entity.get("class_name", "")),
            *(str(value) for value in entity.get("aliases", ())),
        }
        return [
            value
            for value in scene["fast_objects"].values()
            if fast._matches_class_names(value, class_names)
            and fast._attribute_match(entity, value, {"objects": list(scene["fast_objects"].values())})
        ]

    closures: dict[str, dict[str, int]] = {}
    for relation in task.get("relations", ()):
        predicate = _norm(relation.get("predicate"))
        if predicate not in {"closest", "farthest"}:
            continue
        subject_id = str(relation.get("subject_entity", ""))
        anchor_ids = [str(value) for value in relation.get("object_entities", ())]
        if len(anchor_ids) != 1:
            continue
        candidates = base_candidates(subject_id)
        anchors = base_candidates(anchor_ids[0])
        pairs = [
            (candidate, anchor)
            for candidate in candidates
            for anchor in anchors
            if _archive_truth_feature(scene, fast, predicate, candidate, [anchor])
        ]
        if len(pairs) == 1:
            candidate, anchor = pairs[0]
            closures[str(relation.get("id", ""))] = {
                "selected_object_id": int(candidate["object_id"]),
                "selected_anchor_id": int(anchor["object_id"]),
            }
    return closures


def _archive_runtime_candidates(
    task: Mapping[str, Any],
    scene: Mapping[str, Any],
    fast: types.ModuleType,
    entity_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    from integrations.execution import query_executor

    previous_feature = query_executor._geometric_relation_feature

    def archive_feature(
        predicate: str,
        subject: Mapping[str, Any],
        anchors: Sequence[Mapping[str, Any]],
    ) -> bool:
        return _archive_truth_feature(scene, fast, predicate, subject, anchors)

    query_executor._geometric_relation_feature = archive_feature
    try:
        snapshot = {
            "objects": list(scene["fast_objects"].values()),
            "query_domain": {
                "selector_closures": _archive_ordered_selector_closures(task, scene, fast)
            },
        }
        resolver = fast._Resolver(task, snapshot)
        return resolver.resolve(str(entity_id))
    finally:
        query_executor._geometric_relation_feature = previous_feature


def _binding_case(row: Mapping[str, Any], scene: Mapping[str, Any], fast: types.ModuleType, mode: str) -> dict[str, Any]:
    task = _metadata_query(row) if mode == "METADATA_QUERYPROGRAM" else fast.compile_task("Find " + str(row.get("language", "")))
    snapshot = {"objects": copy.deepcopy(list(scene["fast_objects"].values())), "query_domain": {"selector_closures": {}}}
    try:
        if mode == "METADATA_QUERYPROGRAM":
            predicted, failures = _archive_runtime_candidates(task, scene, fast, str(task.get("target_entity", "target_0")))
        else:
            predicted, failures = fast._runtime_candidates(task, {"objects": snapshot["objects"]}, str(task.get("target_entity", "target_0")))
    except Exception as exc:
        predicted, failures = [], [f"{type(exc).__name__}:{exc}"]
    expected_target = _int(row.get("target_index"))
    expected_anchor_ids = sorted(_int(value.get("index")) for value in (row.get("anchors", {}) or {}).values() if isinstance(value, Mapping))
    predicted_ids = sorted(int(value["object_id"]) for value in predicted)
    anchor_predictions = []
    for relation in task.get("relations", ()):
        for entity_id in relation.get("object_entities", ()):
            if mode == "METADATA_QUERYPROGRAM":
                values, anchor_failures = _archive_runtime_candidates(task, scene, fast, str(entity_id))
            else:
                values, anchor_failures = fast._runtime_candidates(task, {"objects": snapshot["objects"]}, str(entity_id))
            anchor_predictions.append({"entity_id": entity_id, "ids": sorted(int(value["object_id"]) for value in values), "failures": anchor_failures[:3]})
    flat_anchor_ids = [value for item in anchor_predictions for value in item["ids"]]
    return {"scene": scene["name"], "relation": _norm(row.get("relation")), "mode": mode, "query": str(row.get("language", "")), "expected_target": expected_target, "predicted_target_ids": predicted_ids, "target_correct": predicted_ids == [expected_target], "expected_anchor_ids": expected_anchor_ids, "predicted_anchor_ids": flat_anchor_ids, "anchor_correct": all(expected in flat_anchor_ids for expected in expected_anchor_ids), "failures": failures[:5], "anchor_predictions": anchor_predictions}


def _binding_calibration(scenes: Mapping[str, dict[str, Any]], fast: types.ModuleType, quota: int = 8) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for scene_name, scene in scenes.items():
        for row in scene.get("referential_cases", ()):
            relation = _norm(row.get("relation"))
            if relation in (*RELATIONS, "closest", "farthest"):
                grouped[(scene_name, relation)].append(row)
    # ``referential_cases`` are attached by the fast adapter only in run();
    # this calibration loader attaches them directly below when needed.
    records = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda value: (str(value.get("target_index")), str(value.get("language", ""))))[:quota]
        for row in rows:
            for mode in ("METADATA_QUERYPROGRAM", "TEXT_COMPILER_PLUS_BINDING"):
                records.append(_binding_case(row, scenes[key[0]], fast, mode))
    by_mode_relation = {}
    for mode in ("METADATA_QUERYPROGRAM", "TEXT_COMPILER_PLUS_BINDING"):
        for relation in sorted({value["relation"] for value in records}):
            subset = [value for value in records if value["mode"] == mode and value["relation"] == relation]
            by_mode_relation[f"{mode}:{relation}"] = {"cases": len(subset), "target_correct": sum(value["target_correct"] for value in subset), "anchor_correct": sum(value["anchor_correct"] for value in subset), "target_accuracy": sum(value["target_correct"] for value in subset) / len(subset) if subset else 0.0, "anchor_accuracy": sum(value["anchor_correct"] for value in subset) / len(subset) if subset else 0.0}
    return {"cases": len(records), "by_mode_relation": by_mode_relation, "records": records[:500], "interpretation": "metadata mode isolates binding with VLA labels as evaluation-only fields; text mode includes current QueryCompiler"}


def _ordered_calibration(scenes: Mapping[str, dict[str, Any]], fast: types.ModuleType, quota: int = 120) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scene_name, scene in scenes.items():
        for row in scene.get("referential_cases", ()):
            relation = _norm(row.get("relation"))
            if relation in {"closest", "farthest"}:
                grouped[relation].append({"scene": scene_name, **row})
    records = []
    ordinal = {"first": 1, "second": 2, "third": 3}
    for relation, rows in sorted(grouped.items()):
        for row in sorted(rows, key=lambda value: (value["scene"], str(value.get("target_index")), str(value.get("language", ""))))[:quota]:
            scene = scenes[row["scene"]]
            anchor_ids = sorted(_int(value.get("index")) for value in (row.get("anchors", {}) or {}).values() if isinstance(value, Mapping))
            target_id = _int(row.get("target_index"))
            anchor_id = anchor_ids[0] if anchor_ids else -1
            target = scene["fast_objects"].get(target_id)
            anchor = scene["fast_objects"].get(anchor_id)
            if target is None or anchor is None:
                continue
            target_row = scene["rows_by_id"][target_id]
            domain = [value for value in scene["fast_objects"].values() if _norm(scene["rows_by_id"][int(value["object_id"])].get("nyu_id")) == _norm(target_row.get("nyu_id")) and str(scene["rows_by_id"][int(value["object_id"])].get("region_id")) == str(row.get("region_id")) and int(value["object_id"]) != anchor_id]
            pure = sorted(domain, key=lambda value: (math.dist(value["center_3d"], anchor["center_3d"]), int(value["object_id"])), reverse=relation == "farthest")
            expected_rank = next((number for word, number in ordinal.items() if re.search(rf"\b{word}\b", str(row.get("language", "")).lower())), 1)
            pure_ids = [int(value["object_id"]) for value in pure[:3]]
            production_ids = sorted(domain, key=lambda value: (fast._xy_distance(value, anchor), int(value["object_id"])), reverse=relation == "farthest")[:3]
            production_ids = [int(value["object_id"]) for value in production_ids]
            task = fast.compile_task("Find " + str(row.get("language", "")))
            predicted, failures = fast._runtime_candidates(task, {"objects": list(scene["fast_objects"].values()), "query_domain": {"selector_closures": {}}}, str(task.get("target_entity", "")))
            records.append({"scene": row["scene"], "region": row.get("region_id"), "relation": relation, "query": row.get("language"), "target_id": target_id, "anchor_id": anchor_id, "expected_rank": expected_rank, "pure_gt_order": pure_ids, "pure_gt_winner": pure_ids[expected_rank - 1] if len(pure_ids) >= expected_rank else None, "production_order": production_ids, "production_winner": production_ids[expected_rank - 1] if len(production_ids) >= expected_rank else None, "binding_ids": sorted(int(value["object_id"]) for value in predicted), "binding_failure": failures[:4], "pure_correct": len(pure_ids) >= expected_rank and pure_ids[expected_rank - 1] == target_id, "production_correct": len(production_ids) >= expected_rank and production_ids[expected_rank - 1] == target_id, "binding_correct": sorted(int(value["object_id"]) for value in predicted) == [target_id]})
    summary = {}
    for relation in sorted(grouped):
        subset = [value for value in records if value["relation"] == relation]
        summary[relation] = {"cases": len(subset), "pure_correct": sum(value["pure_correct"] for value in subset), "production_correct": sum(value["production_correct"] for value in subset), "binding_correct": sum(value["binding_correct"] for value in subset), "binding_unresolved": sum(not value["binding_ids"] for value in subset)}
    return {"by_relation": summary, "records": records, "layers": ["PURE_GT_RANKING", "PRODUCTION_ORDERED_GEOMETRY_XY", "QUERY_BINDING"]}


def _instruction_frame_audit(scenes: Mapping[str, dict[str, Any]], fast: types.ModuleType, questions_path: Path) -> dict[str, Any]:
    source = json.loads(questions_path.read_text(encoding="utf-8"))
    records = []
    for item in source:
        scene_name = str(item["scene"])
        scene = scenes[scene_name]
        for index, question in enumerate(item["questions"].get("instruction_following", ())):
            path = REPO_ROOT / "questions" / scene_name / ("trajectory_q4.ply" if index == 0 else "trajectory_q5.ply")
            if not path.exists():
                continue
            points = fast._read_ascii_trajectory(path)
            task = fast.compile_task(question)
            target_records = []
            for step in task.get("ordered_trajectory_constraints", ()):
                entity_id = str(step.get("target_entity", ""))
                entity = next((value for value in task.get("entities", ()) if str(value.get("id")) == entity_id), {})
                candidates = [value for value in scene["fast_objects"].values() if fast._class_match(entity.get("class_name"), value)]
                if candidates:
                    target = min(candidates, key=lambda value: int(value["object_id"]))
                    distances = [math.hypot(point[0] - target["center_3d"][0], point[1] - target["center_3d"][1]) for point in points]
                    target_records.append({"step": step.get("order"), "action": step.get("action"), "target_id": int(target["object_id"]), "target_center": target["center_3d"], "min_xy_distance_m": min(distances) if distances else None})
            records.append({"scene": scene_name, "question": question, "trajectory": str(path), "point_count": len(points), "start": points[0] if points else None, "end": points[-1] if points else None, "range": [[min(point[index] for point in points), max(point[index] for point in points)] for index in range(3)] if points else [], "target_records": target_records, "z_unique_rounded": sorted({round(point[2], 3) for point in points})[:12]})
    return {"cases": len(records), "same_frame_candidate": sum(bool(value["target_records"]) for value in records), "records": records, "interpretation": "numeric frame audit only; no InstructionExecutionState modification"}


def _attach_referential_cases(scenes: Mapping[str, dict[str, Any]], archive_path: Path, fast: types.ModuleType) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = fast._scene_members(archive)
        for scene_name, scene in scenes.items():
            payload = json.loads(archive.read(members[scene_name]["referential_statements.json"]))
            cases = []
            for region_id, phrases in (payload.get("regions", {}) or {}).items():
                if not isinstance(phrases, Mapping):
                    continue
                for language, rows in phrases.items():
                    for row in rows if isinstance(rows, list) else []:
                        if isinstance(row, Mapping) and "target_index" in row:
                            cases.append({"scene": scene_name, "region_id": str(region_id), "language": str(language), **dict(row)})
            scene["referential_cases"] = cases


def _cardinality_real_comparison() -> dict[str, Any]:
    report_path = AI_ROOT / "artifacts" / "phase3b0" / "PHASE3B0_E2_FORENSIC_REPORT.md"
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return {"recorded_examples": [{"name": "hotel_room_2/q1 bed-fragment/anchor case", "source": str(report_path), "available": bool(text), "evidence": "GT/runtime correspondence in report; runtime bed candidates 1,3,17; stale birth geometry versus later fused geometry; group overlap must not become independent count entity."}], "synthetic_summary": "Synthetic partial fragment uses same acquisition, same center, 0.55 bbox extent, one-third point sample, and same-view mask containment; compare these values against recorded evidence before interpreting the 15/15 failure."}


def _markdown(report: Mapping[str, Any]) -> str:
    if (
        report["official_vla_self_check"]["status"] != "PASS"
        and "binding_calibration" not in report
    ):
        lines = [
            "# VLA Oracle Calibration Report",
            "",
            f"ORACLE_TRUST_STATUS = {report['oracle_trust_status']}",
            "",
            "Official VLA self-check failed or errored. Downstream calibration blocks were intentionally not executed.",
            "",
            "## Scope and provenance",
            "",
            f"- Archive: `{report['provenance']['archive']}`",
            f"- Official source: `{report['provenance']['official_source']}`",
            f"- Official execution mode: `{report['provenance']['official_execution_mode']}`",
            f"- Official geometry dependencies: `{report['official_dependency_shim']}`",
            "- No production algorithm was modified; no Unity/ROS was launched.",
            "",
            "## OFFICIAL_VLA_SELF_CHECK",
            "",
            f"- Status: `{report['official_vla_self_check']['status']}`",
            "",
            "| Relation | Regions | Pass | Fail | Error |",
            "|---|---:|---:|---:|---:|",
        ]
        for relation, value in report["official_vla_self_check"]["by_relation"].items():
            lines.append(f"| {relation} | {value['regions']} | {value['pass']} | {value['fail']} | {value['error']} |")
        lines += [
            "",
            "## RELATION_DIRECTION_AUDIT",
            "",
            f"- Official traces: {len(report['relation_direction_audit']['traces'])}",
            f"- Legacy adapter direction summary: `{report['legacy_adapter_direction_audit']['by_relation']}`",
            "- Binary VLA storage is anchor -> target; semantic evaluation is RELATION(target, anchor).",
            "",
            "## OBB_FRAME_AUDIT",
            "",
            f"- Objects audited: {report['obb_frame_audit']['objects']}",
            f"- Center/size exact: {report['obb_frame_audit']['center_exact']}/{report['obb_frame_audit']['objects']}; axis-adapter corner equivalence: {report['obb_frame_audit']['corner_equivalent']}/{report['obb_frame_audit']['objects']}",
            "",
            "## ARCHIVE_OBB_CORNER_AUDIT",
            "",
            f"- Official bbox_utils corners vs stored scene_graph.json: {report['archive_obb_corner_audit']['corner_equivalent']}/{report['archive_obb_corner_audit']['objects']} order-invariant matches",
            f"- CSV center/size vs archive: {report['archive_obb_corner_audit']['center_exact']}/{report['archive_obb_corner_audit']['objects']} and {report['archive_obb_corner_audit']['size_exact']}/{report['archive_obb_corner_audit']['objects']}",
            f"- Axis-aligned adapter vs archive corners: {report['archive_obb_corner_audit']['axis_corner_equivalent']}/{report['archive_obb_corner_audit']['objects']}",
            "- This separates bbox geometry from the generator-versus-archive relation mismatch.",
            "",
            "## DECISION",
            "",
            report["decision"],
            "",
            "Detailed self-check traces are in `calibration_official_vla_self_check.json` and `vla_oracle_calibration.json`.",
            "",
        ]
        return "\n".join(lines)
    lines = [
        "# VLA Oracle Calibration Report", "", f"ORACLE_TRUST_STATUS = {report['oracle_trust_status']}", "", "This is a development-only calibration. It does not modify production algorithms, tune thresholds, launch Unity/ROS, or declare runtime acceptance.", "", "## Scope and provenance", "", f"- Archive: `{report['provenance']['archive']}`", f"- Official source: `{report['provenance']['official_source']}`", f"- Official execution mode: `{report['provenance']['official_execution_mode']}`", f"- Scenes loaded: {report['provenance']['scene_count']}", "- No hashes/checksums were performed.", "",
        "## OFFICIAL_VLA_SELF_CHECK", "", f"- Status: `{report['official_vla_self_check']['status']}`", f"- Selected scenes: {report['official_vla_self_check']['scene_count']}; selected region checks: {report['official_vla_self_check']['region_count']}", "", "| Relation | Regions | Pass | Fail | Error |", "|---|---:|---:|---:|---:|",
    ]
    for relation, value in report["official_vla_self_check"]["by_relation"].items():
        lines.append(f"| {relation} | {value['regions']} | {value['pass']} | {value['fail']} | {value['error']} |")
    lines += ["", "## RELATION_DIRECTION_AUDIT", "", "Stored binary representation is interpreted as `anchor -> target`; semantic evaluation is `RELATION(target, anchor)`. BETWEEN is stored as `target -> [anchor_1, anchor_2]`.", f"- Traces: {len(report['relation_direction_audit']['traces'])}", "", "## OBB_FRAME_AUDIT", "", f"- Objects audited: {report['obb_frame_audit']['objects']}", f"- Exact CSV center/size: {report['obb_frame_audit']['center_exact']}/{report['obb_frame_audit']['objects']}; z-center reconstruction: {report['obb_frame_audit']['z_center_exact']}/{report['obb_frame_audit']['objects']}", f"- Exact corner equivalence with existing axis-aligned adapter: {report['obb_frame_audit']['corner_equivalent']}/{report['obb_frame_audit']['objects']}", "- Nonzero heading is preserved by the official generator but is not represented by the existing production-shaped `bbox_3d` field; see JSON traces.", "", "## ARCHIVE_OBB_CORNER_AUDIT", "", f"- Official bbox_utils corners vs stored scene_graph.json: {report['archive_obb_corner_audit']['corner_equivalent']}/{report['archive_obb_corner_audit']['objects']} order-invariant matches", f"- CSV center/size vs archive: {report['archive_obb_corner_audit']['center_exact']}/{report['archive_obb_corner_audit']['objects']} and {report['archive_obb_corner_audit']['size_exact']}/{report['archive_obb_corner_audit']['objects']}", f"- Axis-aligned adapter vs archive corners: {report['archive_obb_corner_audit']['axis_corner_equivalent']}/{report['archive_obb_corner_audit']['objects']}", "", "## REGION_SCOPE_AUDIT", "", f"- Truth authority: `{report.get('truth_authority', 'official generator')}`", "- Relation eligibility is restricted to the archived per-region object set. Cross-region absence is not used as a hard negative.", "", "## NEGATIVE_SET_AUDIT", ""]
    for relation, value in report["negative_set_audit"]["by_relation"].items():
        lines.append(f"- `{relation}`: eligible={value['eligible']}, official_positive={value['positive']}, same-region_negative={value['negative']}, cross-region-included={value['cross_region_included']}")
    lines += ["", "## MANUAL_GOLD", "", "The JSON contains per-case target/anchor IDs, official truth, exact geometry diagnostics, geometry-kernel result, full production-contract result, and reasons.", "", "| Relation | Cases | Positive | Matched negative | Kernel correct | Full YES | Missing runtime evidence |", "|---|---:|---:|---:|---:|---:|---:|"]
    for relation, value in report["manual_gold"]["by_relation"].items():
        lines.append(f"| {relation} | {value['cases']} | {value['positive']} | {value['matched_negative']} | {value['kernel_correct']} | {value['full_yes']} | {value['missing_runtime_evidence']} |")
    lines += ["", "## PRODUCTION_GEOMETRY_KERNEL", "", "The development kernel is metric-only and does not read point count, provenance, covariance, joint visibility, Qwen, or role fields. It is diagnostic, not a production replacement.", "", "## FULL_PRODUCTION_RELATION_CONTRACT", "", "Full contract uses clean GT-shaped entities with neutral strict geometry metadata and perfect observation-quality/Qwen role metadata; any remaining UNKNOWN is reported separately from the metric kernel.", "", "## METADATA_BINDING", "", "| Mode/relation | Cases | Target correct | Anchor correct | Target accuracy | Anchor accuracy |", "|---|---:|---:|---:|---:|---:|"]
    for key, value in report["binding_calibration"]["by_mode_relation"].items():
        lines.append(f"| {key} | {value['cases']} | {value['target_correct']} | {value['anchor_correct']} | {value['target_accuracy']:.3f} | {value['anchor_accuracy']:.3f} |")
    lines += ["", "## ORDERED_RELATION_DECOMPOSITION", "", "| Relation | Cases | Pure GT correct | Production geometry correct | Binding correct | Binding unresolved |", "|---|---:|---:|---:|---:|---:|"]
    for relation, value in report["ordered_calibration"]["by_relation"].items():
        lines.append(f"| {relation} | {value['cases']} | {value['pure_correct']} | {value['production_correct']} | {value['binding_correct']} | {value['binding_unresolved']} |")
    lines += ["", "## CARDINALITY_SYNTHETIC_VS_REAL", "", report["cardinality_real_comparison"]["synthetic_summary"], "", "## INSTRUCTION_FRAME_AUDIT", "", f"- Numeric trajectory cases: {report['instruction_frame_audit']['cases']}", f"- Same-frame candidate cases with direct GT target match: {report['instruction_frame_audit']['same_frame_candidate']}", "- This audit does not modify InstructionExecutionState.", "", "## DECISION", "", report["decision"], ""]
    return "\n".join(lines)


def run(archive_path: Path, questions_path: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    fast = _load_fast_tool()
    official, shim = _load_official()
    scenes = _load_raw(archive_path, fast)
    _attach_referential_cases(scenes, archive_path, fast)
    self_check = _self_check(scenes, official)
    archive_obb_audit = _archive_obb_audit(scenes, official)
    archive_geometry_matches = (
        archive_obb_audit["objects"] > 0
        and archive_obb_audit["center_exact"] == archive_obb_audit["objects"]
        and archive_obb_audit["size_exact"] == archive_obb_audit["objects"]
        and archive_obb_audit["corner_equivalent"] == archive_obb_audit["objects"]
    )
    report: dict[str, Any] = {
        "schema_version": "vla_oracle_calibration_v1",
        "oracle_trust_status": "PARTIALLY_TRUSTED" if self_check["status"] == "PASS" else ("ARCHIVED_SCENE_GRAPH_GT" if archive_geometry_matches else "INVALID_ADAPTER"),
        "truth_authority": "stored scene_graph.json" if archive_geometry_matches else "undetermined",
        "provenance": {"archive": str(archive_path.resolve()), "official_source": str((OFFICIAL_ROOT / "generate_scene_info.py").resolve()), "official_execution_mode": "exact official relation functions and official scene_graph/bbox_utils.py with isolated dependencies", "scene_count": len(scenes), "questions": str(questions_path.resolve()), "production_modified": False, "unity_ros_launched": False},
            "official_dependency_shim": shim,
            "official_vla_self_check": self_check,
        "relation_direction_audit": _direction_audit(scenes, official),
        "obb_frame_audit": _obb_audit(scenes, official),
        "archive_obb_corner_audit": archive_obb_audit,
        "legacy_adapter_direction_audit": _legacy_adapter_direction_audit(scenes, fast),
    }
    if self_check["status"] != "PASS":
        if archive_geometry_matches:
            report["region_scope_audit"] = {"status": "PASS", "selected_regions": self_check["region_count"], "policy": "stored scene_graph.json region object set"}
            report["negative_set_audit"] = _archive_negative_audit(scenes)
            report["manual_gold"] = _manual_gold(scenes, official, fast)
            report["binding_calibration"] = _binding_calibration(scenes, fast)
            report["ordered_calibration"] = _ordered_calibration(scenes, fast)
            report["cardinality_real_comparison"] = _cardinality_real_comparison()
            report["instruction_frame_audit"] = _instruction_frame_audit(scenes, fast, questions_path)
            report["decision"] = "Exact official bbox utilities reproduce the archived OBB corners, centers, sizes, and Z-up geometry, but the current official generator does not reproduce stored ON/IN relation edges. Use stored scene_graph.json as archive GT authority for offline binding and ordered diagnostics; do not modify production from generator aggregate scores."
        else:
            report["decision"] = "Exact official VLA geometry did not reproduce the archived OBB fields; keep the oracle invalid and make no production repair."
    else:
        report["region_scope_audit"] = {"status": "PASS", "selected_regions": self_check["region_count"], "policy": "official region object set"}
        report["negative_set_audit"] = _region_negative_audit(scenes, official)
        report["manual_gold"] = _manual_gold(scenes, official, fast)
        report["binding_calibration"] = _binding_calibration(scenes, fast)
        report["ordered_calibration"] = _ordered_calibration(scenes, fast)
        report["cardinality_real_comparison"] = _cardinality_real_comparison()
        report["instruction_frame_audit"] = _instruction_frame_audit(scenes, fast, questions_path)
        report["decision"] = "Exact official generator and stored archive relations agree on the checked regions; keep stored scene_graph.json as the archive GT authority for downstream offline diagnostics and do not modify production from aggregate impact numbers."
    (output / "vla_oracle_calibration.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    (output / "VLA_ORACLE_CALIBRATION_REPORT.md").write_text(_markdown(report) + "\n", encoding="utf-8")
    for key, value in report.items():
        if isinstance(value, (dict, list)):
            (output / f"calibration_{key}.json").write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--questions", type=Path, default=REPO_ROOT / "questions" / "questions.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.archive.resolve(), args.questions.resolve(), args.output.resolve())
    print(json.dumps({"status": report["oracle_trust_status"], "self_check": report["official_vla_self_check"]["status"], "output": str(args.output.resolve())}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
