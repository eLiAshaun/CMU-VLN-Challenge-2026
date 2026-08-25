"""Persistent semantic identity over LiDAR-first map-frame observations.

Every real observation remains represented.  Acquisition-level assignment
prevents many-to-one collapse, conservative complete-link reconciliation
merges duplicate tracks, and explicit same-view evidence prevents adjacent
instances from being merged later.
"""

from __future__ import annotations

import copy
import fcntl
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from orchestration.observation_contract import ObservationContract


SCHEMA_VERSION = "scene_memory_v2"
_UNAVAILABLE_METRIC = 1_000_000.0


def _canonical_class_label(value: object) -> str:
    """Normalize only identity-safe detector aliases.

    The selector asks for the physical lamp instance, while open-vocabulary
    grounding may alternate between ``lamp``, ``table lamp``, ``floor lamp``,
    and ``wall lamp`` across viewpoints.  These labels must share one identity
    pool.  We intentionally keep unrelated classes (for example photo/picture)
    separate because collapsing them globally would change task semantics.
    """
    normalized = " ".join(
        str(value or "").strip().lower().replace("_", " ").replace("-", " ").split()
    )
    if (
        normalized in {"lamp", "lamps", "lamp shade", "lampshade"}
        or normalized.endswith(" lamp")
        or normalized.startswith("lamp ")
    ):
        return "lamp"
    return normalized


def _class_labels_compatible(first: object, second: object) -> bool:
    first_label = _canonical_class_label(first)
    second_label = _canonical_class_label(second)
    return bool(first_label and first_label == second_label)


def _center_uncertainty_radius(value: Mapping[str, Any]) -> float:
    """Return a conservative horizontal 2.5-sigma radius in metres."""
    candidates: list[object] = [value.get("center_cov")]
    candidates.extend(
        item.get("center_cov")
        for item in value.get("evidence", ())
        if isinstance(item, Mapping)
    )
    radii: list[float] = []
    for raw in candidates:
        try:
            covariance = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
            continue
        horizontal = 0.5 * (covariance[:2, :2] + covariance[:2, :2].T)
        try:
            maximum = float(max(0.0, np.linalg.eigvalsh(horizontal).max()))
        except np.linalg.LinAlgError:
            continue
        radii.append(2.5 * math.sqrt(maximum))
    if radii:
        return float(max(0.04, min(0.45, max(radii))))
    try:
        extent = np.asarray(value.get("bbox_3d"), dtype=np.float64)
    except (TypeError, ValueError):
        extent = np.zeros(3, dtype=np.float64)
    if extent.shape == (3,) and np.isfinite(extent).all():
        return float(max(0.05, min(0.30, 0.15 * math.hypot(extent[0], extent[1]))))
    return 0.10


def _geometry_vectors(
    value: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        center = np.asarray(value.get("center_3d"), dtype=np.float64)
        extent = np.asarray(value.get("bbox_3d"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        center.shape != (3,)
        or extent.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(extent).all()
        or np.any(extent <= 0.0)
    ):
        return None
    return center, extent


def _vertical_fragment_metrics(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, float | bool]:
    """Measure whether two same-class tracks can be one vertical fragment.

    Sparse LiDAR can observe different vertical parts of one object from
    different views, shifting the centroid even though the map-frame
    footprint is stable.  This is a geometry/evidence pattern, not a class
    property.  Same-view disjoint boxes remain explicit separate-instance
    evidence unless the fragment geometry is strongly supported.
    """
    first_geometry = _geometry_vectors(first)
    second_geometry = _geometry_vectors(second)
    if first_geometry is None or second_geometry is None:
        return {
            "compatible": False,
            "horizontal_distance_m": _UNAVAILABLE_METRIC,
            "vertical_gap_m": _UNAVAILABLE_METRIC,
            "horizontal_limit_m": 0.0,
            "vertical_limit_m": 0.0,
        }
    first_center, first_extent = first_geometry
    second_center, second_extent = second_geometry
    horizontal_distance = float(np.linalg.norm(first_center[:2] - second_center[:2]))
    first_low = float(first_center[2] - 0.5 * first_extent[2])
    first_high = float(first_center[2] + 0.5 * first_extent[2])
    second_low = float(second_center[2] - 0.5 * second_extent[2])
    second_high = float(second_center[2] + 0.5 * second_extent[2])
    vertical_gap = max(0.0, max(first_low, second_low) - min(first_high, second_high))
    uncertainty = _center_uncertainty_radius(first) + _center_uncertainty_radius(second)
    footprint_sum = float(
        math.hypot(first_extent[0], first_extent[1])
        + math.hypot(second_extent[0], second_extent[1])
    )
    horizontal_limit = min(0.52, max(0.24, 0.30 * footprint_sum + uncertainty))
    vertical_limit = min(
        0.60,
        max(0.25, 0.18 * float(first_extent[2] + second_extent[2]) + uncertainty),
    )
    return {
        "compatible": bool(
            horizontal_distance <= horizontal_limit and vertical_gap <= vertical_limit
        ),
        "horizontal_distance_m": horizontal_distance,
        "vertical_gap_m": vertical_gap,
        "horizontal_limit_m": horizontal_limit,
        "vertical_limit_m": vertical_limit,
    }


def _contained_footprint_fragment_metrics(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, float | bool]:
    """Measure an asymmetric partial footprint against a larger track.

    A clipped depth mask at the edge of a large object can produce a precise,
    very small track whose centre is far from the accumulated full-object
    centre.  Centre distance and symmetric size similarity are therefore the
    wrong geometry for this case.  The small centre must instead lie inside
    the larger map-frame box, allowing only the measured centre uncertainty.
    This metric is only a geometric cue; reconciliation still requires
    independent identity evidence and rejects co-visible/distinct tracks.
    """
    first_geometry = _geometry_vectors(first)
    second_geometry = _geometry_vectors(second)
    if first_geometry is None or second_geometry is None:
        return {
            "compatible": False,
            "small_to_large_volume_ratio": _UNAVAILABLE_METRIC,
            "maximum_linear_extent_ratio": _UNAVAILABLE_METRIC,
            "small_center_outside_large_box_m": _UNAVAILABLE_METRIC,
            "containment_uncertainty_m": 0.0,
        }
    first_center, first_extent = first_geometry
    second_center, second_extent = second_geometry
    first_volume = float(np.prod(first_extent))
    second_volume = float(np.prod(second_extent))
    if first_volume >= second_volume:
        large_center, large_extent = first_center, first_extent
        small_center, small_extent = second_center, second_extent
    else:
        large_center, large_extent = second_center, second_extent
        small_center, small_extent = first_center, first_extent
    volume_ratio = float(np.prod(small_extent) / np.prod(large_extent))
    maximum_linear_extent_ratio = float(np.max(small_extent / large_extent))
    outside_by_axis = np.maximum(
        0.0,
        np.abs(small_center - large_center) - 0.5 * large_extent,
    )
    outside_distance = float(np.linalg.norm(outside_by_axis))
    uncertainty = float(
        _center_uncertainty_radius(first) + _center_uncertainty_radius(second)
    )
    return {
        "compatible": bool(
            # A partial mask must be smaller along every measured axis, not
            # merely have a small total volume.  Requiring at most half the
            # full-track extent on each axis excludes peer-sized instances
            # while retaining edge and quadrant crops of one large object.
            maximum_linear_extent_ratio <= 0.5
            and outside_distance <= uncertainty
        ),
        "small_to_large_volume_ratio": volume_ratio,
        "maximum_linear_extent_ratio": maximum_linear_extent_ratio,
        "small_center_outside_large_box_m": outside_distance,
        "containment_uncertainty_m": uncertainty,
    }


def _adjacent_footprint_partition_metrics(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, float | bool]:
    """Measure complementary map-frame partitions of one extended surface.

    Perspective masks of a large object can cover adjacent rather than nested
    parts of its footprint.  Their centroids may then be farther apart than a
    normal association radius even though the measured boxes meet within
    their localization uncertainty.  This is only a geometry cue; identity
    reconciliation additionally requires repeated image-boundary continuity
    and independent appearance/bearing agreement.
    """
    first_geometry = _geometry_vectors(first)
    second_geometry = _geometry_vectors(second)
    if first_geometry is None or second_geometry is None:
        return {
            "compatible": False,
            "footprint_gap_m": _UNAVAILABLE_METRIC,
            "orthogonal_overlap_ratio": 0.0,
            "vertical_gap_m": _UNAVAILABLE_METRIC,
            "continuity_uncertainty_m": 0.0,
        }
    first_center, first_extent = first_geometry
    second_center, second_extent = second_geometry
    first_low = first_center - 0.5 * first_extent
    first_high = first_center + 0.5 * first_extent
    second_low = second_center - 0.5 * second_extent
    second_high = second_center + 0.5 * second_extent
    axis_gaps = np.maximum(
        0.0,
        np.maximum(first_low[:2], second_low[:2])
        - np.minimum(first_high[:2], second_high[:2]),
    )
    axis_overlaps = np.maximum(
        0.0,
        np.minimum(first_high[:2], second_high[:2])
        - np.maximum(first_low[:2], second_low[:2]),
    )
    minimum_extents = np.minimum(first_extent[:2], second_extent[:2])
    overlap_ratios = axis_overlaps / np.maximum(minimum_extents, 1e-6)
    footprint_gap = float(np.linalg.norm(axis_gaps))
    orthogonal_overlap = float(np.max(overlap_ratios))
    vertical_gap = float(max(
        0.0,
        max(float(first_low[2]), float(second_low[2]))
        - min(float(first_high[2]), float(second_high[2])),
    ))
    uncertainty = float(
        _center_uncertainty_radius(first) + _center_uncertainty_radius(second)
    )
    return {
        "compatible": bool(
            footprint_gap <= uncertainty
            and orthogonal_overlap >= 0.35
            and vertical_gap <= max(0.20, uncertainty)
        ),
        "footprint_gap_m": footprint_gap,
        "orthogonal_overlap_ratio": orthogonal_overlap,
        "vertical_gap_m": vertical_gap,
        "continuity_uncertainty_m": uncertainty,
    }


def _finite_vector(value: object, size: int, *, positive: bool = False) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"expected_{size}d_vector")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("nonfinite_geometry")
    if positive and any(item <= 0.0 for item in result):
        raise ValueError("degenerate_geometry")
    return result


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.dist(tuple(float(v) for v in first[:3]), tuple(float(v) for v in second[:3]))


def _safe_descriptor(value: object) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return np.zeros(0, dtype=np.float64)
    try:
        vector = np.asarray([float(item) for item in value], dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros(0, dtype=np.float64)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        return np.zeros(0, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else np.zeros(0, dtype=np.float64)


def _cosine_similarity(first: object, second: object) -> float | None:
    a = _safe_descriptor(first)
    b = _safe_descriptor(second)
    if not len(a) or not len(b) or a.shape != b.shape:
        return None
    return float(max(-1.0, min(1.0, np.dot(a, b))))


def _resolve_object_id(store: Mapping[str, Any], object_id: int) -> int:
    aliases = store.get("object_id_aliases", {})
    current = int(object_id)
    visited: set[int] = set()
    while current not in visited:
        visited.add(current)
        next_value = aliases.get(str(current), aliases.get(current)) if isinstance(aliases, Mapping) else None
        if next_value is None:
            break
        try:
            next_id = int(next_value)
        except (TypeError, ValueError):
            break
        if next_id == current:
            break
        current = next_id
    return current


def _normalized_size_similarity(first: Mapping[str, Any], second: Mapping[str, Any]) -> float | None:
    try:
        a = np.asarray(first.get("bbox_3d"), dtype=np.float64)
        b = np.asarray(second.get("bbox_3d"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if a.shape != (3,) or b.shape != (3,) or np.any(a <= 0.0) or np.any(b <= 0.0):
        return None
    ratio = np.minimum(a, b) / np.maximum(a, b)
    return float(np.prod(np.clip(ratio, 0.0, 1.0)) ** (1.0 / 3.0))


def _load_map_points(path_value: object) -> np.ndarray:
    path = Path(str(path_value or ""))
    if not path.is_file():
        return np.zeros((0, 3), dtype=np.float32)
    try:
        with np.load(path) as bundle:
            for key in ("world_points", "lidar_support_points_map", "points"):
                if key not in bundle:
                    continue
                points = np.asarray(bundle[key], dtype=np.float32).reshape(-1, 3)
                return points[np.isfinite(points).all(axis=1)]
    except (OSError, ValueError, KeyError):
        pass
    return np.zeros((0, 3), dtype=np.float32)


def _pointcloud_geometry_quality(item: Mapping[str, Any]) -> float | None:
    """Compare robust cloud extent with the observation's declared extent."""
    points = _load_map_points(item.get("pointcloud_path"))
    if len(points) < 8:
        return None
    try:
        declared = np.asarray(item.get("bbox_3d"), dtype=np.float64)
    except (TypeError, ValueError):
        return 0.0
    if (
        declared.shape != (3,)
        or not np.isfinite(declared).all()
        or np.any(declared <= 0.0)
    ):
        return 0.0
    robust_extent = np.quantile(points, 0.95, axis=0) - np.quantile(
        points, 0.05, axis=0
    )
    if not np.isfinite(robust_extent).all():
        return 0.0
    # A real object's sparse support may be somewhat larger than its detector
    # box.  A cloud that exceeds both a multiplicative and an absolute
    # tolerance is instead a projection/mask contamination signal.
    tolerance = np.maximum(1.5 * declared, declared + 0.35)
    excess_ratio = np.max(
        np.maximum(0.0, robust_extent - tolerance)
        / np.maximum(tolerance, 1e-6)
    )
    return float(max(0.0, min(1.0, 1.0 - excess_ratio)))


def _normalized_evidence_verdict(item: Mapping[str, Any]) -> str:
    return "_".join(
        str(item.get("qwen_candidate_verdict", ""))
        .strip()
        .lower()
        .replace("-", "_")
        .split()
    )


def _identity_evidence_tier(item: Mapping[str, Any]) -> str:
    """Classify one observation's identity signal without removing it.

    This is deliberately an evidence annotation, not an acceptance gate.  A
    weak observation remains a candidate; it simply cannot, by itself, make a
    spatially incompatible observation the same canonical instance.
    """
    verdict = _normalized_evidence_verdict(item)
    if item.get("qwen_verified") is True or verdict in {
        "target_wins",
        "positive",
        "verified",
        "supported",
    }:
        return "positive"
    if verdict in {
        "confuser_wins",
        "negative",
        "rejected",
        "false_positive",
        "not_target",
    }:
        return "negative"
    if verdict in {"inconclusive", "unknown", "ambiguous"}:
        return "weak"
    probabilities = [
        float(value)
        for value in item.get("qwen_verification_probabilities", ())
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if probabilities and max(probabilities) <= 0.15 and item.get("qwen_verified") is not True:
        return "weak"
    return "neutral"


def _geometry_support_score(item: Mapping[str, Any]) -> float:
    """Score whether an observation is safe to use for canonical geometry."""
    tier = _identity_evidence_tier(item)
    if tier == "negative":
        return 0.0
    try:
        geometry_confidence = max(
            0.0,
            min(1.0, float(item.get("geometry_confidence", 0.0) or 0.0)),
        )
    except (TypeError, ValueError):
        geometry_confidence = 0.0
    cloud_quality = _pointcloud_geometry_quality(item)
    if cloud_quality is not None and cloud_quality < 0.25:
        return 0.0
    if tier == "positive":
        if geometry_confidence >= 0.35:
            return max(0.75, geometry_confidence)
        if cloud_quality is not None:
            return max(0.35, cloud_quality)
        return 0.30
    if tier == "weak":
        # An inconclusive semantic result can still be retained, but it must
        # not contaminate a geometry consensus that has positive support.
        return min(0.20, geometry_confidence * 0.20)
    if geometry_confidence >= 0.35:
        return geometry_confidence
    if str(item.get("pointcloud_path", "")).strip():
        return 0.45
    return 0.0


def _geometry_evidence_items(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    evidence = [
        item for item in value.get("evidence", ())
        if isinstance(item, Mapping)
    ]
    supported = [item for item in evidence if _geometry_support_score(item) >= 0.35]
    # A candidate with no supported observation still needs a provisional
    # position for targeted acquisition.  It is explicitly marked below by
    # the fallback mode; it is not promoted to a trusted canonical geometry.
    return supported or evidence


def _assigned_metric_geometry_items(
    value: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Return assigned observations with usable physical map support.

    Identity assignment is already authoritative at this boundary.  A weak or
    inconclusive semantic verdict must not delete an assigned LiDAR fact from
    the entity's current geometry; bearing-only and physically poor clouds
    remain provisional.
    """
    metric = []
    for item in value.get("evidence", ()):
        if not isinstance(item, Mapping):
            continue
        cloud_quality = _pointcloud_geometry_quality(item)
        if cloud_quality is not None and cloud_quality < 0.25:
            continue
        if len(_load_map_points(item.get("pointcloud_path"))) >= 3:
            metric.append(item)
    return metric or _geometry_evidence_items(value)


def _map_points_from_items(items: Sequence[Mapping[str, Any]]) -> np.ndarray:
    clouds = [
        _load_map_points(item.get("pointcloud_path"))
        for item in items
    ]
    nonempty = [points for points in clouds if len(points)]
    return (
        np.concatenate(nonempty, axis=0)
        if nonempty else np.zeros((0, 3), dtype=np.float32)
    )


def _value_map_points(value: Mapping[str, Any]) -> np.ndarray:
    if value.get("evidence"):
        selected_ids = {
            str(item) for item in value.get("geometry_observation_ids", ())
            if str(item)
        }
        if selected_ids:
            selected = [
                item for item in value.get("evidence", ())
                if isinstance(item, Mapping)
                and str(item.get("observation_id", "")) in selected_ids
            ]
            if selected:
                return _map_points_from_items(selected)
        return _map_points_from_items(_geometry_evidence_items(value))
    return _load_map_points(value.get("pointcloud_path"))


def _identity_requires_separation(value: Mapping[str, Any]) -> bool:
    evidence = [
        item for item in value.get("evidence", ())
        if isinstance(item, Mapping)
    ]
    tiers = [
        _identity_evidence_tier(item)
        for item in evidence
    ] if evidence else [_identity_evidence_tier(value)]
    return "positive" not in tiers and any(
        tier in {"weak", "negative"} for tier in tiers
    )


def _geometry_has_conflict(value: Mapping[str, Any]) -> bool:
    evidence = [
        item for item in value.get("evidence", ())
        if isinstance(item, Mapping)
    ]
    items = evidence or [value]
    if evidence and any(
        _geometry_support_score(item) >= 0.35 for item in evidence
    ):
        return False
    return any(
        str(item.get("pointcloud_path", "")).strip()
        and (
            _pointcloud_geometry_quality(item) is not None
            and float(_pointcloud_geometry_quality(item) or 0.0) < 0.25
        )
        for item in items
    )


def _voxel_keys(points: np.ndarray, voxel_size_m: float) -> set[tuple[int, int, int]]:
    if not len(points):
        return set()
    scaled = np.rint(points / float(voxel_size_m)).astype(np.int64)
    return {tuple(int(value) for value in row) for row in scaled}


def _pointcloud_support_score(
    object_points: np.ndarray,
    observation_points: np.ndarray,
    voxel_size_m: float,
) -> float | None:
    """Return a symmetric map-cloud mismatch in ``[0, 1)``.

    A one-sided containment score incorrectly treats a tiny observation inside
    a large neighbouring track as a perfect match.  Precision and recall are
    therefore combined with an F1 score; zero means mutually consistent support
    and no overlap returns ``None``.
    """
    object_keys = _voxel_keys(object_points, voxel_size_m)
    observation_keys = _voxel_keys(observation_points, voxel_size_m)
    if not object_keys or not observation_keys:
        return None
    expanded_object = {
        (key[0] + dx, key[1] + dy, key[2] + dz)
        for key in object_keys
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    }
    expanded_observation = {
        (key[0] + dx, key[1] + dy, key[2] + dz)
        for key in observation_keys
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    }
    precision = sum(
        key in expanded_object for key in observation_keys
    ) / len(observation_keys)
    recall = sum(
        key in expanded_observation for key in object_keys
    ) / len(object_keys)
    if precision <= 0.0 or recall <= 0.0:
        return None
    similarity = 2.0 * precision * recall / (precision + recall)
    return 1.0 - similarity


def _evidence_map_points(value: Mapping[str, Any]) -> np.ndarray:
    return _map_points_from_items(_geometry_evidence_items(value))


def _publish_query_geometry_contract(
    obj: dict[str, Any],
    *,
    geometry_evidence: Sequence[Mapping[str, Any]],
) -> None:
    """Refresh the existing relation/navigation geometry fields in-place."""
    geometry_ids = list(dict.fromkeys(
        str(item.get("observation_id", ""))
        for item in geometry_evidence
        if str(item.get("observation_id", ""))
    ))
    center = list(obj.get("center_3d", ()))
    extent = list(obj.get("bbox_3d", ()))
    obj["geometry_observation_ids"] = geometry_ids
    if len(center) == 3 and len(extent) == 3:
        obj["position_mean_xyz"] = list(center)
        obj["position_cov_xyz"] = copy.deepcopy(obj.get("center_cov"))
        obj["footprint_obb"] = {
            "center_xy": center[:2],
            "extent_xy": extent[:2],
            "yaw_rad": 0.0,
        }
        obj["footprint_cov"] = copy.deepcopy(obj.get("extent_cov"))
    obj["geometry_revision"] = int(obj.get("instance_version", 0) or 0)


def _refresh_geometry_from_evidence(
    obj: dict[str, Any],
    voxel_size_m: float,
) -> None:
    """Refresh geometry from every currently assigned metric observation."""
    all_evidence = [
        item for item in obj.get("evidence", ())
        if isinstance(item, Mapping)
    ]
    geometry_evidence = _assigned_metric_geometry_items(obj)
    points = _map_points_from_items(geometry_evidence)
    obj["geometry_evidence_policy"] = (
        "assigned_metric_observations" if len(points)
        else "provisional_bearing_only"
    )
    obj["geometry_excluded_observation_ids"] = [
        str(item.get("observation_id", ""))
        for item in all_evidence
        if item not in geometry_evidence and str(item.get("observation_id", ""))
    ]
    if not len(points):
        centers: list[np.ndarray] = []
        weights: list[float] = []
        for item in geometry_evidence:
            center = np.asarray(item.get("center_3d"), dtype=np.float64)
            if center.shape != (3,) or not np.isfinite(center).all():
                continue
            try:
                weight = max(0.05, float(item.get("semantic_probability", 0.5)))
            except (TypeError, ValueError):
                weight = 0.5
            centers.append(center)
            weights.append(weight)
        if centers:
            fused_center = np.average(
                np.stack(centers, axis=0), axis=0, weights=np.asarray(weights)
            )
            obj["center_3d"] = fused_center.astype(float).tolist()
        obj["geometry_point_count"] = 0
        obj["geometry_frame"] = "map"
        obj["frame_id"] = "map"
        obj["covariance_mode"] = "probabilistic"
        obj["center_estimator"] = "bearing_or_sparse_observation"
        _publish_query_geometry_contract(
            obj,
            geometry_evidence=geometry_evidence,
        )
        return

    scaled = np.rint(points / float(voxel_size_m)).astype(np.int64)
    selected: dict[tuple[int, int, int], np.ndarray] = {}
    for key_values, point in zip(scaled, points):
        key = tuple(int(value) for value in key_values)
        selected.setdefault(key, point)
    fused = np.asarray(list(selected.values()), dtype=np.float32)
    low = np.quantile(fused, 0.05, axis=0)
    high = np.quantile(fused, 0.95, axis=0)
    center = 0.5 * (low + high)
    extent = np.maximum(
        high - low, np.full(3, float(voxel_size_m), dtype=np.float32)
    )
    obj["geometry_point_count"] = int(len(fused))
    obj["geometry_frame"] = "map"
    obj["center_3d"] = center.astype(float).tolist()
    obj["bbox_3d"] = extent.astype(float).tolist()
    covariance = (
        np.cov(fused, rowvar=False) / max(1, len(fused))
        if len(fused) >= 2
        else np.eye(3, dtype=np.float64) * float(voxel_size_m) ** 2
    )
    if np.asarray(covariance).shape == (3, 3) and np.isfinite(covariance).all():
        obj["center_cov"] = np.asarray(covariance, dtype=np.float64).tolist()
    obj["extent_cov"] = np.diag(
        np.maximum(extent, float(voxel_size_m)) ** 2
        / max(12.0, float(len(fused)))
    ).astype(float).tolist()
    provenance = {
        str(item.get("center_covariance_provenance", ""))
        for item in geometry_evidence
    }
    statistical = {
        "pointcloud_statistical",
        "multi_view_statistical",
        "fused_statistical",
        "statistical",
    }
    strict_source = bool(
        len(fused) >= 6 and provenance and provenance.issubset(statistical)
    )
    obj["center_covariance_provenance"] = (
        "fused_statistical" if strict_source else "extent_derived_fallback"
    )
    evidence_frames = {
        str(item.get("frame_id", "")).strip().lstrip("/")
        for item in geometry_evidence
        if str(item.get("frame_id", "")).strip()
    }
    if evidence_frames == {"map"}:
        obj["frame_id"] = "map"
    else:
        obj.pop("frame_id", None)
    obj["extent_covariance_provenance"] = "extent_derived"
    obj["covariance_mode"] = "strict" if strict_source else "probabilistic"
    obj["center_estimator"] = "map_support_quantile_box"
    _publish_query_geometry_contract(
        obj,
        geometry_evidence=geometry_evidence,
    )


def _cardinality_box_contains(
    container: Mapping[str, Any],
    point: Sequence[float],
    *,
    tolerance_m: float,
) -> bool:
    geometry = _geometry_vectors(container)
    if geometry is None or len(point) < 3:
        return False
    center, extent = geometry
    try:
        candidate = np.asarray(point[:3], dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        candidate.shape == (3,)
        and np.isfinite(candidate).all()
        and np.all(
            np.abs(candidate - center)
            <= 0.5 * extent + float(tolerance_m)
        )
    )


def _cardinality_residual_ratio(
    candidate: Mapping[str, Any],
    atomic_peers: Sequence[Mapping[str, Any]],
    *,
    voxel_size_m: float,
) -> float | None:
    candidate_keys = _voxel_keys(
        _value_map_points(candidate), voxel_size_m
    )
    peer_points = [
        points
        for value in atomic_peers
        if len(points := _value_map_points(value))
    ]
    if not candidate_keys or not peer_points:
        return None
    peer_keys = _voxel_keys(
        np.concatenate(peer_points, axis=0), voxel_size_m
    )
    expanded_peer_keys = {
        (key[0] + dx, key[1] + dy, key[2] + dz)
        for key in peer_keys
        for dx in range(-2, 3)
        for dy in range(-2, 3)
        for dz in range(-2, 3)
    }
    covered = sum(key in expanded_peer_keys for key in candidate_keys)
    return 1.0 - float(covered / len(candidate_keys))


def _cardinality_extent_volume(value: Mapping[str, Any]) -> float | None:
    geometry = _geometry_vectors(value)
    if geometry is None:
        return None
    volume = float(np.prod(geometry[1]))
    return volume if math.isfinite(volume) and volume > 0.0 else None


def _cardinality_containment_fraction(
    container: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    tolerance_m: float,
) -> float | None:
    geometry = _geometry_vectors(container)
    points = _value_map_points(candidate)
    if geometry is None or not len(points):
        return None
    center, extent = geometry
    contained = np.all(
        np.abs(points - center)
        <= 0.5 * extent + float(tolerance_m),
        axis=1,
    )
    return float(np.mean(contained))


def _materialize_cardinality_roles(
    objects: Sequence[dict[str, Any]],
    cannot_links: Sequence[Mapping[str, Any]],
    *,
    voxel_size_m: float,
) -> dict[str, Any]:
    """Derive representation roles without changing identity membership."""
    lookup = {int(value["object_id"]): value for value in objects}
    for obj in lookup.values():
        obj["cardinality_role"] = "UNKNOWN_CARDINALITY"
        obj["cardinality_evidence"] = {}
    active = {
        object_id: value
        for object_id, value in lookup.items()
        if value.get("status") in {"tentative", "confirmed"}
    }
    hard_pairs = {
        tuple(sorted(int(item) for item in value.get("entity_ids", ())))
        for value in cannot_links
        if isinstance(value, Mapping)
        and len(value.get("entity_ids", ())) == 2
        and value.get("hard") is not False
    }
    observation_owner = {
        str(evidence.get("observation_id", "")): object_id
        for object_id, obj in active.items()
        for evidence in obj.get("evidence", ())
        if isinstance(evidence, Mapping)
        and str(evidence.get("observation_id", ""))
    }
    reliable_provenance = {
        "pointcloud_statistical",
        "multi_view_statistical",
        "fused_statistical",
        "statistical",
    }

    for object_id, obj in active.items():
        evidence = [
            value for value in obj.get("evidence", ())
            if isinstance(value, Mapping)
        ]
        observation_ids = list(dict.fromkeys(
            str(value.get("observation_id", ""))
            for value in evidence
            if str(value.get("observation_id", ""))
        ))
        explicit_coverage = any(
            value.get("coverage_member_observation_ids")
            for value in evidence
        )
        # Cardinality describes the represented visual instance, not whether
        # LiDAR happened to hit it.  A normal single-instance proposal is an
        # atomic hypothesis even when its geometry remains bearing-only;
        # explicit coverage stays unresolved until the peer evidence below can
        # prove it is an aggregate.
        single_instance = bool(observation_ids and not explicit_coverage)
        obj["cardinality_role"] = (
            "ATOMIC" if single_instance else "UNKNOWN_CARDINALITY"
        )
        if single_instance:
            obj["cardinality_evidence"] = {
                "single_instance_observation_ids": observation_ids,
                "evidence_policy": "instance_proposal_without_coverage_members",
            }

    for object_id, obj in active.items():
        if obj["cardinality_role"] not in {
            "ATOMIC", "UNKNOWN_CARDINALITY"
        }:
            continue
        provenance_peer_ids = {
            observation_owner[str(observation_id)]
            for evidence in obj.get("evidence", ())
            if isinstance(evidence, Mapping)
            for observation_id in evidence.get(
                "coverage_member_observation_ids", ()
            )
            if str(observation_id) in observation_owner
            and observation_owner[str(observation_id)] != object_id
        }
        provenance_peers = [
            active[peer_id]
            for peer_id in provenance_peer_ids
            if peer_id in active
            and active[peer_id].get("cardinality_role") == "ATOMIC"
            and _class_labels_compatible(
                obj.get("class_label"), active[peer_id].get("class_label")
            )
        ]
        provenance_peer_ids = {
            int(value["object_id"]) for value in provenance_peers
        }
        provenance_distinct_pair = any(
            first in provenance_peer_ids and second in provenance_peer_ids
            for first, second in hard_pairs
        )
        if len(provenance_peer_ids) >= 2 and provenance_distinct_pair:
            obj["cardinality_role"] = "AGGREGATE_COVERAGE"
            obj["cardinality_evidence"] = {
                "covered_atomic_entity_ids": sorted(provenance_peer_ids),
                "evidence_policy": "same_station_coverage_provenance",
            }
            continue
        atomic_peers = [
            value
            for peer_id, value in active.items()
            if peer_id != object_id
            and value.get("cardinality_role") == "ATOMIC"
            and _class_labels_compatible(
                obj.get("class_label"), value.get("class_label")
            )
            and _cardinality_box_contains(
                obj,
                value.get("center_3d", ()),
                tolerance_m=2.0 * voxel_size_m,
            )
        ]
        peer_ids = {int(value["object_id"]) for value in atomic_peers}
        has_distinct_pair = any(
            first in peer_ids and second in peer_ids
            for first, second in hard_pairs
        )
        residual_ratio = _cardinality_residual_ratio(
            obj, atomic_peers, voxel_size_m=voxel_size_m
        )
        if (
            len(peer_ids) >= 2
            and has_distinct_pair
            and residual_ratio is not None
            and residual_ratio <= 0.20
        ):
            obj["cardinality_role"] = "AGGREGATE_COVERAGE"
            obj["cardinality_evidence"] = {
                "covered_atomic_entity_ids": sorted(peer_ids),
                "residual_ratio": float(residual_ratio),
            }

    ordered_candidates = sorted(
        active.items(),
        key=lambda item: (
            _cardinality_extent_volume(item[1]) or float("inf"),
            item[0],
        ),
    )
    for object_id, obj in ordered_candidates:
        if obj.get("cardinality_role") not in {
            "ATOMIC", "UNKNOWN_CARDINALITY"
        }:
            continue
        candidate_volume = _cardinality_extent_volume(obj)
        if (
            candidate_volume is None
            or int(obj.get("geometry_point_count", 0) or 0) < 6
            or str(obj.get("center_covariance_provenance", ""))
            not in reliable_provenance
        ):
            continue
        candidate_acquisitions = {
            str(value.get("acquisition_id", ""))
            for value in obj.get("evidence", ())
            if isinstance(value, Mapping)
            and str(value.get("acquisition_id", ""))
        }
        compatible_containers = []
        for peer_id, peer in active.items():
            if (
                peer_id == object_id
                or peer.get("cardinality_role") != "ATOMIC"
                or not _class_labels_compatible(
                    obj.get("class_label"), peer.get("class_label")
                )
                or tuple(sorted((object_id, peer_id))) in hard_pairs
                or int(peer.get("geometry_point_count", 0) or 0) < 6
                or str(peer.get("center_covariance_provenance", ""))
                not in reliable_provenance
            ):
                continue
            peer_volume = _cardinality_extent_volume(peer)
            containment_tolerance = max(
                2.0 * voxel_size_m,
                min(
                    0.25,
                    _center_uncertainty_radius(obj),
                    _center_uncertainty_radius(peer),
                ),
            )
            if (
                peer_volume is None
                or candidate_volume > 0.35 * peer_volume
                or not _cardinality_box_contains(
                    peer,
                    obj.get("center_3d", ()),
                    tolerance_m=containment_tolerance,
                )
            ):
                continue
            shared_acquisitions = sorted(
                candidate_acquisitions.intersection(
                    str(value.get("acquisition_id", ""))
                    for value in peer.get("evidence", ())
                    if isinstance(value, Mapping)
                )
            )
            if not shared_acquisitions:
                continue
            # A broad proposal that also spans an entity known distinct from
            # the candidate is coverage, not a complete single-object parent.
            spans_distinct_peer = any(
                other_id not in {object_id, peer_id}
                and _class_labels_compatible(
                    obj.get("class_label"), other.get("class_label")
                )
                and tuple(sorted((object_id, other_id))) in hard_pairs
                and _cardinality_box_contains(
                    peer,
                    other.get("center_3d", ()),
                    tolerance_m=containment_tolerance,
                )
                for other_id, other in active.items()
            )
            if spans_distinct_peer:
                continue
            containment = _cardinality_containment_fraction(
                peer,
                obj,
                tolerance_m=containment_tolerance,
            )
            if containment is None or containment < 0.65:
                continue
            compatible_containers.append({
                "object_id": peer_id,
                "extent_volume": peer_volume,
                "candidate_extent_ratio": candidate_volume / peer_volume,
                "point_containment_fraction": containment,
                "containment_tolerance_m": containment_tolerance,
                "shared_acquisition_ids": shared_acquisitions,
                "geometry_provenance": str(
                    peer.get("center_covariance_provenance", "")
                ),
            })
        if compatible_containers:
            complete = max(
                compatible_containers,
                key=lambda value: (value["extent_volume"], -value["object_id"]),
            )
            obj["cardinality_role"] = "PARTIAL_FRAGMENT"
            obj["cardinality_evidence"] = {
                "complete_entity_id": int(complete["object_id"]),
                "candidate_extent_ratio": float(
                    complete["candidate_extent_ratio"]
                ),
                "point_containment_fraction": float(
                    complete["point_containment_fraction"]
                ),
                "containment_tolerance_m": float(
                    complete["containment_tolerance_m"]
                ),
                "shared_acquisition_ids": list(
                    complete["shared_acquisition_ids"]
                ),
                "geometry_provenance": complete["geometry_provenance"],
            }

    roles = {
        role: sorted(
            object_id
            for object_id, obj in lookup.items()
            if obj.get("cardinality_role") == role
        )
        for role in (
            "ATOMIC",
            "PARTIAL_FRAGMENT",
            "AGGREGATE_COVERAGE",
            "UNKNOWN_CARDINALITY",
        )
    }
    return {
        "atomic_entity_ids": roles["ATOMIC"],
        "partial_fragment_entity_ids": roles["PARTIAL_FRAGMENT"],
        "aggregate_coverage_entity_ids": roles["AGGREGATE_COVERAGE"],
        "unknown_cardinality_entity_ids": roles["UNKNOWN_CARDINALITY"],
    }


def _empty_store() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "version": 0,
        "geometry_revision": 0,
        "semantic_revision": 0,
        "next_object_id": 0,
        "objects": [],
        "object_id_aliases": {},
        "distinct_object_pairs": [],
        "identity_merge_events": [],
        "identity_resolution_events": [],
        "identity_constraints": {
            "must_link": [],
            "distinct_evidence": [],
            "cannot_link": [],
        },
        "identity_revision": 0,
        "last_observation_transaction": None,
        "observation_ledger": {
            "schema_version": "observation_ledger_v1",
            "records": [],
        },
        "acquisition_traces": [],
        "last_acquisition_trace": None,
        "identity_ambiguity_groups": [],
        "ambiguous_observations": [],
        "relation_evidence": {
            "schema_version": "persistent_relation_evidence_v1",
            "records": {},
            "relation_summaries": {},
            "relation_revision": 0,
            "last_updated_monotonic": 0.0,
        },
        "mast3r_reconstruction": None,
        "viewpoint_history": [],
        "terrain_memory": None,
        "observation_frontier_regions": [],
        "traversable_point_count": 0,
    }


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty_store()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("scene_memory_schema_mismatch")
    payload.setdefault("object_id_aliases", {})
    payload.setdefault("distinct_object_pairs", [])
    payload.setdefault("identity_merge_events", [])
    payload.setdefault("identity_resolution_events", [])
    payload.setdefault(
        "identity_constraints",
        {"must_link": [], "distinct_evidence": [], "cannot_link": []},
    )
    if isinstance(payload.get("identity_constraints"), dict):
        constraints = payload["identity_constraints"]
        constraints.setdefault("must_link", [])
        constraints.setdefault(
            "distinct_evidence",
            copy.deepcopy(payload.get("distinct_object_pairs", ())),
        )
        # Kept only as an empty legacy field.  Distinct evidence is not a
        # cannot-link authority and no active resolver reads this name.
        constraints["cannot_link"] = []
    payload.setdefault("identity_revision", 0)
    payload.setdefault("geometry_revision", 0)
    payload.setdefault("last_observation_transaction", None)
    payload.setdefault(
        "observation_ledger",
        {"schema_version": "observation_ledger_v1", "records": []},
    )
    payload.setdefault("acquisition_traces", [])
    payload.setdefault("last_acquisition_trace", None)
    payload.setdefault("identity_ambiguity_groups", [])
    payload.setdefault("ambiguous_observations", [])
    payload.setdefault("relation_evidence", {
        "schema_version": "persistent_relation_evidence_v1",
        "records": {},
        "relation_summaries": {},
        "relation_revision": 0,
        "last_updated_monotonic": 0.0,
    })
    payload.setdefault("viewpoint_history", [])
    return payload


def _identity_clusters(store: Mapping[str, Any]) -> list[dict[str, Any]]:
    members_by_canonical: dict[int, set[int]] = {}
    for obj in store.get("objects", ()):
        object_id = int(obj.get("object_id", -1))
        if object_id >= 0:
            members_by_canonical.setdefault(object_id, set()).add(object_id)
    aliases = store.get("object_id_aliases", {})
    if isinstance(aliases, Mapping):
        for raw_alias in aliases:
            try:
                alias_id = int(raw_alias)
            except (TypeError, ValueError):
                continue
            canonical_id = _resolve_object_id(store, alias_id)
            members_by_canonical.setdefault(canonical_id, set()).add(alias_id)
            members_by_canonical[canonical_id].add(canonical_id)
    lookup = {int(obj.get("object_id", -1)): obj for obj in store.get("objects", ())}
    clusters = []
    for canonical_id, members in sorted(members_by_canonical.items()):
        obj = lookup.get(canonical_id, {})
        clusters.append({
            "canonical_object_id": canonical_id,
            "member_object_ids": sorted(members),
            "class_label": str(obj.get("class_label", "")),
            "physical_status": str(obj.get("physical_status", obj.get("status", ""))),
            "semantic_status": str(obj.get("semantic_status", "unverified")),
            "evidence_count": len(obj.get("evidence", ())),
        })
    return clusters


def load_scene_memory_snapshot(
    store_path: Path,
    *,
    acquisition_id: str = "memory-read",
) -> dict[str, Any]:
    """Load one immutable episode snapshot without incrementing its version."""
    store_path = store_path.resolve()
    lock_path = store_path.with_suffix(store_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        store = _load(store_path)
        snapshot = {
            "schema_version": "scene_memory_snapshot_v1",
            "scene_version": int(store.get("version", 0)),
            "semantic_revision": int(store.get("semantic_revision", 0)),
            "identity_revision": int(store.get("identity_revision", 0)),
            "geometry_version": int(store.get("geometry_revision", 0)),
            "acquisition_id": str(acquisition_id),
            "frame": "map",
            "objects": copy.deepcopy(store.get("objects", ())),
            "object_id_aliases": copy.deepcopy(store.get("object_id_aliases", {})),
            "identity_clusters": _identity_clusters(store),
            "identity_merge_events": copy.deepcopy(store.get("identity_merge_events", ())[-24:]),
            "identity_resolution_events": copy.deepcopy(
                store.get("identity_resolution_events", ())[-24:]
            ),
            "identity_constraints": copy.deepcopy(
                store.get("identity_constraints", {})
            ),
            "identity_ambiguity_groups": copy.deepcopy(
                store.get("identity_ambiguity_groups", ())
            ),
            "distinct_object_pairs": copy.deepcopy(store.get("distinct_object_pairs", ())[-128:]),
            "associated_object_ids": [],
            "ambiguous_observations": copy.deepcopy(
                store.get("ambiguous_observations", ())
            ),
            "relation_evidence": copy.deepcopy(
                store.get("relation_evidence", {})
            ),
            "relation_revision": int(
                store.get("relation_evidence", {}).get("relation_revision", 0)
            ),
            "mast3r_reconstruction": copy.deepcopy(
                store.get("mast3r_reconstruction")
            ),
            "rejected_observations": [],
            "last_observation_transaction": copy.deepcopy(
                store.get("last_observation_transaction")
            ),
            "observation_ledger": copy.deepcopy(
                store.get("observation_ledger", {})
            ),
            "observation_ledger_size": len(
                store.get("observation_ledger", {}).get("records", ())
            ),
            "last_acquisition_trace": copy.deepcopy(
                store.get("last_acquisition_trace")
            ),
            "viewpoint_history": copy.deepcopy(
                store.get("viewpoint_history", ())
            ),
            "terrain_memory": copy.deepcopy(store.get("terrain_memory")),
            "observation_frontier_regions": copy.deepcopy(
                store.get("observation_frontier_regions", ())
            ),
            "traversable_point_count": int(
                store.get("traversable_point_count", 0)
            ),
            "scene_memory_path": str(store_path),
            "geometry_reconstruction_ids": [],
            # SceneMemory is intentionally an open world.  This is a
            # task-independent fact; numerical answer readiness and relation
            # selector openness are evaluated by their own task consumers.
            "scene_memory_open": True,
            "scene_memory_state": "OPEN_PERSISTENT_TRACKS",
        }
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return snapshot


def _terrain_observation_facts(
    terrain_path: Path | None,
    *,
    obstacle_threshold: float,
    voxel_size_m: float,
    viewpoint_position_map: Sequence[float] | None = None,
    current_terrain_path: Path | None = None,
) -> dict[str, Any]:
    """Extract task-independent traversability/frontier facts from the terrain map."""
    terrain = np.zeros((0, 4), dtype=np.float32)
    if terrain_path is not None and terrain_path.is_file():
        try:
            if terrain_path.suffix == ".npz":
                with np.load(terrain_path) as bundle:
                    terrain = np.asarray(bundle["points"])
            else:
                terrain = np.asarray(np.load(terrain_path))
        except (OSError, ValueError, KeyError):
            terrain = np.zeros((0, 4), dtype=np.float32)
    if terrain.ndim != 2 or terrain.shape[1] < 4:
        return {"observation_frontier_regions": [], "traversable_point_count": 0}
    finite = np.isfinite(terrain[:, :4]).all(axis=1)
    traversable = terrain[finite & (terrain[:, 3] < float(obstacle_threshold))]
    voxel = float(voxel_size_m)
    all_keys = {
        (int(round(float(point[0]) / voxel)), int(round(float(point[1]) / voxel)))
        for point in terrain[finite]
    }
    traversable_by_key = {
        (
            int(round(float(point[0]) / voxel)),
            int(round(float(point[1]) / voxel)),
        ): point
        for point in traversable
    }
    reachable_keys = set(traversable_by_key)
    if viewpoint_position_map is not None and traversable_by_key:
        origin_x = float(viewpoint_position_map[0])
        origin_y = float(viewpoint_position_map[1])
        origin_candidates = set(traversable_by_key)
        if current_terrain_path is not None and current_terrain_path.is_file():
            try:
                current_terrain = np.asarray(np.load(current_terrain_path))
                if current_terrain.ndim == 2 and current_terrain.shape[1] >= 4:
                    current_finite = np.isfinite(current_terrain[:, :4]).all(axis=1)
                    current_traversable = current_terrain[
                        current_finite
                        & (current_terrain[:, 3] < float(obstacle_threshold))
                    ]
                    current_keys = {
                        (
                            int(round(float(point[0]) / voxel)),
                            int(round(float(point[1]) / voxel)),
                        )
                        for point in current_traversable
                    }
                    current_candidates = current_keys & set(traversable_by_key)
                    if current_candidates:
                        origin_candidates = current_candidates
            except (OSError, ValueError, KeyError):
                origin_candidates = set(traversable_by_key)
        origin_key = min(
            origin_candidates,
            key=lambda key: (
                float(traversable_by_key[key][0]) - origin_x
            ) ** 2 + (
                float(traversable_by_key[key][1]) - origin_y
            ) ** 2,
        )
        reachable_keys = {origin_key}
        pending = [origin_key]
        while pending:
            key = pending.pop()
            for dx, dy in (
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ):
                neighbor = (key[0] + dx, key[1] + dy)
                if (
                    neighbor in traversable_by_key
                    and neighbor not in reachable_keys
                ):
                    reachable_keys.add(neighbor)
                    pending.append(neighbor)
    observed_coverage_keys = {
        (key[0] + dx, key[1] + dy)
        for key in all_keys
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    }
    frontier_by_key: dict[tuple[int, int], list[float]] = {}
    for point in traversable:
        key = (
            int(round(float(point[0]) / voxel)),
            int(round(float(point[1]) / voxel)),
        )
        if key in reachable_keys and any(
            (key[0] + dx, key[1] + dy) not in observed_coverage_keys
            for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))
        ):
            frontier_by_key[key] = [float(point[0]), float(point[1])]
    remaining = set(frontier_by_key)
    frontier_regions: list[dict[str, Any]] = []
    while remaining:
        seed = min(remaining)
        component = {seed}
        pending = [seed]
        remaining.remove(seed)
        while pending:
            key = pending.pop()
            for dx, dy in (
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ):
                neighbor = (key[0] + dx, key[1] + dy)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        ordered_keys = sorted(component)
        frontier_regions.append({
            "region_id": (
                f"frontier-component:{ordered_keys[0][0]}:"
                f"{ordered_keys[0][1]}"
            ),
            "points_xy": [frontier_by_key[key] for key in ordered_keys],
        })
    return {
        "observation_frontier_regions": frontier_regions,
        "traversable_point_count": int(len(reachable_keys)),
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_scene_relation_evidence(
    store_path: Path,
    relation_evidence: Mapping[str, Any],
) -> None:
    """Commit cross-view relation evidence into the single world store."""
    store_path = store_path.resolve()
    lock_path = store_path.with_suffix(store_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        store = _load(store_path)
        store["relation_evidence"] = copy.deepcopy(dict(relation_evidence))
        _atomic_write(store_path, store)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _validate_observation(raw: Mapping[str, Any]) -> dict[str, Any]:
    try:
        contract = ObservationContract.from_mapping(raw)
    except ValueError as exc:
        raise ValueError(f"observation_contract_invalid:{exc}") from exc
    if raw.get("frame") != "map" or raw.get("world_aligned") is not True:
        raise ValueError("observation_not_in_map_frame")
    observed_class_label = " ".join(
        str(raw.get("canonical_class", raw.get("class_label", "")))
        .strip().lower().replace("_", " ").replace("-", " ").split()
    )
    class_name = _canonical_class_label(observed_class_label)
    observation_id = str(raw.get("observation_id", "")).strip()
    if not class_name or not observation_id:
        raise ValueError("observation_identity_incomplete")
    center = _finite_vector(contract.world_center, 3)
    bbox = _finite_vector(contract.world_extent, 3, positive=True)
    viewpoint = _finite_vector(raw.get("viewpoint_position_map"), 3)
    probability = raw.get("semantic_probability")
    if probability is None or not math.isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
        raise ValueError("semantic_probability_invalid")
    return {
        "observation_id": observation_id,
        "frame": "map",
        "world_aligned": True,
        "station_id": contract.station_id,
        "acquisition_id": contract.acquisition_id,
        "timestamp": contract.timestamp,
        "timestamp_unix": contract.timestamp,
        "contract_schema_version": "observation_contract_v1",
        "view_id": contract.view_id,
        "camera_id": contract.camera_id,
        "native_image_width": contract.native_image_width,
        "native_image_height": contract.native_image_height,
        "mask_native_width": int(raw.get("mask_native_width", contract.native_image_width)),
        "mask_native_height": int(raw.get("mask_native_height", contract.native_image_height)),
        "camera_model": copy.deepcopy(contract.camera_model),
        "mask_path": contract.mask_path,
        "mask_coordinate_frame": contract.mask_coordinate_frame,
        "bbox_xyxy": list(contract.bbox_xyxy),
        "depth_support": copy.deepcopy(contract.depth_support),
        "T_world_camera": copy.deepcopy(contract.T_world_camera),
        "world_center": list(contract.world_center),
        "world_points_path": contract.world_points_path,
        "world_extent": list(contract.world_extent),
        "semantic_label": contract.semantic_label,
        "semantic_confidence": contract.semantic_confidence,
        "appearance_feature": list(contract.appearance_feature),
        "canonical_object_id": contract.canonical_object_id,
        "identity_hypotheses": [dict(item) for item in contract.identity_hypotheses],
        "frame_id": "map",
        "class_label": class_name,
        "observed_class_label": observed_class_label or class_name,
        "center_3d": center,
        "bbox_3d": bbox,
        "measured_center_3d": [
            float(value)
            for value in raw.get("measured_center_3d", center)
        ],
        "measured_bbox_3d": [
            float(value)
            for value in raw.get("measured_bbox_3d", bbox)
        ],
        "viewpoint_position_map": viewpoint,
        "optical_center_group": str(raw.get("optical_center_group", "")),
        "semantic_probability": float(probability),
        "bearing_observation": copy.deepcopy(raw.get("bearing_observation")),
        "support_parent_id": raw.get("support_parent_id"),
        "support_parent_probability": raw.get("support_parent_probability"),
        "footprint_yaw_rad": raw.get("footprint_yaw_rad", 0.0),
        "pointcloud_path": str(raw.get("pointcloud_path", "")),
        "panorama_mask_path": str(raw.get("panorama_mask_path", "")),
        "panorama_mask_coordinate_frame": str(
            raw.get("panorama_mask_coordinate_frame", "")
        ),
        "panorama_image_path": str(raw.get("panorama_image_path", "")),
        "panorama_mask_projection": copy.deepcopy(
            raw.get("panorama_mask_projection", {})
        ),
        "panorama_pixel_support_bbox_xyxy": copy.deepcopy(
            raw.get("panorama_pixel_support_bbox_xyxy", [])
        ),
        "panorama_pixel_support_area": max(
            0, int(raw.get("panorama_pixel_support_area", 0) or 0)
        ),
        "cannot_link_observation_ids": sorted({
            str(value) for value in raw.get("cannot_link_observation_ids", ())
            if str(value).strip() and str(value) != observation_id
        }),
        "coverage_member_observation_ids": sorted({
            str(value)
            for value in raw.get("coverage_member_observation_ids", ())
            if str(value).strip() and str(value) != observation_id
        }),
        "representative_view_id": str(raw.get("representative_view_id", "")),
        "representative_view_image": str(
            raw.get("representative_view_image", "")
        ),
        "representative_bbox_xyxy": copy.deepcopy(
            raw.get("representative_bbox_xyxy")
        ),
        "relation_roi_image_path": str(
            raw.get("relation_roi_image_path", "")
        ),
        "relation_roi_view_id": str(raw.get("relation_roi_view_id", "")),
        "relation_roi_subject_bbox_xyxy_normalized": copy.deepcopy(
            raw.get("relation_roi_subject_bbox_xyxy_normalized")
        ),
        "relation_roi_anchor_bbox_xyxy_normalized": copy.deepcopy(
            raw.get("relation_roi_anchor_bbox_xyxy_normalized")
        ),
        "relation_roi_anchor_class": str(
            raw.get("relation_roi_anchor_class", "")
        ),
        "relation_roi_anchor_classes": [
            str(value) for value in raw.get("relation_roi_anchor_classes", ())
        ],
        "relation_roi_anchor_bboxes_xyxy_normalized": copy.deepcopy(
            raw.get("relation_roi_anchor_bboxes_xyxy_normalized", [])
        ),
        "relation_roi_anchor_binding_keys": [
            str(value)
            for value in raw.get("relation_roi_anchor_binding_keys", ())
        ],
        "relation_roi_predicate": str(
            raw.get("relation_roi_predicate", "")
        ),
        "relation_binding_context_bbox_xyxy_normalized": copy.deepcopy(
            raw.get("relation_binding_context_bbox_xyxy_normalized")
        ),
        "reconstruction_id": str(raw.get("reconstruction_id", "")),
        "source_cam2w_maps": copy.deepcopy(raw.get("source_cam2w_maps", [])),
        "source_intrinsics": copy.deepcopy(raw.get("source_intrinsics", [])),
        "source_view_ids": list(raw.get("source_view_ids", ())),
        "source_proposal_binding_keys": [
            str(value)
            for value in raw.get("source_proposal_binding_keys", ())
        ],
        "proposal_verification_by_key": {
            str(key): (float(value) if isinstance(value, (int, float)) else None)
            for key, value in dict(
                raw.get("proposal_verification_by_key", {})
            ).items()
        },
        "qwen_verification_probabilities": [
            float(value)
            for value in raw.get("qwen_verification_probabilities", ())
            if isinstance(value, (int, float))
        ],
        "qwen_verified": raw.get("qwen_verified") is True,
        "qwen_candidate_verdict": str(
            raw.get("qwen_candidate_verdict", "unavailable")
        ),
        "geometry_confidence": float(raw.get("median_confidence", 0.0)),
        "geometry_source": str(raw.get("geometry_source", "")),
        "lidar_support_count": int(raw.get("lidar_support_count", 0) or 0),
        "center_cov": copy.deepcopy(raw.get("center_cov")),
        "extent_cov": copy.deepcopy(raw.get("extent_cov")),
        "center_covariance_provenance": str(
            raw.get(
                "center_covariance_provenance",
                raw.get("center_cov_source", ""),
            )
        ),
        "extent_covariance_provenance": str(
            raw.get(
                "extent_covariance_provenance",
                raw.get("extent_cov_source", ""),
            )
        ),
        "covariance_mode": str(raw.get("covariance_mode", "probabilistic")),
        "view_center_dispersion_m": float(raw.get("view_center_dispersion_m", 0.0)),
        "registered_scan_median_distance_m": raw.get("registered_scan_median_distance_m"),
        "registered_scan_inlier_fraction_0_25m": float(
            raw.get("registered_scan_inlier_fraction_0_25m", 0.0)
        ),
        "appearance_descriptor": _safe_descriptor(
            raw.get("appearance_descriptor")
        ).astype(float).tolist(),
        "appearance_quality": max(
            0.0, min(1.0, float(raw.get("appearance_quality", 0.0) or 0.0))
        ),
        "mask_pixel_count": max(0, int(raw.get("mask_pixel_count", 0) or 0)),
        "station_fused_member_count": max(
            1, int(raw.get("station_fused_member_count", 1) or 1)
        ),
        "source_detection_ids": [
            str(value) for value in raw.get("source_detection_ids", ())
            if str(value).strip()
        ],
        "source_view_boxes": _source_view_boxes(raw),
    }


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(first, second))


def _representative_ray(
    value: Mapping[str, Any],
) -> tuple[list[float], list[float], float] | None:
    """Recover one calibrated map-frame observation ray and angular support."""
    view_id = str(value.get("representative_view_id", ""))
    view_ids = [str(item) for item in value.get("source_view_ids", ())]
    try:
        index = view_ids.index(view_id)
        pose = value["source_cam2w_maps"][index]
        intrinsics = value["source_intrinsics"][index]
        bbox = [float(item) for item in value["representative_bbox_xyxy"]]
        fx = float(intrinsics[0][0])
        fy = float(intrinsics[1][1])
        cx = float(intrinsics[0][2])
        cy = float(intrinsics[1][2])
        rotation = [[float(item) for item in row[:3]] for row in pose[:3]]
        origin = [float(pose[axis][3]) for axis in range(3)]
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if fx <= 0.0 or fy <= 0.0 or len(bbox) != 4:
        return None
    u = 0.5 * (bbox[0] + bbox[2])
    v = 0.5 * (bbox[1] + bbox[3])
    camera_ray = [(u - cx) / fx, (v - cy) / fy, 1.0]
    direction = [
        sum(rotation[row][column] * camera_ray[column] for column in range(3))
        for row in range(3)
    ]
    norm = math.sqrt(_dot(direction, direction))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    direction = [item / norm for item in direction]
    half_angular_extent = math.atan(math.hypot(
        0.5 * (bbox[2] - bbox[0]) / fx,
        0.5 * (bbox[3] - bbox[1]) / fy,
    ))
    if not math.isfinite(half_angular_extent) or half_angular_extent <= 0.0:
        return None
    return origin, direction, half_angular_extent


def _registered_geometry_reliable(value: Mapping[str, Any]) -> bool:
    """Whether the local centre is independently supported by the 0.25 m scan."""
    try:
        median = float(value.get("registered_scan_median_distance_m"))
        inlier = float(value.get("registered_scan_inlier_fraction_0_25m"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(median) and math.isfinite(inlier) and median <= 0.25 and inlier >= 0.5


def _ray_depth_pair_allowed(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    """Reject a ray-only match when two reliable 3D supports are disjoint."""
    if not (
        _registered_geometry_reliable(first)
        and _registered_geometry_reliable(second)
    ):
        return True
    return _distance(first["center_3d"], second["center_3d"]) <= _association_limit(
        first, second
    )


def _ray_support_score(
    obj: Mapping[str, Any], observation: Mapping[str, Any]
) -> float | None:
    """Return image-support-normalized ray miss for independent viewpoints.

    A score at or below one means the two calibrated 2D supports intersect in
    front of both cameras.  This lets independent rays repair a bad monocular
    depth without widening the ordinary 3D association radius.
    """
    second = _representative_ray(observation)
    if second is None:
        return None
    best: float | None = None
    for evidence in obj.get("evidence", ()):
        if str(evidence.get("optical_center_group", "")) == str(
            observation.get("optical_center_group", "")
        ):
            continue
        first = _representative_ray(evidence)
        if first is None:
            continue
        if not _ray_depth_pair_allowed(evidence, observation):
            continue
        first_origin, first_direction, first_extent = first
        second_origin, second_direction, second_extent = second
        cosine = _dot(first_direction, second_direction)
        denominator = 1.0 - cosine * cosine
        if denominator <= 1e-8:
            continue
        offset = [
            first_origin[index] - second_origin[index]
            for index in range(3)
        ]
        first_range = (
            cosine * _dot(second_direction, offset)
            - _dot(first_direction, offset)
        ) / denominator
        second_range = (
            _dot(second_direction, offset)
            - cosine * _dot(first_direction, offset)
        ) / denominator
        if first_range <= 0.0 or second_range <= 0.0:
            continue
        first_point = [
            first_origin[index] + first_range * first_direction[index]
            for index in range(3)
        ]
        second_point = [
            second_origin[index] + second_range * second_direction[index]
            for index in range(3)
        ]
        miss = _distance(first_point, second_point)
        image_support = (
            first_range * math.tan(first_extent)
            + second_range * math.tan(second_extent)
        )
        if image_support <= 0.0:
            continue
        score = miss / image_support
        if score <= 1.0 and (best is None or score < best):
            best = score
    return best


def _same_station_support_score(
    obj: Mapping[str, Any], observation: Mapping[str, Any]
) -> float | None:
    """Match duplicate detections from overlapping views at one optical centre."""
    second = _representative_ray(observation)
    if second is None:
        return None
    _second_origin, second_direction, second_extent = second
    group = str(observation.get("optical_center_group", ""))
    best: float | None = None
    for evidence in obj.get("evidence", ()):
        if str(evidence.get("optical_center_group", "")) != group:
            continue
        first = _representative_ray(evidence)
        if first is None:
            continue
        _first_origin, first_direction, first_extent = first
        cosine = max(-1.0, min(1.0, _dot(first_direction, second_direction)))
        angular_separation = math.acos(cosine)
        angular_support = first_extent + second_extent
        if angular_support <= 0.0:
            continue
        score = angular_separation / angular_support
        if score <= 1.0 and (best is None or score < best):
            best = score
    return best


def _association_radius(value: Mapping[str, Any]) -> float:
    """Return the footprint radius that identity association may trust."""
    # Negative or inconclusive identity evidence is a soft association cost,
    # not a radius clamp.  A bearing-only split may acquire strong same-ID
    # support from a later view; shrinking the radius here would make that
    # evidence unreachable before the resolver can evaluate it.
    geometry_items = _geometry_evidence_items(value)
    extents: list[float] = []
    for item in geometry_items:
        try:
            extent = np.asarray(item.get("bbox_3d"), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if extent.shape == (3,) and np.isfinite(extent).all() and np.all(extent > 0.0):
            extents.append(0.5 * math.hypot(float(extent[0]), float(extent[1])))
    if not extents:
        try:
            extent = np.asarray(value.get("bbox_3d"), dtype=np.float64)
        except (TypeError, ValueError):
            extent = np.zeros(3, dtype=np.float64)
        if extent.shape == (3,) and np.isfinite(extent).all() and np.all(extent > 0.0):
            extents.append(0.5 * math.hypot(float(extent[0]), float(extent[1])))
    footprint_radius = float(np.median(extents)) if extents else 0.10
    return max(
        0.05,
        min(0.85, footprint_radius + min(0.30, _center_uncertainty_radius(value))),
    )


def _association_limit(obj: Mapping[str, Any], observation: Mapping[str, Any]) -> float:
    old_radius = _association_radius(obj)
    new_radius = _association_radius(observation)
    return min(1.5, max(0.45, old_radius + new_radius + 0.20))


def _strong_identity_support(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    voxel_size_m: float,
) -> bool:
    """Check whether a weak candidate has independent same-instance support."""
    try:
        distance = _distance(first["center_3d"], second["center_3d"])
    except (KeyError, TypeError, ValueError):
        return False
    association_limit = _association_limit(first, second)
    shared = _shared_view_box_metrics(first, second)
    if (
        float(shared.get("best_iou", 0.0)) >= 0.45
        and distance <= max(0.55, association_limit)
    ):
        return True

    mismatch = _pointcloud_support_score(
        _value_map_points(first),
        _value_map_points(second),
        voxel_size_m,
    )
    cloud_similarity = None if mismatch is None else 1.0 - mismatch
    appearance, appearance_quality = _track_appearance_similarity(first, second)
    size = _normalized_size_similarity(first, second)
    ray = _ray_support_score(first, second)
    if (
        cloud_similarity is not None
        and cloud_similarity >= 0.58
        and distance <= max(0.85, association_limit)
    ):
        return True
    if (
        appearance is not None
        and appearance_quality > 0.0
        and appearance >= 0.90
        and distance <= 0.45
        and (size is None or size >= 0.30)
        and (ray is None or ray >= 0.30)
    ):
        return True
    same_station = _same_station_support_score(first, second)
    return bool(
        same_station is not None
        and same_station <= 0.35
        and distance <= max(0.55, association_limit)
    )


SAME_STATION_SEPARATION_SCORE = 0.5
def _independent_view_count(evidence: Sequence[Mapping[str, Any]], minimum_separation_m: float) -> int:
    representatives: list[list[float]] = []
    for item in evidence:
        viewpoint = list(item["viewpoint_position_map"])
        if representatives and any(
            _distance(viewpoint, existing) < minimum_separation_m
            for existing in representatives
        ):
            continue
        representatives.append(viewpoint)
    return len(representatives)


def _semantic_evidence_weight(item: Mapping[str, Any]) -> float:
    geometry = max(0.0, min(1.0, float(item.get("geometry_confidence", 0.0) or 0.0)))
    lidar_support = max(0.0, float(item.get("lidar_support_count", 0.0) or 0.0))
    lidar_quality = min(1.0, math.log1p(lidar_support) / math.log(41.0))
    appearance_quality = max(0.0, min(1.0, float(item.get("appearance_quality", 0.0) or 0.0)))
    registered_quality = max(
        0.0,
        min(1.0, float(item.get("registered_scan_inlier_fraction_0_25m", 0.0) or 0.0)),
    )
    verified = 1.0 if item.get("qwen_verified") is True else 0.0
    geometry_weight = max(
        0.08,
        0.12
        + 0.24 * geometry
        + 0.24 * lidar_quality
        + 0.12 * appearance_quality
        + 0.12 * registered_quality
        + 0.16 * verified,
    )
    verdict = _normalized_evidence_verdict(item)
    target_probability = max(
        0.0,
        min(1.0, float(item.get("semantic_probability", 0.0) or 0.0)),
    )
    # A strict visual class verdict is semantic evidence even when the mask
    # has no usable depth.  Preserve its calibrated strength directly;
    # geometry quality continues to govern localization and identity fusion,
    # but cannot dilute a visible target/confuser judgment.
    semantic_weight = (
        target_probability
        if item.get("qwen_verified") is True or verdict == "target_wins"
        else 1.0 - target_probability
        if verdict == "confuser_wins"
        else 0.0
    )
    return max(geometry_weight, semantic_weight)


def _verification_weight(item: Mapping[str, Any]) -> float:
    for key in ("confidence", "probability", "target_probability", "score"):
        value = item.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.15, min(1.0, float(value)))
    return 0.75


def _refresh_appearance_from_evidence(obj: dict[str, Any]) -> None:
    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for item in obj.get("evidence", ()):
        vector = _safe_descriptor(item.get("appearance_descriptor"))
        if not len(vector):
            continue
        quality = max(0.05, min(1.0, float(item.get("appearance_quality", 0.0) or 0.0)))
        vectors.append(vector)
        weights.append(quality * _semantic_evidence_weight(item))
    if not vectors or len({tuple(vector.shape) for vector in vectors}) != 1:
        obj["appearance_prototype"] = []
        obj["appearance_quality"] = 0.0
        return
    prototype = np.average(
        np.stack(vectors, axis=0),
        axis=0,
        weights=np.asarray(weights, dtype=np.float64),
    )
    norm = float(np.linalg.norm(prototype))
    if norm <= 1e-9:
        obj["appearance_prototype"] = []
        obj["appearance_quality"] = 0.0
        return
    obj["appearance_prototype"] = (prototype / norm).astype(float).tolist()
    obj["appearance_quality"] = float(min(1.0, sum(weights)))


def _recompute_track_state(
    obj: dict[str, Any],
    minimum_separation_m: float,
) -> None:
    deduplicated: dict[str, dict[str, Any]] = {}
    for item in obj.get("evidence", ()):
        observation_id = str(item.get("observation_id", "")).strip()
        if not observation_id:
            continue
        deduplicated[observation_id] = copy.deepcopy(dict(item))
    evidence = list(deduplicated.values())
    obj["evidence"] = evidence
    obj["evidence_count"] = len(evidence)
    observed_labels = {
        str(item.get("observed_class_label", item.get("class_label", ""))).strip().lower()
        for item in evidence
        if str(item.get("observed_class_label", item.get("class_label", ""))).strip()
    }
    existing_labels = obj.get("observed_class_labels", ())
    if isinstance(existing_labels, Sequence) and not isinstance(existing_labels, (str, bytes)):
        observed_labels.update(
            str(value).strip().lower() for value in existing_labels if str(value).strip()
        )
    obj["class_label"] = _canonical_class_label(obj.get("class_label", ""))
    obj["observed_class_labels"] = sorted(observed_labels or {obj["class_label"]})

    if evidence:
        weights = [_semantic_evidence_weight(item) for item in evidence]
        probability = sum(
            weight * float(item.get("semantic_probability", 0.0) or 0.0)
            for weight, item in zip(weights, evidence)
        ) / max(sum(weights), 1e-9)
        obj["semantic_probability"] = float(max(0.0, min(1.0, probability)))
        obj["last_acquisition_id"] = str(evidence[-1].get("acquisition_id", ""))
    else:
        obj["semantic_probability"] = float(obj.get("semantic_probability", 0.0) or 0.0)

    independent = _independent_view_count(evidence, minimum_separation_m)
    station_ids = {
        str(item.get("optical_center_group", ""))
        for item in evidence
        if str(item.get("optical_center_group", ""))
    }
    obj["independent_viewpoint_count"] = independent
    obj["station_count"] = len(station_ids)
    physical_status = "confirmed" if independent >= 2 else "tentative"
    obj["physical_status"] = physical_status
    obj["status"] = physical_status

    target_support = 0.0
    confuser_support = 0.0
    verification_confuser_support = 0.0
    for item in evidence:
        weight = _semantic_evidence_weight(item)
        verdict = str(item.get("qwen_candidate_verdict", ""))
        if verdict == "target_wins" or item.get("qwen_verified") is True:
            target_support += weight
        elif verdict == "confuser_wins":
            confuser_support += weight
    for item in obj.get("semantic_verifications", ()):
        verdict = str(item.get("verdict", ""))
        weight = _verification_weight(item)
        if verdict == "target_wins":
            target_support += weight
        elif verdict == "confuser_wins":
            confuser_support += weight
            verification_confuser_support += weight

    margin = target_support - confuser_support
    probability = float(obj.get("semantic_probability", 0.0) or 0.0)
    if target_support >= 0.35 and margin >= 0.25:
        semantic_status = "verified"
    elif (
        confuser_support >= 0.50
        and margin <= -0.35
        and (probability < 0.45 or verification_confuser_support > 0.0)
    ):
        semantic_status = "rejected"
    elif target_support > 0.0 or confuser_support > 0.0:
        semantic_status = "ambiguous"
    else:
        semantic_status = str(obj.get("semantic_status", "unverified"))
        if semantic_status not in {"verified", "rejected"}:
            semantic_status = "unverified"
    obj["semantic_status"] = semantic_status
    obj["semantic_target_support"] = float(target_support)
    obj["semantic_confuser_support"] = float(confuser_support)
    obj["semantic_margin"] = float(margin)
    _refresh_appearance_from_evidence(obj)


def _attach_observation(
    obj: dict[str, Any],
    observation: Mapping[str, Any],
    acquisition_id: str,
    minimum_separation_m: float,
) -> None:
    observation_id = str(observation["observation_id"])
    evidence = [
        item
        for item in obj.get("evidence", ())
        if str(item.get("observation_id", "")) != observation_id
    ]
    evidence.append({**copy.deepcopy(dict(observation)), "acquisition_id": acquisition_id})
    obj["evidence"] = evidence
    obj["instance_version"] = int(obj.get("instance_version", 0)) + 1
    obj["observation_count"] = len(evidence)
    obj["last_seen_time"] = observation.get("timestamp", acquisition_id)
    origins = [
        list(item.get("viewpoint_position_map", ()))
        for item in evidence
        if isinstance(item.get("viewpoint_position_map"), Sequence)
        and len(item.get("viewpoint_position_map", ())) == 3
    ]
    obj["viewpoint_origins"] = origins[-32:]
    bearing_records = [
        copy.deepcopy(item.get("bearing_observation"))
        for item in evidence
        if isinstance(item.get("bearing_observation"), Mapping)
    ]
    obj["bearing_observations"] = bearing_records[-32:]
    obj["position_mean_xyz"] = list(observation.get("center_3d", obj.get("center_3d", ())))
    obj["position_cov_xyz"] = copy.deepcopy(observation.get("center_cov", obj.get("center_cov")))
    bbox = list(observation.get("bbox_3d", obj.get("bbox_3d", ())))
    center = list(observation.get("center_3d", obj.get("center_3d", ())))
    obj["footprint_obb"] = {
        "center_xy": center[:2],
        "extent_xy": bbox[:2],
        "yaw_rad": float(observation.get("footprint_yaw_rad", 0.0) or 0.0),
    }
    obj["footprint_cov"] = copy.deepcopy(observation.get("extent_cov", obj.get("extent_cov")))
    appearance = observation.get("appearance_descriptor")
    if appearance:
        obj["appearance_embedding"] = copy.deepcopy(appearance)
    class_log_probs = dict(obj.get("class_log_probs", {}))
    label = str(observation.get("class_label", obj.get("class_label", "")))
    class_log_probs[label] = float(observation.get("semantic_probability", 0.5))
    obj["class_log_probs"] = class_log_probs
    if observation.get("support_parent_id") is not None:
        obj["support_parent_id"] = observation.get("support_parent_id")
        obj["support_parent_probability"] = observation.get("support_parent_probability")
    _recompute_track_state(obj, minimum_separation_m)


def _new_object(
    store: dict[str, Any],
    observation: Mapping[str, Any],
    acquisition_id: str,
    minimum_separation_m: float,
) -> dict[str, Any]:
    object_id = int(store["next_object_id"])
    store["next_object_id"] = object_id + 1
    obj = {
        "object_id": object_id,
        "frame_id": "map",
        "class_label": _canonical_class_label(observation["class_label"]),
        "observed_class_labels": [str(observation.get("observed_class_label", observation["class_label"]))],
        "center_3d": list(observation["center_3d"]),
        "bbox_3d": list(observation["bbox_3d"]),
        "center_cov": copy.deepcopy(observation.get("center_cov")),
        "extent_cov": copy.deepcopy(observation.get("extent_cov")),
        "center_covariance_provenance": str(
            observation.get("center_covariance_provenance", "")
        ),
        "extent_covariance_provenance": str(
            observation.get("extent_covariance_provenance", "")
        ),
        "covariance_mode": str(
            observation.get("covariance_mode", "probabilistic")
        ),
        "semantic_probability": float(observation["semantic_probability"]),
        "instance_version": 0,
        "status": "tentative",
        "physical_status": "tentative",
        "semantic_status": "unverified",
        "identity_state": "NEW_TRACK",
        "identity_association_hypotheses": [],
        "semantic_verifications": [],
        "independent_viewpoint_count": 1,
        "identity_member_ids": [object_id],
        "class_log_probs": {
            str(observation["class_label"]): float(observation["semantic_probability"])
        },
        "position_mean_xyz": list(observation["center_3d"]),
        "position_cov_xyz": copy.deepcopy(observation.get("center_cov")),
        "footprint_obb": {
            "center_xy": list(observation["center_3d"][:2]),
            "extent_xy": list(observation["bbox_3d"][:2]),
            "yaw_rad": float(observation.get("footprint_yaw_rad", 0.0) or 0.0),
        },
        "footprint_cov": copy.deepcopy(observation.get("extent_cov")),
        "appearance_embedding": copy.deepcopy(observation.get("appearance_descriptor", [])),
        "support_parent_id": observation.get("support_parent_id"),
        "support_parent_probability": observation.get("support_parent_probability"),
        "last_seen_time": observation.get("timestamp", acquisition_id),
        "observation_count": 1,
        "viewpoint_origins": [list(observation["viewpoint_position_map"])],
        "bearing_observations": [
            copy.deepcopy(observation["bearing_observation"])
        ] if isinstance(observation.get("bearing_observation"), Mapping) else [],
        "evidence": [],
    }
    _attach_observation(obj, observation, acquisition_id, minimum_separation_m)
    store["objects"].append(obj)
    return obj


def relation_role_semantic_verifications(
    relation_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Translate explicit Qwen role contradictions into semantic evidence.

    RelationEngine remains the predicate-verdict owner.  This adapter consumes
    only its structured role audit and never reinterprets the relation state.
    A jointly visible participant whose declared semantic role is explicitly
    NO becomes negative class evidence for SceneMemory after the current
    frozen-revision action has been produced.
    """
    verifications: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in relation_records:
        if not isinstance(raw, Mapping):
            continue
        qwen = raw.get("qwen")
        if not isinstance(qwen, Mapping):
            continue
        nested = qwen.get("role_audit")
        report = nested if isinstance(nested, Mapping) else qwen
        if (
            str(report.get("verification_pass", "")).lower() != "roles"
            or report.get("jointly_observable") is not True
        ):
            continue
        try:
            subject_id = int(report.get("subject_id", -1))
            object_ids = [int(value) for value in report.get("object_ids", ())]
            confidence = float(report.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        object_states = [
            str(value).upper()
            for value in report.get("object_role_states", ())
        ]
        visible_objects = [
            bool(value) for value in report.get("visible_objects", ())
        ]
        if (
            subject_id < 0
            or len(object_ids) != len(object_states)
            or len(object_ids) != len(visible_objects)
            or not math.isfinite(confidence)
        ):
            continue
        relation_id = str(raw.get("relation_id", ""))
        predicate = str(raw.get("predicate", ""))
        station_id = str(raw.get("station_id", ""))
        evidence_ids = [
            str(value) for value in raw.get("evidence_ids", ())
            if str(value).strip()
        ]
        tuple_key = ",".join(str(value) for value in object_ids)

        def append_confuser(object_id: int, role: str) -> None:
            verification_id = "|".join((
                "relation_role",
                relation_id,
                predicate,
                f"s{subject_id}",
                f"o{tuple_key}",
                station_id,
                role,
            ))
            if verification_id in seen:
                return
            seen.add(verification_id)
            verifications.append({
                "verification_id": verification_id,
                "object_id": int(object_id),
                "verdict": "confuser_wins",
                "confidence": max(0.0, min(1.0, confidence)),
                "source": "qwen_relation_role_audit",
                "semantic_role": role,
                "relation_id": relation_id,
                "predicate": predicate,
                "reason_code": str(report.get("reason_code", "")),
                "station_id": station_id,
                "evidence_ids": evidence_ids,
                "identity_version": int(raw.get("identity_version", 0) or 0),
                "geometry_version": int(raw.get("geometry_version", 0) or 0),
            })

        if (
            str(report.get("subject_role_state", "")).upper() == "NO"
            and report.get("visible_subject") is True
        ):
            append_confuser(subject_id, "subject")
        for index, (object_id, state, visible) in enumerate(zip(
            object_ids, object_states, visible_objects
        ), start=1):
            if state == "NO" and visible:
                append_confuser(object_id, f"object_{index}")
    return verifications


def apply_scene_memory_semantic_verifications(
    store_path: Path,
    verifications: Sequence[Mapping[str, Any]],
) -> dict[int, str]:
    """Persist semantic decisions against canonical, alias-resolved tracks."""
    store_path = store_path.resolve()
    lock_path = store_path.with_suffix(store_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    touched: dict[int, str] = {}
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        store = _load(store_path)
        by_id = {int(obj["object_id"]): obj for obj in store.get("objects", ())}
        for raw in verifications:
            try:
                requested_id = int(raw.get("object_id", -1))
            except (TypeError, ValueError):
                continue
            object_id = _resolve_object_id(store, requested_id)
            obj = by_id.get(object_id)
            if obj is None:
                continue
            record = copy.deepcopy(dict(raw))
            record["object_id"] = object_id
            if requested_id != object_id:
                record["requested_object_id"] = requested_id
            verification_id = str(record.get("verification_id", ""))
            history = [
                item
                for item in obj.get("semantic_verifications", ())
                if not verification_id
                or str(item.get("verification_id", "")) != verification_id
            ]
            history.append(record)
            obj["semantic_verifications"] = history[-32:]
            _recompute_track_state(obj, 0.30)
            touched[object_id] = str(obj.get("semantic_status", "unverified"))
        if touched:
            store["semantic_revision"] = int(store.get("semantic_revision", 0)) + 1
            _atomic_write(store_path, store)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return touched


def _track_appearance_similarity(
    obj: Mapping[str, Any], observation: Mapping[str, Any]
) -> tuple[float | None, float]:
    prototype = obj.get("appearance_prototype")
    if not prototype:
        evidence_vectors = [
            item.get("appearance_descriptor")
            for item in obj.get("evidence", ())
            if item.get("appearance_descriptor")
        ]
        prototype = evidence_vectors[-1] if evidence_vectors else None
    similarity = _cosine_similarity(prototype, observation.get("appearance_descriptor"))
    quality = min(
        float(obj.get("appearance_quality", 0.0) or 0.0),
        float(observation.get("appearance_quality", 0.0) or 0.0),
    )
    return similarity, max(0.0, min(1.0, quality))


def _coverage_provenance_ids(value: Mapping[str, Any]) -> set[str]:
    result = {
        str(item)
        for item in value.get("coverage_member_observation_ids", ())
        if str(item)
    }
    for evidence in value.get("evidence", ()):
        if not isinstance(evidence, Mapping):
            continue
        result.update(
            str(item)
            for item in evidence.get("coverage_member_observation_ids", ())
            if str(item)
        )
    return result


def _evidence_observation_ids(value: Mapping[str, Any]) -> set[str]:
    result = {
        str(value.get("observation_id", ""))
    } if str(value.get("observation_id", "")) else set()
    result.update(
        str(evidence.get("observation_id", ""))
        for evidence in value.get("evidence", ())
        if isinstance(evidence, Mapping)
        and str(evidence.get("observation_id", ""))
    )
    return result


def _association_cost(
    obj: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    voxel_size_m: float,
) -> tuple[float, float] | None:
    """Return a compound identity cost; no single weak signal can dominate."""
    if not _class_labels_compatible(obj.get("class_label"), observation.get("class_label")):
        return None
    if bool(_coverage_provenance_ids(obj)) != bool(
        _coverage_provenance_ids(observation)
    ):
        return None

    same_station_score = _same_station_support_score(obj, observation)
    same_station_reuse = any(
        str(item.get("optical_center_group", ""))
        == str(observation.get("optical_center_group", ""))
        for item in obj.get("evidence", ())
    )
    shared_view = _shared_view_box_metrics(obj, observation)
    same_station_disjoint = bool(
        same_station_reuse
        and _shared_view_boxes_are_disjoint(shared_view)
        and not _shared_view_fragment_compatible(
            obj, observation, shared=shared_view
        )
    )

    distance = _distance(obj["center_3d"], observation["center_3d"])
    # A ray or appearance match can refine a plausible track, but it cannot
    # turn a several-metre displacement into the same physical instance.  The
    # association limit is derived from both 3-D footprints and is the shared
    # geometric feasibility boundary for every class.
    if distance > _association_limit(obj, observation):
        return None
    first_cov = np.asarray(obj.get("center_cov"), dtype=np.float64)
    second_cov = np.asarray(observation.get("center_cov"), dtype=np.float64)
    covariance = first_cov + second_cov
    delta = np.asarray(observation["center_3d"], dtype=np.float64) - np.asarray(
        obj["center_3d"], dtype=np.float64
    )
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        covariance = np.eye(3, dtype=np.float64) * 0.25 ** 2
    covariance = covariance + np.eye(3, dtype=np.float64) * 1e-4
    try:
        mahalanobis = math.sqrt(max(0.0, float(delta @ np.linalg.solve(covariance, delta))))
    except (np.linalg.LinAlgError, ValueError):
        mahalanobis = distance / 0.5
    mahalanobis_cost = min(1.0, mahalanobis / 6.0)

    appearance_similarity, appearance_quality = _track_appearance_similarity(obj, observation)
    appearance_cost = (
        1.0 - float(appearance_similarity)
        if appearance_similarity is not None and appearance_quality > 0.0
        else 0.5
    )
    size_similarity = _normalized_size_similarity(obj, observation)
    footprint_cost = 1.0 - float(size_similarity) if size_similarity is not None else 0.5
    ray_score = _ray_support_score(obj, observation)
    bearing_cost = min(1.0, float(ray_score)) if ray_score is not None else 0.5
    class_cost = 0.0
    same_station_cost = (
        min(1.0, float(same_station_score))
        if same_station_reuse and same_station_score is not None
        else 0.0
    )
    # One soft Hungarian assignment: Mahalanobis .40, appearance .25,
    # class .15, footprint .10, bearing .10.
    cost = (
        0.40 * mahalanobis_cost
        + 0.25 * appearance_cost
        + 0.15 * class_cost
        + 0.10 * footprint_cost
        + 0.10 * bearing_cost
    )
    if same_station_reuse:
        cost = min(1.0, cost + 0.05 * same_station_cost)
    # Same-station disjoint boxes and weak/negative semantic evidence are
    # useful negative signals, but never an irreversible cannot-link.  Keep
    # the proposal available for later cross-view reconciliation.
    if same_station_disjoint:
        cost = min(1.0, cost + 0.12)
    if _identity_requires_separation(obj) or _identity_requires_separation(observation):
        cost = min(1.0, cost + 0.08)
    if _geometry_has_conflict(obj) or _geometry_has_conflict(observation):
        cost = min(1.0, cost + 0.08)
    return float(cost), float(distance)


def _bbox_iou_2d(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != 4 or len(second) != 4:
        return 0.0
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 1e-9 else 0.0


def _source_view_boxes(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    # Track-level evidence stores the per-view boxes on each observation.  A
    # track itself may not have a flattened copy, so collect boxes from both
    # shapes.  Without this aggregation the identity reconciler silently sees
    # no shared views and cannot use cross-view correspondence to merge a
    # detector split.
    raw_boxes: list[Any] = list(value.get("source_view_boxes", ()) or ())
    for evidence in value.get("evidence", ()):
        if isinstance(evidence, Mapping):
            raw_boxes.extend(evidence.get("source_view_boxes", ()) or ())
    seen: set[tuple[str, tuple[float, ...]]] = set()
    for raw in raw_boxes:
        if not isinstance(raw, Mapping):
            continue
        try:
            bbox = [float(item) for item in raw.get("bbox_xyxy", ())]
            width = int(raw.get("image_width", 0))
            height = int(raw.get("image_height", 0))
        except (TypeError, ValueError):
            continue
        view_id = str(raw.get("view_id", ""))
        key = (view_id, tuple(bbox))
        if (
            len(bbox) == 4
            and width > 0
            and height > 0
            and view_id
            and key not in seen
        ):
            seen.add(key)
            output.append({
                "view_id": view_id,
                "bbox_xyxy": bbox,
                "image_width": width,
                "image_height": height,
            })
    return output


def _shared_view_box_metrics(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Return same-image overlap and fragment geometry without non-finite values."""
    best_iou = 0.0
    maximum_separation = 0.0
    maximum_horizontal_overlap_ratio = 0.0
    minimum_normalized_vertical_gap = 1.0
    shared_view_ids: set[str] = set()
    for a in _source_view_boxes(first):
        for b in _source_view_boxes(second):
            if a["view_id"] != b["view_id"]:
                continue
            shared_view_ids.add(a["view_id"])
            first_box = a["bbox_xyxy"]
            second_box = b["bbox_xyxy"]
            best_iou = max(best_iou, _bbox_iou_2d(first_box, second_box))
            ax = 0.5 * (first_box[0] + first_box[2])
            ay = 0.5 * (first_box[1] + first_box[3])
            bx = 0.5 * (second_box[0] + second_box[2])
            by = 0.5 * (second_box[1] + second_box[3])
            scale = max(1.0, math.hypot(a["image_width"], a["image_height"]))
            maximum_separation = max(
                maximum_separation, math.hypot(ax - bx, ay - by) / scale
            )

            first_width = max(1e-6, float(first_box[2]) - float(first_box[0]))
            second_width = max(1e-6, float(second_box[2]) - float(second_box[0]))
            horizontal_overlap = max(
                0.0,
                min(float(first_box[2]), float(second_box[2]))
                - max(float(first_box[0]), float(second_box[0])),
            )
            maximum_horizontal_overlap_ratio = max(
                maximum_horizontal_overlap_ratio,
                horizontal_overlap / min(first_width, second_width),
            )
            vertical_gap = max(
                0.0,
                max(float(first_box[1]), float(second_box[1]))
                - min(float(first_box[3]), float(second_box[3])),
            )
            image_height = max(1.0, float(a["image_height"]), float(b["image_height"]))
            minimum_normalized_vertical_gap = min(
                minimum_normalized_vertical_gap, vertical_gap / image_height
            )
    return {
        "shared_view_ids": sorted(shared_view_ids),
        "best_iou": float(best_iou),
        "maximum_normalized_center_separation": float(maximum_separation),
        "maximum_horizontal_overlap_ratio": float(maximum_horizontal_overlap_ratio),
        "minimum_normalized_vertical_gap": float(minimum_normalized_vertical_gap),
    }


def _shared_view_boxes_are_disjoint(metrics: Mapping[str, Any]) -> bool:
    """Require actual image-plane separation before recording cannot-link.

    A tiny IoU can still mean that one same-class detector box is a fragment
    nested in a larger box.  At least one image axis must therefore have a
    real gap before the pair is treated as separate instances.
    """
    # The helper is used both with the raw shared-view metric names and with
    # the track-level names emitted by ``_track_identity_metrics``.  Treat
    # them identically; otherwise reconciliation silently loses the image
    # separation that was recorded during ingestion.
    best_iou = float(
        metrics.get(
            "best_iou", metrics.get("shared_view_best_iou", 0.0)
        )
        or 0.0
    )
    center_separation = float(
        metrics.get(
            "maximum_normalized_center_separation",
            metrics.get("shared_view_center_separation", 0.0),
        )
        or 0.0
    )
    horizontal_overlap = float(
        metrics.get(
            "maximum_horizontal_overlap_ratio",
            metrics.get("shared_view_horizontal_overlap_ratio", 0.0),
        )
        or 0.0
    )
    vertical_gap = float(
        metrics.get(
            "minimum_normalized_vertical_gap",
            metrics.get("shared_view_vertical_gap_normalized", 1.0),
        )
        or 0.0
    )
    return bool(
        metrics.get("shared_view_ids")
        and best_iou <= 0.08
        and center_separation >= 0.025
        and (
            horizontal_overlap <= 0.05
            or vertical_gap >= 0.025
        )
    )


def _shared_view_fragment_compatible(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    shared: Mapping[str, Any] | None = None,
) -> bool:
    """Allow a supported same-class fragment without collapsing separate objects.

    A pair is treated as fragments only when map-frame footprints are tightly
    compatible *and* their disjoint same-view boxes overlap horizontally like
    vertically stacked parts.  Side-by-side objects remain explicit distinct
    instances.
    """
    if not _class_labels_compatible(
        first.get("class_label", first.get("canonical_class", "")),
        second.get("class_label", second.get("canonical_class", "")),
    ):
        return False
    shared_metrics = dict(shared or _shared_view_box_metrics(first, second))
    if not shared_metrics.get("shared_view_ids"):
        return False
    fragment_metrics = _vertical_fragment_metrics(first, second)
    if not bool(fragment_metrics.get("compatible", False)):
        return False
    tight_footprint = bool(
        float(fragment_metrics.get("horizontal_distance_m", _UNAVAILABLE_METRIC))
        <= min(0.24, float(fragment_metrics.get("horizontal_limit_m", 0.24)))
        and float(fragment_metrics.get("vertical_gap_m", _UNAVAILABLE_METRIC)) <= 0.28
    )
    return bool(
        tight_footprint
        and float(shared_metrics.get("best_iou", 0.0)) < 0.08
        and float(shared_metrics.get("maximum_horizontal_overlap_ratio", 0.0)) >= 0.40
        and float(shared_metrics.get("minimum_normalized_vertical_gap", 1.0)) <= 0.12
    )


def _shared_view_clipped_continuation(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    """Return whether a small boundary crop continues a larger same-view box.

    The fragment must be substantially smaller, touch an image boundary, and
    overlap the larger box only across the boundary-facing seam while strongly
    overlapping it on the orthogonal axis.  This distinguishes a clipped
    continuation from two peer-sized side-by-side detections.
    """
    if not _class_labels_compatible(
        first.get("class_label", first.get("canonical_class", "")),
        second.get("class_label", second.get("canonical_class", "")),
    ):
        return False
    for first_box_record in _source_view_boxes(first):
        for second_box_record in _source_view_boxes(second):
            if first_box_record["view_id"] != second_box_record["view_id"]:
                continue
            records = (first_box_record, second_box_record)
            areas = []
            for record in records:
                box = record["bbox_xyxy"]
                areas.append(
                    max(0.0, float(box[2]) - float(box[0]))
                    * max(0.0, float(box[3]) - float(box[1]))
                )
            if min(areas) <= 0.0 or min(areas) / max(areas) > 0.35:
                continue
            small_index = 0 if areas[0] <= areas[1] else 1
            small_record = records[small_index]
            large_record = records[1 - small_index]
            small = [float(value) for value in small_record["bbox_xyxy"]]
            large = [float(value) for value in large_record["bbox_xyxy"]]
            width = float(small_record["image_width"])
            height = float(small_record["image_height"])
            small_width = max(1e-6, small[2] - small[0])
            small_height = max(1e-6, small[3] - small[1])
            overlap_x = max(0.0, min(small[2], large[2]) - max(small[0], large[0]))
            overlap_y = max(0.0, min(small[3], large[3]) - max(small[1], large[1]))
            seam_x = overlap_x / small_width
            seam_y = overlap_y / small_height
            touches_left = small[0] <= 1.0 and large[0] > small[0]
            touches_right = small[2] >= width - 1.0 and large[2] < small[2]
            touches_top = small[1] <= 1.0 and large[1] > small[1]
            touches_bottom = small[3] >= height - 1.0 and large[3] < small[3]
            horizontal_continuation = bool(
                (touches_left or touches_right)
                and 0.05 <= seam_x <= 0.35
                and seam_y >= 0.65
            )
            vertical_continuation = bool(
                (touches_top or touches_bottom)
                and 0.05 <= seam_y <= 0.35
                and seam_x >= 0.65
            )
            if horizontal_continuation or vertical_continuation:
                return True
    return False


def _view_yaw_deg(view_id: object) -> float | None:
    marker = "_yaw_"
    value = str(view_id)
    if marker not in value:
        return None
    try:
        return float(value.rsplit(marker, 1)[1]) % 360.0
    except (TypeError, ValueError):
        return None


def _source_view_horizontal_fov(value: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in (value, *value.get("evidence", ())):
        if not isinstance(item, Mapping):
            continue
        camera_model = item.get("camera_model", {})
        if not isinstance(camera_model, Mapping):
            continue
        try:
            horizontal_fov = float(camera_model.get("horizontal_fov_deg"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(horizontal_fov) or horizontal_fov <= 0.0:
            continue
        view_ids = {
            str(item.get("view_id", "")),
            str(item.get("camera_id", "")),
            *(str(raw) for raw in item.get("source_view_ids", ())),
        }
        for view_id in view_ids:
            if view_id:
                result[view_id] = horizontal_fov
    return result


def _shared_acquisition_tile_continuation(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    """Return whether two perspective tiles continue one panorama object."""
    first_fovs = _source_view_horizontal_fov(first)
    second_fovs = _source_view_horizontal_fov(second)
    for first_record in _source_view_boxes(first):
        first_view = str(first_record["view_id"])
        first_yaw = _view_yaw_deg(first_view)
        first_acquisition = first_view.split("__pitch_", 1)[0]
        if first_yaw is None or first_view not in first_fovs:
            continue
        for second_record in _source_view_boxes(second):
            second_view = str(second_record["view_id"])
            second_yaw = _view_yaw_deg(second_view)
            if (
                second_view == first_view
                or second_yaw is None
                or second_view not in second_fovs
                or second_view.split("__pitch_", 1)[0] != first_acquisition
            ):
                continue
            yaw_separation = abs(
                ((second_yaw - first_yaw + 180.0) % 360.0) - 180.0
            )
            if yaw_separation > 0.5 * (
                first_fovs[first_view] + second_fovs[second_view]
            ):
                continue
            first_box = [float(raw) for raw in first_record["bbox_xyxy"]]
            second_box = [float(raw) for raw in second_record["bbox_xyxy"]]
            first_width = float(first_record["image_width"])
            second_width = float(second_record["image_width"])
            opposite_horizontal_edges = bool(
                (first_box[2] >= first_width - 1.0 and second_box[0] <= 1.0)
                or (second_box[2] >= second_width - 1.0 and first_box[0] <= 1.0)
            )
            if not opposite_horizontal_edges:
                continue
            vertical_overlap = max(
                0.0,
                min(first_box[3], second_box[3])
                - max(first_box[1], second_box[1]),
            )
            minimum_height = min(
                max(1.0, first_box[3] - first_box[1]),
                max(1.0, second_box[3] - second_box[1]),
            )
            if vertical_overlap / minimum_height >= 0.65:
                return True
    return False


def _canonical_pair_key(first_id: int, second_id: int) -> str:
    low, high = sorted((int(first_id), int(second_id)))
    return f"{low}:{high}"


def _normalized_cost_distribution(
    costs: Sequence[float],
) -> tuple[list[float], float]:
    """Return soft association probabilities and normalized entropy."""
    if not costs:
        return [], 0.0
    values = np.asarray([float(value) for value in costs], dtype=np.float64)
    values = values - float(np.min(values))
    weights = np.exp(-values / 0.12)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        probabilities = [1.0 / len(costs)] * len(costs)
    else:
        probabilities = [float(value / total) for value in weights]
    if len(probabilities) <= 1:
        return probabilities, 0.0
    entropy = -sum(
        probability * math.log(max(probability, 1e-12))
        for probability in probabilities
    ) / math.log(float(len(probabilities)))
    return probabilities, float(max(0.0, min(1.0, entropy)))


def _refresh_identity_ambiguity_groups(store: dict[str, Any]) -> None:
    """Materialize unresolved hypotheses without cross-observation chaining.

    A candidate set belongs to one ambiguous observation.  Two observations
    that happen to share one candidate are not automatically the same
    identity question; connected-component union made unrelated detector
    proposals expand one group forever and prevented later merge/split
    resolution.  Exact canonical candidate sets may be aggregated for
    telemetry, while distinct sets remain distinct groups.
    """
    records_by_key: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for raw in store.get("ambiguous_observations", ()):
        if not isinstance(raw, Mapping):
            continue
        try:
            ids = {
                _resolve_object_id(store, int(value))
                for value in raw.get("candidate_object_ids", ())
            }
        except (TypeError, ValueError):
            continue
        if len(ids) < 2:
            continue
        records_by_key.setdefault(tuple(sorted(ids)), []).append(
            copy.deepcopy(dict(raw))
        )
    groups: list[dict[str, Any]] = []
    for candidate_key, records in records_by_key.items():
        ordered = list(candidate_key)
        labels = sorted({
            str((record.get("observation") or {}).get("class_label", ""))
            for record in records
            if isinstance(record.get("observation"), Mapping)
            and str((record.get("observation") or {}).get("class_label", ""))
        })
        groups.append({
            "group_id": "identity-group:" + ":".join(str(value) for value in ordered),
            "canonical_object_ids": ordered,
            "class_labels": labels,
            "observation_count": len(records),
            "last_acquisition_id": str(records[-1].get("acquisition_id", "")),
            "association_entropy": float(
                records[-1].get("association_entropy", 1.0) or 0.0
            ),
            "association_margin": records[-1].get("association_margin"),
            "resolution_state": "OPEN",
            "last_resolution_update": str(
                records[-1].get("resolution_state", "OPEN")
            ),
            "observation_ids": [
                str((record.get("observation") or {}).get("observation_id", ""))
                for record in records
                if str((record.get("observation") or {}).get("observation_id", ""))
            ],
        })
    store["identity_ambiguity_groups"] = groups[-128:]


def _resolve_identity_ambiguities(
    store: dict[str, Any],
    *,
    minimum_separation_m: float,
    voxel_size_m: float,
) -> None:
    """Resolve old one-to-many assignments when a later view separates them."""
    objects = {
        int(value["object_id"]): value
        for value in store.get("objects", ())
        if isinstance(value, Mapping)
    }
    unresolved: list[dict[str, Any]] = []
    resolved_events = list(store.get("identity_resolution_events", ()))
    for raw in store.get("ambiguous_observations", ()):
        if not isinstance(raw, Mapping):
            continue
        observation = raw.get("observation")
        if not isinstance(observation, Mapping):
            unresolved.append(copy.deepcopy(dict(raw)))
            continue
        try:
            candidate_ids = sorted({
                _resolve_object_id(store, int(value))
                for value in raw.get("candidate_object_ids", ())
            })
        except (TypeError, ValueError):
            unresolved.append(copy.deepcopy(dict(raw)))
            continue
        prior_entropy = raw.get("association_entropy")
        try:
            prior_entropy = float(prior_entropy)
        except (TypeError, ValueError):
            prior_entropy = 1.0
        scored: list[tuple[float, int]] = []
        for object_id in candidate_ids:
            obj = objects.get(object_id)
            if obj is None:
                continue
            cost = _association_cost(
                obj,
                observation,
                voxel_size_m=voxel_size_m,
            )
            if cost is not None:
                scored.append((float(cost[0]), object_id))
        scored.sort()
        if not scored:
            unresolved.append(copy.deepcopy(dict(raw)))
            continue
        current_probabilities, current_entropy = _normalized_cost_distribution(
            [cost for cost, _ in scored]
        )
        winner_cost, winner_id = scored[0]
        # A singleton feasible set has no second candidate, so its margin is
        # undefined rather than infinite.  Keep the distinction explicit in
        # persisted telemetry; strict JSON serializers must not receive NaN or
        # Infinity from scene-memory identity resolution.
        margin = (
            float(scored[1][0] - winner_cost)
            if len(scored) > 1 else None
        )
        winner = objects.get(winner_id, {})
        decisive = bool(
            len(scored) == 1
            or (
                margin is not None
                and margin >= 0.12
                and winner.get("status") == "confirmed"
                and int(winner.get("independent_viewpoint_count", 0) or 0) >= 2
            )
        )
        if not decisive:
            record = copy.deepcopy(dict(raw))
            record["candidate_object_ids"] = candidate_ids
            record["candidate_costs"] = [float(cost) for cost, _ in scored]
            record["candidate_probabilities"] = current_probabilities
            record["association_entropy"] = current_entropy
            record["association_margin"] = margin
            entropy_delta = current_entropy - prior_entropy
            if entropy_delta < -0.05:
                resolution_state = "ENTROPY_DECREASED"
            elif entropy_delta > 0.05:
                resolution_state = "ENTROPY_INCREASED"
            else:
                resolution_state = "ENTROPY_UNCHANGED"
            record["resolution_state"] = resolution_state
            resolved_events.append({
                "observation_id": str(observation.get("observation_id", "")),
                "acquisition_id": str(raw.get("acquisition_id", "")),
                "candidate_object_ids": candidate_ids,
                "previous_entropy": prior_entropy,
                "current_entropy": current_entropy,
                "entropy_delta": entropy_delta,
                "resolution_state": resolution_state.lower(),
                "reason": "later_view_identity_information_update",
            })
            unresolved.append(record)
            continue
        assigned_id = _resolve_object_id(
            store,
            int(raw.get("resolved_object_id", winner_id)),
        )
        observation_id = str(observation.get("observation_id", ""))
        if assigned_id != winner_id and observation_id:
            assigned = objects.get(assigned_id)
            if assigned is not None:
                assigned["evidence"] = [
                    value for value in assigned.get("evidence", ())
                    if str(value.get("observation_id", "")) != observation_id
                ]
                _recompute_track_state(assigned, minimum_separation_m)
            _attach_observation(
                winner,
                observation,
                str(raw.get("acquisition_id", "identity-resolution")),
                minimum_separation_m,
            )
        winner["identity_state"] = "IDENTITY_RESOLVED_BY_LATER_VIEW"
        resolved_events.append({
            "observation_id": observation_id,
            "acquisition_id": str(raw.get("acquisition_id", "")),
            "previous_object_id": assigned_id,
            "resolved_object_id": winner_id,
            "candidate_object_ids": candidate_ids,
            "cost_margin": margin,
            "association_entropy_before": prior_entropy,
            "association_entropy_after": current_entropy,
            "resolution_state": (
                "confirmed_same_identity"
                if assigned_id == winner_id
                else "confirmed_different_identity"
            ),
            "reason": "later_view_identity_separation",
        })
    store["ambiguous_observations"] = unresolved[-128:]
    store["identity_resolution_events"] = resolved_events[-64:]


def _record_distinct_pairs(
    store: dict[str, Any],
    observation_to_object_id: Mapping[str, int],
    observations: Sequence[Mapping[str, Any]],
    acquisition_id: str,
    *,
    geometry_version: int = 0,
    voxel_size_m: float = 0.05,
) -> None:
    records = [
        copy.deepcopy(dict(value))
        for value in store.get("distinct_object_pairs", ())
        if isinstance(value, Mapping)
    ]
    records_by_key: dict[str, dict[str, Any]] = {}
    for value in records:
        key = str(value.get("pair_key", ""))
        if not key:
            try:
                ids = sorted({
                    int(item) for item in value.get("object_ids", ())
                })
            except (TypeError, ValueError):
                ids = []
            if len(ids) == 2:
                key = _canonical_pair_key(ids[0], ids[1])
                value["pair_key"] = key
        if key:
            records_by_key[key] = value
    for index, first in enumerate(observations):
        first_id = observation_to_object_id.get(str(first.get("observation_id", "")))
        if first_id is None:
            continue
        for second in observations[index + 1 :]:
            second_id = observation_to_object_id.get(str(second.get("observation_id", "")))
            if second_id is None or int(first_id) == int(second_id):
                continue
            if not _class_labels_compatible(first.get("class_label"), second.get("class_label")):
                continue
            metrics = _shared_view_box_metrics(first, second)
            try:
                world_distance = _distance(
                    first["center_3d"], second["center_3d"]
                )
                world_extent = max(
                    max(float(value) for value in first.get("bbox_3d", ())[:3]),
                    max(float(value) for value in second.get("bbox_3d", ())[:3]),
                )
            except (KeyError, TypeError, ValueError):
                world_distance = 0.0
                world_extent = 0.0
            metric_geometry_reliable = bool(
                _metric_geometry_reliable(first)
                and _metric_geometry_reliable(second)
            )
            first_points = _value_map_points(first)
            second_points = _value_map_points(second)
            cloud_mismatch = _pointcloud_support_score(
                first_points,
                second_points,
                voxel_size_m,
            )
            shared_view_fragment = _shared_view_fragment_compatible(
                first, second, shared=metrics
            )
            # Two independently supported clouds with zero voxel overlap in
            # the same real image are co-visible physical instances, unless
            # their boxes and footprint satisfy the explicit vertical-
            # fragment contract.  This is stronger than centroid distance,
            # which is easily biased by sparse support.
            shared_view_pointcloud_separate = bool(
                metrics["shared_view_ids"]
                and metric_geometry_reliable
                and len(first_points)
                and len(second_points)
                and cloud_mismatch is None
                and not shared_view_fragment
            )
            # A distance difference between two views from the same station
            # is not, by itself, proof of two instances: the views can observe
            # different parts of one object, and bearing-only centers can be
            # biased.  Cross-view separation is retained as weighted negative
            # evidence only when the metric geometry is independently
            # supported.
            shared_view_world_separate = bool(
                metrics["shared_view_ids"]
                and metric_geometry_reliable
                and world_distance >= max(0.40, 2.0 * max(world_extent, 0.05))
            )
            shared_view_disjoint = _shared_view_boxes_are_disjoint(metrics)
            if (
                not shared_view_disjoint
                and not shared_view_world_separate
                and not shared_view_pointcloud_separate
            ):
                continue
            if shared_view_fragment:
                # Do not persist negative evidence for two sparse boxes that
                # are tightly aligned vertical fragments.
                continue
            key = _canonical_pair_key(int(first_id), int(second_id))
            # Within the disjoint branch, less overlap is the stronger
            # negative signal.  The old expression increased distinct support
            # as IoU increased, which inverted that evidence term.
            disjointness = max(
                0.0,
                min(1.0, 1.0 - float(metrics["best_iou"]) / 0.08),
            )
            weak_strength = float(min(
                0.45,
                0.18
                + 0.12 * min(
                    1.0,
                    float(metrics["maximum_normalized_center_separation"])
                    / 0.25,
                )
                + 0.08 * disjointness,
            ))
            strong_strength = float(
                0.80
                if shared_view_world_separate or shared_view_pointcloud_separate
                else weak_strength
            )
            record = records_by_key.get(key)
            if record is None:
                record = {
                    "pair_key": key,
                    "object_ids": sorted((int(first_id), int(second_id))),
                    "class_label": str(first.get("class_label", "")),
                    "identity_version": int(store.get("identity_revision", 0)),
                    "evidence_type": "identity_pair_evidence",
                    "hard": False,
                    "distinct_identity_support": 0.0,
                    "same_identity_support": 0.0,
                    "distinct_acquisition_ids": [],
                    "same_supporting_acquisition_ids": [],
                    "contradicting_acquisition_ids": [],
                    "evidence_events": [],
                }
                records.append(record)
                records_by_key[key] = record
            acquisition_key = str(acquisition_id)
            event = {
                "acquisition_id": acquisition_key,
                "station_id": str(
                    first.get("station_id") or second.get("station_id") or ""
                ),
                "timestamp": float(
                    first.get("timestamp", first.get("timestamp_unix", time.time()))
                    or time.time()
                ),
                "source_observation_ids": [
                    str(first.get("observation_id", "")),
                    str(second.get("observation_id", "")),
                ],
                "identity_version": int(store.get("identity_revision", 0)),
                "geometry_version": int(geometry_version),
                "strength": strong_strength,
                "reliable_metric_separation": bool(
                    shared_view_world_separate
                    or shared_view_pointcloud_separate
                ),
                "shared_view_ids": list(metrics["shared_view_ids"]),
                "best_iou": float(metrics["best_iou"]),
                "world_distance_m": float(world_distance),
                "world_separation": bool(shared_view_world_separate),
                "pointcloud_separation": bool(shared_view_pointcloud_separate),
                "normalized_center_separation": float(
                    metrics["maximum_normalized_center_separation"]
                ),
                "reason": (
                    "simultaneously_visible_world_separated_instances"
                    if shared_view_world_separate
                    else "simultaneously_visible_disjoint_pointcloud_instances"
                    if shared_view_pointcloud_separate
                    else "simultaneously_visible_disjoint_boxes_soft_evidence"
                ),
            }
            events = [
                dict(value)
                for value in record.get("evidence_events", ())
                if isinstance(value, Mapping)
            ]
            event_key = (
                acquisition_key,
                tuple(sorted(event["source_observation_ids"])),
                event["reason"],
            )
            events = [
                value for value in events
                if (
                    str(value.get("acquisition_id", "")),
                    tuple(sorted(
                        str(item)
                        for item in value.get("source_observation_ids", ())
                    )),
                    str(value.get("reason", "")),
                ) != event_key
            ]
            events.append(event)
            events = events[-64:]
            record["evidence_events"] = events

            # Identity evidence must grow with independent acquisitions, not
            # with duplicate tuple bookkeeping inside one acquisition.  Keep
            # the strongest event from each acquisition and derive the
            # aggregate fields from those independent observations.
            strength_by_acquisition: dict[str, float] = {}
            reliable_by_acquisition: dict[str, float] = {}
            for evidence_event in events:
                event_acquisition = str(
                    evidence_event.get("acquisition_id", "")
                )
                if not event_acquisition:
                    continue
                event_strength = max(
                    0.0,
                    float(evidence_event.get("strength", 0.0) or 0.0),
                )
                strength_by_acquisition[event_acquisition] = max(
                    strength_by_acquisition.get(event_acquisition, 0.0),
                    event_strength,
                )
                if evidence_event.get("reliable_metric_separation") is True:
                    reliable_by_acquisition[event_acquisition] = max(
                        reliable_by_acquisition.get(event_acquisition, 0.0),
                        event_strength,
                    )
            record["distinct_acquisition_ids"] = sorted(
                strength_by_acquisition
            )[-32:]
            record["contradicting_acquisition_ids"] = sorted(
                reliable_by_acquisition
            )[-32:]
            record["distinct_identity_support"] = float(min(
                3.0, sum(strength_by_acquisition.values())
            ))
            record["reliable_distinct_support"] = float(min(
                3.0, sum(reliable_by_acquisition.values())
            ))
            record.update({
                "acquisition_id": acquisition_key,
                "station_id": event["station_id"],
                "timestamp": event["timestamp"],
                "source_observation_ids": event["source_observation_ids"],
                "identity_version": event["identity_version"],
                "geometry_version": event["geometry_version"],
                "world_distance_m": event["world_distance_m"],
                "world_separation": event["world_separation"],
                "confidence": float(min(1.0, strong_strength)),
                "shared_view_ids": event["shared_view_ids"],
                "best_iou": event["best_iou"],
                "normalized_center_separation": event[
                    "normalized_center_separation"
                ],
                "reason": event["reason"],
            })
    store["distinct_object_pairs"] = records[-256:]


def _record_must_link_hypotheses(
    store: dict[str, Any],
    observations: Sequence[Mapping[str, Any]],
    observation_to_object_id: Mapping[str, int],
    *,
    acquisition_id: str,
    geometry_version: int,
) -> None:
    """Persist soft same-track evidence without making it an irreversible merge."""
    constraints = store.setdefault(
        "identity_constraints",
        {"must_link": [], "distinct_evidence": [], "cannot_link": []},
    )
    if not isinstance(constraints, dict):
        constraints = {
            "must_link": [],
            "distinct_evidence": [],
            "cannot_link": [],
        }
        store["identity_constraints"] = constraints
    records = list(constraints.get("must_link", ()))
    known = {
        str(value.get("constraint_key", ""))
        for value in records
        if isinstance(value, Mapping)
    }
    incoming_by_object: dict[int, list[Mapping[str, Any]]] = {}
    for observation in observations:
        observation_id = str(observation.get("observation_id", ""))
        object_id = observation_to_object_id.get(observation_id)
        if not observation_id or object_id is None:
            continue
        incoming_by_object.setdefault(int(object_id), []).append(observation)
    objects = {
        int(value.get("object_id", -1)): value
        for value in store.get("objects", ())
        if isinstance(value, Mapping)
    }
    for object_id, incoming in incoming_by_object.items():
        obj = objects.get(object_id)
        if obj is None:
            continue
        prior = [
            value for value in obj.get("evidence", ())
            if str(value.get("observation_id", "")) not in {
                str(item.get("observation_id", "")) for item in incoming
            }
        ]
        if not prior:
            continue
        for observation in incoming:
            current_id = str(observation.get("observation_id", ""))
            previous = prior[-1]
            previous_id = str(previous.get("observation_id", ""))
            if not current_id or not previous_id or current_id == previous_id:
                continue
            pair = sorted((previous_id, current_id))
            key = f"{object_id}:{pair[0]}:{pair[1]}"
            if key in known:
                continue
            known.add(key)
            records.append({
                "constraint_key": key,
                "object_ids": [int(object_id)],
                "source_observation_ids": pair,
                "acquisition_id": str(acquisition_id),
                "station_id": str(observation.get("station_id", "")),
                "timestamp": float(
                    observation.get(
                        "timestamp", observation.get("timestamp_unix", time.time())
                    )
                    or time.time()
                ),
                "identity_version": int(store.get("identity_revision", 0)),
                "geometry_version": int(geometry_version),
                "evidence_type": "must_link_hypothesis",
                "hard": False,
                "confidence": float(
                    0.60
                    if str(obj.get("identity_state", ""))
                    == "TRACKED_WITH_ASSOCIATION_AMBIGUITY"
                    else 0.90
                ),
                "resolution_state": "ACTIVE",
            })
    constraints["must_link"] = records[-256:]
    constraints["distinct_evidence"] = copy.deepcopy(
        store.get("distinct_object_pairs", ())[-256:]
    )
    constraints["cannot_link"] = []


def _track_identity_metrics(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    voxel_size_m: float,
) -> dict[str, Any]:
    distance = _distance(first["center_3d"], second["center_3d"])
    first_points = _evidence_map_points(first)
    second_points = _evidence_map_points(second)
    mismatch = _pointcloud_support_score(first_points, second_points, voxel_size_m)
    appearance = _cosine_similarity(
        first.get("appearance_prototype"), second.get("appearance_prototype")
    )
    if appearance is None:
        first_evidence = list(first.get("evidence", ()))
        second_evidence = list(second.get("evidence", ()))
        if first_evidence and second_evidence:
            appearance = _cosine_similarity(
                first_evidence[-1].get("appearance_descriptor"),
                second_evidence[-1].get("appearance_descriptor"),
            )
    size = _normalized_size_similarity(first, second)
    ray_scores = [
        score
        for item in second.get("evidence", ())
        for score in [_ray_support_score(first, item)]
        if score is not None
    ]
    shared = _shared_view_box_metrics(first, second)
    fragment_metrics = _vertical_fragment_metrics(first, second)
    footprint_fragment_metrics = _contained_footprint_fragment_metrics(first, second)
    footprint_partition_metrics = _adjacent_footprint_partition_metrics(first, second)
    first_acquisitions = {
        str(item.get("acquisition_id", ""))
        for item in first.get("evidence", ())
        if isinstance(item, Mapping) and str(item.get("acquisition_id", ""))
    }
    second_acquisitions = {
        str(item.get("acquisition_id", ""))
        for item in second.get("evidence", ())
        if isinstance(item, Mapping) and str(item.get("acquisition_id", ""))
    }
    return {
        "distance_m": float(distance),
        "association_limit_m": float(_association_limit(first, second)),
        "first_center_uncertainty_radius_m": _center_uncertainty_radius(first),
        "second_center_uncertainty_radius_m": _center_uncertainty_radius(second),
        "pointcloud_similarity": None if mismatch is None else float(1.0 - mismatch),
        "pointcloud_disjoint": bool(
            len(first_points) and len(second_points) and mismatch is None
        ),
        "appearance_similarity": appearance,
        "size_similarity": size,
        "ray_similarity": None if not ray_scores else float(1.0 - min(1.0, min(ray_scores))),
        "shared_view_best_iou": shared["best_iou"],
        "shared_view_ids": shared["shared_view_ids"],
        "shared_view_center_separation": shared[
            "maximum_normalized_center_separation"
        ],
        "shared_view_horizontal_overlap_ratio": shared[
            "maximum_horizontal_overlap_ratio"
        ],
        "shared_view_vertical_gap_normalized": shared[
            "minimum_normalized_vertical_gap"
        ],
        "shared_view_fragment_compatible": _shared_view_fragment_compatible(
            first, second, shared=shared
        ),
        "shared_view_clipped_continuation": _shared_view_clipped_continuation(
            first, second
        ),
        "shared_acquisition_tile_continuation": (
            _shared_acquisition_tile_continuation(first, second)
        ),
        "fragment_footprint_compatible": bool(fragment_metrics["compatible"]),
        "fragment_horizontal_distance_m": float(fragment_metrics["horizontal_distance_m"]),
        "fragment_vertical_gap_m": float(fragment_metrics["vertical_gap_m"]),
        "fragment_horizontal_limit_m": float(fragment_metrics["horizontal_limit_m"]),
        "fragment_vertical_limit_m": float(fragment_metrics["vertical_limit_m"]),
        "contained_footprint_fragment_compatible": bool(
            footprint_fragment_metrics["compatible"]
        ),
        "fragment_small_to_large_volume_ratio": float(
            footprint_fragment_metrics["small_to_large_volume_ratio"]
        ),
        "fragment_maximum_linear_extent_ratio": float(
            footprint_fragment_metrics["maximum_linear_extent_ratio"]
        ),
        "fragment_small_center_outside_large_box_m": float(
            footprint_fragment_metrics["small_center_outside_large_box_m"]
        ),
        "fragment_containment_uncertainty_m": float(
            footprint_fragment_metrics["containment_uncertainty_m"]
        ),
        "adjacent_footprint_partition_compatible": bool(
            footprint_partition_metrics["compatible"]
        ),
        "adjacent_footprint_gap_m": float(
            footprint_partition_metrics["footprint_gap_m"]
        ),
        "adjacent_footprint_orthogonal_overlap_ratio": float(
            footprint_partition_metrics["orthogonal_overlap_ratio"]
        ),
        "adjacent_footprint_vertical_gap_m": float(
            footprint_partition_metrics["vertical_gap_m"]
        ),
        "adjacent_footprint_continuity_uncertainty_m": float(
            footprint_partition_metrics["continuity_uncertainty_m"]
        ),
        "shared_acquisition_ids": sorted(
            first_acquisitions.intersection(second_acquisitions)
        ),
        "metric_geometry_reliable": bool(
            _metric_geometry_reliable(first)
            and _metric_geometry_reliable(second)
        ),
    }


def _metric_geometry_reliable(value: Mapping[str, Any]) -> bool:
    """Return whether an observation has independent metric support.

    Covariance and extent alone are not enough for a strong distinct-instance
    claim: bearing-only lifts can carry finite covariance while their centres
    are biased.  Strict multi-view/point-cloud provenance is required before
    world separation receives more than soft weight.
    """
    items = _geometry_evidence_items(value)
    if not items:
        items = [value]
    for item in items:
        mode = str(item.get("covariance_mode", "")).lower()
        provenance = str(
            item.get("center_covariance_provenance", "")
        ).lower()
        try:
            point_count = int(item.get("lidar_support_count", 0) or 0)
        except (TypeError, ValueError):
            point_count = 0
        if (
            mode == "strict"
            and point_count >= 6
            and provenance not in {
                "",
                "bearing_only",
                "extent_derived_fallback",
                "single_point_fallback",
            }
        ):
            return True
    return False


def _identity_pair_metric_support(
    metrics: Mapping[str, Any],
) -> tuple[float, float]:
    """Convert current pair metrics into continuous same/distinct support."""
    same = 0.0
    distinct = 0.0
    appearance = metrics.get("appearance_similarity")
    cloud = metrics.get("pointcloud_similarity")
    size = metrics.get("size_similarity")
    ray = metrics.get("ray_similarity")
    try:
        if appearance is not None:
            appearance_value = float(appearance)
            if appearance_value >= 0.90:
                same += 0.85
            elif appearance_value >= 0.75:
                same += 0.40
            elif appearance_value < 0.40:
                distinct += 0.75
    except (TypeError, ValueError):
        pass
    try:
        if cloud is not None:
            cloud_value = float(cloud)
            if cloud_value >= 0.58:
                same += 0.85
            elif cloud_value >= 0.30:
                same += 0.35
    except (TypeError, ValueError):
        pass
    try:
        if size is not None and float(size) >= 0.50:
            same += 0.20
    except (TypeError, ValueError):
        pass
    try:
        if ray is not None and float(ray) >= 0.30:
            same += 0.20
    except (TypeError, ValueError):
        pass
    if float(metrics.get("shared_view_best_iou", 0.0) or 0.0) >= 0.30:
        same += 0.45
    if bool(metrics.get("shared_view_fragment_compatible", False)):
        same += 0.45
    uncertainty = (
        float(metrics.get("first_center_uncertainty_radius_m", 0.0) or 0.0)
        + float(metrics.get("second_center_uncertainty_radius_m", 0.0) or 0.0)
    )
    if float(metrics.get("distance_m", _UNAVAILABLE_METRIC)) <= 0.20 + uncertainty:
        same += 0.25
    if _shared_view_boxes_are_disjoint(metrics) and not bool(
        metrics.get("shared_view_fragment_compatible", False)
    ):
        # A same-view box gap is useful negative evidence, but intentionally
        # weak: detector fragments and partial masks commonly produce it.
        distinct += 0.25
    if (
        bool(metrics.get("metric_geometry_reliable", False))
        and _shared_view_boxes_are_disjoint(metrics)
        and float(metrics.get("distance_m", 0.0))
        >= max(0.40, 1.25 * float(metrics.get("association_limit_m", 0.45)))
    ):
        distinct += 0.80
    return float(same), float(distinct)


def _identity_pair_evidence_summary(
    store: Mapping[str, Any] | None,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse historical positive/negative pair evidence without veto state."""
    first_id = int(first.get("object_id", -1))
    second_id = int(second.get("object_id", -1))
    pair_key = _canonical_pair_key(first_id, second_id)
    same_support = 0.0
    distinct_support = 0.0
    reliable_distinct_support = 0.0
    same_acquisitions: set[str] = set()
    distinct_acquisitions: set[str] = set()
    contradicting_acquisitions: set[str] = set()
    if isinstance(store, Mapping):
        for raw in store.get("distinct_object_pairs", ()):
            if not isinstance(raw, Mapping):
                continue
            try:
                ids = sorted({
                    _resolve_object_id(store, int(value))
                    for value in raw.get("object_ids", ())
                })
            except (TypeError, ValueError):
                continue
            if len(ids) != 2 or _canonical_pair_key(ids[0], ids[1]) != pair_key:
                continue
            distinct_support += float(
                raw.get("distinct_identity_support", raw.get("confidence", 0.0))
                or 0.0
            )
            stored_reliable_support = float(
                raw.get("reliable_distinct_support", 0.0) or 0.0
            )
            distinct_acquisitions.update(
                str(value) for value in raw.get("distinct_acquisition_ids", ())
                if str(value).strip()
            )
            contradicting_acquisitions.update(
                str(value)
                for value in raw.get("contradicting_acquisition_ids", ())
                if str(value).strip()
            )
            reliable_event_support: dict[str, float] = {}
            for event in raw.get("evidence_events", ()):
                if not isinstance(event, Mapping):
                    continue
                if event.get("reliable_metric_separation") is True:
                    event_acquisition = str(event.get("acquisition_id", ""))
                    reliable_event_support[event_acquisition] = max(
                        reliable_event_support.get(event_acquisition, 0.0),
                        float(event.get("strength", 0.0) or 0.0),
                    )
            # New records store the acquisition-deduplicated aggregate.  For
            # legacy records, the event-derived value can recover a missing or
            # stale aggregate; never add both representations of the same
            # evidence.
            reliable_distinct_support += max(
                stored_reliable_support,
                sum(reliable_event_support.values()),
            )
        observed_pair_acquisitions: set[str] = set()
        for raw in store.get("ambiguous_observations", ()):
            if not isinstance(raw, Mapping):
                continue
            try:
                ids = {
                    _resolve_object_id(store, int(value))
                    for value in raw.get("candidate_object_ids", ())
                }
            except (TypeError, ValueError):
                continue
            if first_id not in ids or second_id not in ids:
                continue
            acquisition = str(raw.get("acquisition_id", ""))
            if acquisition and acquisition not in observed_pair_acquisitions:
                same_support += 0.25
                same_acquisitions.add(acquisition)
                observed_pair_acquisitions.add(acquisition)
        for obj in (first, second):
            for raw in obj.get("identity_association_hypotheses", ()):
                if not isinstance(raw, Mapping):
                    continue
                try:
                    ids = {
                        _resolve_object_id(store, int(value))
                        for value in raw.get("candidate_object_ids", ())
                    }
                except (TypeError, ValueError):
                    continue
                if first_id not in ids or second_id not in ids:
                    continue
                acquisition = str(raw.get("acquisition_id", ""))
                if acquisition and acquisition not in same_acquisitions:
                    same_support += 0.15
                    same_acquisitions.add(acquisition)
    current_same, current_distinct = _identity_pair_metric_support(metrics)
    same_support += current_same
    distinct_support += current_distinct
    return {
        "pair_key": pair_key,
        "same_identity_support": float(min(4.0, same_support)),
        "distinct_identity_support": float(min(4.0, distinct_support)),
        "reliable_distinct_support": float(min(4.0, reliable_distinct_support)),
        "supporting_acquisition_ids": sorted(same_acquisitions)[-32:],
        "distinct_acquisition_ids": sorted(distinct_acquisitions)[-32:],
        "contradicting_acquisition_ids": sorted(contradicting_acquisitions)[-32:],
        "latest_metrics": copy.deepcopy(dict(metrics)),
    }


def _track_merge_allowed(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    store: Mapping[str, Any] | None = None,
    voxel_size_m: float,
) -> tuple[bool, dict[str, Any]]:
    first_id = int(first.get("object_id", -1))
    second_id = int(second.get("object_id", -1))
    metrics = _track_identity_metrics(first, second, voxel_size_m=voxel_size_m)
    if not _class_labels_compatible(first.get("class_label"), second.get("class_label")):
        return False, {**metrics, "reason": "class_mismatch"}
    first_coverage = _coverage_provenance_ids(first)
    second_coverage = _coverage_provenance_ids(second)
    if bool(first_coverage) != bool(second_coverage):
        return False, {
            **metrics,
            "reason": "coverage_representation_mismatch",
        }
    if (
        first_coverage.intersection(_evidence_observation_ids(second))
        or second_coverage.intersection(_evidence_observation_ids(first))
    ):
        return False, {
            **metrics,
            "reason": "coverage_member_identity_boundary",
        }

    pair_evidence = _identity_pair_evidence_summary(
        store,
        first,
        second,
        metrics,
    )
    if (
        bool(metrics.get("pointcloud_disjoint", False))
        and bool(metrics.get("shared_view_ids"))
        and bool(metrics.get("metric_geometry_reliable", False))
        and not bool(metrics.get("shared_view_fragment_compatible", False))
    ):
        return False, {
            **metrics,
            "identity_pair_evidence": pair_evidence,
            "reason": "co_visible_disjoint_pointcloud_support",
        }
    # One reliable co-visible separation is direct evidence for two physical
    # instances.  Later appearance or fragment similarity cannot reverse that
    # observation into one identity; only the explicit shared-view fragment
    # contract can show that the separated geometry was one split object.
    if (
        float(pair_evidence.get("reliable_distinct_support", 0.0)) > 0.0
        and bool(pair_evidence.get("distinct_acquisition_ids", ()))
        and not bool(metrics.get("shared_view_fragment_compatible", False))
    ):
        return False, {
            **metrics,
            "identity_pair_evidence": pair_evidence,
            "reason": "co_visible_reliable_distinct_identity_evidence",
        }
    if (
        float(pair_evidence.get("reliable_distinct_support", 0.0)) >= 1.0
        and float(pair_evidence.get("distinct_identity_support", 0.0))
        > float(pair_evidence.get("same_identity_support", 0.0)) + 0.55
    ):
        return False, {
            **metrics,
            "identity_pair_evidence": pair_evidence,
            "reason": "reliable_distinct_identity_evidence",
        }

    distance = float(metrics["distance_m"])
    cloud = metrics["pointcloud_similarity"]
    appearance = metrics["appearance_similarity"]
    size = metrics["size_similarity"]
    ray = metrics["ray_similarity"]
    shared_iou = float(metrics["shared_view_best_iou"])
    same_view_distinct = bool(
        _shared_view_boxes_are_disjoint(metrics)
        and not bool(metrics.get("shared_view_fragment_compatible", False))
    )

    if appearance is not None and appearance < 0.35:
        return False, {**metrics, "reason": "appearance_contradiction"}

    strong_cloud = cloud is not None and cloud >= 0.58 and distance <= max(
        0.85, 1.25 * float(metrics["association_limit_m"])
    )
    strong_appearance = (
        appearance is not None
        and appearance >= 0.90
        and distance <= 0.45
        and (size is None or size >= 0.30)
        and (ray is None or ray >= 0.30)
        # Two non-overlapping boxes in the same real image are distinct
        # hypotheses on their first observation.  Appearance alone (which is
        # often nearly identical for paired decorations) may not collapse
        # them before a later acquisition supplies reversible positive
        # association evidence.
        and not same_view_distinct
    )
    shared_view_duplicate = (
        shared_iou >= 0.45
        and distance <= 0.55
        and ((cloud is not None and cloud >= 0.15) or (appearance is not None and appearance >= 0.72))
    )
    # A detector split can have biased centers when one track is built from a
    # broad bearing-only box and the other from a precise depth-supported
    # fragment.  Repeated same-view overlap is then stronger identity
    # evidence than the distance between the two accumulated centroids.  It
    # is deliberately a positive, multi-view evidence rule: one accidental
    # overlap is not enough, and a stored ambiguity hypothesis is not a
    # permanent cannot-link constraint.
    repeated_shared_view_duplicate = bool(
        len(metrics.get("shared_view_ids", ())) >= 2
        and shared_iou >= 0.30
        and distance <= float(metrics["association_limit_m"])
        and appearance is not None
        and appearance >= 0.75
    )
    first_uncertainty = float(metrics["first_center_uncertainty_radius_m"])
    second_uncertainty = float(metrics["second_center_uncertainty_radius_m"])
    uncertainty_consistent = distance <= 0.20 + first_uncertainty + second_uncertainty
    # Partial views of one large object can have stable but separated visible
    # centroids.  Repeated association competition may cross that centroid
    # gate only when independent appearance, ray, and size evidence all agree
    # and no co-visible or stored distinct-instance evidence exists.
    repeated_unopposed_cross_view_identity = bool(
        len(pair_evidence.get("supporting_acquisition_ids", ())) >= 2
        and not metrics.get("shared_view_ids")
        and float(pair_evidence.get("reliable_distinct_support", 0.0)) == 0.0
        and float(pair_evidence.get("distinct_identity_support", 0.0)) == 0.0
        and float(pair_evidence.get("same_identity_support", 0.0)) >= 1.50
        and appearance is not None
        and appearance >= 0.90
        and ray is not None
        and ray >= 0.60
        and size is not None
        and size >= 0.35
    )
    repeated_contained_footprint_fragment = bool(
        len(pair_evidence.get("supporting_acquisition_ids", ())) >= 2
        and not metrics.get("shared_view_ids")
        and float(pair_evidence.get("reliable_distinct_support", 0.0)) == 0.0
        and float(pair_evidence.get("distinct_identity_support", 0.0)) == 0.0
        and float(pair_evidence.get("same_identity_support", 0.0)) >= 1.0
        and bool(metrics.get("metric_geometry_reliable", False))
        and bool(metrics.get("contained_footprint_fragment_compatible", False))
        and appearance is not None
        and appearance >= 0.75
        and ray is not None
        and ray >= 0.60
    )
    # A clipped cross-view mask of a large object can be a one-off event: the
    # robot may never return to the exact edge view that produced it.  When
    # that fragment is strictly contained by an established 3-D footprint,
    # independent appearance and bearing evidence agree, and no co-visible or
    # stored separation evidence exists, waiting for a repeated Hungarian
    # ambiguity only preserves a duplicate physical identity indefinitely.
    direct_contained_footprint_fragment = bool(
        not metrics.get("shared_view_ids")
        and not metrics.get("shared_acquisition_ids")
        and float(pair_evidence.get("reliable_distinct_support", 0.0)) == 0.0
        and float(pair_evidence.get("distinct_identity_support", 0.0)) == 0.0
        and bool(metrics.get("metric_geometry_reliable", False))
        and bool(metrics.get("contained_footprint_fragment_compatible", False))
        and appearance is not None
        and appearance >= 0.85
        and ray is not None
        and ray >= 0.60
    )
    # Repeated observations can partition one extended surface into adjacent
    # tracks rather than a full track plus a contained fragment.  Merge only
    # when the map footprints meet within measured uncertainty, the same-view
    # evidence contains a boundary continuation, two acquisitions saw both
    # tracks, and no stored separation evidence opposes the identity.
    repeated_boundary_continuation_partition = bool(
        len(metrics.get("shared_acquisition_ids", ())) >= 2
        and bool(metrics.get("shared_view_clipped_continuation", False))
        and bool(metrics.get("adjacent_footprint_partition_compatible", False))
        and bool(metrics.get("metric_geometry_reliable", False))
        and float(pair_evidence.get("reliable_distinct_support", 0.0)) == 0.0
        and float(pair_evidence.get("distinct_identity_support", 0.0)) == 0.0
        and bool(pair_evidence.get("supporting_acquisition_ids", ()))
        and appearance is not None
        and appearance >= 0.90
        and ray is not None
        and ray >= 0.60
        and size is not None
        and size >= 0.35
    )
    # A mature extended track and a new surface crop can carry direct
    # same-identity evidence before the crop is repeated.  Two complementary
    # evidence shapes are valid here: a large same-view overlap whose map
    # footprints touch, or a contained cross-view crop whose especially strong
    # bearing and non-zero cloud agreement compensate for viewpoint-dependent
    # appearance.  Neither path may override any stored distinct-instance
    # evidence.
    direct_shared_view_surface_partition = bool(
        bool(metrics.get("shared_acquisition_ids", ()))
        and float(metrics.get("shared_view_best_iou", 0.0)) >= 0.60
        and float(metrics.get("adjacent_footprint_gap_m", _UNAVAILABLE_METRIC))
        <= float(metrics.get("adjacent_footprint_continuity_uncertainty_m", 0.0))
        and bool(metrics.get("metric_geometry_reliable", False))
        and float(pair_evidence.get("reliable_distinct_support", 0.0)) == 0.0
        and float(pair_evidence.get("distinct_identity_support", 0.0)) == 0.0
        and appearance is not None
        and appearance >= 0.90
        and ray is not None
        and ray >= 0.60
        and size is not None
        and size >= 0.35
    )
    # One panorama can split an extended object across two perspective tiles.
    # Complementary clipped tile edges are the image-bearing identity signal
    # in that case; requiring same-view IoU or crossing rays would make the
    # seam evidence unreachable by construction.
    direct_panorama_tile_surface_partition = bool(
        bool(metrics.get("shared_acquisition_ids", ()))
        and bool(metrics.get("shared_acquisition_tile_continuation", False))
        and bool(metrics.get("adjacent_footprint_partition_compatible", False))
        and bool(metrics.get("metric_geometry_reliable", False))
        and float(pair_evidence.get("reliable_distinct_support", 0.0)) == 0.0
        and float(pair_evidence.get("distinct_identity_support", 0.0)) == 0.0
        and str(first.get("semantic_status", "")).lower() == "verified"
        and str(second.get("semantic_status", "")).lower() == "verified"
        and appearance is not None
        and appearance >= 0.90
        and size is not None
        and size >= 0.35
    )
    direct_compound_contained_fragment = bool(
        not metrics.get("shared_view_ids")
        and not metrics.get("shared_acquisition_ids")
        and bool(metrics.get("contained_footprint_fragment_compatible", False))
        and bool(metrics.get("metric_geometry_reliable", False))
        and float(pair_evidence.get("reliable_distinct_support", 0.0)) == 0.0
        and float(pair_evidence.get("distinct_identity_support", 0.0)) == 0.0
        and appearance is not None
        and appearance >= 0.75
        and ray is not None
        and ray >= 0.90
        and cloud is not None
        and cloud >= 0.03
    )
    supporting_acquisitions = set(
        str(value)
        for value in pair_evidence.get("supporting_acquisition_ids", ())
        if str(value)
    )
    shared_track_acquisitions = set(
        str(value)
        for value in metrics.get("shared_acquisition_ids", ())
        if str(value)
    )
    contained_fragment_cross_acquisition_reassociation = bool(
        supporting_acquisitions
        and shared_track_acquisitions
        and supporting_acquisitions.difference(shared_track_acquisitions)
        and not metrics.get("shared_view_ids")
        and float(pair_evidence.get("reliable_distinct_support", 0.0)) == 0.0
        and float(pair_evidence.get("distinct_identity_support", 0.0)) == 0.0
        and float(pair_evidence.get("same_identity_support", 0.0)) >= 0.60
        and bool(metrics.get("metric_geometry_reliable", False))
        and bool(metrics.get("contained_footprint_fragment_compatible", False))
        and appearance is not None
        and appearance >= 0.85
        and (ray is None or ray >= 0.60)
    )
    repeated_association_duplicate = bool(
        (
            len(pair_evidence.get("supporting_acquisition_ids", ())) >= 2
            or contained_fragment_cross_acquisition_reassociation
        )
        and float(pair_evidence.get("same_identity_support", 0.0))
        >= float(pair_evidence.get("distinct_identity_support", 0.0)) + 0.20
        and (
            distance <= float(metrics["association_limit_m"])
            or repeated_contained_footprint_fragment
            or contained_fragment_cross_acquisition_reassociation
        )
        and (
            uncertainty_consistent
            or repeated_unopposed_cross_view_identity
            or repeated_contained_footprint_fragment
            or contained_fragment_cross_acquisition_reassociation
        )
        and (appearance is None or appearance >= 0.68)
        and (
            size is None
            or size >= 0.20
            or repeated_contained_footprint_fragment
            or contained_fragment_cross_acquisition_reassociation
        )
    )
    # Fragment reconciliation must carry an identity-bearing signal that is
    # independent of mere footprint proximity/size.  Overlapping detections
    # of two adjacent physical instances can otherwise satisfy the geometric
    # fragment envelope and be irreversibly collapsed before later views can
    # preserve their multiplicity.  A deliberately aligned shared-view
    # fragment remains valid evidence; generic box overlap is already handled
    # by ``shared_view_duplicate`` above when appearance/cloud support agrees.
    fragment_identity_signals = [
        cloud is not None and cloud >= 0.10,
        appearance is not None and appearance >= 0.68,
        ray is not None and ray >= 0.18,
        bool(metrics.get("shared_view_fragment_compatible", False)),
    ]
    fragment_support_signals = [
        *fragment_identity_signals,
        size is not None and size >= 0.20,
        uncertainty_consistent,
    ]
    fragment_distance_limit = min(
        0.95,
        max(0.60, 0.36 + first_uncertainty + second_uncertainty),
    )
    fragment_footprint_compatible = bool(
        metrics.get("fragment_footprint_compatible", False)
    )
    supported_fragment_duplicate = (
        (distance <= fragment_distance_limit or fragment_footprint_compatible)
        and (appearance is None or appearance >= 0.50)
        and not same_view_distinct
        and fragment_footprint_compatible
        and any(fragment_identity_signals)
        and sum(bool(value) for value in fragment_support_signals) >= 2
    )
    allowed = (
        strong_cloud
        or strong_appearance
        or shared_view_duplicate
        or repeated_shared_view_duplicate
        or repeated_association_duplicate
        or direct_contained_footprint_fragment
        or repeated_boundary_continuation_partition
        or direct_shared_view_surface_partition
        or direct_panorama_tile_surface_partition
        or direct_compound_contained_fragment
        or supported_fragment_duplicate
    )
    reason = (
        "strong_symmetric_cloud"
        if strong_cloud
        else "strong_appearance_geometry"
        if strong_appearance
        else "shared_view_duplicate"
        if shared_view_duplicate
        else "repeated_shared_view_detector_split"
        if repeated_shared_view_duplicate
        else "contained_fragment_cross_acquisition_reassociation"
        if (
            repeated_association_duplicate
            and contained_fragment_cross_acquisition_reassociation
        )
        else "repeated_cross_acquisition_association_competition"
        if repeated_association_duplicate
        else "direct_contained_cross_view_fragment"
        if direct_contained_footprint_fragment
        else "repeated_boundary_continuation_partition"
        if repeated_boundary_continuation_partition
        else "direct_shared_view_surface_partition"
        if direct_shared_view_surface_partition
        else "direct_panorama_tile_surface_partition"
        if direct_panorama_tile_surface_partition
        else "direct_compound_contained_fragment"
        if direct_compound_contained_fragment
        else "supported_vertical_fragment"
        if supported_fragment_duplicate
        else "insufficient_compound_identity_support"
    )
    return bool(allowed), {
        **metrics,
        "identity_pair_evidence": pair_evidence,
        "reason": reason,
    }


def _canonical_track_rank(obj: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(str(obj.get("semantic_status", "")) == "verified"),
        int(str(obj.get("physical_status", obj.get("status", ""))) == "confirmed"),
        int(obj.get("independent_viewpoint_count", 0) or 0),
        int(obj.get("station_count", 0) or 0),
        len(obj.get("evidence", ())),
        int(obj.get("geometry_point_count", 0) or 0),
        float(obj.get("semantic_probability", 0.0) or 0.0),
        -int(obj.get("object_id", 0)),
    )


def _canonicalize_store_references(store: dict[str, Any]) -> None:
    aliases = store.get("object_id_aliases", {})
    canonical_ambiguities: list[dict[str, Any]] = []
    for raw in store.get("ambiguous_observations", ()):
        record = copy.deepcopy(dict(raw))
        canonical_ids = sorted({
            _resolve_object_id(store, int(value))
            for value in record.get("candidate_object_ids", ())
        })
        if len(canonical_ids) <= 1:
            continue
        record["candidate_object_ids"] = canonical_ids
        canonical_ambiguities.append(record)
    store["ambiguous_observations"] = canonical_ambiguities[-128:]

    canonical_distinct_by_key: dict[str, dict[str, Any]] = {}
    for raw in store.get("distinct_object_pairs", ()):
        record = copy.deepcopy(dict(raw))
        try:
            ids = sorted({
                _resolve_object_id(store, int(value))
                for value in record.get("object_ids", ())
            })
        except (TypeError, ValueError):
            continue
        if len(ids) != 2:
            continue
        key = _canonical_pair_key(ids[0], ids[1])
        record["pair_key"] = key
        record["object_ids"] = ids
        record["evidence_type"] = "identity_pair_evidence"
        record["hard"] = False
        record.setdefault(
            "distinct_identity_support",
            float(record.get("confidence", 0.0) or 0.0),
        )
        record.setdefault("same_identity_support", 0.0)
        record.setdefault("reliable_distinct_support", 0.0)
        record.setdefault("evidence_events", [])
        existing = canonical_distinct_by_key.get(key)
        if existing is None:
            canonical_distinct_by_key[key] = record
            continue
        merged_events = [
            dict(value)
            for value in (
                *existing.get("evidence_events", ()),
                *record.get("evidence_events", ()),
            )
            if isinstance(value, Mapping)
        ]
        deduplicated_events: dict[tuple[str, tuple[str, ...], str], dict[str, Any]] = {}
        for event in merged_events:
            event_key = (
                str(event.get("acquisition_id", "")),
                tuple(sorted(
                    str(value)
                    for value in event.get("source_observation_ids", ())
                )),
                str(event.get("reason", "")),
            )
            prior = deduplicated_events.get(event_key)
            if prior is None or float(event.get("strength", 0.0) or 0.0) > float(
                prior.get("strength", 0.0) or 0.0
            ):
                deduplicated_events[event_key] = event
        existing["evidence_events"] = list(deduplicated_events.values())[-64:]
        for field in (
            "distinct_acquisition_ids",
            "same_supporting_acquisition_ids",
            "contradicting_acquisition_ids",
        ):
            existing[field] = sorted({
                str(value)
                for value in (*existing.get(field, ()), *record.get(field, ()))
                if str(value).strip()
            })[-32:]
        existing["same_identity_support"] = max(
            float(existing.get("same_identity_support", 0.0) or 0.0),
            float(record.get("same_identity_support", 0.0) or 0.0),
        )
        existing["distinct_identity_support"] = max(
            float(existing.get("distinct_identity_support", 0.0) or 0.0),
            float(record.get("distinct_identity_support", 0.0) or 0.0),
        )
        existing["reliable_distinct_support"] = max(
            float(existing.get("reliable_distinct_support", 0.0) or 0.0),
            float(record.get("reliable_distinct_support", 0.0) or 0.0),
        )
        if float(record.get("timestamp", 0.0) or 0.0) >= float(
            existing.get("timestamp", 0.0) or 0.0
        ):
            for field in (
                "acquisition_id",
                "station_id",
                "timestamp",
                "source_observation_ids",
                "identity_version",
                "geometry_version",
                "world_distance_m",
                "world_separation",
                "shared_view_ids",
                "best_iou",
                "normalized_center_separation",
                "reason",
            ):
                if field in record:
                    existing[field] = copy.deepcopy(record[field])
    canonical_distinct = list(canonical_distinct_by_key.values())
    for record in canonical_distinct:
        strength_by_acquisition: dict[str, float] = {}
        reliable_by_acquisition: dict[str, float] = {}
        for event in record.get("evidence_events", ()):
            acquisition = str(event.get("acquisition_id", ""))
            if not acquisition:
                continue
            strength = max(0.0, float(event.get("strength", 0.0) or 0.0))
            strength_by_acquisition[acquisition] = max(
                strength_by_acquisition.get(acquisition, 0.0), strength
            )
            if event.get("reliable_metric_separation") is True:
                reliable_by_acquisition[acquisition] = max(
                    reliable_by_acquisition.get(acquisition, 0.0), strength
                )
        if strength_by_acquisition:
            record["distinct_acquisition_ids"] = sorted(
                strength_by_acquisition
            )[-32:]
            record["distinct_identity_support"] = float(min(
                3.0, sum(strength_by_acquisition.values())
            ))
            record["reliable_distinct_support"] = float(min(
                3.0, sum(reliable_by_acquisition.values())
            ))
    store["distinct_object_pairs"] = canonical_distinct[-256:]

    constraints = store.get("identity_constraints")
    if isinstance(constraints, Mapping):
        normalized_must_link: list[dict[str, Any]] = []
        for raw in constraints.get("must_link", ()):
            if not isinstance(raw, Mapping):
                continue
            record = copy.deepcopy(dict(raw))
            try:
                record["object_ids"] = sorted({
                    _resolve_object_id(store, int(value))
                    for value in record.get("object_ids", ())
                })
            except (TypeError, ValueError):
                continue
            if not record["object_ids"]:
                continue
            normalized_must_link.append(record)
        constraints["must_link"] = normalized_must_link[-256:]
        constraints["distinct_evidence"] = copy.deepcopy(
            store.get("distinct_object_pairs", ())[-256:]
        )
        constraints["cannot_link"] = []

    for record in store.get("viewpoint_history", ()):
        by_class = record.get("object_ids_by_class", {})
        if not isinstance(by_class, Mapping):
            continue
        record["object_ids_by_class"] = {
            str(class_name): sorted({
                _resolve_object_id(store, int(value)) for value in values
            })
            for class_name, values in by_class.items()
        }


def _merge_track_group(
    store: dict[str, Any],
    members: Sequence[dict[str, Any]],
    *,
    minimum_separation_m: float,
    voxel_size_m: float,
    pair_metrics: Sequence[Mapping[str, Any]],
) -> int:
    canonical = max(members, key=_canonical_track_rank)
    canonical_id = int(canonical["object_id"])
    member_ids = sorted({
        int(value)
        for member in members
        for value in member.get("identity_member_ids", (member.get("object_id"),))
    })
    evidence_by_id: dict[str, dict[str, Any]] = {}
    verifications: list[dict[str, Any]] = []
    association_hypotheses: list[dict[str, Any]] = []
    for member in members:
        for item in member.get("evidence", ()):
            observation_id = str(item.get("observation_id", ""))
            if observation_id:
                evidence_by_id[observation_id] = copy.deepcopy(dict(item))
        verifications.extend(copy.deepcopy(list(member.get("semantic_verifications", ()))))
        association_hypotheses.extend(
            copy.deepcopy(list(member.get("identity_association_hypotheses", ())))
        )
    canonical["evidence"] = list(evidence_by_id.values())
    canonical["semantic_verifications"] = verifications[-64:]
    canonical["identity_association_hypotheses"] = association_hypotheses[-64:]
    canonical["identity_member_ids"] = member_ids
    canonical["instance_version"] = max(
        int(member.get("instance_version", 0)) for member in members
    ) + 1

    aliases = store.setdefault("object_id_aliases", {})
    for member_id in member_ids:
        if member_id != canonical_id:
            aliases[str(member_id)] = canonical_id
    for raw_alias in list(aliases):
        try:
            if _resolve_object_id(store, int(raw_alias)) in member_ids:
                aliases[str(raw_alias)] = canonical_id
        except (TypeError, ValueError):
            continue

    removed = {int(member["object_id"]) for member in members if member is not canonical}
    store["objects"] = [
        obj for obj in store.get("objects", ()) if int(obj.get("object_id", -1)) not in removed
    ]
    _refresh_geometry_from_evidence(canonical, voxel_size_m)
    _recompute_track_state(canonical, minimum_separation_m)
    store["identity_revision"] = int(store.get("identity_revision", 0)) + 1
    events = list(store.get("identity_merge_events", ()))
    events.append({
        "identity_revision": int(store["identity_revision"]),
        "canonical_object_id": canonical_id,
        "merged_object_ids": sorted(removed),
        "member_object_ids": member_ids,
        "class_label": str(canonical.get("class_label", "")),
        "pair_metrics": [copy.deepcopy(dict(value)) for value in pair_metrics],
        "reason": "complete_link_identity_reconciliation",
    })
    store["identity_merge_events"] = events[-64:]
    return canonical_id


def _reconcile_duplicate_tracks(
    store: dict[str, Any],
    *,
    minimum_separation_m: float,
    voxel_size_m: float,
) -> dict[int, int]:
    """Merge only complete-link duplicate groups, preserving distinct evidence."""
    objects = [value for value in store.get("objects", ())]
    by_class: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        canonical_class = _canonical_class_label(obj.get("class_label", ""))
        obj["class_label"] = canonical_class
        by_class.setdefault(canonical_class, []).append(obj)
    merged_to: dict[int, int] = {}

    for class_objects in by_class.values():
        groups: list[list[dict[str, Any]]] = [[obj] for obj in class_objects]
        group_metrics: dict[tuple[int, int], list[dict[str, Any]]] = {}
        changed = True
        while changed:
            changed = False
            best_choice: tuple[float, int, int, list[dict[str, Any]]] | None = None
            for first_index in range(len(groups)):
                for second_index in range(first_index + 1, len(groups)):
                    metrics_records: list[dict[str, Any]] = []
                    all_allowed = True
                    quality_values: list[float] = []
                    for first in groups[first_index]:
                        for second in groups[second_index]:
                            allowed, metrics = _track_merge_allowed(
                                first,
                                second,
                                store=store,
                                voxel_size_m=voxel_size_m,
                            )
                            metrics_records.append({
                                "first_object_id": int(first["object_id"]),
                                "second_object_id": int(second["object_id"]),
                                **metrics,
                            })
                            if not allowed:
                                all_allowed = False
                                break
                            cloud = metrics.get("pointcloud_similarity")
                            appearance = metrics.get("appearance_similarity")
                            quality_values.append(max(
                                float(cloud) if cloud is not None else 0.0,
                                float(appearance) if appearance is not None else 0.0,
                                float(metrics.get("shared_view_best_iou", 0.0)),
                            ))
                        if not all_allowed:
                            break
                    if not all_allowed:
                        continue
                    quality = min(quality_values) if quality_values else 0.0
                    choice = (quality, first_index, second_index, metrics_records)
                    if best_choice is None or choice[0] > best_choice[0]:
                        best_choice = choice
            if best_choice is None:
                break
            _, first_index, second_index, metrics_records = best_choice
            groups[first_index] = groups[first_index] + groups[second_index]
            groups.pop(second_index)
            group_metrics[(id(groups[first_index]), len(groups[first_index]))] = metrics_records
            changed = True

        for group in groups:
            if len(group) <= 1:
                continue
            metrics_records: list[dict[str, Any]] = []
            for first_index, first in enumerate(group):
                for second in group[first_index + 1 :]:
                    _, metrics = _track_merge_allowed(
                        first,
                        second,
                        store=store,
                        voxel_size_m=voxel_size_m,
                    )
                    metrics_records.append({
                        "first_object_id": int(first["object_id"]),
                        "second_object_id": int(second["object_id"]),
                        **metrics,
                    })
            canonical_id = _merge_track_group(
                store,
                group,
                minimum_separation_m=minimum_separation_m,
                voxel_size_m=voxel_size_m,
                pair_metrics=metrics_records,
            )
            for member in group:
                merged_to[int(member["object_id"])] = canonical_id

    _canonicalize_store_references(store)
    return merged_to


def _linear_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Solve one-to-one assignment, with deterministic fallback if SciPy is absent."""
    if cost_matrix.ndim != 2 or not cost_matrix.size:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(cost_matrix)
        return [(int(row), int(column)) for row, column in zip(rows, columns)]
    except (ImportError, ValueError):
        used_rows: set[int] = set()
        used_columns: set[int] = set()
        result: list[tuple[int, int]] = []
        for flat_index in np.argsort(cost_matrix, axis=None):
            row, column = np.unravel_index(int(flat_index), cost_matrix.shape)
            row, column = int(row), int(column)
            if row in used_rows or column in used_columns:
                continue
            result.append((row, column))
            used_rows.add(row)
            used_columns.add(column)
            if len(used_rows) == cost_matrix.shape[0]:
                break
        return result


def _append_observation_ledger(
    store: dict[str, Any],
    observations: Sequence[Mapping[str, Any]],
    *,
    acquisition_id: str,
) -> dict[str, Any]:
    """Append immutable observation exports without influencing association.

    Replayed IDs retain their first record.  A content mismatch is traced but
    does not alter the legacy association path in this audit-only commit.
    """
    ledger = store.setdefault(
        "observation_ledger",
        {"schema_version": "observation_ledger_v1", "records": []},
    )
    records = ledger.setdefault("records", [])
    existing = {
        str(value.get("observation_id", "")): value
        for value in records
        if isinstance(value, Mapping) and str(value.get("observation_id", ""))
    }
    appended: list[str] = []
    replayed: list[str] = []
    conflicts: list[str] = []
    for observation in observations:
        observation_id = str(observation.get("observation_id", ""))
        record = copy.deepcopy(dict(observation))
        record["acquisition_id"] = str(acquisition_id)
        prior = existing.get(observation_id)
        if prior is None:
            records.append(record)
            existing[observation_id] = record
            appended.append(observation_id)
        elif prior == record:
            replayed.append(observation_id)
        else:
            conflicts.append(observation_id)
    return {
        "ledger_size": len(records),
        "appended_observation_ids": appended,
        "replayed_observation_ids": replayed,
        "content_conflict_observation_ids": conflicts,
    }


def _entity_transition_trace(
    before: Mapping[int, Mapping[str, Any]],
    after: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for object_id in sorted(set(before) | set(after)):
        prior = before.get(object_id)
        current = after.get(object_id)
        if prior is None:
            transitions.append({
                "entity_id": object_id,
                "transition": "BIRTH",
                "class_name": str(current.get("class_label", "")),
                "physical_after": str(current.get("physical_status", current.get("status", ""))),
                "semantic_after": str(current.get("semantic_status", "unverified")),
            })
            continue
        if current is None:
            transitions.append({
                "entity_id": object_id,
                "transition": "ALIASED_OR_REMOVED",
                "class_name": str(prior.get("class_label", "")),
            })
            continue
        fields = {
            "physical": (
                str(prior.get("physical_status", prior.get("status", ""))),
                str(current.get("physical_status", current.get("status", ""))),
            ),
            "semantic": (
                str(prior.get("semantic_status", "unverified")),
                str(current.get("semantic_status", "unverified")),
            ),
            "identity": (
                str(prior.get("identity_state", "")),
                str(current.get("identity_state", "")),
            ),
            "observation_count": (
                int(prior.get("observation_count", 0)),
                int(current.get("observation_count", 0)),
            ),
        }
        changed = {
            name: {"before": values[0], "after": values[1]}
            for name, values in fields.items() if values[0] != values[1]
        }
        if changed:
            transitions.append({
                "entity_id": object_id,
                "transition": "UPDATED",
                "class_name": str(current.get("class_label", "")),
                "changes": changed,
            })
    return transitions


def _associate_acquisition_one_to_one(
    *,
    store: dict[str, Any],
    observations: Sequence[Mapping[str, Any]],
    acquisition_id: str,
    minimum_viewpoint_separation_m: float,
    terrain_voxel_size_m: float,
    claimed_object_ids: set[int],
) -> tuple[list[int], list[dict[str, Any]], dict[str, int]]:
    """Globally assign every station-fused observation exactly once.

    A feasible existing track remains the persistent identity even when the
    assignment is close.  The observation is attached to that best track and
    the competing IDs/costs are retained as identity evidence.  Creating a new
    object for every close assignment makes repeated views look like new
    physical objects and corrupts every later count or geometric relation.
    """
    default_new_track_cost = 0.74
    impossible_cost = 1_000_000.0
    ambiguity_margin = 0.08
    associated: list[int] = []
    ambiguous: list[dict[str, Any]] = []
    observation_to_object_id: dict[str, int] = {}
    classes = sorted({_canonical_class_label(value.get("class_label", "")) for value in observations})
    for class_name in classes:
        new_track_cost = default_new_track_cost
        class_observations = [
            dict(value)
            for value in observations
            if _canonical_class_label(value.get("class_label", "")) == class_name
        ]
        tracks = [
            value
            for value in store.get("objects", ())
            if _canonical_class_label(value.get("class_label", "")) == class_name
            and int(value.get("object_id", -1)) not in claimed_object_ids
        ]
        if not class_observations:
            continue
        if not tracks:
            for observation in class_observations:
                obj = _new_object(
                    store,
                    observation,
                    acquisition_id,
                    minimum_viewpoint_separation_m,
                )
                object_id = int(obj["object_id"])
                claimed_object_ids.add(object_id)
                associated.append(object_id)
                observation_to_object_id[str(observation["observation_id"])] = object_id
            continue

        row_count = len(class_observations)
        track_count = len(tracks)
        cost_matrix = np.full(
            (row_count, track_count + row_count),
            impossible_cost,
            dtype=np.float64,
        )
        for row, observation in enumerate(class_observations):
            for column, obj in enumerate(tracks):
                candidate = _association_cost(
                    obj,
                    observation,
                    voxel_size_m=terrain_voxel_size_m,
                )
                if candidate is None:
                    continue
                cost_matrix[row, column] = float(candidate[0])
            cost_matrix[row, track_count + row] = new_track_cost

        assignments = {row: column for row, column in _linear_assignment(cost_matrix)}
        for row, observation in enumerate(class_observations):
            column = assignments.get(row, track_count + row)
            assigned_existing = (
                column < track_count and cost_matrix[row, column] < new_track_cost
            )
            feasible = sorted(
                (
                    float(cost_matrix[row, candidate_column]),
                    candidate_column,
                )
                for candidate_column in range(track_count)
                if cost_matrix[row, candidate_column] < impossible_cost
            )
            ambiguous_assignment = False
            if assigned_existing:
                assigned_cost = float(cost_matrix[row, column])
                alternatives = [
                    (cost, candidate_column)
                    for cost, candidate_column in feasible
                    if candidate_column != column
                ]
                ambiguous_assignment = bool(
                    alternatives and alternatives[0][0] - assigned_cost < ambiguity_margin
                )

            if assigned_existing:
                obj = tracks[column]
                _attach_observation(
                    obj,
                    observation,
                    acquisition_id,
                    minimum_viewpoint_separation_m,
                )
                object_id = int(obj["object_id"])
                obj["identity_state"] = "TRACKED"
                if ambiguous_assignment:
                    candidate_costs = [float(cost) for cost, _ in feasible]
                    candidate_probabilities, association_entropy = (
                        _normalized_cost_distribution(candidate_costs)
                    )
                    association_margin = (
                        float(feasible[1][0] - feasible[0][0])
                        if len(feasible) > 1 else None
                    )
                    candidate_ids = list(dict.fromkeys(
                        int(tracks[candidate_column]["object_id"])
                        for _, candidate_column in feasible
                    ))
                    record = {
                        "acquisition_id": acquisition_id,
                        "observation": copy.deepcopy(observation),
                        "created_tentative_object_id": None,
                        "resolved_object_id": object_id,
                        "candidate_object_ids": candidate_ids,
                        "candidate_costs": candidate_costs,
                        "candidate_probabilities": candidate_probabilities,
                        "association_entropy": association_entropy,
                        "association_margin": association_margin,
                        "assigned_cost": assigned_cost,
                        "resolution": "attached_to_best_existing_track",
                        "reason": "existing_track_updated_under_global_assignment_margin",
                    }
                    hypotheses = list(obj.get("identity_association_hypotheses", ()))
                    hypotheses.append({
                        "acquisition_id": acquisition_id,
                        "observation_id": str(observation["observation_id"]),
                        "candidate_object_ids": candidate_ids,
                        "candidate_costs": candidate_costs,
                        "candidate_probabilities": candidate_probabilities,
                        "association_entropy": association_entropy,
                        "association_margin": association_margin,
                        "resolved_object_id": object_id,
                        "resolution": "attached_to_best_existing_track",
                    })
                    obj["identity_association_hypotheses"] = hypotheses[-32:]
                    obj["identity_state"] = "TRACKED_WITH_ASSOCIATION_AMBIGUITY"
                    ambiguous.append(record)
                    store.setdefault("ambiguous_observations", []).append(record)
            else:
                obj = _new_object(
                    store,
                    observation,
                    acquisition_id,
                    minimum_viewpoint_separation_m,
                )
                object_id = int(obj["object_id"])
                obj["identity_state"] = "NEW_TRACK"

            claimed_object_ids.add(object_id)
            associated.append(object_id)
            observation_to_object_id[str(observation["observation_id"])] = object_id

    return associated, ambiguous, observation_to_object_id


def materialize_query_view(
    query_program: Mapping[str, Any],
    observation_ledger: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    current_snapshot: Mapping[str, Any] | None = None,
    minimum_viewpoint_separation_m: float = 0.30,
    terrain_voxel_size_m: float = 0.05,
) -> dict[str, Any]:
    """Project the current SceneMemory state for one query.

    Live callers supply ``current_snapshot`` so this layer preserves the
    identity membership already owned by SceneMemory.  The ledger-only path is
    retained for recorded diagnostics that predate current-state exports.
    """
    raw_records = (
        observation_ledger.get("records", ())
        if isinstance(observation_ledger, Mapping)
        else observation_ledger
    )
    required_classes = {
        _canonical_class_label(value.get("class_name", ""))
        for value in query_program.get("entities", ())
        if isinstance(value, Mapping)
    }
    if isinstance(current_snapshot, Mapping) and isinstance(
        current_snapshot.get("objects"), Sequence
    ):
        objects = [
            copy.deepcopy(dict(value))
            for value in current_snapshot.get("objects", ())
            if isinstance(value, Mapping)
            and _canonical_class_label(
                value.get("class_label", value.get("canonical_class", ""))
            ) in required_classes
        ]
        for obj in objects:
            obj["class_label"] = _canonical_class_label(
                obj.get("class_label", obj.get("canonical_class", ""))
            )
            _refresh_geometry_from_evidence(obj, terrain_voxel_size_m)
        objects.sort(key=lambda value: int(value["object_id"]))
        object_ids = {int(value["object_id"]) for value in objects}
        raw_constraints = current_snapshot.get("identity_constraints", {})
        constraints = (
            copy.deepcopy(dict(raw_constraints))
            if isinstance(raw_constraints, Mapping)
            else {"must_link": [], "distinct_evidence": [], "cannot_link": []}
        )
        cannot_links = [
            value for value in constraints.get("cannot_link", ())
            if isinstance(value, Mapping)
            and set(int(item) for item in value.get("entity_ids", ())).issubset(
                object_ids
            )
        ]
        reliable_distinct: list[dict[str, Any]] = []
        distinct_records = [
            *constraints.get("distinct_evidence", ()),
            *current_snapshot.get("distinct_object_pairs", ()),
        ]
        for value in distinct_records:
            if not isinstance(value, Mapping):
                continue
            entity_ids = value.get("entity_ids", value.get("object_ids", ()))
            try:
                pair = sorted({int(item) for item in entity_ids})
                reliable_support = float(
                    value.get("reliable_distinct_support", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                continue
            reliable = bool(
                value.get("hard") is True
                or reliable_support > 0.0
                or any(
                    isinstance(event, Mapping)
                    and event.get("reliable_metric_separation") is True
                    for event in value.get("evidence_events", ())
                )
            )
            if (
                len(pair) != 2
                or not reliable
                or not set(pair).issubset(object_ids)
            ):
                continue
            reliable_distinct.append({
                **copy.deepcopy(dict(value)),
                "entity_ids": pair,
                "hard": True,
            })
        by_pair = {
            tuple(value.get("entity_ids", ())): value
            for value in [*cannot_links, *reliable_distinct]
            if len(value.get("entity_ids", ())) == 2
        }
        cannot_links = list(by_pair.values())
        constraints["cannot_link"] = copy.deepcopy(cannot_links)
        cardinality_summary = _materialize_cardinality_roles(
            objects,
            cannot_links,
            voxel_size_m=terrain_voxel_size_m,
        )
        history = [
            copy.deepcopy(dict(value))
            for value in current_snapshot.get("viewpoint_history", ())
            if isinstance(value, Mapping)
        ]
        observation_to_entity = {
            str(evidence.get("observation_id", "")): int(obj["object_id"])
            for obj in objects
            for evidence in obj.get("evidence", ())
            if isinstance(evidence, Mapping)
            and str(evidence.get("observation_id", ""))
        }
        aliases = copy.deepcopy(
            current_snapshot.get("object_id_aliases", {})
        )
        provisional = sorted(
            int(value["object_id"])
            for value in objects
            if value.get("entity_lifecycle") == "NEW_SPACE_SINGLETON"
        )
        supported = sorted(
            int(value["object_id"])
            for value in objects
            if value.get("entity_lifecycle") == "SUPPORTED_ENTITY"
        )
        ambiguous = copy.deepcopy(
            current_snapshot.get("ambiguous_observations", ())
        )
        return {
            "schema_version": "query_entity_view_v1",
            "authority": "ObservationLedger+QueryProgram",
            "query_key": str(query_program.get("original_question", "")),
            "scene_version": int(current_snapshot.get("scene_version", 0)),
            "identity_revision": int(
                current_snapshot.get("identity_revision", 0)
            ),
            "geometry_version": int(
                current_snapshot.get("geometry_version", 0)
            ),
            "objects": objects,
            "object_id_aliases": aliases,
            "identity_clusters": copy.deepcopy(
                current_snapshot.get(
                    "identity_clusters",
                    _identity_clusters({
                        "objects": objects,
                        "object_id_aliases": aliases,
                    }),
                )
            ),
            "identity_constraints": constraints,
            "identity_ambiguity_groups": copy.deepcopy(
                current_snapshot.get("identity_ambiguity_groups", ())
            ),
            "ambiguous_observations": ambiguous,
            "duplicate_risk_orphans": copy.deepcopy(
                current_snapshot.get("duplicate_risk_orphans", ambiguous)
            ),
            "provisional_singleton_ids": provisional,
            "supported_entity_ids": supported,
            "cardinality_summary": cardinality_summary,
            "observation_to_entity": dict(sorted(observation_to_entity.items())),
            "viewpoint_history": history,
            "observation_ledger_size": int(
                current_snapshot.get(
                    "observation_ledger_size",
                    len(raw_records),
                )
            ),
            "place_states": [
                {"place_id": value, "kind": "pose_cluster"}
                for value in sorted({
                    str(item.get("coverage_region", ""))
                    for item in history
                    if str(item.get("coverage_region", ""))
                })
            ],
        }
    records = []
    for value in raw_records:
        if not isinstance(value, Mapping):
            continue
        class_label = _canonical_class_label(
            value.get("class_label", value.get("canonical_class", ""))
        )
        if class_label not in required_classes:
            continue
        record = copy.deepcopy(dict(value))
        record["class_label"] = class_label
        records.append(record)
    records.sort(key=lambda value: (
        float(value.get("timestamp", value.get("timestamp_unix", 0.0)) or 0.0),
        str(value.get("acquisition_id", "")),
        str(value.get("observation_id", "")),
    ))
    acquisitions: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        acquisitions.setdefault(str(record.get("acquisition_id", "")), []).append(record)

    store = _empty_store()
    store["observation_ledger"] = {
        "schema_version": "observation_ledger_v1",
        "records": copy.deepcopy(records),
    }
    cannot_links: list[dict[str, Any]] = []
    orphans: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    observation_to_entity: dict[str, int] = {}

    for acquisition_index, (acquisition_id, observations) in enumerate(acquisitions.items()):
        observations.sort(key=lambda value: str(value.get("observation_id", "")))
        before = {
            int(value["object_id"]): copy.deepcopy(value)
            for value in store["objects"]
        }
        associated, ambiguous, assignment = _associate_acquisition_one_to_one(
            store=store,
            observations=observations,
            acquisition_id=acquisition_id,
            minimum_viewpoint_separation_m=minimum_viewpoint_separation_m,
            terrain_voxel_size_m=terrain_voxel_size_m,
            claimed_object_ids=set(),
        )
        observation_to_entity.update(assignment)
        object_lookup = {int(value["object_id"]): value for value in store["objects"]}
        births = sorted(set(object_lookup).difference(before))
        observation_by_id = {
            str(value.get("observation_id", "")): value for value in observations
        }

        for observation in observations:
            first_id = assignment.get(str(observation.get("observation_id", "")))
            if first_id is None:
                continue
            for linked_id in observation.get("cannot_link_observation_ids", ()):
                second_id = assignment.get(str(linked_id))
                if second_id is None or first_id == second_id:
                    continue
                pair = sorted((int(first_id), int(second_id)))
                if any(value["entity_ids"] == pair for value in cannot_links):
                    continue
                cannot_links.append({
                    "entity_ids": pair,
                    "observation_ids": sorted((str(observation["observation_id"]), str(linked_id))),
                    "acquisition_id": acquisition_id,
                    "reason": "simultaneous_panorama_support_and_3d_core_separation",
                    "hard": True,
                })

        for object_id in births:
            obj = object_lookup[object_id]
            evidence = obj.get("evidence", ())[-1]
            prior_same_class = [
                value for value in before.values()
                if _class_labels_compatible(value.get("class_label"), obj.get("class_label"))
            ]
            distances = [
                _distance(value["center_3d"], obj["center_3d"])
                for value in prior_same_class
            ]
            covered_space = bool(distances) and min(distances) <= max(
                1.50,
                2.50 * max(float(value) for value in obj.get("bbox_3d", (0.1,))),
            )
            if acquisition_index > 0 and covered_space:
                obj["entity_lifecycle"] = "DUPLICATE_RISK_ORPHAN"
                obj["status"] = "orphan"
                obj["physical_status"] = "orphan"
                orphan = {
                    "observation_id": str(evidence.get("observation_id", "")),
                    "entity_id": object_id,
                    "candidate_entity_ids": [
                        int(value["object_id"]) for value in prior_same_class
                        if _distance(value["center_3d"], obj["center_3d"])
                        <= max(1.50, 2.50 * max(float(item) for item in obj.get("bbox_3d", (0.1,))))
                    ],
                    "reason": "unmatched_in_previously_covered_space",
                }
                orphans.append(orphan)
            else:
                obj["entity_lifecycle"] = "NEW_SPACE_SINGLETON"
                strong_single_view = bool(
                    evidence.get("qwen_verified") is True
                    and int(evidence.get("lidar_support_count", 0) or 0) >= 24
                    and evidence.get("cannot_link_observation_ids")
                )
                if strong_single_view:
                    obj["status"] = "confirmed"
                    obj["physical_status"] = "confirmed"
                    obj["entity_lifecycle"] = "SUPPORTED_ENTITY"

        for obj in store["objects"]:
            lifecycle = str(obj.get("entity_lifecycle", ""))
            if lifecycle == "DUPLICATE_RISK_ORPHAN":
                obj["status"] = "orphan"
                obj["physical_status"] = "orphan"
            elif int(obj.get("independent_viewpoint_count", 0)) >= 2:
                obj["entity_lifecycle"] = "SUPPORTED_ENTITY"
            obj["deterministic_entity_key"] = min(
                str(value.get("observation_id", ""))
                for value in obj.get("evidence", ())
                if str(value.get("observation_id", ""))
            )

        viewpoint = list(observations[0].get("viewpoint_position_map", ())) if observations else []
        coverage_region = (
            f"xy:{math.floor(float(viewpoint[0]))}:{math.floor(float(viewpoint[1]))}"
            if len(viewpoint) >= 2 else ""
        )
        view_ids = {
            str(value) for observation in observations
            for value in observation.get("source_view_ids", ())
        }
        history.append({
            "acquisition_id": acquisition_id,
            "viewpoint_position_map": viewpoint,
            "valid_for_count_closure": bool(observations and len(viewpoint) >= 3),
            "independent_viewpoint": all(
                not prior.get("viewpoint_position_map")
                or _distance(viewpoint, prior["viewpoint_position_map"])
                >= minimum_viewpoint_separation_m
                for prior in history
            ) if len(viewpoint) >= 3 else False,
            "coverage_region": coverage_region,
            "full_panorama": len(view_ids) >= 8,
            "view_count": len(view_ids),
            "object_ids_by_class": {
                class_name: sorted(
                    object_id for observation_id, object_id in assignment.items()
                    if _canonical_class_label(observation_by_id[observation_id].get("class_label")) == class_name
                )
                for class_name in sorted(required_classes)
            },
            "new_instance_ids": [
                value for value in births
                if object_lookup[value].get("entity_lifecycle") != "DUPLICATE_RISK_ORPHAN"
            ],
            "new_instance_count": sum(
                object_lookup[value].get("entity_lifecycle") != "DUPLICATE_RISK_ORPHAN"
                for value in births
            ),
            "ambiguous": bool(ambiguous),
        })

    # Preserve the established deterministic ledger assignment exactly, then
    # refresh relation-facing geometry from every observation assigned to that
    # identity. Geometry refresh must never change entity membership or IDs.
    for obj in store["objects"]:
        _refresh_geometry_from_evidence(obj, terrain_voxel_size_m)
    objects = sorted(store["objects"], key=lambda value: int(value["object_id"]))
    provisional = [
        int(value["object_id"]) for value in objects
        if value.get("entity_lifecycle") == "NEW_SPACE_SINGLETON"
    ]
    supported = [
        int(value["object_id"]) for value in objects
        if value.get("entity_lifecycle") == "SUPPORTED_ENTITY"
    ]
    constraints = {
        "must_link": [],
        "distinct_evidence": copy.deepcopy(cannot_links),
        "cannot_link": copy.deepcopy(cannot_links),
    }
    cardinality_summary = _materialize_cardinality_roles(
        objects,
        cannot_links,
        voxel_size_m=terrain_voxel_size_m,
    )
    return {
        "schema_version": "query_entity_view_v1",
        "authority": "ObservationLedger+QueryProgram",
        "query_key": str(query_program.get("original_question", "")),
        "scene_version": len(acquisitions),
        "identity_revision": len(records),
        "geometry_version": len(acquisitions),
        "objects": copy.deepcopy(objects),
        "object_id_aliases": {},
        "identity_clusters": _identity_clusters({"objects": objects, "object_id_aliases": {}}),
        "identity_constraints": constraints,
        "identity_ambiguity_groups": [],
        "ambiguous_observations": copy.deepcopy(orphans),
        "duplicate_risk_orphans": copy.deepcopy(orphans),
        "provisional_singleton_ids": provisional,
        "supported_entity_ids": supported,
        "cardinality_summary": cardinality_summary,
        "observation_to_entity": dict(sorted(observation_to_entity.items())),
        "viewpoint_history": history,
        "observation_ledger_size": len(records),
        "place_states": [
            {"place_id": value, "kind": "pose_cluster"}
            for value in sorted({item["coverage_region"] for item in history if item["coverage_region"]})
        ],
    }


def update_scene_memory(
    store_path: Path,
    observations: Sequence[Mapping[str, Any]],
    *,
    acquisition_id: str,
    minimum_viewpoint_separation_m: float = 0.30,
    terrain_map_path: str | None = None,
    accumulated_terrain_map_path: str | None = None,
    terrain_obstacle_height_threshold: float = 0.05,
    terrain_voxel_size_m: float = 0.05,
    viewpoint_position_map: Sequence[float] | None = None,
    geometry_manifest_path: str | None = None,
    panorama_view_count: int = 0,
) -> dict[str, Any]:
    """Atomically ingest one acquisition and return a canonical snapshot."""
    if not acquisition_id.strip():
        raise ValueError("acquisition_id_empty")
    store_path = store_path.resolve()
    lock_path = store_path.with_suffix(store_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        store = _load(store_path)
        store.setdefault("objects", [])
        store.setdefault("ambiguous_observations", [])
        store.setdefault("viewpoint_history", [])
        store.setdefault("object_id_aliases", {})
        store.setdefault("distinct_object_pairs", [])
        store.setdefault("identity_merge_events", [])
        store.setdefault("identity_resolution_events", [])
        store.setdefault("identity_constraints", {
            "must_link": [],
            "distinct_evidence": [],
            "cannot_link": [],
        })
        store.setdefault("identity_revision", 0)
        store.setdefault("geometry_revision", 0)
        store.setdefault("last_observation_transaction", None)
        store.setdefault(
            "observation_ledger",
            {"schema_version": "observation_ledger_v1", "records": []},
        )
        store.setdefault("acquisition_traces", [])
        store.setdefault("last_acquisition_trace", None)
        store.setdefault("identity_ambiguity_groups", [])

        objects_before = {
            int(value.get("object_id", -1)): copy.deepcopy(value)
            for value in store.get("objects", ())
            if int(value.get("object_id", -1)) >= 0
        }
        merge_event_count_before = len(store.get("identity_merge_events", ()))

        prior_canonical_object_ids = {
            _resolve_object_id(store, int(obj.get("object_id", -1)))
            for obj in store.get("objects", ())
            if int(obj.get("object_id", -1)) >= 0
            and obj.get("status") in {"tentative", "confirmed"}
        }

        validated: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for raw in observations:
            try:
                validated.append(_validate_observation(raw))
            except (TypeError, ValueError) as exc:
                rejected.append({
                    "observation_id": raw.get("observation_id"),
                    "class": raw.get("canonical_class", raw.get("class_label")),
                    "reason": str(exc),
                })
        ledger_trace = _append_observation_ledger(
            store,
            validated,
            acquisition_id=acquisition_id,
        )

        exact_objects: dict[tuple[str, str], dict[str, Any]] = {}
        for obj in store["objects"]:
            for evidence in obj.get("evidence", ()):
                observation_id = str(evidence.get("observation_id", ""))
                if observation_id:
                    exact_objects.setdefault(
                        (observation_id, str(obj.get("class_label", ""))), obj
                    )

        incoming_ids = {
            str(observation["observation_id"]) for observation in validated
        }
        store["ambiguous_observations"] = [
            record
            for record in store.get("ambiguous_observations", ())
            if str((record.get("observation") or {}).get("observation_id", ""))
            not in incoming_ids
        ]

        associated_ids: list[int] = []
        ambiguous: list[dict[str, Any]] = []
        observation_to_object_id: dict[str, int] = {}
        refreshed_observation_ids: set[str] = set()
        for observation in validated:
            observation_id = str(observation["observation_id"])
            obj = exact_objects.get((observation_id, str(observation["class_label"])))
            if obj is None:
                continue
            _attach_observation(
                obj,
                observation,
                acquisition_id,
                minimum_viewpoint_separation_m,
            )
            object_id = int(obj["object_id"])
            associated_ids.append(object_id)
            observation_to_object_id[observation_id] = object_id
            refreshed_observation_ids.add(observation_id)

        remaining_observations = [
            observation
            for observation in validated
            if str(observation["observation_id"]) not in refreshed_observation_ids
        ]
        claimed_object_ids = set(associated_ids)
        (
            globally_associated,
            global_ambiguous,
            global_observation_map,
        ) = _associate_acquisition_one_to_one(
            store=store,
            observations=remaining_observations,
            acquisition_id=acquisition_id,
            minimum_viewpoint_separation_m=minimum_viewpoint_separation_m,
            terrain_voxel_size_m=terrain_voxel_size_m,
            claimed_object_ids=claimed_object_ids,
        )
        associated_ids.extend(globally_associated)
        ambiguous.extend(global_ambiguous)
        observation_to_object_id.update(global_observation_map)

        _record_distinct_pairs(
            store,
            observation_to_object_id,
            validated,
            acquisition_id,
            geometry_version=int(store.get("geometry_revision", 0)) + 1,
            voxel_size_m=terrain_voxel_size_m,
        )
        _record_must_link_hypotheses(
            store,
            validated,
            observation_to_object_id,
            acquisition_id=acquisition_id,
            geometry_version=int(store.get("geometry_revision", 0)) + 1,
        )

        object_lookup = {int(obj["object_id"]): obj for obj in store["objects"]}
        for object_id in list(dict.fromkeys(associated_ids)):
            obj = object_lookup.get(object_id)
            if obj is None:
                continue
            _refresh_geometry_from_evidence(obj, terrain_voxel_size_m)
            _recompute_track_state(obj, minimum_viewpoint_separation_m)

        _resolve_identity_ambiguities(
            store,
            minimum_separation_m=minimum_viewpoint_separation_m,
            voxel_size_m=terrain_voxel_size_m,
        )

        # Reconcile only complete-link duplicate groups.  The reconciler uses
        # stored simultaneous-view distinctness and compound identity evidence
        # for every class; it is not a class-specific query or closure gate.
        merged_to = _reconcile_duplicate_tracks(
            store,
            minimum_separation_m=minimum_viewpoint_separation_m,
            voxel_size_m=terrain_voxel_size_m,
        )
        associated_ids = list(dict.fromkeys(
            _resolve_object_id(store, merged_to.get(object_id, object_id))
            for object_id in associated_ids
        ))
        _canonicalize_store_references(store)
        constraints = store.setdefault(
            "identity_constraints",
            {"must_link": [], "distinct_evidence": [], "cannot_link": []},
        )
        if isinstance(constraints, Mapping):
            constraints["distinct_evidence"] = copy.deepcopy(
                store.get("distinct_object_pairs", ())[-256:]
            )
            constraints["cannot_link"] = []

        object_lookup = {int(obj["object_id"]): obj for obj in store["objects"]}
        associated_ids = [
            object_id for object_id in associated_ids if object_id in object_lookup
        ]
        for obj in store["objects"]:
            _recompute_track_state(obj, minimum_viewpoint_separation_m)

        _refresh_identity_ambiguity_groups(store)

        store["version"] = int(store.get("version", 0)) + 1
        if validated:
            # Geometry and identity are separate monotonically increasing
            # revisions.  A new acquisition invalidates relation evidence even
            # when association ultimately keeps the same canonical IDs.
            store["geometry_revision"] = int(
                store.get("geometry_revision", 0)
            ) + 1
            store["identity_revision"] = int(
                store.get("identity_revision", 0)
            ) + 1
        observed_classes = sorted({
            _canonical_class_label(object_lookup[object_id].get("class_label", ""))
            for object_id in associated_ids
            if object_id in object_lookup
        })
        object_ids_by_class = {
            class_name: sorted(
                object_id
                for object_id in associated_ids
                if object_id in object_lookup
                and _canonical_class_label(object_lookup[object_id].get("class_label", "")) == class_name
            )
            for class_name in observed_classes
        }
        new_canonical_object_ids = sorted(
            set(associated_ids).difference(prior_canonical_object_ids)
        )
        new_instance_ids_by_class = {
            class_name: sorted(
                object_id for object_id in new_canonical_object_ids
                if object_id in object_lookup
                and _canonical_class_label(
                    object_lookup[object_id].get("class_label", "")
                ) == class_name
            )
            for class_name in observed_classes
        }
        viewpoint = (
            validated[0]["viewpoint_position_map"]
            if validated
            else (
                _finite_vector(viewpoint_position_map, 3)
                if viewpoint_position_map is not None
                else None
            )
        )
        finite_viewpoint = None
        if viewpoint is not None:
            try:
                finite_viewpoint = _finite_vector(viewpoint, 3)
            except (TypeError, ValueError):
                finite_viewpoint = None
        history = store["viewpoint_history"]
        prior_viewpoints = [
            record.get("viewpoint_position_map")
            for record in history
            if isinstance(record, Mapping)
            and record.get("independent_viewpoint") is True
            and isinstance(record.get("viewpoint_position_map"), Sequence)
        ]
        independent_viewpoint = bool(finite_viewpoint) and all(
            _distance(finite_viewpoint, prior)
            >= float(minimum_viewpoint_separation_m)
            for prior in prior_viewpoints
            if len(prior) >= 3
        )
        if finite_viewpoint is not None and not prior_viewpoints:
            independent_viewpoint = True
        coverage_region = (
            f"xy:{math.floor(finite_viewpoint[0]):d}:"
            f"{math.floor(finite_viewpoint[1]):d}"
            if finite_viewpoint is not None else ""
        )
        unresolved_ambiguous = [
            record
            for record in store.get("ambiguous_observations", ())
            if len({
                _resolve_object_id(store, int(value))
                for value in record.get("candidate_object_ids", ())
            }) > 1
        ]
        history.append({
            "acquisition_id": acquisition_id,
            "viewpoint_position_map": viewpoint,
            "view_count": max(0, int(panorama_view_count)),
            "full_panorama": int(panorama_view_count) >= 8,
            "valid_for_count_closure": bool(validated and finite_viewpoint),
            "independent_viewpoint": bool(independent_viewpoint),
            "coverage_region": coverage_region,
            "recorded_at_unix": float(time.time()),
            "object_ids_by_class": object_ids_by_class,
            "new_instance_ids": new_canonical_object_ids,
            "new_instance_ids_by_class": new_instance_ids_by_class,
            "new_instance_count": len(new_canonical_object_ids),
            "ambiguous": bool(unresolved_ambiguous or rejected),
            "ambiguous_classes": sorted({
                str((record.get("observation") or {}).get("class_label", ""))
                for record in unresolved_ambiguous
                if str((record.get("observation") or {}).get("class_label", ""))
            }),
            "rejected_observation_count": len(rejected),
            "identity_revision": int(store.get("identity_revision", 0)),
        })
        history[:] = history[-16:]
        _refresh_identity_ambiguity_groups(store)

        manifest_path = Path(str(geometry_manifest_path or ""))
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                manifest = {}
            if manifest.get("frame") == "map" and manifest.get("world_aligned") is True:
                store["mast3r_reconstruction"] = {
                    "geometry_manifest_path": str(manifest_path.resolve()),
                    "reconstruction_id": str(manifest.get("reconstruction_id", "")),
                    "frame": "map",
                    "mapping_mode": str(manifest.get("mapping_mode", "")),
                    "optical_center_groups": [
                        str(value) for value in manifest.get("optical_center_groups", ())
                    ],
                    "view_count": len(manifest.get("views", ())),
                }

        terrain_path = (
            Path(accumulated_terrain_map_path)
            if accumulated_terrain_map_path
            else Path(terrain_map_path)
            if terrain_map_path
            else None
        )
        terrain_facts = _terrain_observation_facts(
            terrain_path,
            obstacle_threshold=terrain_obstacle_height_threshold,
            voxel_size_m=terrain_voxel_size_m,
            viewpoint_position_map=viewpoint,
            current_terrain_path=(Path(terrain_map_path) if terrain_map_path else None),
        )
        for region in terrain_facts["observation_frontier_regions"]:
            region["region_id"] = f"scene-v{int(store['version'])}:{region['region_id']}"
        store.update(terrain_facts)
        store["terrain_memory"] = (
            {
                "frame": "map",
                "pointcloud_path": str(terrain_path.resolve()),
                "voxel_size_m": float(terrain_voxel_size_m),
                "obstacle_height_threshold": float(terrain_obstacle_height_threshold),
                "reachable_point_count": int(terrain_facts["traversable_point_count"]),
                "frontier_region_count": len(
                    terrain_facts["observation_frontier_regions"]
                ),
            }
            if terrain_path is not None and terrain_path.is_file()
            else None
        )
        _canonicalize_store_references(store)
        station_ids = sorted({
            str(value.get("station_id", ""))
            for value in validated
            if str(value.get("station_id", ""))
        })
        store["last_observation_transaction"] = {
            "transaction_id": f"{acquisition_id}:scene-memory:{int(store['version'])}",
            "acquisition_id": str(acquisition_id),
            "station_ids": station_ids,
            "accepted_observation_ids": [
                str(value.get("observation_id", "")) for value in validated
            ],
            "rejected_observations": copy.deepcopy(rejected),
            "associated_object_ids": list(associated_ids),
            "identity_version": int(store.get("identity_revision", 0)),
            "geometry_version": int(store.get("geometry_revision", 0)),
            "observation_commit_contains_new_geometry": bool(validated),
            "completed_at_unix": float(time.time()),
        }
        objects_after = {
            int(value.get("object_id", -1)): value
            for value in store.get("objects", ())
            if int(value.get("object_id", -1)) >= 0
        }
        acquisition_trace = {
            "schema_version": "scene_memory_acquisition_trace_v1",
            "acquisition_id": str(acquisition_id),
            "scene_version_before": int(store["version"]) - 1,
            "scene_version_after": int(store["version"]),
            "raw_observation_count": len(observations),
            "accepted_observation_count": len(validated),
            "rejected_observations": copy.deepcopy(rejected),
            "observation_ledger": ledger_trace,
            "observation_to_entity": [
                {
                    "observation_id": observation_id,
                    "entity_id": _resolve_object_id(store, object_id),
                }
                for observation_id, object_id in sorted(
                    observation_to_object_id.items()
                )
            ],
            "new_entity_ids": list(new_canonical_object_ids),
            "ambiguous_observation_ids": sorted(
                str((value.get("observation") or {}).get("observation_id", ""))
                for value in ambiguous
            ),
            "identity_merge_events": copy.deepcopy(
                store.get("identity_merge_events", ())[merge_event_count_before:]
            ),
            "entity_transitions": _entity_transition_trace(
                objects_before,
                objects_after,
            ),
            "entity_count_before": len(objects_before),
            "entity_count_after": len(objects_after),
            "supported_entity_count_after": sum(
                str(value.get("physical_status", value.get("status", "")))
                == "confirmed"
                for value in objects_after.values()
            ),
            "singleton_entity_count_after": sum(
                len(value.get("evidence", ())) == 1
                for value in objects_after.values()
            ),
            "cannot_link_count_after": len(
                store.get("identity_constraints", {}).get("cannot_link", ())
            ),
        }
        store["last_acquisition_trace"] = acquisition_trace
        traces = store.setdefault("acquisition_traces", [])
        traces.append(copy.deepcopy(acquisition_trace))
        traces[:] = traces[-64:]
        _atomic_write(store_path, store)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    return {
        "schema_version": "scene_memory_snapshot_v2",
        "scene_version": int(store["version"]),
        "semantic_revision": int(store.get("semantic_revision", 0)),
        "identity_revision": int(store.get("identity_revision", 0)),
        "geometry_version": int(store.get("geometry_revision", 0)),
        "acquisition_id": acquisition_id,
        "frame": "map",
        "objects": copy.deepcopy(store["objects"]),
        "object_id_aliases": copy.deepcopy(store.get("object_id_aliases", {})),
        "identity_clusters": _identity_clusters(store),
        "identity_merge_events": copy.deepcopy(
            store.get("identity_merge_events", ())[-24:]
        ),
        "identity_resolution_events": copy.deepcopy(
            store.get("identity_resolution_events", ())[-24:]
        ),
        "identity_constraints": copy.deepcopy(
            store.get("identity_constraints", {})
        ),
        "identity_ambiguity_groups": copy.deepcopy(
            store.get("identity_ambiguity_groups", ())
        ),
        "distinct_object_pairs": copy.deepcopy(
            store.get("distinct_object_pairs", ())[-128:]
        ),
        "associated_object_ids": associated_ids,
        "ambiguous_observations": copy.deepcopy(
            store.get("ambiguous_observations", ())
        ),
        "relation_evidence": copy.deepcopy(
            store.get("relation_evidence", {})
        ),
        "relation_revision": int(
            store.get("relation_evidence", {}).get("relation_revision", 0)
        ),
        "mast3r_reconstruction": copy.deepcopy(store.get("mast3r_reconstruction")),
        "rejected_observations": rejected,
        "last_observation_transaction": copy.deepcopy(
            store.get("last_observation_transaction")
        ),
        "observation_ledger": copy.deepcopy(
            store.get("observation_ledger", {})
        ),
        "observation_ledger_size": len(
            store.get("observation_ledger", {}).get("records", ())
        ),
        "last_acquisition_trace": copy.deepcopy(
            store.get("last_acquisition_trace")
        ),
        "viewpoint_history": copy.deepcopy(store.get("viewpoint_history", ())),
        "terrain_memory": copy.deepcopy(store.get("terrain_memory")),
        "observation_frontier_regions": copy.deepcopy(
            store.get("observation_frontier_regions", ())
        ),
        "traversable_point_count": int(store.get("traversable_point_count", 0)),
        "scene_memory_path": str(store_path),
        "geometry_reconstruction_ids": sorted({
            item["reconstruction_id"]
            for item in validated
            if item["reconstruction_id"]
        }),
    }


# Hungarian matching for one-to-one track association
def hungarian_match(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Return a globally one-to-one minimum-cost assignment."""
    matrix = np.asarray(cost_matrix, dtype=np.float64)
    return _linear_assignment(matrix)


def compute_iou(bbox1: list, bbox2: list) -> float:
    """Compute IoU between two bboxes [x_min, y_min, x_max, y_max]."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_iou_cost_matrix(observations: list, tracks: list) -> np.ndarray:
    """Compute IoU-based cost matrix (lower cost = better match)."""
    n_obs = len(observations)
    n_tracks = len(tracks)

    cost_matrix = np.ones((n_obs, n_tracks), dtype=np.float32)

    for i, obs in enumerate(observations):
        for j, track in enumerate(tracks):
            iou = compute_iou(obs["bbox"], track["bbox"])
            cost_matrix[i, j] = 1.0 - iou  # Convert IoU to cost

    return cost_matrix


def compute_cost_matrix(observations: list, tracks: list, use_class_penalty: bool = True) -> np.ndarray:
    """Compute cost matrix with IoU and optional class penalty."""
    cost_matrix = compute_iou_cost_matrix(observations, tracks)

    if use_class_penalty:
        for i, obs in enumerate(observations):
            for j, track in enumerate(tracks):
                if obs.get("class_name") != track.get("class_name"):
                    cost_matrix[i, j] += 0.5  # Class mismatch penalty

    return cost_matrix


def associate_observations_to_tracks(observations: list, tracks: list) -> dict:
    """Associate observations to tracks using Hungarian matching."""
    if len(observations) == 0 or len(tracks) == 0:
        return {
            "matches": [],
            "unmatched_observations": list(range(len(observations))),
            "unmatched_tracks": list(range(len(tracks))),
        }

    cost_matrix = compute_cost_matrix(observations, tracks)
    matches = hungarian_match(cost_matrix)

    # Filter low-quality matches (IoU threshold)
    good_matches = []
    for obs_idx, track_idx in matches:
        iou = 1.0 - cost_matrix[obs_idx, track_idx]
        if iou > 0.3:  # Minimum IoU threshold
            good_matches.append((obs_idx, track_idx))

    matched_obs = {m[0] for m in good_matches}
    matched_tracks = {m[1] for m in good_matches}

    return {
        "matches": good_matches,
        "unmatched_observations": [i for i in range(len(observations)) if i not in matched_obs],
        "unmatched_tracks": [i for i in range(len(tracks)) if i not in matched_tracks],
    }
