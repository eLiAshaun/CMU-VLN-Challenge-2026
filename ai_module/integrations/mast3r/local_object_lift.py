#!/usr/bin/env python3
"""Lift verified masks through SGA pointmaps already aligned to ROS map."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback

import cv2
import numpy as np


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _quaternion_rotation(xyzw: list[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in xyzw)
    norm = np.linalg.norm([x, y, z, w])
    if norm <= 0:
        raise ValueError("state_estimation_quaternion_invalid")
    x, y, z, w = np.asarray([x, y, z, w], dtype=np.float64) / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _map_from_sensor(state: dict) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quaternion_rotation(state["orientation_xyzw"])
    matrix[:3, 3] = np.asarray(state["position_xyz"], dtype=np.float64)
    return matrix


def _lidar_mask_support(
    sensor_points: np.ndarray,
    panorama_mask: np.ndarray,
    sensor_to_camera: np.ndarray,
    adapter_to_camera: np.ndarray,
    panorama_vertical_fov_deg: float,
    *,
    dilate: bool = True,
) -> np.ndarray:
    camera_canonical = _transform(sensor_points, sensor_to_camera)
    camera_to_adapter = np.linalg.inv(adapter_to_camera)
    adapter = _transform(camera_canonical, camera_to_adapter)
    ranges = np.linalg.norm(adapter, axis=1)
    finite = np.isfinite(adapter).all(axis=1) & (ranges > 1e-4)
    longitude = np.arctan2(adapter[:, 0], adapter[:, 2])
    latitude = np.arcsin(np.clip(-adapter[:, 1] / np.maximum(ranges, 1e-4), -1.0, 1.0))
    height, width = panorama_mask.shape
    columns = np.rint((longitude / (2.0 * np.pi) + 0.5) * width - 0.5).astype(np.int64) % width
    vertical_fov = np.radians(float(panorama_vertical_fov_deg))
    rows = np.rint((0.5 - latitude / vertical_fov) * height - 0.5).astype(np.int64)
    valid = finite & (rows >= 0) & (rows < height)
    sampled_mask = (
        cv2.dilate(
            panorama_mask.astype(np.uint8),
            np.ones((5, 5), np.uint8),
        ) > 0
        if dilate
        else panorama_mask > 0
    )
    supported = np.zeros(len(sensor_points), dtype=bool)
    supported[valid] = sampled_mask[rows[valid], columns[valid]]
    return supported


def _nearest_registered_distances(
    points: np.ndarray, registered_points: np.ndarray
) -> np.ndarray:
    """Bound memory and latency while checking map-cloud consistency."""
    if not len(points) or not len(registered_points):
        return np.zeros((0,), dtype=np.float32)
    if len(points) > 1000:
        points = points[np.linspace(0, len(points) - 1, 1000, dtype=np.int64)]
    if len(registered_points) > 20000:
        registered_points = registered_points[
            np.linspace(0, len(registered_points) - 1, 20000, dtype=np.int64)
        ]
    nearest = np.full(len(points), np.inf, dtype=np.float32)
    for start in range(0, len(points), 100):
        query = points[start:start + 100]
        distances_squared = np.sum(
            (query[:, None, :] - registered_points[None, :, :]) ** 2,
            axis=-1,
        )
        nearest[start:start + len(query)] = np.sqrt(distances_squared.min(axis=1))
    return nearest


def _mask_geometry_candidate(
    *,
    source: str,
    points: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    confidence_threshold: float,
    sample_budget: int,
    registered_points: np.ndarray,
) -> dict | None:
    """Measure one dense geometry source inside one semantic instance mask."""
    if mask.shape != points.shape[:2] or confidence.shape != mask.shape:
        raise ValueError("mask_pointmap_shape_mismatch")
    valid = (
        (mask > 0)
        & np.isfinite(points).all(axis=-1)
        & np.isfinite(confidence)
        & (confidence > 0.0)
    )
    selected = points[valid].astype(np.float32)
    selected_confidence = confidence[valid].astype(np.float32)
    if not len(selected):
        return None
    if len(selected) > sample_budget:
        indices = np.linspace(
            0, len(selected) - 1, sample_budget, dtype=np.int64
        )
        sample = selected[indices]
        sample_confidence = selected_confidence[indices]
    else:
        sample = selected
        sample_confidence = selected_confidence
    distances = _nearest_registered_distances(sample, registered_points)
    registered_median = (
        float(np.median(distances)) if len(distances) else None
    )
    median_confidence = float(np.median(selected_confidence))
    confidence_scale = max(float(confidence_threshold), 1e-6)
    source_quality = float(np.log1p(median_confidence / confidence_scale))
    if registered_median is not None:
        source_quality *= float(np.exp(-registered_median / 0.25))
    low = np.percentile(selected, 2.5, axis=0)
    high = np.percentile(selected, 97.5, axis=0)
    center = 0.5 * (low + high)
    extent = np.maximum(high - low, 0.02)
    return {
        "source": source,
        "point_count": int(len(selected)),
        "sampled_point_count": int(len(sample)),
        "center_3d": center.astype(np.float64),
        "bbox_3d": extent.astype(np.float64),
        "median_confidence": median_confidence,
        "source_quality": source_quality,
        "registered_scan_median_distance_m": registered_median,
        "registered_scan_inlier_fraction_0_25m": (
            float(np.mean(distances <= 0.25)) if len(distances) else 0.0
        ),
        "sample": sample,
        "sample_confidence": sample_confidence,
    }


def _weighted_coordinate_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = []
    normalized = np.maximum(np.asarray(weights, dtype=np.float64), 1e-9)
    for axis in range(values.shape[1]):
        order = np.argsort(values[:, axis])
        cumulative = np.cumsum(normalized[order])
        index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
        result.append(float(values[order[min(index, len(order) - 1)], axis]))
    return np.asarray(result, dtype=np.float64)


def _common_radial_mode(samples_by_view: list[np.ndarray]) -> float:
    """Estimate one latent object range shared by overlapping camera views.

    Dense completion commonly mixes foreground and background at a small
    instance boundary.  Treating every completed pixel as one solid 3D box
    turns that mixture into metres of false radial extent.  Overlapping views
    from the same optical centre observe the same physical range, so combine
    their one-dimensional kernel densities with equal per-view weight.  The
    bandwidth is estimated from each sample distribution (Silverman's rule),
    rather than from an object class or relation predicate.
    """
    finite_samples = [
        np.asarray(values, dtype=np.float64)[
            np.isfinite(np.asarray(values, dtype=np.float64))
            & (np.asarray(values, dtype=np.float64) > 1e-6)
        ]
        for values in samples_by_view
    ]
    finite_samples = [values for values in finite_samples if len(values)]
    if not finite_samples:
        raise ValueError("radial_geometry_samples_empty")
    low = min(float(np.percentile(values, 1.0)) for values in finite_samples)
    high = max(float(np.percentile(values, 99.0)) for values in finite_samples)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("radial_geometry_samples_invalid")
    if high <= low + 1e-6:
        return float(np.median(np.concatenate(finite_samples)))
    grid = np.linspace(low, high, 512, dtype=np.float64)
    joint_log_density = np.zeros_like(grid)
    for values in finite_samples:
        if len(values) == 1:
            bandwidth = max(1e-3, 0.01 * max(1.0, abs(float(values[0]))))
        else:
            standard_deviation = float(np.std(values))
            interquartile_range = float(
                np.percentile(values, 75.0) - np.percentile(values, 25.0)
            )
            robust_scale = (
                min(standard_deviation, interquartile_range / 1.34)
                if interquartile_range > 0.0
                else standard_deviation
            )
            bandwidth = 0.9 * robust_scale * (len(values) ** -0.2)
            bandwidth = max(
                bandwidth,
                1e-3 * max(1.0, abs(float(np.median(values)))),
            )
        offsets = (grid[:, None] - values[None, :]) / bandwidth
        density = np.exp(-0.5 * offsets * offsets).mean(axis=1) / bandwidth
        density /= max(float(density.max()), 1e-12)
        joint_log_density += np.log(np.maximum(density, 1e-12))
    return float(grid[int(np.argmax(joint_log_density))])


def _visual_hull_from_optical_center(
    samples: list[dict],
) -> dict[str, np.ndarray | float]:
    """Fit a mask visual hull after resolving its shared radial depth."""
    ranges_by_view: list[np.ndarray] = []
    usable: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for item in samples:
        points = np.asarray(item["points"], dtype=np.float64)
        camera = np.asarray(item["camera_position"], dtype=np.float64)
        offsets = points - camera[None, :]
        ranges = np.linalg.norm(offsets, axis=1)
        valid = np.isfinite(offsets).all(axis=1) & np.isfinite(ranges) & (ranges > 1e-6)
        if not valid.any():
            continue
        ranges_by_view.append(ranges[valid])
        usable.append((camera, offsets[valid], ranges[valid]))
    common_range = _common_radial_mode(ranges_by_view)
    projected = []
    for camera, offsets, ranges in usable:
        projected.append(
            camera[None, :] + offsets / ranges[:, None] * common_range
        )
    points = np.concatenate(projected).astype(np.float32)
    low = np.percentile(points, 2.5, axis=0)
    high = np.percentile(points, 97.5, axis=0)
    center = 0.5 * (low + high)
    extent = np.maximum(high - low, 0.02)
    # A single optical centre constrains silhouette width and height but not
    # the hidden radial thickness.  Use the horizontal silhouette diameter as
    # the neutral visual-hull thickness; independent stations can refine it.
    horizontal_diameter = max(float(extent[0]), float(extent[1]), 0.02)
    extent[:2] = horizontal_diameter
    return {
        "center": center.astype(np.float64),
        "extent": extent.astype(np.float64),
        "points": points,
        "common_range": common_range,
    }


def _apply_near_anchor_ranges(lifted: list[dict]) -> None:
    """Recover an unmeasured small target's range from its bound near anchor.

    A small mask can fall between LiDAR beams.  In that case dense completion
    still supplies the target ray and silhouette, but its radial mode can land
    on the background.  A visually bound ``near`` anchor with direct masked
    LiDAR is metric evidence for the local depth layer.  Reproject only the
    target's existing visual hull to that measured range; never replace its
    semantic identity, ray, or silhouette with the anchor.
    """
    by_binding_key = {
        str(key): observation
        for observation in lifted
        for key in observation.get("source_proposal_binding_keys", ())
    }
    for observation in lifted:
        if (
            observation.get("relation_roi_predicate") != "near"
            or int(observation.get("lidar_mask_support_point_count", 0)) > 0
        ):
            continue
        anchors = [
            by_binding_key.get(str(key))
            for key in observation.get("relation_roi_anchor_binding_keys", ())
        ]
        anchors = [
            anchor for anchor in anchors
            if isinstance(anchor, dict)
            and int(anchor.get("lidar_foreground_point_count", 0)) > 0
        ]
        if not anchors:
            continue
        view_id = str(observation.get("relation_roi_view_id", ""))
        view = next((
            item for item in observation.get("per_view_geometry", ())
            if str(item.get("view_id", "")) == view_id
        ), None)
        if not isinstance(view, dict):
            continue
        intrinsics = np.asarray(view.get("intrinsics", ()), dtype=np.float64)
        cam2w = np.asarray(view.get("cam2w_map", ()), dtype=np.float64)
        bbox = np.asarray(view.get("bbox_xyxy", ()), dtype=np.float64)
        if intrinsics.shape != (3, 3) or cam2w.shape != (4, 4) or bbox.shape != (4,):
            continue
        camera = cam2w[:3, 3]
        anchor = min(
            anchors,
            key=lambda item: np.linalg.norm(
                np.asarray(item["center_3d"], dtype=np.float64) - camera
            ),
        )
        anchor_center = np.asarray(anchor["center_3d"], dtype=np.float64)
        anchor_range = float(np.linalg.norm(anchor_center - camera))
        old_center = np.asarray(observation["center_3d"], dtype=np.float64)
        old_extent = np.asarray(observation["bbox_3d"], dtype=np.float64)
        old_range = float(np.linalg.norm(old_center - camera))
        if not math.isfinite(anchor_range) or not math.isfinite(old_range) or min(anchor_range, old_range) <= 1e-6:
            continue
        pixel = np.asarray([
            0.5 * (bbox[0] + bbox[2]),
            0.5 * (bbox[1] + bbox[3]),
            1.0,
        ], dtype=np.float64)
        ray_camera = np.linalg.solve(intrinsics, pixel)
        ray_map = cam2w[:3, :3] @ ray_camera
        ray_norm = float(np.linalg.norm(ray_map))
        if not math.isfinite(ray_norm) or ray_norm <= 1e-9:
            continue
        scale = anchor_range / old_range
        new_center = camera + (ray_map / ray_norm) * anchor_range
        new_extent = np.maximum(old_extent * scale, 0.02)
        cloud_path = Path(str(observation.get("pointcloud_path", "")))
        if cloud_path.is_file():
            with np.load(cloud_path) as cloud:
                payload = {key: cloud[key] for key in cloud.files}
            for key in ("points", "world_points"):
                if key in payload:
                    points = np.asarray(payload[key], dtype=np.float32)
                    payload[key] = (
                        camera[None, :] + (points - camera[None, :]) * scale
                    ).astype(np.float32)
            np.savez_compressed(cloud_path, **payload)
        observation["geometry_source_before_relation_anchor"] = observation.get(
            "geometry_source"
        )
        observation["geometry_source"] = (
            "near_anchor_lidar_range_projected_visual_hull"
        )
        observation["relation_anchor_observation_id"] = anchor.get(
            "observation_id"
        )
        observation["relation_anchor_range_m"] = anchor_range
        observation["visual_range_before_anchor_m"] = old_range
        observation["center_3d"] = new_center.astype(float).tolist()
        observation["centroid_xyz"] = new_center.astype(float).tolist()
        observation["measured_center_3d"] = new_center.astype(float).tolist()
        observation["bbox_3d"] = new_extent.astype(float).tolist()
        observation["measured_bbox_3d"] = new_extent.astype(float).tolist()
        observation["lidar_validated"] = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    output = Path(request["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    response_path = output / "worker_response.json"
    response = {"status": "failed", "request_path": str(args.request)}
    try:
        observations = json.loads(
            Path(request["verified_observations_path"]).read_text(encoding="utf-8")
        )
        geometry = json.loads(
            Path(request["geometry_manifest_path"]).read_text(encoding="utf-8")
        )
        pointmaps = {item["view_id"]: {**item, "optical_center_group": geometry.get("optical_center_group", "unknown")} for item in geometry["views"]}
        calibration = json.loads(Path(request["calibration_path"]).read_text(encoding="utf-8"))
        state = json.loads(Path(request["state_estimation_path"]).read_text(encoding="utf-8"))
        if not calibration.get("configured") or state is None:
            raise ValueError("competition_geometry_transform_unavailable")
        # Only validate child_frame_id if it's present (it's optional)
        if state.get("frame_id") != "map":
            raise ValueError("state_estimation_map_to_sensor_contract_invalid")
        if "child_frame_id" in state and state["child_frame_id"] != "sensor":
            raise ValueError("state_estimation_map_to_sensor_contract_invalid")
        sensor_to_camera = np.asarray(calibration["source_to_camera"], dtype=np.float64)
        adapter_to_camera = np.asarray(calibration["adapter_to_camera_canonical"], dtype=np.float64)
        map_from_sensor = _map_from_sensor(state)
        sensor_points = np.load(request["sensor_scan_path"]).astype(np.float32)
        registered_points = np.load(request["registered_scan_path"]).astype(np.float32)
        threshold = float(request["confidence_threshold"])
        maximum = int(request["max_saved_points_per_observation"])
        clouds_dir = output / "object_pointclouds"
        clouds_dir.mkdir(exist_ok=True)
        lifted = []
        failures = []
        for observation in observations:
            observation_sensor_points = np.load(
                observation.get("source_sensor_scan_path", request["sensor_scan_path"])
            ).astype(np.float32)
            observation_registered_points = np.load(
                observation.get("source_registered_scan_path", request["registered_scan_path"])
            ).astype(np.float32)
            observation_state = json.loads(Path(
                observation.get("source_state_estimation_path", request["state_estimation_path"])
            ).read_text(encoding="utf-8"))
            observation_map_from_sensor = _map_from_sensor(observation_state)
            panorama_mask = cv2.imread(
                observation["panorama_mask_path"], cv2.IMREAD_GRAYSCALE
            )
            if panorama_mask is None:
                raise ValueError(
                    f"panorama_mask_unreadable:{observation['observation_id']}"
                )
            sensor_mask_support = _lidar_mask_support(
                observation_sensor_points,
                panorama_mask > 0,
                sensor_to_camera,
                adapter_to_camera,
                float(request["panorama_vertical_fov_deg"]),
                dilate=False,
            )
            supported_sensor_points = observation_sensor_points[
                sensor_mask_support
            ]
            supported_sensor_world_points = _transform(
                supported_sensor_points, observation_map_from_sensor
            ).astype(np.float32)
            members = list(observation.get("members", ()))
            # If no members, use representative view
            if not members:
                representative_id = str(observation.get("representative_view_id", ""))
                representative_mask = str(observation.get("representative_mask_path", ""))
                if representative_id and representative_mask:
                    members = [{"view_id": representative_id, "mask_path": representative_mask}]
            per_view_budget = max(256, maximum // max(1, len(members)))
            view_geometry = []
            geometry_samples = []
            for member in members:
                view_id = str(member["view_id"])
                pointmap_record = pointmaps.get(view_id)
                if pointmap_record is None:
                    continue
                mask = cv2.imread(str(member["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                with np.load(pointmap_record["pointmap_path"]) as data:
                    candidates = []
                    primary_source = str(pointmap_record.get(
                        "dense_geometry_source", "dense_pointmap"
                    ))
                    # Support both points_map and points_panorama for backward compatibility
                    points_key = "points_map" if "points_map" in data.files else "points_panorama"
                    primary = _mask_geometry_candidate(
                        source=primary_source,
                        points=data[points_key],
                        confidence=data["confidence"],
                        mask=mask,
                        confidence_threshold=threshold,
                        sample_budget=per_view_budget,
                        registered_points=observation_registered_points,
                    )
                    if primary is not None:
                        candidates.append(primary)
                    if (
                        "mast3r_points_map" in data.files
                        and primary_source != "mast3r_sga_dense_pointmap"
                    ):
                        mast3r = _mask_geometry_candidate(
                            source="mast3r_sga_dense_pointmap",
                            points=data["mast3r_points_map"],
                            confidence=data["mast3r_confidence"],
                            mask=mask,
                            confidence_threshold=threshold,
                            sample_budget=per_view_budget,
                            registered_points=observation_registered_points,
                        )
                        if mast3r is not None:
                            candidates.append(mast3r)
                if not candidates:
                    continue
                # Preserve every positive-confidence mask point and choose the
                # dense source by continuous model confidence and registered
                # map consistency. No class, relation, or acceptance threshold
                # participates in the choice.
                chosen = max(
                    candidates,
                    key=lambda item: (
                        item["source_quality"],
                        item["point_count"],
                    ),
                )
                registered_median = chosen[
                    "registered_scan_median_distance_m"
                ]
                registered_fraction = chosen[
                    "registered_scan_inlier_fraction_0_25m"
                ]
                confidence_median = chosen["median_confidence"]
                geometry_weight = (
                    max(1e-9, float(chosen["source_quality"]))
                    * max(1e-3, float(member.get("sam_score", 0.0)))
                    * max(1e-3, float(member.get("qwen_target_probability") or observation.get("qwen_target_probability") or 0.5))
                )
                if chosen["source"] == "mast3r_sga_dense_pointmap":
                    chosen_cam2w = pointmap_record.get(
                        "sga_cam2w_map", pointmap_record.get("cam2w_map", np.eye(4).tolist())
                    )
                elif chosen["source"] == "any2full_metric_depth_ros_camera_pose":
                    chosen_cam2w = pointmap_record.get(
                        "depth_cam2w_map", pointmap_record.get("cam2w_map", np.eye(4).tolist())
                    )
                else:
                    chosen_cam2w = pointmap_record.get("cam2w_map", np.eye(4).tolist())
                view_geometry.append({
                    "view_id": view_id,
                    "optical_center_group": pointmap_record["optical_center_group"],
                    "intrinsics": pointmap_record.get("intrinsics", []),
                    "point_count": chosen["point_count"],
                    "sampled_point_count": chosen["sampled_point_count"],
                    "center_3d": chosen["center_3d"].astype(float).tolist(),
                    "bbox_3d": chosen["bbox_3d"].astype(float).tolist(),
                    "median_confidence": confidence_median,
                    "registered_scan_median_distance_m": registered_median,
                    "registered_scan_inlier_fraction_0_25m": registered_fraction,
                    "fusion_weight": geometry_weight,
                    "cam2w_map": chosen_cam2w,
                    "bbox_xyxy": list(member.get("bbox_xyxy", ())),
                    "geometry_source": chosen["source"],
                    "geometry_source_candidates": [{
                        "source": item["source"],
                        "point_count": item["point_count"],
                        "source_quality": item["source_quality"],
                        "registered_scan_median_distance_m": item[
                            "registered_scan_median_distance_m"
                        ],
                        "registered_scan_inlier_fraction_0_25m": item[
                            "registered_scan_inlier_fraction_0_25m"
                        ],
                    } for item in candidates],
                })
                geometry_samples.append({
                    "optical_center_group": pointmap_record[
                        "optical_center_group"
                    ],
                    "points": chosen["sample"],
                    "confidence": chosen["sample_confidence"],
                    "camera_position": np.asarray(
                        chosen_cam2w, dtype=np.float64
                    )[:3, 3],
                    "fusion_weight": geometry_weight,
                })
            if not view_geometry:
                failures.append({
                    "observation_id": observation["observation_id"],
                    "reason": (
                        "no_dense_geometry_source_has_confident_mask_points"
                    ),
                })
                continue
            samples_by_group: dict[str, list[dict]] = {}
            for item in geometry_samples:
                samples_by_group.setdefault(
                    str(item["optical_center_group"]), []
                ).append(item)
            group_geometry = []
            group_points = []
            group_confidences = []
            for group, samples in samples_by_group.items():
                hull = _visual_hull_from_optical_center(samples)
                hull_points = np.asarray(hull["points"], dtype=np.float32)
                confidence_values = np.concatenate([
                    np.asarray(item["confidence"], dtype=np.float32)
                    for item in samples
                ])
                group_weight = max(
                    float(item["fusion_weight"]) for item in samples
                )
                group_geometry.append({
                    "optical_center_group": group,
                    "center_3d": np.asarray(hull["center"], dtype=np.float64),
                    "bbox_3d": np.asarray(hull["extent"], dtype=np.float64),
                    "common_radial_range_m": float(hull["common_range"]),
                    "fusion_weight": group_weight,
                })
                group_points.append(hull_points)
                group_confidences.append(np.full(
                    len(hull_points),
                    float(np.median(confidence_values)),
                    dtype=np.float32,
                ))
            centers = np.asarray([
                item["center_3d"] for item in group_geometry
            ], dtype=np.float64)
            extents = np.asarray([
                item["bbox_3d"] for item in group_geometry
            ], dtype=np.float64)
            weights = np.asarray([
                item["fusion_weight"] for item in group_geometry
            ], dtype=np.float64)

            lower_bounds = centers - 0.5 * extents
            upper_bounds = centers + 0.5 * extents
            world_lower = _weighted_coordinate_median(lower_bounds, weights)
            world_upper = _weighted_coordinate_median(upper_bounds, weights)
            world_center = 0.5 * (world_lower + world_upper)
            world_extent = np.maximum(world_upper - world_lower, 0.02)
            world_points = np.concatenate(group_points).astype(np.float32)
            world_confidence = np.concatenate(group_confidences).astype(
                np.float32
            )
            geometry_source = (
                "mask_local_multiview_radial_consensus_visual_hull"
            )

            # The timestamp-aligned sensor scan is metric geometry in the ROS
            # map frame.  Once SAM's panorama mask has selected its returns,
            # those measured points are the final object geometry; dense visual
            # depth remains useful only when the mask contains no LiDAR return.
            # Do not reduce LiDAR to a consistency score for a monocular box.
            lidar_foreground_points = np.zeros((0, 3), dtype=np.float32)
            if len(supported_sensor_world_points):
                # A 2D mask can also contain returns from surfaces visible
                # through holes or just behind an object boundary.  Seed the
                # foreground at the LiDAR return nearest the dense visual
                # hypothesis, then retain one visual-box diagonal of metric
                # support around that seed.  Vision selects the physical LiDAR
                # component; it does not contribute the published geometry.
                visual_diagonal = float(np.linalg.norm(world_extent))
                lidar_distances = np.linalg.norm(
                    supported_sensor_world_points
                    - world_center[None, :],
                    axis=1,
                )
                foreground_radius = (
                    float(np.min(lidar_distances)) + visual_diagonal
                )
                lidar_foreground_points = supported_sensor_world_points[
                    lidar_distances <= foreground_radius
                ]
                lidar_lower = np.min(
                    lidar_foreground_points, axis=0
                ).astype(np.float64)
                lidar_upper = np.max(
                    lidar_foreground_points, axis=0
                ).astype(np.float64)
                world_center = 0.5 * (lidar_lower + lidar_upper)
                world_extent = np.maximum(
                    lidar_upper - lidar_lower, 0.02
                )
                measured_center = world_center.copy()
                measured_extent = world_extent.copy()
                world_points = lidar_foreground_points.astype(
                    np.float32
                )
                world_confidence = np.ones(
                    len(world_points), dtype=np.float32
                )
                geometry_source = "sam_mask_projected_lidar_aabb"

            # Fit a top-surface plane for anchor-class objects (e.g. cabinet tops).
            # Ray-surface verification needs this to check whether a target bbox
            # centre projects onto the supporting surface.
            surface_plane = None
            if len(world_points) >= 30:
                z_vals = world_points[:, 2]
                z_top = float(np.percentile(z_vals, 75))
                top_pts = world_points[z_vals >= z_top]
                if len(top_pts) >= 15:
                    centroid = top_pts.mean(axis=0)
                    _u, _s, vh = np.linalg.svd(top_pts - centroid)
                    normal = vh[2].astype(np.float64)
                    if normal[2] < 0:
                        normal = -normal
                    # Only trust near-horizontal surfaces (within 30° of up)
                    if normal[2] >= 0.866:
                        horizontal_pts = top_pts - centroid
                        lateral = horizontal_pts - np.outer(
                            np.dot(horizontal_pts, normal), normal
                        )
                        extents_xy = np.abs(lateral).max(axis=0).astype(float)
                        surface_plane = {
                            "center": centroid.astype(float).tolist(),
                            "normal": normal.astype(float).tolist(),
                            "extents_xy_m": [
                                max(0.08, float(extents_xy[0])),
                                max(0.08, float(extents_xy[1])),
                            ],
                            "point_count": int(len(top_pts)),
                        }

            if len(world_points) > maximum:
                indices = np.linspace(0, len(world_points) - 1, maximum, dtype=np.int64)
                saved_points = world_points[indices]
                saved_confidence = world_confidence[indices]
            else:
                saved_points = world_points
                saved_confidence = world_confidence
            cloud_path = clouds_dir / f"{observation['observation_id']}.npz"
            # SGA has already applied the single reconstruction-level Sim(3).
            # Do not apply this acquisition's sensor pose a second time.
            registered_views = [
                (item, float(item["fusion_weight"]))
                for item in view_geometry
                if item["registered_scan_median_distance_m"] is not None
            ]
            if registered_views:
                registered_values = np.asarray([
                    [item["registered_scan_median_distance_m"]]
                    for item, _weight in registered_views
                ], dtype=np.float64)
                registered_weights = np.asarray([
                    weight for _item, weight in registered_views
                ], dtype=np.float64)
                registered_median_distance = float(_weighted_coordinate_median(
                    registered_values, registered_weights
                )[0])
                registered_inlier_values = np.asarray([
                    item["registered_scan_inlier_fraction_0_25m"]
                    for item, _weight in registered_views
                ], dtype=np.float64)
                if float(registered_weights.sum()) > 0.0:
                    registered_inlier_fraction = float(np.average(
                        registered_inlier_values,
                        weights=registered_weights,
                    ))
                else:
                    # A valid view can have zero retained point weight.  Its
                    # unweighted diagnostic mean is still defined; numpy's
                    # weighted average is not.
                    registered_inlier_fraction = float(
                        registered_inlier_values.mean()
                    )
            else:
                registered_median_distance = None
                registered_inlier_fraction = 0.0
            np.savez_compressed(
                cloud_path,
                points=saved_points,
                confidence=saved_confidence,
                world_points=saved_points,
                lidar_support_points_sensor=supported_sensor_points,
                lidar_support_points_map=supported_sensor_world_points,
            )
            lifted.append({
                "schema_version": "1.0",
                "observation_id": observation["observation_id"],
                "canonical_class": observation["canonical_class"],
                "semantic_probability": observation.get("qwen_target_probability"),
                "panorama_mask_path": observation.get("panorama_mask_path"),
                "representative_view_id": observation.get(
                    "representative_view_id"
                ),
                "representative_view_image": observation.get(
                    "representative_view_image"
                ),
                "representative_bbox_xyxy": observation.get(
                    "representative_bbox_xyxy"
                ),
                **{
                    key: observation[key]
                    for key in (
                        "relation_roi_image_path",
                        "relation_roi_view_id",
                        "relation_roi_subject_bbox_xyxy_normalized",
                        "relation_roi_anchor_bbox_xyxy_normalized",
                        "relation_roi_anchor_class",
                        "relation_roi_anchor_classes",
                        "relation_roi_anchor_bboxes_xyxy_normalized",
                        "relation_roi_anchor_binding_keys",
                        "relation_roi_predicate",
                        "relation_binding_context_bbox_xyxy_normalized",
                    )
                    if key in observation
                },
                "source_view_ids": [item["view_id"] for item in view_geometry],
                "source_proposal_binding_keys": list(
                    observation.get("source_proposal_binding_keys", ())
                ),
                "proposal_verification_by_key": dict(
                    observation.get("proposal_verification_by_key", {})
                ),
                "qwen_verification_probabilities": list(
                    observation.get("qwen_verification_probabilities", ())
                ),
                "source_view_count": len(view_geometry),
                "per_view_geometry": view_geometry,
                "per_optical_center_geometry": [{
                    **item,
                    "center_3d": item["center_3d"].astype(float).tolist(),
                    "bbox_3d": item["bbox_3d"].astype(float).tolist(),
                } for item in group_geometry],
                "frame": geometry["frame"],
                "source_geometry_frame": geometry["frame"],
                "optical_center_group": view_geometry[0]["optical_center_group"] if view_geometry else "unknown",
                "optical_center_groups": list(dict.fromkeys(
                    str(item["optical_center_group"]) for item in view_geometry
                )),
                "independent_optical_center_count": len(set(
                    str(item["optical_center_group"]) for item in view_geometry
                )),
                "world_aligned": True,
                "map_transform_source": "mast3r_sga_post_alignment_sim3",
                "geometry_source": geometry_source,
                "reconstruction_id": geometry.get("reconstruction_id", "unknown"),
                "source_cam2w_maps": [item["cam2w_map"] for item in view_geometry],
                "source_intrinsics": [item["intrinsics"] for item in view_geometry],
                "point_count": int(sum(item["point_count"] for item in view_geometry)),
                "saved_point_count": int(len(saved_points)),
                "centroid_xyz": world_center.astype(float).tolist(),
                "center_3d": world_center.astype(float).tolist(),
                "bbox_3d": world_extent.astype(float).tolist(),
                "measured_center_3d": (
                    measured_center.astype(float).tolist()
                    if len(lidar_foreground_points)
                    else world_center.astype(float).tolist()
                ),
                "measured_bbox_3d": (
                    measured_extent.astype(float).tolist()
                    if len(lidar_foreground_points)
                    else world_extent.astype(float).tolist()
                ),
                "viewpoint_position_map": np.median(np.asarray([
                    np.asarray(item["cam2w_map"], dtype=np.float64)[:3, 3]
                    for item in view_geometry
                ]), axis=0).astype(float).tolist(),
                "median_confidence": float(np.median([
                    item["median_confidence"] for item in view_geometry
                ])),
                "view_center_dispersion_m": float(np.median(
                    np.linalg.norm(centers - world_center[None, :], axis=1)
                )),
                "lidar_mask_support_point_count": int(len(supported_sensor_points)),
                "lidar_foreground_point_count": int(
                    len(lidar_foreground_points)
                ),
                "registered_scan_median_distance_m": registered_median_distance,
                "registered_scan_inlier_fraction_0_25m": registered_inlier_fraction,
                "lidar_validated": bool(
                    len(supported_sensor_points) > 0 or registered_inlier_fraction > 0.0
                ),
                "persistent_instance_authorized": False,
                "surface_plane": surface_plane,
                "pointcloud_path": str(cloud_path),
            })
        _apply_near_anchor_ranges(lifted)
        lifted_path = output / "local_3d_observations.json"
        _write_json(lifted_path, lifted)
        response = {
            "status": "completed" if lifted else "blocked",
            "completion_scope": "partial" if lifted and failures else "complete",
            "input_verified_observation_count": len(observations),
            "lifted_observation_count": len(lifted),
            "failed_observation_count": len(failures),
            "failures": failures,
            "local_3d_observations_path": str(lifted_path),
            "frame": geometry["frame"],
            "world_aligned": bool(lifted),
            "sensor_scan_point_count": int(len(sensor_points)),
            "registered_scan_point_count": int(len(registered_points)),
            "state_estimation_frame": state["frame_id"],
            "calibration_path": request["calibration_path"],
            "reconstruction_id": geometry.get("reconstruction_id", "unknown"),
            "reconstruction_mode": geometry.get("reconstruction_mode", "unknown"),
        }
        _write_json(response_path, response)
        return 0 if response["status"] == "completed" else 1
    except Exception as exc:
        response.update({
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        _write_json(response_path, response)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
