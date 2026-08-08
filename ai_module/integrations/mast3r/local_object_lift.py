#!/usr/bin/env python3
"""Lift verified masks through SGA pointmaps already aligned to ROS map."""

from __future__ import annotations

import argparse
import json
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
    dilated = cv2.dilate(panorama_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    supported = np.zeros(len(sensor_points), dtype=bool)
    supported[valid] = dilated[rows[valid], columns[valid]]
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


def _weighted_coordinate_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = []
    normalized = np.maximum(np.asarray(weights, dtype=np.float64), 1e-9)
    for axis in range(values.shape[1]):
        order = np.argsort(values[:, axis])
        cumulative = np.cumsum(normalized[order])
        index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
        result.append(float(values[order[min(index, len(order) - 1)], axis]))
    return np.asarray(result, dtype=np.float64)


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
        pointmaps = {item["view_id"]: item for item in geometry["views"]}
        calibration = json.loads(Path(request["calibration_path"]).read_text(encoding="utf-8"))
        state = json.loads(Path(request["state_estimation_path"]).read_text(encoding="utf-8"))
        if not calibration.get("configured") or state is None:
            raise ValueError("competition_geometry_transform_unavailable")
        if state.get("frame_id") != "map" or state.get("child_frame_id") != "sensor":
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
            members = list(observation.get("members", ()))
            per_view_budget = max(256, maximum // max(1, len(members)))
            view_geometry = []
            sampled_points = []
            sampled_confidences = []
            for member in members:
                view_id = str(member["view_id"])
                pointmap_record = pointmaps.get(view_id)
                if pointmap_record is None:
                    continue
                mask = cv2.imread(str(member["mask_path"]), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                with np.load(pointmap_record["pointmap_path"]) as data:
                    points = data["points_map"]
                    confidence = data["confidence"]
                if mask.shape != points.shape[:2] or confidence.shape != mask.shape:
                    raise ValueError(
                        f"mask_pointmap_shape_mismatch:{observation['observation_id']}:{view_id}"
                    )
                valid = (
                    (mask > 0)
                    & np.isfinite(points).all(axis=-1)
                    & np.isfinite(confidence)
                    & (confidence >= threshold)
                )
                selected = points[valid].astype(np.float32)
                selected_confidence = confidence[valid].astype(np.float32)
                if not len(selected):
                    continue
                if len(selected) > per_view_budget:
                    indices = np.linspace(
                        0, len(selected) - 1, per_view_budget, dtype=np.int64
                    )
                    sample = selected[indices]
                    sample_confidence = selected_confidence[indices]
                else:
                    sample = selected
                    sample_confidence = selected_confidence
                distances = _nearest_registered_distances(
                    sample, observation_registered_points
                )
                registered_median = float(np.median(distances)) if len(distances) else None
                registered_fraction = float(np.mean(distances <= 0.25)) if len(distances) else 0.0
                center = np.median(selected, axis=0).astype(np.float64)
                low = np.percentile(selected, 2.5, axis=0)
                high = np.percentile(selected, 97.5, axis=0)
                extent = np.maximum(high - low, 0.02).astype(np.float64)
                confidence_median = float(np.median(selected_confidence))
                geometry_weight = (
                    confidence_median
                    * max(1e-3, float(member.get("sam_score", 0.0)))
                    * max(1e-3, float(member.get("qwen_target_probability") or observation.get("qwen_target_probability") or 0.5))
                    * (1.0 if registered_median is None else float(np.exp(-registered_median / 0.25)))
                )
                view_geometry.append({
                    "view_id": view_id,
                    "optical_center_group": pointmap_record["optical_center_group"],
                    "point_count": int(len(selected)),
                    "sampled_point_count": int(len(sample)),
                    "center_3d": center.astype(float).tolist(),
                    "bbox_3d": extent.astype(float).tolist(),
                    "median_confidence": confidence_median,
                    "registered_scan_median_distance_m": registered_median,
                    "registered_scan_inlier_fraction_0_25m": registered_fraction,
                    "fusion_weight": geometry_weight,
                    "cam2w_map": pointmap_record["cam2w_map"],
                })
                sampled_points.append(sample)
                sampled_confidences.append(sample_confidence)
            if not view_geometry:
                failures.append({
                    "observation_id": observation["observation_id"],
                    "reason": "no_member_view_has_confident_mast3r_points",
                })
                continue
            centers = np.asarray([item["center_3d"] for item in view_geometry], dtype=np.float64)
            extents = np.asarray([item["bbox_3d"] for item in view_geometry], dtype=np.float64)
            weights = np.asarray([item["fusion_weight"] for item in view_geometry], dtype=np.float64)
            world_center = _weighted_coordinate_median(centers, weights)
            world_extent = _weighted_coordinate_median(extents, weights)
            world_points = np.concatenate(sampled_points).astype(np.float32)
            world_confidence = np.concatenate(sampled_confidences).astype(np.float32)

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
            panorama_mask = cv2.imread(observation["panorama_mask_path"], cv2.IMREAD_GRAYSCALE)
            if panorama_mask is None:
                raise ValueError(f"panorama_mask_unreadable:{observation['observation_id']}")
            lidar_support = _lidar_mask_support(
                observation_sensor_points,
                panorama_mask > 0,
                sensor_to_camera,
                adapter_to_camera,
                float(request["panorama_vertical_fov_deg"]),
            )
            supported_sensor_points = observation_sensor_points[lidar_support]
            supported_world_points = _transform(
                supported_sensor_points, observation_map_from_sensor
            ).astype(np.float32)
            registered_views = [
                (item, weight) for item, weight in zip(view_geometry, weights)
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
                registered_inlier_fraction = float(np.average(
                    [item["registered_scan_inlier_fraction_0_25m"] for item, _weight in registered_views],
                    weights=registered_weights,
                ))
            else:
                registered_median_distance = None
                registered_inlier_fraction = 0.0
            np.savez_compressed(
                cloud_path,
                points=saved_points,
                confidence=saved_confidence,
                world_points=saved_points,
                lidar_support_points_sensor=supported_sensor_points,
                lidar_support_points_map=supported_world_points,
            )
            lifted.append({
                "schema_version": "1.0",
                "observation_id": observation["observation_id"],
                "canonical_class": observation["canonical_class"],
                "semantic_probability": observation.get("qwen_target_probability"),
                "panorama_mask_path": observation.get("panorama_mask_path"),
                "source_view_ids": [item["view_id"] for item in view_geometry],
                "source_view_count": len(view_geometry),
                "per_view_geometry": view_geometry,
                "frame": "map",
                "source_geometry_frame": geometry["frame"],
                "optical_center_group": observation["optical_center_group"],
                "optical_center_groups": list(dict.fromkeys(
                    str(item["optical_center_group"]) for item in view_geometry
                )),
                "independent_optical_center_count": len(set(
                    str(item["optical_center_group"]) for item in view_geometry
                )),
                "world_aligned": True,
                "map_transform_source": "mast3r_sga_post_alignment_sim3",
                "reconstruction_id": geometry["reconstruction_id"],
                "source_cam2w_maps": [item["cam2w_map"] for item in view_geometry],
                "point_count": int(sum(item["point_count"] for item in view_geometry)),
                "saved_point_count": int(len(saved_points)),
                "centroid_xyz": world_center.astype(float).tolist(),
                "center_3d": world_center.astype(float).tolist(),
                "bbox_3d": world_extent.astype(float).tolist(),
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
                "registered_scan_median_distance_m": registered_median_distance,
                "registered_scan_inlier_fraction_0_25m": registered_inlier_fraction,
                "lidar_validated": bool(
                    len(supported_sensor_points) > 0 or registered_inlier_fraction > 0.0
                ),
                "surface_plane": surface_plane,
                "pointcloud_path": str(cloud_path),
            })
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
            "reconstruction_id": geometry["reconstruction_id"],
            "reconstruction_mode": geometry["reconstruction_mode"],
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
