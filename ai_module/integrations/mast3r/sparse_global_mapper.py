#!/usr/bin/env python3
"""Cross-station MASt3R sparse global reconstruction aligned to ROS map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import torch

import mast3r.utils.path_to_dust3r  # noqa: F401
from dust3r.utils.image import load_images
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.model import AsymmetricMASt3R


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rotation_from_quaternion(xyzw: list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(xyzw, dtype=np.float64)
    x, y, z, w = np.asarray([x, y, z, w]) / np.linalg.norm([x, y, z, w])
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _map_from_sensor(state: dict[str, Any]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rotation_from_quaternion(state["orientation_xyzw"])
    result[:3, 3] = np.asarray(state["position_xyz"], dtype=np.float64)
    return result


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _singular, vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _align_direction(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine > 1e-9:
        return _axis_angle_rotation(cross / sine, np.arctan2(sine, cosine))
    if cosine > 0.0:
        return np.eye(3)
    helper = np.asarray([1.0, 0.0, 0.0])
    if abs(source[0]) > 0.8:
        helper = np.asarray([0.0, 1.0, 0.0])
    return _axis_angle_rotation(np.cross(source, helper), np.pi)


def _build_pair_graph(images: list[dict], views: list[dict]) -> tuple[list[tuple[dict, dict]], list[dict]]:
    by_group: dict[str, list[int]] = {}
    for index, view in enumerate(views):
        by_group.setdefault(str(view["optical_center_group"]), []).append(index)
    ordered_groups = list(by_group)
    edges: set[tuple[int, int]] = set()

    # Circular overlap inside each panorama preserves its eight-slice optical
    # centre, while temporal same-heading edges carry translation between robot
    # stations into one connected reconstruction.
    for indices in by_group.values():
        indices.sort(key=lambda value: float(views[value]["yaw_deg"]))
        for position, first in enumerate(indices):
            second = indices[(position + 1) % len(indices)]
            edges.add(tuple(sorted((first, second))))
    for first_group, second_group in zip(ordered_groups, ordered_groups[1:]):
        first_by_yaw = {round(float(views[i]["yaw_deg"]), 3): i for i in by_group[first_group]}
        second_by_yaw = {round(float(views[i]["yaw_deg"]), 3): i for i in by_group[second_group]}
        for yaw in sorted(set(first_by_yaw) & set(second_by_yaw)):
            edges.add(tuple(sorted((first_by_yaw[yaw], second_by_yaw[yaw]))))

    directed = []
    records = []
    for first, second in sorted(edges):
        records.append({
            "view1": views[first]["view_id"],
            "view2": views[second]["view_id"],
            "cross_station": (
                views[first]["optical_center_group"]
                != views[second]["optical_center_group"]
            ),
        })
        directed.extend(((images[first], images[second]), (images[second], images[first])))
    return directed, records


def _target_camera_poses(
    views: list[dict],
    state_by_group: dict[str, dict],
    calibration: dict,
) -> np.ndarray:
    source_to_camera = np.asarray(calibration["source_to_camera"], dtype=np.float64)
    adapter_to_camera = np.asarray(
        calibration["adapter_to_camera_canonical"], dtype=np.float64
    )
    adapter_to_sensor = np.linalg.inv(source_to_camera) @ adapter_to_camera
    targets = []
    for view in views:
        map_from_adapter = (
            _map_from_sensor(state_by_group[str(view["optical_center_group"])])
            @ adapter_to_sensor
        )
        adapter_from_view = np.eye(4, dtype=np.float64)
        adapter_from_view[:3, :3] = np.asarray(
            view["R_panorama_from_camera"], dtype=np.float64
        )
        targets.append(map_from_adapter @ adapter_from_view)
    return np.stack(targets)


def _estimate_sim3(
    cam2w_sga: np.ndarray,
    cam2w_map_targets: np.ndarray,
    groups: list[str],
) -> tuple[float, np.ndarray, np.ndarray]:
    source_centres = {}
    target_centres = {}
    for group in dict.fromkeys(groups):
        indices = [index for index, value in enumerate(groups) if value == group]
        source_centres[group] = np.median(cam2w_sga[indices, :3, 3], axis=0)
        target_centres[group] = np.median(cam2w_map_targets[indices, :3, 3], axis=0)
    scale_samples = []
    ordered = list(source_centres)
    baselines = []
    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1:]:
            source_delta = source_centres[second] - source_centres[first]
            target_delta = target_centres[second] - target_centres[first]
            source_distance = np.linalg.norm(source_delta)
            target_distance = np.linalg.norm(target_delta)
            if source_distance > 1e-6 and target_distance > 1e-6:
                scale_samples.append(target_distance / source_distance)
                baselines.append((source_distance, source_delta, target_delta))
    if baselines:
        _distance, source_axis, target_axis = max(baselines, key=lambda item: item[0])
        rotation = _align_direction(source_axis, target_axis)
        target_axis = target_axis / np.linalg.norm(target_axis)
        sine_sum = 0.0
        cosine_sum = 0.0
        for source_pose, target_pose in zip(cam2w_sga, cam2w_map_targets):
            for axis_index in range(3):
                source_vector = rotation @ source_pose[:3, axis_index]
                target_vector = target_pose[:3, axis_index]
                source_plane = source_vector - target_axis * np.dot(target_axis, source_vector)
                target_plane = target_vector - target_axis * np.dot(target_axis, target_vector)
                sine_sum += float(np.dot(target_axis, np.cross(source_plane, target_plane)))
                cosine_sum += float(np.dot(source_plane, target_plane))
        rotation = _axis_angle_rotation(
            target_axis, np.arctan2(sine_sum, cosine_sum)
        ) @ rotation
    else:
        rotation = _project_rotation(sum(
            target[:3, :3] @ source[:3, :3].T
            for source, target in zip(cam2w_sga, cam2w_map_targets)
        ))
    scale = float(np.median(scale_samples)) if scale_samples else 1.0
    translation = np.mean([
        target_centres[group] - scale * (source_centres[group] @ rotation.T)
        for group in ordered
    ], axis=0)
    return scale, rotation, translation


def _consolidate_optical_centres(
    cam2w_native: np.ndarray,
    dense_points_native: list[np.ndarray],
    groups: list[str],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Remove virtual translation between slices from one panorama.

    This happens only after native SGA.  Each dense pointmap is translated with
    its camera, preserving camera-relative geometry while restoring the known
    shared physical optical centre before the reconstruction-level Sim(3).
    """
    cam2w = cam2w_native.copy()
    dense_points = []
    for group in dict.fromkeys(groups):
        indices = [index for index, value in enumerate(groups) if value == group]
        centre = np.median(cam2w_native[indices, :3, 3], axis=0)
        for index in indices:
            delta = centre - cam2w_native[index, :3, 3]
            cam2w[index, :3, 3] = centre
            dense_points.append((index, dense_points_native[index] + delta))
    dense_points.sort(key=lambda item: item[0])
    return cam2w, [item[1] for item in dense_points]


def _apply_sim3(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (scale * (points @ rotation.T) + translation).astype(np.float32)


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_centered ** 2, axis=1)))
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * (source_mean @ rotation.T)
    return scale, rotation, translation


def _lidar_dense_correspondences(
    views: list[dict],
    dense_points: list[np.ndarray],
    confidences: list[np.ndarray],
    keyframes_by_group: dict[str, dict],
    calibration: dict,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    sensor_to_camera = np.asarray(calibration["source_to_camera"], dtype=np.float64)
    adapter_to_camera = np.asarray(
        calibration["adapter_to_camera_canonical"], dtype=np.float64
    )
    camera_to_adapter = np.linalg.inv(adapter_to_camera)
    source_points = []
    target_points = []
    for view, points_flat, confidence in zip(views, dense_points, confidences):
        keyframe = keyframes_by_group[str(view["optical_center_group"])]
        sensor = np.load(keyframe["sensor_scan_path"]).astype(np.float64)
        registered = np.load(keyframe["registered_scan_path"]).astype(np.float64)
        adapter = _transform(_transform(sensor, sensor_to_camera), camera_to_adapter)
        rotation_adapter_from_view = np.asarray(
            view["R_panorama_from_camera"], dtype=np.float64
        )
        camera = adapter @ rotation_adapter_from_view
        depth = camera[:, 2]
        intrinsics = np.asarray(view["K"], dtype=np.float64)
        u = np.rint(intrinsics[0, 0] * camera[:, 0] / np.maximum(depth, 1e-6) + intrinsics[0, 2]).astype(np.int64)
        v = np.rint(intrinsics[1, 1] * camera[:, 1] / np.maximum(depth, 1e-6) + intrinsics[1, 2]).astype(np.int64)
        height, width = confidence.shape
        visible = (depth > 0.05) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indices = np.flatnonzero(visible)
        if not len(indices):
            continue
        points = points_flat.reshape(height, width, 3)[v[indices], u[indices]]
        conf = confidence[v[indices], u[indices]]
        valid = (
            np.isfinite(points).all(axis=1)
            & np.isfinite(conf)
            & (conf >= confidence_threshold)
        )
        source_points.append(points[valid])
        target_points.append(registered[indices][valid])
    source = np.concatenate(source_points).astype(np.float64)
    target = np.concatenate(target_points).astype(np.float64)
    if len(source) > 12000:
        selected = np.linspace(0, len(source) - 1, 12000, dtype=np.int64)
        source = source[selected]
        target = target[selected]
    return source, target


def _refine_sim3_with_lidar(
    source: np.ndarray,
    target: np.ndarray,
    cam2w_sga: np.ndarray,
    cam2w_targets: np.ndarray,
    groups: list[str],
    initial: tuple[float, np.ndarray, np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    scale, rotation, translation = initial
    initial_residual = np.linalg.norm(
        _apply_sim3(source, scale, rotation, translation) - target, axis=1
    )
    retained = initial_residual <= np.percentile(initial_residual, 70.0)
    source = source[retained]
    target = target[retained]

    source_centres = []
    target_centres = []
    for group in dict.fromkeys(groups):
        indices = [index for index, value in enumerate(groups) if value == group]
        source_centres.append(np.median(cam2w_sga[indices, :3, 3], axis=0))
        target_centres.append(np.median(cam2w_targets[indices, :3, 3], axis=0))
    repeats = max(100, len(source) // max(1, 10 * len(source_centres)))
    fit_source = np.concatenate((source, np.repeat(source_centres, repeats, axis=0)))
    fit_target = np.concatenate((target, np.repeat(target_centres, repeats, axis=0)))
    scale, rotation, translation = _umeyama(fit_source, fit_target)
    residual = np.linalg.norm(_apply_sim3(source, scale, rotation, translation) - target, axis=1)
    return scale, rotation, translation, residual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    output = Path(request["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    response_path = output / "worker_response.json"
    response: dict[str, Any] = {"status": "failed", "request_path": str(args.request)}
    try:
        bundles = []
        keyframes_by_group = {}
        for keyframe in request["keyframes"]:
            manifest = json.loads(Path(keyframe["manifest_path"]).read_text(encoding="utf-8"))
            bundles.append(manifest)
            keyframes_by_group[str(keyframe["optical_center_group"])] = keyframe
        views = [view for bundle in bundles for view in bundle["views"]]
        image_paths = [str(view["image_path"]) for view in views]
        images = load_images(image_paths, size=512, verbose=False)
        for index, image in enumerate(images):
            image["idx"] = index
            image["instance"] = str(views[index]["view_id"])
        pairs, pair_records = _build_pair_graph(images, views)

        device = str(request.get("device", "cuda"))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = AsymmetricMASt3R.from_pretrained(str(request["checkpoint"])).to(device).eval()
        model_load_seconds = time.perf_counter() - started
        alignment_started = time.perf_counter()
        scene = sparse_global_alignment(
            image_paths,
            pairs,
            str(output / "sga_cache"),
            model,
            device=device,
            lr1=float(request.get("coarse_learning_rate", 0.07)),
            niter1=int(request["coarse_iterations"]),
            lr2=float(request.get("refine_learning_rate", 0.01)),
            niter2=int(request["refine_iterations"]),
            matching_conf_thr=float(request["matching_confidence_threshold"]),
            shared_intrinsics=False,
        )
        cam2w_sga_native = scene.get_im_poses().detach().cpu().numpy()
        dense_points, _depthmaps, confidences = scene.get_dense_pts3d(clean_depth=False)
        dense_points = [item.detach().cpu().numpy() for item in dense_points]
        confidences = [item.detach().cpu().numpy() for item in confidences]
        alignment_seconds = time.perf_counter() - alignment_started

        state_by_group = {
            group: json.loads(Path(item["state_estimation_path"]).read_text(encoding="utf-8"))
            for group, item in keyframes_by_group.items()
        }
        calibration = json.loads(Path(request["calibration_path"]).read_text(encoding="utf-8"))
        target_poses = _target_camera_poses(views, state_by_group, calibration)
        groups = [str(view["optical_center_group"]) for view in views]
        cam2w_sga, dense_points = _consolidate_optical_centres(
            cam2w_sga_native, dense_points, groups
        )
        initial_sim3 = _estimate_sim3(cam2w_sga, target_poses, groups)
        lidar_source, lidar_target = _lidar_dense_correspondences(
            views,
            dense_points,
            confidences,
            keyframes_by_group,
            calibration,
            float(request["pointmap_confidence_threshold"]),
        )
        scale, rotation, translation, lidar_residual = _refine_sim3_with_lidar(
            lidar_source,
            lidar_target,
            cam2w_sga,
            target_poses,
            groups,
            initial_sim3,
        )

        # ── Optional Any2Full refinement ───────────────────────────────
        any2full_config = request.get("any2full", {})
        any2full_refined = False
        any2full_runtime = 0.0
        any2full_views_refined = 0
        if any2full_config.get("enabled"):
            try:
                from integrations.mast3r.any2full_refiner import refine_view_pointmaps

                any2full_dense, any2full_conf, any2full_runtime, lidar_covered = (
                    refine_view_pointmaps(
                        views,
                        keyframes_by_group,
                        calibration,
                        str(any2full_config["checkpoint"]),
                        encoder=str(any2full_config.get("encoder", "vitb")),
                        depth_scale=float(any2full_config.get("depth_scale", 100.0)),
                        denoise=bool(any2full_config.get("denoise", True)),
                    )
                )
                # Per-view hybrid: use Any2Full where LiDAR covers, keep MASt3R otherwise
                for idx, has_lidar in enumerate(lidar_covered):
                    if has_lidar:
                        dense_points[idx] = any2full_dense[idx]
                        confidences[idx] = any2full_conf[idx]
                        any2full_views_refined += 1
                any2full_refined = any2full_views_refined > 0
                print(
                    f"[any2full] refined {any2full_views_refined}/{len(views)} views "
                    f"in {any2full_runtime:.1f}s"
                )
            except Exception as exc:
                print(f"[any2full] refinement failed, falling back to MASt3R: {exc}")
                # Continue with MASt3R pointmaps unchanged

        pointmaps_dir = output / "pointmaps"
        pointmaps_dir.mkdir(exist_ok=True)
        geometry_views = []
        cam2w_map_all = []
        for index, (view, points, confidence) in enumerate(zip(views, dense_points, confidences)):
            points = points.reshape(*confidence.shape, 3)
            points_map = _apply_sim3(points, scale, rotation, translation)
            cam2w_map = np.eye(4, dtype=np.float64)
            cam2w_map[:3, :3] = rotation @ cam2w_sga[index, :3, :3]
            cam2w_map[:3, 3] = _apply_sim3(
                cam2w_sga[index, None, :3, 3], scale, rotation, translation
            )[0]
            cam2w_map_all.append(cam2w_map)
            pointmap_path = pointmaps_dir / f"{view['view_id']}.npz"
            np.savez_compressed(
                pointmap_path,
                points_sga=points.astype(np.float32),
                points_map=points_map,
                confidence=confidence.astype(np.float32),
            )
            geometry_views.append({
                "view_id": view["view_id"],
                "image_path": view["image_path"],
                "pointmap_path": str(pointmap_path),
                "width": int(view["width"]),
                "height": int(view["height"]),
                "frame": "map",
                "world_aligned": True,
                "optical_center_group": view["optical_center_group"],
                "cam2w_sga": cam2w_sga[index].astype(float).tolist(),
                "cam2w_sga_native": cam2w_sga_native[index].astype(float).tolist(),
                "cam2w_map": cam2w_map.astype(float).tolist(),
            })

        camera_errors = np.linalg.norm(
            np.asarray(cam2w_map_all)[:, :3, 3] - target_poses[:, :3, 3], axis=1
        )
        sensor_count = sum(
            int(len(np.load(item["sensor_scan_path"], mmap_mode="r")))
            for item in keyframes_by_group.values()
        )
        registered_count = sum(
            int(len(np.load(item["registered_scan_path"], mmap_mode="r")))
            for item in keyframes_by_group.values()
        )
        geometry_manifest = {
            "schema_version": "mast3r_sga_geometry_v1",
            "reconstruction_mode": "sparse_global_alignment",
            "frame": "map",
            "world_aligned": True,
            "reconstruction_id": output.name,
            "optical_center_groups": list(dict.fromkeys(groups)),
            "sim3_sga_to_map": {
                "scale": scale,
                "rotation": rotation.astype(float).tolist(),
                "translation": translation.astype(float).tolist(),
                "source": "post_sga_ros_pose_alignment",
                "pre_alignment": "post_sga_optical_center_consolidation",
            },
            "views": geometry_views,
        }
        geometry_path = output / "geometry_manifest.json"
        _write_json(geometry_path, geometry_manifest)
        report = {
            "status": "completed",
            "reconstruction_mode": "sparse_global_alignment",
            "native_sga_executed": True,
            "get_im_poses_executed": True,
            "get_dense_pts3d_executed": True,
            "any2full_refinement_applied": any2full_refined,
            "geometry_manifest_path": str(geometry_path),
            "reconstruction_id": output.name,
            "optical_center_group_count": len(set(groups)),
            "view_count": len(views),
            "pair_count": len(pair_records),
            "cross_station_pair_count": sum(item["cross_station"] for item in pair_records),
            "pair_graph": pair_records,
            "sim3": geometry_manifest["sim3_sga_to_map"],
            "camera_alignment_rmse_m": float(np.sqrt(np.mean(camera_errors ** 2))),
            "sensor_scan_point_count": sensor_count,
            "registered_scan_point_count": registered_count,
            "sim3_lidar_correspondence_count": int(len(lidar_source)),
            "sim3_lidar_residual_median_m": float(np.median(lidar_residual)),
            "sim3_lidar_residual_p90_m": float(np.percentile(lidar_residual, 90)),
            "runtime": {
                "model_load_seconds": model_load_seconds,
                "alignment_seconds": alignment_seconds,
                "any2full_seconds": any2full_runtime,
                "peak_cuda_gib": float(torch.cuda.max_memory_allocated() / 1024 ** 3),
            },
        }
        _write_json(output / "report.json", report)
        _write_json(response_path, report)
        return 0
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
