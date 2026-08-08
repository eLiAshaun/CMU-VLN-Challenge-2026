#!/usr/bin/env python3
"""Any2Full depth refinement for MASt3R perspective views.

Replaces MASt3R dense pointmaps with LiDAR-anchored Any2Full depth,
keeping MASt3R's multi-view alignment (camera poses, Sim3) unchanged.

Architecture:
  LiDAR (sparse, accurate) + RGB → Any2Full → dense depth → pointmap
  MASt3R SGA → camera poses + Sim(3) alignment (reused as-is)

Usage as a library::

    from integrations.mast3r.any2full_refiner import refine_view_pointmaps
    refined = refine_view_pointmaps(views, keyframes_by_group, calibration, config)

Usage as a worker::

    python -m integrations.mast3r.any2full_refiner --request /path/to/request.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

# ── path setup ──────────────────────────────────────────────────────────
# __file__ = .../integrations/mast3r/any2full_refiner.py
# parents[2] = ai_module/
# Any2Full lives at ai_module/third_party/Any2Full/
_ANY2FULL_ROOT = (Path(__file__).resolve().parents[2] / "third_party" / "Any2Full")
if str(_ANY2FULL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANY2FULL_ROOT))

from model.ours.any2full import Any2Full  # noqa: E402
from utils.denoise import remove_outliers  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _resolve_path(path: str) -> str:
    """Resolve a path — no-op inside Docker, translates Docker→host on bare metal."""
    # Inside the Docker container, /home/docker/ is the real filesystem.
    if Path("/home/docker/ai_module").is_dir():
        return path
    # On the host, translate Docker paths to the host mount.
    if path.startswith("/home/docker/"):
        path = path.replace("/home/docker/", "/home/robot/cmu_vln/CMU-VLN-Challenge-2026/", 1)
    return path


def _load_rgb_tensor(path: str, device: str) -> torch.Tensor:
    """Load RGB image as normalised (1,3,H,W) tensor."""
    resolved = _resolve_path(path)
    img = Image.open(resolved).convert("RGB")
    t = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return t(img).unsqueeze(0).to(device)


def _depth_from_lidar_projection(
    sensor_points: np.ndarray,
    sensor_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    height: int,
    width: int,
    R_panorama_from_camera: np.ndarray | None = None,
) -> np.ndarray:
    """Project LiDAR points into the camera and return a sparse depth image (H,W).

    Accounts for per-view rotation (R_panorama_from_camera) if provided.

    Returns a float32 array where 0 = no measurement.
    """
    # Transform: sensor → canonical camera
    camera_points = _transform(sensor_points, sensor_to_camera)

    # Apply per-view rotation: canonical camera → view camera
    if R_panorama_from_camera is not None:
        camera_points = camera_points @ R_panorama_from_camera[:3, :3].T

    depths = camera_points[:, 2]
    valid_depth = depths > 0.05

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    u = np.rint(fx * camera_points[:, 0] / np.maximum(depths, 1e-6) + cx).astype(np.int64)
    v = np.rint(fy * camera_points[:, 1] / np.maximum(depths, 1e-6) + cy).astype(np.int64)

    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    keep = valid_depth & in_bounds

    depth_map = np.zeros((height, width), dtype=np.float32)
    if keep.any():
        # When multiple points project to the same pixel, keep the closest.
        order = np.argsort(depths[keep])
        u_keep, v_keep, d_keep = u[keep][order], v[keep][order], depths[keep][order]
        depth_map[v_keep, u_keep] = d_keep
    return depth_map


def _depth_to_pointmap(
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Unproject a dense depth map (H,W) to a 3D pointmap (H,W,3) in camera frame."""
    height, width = depth_map.shape
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    u, v = np.meshgrid(np.arange(width, dtype=np.float32),
                       np.arange(height, dtype=np.float32))
    z = np.maximum(depth_map, 1e-6)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _apply_sim3(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return (scale * (points @ rotation.T) + translation).astype(np.float32)


def _map_from_sensor(state: dict) -> np.ndarray:
    """Build 4x4 sensor→map transform from state estimation."""
    import numpy as np
    x, y, z, w = (float(v) for v in state["orientation_xyzw"])
    norm = np.linalg.norm([x, y, z, w])
    if norm <= 0:
        raise ValueError("state_estimation_quaternion_invalid")
    x, y, z, w = np.asarray([x, y, z, w], dtype=np.float64) / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    matrix[:3, 3] = np.asarray(state["position_xyz"], dtype=np.float64)
    return matrix


def _camera_to_map_transform(
    view: dict,
    state: dict,
    calibration: dict,
) -> np.ndarray:
    """Build view-camera → map transform using real metric geometry.

    Chain (matching sparse_global_mapper._target_camera_poses):
      view-camera → adapter → sensor → map

    - adapter_from_view:  view_camera → adapter  (3×3 rotation from panorama adapter)
    - adapter_to_sensor:  adapter → sensor       (from calibration)
    - map_from_sensor:    sensor → map           (from state estimation)
    """
    source_to_camera = np.asarray(calibration["source_to_camera"], dtype=np.float64)
    adapter_to_camera = np.asarray(
        calibration["adapter_to_camera_canonical"], dtype=np.float64
    )

    # adapter → sensor
    adapter_to_sensor = np.linalg.inv(source_to_camera) @ adapter_to_camera
    # sensor → map
    map_from_sensor = _map_from_sensor(state)
    # adapter → map
    map_from_adapter = map_from_sensor @ adapter_to_sensor

    # view → adapter (rotation only, views share the same optical center)
    R = np.asarray(view.get("R_panorama_from_camera", [[1,0,0],[0,1,0],[0,0,1]]), dtype=np.float64)
    adapter_from_view = np.eye(4, dtype=np.float64)
    adapter_from_view[:3, :3] = R

    return map_from_adapter @ adapter_from_view


# ═══════════════════════════════════════════════════════════════════════════
# Model loader (singleton per process)
# ═══════════════════════════════════════════════════════════════════════════

_model: Any2Full | None = None
_device: str = "cuda"


def _load_checkpoint(model: Any2Full, ckpt_path: str, device: str) -> Any2Full:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    cleaned = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
    model.load_state_dict(cleaned, strict=True)
    return model


class _InferenceArgs:
    """Minimal args namespace matching what Any2Full.forward() expects."""
    def __init__(self):
        self.init_scailing = True
        self.max_depth = 1e3
        self.min_depth = 1e-6
        self.stage = 2  # inference-only, skip pretrained DA loading branch


def _get_model(checkpoint_path: str, encoder: str = "vitb") -> Any2Full:
    global _model
    if _model is not None:
        return _model
    _model = Any2Full(encoder=encoder, da_ckpt_path=None, args=_InferenceArgs())
    _model = _load_checkpoint(_model, checkpoint_path, _device)
    _model = _model.to(_device).eval()
    return _model


# ═══════════════════════════════════════════════════════════════════════════
# Core refinement
# ═══════════════════════════════════════════════════════════════════════════

def refine_single_view(
    view: dict,
    sensor_points: np.ndarray,
    sensor_to_camera: np.ndarray,
    adapter_to_camera_canonical: np.ndarray,
    model: Any2Full,
    *,
    depth_scale: float = 100.0,
    denoise: bool = True,
    denoise_threshold: float = 2.0,
    denoise_kernel_size: int | None = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Any2Full on a single perspective view.

    Parameters
    ----------
    view : dict
        Must contain ``image_path``, ``K`` (3×3 intrinsics), ``height``, ``width``.
    sensor_points : np.ndarray
        (N,3) LiDAR points in sensor frame.
    sensor_to_camera : np.ndarray
        (4,4) sensor-to-camera transform.
    adapter_to_camera_canonical : np.ndarray
        (4,4) adapter-to-camera canonical transform.
    model : Any2Full
        Pre-loaded Any2Full model.

    Returns
    -------
    pointmap : np.ndarray (H,W,3)
        Dense 3D points in camera frame.
    confidence : np.ndarray (H,W)
        Per-pixel confidence (heuristic: 1.0 for all valid depth pixels).
    """
    height, width = int(view["height"]), int(view["width"])
    intrinsics = np.asarray(view["K"], dtype=np.float64)

    # 1. Load RGB
    rgb_tensor = _load_rgb_tensor(str(view["image_path"]), _device)

    # 2. Project LiDAR → sparse depth
    # Full chain: sensor → canonical_camera → adapter → view_camera
    camera_to_adapter = np.linalg.inv(adapter_to_camera_canonical)
    # sensor → canonical_camera
    camera_canonical = _transform(sensor_points, sensor_to_camera)
    # canonical_camera → adapter
    adapter_pts = _transform(camera_canonical, camera_to_adapter)
    # adapter → view_camera (apply view rotation)
    R_view = np.asarray(view.get("R_panorama_from_camera", [[1,0,0],[0,1,0],[0,0,1]]), dtype=np.float64)
    view_camera_pts = adapter_pts @ R_view[:3, :3].T

    # Now project view_camera_pts to image plane
    depths = view_camera_pts[:, 2]
    valid_depth = depths > 0.05
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    u = np.rint(fx * view_camera_pts[:, 0] / np.maximum(depths, 1e-6) + cx).astype(np.int64)
    v = np.rint(fy * view_camera_pts[:, 1] / np.maximum(depths, 1e-6) + cy).astype(np.int64)
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    keep = valid_depth & in_bounds

    sparse_depth = np.zeros((height, width), dtype=np.float32)
    if keep.any():
        order = np.argsort(depths[keep])
        sparse_depth[v[keep][order], u[keep][order]] = depths[keep][order]

    # Convert to tensor (1,1,H,W)
    depth_img = Image.fromarray(sparse_depth)
    dep_tensor = T.ToTensor()(depth_img).unsqueeze(0)

    # Optional denoise
    if denoise and dep_tensor.sum() > 0:
        dep_tensor = remove_outliers(
            dep_tensor,
            min_valid=5,
            threshold=denoise_threshold,
            kernel_size=denoise_kernel_size,
        )
    dep_tensor = dep_tensor.to(_device)

    # 3. Run Any2Full
    sample = {"rgb": rgb_tensor, "dep": dep_tensor}
    with torch.no_grad():
        pred = model(sample)["pred"].squeeze(0).squeeze(0).cpu().numpy()

    # 4. Depth → pointmap
    pointmap = _depth_to_pointmap(pred, intrinsics)

    # 5. Confidence: mark valid (non-zero depth) pixels as confident
    confidence = (pred > 1e-6).astype(np.float32) * 10.0  # heuristic confidence value

    return pointmap, confidence


def refine_view_pointmaps(
    views: list[dict],
    keyframes_by_group: dict[str, dict],
    calibration: dict,
    checkpoint_path: str,
    *,
    encoder: str = "vitb",
    depth_scale: float = 100.0,
    denoise: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], float, list[bool]]:
    """Replace MASt3R pointmaps with Any2Full-refined ones.

    Views without LiDAR coverage are skipped (returns empty pointmaps)
    so the caller can fall back to MASt3R per-view.

    Returns
    -------
    dense_points : list[np.ndarray]
        Refined pointmaps (camera-frame), one per view.
    confidences : list[np.ndarray]
        Per-pixel confidence maps.
    runtime_seconds : float
        Total inference time.
    lidar_covered : list[bool]
        Whether each view had sufficient LiDAR coverage.
    """
    sensor_to_camera = np.asarray(calibration["source_to_camera"], dtype=np.float64)
    adapter_to_camera = np.asarray(
        calibration["adapter_to_camera_canonical"], dtype=np.float64
    )

    model = _get_model(checkpoint_path, encoder)
    model.eval()

    dense_points: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    lidar_covered: list[bool] = []

    started = time.perf_counter()
    for view in views:
        group = str(view["optical_center_group"])
        keyframe = keyframes_by_group[group]
        sensor_points = np.load(_resolve_path(keyframe["sensor_scan_path"])).astype(np.float64)

        pointmap, confidence = refine_single_view(
            view,
            sensor_points,
            sensor_to_camera,
            adapter_to_camera,
            model,
            depth_scale=depth_scale,
            denoise=denoise,
        )

        # Check if Any2Full produced meaningful depth (not saturated at max_depth)
        cam_z = pointmap.reshape(-1, 3)[:, 2]
        finite_mask = np.isfinite(cam_z) & (cam_z > 0.05)
        if finite_mask.any():
            median_z = float(np.median(cam_z[finite_mask]))
            # If median depth is near MAX_DEPTH, LiDAR didn't cover this view
            has_lidar = median_z < 900.0
        else:
            has_lidar = False

        if has_lidar:
            dense_points.append(pointmap.reshape(-1, 3))
            confidences.append(confidence)
            lidar_covered.append(True)
        else:
            # Return empty placeholders — caller should use MASt3R fallback
            dense_points.append(np.zeros((0, 3), dtype=np.float32))
            confidences.append(np.zeros_like(confidence))
            lidar_covered.append(False)

    runtime = time.perf_counter() - started
    return dense_points, confidences, runtime, lidar_covered


# ═══════════════════════════════════════════════════════════════════════════
# CLI worker entry-point (compatible with run_worker convention)
# ═══════════════════════════════════════════════════════════════════════════

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
        # Load geometry manifest (views + existing MASt3R alignment)
        geometry = json.loads(
            Path(request["geometry_manifest_path"]).read_text(encoding="utf-8")
        )
        views = geometry["views"]
        sim3 = geometry["sim3_sga_to_map"]

        # Load keyframes
        keyframes_by_group: dict[str, dict] = {}
        for keyframe in request["keyframes"]:
            keyframes_by_group[str(keyframe["optical_center_group"])] = keyframe

        # Load perspective adapter manifests to get camera intrinsics (K) per view
        manifest_views_lookup: dict[str, dict] = {}
        for keyframe in keyframes_by_group.values():
            manifest_path = Path(keyframe["manifest_path"])
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for mv in manifest.get("views", []):
                    manifest_views_lookup[mv["view_id"]] = mv

        # Merge intrinsics and camera rotation from manifest views into geometry views
        for view in views:
            mv = manifest_views_lookup.get(view["view_id"])
            if mv:
                view["K"] = mv.get("K")
                view["image_path"] = mv.get("image_path", view.get("image_path", ""))
                view["R_panorama_from_camera"] = mv.get("R_panorama_from_camera", [[1,0,0],[0,1,0],[0,0,1]])

        calibration = json.loads(
            Path(request["calibration_path"]).read_text(encoding="utf-8")
        )

        # Run Any2Full refinement (produces camera-frame pointmaps)
        torch.cuda.empty_cache()
        dense_points, confidences, runtime, lidar_covered = refine_view_pointmaps(
            views,
            keyframes_by_group,
            calibration,
            str(request["checkpoint"]),
            encoder=str(request.get("encoder", "vitb")),
            depth_scale=float(request.get("depth_scale", 100.0)),
            denoise=bool(request.get("denoise", True)),
        )

        # Transform camera-frame pointmaps → map frame using REAL geometry
        pointmaps_dir = output / "pointmaps"
        pointmaps_dir.mkdir(exist_ok=True)
        refined_views: list[dict] = []

        views_with_lidar = sum(1 for v in lidar_covered if v)
        print(f"[any2full] LiDAR coverage: {views_with_lidar}/{len(views)} views")

        for idx, (view, points, confidence, has_lidar) in enumerate(
            zip(views, dense_points, confidences, lidar_covered)
        ):
            if not has_lidar:
                # Keep the original MASt3R pointmap — don't modify this view
                refined_views.append(view)
                continue

            points_reshaped = points.reshape(*confidence.shape, 3)
            h, w = confidence.shape

            # Get state estimation for this view's station
            group = str(view["optical_center_group"])
            keyframe = keyframes_by_group[group]
            state = json.loads(
                Path(_resolve_path(keyframe["state_estimation_path"])).read_text(encoding="utf-8")
            )

            # Build camera→map transform using real metric geometry
            cam2map = _camera_to_map_transform(view, state, calibration)

            # Transform camera points (H*W, 3) → map frame
            points_homog = np.concatenate([
                points_reshaped.reshape(-1, 3),
                np.ones((h * w, 1), dtype=np.float32),
            ], axis=-1)  # (N, 4)
            points_map_flat = (points_homog @ cam2map.T)[:, :3]  # (N, 3)
            points_map = points_map_flat.reshape(h, w, 3).astype(np.float32)

            pointmap_path = pointmaps_dir / f"{view['view_id']}_any2full.npz"
            np.savez_compressed(
                pointmap_path,
                points_sga=points_reshaped.astype(np.float32),    # camera frame
                points_map=points_map,                             # map frame (real geometry)
                confidence=confidence.astype(np.float32),
            )
            refined_views.append({
                **view,
                "pointmap_path": str(pointmap_path),
                "refinement": "any2full",
            })

        # Write updated geometry manifest
        refined_manifest = {
            **geometry,
            "views": refined_views,
            "refinement": {
                "method": "any2full",
                "encoder": request.get("encoder", "vitb"),
                "runtime_seconds": runtime,
            },
        }
        manifest_path = output / "geometry_manifest.json"
        _write_json(manifest_path, refined_manifest)

        response = {
            "status": "completed",
            "geometry_manifest_path": str(manifest_path),
            "view_count": len(refined_views),
            "runtime_seconds": runtime,
            "reconstruction_id": geometry.get("reconstruction_id", output.name),
            "frame": geometry["frame"],
            "world_aligned": geometry.get("world_aligned", True),
            "optical_center_group_count": len(
                set(v["optical_center_group"] for v in refined_views)
            ),
        }
        _write_json(response_path, response)
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
