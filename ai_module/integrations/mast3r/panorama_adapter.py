"""Calibrated equirectangular-to-pinhole adapter for live MASt3R input.

The generated views all share one physical optical centre.  Their metadata
therefore carries a common ``optical_center_group`` and must not be interpreted
as independent spatial observations by any object-persistence layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class PerspectiveProjection:
    image_bgr: np.ndarray
    intrinsics: np.ndarray
    rotation_panorama_from_camera: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    yaw_deg: float
    pitch_deg: float
    horizontal_fov_deg: float
    panorama_vertical_fov_deg: float


def _rotation_panorama_from_camera(yaw_rad: float, pitch_rad: float) -> np.ndarray:
    """Return optical-camera to panorama-frame rotation (right/down/forward)."""
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    yaw = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float64,
    )
    # Positive pitch looks upward, hence the central ray obtains negative y.
    pitch = np.array(
        [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]],
        dtype=np.float64,
    )
    return yaw @ pitch


def project_equirectangular(
    panorama_bgr: np.ndarray,
    *,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    width: int = 512,
    height: int = 384,
    horizontal_fov_deg: float = 90.0,
    panorama_vertical_fov_deg: float = 120.0,
) -> PerspectiveProjection:
    if panorama_bgr.ndim != 3 or panorama_bgr.shape[2] != 3:
        raise ValueError("panorama_must_be_hwc_bgr")
    if width <= 1 or height <= 1:
        raise ValueError("perspective_size_invalid")
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError("perspective_hfov_invalid")
    if not 0.0 < panorama_vertical_fov_deg <= 180.0:
        raise ValueError("panorama_vertical_fov_invalid")

    pano_height, pano_width = panorama_bgr.shape[:2]
    focal = 0.5 * width / math.tan(math.radians(horizontal_fov_deg) * 0.5)
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5
    intrinsics = np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    rays = np.stack(
        ((u - cx) / focal, (v - cy) / focal, np.ones_like(u)),
        axis=-1,
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    rotation = _rotation_panorama_from_camera(
        math.radians(yaw_deg), math.radians(pitch_deg)
    )
    panorama_rays = rays @ rotation.T
    longitude = np.arctan2(panorama_rays[..., 0], panorama_rays[..., 2])
    latitude = np.arcsin(np.clip(-panorama_rays[..., 1], -1.0, 1.0))
    map_x = ((longitude / (2.0 * math.pi) + 0.5) * pano_width - 0.5)
    map_x = np.mod(map_x, pano_width).astype(np.float32)
    vertical_fov = math.radians(panorama_vertical_fov_deg)
    map_y = ((0.5 - latitude / vertical_fov) * pano_height - 0.5).astype(np.float32)
    tolerance = 1e-3
    if float(map_y.min()) < -0.5 - tolerance or float(map_y.max()) > pano_height - 0.5 + tolerance:
        raise ValueError("perspective_view_exceeds_cropped_panorama_vertical_fov")
    image = cv2.remap(
        panorama_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return PerspectiveProjection(
        image_bgr=image,
        intrinsics=intrinsics,
        rotation_panorama_from_camera=rotation,
        map_x=map_x,
        map_y=map_y,
        yaw_deg=float(yaw_deg),
        pitch_deg=float(pitch_deg),
        horizontal_fov_deg=float(horizontal_fov_deg),
        panorama_vertical_fov_deg=float(panorama_vertical_fov_deg),
    )


def project_mask_to_panorama(
    mask: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    panorama_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize a perspective mask in the panorama without sparse splat holes."""
    if mask.shape != map_x.shape or mask.shape != map_y.shape:
        raise ValueError("mask_and_remap_shape_mismatch")
    pano_height, pano_width = map(int, panorama_shape)
    result = np.zeros((pano_height, pano_width), dtype=np.uint8)
    binary = np.asarray(mask > 0, dtype=np.uint8)
    if not binary.any():
        return result
    contours, _hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    for contour in contours:
        if len(contour) < 3:
            continue
        coordinates = contour[:, 0, :]
        source_x = coordinates[:, 0]
        source_y = coordinates[:, 1]
        longitude = map_x[source_y, source_x] / pano_width * (2.0 * math.pi)
        unwrapped_x = np.unwrap(longitude) / (2.0 * math.pi) * pano_width
        projected_y = np.clip(map_y[source_y, source_x], 0, pano_height - 1)
        for shift in range(-2, 3):
            shifted_x = unwrapped_x + shift * pano_width
            if shifted_x.max() < 0 or shifted_x.min() >= pano_width:
                continue
            polygon = np.rint(np.stack((shifted_x, projected_y), axis=1)).astype(np.int32)
            cv2.fillPoly(result, [polygon], 255)
    # Retain isolated components that are too small to form a contour polygon.
    foreground_y, foreground_x = np.nonzero(binary)
    pano_x = np.rint(map_x[foreground_y, foreground_x]).astype(np.int64) % pano_width
    pano_y = np.clip(np.rint(map_y[foreground_y, foreground_x]), 0, pano_height - 1).astype(np.int64)
    result[pano_y, pano_x] = 255
    return result


def write_perspective_bundle(
    panorama_bgr: np.ndarray,
    output_dir: Path,
    *,
    yaws_deg: Iterable[float],
    pitches_deg: Iterable[float] = (0.0,),
    width: int = 512,
    height: int = 384,
    horizontal_fov_deg: float = 90.0,
    panorama_vertical_fov_deg: float = 120.0,
    optical_center_group: str,
    view_id_prefix: str = "",
    write_remap_arrays: bool = True,
) -> dict:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for pitch_deg in pitches_deg:
        for yaw_deg in yaws_deg:
            projection = project_equirectangular(
                panorama_bgr,
                yaw_deg=float(yaw_deg),
                pitch_deg=float(pitch_deg),
                width=width,
                height=height,
                horizontal_fov_deg=horizontal_fov_deg,
                panorama_vertical_fov_deg=panorama_vertical_fov_deg,
            )
            stem = f"pitch_{float(pitch_deg):+06.1f}_yaw_{float(yaw_deg) % 360.0:06.1f}"
            view_id = f"{view_id_prefix}__{stem}" if view_id_prefix else stem
            image_path = output_dir / f"{stem}.png"
            map_x_path = output_dir / f"{stem}.map_x.npy"
            map_y_path = output_dir / f"{stem}.map_y.npy"
            if not cv2.imwrite(str(image_path), projection.image_bgr):
                raise RuntimeError(f"could_not_write_perspective_view:{image_path}")
            if write_remap_arrays:
                np.save(map_x_path, projection.map_x)
                np.save(map_y_path, projection.map_y)
            records.append(
                {
                    "view_id": view_id,
                    "image_path": str(image_path),
                    "map_x_path": str(map_x_path) if write_remap_arrays else None,
                    "map_y_path": str(map_y_path) if write_remap_arrays else None,
                    "width": width,
                    "height": height,
                    "yaw_deg": float(yaw_deg) % 360.0,
                    "pitch_deg": float(pitch_deg),
                    "horizontal_fov_deg": float(horizontal_fov_deg),
                    "K": projection.intrinsics.tolist(),
                    "R_panorama_from_camera": projection.rotation_panorama_from_camera.tolist(),
                    "optical_center_group": optical_center_group,
                    "independent_spatial_evidence": False,
                }
            )
    manifest = {
        "schema_version": "1.0",
        "source_projection": "central_cropped_equirectangular",
        "source_size": [int(panorama_bgr.shape[1]), int(panorama_bgr.shape[0])],
        "panorama_vertical_fov_deg": float(panorama_vertical_fov_deg),
        "optical_center_group": optical_center_group,
        "views_are_independent_spatial_evidence": False,
        "views": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path)}
