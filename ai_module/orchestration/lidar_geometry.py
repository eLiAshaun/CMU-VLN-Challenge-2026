"""LiDAR-first lifting for perspective-view masks.

The competition already supplies a timestamped sensor-frame scan, a map-frame
registered scan, and the map<-sensor pose.  This module uses those authorities
directly.  MASt3R is intentionally absent from the normal path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from integrations.mast3r.panorama_adapter import (
    project_mask_to_panorama,
)
from orchestration.observation_contract import (
    MASK_PANORAMA_PIXELS,
    MASK_PERSPECTIVE_PIXELS,
    ObservationContract,
    validate_mask_contract,
)


_UNAVAILABLE_METRIC = 1_000_000.0
_MIN_STRICT_LIDAR_POINTS = 6


_PANORAMA_OPTICAL_FROM_PHYSICAL = np.asarray(
    [
        [0.0, 1.0, 0.0],   # right
        [0.0, 0.0, -1.0],  # down
        [1.0, 0.0, 0.0],   # forward
    ],
    dtype=np.float64,
)


def _find_camera_panorama(image_path: object) -> Path | None:
    image = Path(str(image_path or "")).resolve()
    if not image.name:
        return None
    for parent in (image.parent, *image.parents[:6]):
        candidate = parent / "camera_panorama.png"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _canonical_class_label(value: object) -> str:
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


def _appearance_similarity(first: Mapping[str, Any], second: Mapping[str, Any]) -> float | None:
    try:
        a = np.asarray(first.get("appearance_descriptor"), dtype=np.float64)
        b = np.asarray(second.get("appearance_descriptor"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if a.ndim != 1 or b.ndim != 1 or not len(a) or a.shape != b.shape:
        return None
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / norm) if norm > 1e-9 else None


def _center_uncertainty_radius(value: Mapping[str, Any]) -> float:
    try:
        covariance = np.asarray(value.get("center_cov"), dtype=np.float64)
    except (TypeError, ValueError):
        covariance = np.zeros((0, 0), dtype=np.float64)
    if covariance.shape == (3, 3) and np.isfinite(covariance).all():
        horizontal = 0.5 * (covariance[:2, :2] + covariance[:2, :2].T)
        try:
            variance = float(max(0.0, np.linalg.eigvalsh(horizontal).max()))
            return float(max(0.04, min(0.35, 2.5 * math.sqrt(variance))))
        except np.linalg.LinAlgError:
            pass
    return 0.08


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
    first_geometry = _geometry_vectors(first)
    second_geometry = _geometry_vectors(second)
    if first_geometry is None or second_geometry is None:
        return {
            "compatible": False,
            "horizontal_distance_m": _UNAVAILABLE_METRIC,
            "vertical_gap_m": _UNAVAILABLE_METRIC,
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
    }


def _bbox_iou_2d(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != 4 or len(second) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(
        0.0, bx2 - bx1
    ) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 1e-9 else 0.0


def _same_view_fragment_compatible(
    first_box: Sequence[float],
    second_box: Sequence[float],
    fragment_metrics: Mapping[str, float | bool],
) -> bool:
    """Recognize vertically stacked same-class boxes for one object."""
    if len(first_box) != 4 or len(second_box) != 4:
        return False
    try:
        ax1, ay1, ax2, ay2 = [float(value) for value in first_box]
        bx1, by1, bx2, by2 = [float(value) for value in second_box]
    except (TypeError, ValueError):
        return False
    first_width = max(1e-6, ax2 - ax1)
    second_width = max(1e-6, bx2 - bx1)
    first_height = max(1e-6, ay2 - ay1)
    second_height = max(1e-6, by2 - by1)
    horizontal_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    horizontal_ratio = horizontal_overlap / min(first_width, second_width)
    vertical_gap = max(0.0, max(ay1, by1) - min(ay2, by2))
    tight_footprint = bool(
        fragment_metrics.get("compatible", False)
        and float(fragment_metrics.get("horizontal_distance_m", _UNAVAILABLE_METRIC)) <= 0.24
        and float(fragment_metrics.get("vertical_gap_m", _UNAVAILABLE_METRIC)) <= 0.28
    )
    return bool(
        tight_footprint
        and horizontal_ratio >= 0.40
        and vertical_gap <= 1.25 * max(first_height, second_height)
    )

def _load_points(path_value: object) -> np.ndarray:
    path = Path(str(path_value or ""))
    if not path.is_file():
        return np.zeros((0, 3), dtype=np.float64)
    value = np.load(path)
    if isinstance(value, np.lib.npyio.NpzFile):
        try:
            for key in ("points", "world_points", "lidar_support_points_map"):
                if key in value:
                    points = np.asarray(value[key], dtype=np.float64)
                    break
            else:
                return np.zeros((0, 3), dtype=np.float64)
        finally:
            value.close()
    else:
        points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float64)
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) <= 1:
        return points
    # Repeated rows are repeated encodings of one physical LiDAR return, not
    # independent spatial support. Preserve scan order while ensuring a
    # duplicated return cannot satisfy the strict-geometry point count.
    _, unique_indices = np.unique(points, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def _quaternion_rotation(xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(value) for value in xyzw[:4]]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("state_estimation_quaternion_invalid")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _load_map_from_sensor(path_value: object) -> np.ndarray:
    path = Path(str(path_value or ""))
    if not path.is_file():
        raise FileNotFoundError(f"state_estimation_missing:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("state_estimation_payload_invalid")
    if str(payload.get("frame_id", "")).lstrip("/") != "map":
        raise ValueError("state_estimation_frame_not_map")
    child = str(payload.get("child_frame_id", "")).lstrip("/")
    if child not in {"sensor", "sensor_at_scan"}:
        raise ValueError("state_estimation_child_not_sensor")
    position = payload.get("position_xyz")
    orientation = payload.get("orientation_xyzw")
    if not isinstance(position, Sequence) or len(position) < 3:
        raise ValueError("state_estimation_position_invalid")
    if not isinstance(orientation, Sequence) or len(orientation) < 4:
        raise ValueError("state_estimation_orientation_invalid")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quaternion_rotation(orientation)
    transform[:3, 3] = [float(value) for value in position[:3]]
    return transform


def _load_sensor_to_panorama_physical(path_value: object) -> np.ndarray:
    path = Path(str(path_value or ""))
    if not path.is_file():
        raise FileNotFoundError(f"camera_lidar_calibration_missing:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload.get("source_to_camera"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("camera_lidar_calibration_invalid")
    return matrix


def _map_points(points_sensor: np.ndarray, map_from_sensor: np.ndarray) -> np.ndarray:
    return points_sensor @ map_from_sensor[:3, :3].T + map_from_sensor[:3, 3]


def _sensor_points(points_map: np.ndarray, map_from_sensor: np.ndarray) -> np.ndarray:
    return (points_map - map_from_sensor[:3, 3]) @ map_from_sensor[:3, :3]


def _project_sensor_points(
    points_sensor: np.ndarray,
    *,
    sensor_to_panorama_physical: np.ndarray,
    rotation_panorama_from_view: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    physical = (
        points_sensor @ sensor_to_panorama_physical[:3, :3].T
        + sensor_to_panorama_physical[:3, 3]
    )
    panorama_optical = physical @ _PANORAMA_OPTICAL_FROM_PHYSICAL.T
    view_optical = panorama_optical @ rotation_panorama_from_view
    depth = view_optical[:, 2]
    valid = np.isfinite(view_optical).all(axis=1) & (depth > 0.05)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return indices, np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    camera = view_optical[indices]
    pixels_h = camera @ intrinsics.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    inside = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= height - 1)
    )
    return indices[inside], pixels[inside], camera[inside, 2]


def _select_foreground_depth(points: np.ndarray, depths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select the nearest sufficiently supported depth mode inside a SAM mask.

    Choosing the largest mode frequently selects the wall or table behind a
    small lamp.  We reject tiny near-range noise, then prefer the nearest mode
    whose support is a meaningful fraction of the strongest mode.
    """
    if len(points) <= 3:
        return points, depths
    order = np.argsort(depths)
    sorted_depth = depths[order]
    median = float(np.median(sorted_depth))
    gap = max(0.08, min(0.30, 0.035 * max(median, 1.0)))
    cuts = np.flatnonzero(np.diff(sorted_depth) > gap) + 1
    groups = [group for group in np.split(order, cuts) if len(group)]
    if not groups:
        return points, depths
    maximum_support = max(len(group) for group in groups)
    minimum_viable = max(3, int(math.ceil(0.18 * maximum_support)))
    viable = [group for group in groups if len(group) >= minimum_viable]
    if not viable:
        viable = [max(groups, key=len)]

    def score(group: np.ndarray) -> tuple[float, float, float]:
        group_depth = depths[group]
        depth_median = float(np.median(group_depth))
        spread = float(np.quantile(group_depth, 0.90) - np.quantile(group_depth, 0.10))
        support_penalty = 0.18 * max(0.0, maximum_support / max(1, len(group)) - 1.0)
        return (depth_median + support_penalty + 0.25 * spread, spread, -len(group))

    chosen = min(viable, key=score)
    selected_points = points[chosen]
    selected_depths = depths[chosen]
    if len(selected_points) >= 6:
        center = np.median(selected_points, axis=0)
        radial = np.linalg.norm(selected_points - center, axis=1)
        radial_median = float(np.median(radial))
        radial_mad = float(np.median(np.abs(radial - radial_median)))
        threshold = radial_median + max(0.08, 4.0 * 1.4826 * radial_mad)
        keep = radial <= threshold
        if int(keep.sum()) >= 3:
            selected_points = selected_points[keep]
            selected_depths = selected_depths[keep]
    return selected_points, selected_depths


def _mask_membership(mask: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    if not len(pixels):
        return np.zeros(0, dtype=bool)
    binary = np.asarray(mask > 0, dtype=np.uint8)
    kernel_size = max(1, int(round(min(binary.shape) * 0.005)))
    if kernel_size >= 2:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded = cv2.erode(binary, kernel)
        if int(eroded.sum()) >= 20:
            binary = eroded
    u = np.clip(np.rint(pixels[:, 0]).astype(np.int64), 0, binary.shape[1] - 1)
    v = np.clip(np.rint(pixels[:, 1]).astype(np.int64), 0, binary.shape[0] - 1)
    return binary[v, u] > 0


def _sparse_local_range(
    *,
    mask: np.ndarray,
    points_sensor: np.ndarray,
    projected_pixels: np.ndarray,
    projected_depths: np.ndarray,
    map_from_sensor: np.ndarray,
    map_from_view: np.ndarray,
) -> tuple[float, int] | None:
    """Estimate a radial layer near a sparse mask without promoting it to 3D.

    Thin foliage, chair legs, and other small silhouettes often contain only
    LiDAR returns from the background.  A narrow image-space expansion can
    recover the locally supported foreground depth mode, but those adjacent
    points still do not belong to the object.  They therefore inform only the
    range of a broad bearing observation and are never saved as object support.
    """
    if (
        not len(points_sensor)
        or len(points_sensor) != len(projected_pixels)
        or len(points_sensor) != len(projected_depths)
    ):
        return None
    binary = np.asarray(mask > 0, dtype=np.uint8)
    kernel_size = int(round(min(binary.shape[:2]) * 0.0065))
    kernel_size = max(3, min(7, kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    expanded = cv2.dilate(
        binary,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
    )
    u = np.clip(
        np.rint(projected_pixels[:, 0]).astype(np.int64),
        0,
        expanded.shape[1] - 1,
    )
    v = np.clip(
        np.rint(projected_pixels[:, 1]).astype(np.int64),
        0,
        expanded.shape[0] - 1,
    )
    inside = expanded[v, u] > 0
    local_sensor, local_depths = _select_foreground_depth(
        points_sensor[inside], projected_depths[inside]
    )
    if len(local_sensor) < 3:
        return None
    local_map = _map_points(local_sensor, map_from_sensor)
    camera = np.asarray(map_from_view[:3, 3], dtype=np.float64)
    ranges = np.linalg.norm(local_map - camera[None, :], axis=1)
    ranges = ranges[np.isfinite(ranges) & (ranges > 0.05)]
    if len(ranges) < 3:
        return None
    return float(np.median(ranges)), int(len(ranges))



def _appearance_descriptor(
    image: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: Sequence[float],
) -> tuple[list[float], float]:
    """Return a compact foreground colour/shape descriptor.

    The descriptor is intentionally model-free and cheap. It gives persistent
    association a cue that is independent of sparse LiDAR geometry, while its
    quality value prevents tiny or badly clipped masks from dominating.
    """
    if image is None or mask is None:
        return [], 0.0

    if image.shape[:2] != mask.shape[:2]:
        return [], 0.0

    height, width = mask.shape[:2]
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy[:4]]
    except (TypeError, ValueError):
        return [], 0.0

    x1i = max(0, min(width - 1, int(math.floor(x1))))
    y1i = max(0, min(height - 1, int(math.floor(y1))))
    x2i = max(x1i + 1, min(width, int(math.ceil(x2))))
    y2i = max(y1i + 1, min(height, int(math.ceil(y2))))

    roi = image[y1i:y2i, x1i:x2i]
    roi_mask = mask[y1i:y2i, x1i:x2i]

    if roi.shape[:2] != roi_mask.shape[:2]:
        return [], 0.0

    roi_mask = np.asarray(roi_mask > 0, dtype=np.uint8)
    pixel_count = int(roi_mask.sum())
    roi_area = max(1, int(roi_mask.size))

    if pixel_count < 24:
        return [], 0.0

    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    except Exception:
        return [], 0.0

    descriptor: list[float] = []
    for channel, bins, value_range in ((0, 12, (0, 180)), (1, 4, (0, 256)), (2, 4, (0, 256))):
        try:
            hist_mask = np.ascontiguousarray(roi_mask)
            hist = cv2.calcHist([hsv], [channel], hist_mask, [bins], list(value_range)).reshape(-1)
            total = float(hist.sum())
            descriptor.extend((hist / total).astype(float).tolist() if total > 0 else [0.0] * bins)
        except Exception:
            descriptor.extend([0.0] * bins)

    try:
        pixels = roi[roi_mask > 0].astype(np.float32) / 255.0
        descriptor.extend(np.mean(pixels, axis=0).astype(float).tolist())
        descriptor.extend(np.std(pixels, axis=0).astype(float).tolist())
    except Exception:
        descriptor.extend([0.0] * 6)

    fill_fraction = pixel_count / roi_area
    aspect = (x2i - x1i) / max(1.0, float(y2i - y1i))
    descriptor.extend([
        float(fill_fraction),
        float(math.tanh(math.log(max(aspect, 1e-3)))),
    ])

    vector = np.asarray(descriptor, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-9:
        return [], 0.0

    quality = min(1.0, pixel_count / 800.0) * min(1.0, fill_fraction / 0.35)
    touches_edge = x1i <= 1 or y1i <= 1 or x2i >= width - 1 or y2i >= height - 1
    if touches_edge:
        quality *= 0.75

    return (vector / norm).astype(float).tolist(), float(max(0.0, min(1.0, quality)))


def _registered_support_metrics(
    points_map: np.ndarray,
    registered_map: np.ndarray,
) -> tuple[float | None, float]:
    if not len(points_map) or not len(registered_map):
        return None, 0.0
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(registered_map).query(points_map, k=1, workers=-1)
    except (ImportError, TypeError, ValueError):
        sample = registered_map
        if len(sample) > 20000:
            stride = max(1, len(sample) // 20000)
            sample = sample[::stride]
        distances = np.sqrt(
            np.min(
                np.sum((points_map[:, None, :] - sample[None, :, :]) ** 2, axis=2),
                axis=1,
            )
        )
    distances = np.asarray(distances, dtype=np.float64)
    finite = distances[np.isfinite(distances)]
    if not len(finite):
        return None, 0.0
    return float(np.median(finite)), float(np.mean(finite <= 0.25))

def _robust_geometry(
    points_map: np.ndarray,
) -> tuple[
    list[float],
    list[float],
    list[list[float]],
    list[list[float]],
    str,
    str,
]:
    low = np.quantile(points_map, 0.05, axis=0)
    high = np.quantile(points_map, 0.95, axis=0)
    center = 0.5 * (low + high)
    extent = np.maximum(high - low, np.asarray([0.03, 0.03, 0.03]))
    center_covariance_provenance = "extent_derived_fallback"
    if len(points_map) >= _MIN_STRICT_LIDAR_POINTS:
        covariance = np.cov(points_map.T) / max(1, len(points_map))
        valid_statistical = bool(
            covariance.shape == (3, 3)
            and np.isfinite(covariance).all()
            and np.allclose(covariance, covariance.T, atol=1e-6)
            and np.all(np.diag(covariance) >= 0.0)
        )
        if not valid_statistical:
            covariance = np.diag(np.maximum(extent * 0.15, 0.03) ** 2)
        else:
            center_covariance_provenance = "pointcloud_statistical"
    else:
        covariance = np.diag(np.maximum(extent * 0.25, 0.05) ** 2)
    extent_cov = np.diag(np.maximum(extent * 0.15, 0.02) ** 2)
    return (
        center.astype(float).tolist(),
        extent.astype(float).tolist(),
        covariance.astype(float).tolist(),
        extent_cov.astype(float).tolist(),
        center_covariance_provenance,
        "extent_derived",
    )


def _camera_pose_map(
    *,
    map_from_sensor: np.ndarray,
    sensor_to_panorama_physical: np.ndarray,
    rotation_panorama_from_view: np.ndarray,
) -> list[list[float]]:
    r_panoopt_sensor = _PANORAMA_OPTICAL_FROM_PHYSICAL @ sensor_to_panorama_physical[:3, :3]
    t_panoopt_sensor = _PANORAMA_OPTICAL_FROM_PHYSICAL @ sensor_to_panorama_physical[:3, 3]
    origin_sensor = -r_panoopt_sensor.T @ t_panoopt_sensor
    rotation_sensor_from_view = r_panoopt_sensor.T @ rotation_panorama_from_view
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = map_from_sensor[:3, :3] @ rotation_sensor_from_view
    transform[:3, 3] = map_from_sensor[:3, :3] @ origin_sensor + map_from_sensor[:3, 3]
    return transform.astype(float).tolist()


def _bearing_geometry(
    *,
    bbox: Sequence[float],
    intrinsics: np.ndarray,
    map_from_view: np.ndarray,
    range_m: float,
) -> tuple[
    list[float],
    list[float],
    list[list[float]],
    list[list[float]],
    dict[str, Any],
]:
    """Build a conservative bearing observation when LiDAR support is sparse."""
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0
    pixel = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5, 1.0], dtype=np.float64)
    ray_view = np.linalg.solve(intrinsics, pixel)
    ray_view /= max(float(np.linalg.norm(ray_view)), 1e-9)
    ray_map = np.asarray(map_from_view[:3, :3], dtype=np.float64) @ ray_view
    ray_map /= max(float(np.linalg.norm(ray_map)), 1e-9)
    origin = np.asarray(map_from_view[:3, 3], dtype=np.float64)
    range_value = max(0.5, float(range_m))
    center = origin + range_value * ray_map
    angular_sigma = max(0.035, 1.0 / max(12.0, float(max(x2 - x1, y2 - y1))))
    radial_sigma = max(0.45, 0.18 * range_value)
    lateral_sigma = max(0.20, range_value * angular_sigma)
    covariance = (
        lateral_sigma * lateral_sigma * (np.eye(3) - np.outer(ray_map, ray_map))
        + radial_sigma * radial_sigma * np.outer(ray_map, ray_map)
    )
    extent = np.asarray([max(0.20, 2.0 * lateral_sigma)] * 2 + [0.35], dtype=np.float64)
    bearing = {
        "camera_origin_map": origin.astype(float).tolist(),
        "unit_ray_map": ray_map.astype(float).tolist(),
        "range_mean_m": range_value,
        "range_variance_m2": radial_sigma * radial_sigma,
        "angular_sigma_rad": angular_sigma,
        "source": "mask_bearing",
    }
    return (
        center.astype(float).tolist(),
        extent.astype(float).tolist(),
        covariance.astype(float).tolist(),
        np.diag(np.maximum(extent * 0.25, 0.05) ** 2).astype(float).tolist(),
        bearing,
    )



def _aabb_iou_3d(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    first_center = np.asarray(first.get("center_3d", ()), dtype=np.float64)
    second_center = np.asarray(second.get("center_3d", ()), dtype=np.float64)
    first_extent = np.asarray(first.get("bbox_3d", ()), dtype=np.float64)
    second_extent = np.asarray(second.get("bbox_3d", ()), dtype=np.float64)
    if (
        first_center.shape != (3,)
        or second_center.shape != (3,)
        or first_extent.shape != (3,)
        or second_extent.shape != (3,)
    ):
        return 0.0
    first_low, first_high = first_center - 0.5 * first_extent, first_center + 0.5 * first_extent
    second_low, second_high = second_center - 0.5 * second_extent, second_center + 0.5 * second_extent
    overlap = np.maximum(0.0, np.minimum(first_high, second_high) - np.maximum(first_low, second_low))
    intersection = float(np.prod(overlap))
    first_volume = float(np.prod(np.maximum(first_extent, 0.0)))
    second_volume = float(np.prod(np.maximum(second_extent, 0.0)))
    union = first_volume + second_volume - intersection
    return intersection / union if union > 0.0 else 0.0


def _panorama_support_overlap(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, float] | None:
    """Compare masks in their shared original-panorama coordinate frame."""
    first_path = Path(str(first.get("panorama_mask_path", "")))
    second_path = Path(str(second.get("panorama_mask_path", "")))
    if not first_path.is_file() or not second_path.is_file():
        return None
    first_mask = cv2.imread(str(first_path), cv2.IMREAD_GRAYSCALE)
    second_mask = cv2.imread(str(second_path), cv2.IMREAD_GRAYSCALE)
    if first_mask is None or second_mask is None or first_mask.shape != second_mask.shape:
        return None
    first_support = first_mask > 0
    second_support = second_mask > 0
    intersection = int(np.count_nonzero(first_support & second_support))
    first_area = int(np.count_nonzero(first_support))
    second_area = int(np.count_nonzero(second_support))
    union = first_area + second_area - intersection
    return {
        "intersection_px": float(intersection),
        "first_area_px": float(first_area),
        "second_area_px": float(second_area),
        "iou": intersection / union if union else 0.0,
        "containment_iom": (
            intersection / min(first_area, second_area)
            if min(first_area, second_area) else 0.0
        ),
        "first_containment": intersection / first_area if first_area else 0.0,
        "second_containment": intersection / second_area if second_area else 0.0,
    }


def _metric_geometry_reliable(value: Mapping[str, Any]) -> bool:
    return bool(
        str(value.get("covariance_mode", "")) == "strict"
        and int(value.get("lidar_support_count", 0) or 0) >= _MIN_STRICT_LIDAR_POINTS
    )


def _voxel_support_f1(first_path: object, second_path: object, voxel_size_m: float = 0.06) -> float:
    first = _load_points(first_path)
    second = _load_points(second_path)
    if not len(first) or not len(second):
        return 0.0
    first_keys = {
        tuple(int(value) for value in row)
        for row in np.rint(first / float(voxel_size_m)).astype(np.int64)
    }
    second_keys = {
        tuple(int(value) for value in row)
        for row in np.rint(second / float(voxel_size_m)).astype(np.int64)
    }
    if not first_keys or not second_keys:
        return 0.0
    expanded_first = {
        (key[0] + dx, key[1] + dy, key[2] + dz)
        for key in first_keys
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    }
    expanded_second = {
        (key[0] + dx, key[1] + dy, key[2] + dz)
        for key in second_keys
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    }
    precision = sum(key in expanded_first for key in second_keys) / len(second_keys)
    recall = sum(key in expanded_second for key in first_keys) / len(first_keys)
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _same_station_instance(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    class_name = _canonical_class_label(first.get("canonical_class", ""))
    if not class_name or class_name != _canonical_class_label(second.get("canonical_class", "")):
        return False
    if str(first.get("optical_center_group", "")) != str(second.get("optical_center_group", "")):
        return False
    first_id = str(first.get("observation_id", ""))
    second_id = str(second.get("observation_id", ""))
    if (
        second_id in {
            str(value)
            for value in first.get("coverage_member_observation_ids", ())
        }
        or first_id in {
            str(value)
            for value in second.get("coverage_member_observation_ids", ())
        }
    ):
        return False
    panorama_overlap = _panorama_support_overlap(first, second)
    if (
        panorama_overlap is not None
        and panorama_overlap["containment_iom"] >= 0.80
        and panorama_overlap["iou"] >= 0.20
    ):
        # Direct support containment in the original panorama is stronger
        # same-instance evidence than sparse or biased lifted centroids.
        return True
    same_representative_view = bool(
        str(first.get("representative_view_id", ""))
        and str(first.get("representative_view_id", ""))
        == str(second.get("representative_view_id", ""))
    )
    first_box = first.get("representative_bbox_xyxy") or ()
    second_box = second.get("representative_bbox_xyxy") or ()
    fragment_metrics = _vertical_fragment_metrics(first, second)
    same_view_disjoint = bool(
        same_representative_view
        and _bbox_iou_2d(first_box, second_box) < 0.08
    )
    try:
        center_distance = float(np.linalg.norm(
            np.asarray(first["center_3d"], dtype=np.float64)
            - np.asarray(second["center_3d"], dtype=np.float64)
        ))
        first_diag = float(np.linalg.norm(np.asarray(first["bbox_3d"], dtype=np.float64)))
        second_diag = float(np.linalg.norm(np.asarray(second["bbox_3d"], dtype=np.float64)))
    except (KeyError, TypeError, ValueError):
        return False
    uncertainty = _center_uncertainty_radius(first) + _center_uncertainty_radius(second)
    center_limit = max(0.12, min(0.60, 0.24 * (first_diag + second_diag) + uncertainty))
    if center_distance > center_limit and not bool(fragment_metrics["compatible"]):
        return False
    support_f1 = _voxel_support_f1(first.get("pointcloud_path"), second.get("pointcloud_path"))
    box_iou = _aabb_iou_3d(first, second)
    appearance = _appearance_similarity(first, second)
    uncertainty_consistent = center_distance <= 0.18 + uncertainty
    if appearance is not None and appearance < 0.40:
        return False
    fragment_compatible = _same_view_fragment_compatible(
        first_box, second_box, fragment_metrics
    )
    # Score the available identity evidence continuously.  A disjoint
    # same-image box lowers the affinity, but is no longer a separate boolean
    # veto; sufficiently consistent metric/appearance/fragment evidence can
    # still recover detector or SAM splits at this station.
    identity_affinity = 0.0
    identity_affinity += 0.35 * min(1.0, support_f1 / 0.30)
    identity_affinity += 0.25 * min(1.0, box_iou / 0.25)
    if appearance is not None:
        identity_affinity += 0.25 * max(
            0.0, min(1.0, (appearance - 0.40) / 0.50)
        )
    if uncertainty_consistent:
        identity_affinity += 0.20
    if fragment_compatible:
        identity_affinity += 0.25
    if same_view_disjoint and not fragment_compatible:
        identity_affinity -= 0.20
    tight_fragment_footprint = bool(
        fragment_metrics["compatible"]
        and float(fragment_metrics["horizontal_distance_m"]) <= 0.24
        and float(fragment_metrics["vertical_gap_m"]) <= 0.28
    )
    return bool(
        identity_affinity >= 0.50
        or tight_fragment_footprint
    )


def _coverage_children_are_distinct(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    overlap = _panorama_support_overlap(first, second)
    if overlap is None or overlap["containment_iom"] > 0.10:
        return False
    if not (_metric_geometry_reliable(first) and _metric_geometry_reliable(second)):
        return False
    try:
        distance = float(np.linalg.norm(
            np.asarray(first["center_3d"], dtype=np.float64)
            - np.asarray(second["center_3d"], dtype=np.float64)
        ))
        core_scale = 0.25 * min(
            float(np.linalg.norm(np.asarray(first["bbox_3d"], dtype=np.float64))),
            float(np.linalg.norm(np.asarray(second["bbox_3d"], dtype=np.float64))),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        _aabb_iou_3d(first, second) <= 0.0
        and distance > max(0.12, core_scale)
    )


def _safe_component(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in value.lower())
    return normalized.strip("_") or "object"


def _merge_station_group(
    group: Sequence[Mapping[str, Any]],
    *,
    acquisition_id: str,
    group_index: int,
    geometry_dir: Path,
) -> dict[str, Any]:
    if len(group) == 1:
        return dict(group[0])
    members = [dict(value) for value in group]
    representative = max(
        members,
        key=lambda value: (
            float(value.get("semantic_probability", 0.0)),
            int(value.get("lidar_support_count", 0)),
        ),
    )
    clouds = [_load_points(value.get("pointcloud_path")) for value in members]
    nonempty = [value for value in clouds if len(value)]
    fused = np.concatenate(nonempty, axis=0) if nonempty else np.zeros((0, 3), dtype=np.float64)
    if len(fused):
        voxel = np.rint(fused / 0.03).astype(np.int64)
        _, selected = np.unique(voxel, axis=0, return_index=True)
        fused = fused[np.sort(selected)]
    if len(fused):
        (
            center,
            extent,
            center_cov,
            extent_cov,
            center_covariance_provenance,
            extent_covariance_provenance,
        ) = _robust_geometry(fused)
    else:
        center = list(representative.get("center_3d", (0.0, 0.0, 0.0)))
        extent = list(representative.get("bbox_3d", (0.1, 0.1, 0.1)))
        center_cov = list(representative.get(
            "center_cov", np.diag([4.0, 4.0, 1.0]).tolist()
        ))
        extent_cov = list(representative.get(
            "extent_cov", np.diag([1.0, 1.0, 0.25]).tolist()
        ))
        center_covariance_provenance = str(
            representative.get(
                "center_covariance_provenance", "bearing_model"
            )
        )
        extent_covariance_provenance = str(
            representative.get(
                "extent_covariance_provenance", "bearing_extent_model"
            )
        )
    class_name = str(representative.get("canonical_class", "object"))
    support_path = geometry_dir / (
        f"station_fused_{_safe_component(class_name)}_{group_index:04d}.npz"
    )
    np.savez_compressed(
        support_path,
        world_points=fused.astype(np.float32),
        lidar_support_points_map=fused.astype(np.float32),
    )

    source_records: dict[str, tuple[Any, Any]] = {}
    proposal_keys: list[str] = []
    verification_by_key: dict[str, float | None] = {}
    verification_probabilities: list[float] = []
    source_detection_ids: list[str] = []
    source_observation_ids: list[str] = []
    coverage_member_observation_ids: list[str] = []
    source_view_boxes: list[dict[str, Any]] = []
    proposal_probabilities: list[float] = []
    for member in members:
        source_observation_ids.append(str(member.get("observation_id", "")))
        source_observation_ids.extend(
            str(value) for value in member.get("source_observation_ids", ())
        )
        coverage_member_observation_ids.extend(
            str(value)
            for value in member.get("coverage_member_observation_ids", ())
        )
        for view_id, pose, intrinsics in zip(
            member.get("source_view_ids", ()),
            member.get("source_cam2w_maps", ()),
            member.get("source_intrinsics", ()),
        ):
            source_records.setdefault(str(view_id), (pose, intrinsics))
        proposal_keys.extend(str(value) for value in member.get("source_proposal_binding_keys", ()))
        verification_by_key.update({
            str(key): (float(value) if isinstance(value, (int, float)) else None)
            for key, value in dict(member.get("proposal_verification_by_key", {})).items()
        })
        verification_probabilities.extend(
            float(value)
            for value in member.get("qwen_verification_probabilities", ())
            if isinstance(value, (int, float))
        )
        source_detection_ids.extend(
            str(value) for value in member.get("source_detection_ids", ())
        )
        if not member.get("source_detection_ids"):
            source_detection_ids.extend(
                str(value) for value in member.get("source_proposal_binding_keys", ())
            )
        for raw_box in member.get("source_view_boxes", ()):
            if not isinstance(raw_box, Mapping):
                continue
            try:
                record = {
                    "view_id": str(raw_box.get("view_id", "")),
                    "bbox_xyxy": [float(value) for value in raw_box.get("bbox_xyxy", ())],
                    "image_width": int(raw_box.get("image_width", 0)),
                    "image_height": int(raw_box.get("image_height", 0)),
                }
            except (TypeError, ValueError):
                continue
            if (
                record["view_id"]
                and len(record["bbox_xyxy"]) == 4
                and record["image_width"] > 0
                and record["image_height"] > 0
            ):
                source_view_boxes.append(record)
        proposal_probabilities.append(max(
            0.0,
            min(1.0, float(member.get("semantic_probability", 0.0))),
        ))

    appearance_values = []
    appearance_weights = []
    for member in members:
        descriptor = member.get("appearance_descriptor")
        if not isinstance(descriptor, Sequence) or isinstance(descriptor, (str, bytes)):
            continue
        try:
            vector = np.asarray([float(value) for value in descriptor], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if not len(vector) or not np.isfinite(vector).all():
            continue
        quality = max(0.05, float(member.get("appearance_quality", 0.0)))
        appearance_values.append(vector)
        appearance_weights.append(quality)
    fused_appearance: list[float] = []
    fused_appearance_quality = 0.0
    if appearance_values and len({len(value) for value in appearance_values}) == 1:
        vector = np.average(
            np.stack(appearance_values, axis=0),
            axis=0,
            weights=np.asarray(appearance_weights, dtype=np.float64),
        )
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            fused_appearance = (vector / norm).astype(float).tolist()
            fused_appearance_quality = float(min(1.0, sum(appearance_weights)))

    result = dict(representative)
    result.update({
        "observation_id": (
            f"{acquisition_id}:station_fused:{_safe_component(class_name)}:{group_index:04d}"
        ),
        "center_3d": center,
        "bbox_3d": extent,
        "world_center": center,
        "world_extent": extent,
        "measured_center_3d": center,
        "measured_bbox_3d": extent,
        "center_cov": center_cov,
        "extent_cov": extent_cov,
        "center_covariance_provenance": (
            "multi_view_statistical"
            if center_covariance_provenance == "pointcloud_statistical"
            else center_covariance_provenance
        ),
        "extent_covariance_provenance": extent_covariance_provenance,
        "covariance_mode": (
            "strict"
            if center_covariance_provenance == "pointcloud_statistical"
            and len(fused) >= _MIN_STRICT_LIDAR_POINTS
            else "probabilistic"
        ),
        "pointcloud_path": str(support_path.resolve()),
        "world_points_path": str(support_path.resolve()),
        # Members are duplicate proposals for one physical instance, not
        # independent objects. Lidar mask size must not become a class-vote
        # weight; retain the strongest continuous class hypothesis and keep
        # every per-proposal score below as diagnostics.
        "semantic_probability": max(proposal_probabilities, default=0.0),
        "source_view_ids": list(source_records),
        "source_cam2w_maps": [value[0] for value in source_records.values()],
        "source_intrinsics": [value[1] for value in source_records.values()],
        "source_proposal_binding_keys": list(dict.fromkeys(proposal_keys)),
        "proposal_verification_by_key": verification_by_key,
        "qwen_verification_probabilities": verification_probabilities,
        "qwen_verified": any(value.get("qwen_verified") is True for value in members),
        "qwen_candidate_verdict": (
            "target_wins"
            if any(value.get("qwen_verified") is True for value in members)
            else str(representative.get("qwen_candidate_verdict", "unavailable"))
        ),
        "median_confidence": max(float(value.get("median_confidence", 0.0)) for value in members),
        "lidar_support_count": int(len(fused)),
        "geometry_source": "same_station_multiview_lidar_fusion",
        "station_fused_member_count": len(members),
        "source_detection_ids": list(dict.fromkeys(source_detection_ids)),
        "source_observation_ids": [
            value for value in dict.fromkeys(source_observation_ids) if value
        ],
        "coverage_member_observation_ids": [
            value
            for value in dict.fromkeys(coverage_member_observation_ids)
            if value
        ],
        "source_view_boxes": list({
            (
                str(record["view_id"]),
                tuple(round(float(value), 3) for value in record["bbox_xyxy"]),
                int(record["image_width"]),
                int(record["image_height"]),
            ): record
            for record in source_view_boxes
        }.values()),
        "appearance_descriptor": fused_appearance or list(
            representative.get("appearance_descriptor", ())
        ),
        "appearance_quality": max(
            fused_appearance_quality,
            float(representative.get("appearance_quality", 0.0)),
        ),
    })
    panorama_masks = [
        cv2.imread(str(value.get("panorama_mask_path", "")), cv2.IMREAD_GRAYSCALE)
        for value in members
        if Path(str(value.get("panorama_mask_path", ""))).is_file()
    ]
    panorama_masks = [value for value in panorama_masks if value is not None]
    if panorama_masks and len({value.shape for value in panorama_masks}) == 1:
        fused_mask = np.logical_or.reduce([value > 0 for value in panorama_masks])
        fused_mask_path = geometry_dir / (
            f"station_fused_{_safe_component(class_name)}_{group_index:04d}.panorama_mask.png"
        )
        cv2.imwrite(str(fused_mask_path), fused_mask.astype(np.uint8) * 255)
        rows, columns = np.nonzero(fused_mask)
        result.update({
            "panorama_mask_path": str(fused_mask_path.resolve()),
            "panorama_mask_coordinate_frame": MASK_PANORAMA_PIXELS,
            "panorama_pixel_support_bbox_xyxy": [
                int(columns.min()), int(rows.min()),
                int(columns.max() + 1), int(rows.max() + 1),
            ],
            "panorama_pixel_support_area": int(np.count_nonzero(fused_mask)),
        })
    return result


def _deduplicate_station_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    acquisition_id: str,
    geometry_dir: Path,
) -> list[dict[str, Any]]:
    """Fuse only complete-link, same-class, same-station 3D duplicates.

    Complete-link grouping prevents the transitive A-overlaps-B-overlaps-C chain
    that previously merged adjacent pillows, cups, or pictures into one object.
    """
    prepared = [dict(value) for value in observations]
    for candidate in prepared:
        candidate_id = str(candidate.get("observation_id", ""))
        covered: list[dict[str, Any]] = []
        for peer in prepared:
            if peer is candidate:
                continue
            if _canonical_class_label(
                candidate.get("canonical_class")
            ) != _canonical_class_label(peer.get("canonical_class")):
                continue
            if str(candidate.get("optical_center_group", "")) != str(
                peer.get("optical_center_group", "")
            ):
                continue
            overlap = _panorama_support_overlap(candidate, peer)
            if overlap is None:
                continue
            if (
                overlap["second_containment"] >= 0.75
                and overlap["first_area_px"]
                >= 1.35 * overlap["second_area_px"]
                and _metric_geometry_reliable(peer)
            ):
                covered.append(peer)
        distinct_children = any(
            _coverage_children_are_distinct(first, second)
            for index, first in enumerate(covered)
            for second in covered[index + 1 :]
        )
        if distinct_children:
            candidate["coverage_member_observation_ids"] = sorted({
                str(value.get("observation_id", ""))
                for value in covered
                if str(value.get("observation_id", ""))
                and str(value.get("observation_id", "")) != candidate_id
            })

    ordered = sorted(
        prepared,
        key=lambda value: (
            str(value.get("canonical_class", "")),
            -float(value.get("semantic_probability", 0.0)),
            str(value.get("observation_id", "")),
        ),
    )

    def build_groups() -> list[list[dict[str, Any]]]:
        result: list[list[dict[str, Any]]] = []
        for observation in ordered:
            matching_index = None
            for index, group in enumerate(result):
                if all(
                    _same_station_instance(observation, member)
                    for member in group
                ):
                    matching_index = index
                    break
            if matching_index is None:
                result.append([observation])
            else:
                result[matching_index].append(observation)
        return result

    groups = build_groups()
    raw_group_index = {
        str(value.get("observation_id", "")): index
        for index, group in enumerate(groups)
        for value in group
    }
    coverage_changed = False
    for value in ordered:
        value_id = str(value.get("observation_id", ""))
        covered_groups = {
            raw_group_index[str(item)]
            for item in value.get("coverage_member_observation_ids", ())
            if str(item) in raw_group_index
        }
        candidate_group = raw_group_index.get(value_id)
        if candidate_group is not None:
            covered_groups.discard(candidate_group)
        if value.get("coverage_member_observation_ids") and len(covered_groups) < 2:
            value.pop("coverage_member_observation_ids", None)
            coverage_changed = True
    if coverage_changed:
        groups = build_groups()
    fused = [
        _merge_station_group(
            group,
            acquisition_id=acquisition_id,
            group_index=index,
            geometry_dir=geometry_dir,
        )
        for index, group in enumerate(groups, start=1)
    ]
    raw_to_fused: dict[str, str] = {}
    for value in fused:
        fused_id = str(value.get("observation_id", ""))
        source_ids = {
            fused_id,
            *(str(item) for item in value.get("source_observation_ids", ())),
        }
        for source_id in source_ids:
            if source_id:
                raw_to_fused[source_id] = fused_id
    for value in fused:
        fused_id = str(value.get("observation_id", ""))
        mapped = {
            raw_to_fused.get(str(item), str(item))
            for item in value.get("coverage_member_observation_ids", ())
        }
        value["coverage_member_observation_ids"] = sorted(
            item for item in mapped if item and item != fused_id
        )
    for index, first in enumerate(fused):
        for second in fused[index + 1 :]:
            if _canonical_class_label(first.get("canonical_class")) != _canonical_class_label(second.get("canonical_class")):
                continue
            overlap = _panorama_support_overlap(first, second)
            if overlap is None or overlap["containment_iom"] > 0.01:
                continue
            if not (_metric_geometry_reliable(first) and _metric_geometry_reliable(second)):
                continue
            distance = float(np.linalg.norm(
                np.asarray(first.get("center_3d"), dtype=np.float64)
                - np.asarray(second.get("center_3d"), dtype=np.float64)
            ))
            core_scale = 0.25 * min(
                float(np.linalg.norm(np.asarray(first.get("bbox_3d"), dtype=np.float64))),
                float(np.linalg.norm(np.asarray(second.get("bbox_3d"), dtype=np.float64))),
            )
            if _aabb_iou_3d(first, second) > 0.0 or distance <= max(0.12, core_scale):
                continue
            for subject, other in ((first, second), (second, first)):
                links = list(subject.get("cannot_link_observation_ids", ()))
                links.append(str(other.get("observation_id", "")))
                subject["cannot_link_observation_ids"] = sorted(set(links))
    return fused

def lift_detections_to_observations(
    *,
    detections: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, Any]],
    competition_geometry: Mapping[str, Any],
    station_id: str,
    acquisition_id: str,
    output_dir: Path,
    calibration_path: Path,
    minimum_support_points: int = _MIN_STRICT_LIDAR_POINTS,
    timestamp_unix: float | None = None,
) -> dict[str, Any]:
    """Lift all mask-backed detections and return SceneMemory observations."""
    sensor_points = _load_points(competition_geometry.get("sensor_scan_path"))
    registered_map = _load_points(competition_geometry.get("registered_scan_path"))
    map_from_sensor = _load_map_from_sensor(competition_geometry.get("state_estimation_path"))
    sensor_to_panorama = _load_sensor_to_panorama_physical(calibration_path)
    registered_sensor = _sensor_points(registered_map, map_from_sensor) if len(registered_map) else np.zeros((0, 3))
    view_lookup = {str(value["view_id"]): dict(value) for value in views}
    view_image_cache: dict[str, np.ndarray | None] = {}
    projection_cache: dict[str, dict[str, Any]] = {}
    viewpoint = map_from_sensor[:3, 3].astype(float).tolist()

    geometry_dir = output_dir / "05_lidar_geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    observations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for raw in detections:
        detection = dict(raw)
        detection_id = str(detection.get("detection_id", "unknown"))
        view_id = str(detection.get("view_id", ""))
        view = view_lookup.get(view_id)
        mask_path = Path(str(detection.get("mask_path", "")))

        if view is None:
            rejected.append({
                "detection_id": detection_id,
                "class": detection.get("canonical_class"),
                "reason": "view_missing",
            })
            continue

        if not mask_path.is_file():
            rejected.append({
                "detection_id": detection_id,
                "class": detection.get("canonical_class"),
                "reason": "mask_missing",
            })
            continue

        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        except Exception:
            rejected.append({
                "detection_id": detection_id,
                "class": detection.get("canonical_class"),
                "reason": "mask_unreadable",
            })
            continue

        if mask is None:
            rejected.append({
                "detection_id": detection_id,
                "class": detection.get("canonical_class"),
                "reason": "mask_unreadable",
            })
            continue

        # Normalize mask to single channel
        if mask.ndim == 3:
            mask = mask[..., 0]

        try:
            k = np.asarray(view.get("K"), dtype=np.float64)
            rotation = np.asarray(view.get("R_panorama_from_camera"), dtype=np.float64)
        except (TypeError, ValueError):
            rejected.append({
                "detection_id": detection_id,
                "class": detection.get("canonical_class"),
                "reason": "view_calibration_invalid",
            })
            continue

        if k.shape != (3, 3) or rotation.shape != (3, 3):
            rejected.append({
                "detection_id": detection_id,
                "class": detection.get("canonical_class"),
                "reason": "view_calibration_invalid",
            })
            continue

        width, height = int(view["width"]), int(view["height"])

        # The detector/SAM2 mask owns the generated perspective-view frame.
        # A different raster shape is a producer contract failure, not a
        # reason to resize the mask into an unrelated coordinate system.
        mask_coordinate_frame = str(
            detection.get("mask_coordinate_frame", MASK_PERSPECTIVE_PIXELS)
        ).strip()
        if mask_coordinate_frame != MASK_PERSPECTIVE_PIXELS:
            rejected.append({
                "detection_id": detection_id,
                "class": detection.get("canonical_class"),
                "status": "INVALID",
                "reason": "mask_coordinate_frame_mismatch",
                "expected_frame": MASK_PERSPECTIVE_PIXELS,
                "actual_frame": mask_coordinate_frame,
            })
            continue
        try:
            validate_mask_contract(
                mask_coordinate_frame=mask_coordinate_frame,
                mask_shape=mask.shape[:2],
                native_image_width=width,
                native_image_height=height,
            )
        except ValueError as exc:
            rejected.append({
                "detection_id": detection_id,
                "class": detection.get("canonical_class"),
                "status": "INVALID",
                "reason": str(exc),
                "expected": (height, width),
                "actual": tuple(int(value) for value in mask.shape[:2]),
            })
            continue

        cached_projection = projection_cache.get(view_id)
        if cached_projection is None:
            current_projection = _project_sensor_points(
                sensor_points,
                sensor_to_panorama_physical=sensor_to_panorama,
                rotation_panorama_from_view=rotation,
                intrinsics=k,
                width=width,
                height=height,
            )
            registered_projection = _project_sensor_points(
                registered_sensor,
                sensor_to_panorama_physical=sensor_to_panorama,
                rotation_panorama_from_view=rotation,
                intrinsics=k,
                width=width,
                height=height,
            ) if len(registered_sensor) else (
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 2), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
            map_from_view = np.asarray(
                _camera_pose_map(
                    map_from_sensor=map_from_sensor,
                    sensor_to_panorama_physical=sensor_to_panorama,
                    rotation_panorama_from_view=rotation,
                ),
                dtype=np.float64,
            )
            cached_projection = {
                "current": current_projection,
                "registered": registered_projection,
                "map_from_view": map_from_view,
            }
            projection_cache[view_id] = cached_projection

        current_indices, current_pixels, current_depths = cached_projection["current"]
        current_inside = _mask_membership(mask, current_pixels)
        selected_sensor = sensor_points[current_indices[current_inside]]
        selected_depth = current_depths[current_inside]
        source = "sensor_scan"

        if len(selected_sensor) < minimum_support_points and len(registered_sensor):
            reg_indices, reg_pixels, reg_depths = cached_projection["registered"]
            reg_inside = _mask_membership(mask, reg_pixels)
            selected_registered = registered_sensor[reg_indices[reg_inside]]
            selected_registered_depth = reg_depths[reg_inside]
            if len(selected_registered) > len(selected_sensor):
                selected_sensor = selected_registered
                selected_depth = selected_registered_depth
                source = "registered_scan_reprojection"

        foreground_sensor, foreground_depth = _select_foreground_depth(selected_sensor, selected_depth)
        foreground_map = _map_points(foreground_sensor, map_from_sensor)
        raw_foreground_sensor = foreground_sensor
        raw_foreground_map = foreground_map
        raw_lidar_support_count = int(len(raw_foreground_map))
        sparse_range_support_count = 0
        bearing_observation: dict[str, Any] | None = None
        if len(foreground_map) >= _MIN_STRICT_LIDAR_POINTS:
            (
                center,
                extent,
                center_cov,
                extent_cov,
                center_covariance_provenance,
                extent_covariance_provenance,
            ) = _robust_geometry(foreground_map)
        else:
            # Zero-to-five points are useful range hints, not a statistically
            # supported object box.  In particular, three collinear returns
            # through foliage are commonly the wall behind the object.
            bbox_for_bearing = detection.get("bbox_xyxy", (0.0, 0.0, 1.0, 1.0))
            bearing_range = 3.0
            current_points = sensor_points[current_indices]
            local_range = _sparse_local_range(
                mask=mask,
                points_sensor=current_points,
                projected_pixels=current_pixels,
                projected_depths=current_depths,
                map_from_sensor=map_from_sensor,
                map_from_view=cached_projection["map_from_view"],
            )
            if local_range is None and len(registered_sensor):
                reg_indices, reg_pixels, reg_depths = cached_projection["registered"]
                local_range = _sparse_local_range(
                    mask=mask,
                    points_sensor=registered_sensor[reg_indices],
                    projected_pixels=reg_pixels,
                    projected_depths=reg_depths,
                    map_from_sensor=map_from_sensor,
                    map_from_view=cached_projection["map_from_view"],
                )
            if local_range is not None:
                bearing_range, sparse_range_support_count = local_range
            elif len(raw_foreground_map):
                camera = np.asarray(
                    cached_projection["map_from_view"][:3, 3],
                    dtype=np.float64,
                )
                ranges = np.linalg.norm(
                    raw_foreground_map - camera[None, :], axis=1
                )
                finite_ranges = ranges[np.isfinite(ranges) & (ranges > 0.05)]
                if len(finite_ranges):
                    bearing_range = float(np.median(finite_ranges))
            (
                center,
                extent,
                center_cov,
                extent_cov,
                bearing_observation,
            ) = _bearing_geometry(
                bbox=bbox_for_bearing,
                intrinsics=k,
                map_from_view=cached_projection["map_from_view"],
                range_m=bearing_range,
            )
            center_covariance_provenance = "bearing_model"
            extent_covariance_provenance = "bearing_extent_model"
            bearing_observation.update({
                "source": "mask_bearing_sparse_lidar",
                "raw_lidar_support_count": raw_lidar_support_count,
                "local_range_support_count": sparse_range_support_count,
            })
            source = (
                "bearing_only"
                if raw_lidar_support_count == 0
                else f"{source}_sparse_bearing"
            )
            # Do not let rejected sparse/background points become strict later
            # merely because SceneMemory accumulated three of them.
            foreground_sensor = np.zeros((0, 3), dtype=np.float64)
            foreground_depth = np.zeros(0, dtype=np.float64)
            foreground_map = np.zeros((0, 3), dtype=np.float64)

        # Compute geometry confidence from center covariance trace
        center_cov_array = np.asarray(center_cov, dtype=np.float64)
        if center_cov_array.shape == (3, 3) and np.isfinite(center_cov_array).all():
            trace = float(np.trace(center_cov_array))
            geometry_confidence = 1.0 / (1.0 + trace) if trace >= 0.0 else 0.0
        else:
            geometry_confidence = 0.0

        support_path = geometry_dir / f"{detection_id}.npz"
        np.savez_compressed(
            support_path,
            world_points=foreground_map.astype(np.float32),
            lidar_support_points_map=foreground_map.astype(np.float32),
            points_sensor=foreground_sensor.astype(np.float32),
            raw_mask_points_map=raw_foreground_map.astype(np.float32),
            raw_mask_points_sensor=raw_foreground_sensor.astype(np.float32),
        )

        probability = max(0.0, min(1.0, float(detection.get("semantic_probability", 0.55))))
        bbox = [float(value) for value in detection.get("bbox_xyxy", (0, 0, 0, 0))]

        # Clip and validate bbox
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width, int(np.floor(x1))))
        y1 = max(0, min(height, int(np.floor(y1))))
        x2 = max(0, min(width, int(np.ceil(x2))))
        y2 = max(0, min(height, int(np.ceil(y2))))

        if x2 <= x1 or y2 <= y1:
            rejected.append({
                "detection_id": detection_id,
                "class": detection.get("canonical_class"),
                "reason": "bbox_invalid",
                "bbox": bbox,
            })
            continue

        bbox = [float(x1), float(y1), float(x2), float(y2)]

        # Appearance is optional - failures must not block geometry
        appearance_descriptor: list[float] = []
        appearance_quality = 0.0
        if view_id not in view_image_cache:
            try:
                view_image_cache[view_id] = cv2.imread(
                    str(view.get("image_path", "")), cv2.IMREAD_COLOR
                )
            except Exception:
                view_image_cache[view_id] = None

        if view_image_cache[view_id] is not None:
            try:
                appearance_descriptor, appearance_quality = _appearance_descriptor(
                    view_image_cache[view_id], mask, bbox
                )
            except Exception:
                # Appearance failure is not fatal
                appearance_descriptor = []
                appearance_quality = 0.0

        registered_median, registered_inlier = _registered_support_metrics(
            foreground_map, registered_map
        )
        observation_id = f"{acquisition_id}:{view_id}:{detection_id}"
        camera_pose_map = _camera_pose_map(
            map_from_sensor=map_from_sensor,
            sensor_to_panorama_physical=sensor_to_panorama,
            rotation_panorama_from_view=rotation,
        )
        panorama_path = _find_camera_panorama(view.get("image_path"))
        panorama_mask_path = ""
        panorama_projection: dict[str, Any] = {
            "status": "unavailable",
            "source_mask_coordinate_frame": mask_coordinate_frame,
        }
        panorama_support_bbox: list[int] = []
        panorama_support_area = 0
        if panorama_path is not None:
            try:
                panorama_image = cv2.imread(
                    str(panorama_path), cv2.IMREAD_COLOR
                )
                map_x_path = Path(str(view.get("map_x_path", "")))
                map_y_path = Path(str(view.get("map_y_path", "")))
                map_x = np.load(map_x_path, allow_pickle=False)
                map_y = np.load(map_y_path, allow_pickle=False)
                if (
                    panorama_image is None
                    or map_x.shape != mask.shape
                    or map_y.shape != mask.shape
                ):
                    raise ValueError("perspective_remap_shape_mismatch")
                panorama_mask = project_mask_to_panorama(
                    mask,
                    map_x,
                    map_y,
                    panorama_image.shape[:2],
                )
                panorama_mask_file = geometry_dir / (
                    f"{detection_id}.panorama_mask.png"
                )
                if not cv2.imwrite(str(panorama_mask_file), panorama_mask):
                    raise ValueError("panorama_mask_write_failed")
                panorama_mask_path = str(panorama_mask_file.resolve())
                support_rows, support_columns = np.nonzero(panorama_mask > 0)
                panorama_support_bbox = [
                    int(support_columns.min()), int(support_rows.min()),
                    int(support_columns.max() + 1), int(support_rows.max() + 1),
                ]
                panorama_support_area = int(np.count_nonzero(panorama_mask))
                panorama_projection = {
                    "status": "completed",
                    "source_mask_coordinate_frame": mask_coordinate_frame,
                    "target_mask_coordinate_frame": MASK_PANORAMA_PIXELS,
                    "map_x_path": str(map_x_path.resolve()),
                    "map_y_path": str(map_y_path.resolve()),
                }
            except (OSError, ValueError, TypeError):
                # The world-space observation remains valid.  Panorama pixels
                # are optional visual provenance; they are not allowed to
                # overwrite or resize the native perspective mask.
                panorama_projection = {
                    "status": "unavailable",
                    "source_mask_coordinate_frame": mask_coordinate_frame,
                    "reason": "panorama_mask_projection_unavailable",
                }
        observation_timestamp = (
            float(timestamp_unix)
            if timestamp_unix is not None and math.isfinite(float(timestamp_unix))
            else time.time()
        )
        observation = {
            "observation_id": observation_id,
            "station_id": str(station_id),
            "acquisition_id": str(acquisition_id),
            "timestamp": observation_timestamp,
            "timestamp_unix": observation_timestamp,
            "contract_schema_version": "observation_contract_v1",
            "view_id": view_id,
            "camera_id": view_id,
            "native_image_width": width,
            "native_image_height": height,
            "mask_native_width": int(mask.shape[1]),
            "mask_native_height": int(mask.shape[0]),
            "camera_model": {
                "projection": "pinhole",
                "K": k.astype(float).tolist(),
                "R_panorama_from_camera": rotation.astype(float).tolist(),
                "horizontal_fov_deg": float(view.get("horizontal_fov_deg", 90.0)),
                "source_projection": "central_cropped_equirectangular",
            },
            "mask_path": str(mask_path.resolve()),
            "mask_coordinate_frame": mask_coordinate_frame,
            "bbox_xyxy": list(bbox),
            "panorama_mask_path": panorama_mask_path,
            "panorama_mask_coordinate_frame": (
                MASK_PANORAMA_PIXELS if panorama_mask_path else ""
            ),
            "panorama_image_path": str(panorama_path or ""),
            "panorama_mask_projection": panorama_projection,
            "panorama_pixel_support_bbox_xyxy": panorama_support_bbox,
            "panorama_pixel_support_area": panorama_support_area,
            "canonical_class": _canonical_class_label(detection.get("canonical_class", "")),
            "observed_class_label": str(detection.get("canonical_class", "")).strip().lower(),
            "semantic_label": _canonical_class_label(detection.get("canonical_class", "")),
            "semantic_confidence": probability,
            "frame": "map",
            "world_aligned": True,
            "center_3d": center,
            "bbox_3d": extent,
            "world_center": center,
            "world_extent": extent,
            "world_points_path": str(support_path.resolve()),
            "measured_center_3d": center,
            "measured_bbox_3d": extent,
            "center_cov": center_cov,
            "extent_cov": extent_cov,
            "center_covariance_provenance": center_covariance_provenance,
            "extent_covariance_provenance": extent_covariance_provenance,
            "covariance_mode": (
                "strict"
                if center_covariance_provenance == "pointcloud_statistical"
                and len(foreground_map) >= _MIN_STRICT_LIDAR_POINTS
                else "probabilistic"
            ),
            "viewpoint_position_map": viewpoint,
            "optical_center_group": str(station_id),
            "semantic_probability": probability,
            "pointcloud_path": str(support_path.resolve()),
            "representative_view_id": view_id,
            "representative_view_image": str(view["image_path"]),
            "representative_bbox_xyxy": bbox,
            "source_cam2w_maps": [camera_pose_map],
            "T_world_camera": camera_pose_map,
            "source_intrinsics": [k.astype(float).tolist()],
            "source_view_ids": [view_id],
            "source_proposal_binding_keys": [detection_id],
            "source_detection_ids": [detection_id],
            "source_view_boxes": [{
                "view_id": view_id,
                "bbox_xyxy": list(bbox),
                "image_width": int(mask.shape[1]),
                "image_height": int(mask.shape[0]),
            }],
            "proposal_verification_by_key": {detection_id: probability},
            "qwen_verification_probabilities": [probability],
            "qwen_verified": bool(detection.get("qwen_verified", False)),
            "qwen_candidate_verdict": str(detection.get("qwen_candidate_verdict", "unavailable")),
            "median_confidence": min(1.0, probability * min(1.0, len(foreground_map) / 20.0)),
            "view_center_dispersion_m": 0.0,
            "registered_scan_median_distance_m": registered_median,
            "registered_scan_inlier_fraction_0_25m": registered_inlier,
            "appearance_descriptor": appearance_descriptor,
            "appearance_feature": appearance_descriptor,
            "appearance_quality": appearance_quality,
            "mask_pixel_count": int(np.asarray(mask > 0, dtype=np.uint8).sum()),
            "geometry_source": source,
            "lidar_support_count": int(len(foreground_map)),
            "raw_lidar_support_count": raw_lidar_support_count,
            "sparse_range_support_count": sparse_range_support_count,
            "depth_median": (
                float(np.median(foreground_depth))
                if len(foreground_depth)
                else float((bearing_observation or {}).get("range_mean_m", 3.0))
            ),
            "bearing_observation": bearing_observation,
            "depth_support": {
                "raw_lidar_support_count": raw_lidar_support_count,
                "foreground_lidar_support_count": int(len(foreground_map)),
                "sparse_range_support_count": sparse_range_support_count,
                "depth_median_m": (
                    float(np.median(foreground_depth))
                    if len(foreground_depth)
                    else float((bearing_observation or {}).get("range_mean_m", 3.0))
                ),
            },
            "canonical_object_id": None,
            "identity_hypotheses": [],
        }
        contract = ObservationContract.from_mapping(observation)
        observation.update(contract.to_dict())
        observations.append(observation)

    raw_observation_count = len(observations)
    observations = _deduplicate_station_observations(
        observations,
        acquisition_id=acquisition_id,
        geometry_dir=geometry_dir,
    )
    manifest = {
        "schema_version": "lidar_first_geometry_v3",
        "frame": "map",
        "acquisition_id": acquisition_id,
        "station_id": station_id,
        "raw_observation_count": raw_observation_count,
        "observation_count": len(observations),
        "same_station_duplicates_removed": raw_observation_count - len(observations),
        "rejected_count": len(rejected),
        "observations": observations,
        "rejected": rejected,
    }
    (geometry_dir / "observations.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
