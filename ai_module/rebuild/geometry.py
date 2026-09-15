"""Sensor anchored geometry for the rebuilt AI module.

The rebuilt chain has one source of metric scale: the registered scan and the
pose measured at image time.  This module deliberately contains only numpy
and a small amount of image remapping code.  It does not import a model or a
legacy perception pipeline.

Transforms use column vectors and ``T_destination_source`` naming.  A
``View`` is a virtual optical camera looking into the cropped equirectangular
panorama.  ``View.map_x`` and ``View.map_y`` map view pixels back to the
original panorama pixels.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import Detection, Frame, Observation, View


# The calibration file describes the camera in a physical ROS-like basis:
# x-forward, y-right, z-up.  The panorama/pinhole camera basis is optical:
# x-right, y-down, z-forward.
_PHYSICAL_TO_OPTICAL = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

_DEFAULT_VIEW_WIDTH = 768
_DEFAULT_VIEW_HEIGHT = 576
_DEFAULT_VIEW_YAWS = (0.0, 90.0, 180.0, 270.0)
_DEFAULT_HORIZONTAL_FOV = 105.0
_DEFAULT_PANORAMA_VERTICAL_FOV = 120.0
_MIN_DEPTH_M = 0.05
_MAX_DEPTH_M = 100.0
_MAX_ESTIMATED_PIXELS = 8192


def _as_transform(value: object, name: str) -> np.ndarray:
    """Validate and copy a homogeneous transform."""

    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}_invalid") from exc
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name}_invalid")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name}_not_homogeneous")
    return transform


def _source_to_camera_optical(calibration: Mapping[str, Any]) -> np.ndarray:
    """Return ``T_sensor_camera`` for the optical panorama camera.

    ``source_to_camera`` in the checked-in calibration is the physical camera
    transform (camera physical <- sensor).  Prefixing the physical-to-optical
    basis change gives camera optical <- sensor; inverting that transform is
    the requested sensor <- camera optical transform.
    """

    if not isinstance(calibration, Mapping):
        raise ValueError("calibration_invalid")
    source_to_camera = calibration.get("source_to_camera")
    if source_to_camera is None:
        # This spelling is useful for callers that have already unpacked the
        # same file, while the checked-in source remains authoritative.
        source_to_camera = calibration.get("T_camera_sensor_physical")
    if source_to_camera is None:
        raise ValueError("calibration_source_to_camera_missing")
    physical = _as_transform(source_to_camera, "calibration_source_to_camera")
    optical = np.eye(4, dtype=np.float64)
    optical[:3, :3] = _PHYSICAL_TO_OPTICAL
    camera_optical_from_sensor = optical @ physical
    try:
        sensor_from_camera = np.linalg.inv(camera_optical_from_sensor)
    except np.linalg.LinAlgError as exc:
        raise ValueError("calibration_source_to_camera_singular") from exc
    if not np.isfinite(sensor_from_camera).all():
        raise ValueError("calibration_source_to_camera_invalid_inverse")
    return sensor_from_camera


def _rotation_panorama_from_view(yaw_rad: float, pitch_rad: float = 0.0) -> np.ndarray:
    """Return optical view -> panorama optical rotation.

    Positive yaw turns the view toward increasing panorama longitude.  A
    positive pitch looks upward, which decreases optical image ``y``.
    """

    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    yaw = np.asarray(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64
    )
    pitch = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64
    )
    return yaw @ pitch


def _view_parameters(config: Mapping[str, Any] | None) -> tuple[int, int, list[float], float, float]:
    config = config if isinstance(config, Mapping) else {}
    try:
        width = int(config.get("view_width", _DEFAULT_VIEW_WIDTH))
        height = int(config.get("view_height", _DEFAULT_VIEW_HEIGHT))
        horizontal_fov = float(
            config.get("horizontal_fov_deg", _DEFAULT_HORIZONTAL_FOV)
        )
        vertical_fov = float(
            config.get(
                "panorama_vertical_fov_deg", _DEFAULT_PANORAMA_VERTICAL_FOV
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("view_config_invalid") from exc
    raw_yaws = config.get("view_yaws_deg", _DEFAULT_VIEW_YAWS)
    if isinstance(raw_yaws, (str, bytes)):
        raise ValueError("view_yaws_invalid")
    try:
        yaws = [float(value) for value in raw_yaws]
    except (TypeError, ValueError) as exc:
        raise ValueError("view_yaws_invalid") from exc
    if width <= 1 or height <= 1:
        raise ValueError("view_size_invalid")
    if not yaws or not np.isfinite(yaws).all():
        raise ValueError("view_yaws_invalid")
    if not 0.0 < horizontal_fov < 180.0:
        raise ValueError("horizontal_fov_invalid")
    if not 0.0 < vertical_fov <= 180.0:
        raise ValueError("panorama_vertical_fov_invalid")
    return width, height, yaws, horizontal_fov, vertical_fov


def _project_equirectangular(
    panorama_rgb: np.ndarray,
    *,
    yaw_deg: float,
    width: int,
    height: int,
    horizontal_fov_deg: float,
    panorama_vertical_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project one perspective view and return image, K, and source maps."""

    pano_height, pano_width = panorama_rgb.shape[:2]
    focal = 0.5 * float(width) / math.tan(math.radians(horizontal_fov_deg) * 0.5)
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    intrinsics = np.asarray(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    rays = np.stack(((u - cx) / focal, (v - cy) / focal, np.ones_like(u)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    rotation = _rotation_panorama_from_view(math.radians(yaw_deg))
    panorama_rays = rays @ rotation.T
    longitude = np.arctan2(panorama_rays[..., 0], panorama_rays[..., 2])
    latitude = np.arcsin(np.clip(-panorama_rays[..., 1], -1.0, 1.0))
    map_x = np.mod(
        (longitude / (2.0 * math.pi) + 0.5) * float(pano_width) - 0.5,
        float(pano_width),
    ).astype(np.float32)
    vertical_fov = math.radians(panorama_vertical_fov_deg)
    map_y = (
        (0.5 - latitude / vertical_fov) * float(pano_height) - 0.5
    ).astype(np.float32)
    tolerance = 1e-3
    if float(map_y.min()) < -0.5 - tolerance or float(map_y.max()) > (
        float(pano_height) - 0.5 + tolerance
    ):
        raise ValueError("perspective_view_exceeds_cropped_panorama_vertical_fov")
    image = _remap_image(panorama_rgb, map_x, map_y)
    return image, intrinsics, rotation, map_x, map_y


def _remap_image(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Bilinearly sample an equirectangular image with horizontal wrapping."""

    height, width = image.shape[:2]
    x = np.mod(np.asarray(map_x, dtype=np.float64), float(width))
    y = np.clip(np.asarray(map_y, dtype=np.float64), 0.0, float(height - 1))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = (x0 + 1) % width
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]
    source = np.asarray(image)
    if source.ndim == 2:
        source = source[..., None]
        squeeze = True
    else:
        squeeze = False
    top = source[y0, x0] * (1.0 - wx) + source[y0, x1] * wx
    bottom = source[y1, x0] * (1.0 - wx) + source[y1, x1] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    if np.issubdtype(source.dtype, np.integer):
        sampled = np.rint(sampled).clip(
            np.iinfo(source.dtype).min, np.iinfo(source.dtype).max
        ).astype(source.dtype)
    else:
        sampled = sampled.astype(source.dtype, copy=False)
    return sampled[..., 0] if squeeze else sampled


def _project_mask_to_panorama(
    mask: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    panorama_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize a perspective mask in original panorama pixels.

    The dense inverse map is sufficient for the normal masks.  When OpenCV is
    available, contour filling closes the small holes introduced by sampling a
    curved mask and handles a longitude seam.  A numpy scatter path remains
    available for the CPU-only/minimal environment.
    """

    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = np.any(mask_array > 0, axis=-1)
    if mask_array.ndim != 2:
        raise ValueError("mask_must_be_2d")
    if mask_array.shape != np.asarray(map_x).shape or mask_array.shape != np.asarray(map_y).shape:
        raise ValueError("mask_and_remap_shape_mismatch")
    pano_height, pano_width = [int(value) for value in panorama_shape]
    if pano_height <= 0 or pano_width <= 0:
        raise ValueError("panorama_shape_invalid")
    result = np.zeros((pano_height, pano_width), dtype=np.uint8)
    binary = np.asarray(mask_array > 0, dtype=np.uint8)
    if not binary.any():
        return result

    # Fill contours using the same coordinates as the remap.  This is an
    # optional acceleration/quality aid; the scatter below is authoritative.
    try:
        import cv2  # type: ignore
    except ImportError:  # pragma: no cover - exercised only in minimal images
        cv2 = None
    if cv2 is not None:
        contours, _hierarchy = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        for contour in contours:
            if len(contour) < 3:
                continue
            coordinates = contour[:, 0, :]
            source_x = coordinates[:, 0]
            source_y = coordinates[:, 1]
            longitude = (
                np.asarray(map_x)[source_y, source_x] / float(pano_width)
            ) * (2.0 * math.pi)
            unwrapped_x = np.unwrap(longitude) / (2.0 * math.pi) * pano_width
            projected_y = np.clip(
                np.asarray(map_y)[source_y, source_x], 0.0, pano_height - 1.0
            )
            for shift in range(-2, 3):
                shifted_x = unwrapped_x + shift * pano_width
                if shifted_x.max() < 0.0 or shifted_x.min() >= pano_width:
                    continue
                polygon = np.rint(
                    np.stack((shifted_x, projected_y), axis=1)
                ).astype(np.int32)
                cv2.fillPoly(result, [polygon], 1)

    rows, columns = np.nonzero(binary)
    pano_x = np.rint(np.asarray(map_x)[rows, columns]).astype(np.int64) % pano_width
    pano_y_float = np.asarray(map_y)[rows, columns]
    valid_y = np.isfinite(pano_y_float)
    pano_y = np.clip(np.rint(pano_y_float[valid_y]), 0, pano_height - 1).astype(np.int64)
    result[pano_y, pano_x[valid_y]] = 1
    return result


def make_views(
    frame: Frame, calibration: dict, config: dict
) -> list[View]:
    """Create calibrated overlapping perspective views from one panorama.

    ``frame.T_map_sensor`` is the pose at ``frame.stamp``.  The returned view
    poses therefore keep the image-time pose even if the caller receives a
    newer robot pose while inference is running.
    """

    if not isinstance(frame, Frame):
        raise TypeError("frame_must_be_Frame")
    panorama = np.asarray(frame.panorama_rgb)
    if panorama.ndim != 3 or panorama.shape[2] != 3:
        raise ValueError("panorama_must_be_hwc_rgb")
    if panorama.shape[0] < 2 or panorama.shape[1] < 2:
        raise ValueError("panorama_shape_invalid")
    map_from_sensor = _as_transform(frame.T_map_sensor, "T_map_sensor")
    sensor_from_view = _source_to_camera_optical(calibration)
    map_from_panorama = map_from_sensor @ sensor_from_view
    width, height, yaws, horizontal_fov, vertical_fov = _view_parameters(config)
    views: list[View] = []
    for index, yaw_deg in enumerate(yaws):
        image, intrinsics, rotation, map_x, map_y = _project_equirectangular(
            panorama,
            yaw_deg=yaw_deg,
            width=width,
            height=height,
            horizontal_fov_deg=horizontal_fov,
            panorama_vertical_fov_deg=vertical_fov,
        )
        view_from_panorama = np.eye(4, dtype=np.float64)
        view_from_panorama[:3, :3] = rotation
        map_from_view = map_from_panorama @ view_from_panorama
        views.append(
            View(
                observation_id=str(frame.observation_id),
                stamp=float(frame.stamp),
                view_id=f"view_{index:02d}",
                image_rgb=np.ascontiguousarray(image),
                intrinsics=intrinsics,
                T_map_view=map_from_view,
                map_x=map_x,
                map_y=map_y,
                panorama_shape=(int(panorama.shape[0]), int(panorama.shape[1])),
            )
        )
    return views


def sample_view_region(frame: Frame, view: View, calibration: dict,
                       bounds: tuple[int, int, int, int], output_size: tuple[int, int],
                       panorama_vertical_fov_deg: float) -> np.ndarray:
    """Sample a perspective region directly from its source panorama.

    Horizontal bounds may extend beyond an artificial perspective edge. This
    preserves the context that exists in the same camera image instead of
    truncating a physical object at one of the four view boundaries.
    """
    left, top, right, bottom = bounds
    width, height = output_size
    u, v = np.meshgrid(left + (np.arange(width) + 0.5) * (right-left) / width,
                       top + (np.arange(height) + 0.5) * (bottom-top) / height)
    K = np.asarray(view.intrinsics, dtype=np.float64)
    rays = np.stack(((u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], np.ones_like(u)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    map_from_panorama = frame.T_map_sensor @ _source_to_camera_optical(calibration)
    panorama_from_view = map_from_panorama[:3, :3].T @ view.T_map_view[:3, :3]
    rays = rays @ panorama_from_view.T
    longitude = np.arctan2(rays[..., 0], rays[..., 2])
    latitude = np.arcsin(np.clip(-rays[..., 1], -1.0, 1.0))
    pano_height, pano_width = frame.panorama_rgb.shape[:2]
    map_x = ((longitude/(2*math.pi)+0.5)*pano_width-0.5) % pano_width
    map_y = (0.5-latitude/math.radians(panorama_vertical_fov_deg))*pano_height-0.5
    return _remap_image(frame.panorama_rgb, map_x, map_y)


def _valid_points(points: object) -> np.ndarray:
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        return np.zeros((0, 3), dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points_must_be_nx3")
    return array[np.isfinite(array).all(axis=1)]


def _project_map_points(
    points_map: np.ndarray,
    view: View,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project already-map-frame points to a view and expose z-buffer data."""

    if len(points_map) == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
        )
    transform = _as_transform(view.T_map_view, "T_map_view")
    intrinsics = np.asarray(view.intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("view_intrinsics_invalid")
    width = int(view.image_rgb.shape[1])
    height = int(view.image_rgb.shape[0])
    # Registered points are already in map.  This is the only operation that
    # moves them into the camera view: inverse(T_map_view).
    view_points = (points_map - transform[:3, 3]) @ transform[:3, :3]
    depth = view_points[:, 2]
    valid = np.isfinite(view_points).all(axis=1) & (depth > _MIN_DEPTH_M)
    source_indices = np.flatnonzero(valid)
    if not len(source_indices):
        return (
            source_indices,
            np.zeros((0, 2), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
        )
    camera = view_points[source_indices]
    pixels_h = camera @ intrinsics.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= height - 1)
    )
    source_indices = source_indices[inside]
    pixels = pixels[inside]
    depths = camera[inside, 2]
    pixel_x = np.rint(pixels[:, 0]).astype(np.int64)
    pixel_y = np.rint(pixels[:, 1]).astype(np.int64)
    pixel_index = pixel_y * width + pixel_x
    # A registered return may occur more than once in the same raster pixel
    # (fixed-size scan padding is one concrete source of this).  Keep the
    # nearest source return once per projected pixel, with source index as the
    # deterministic tie-break.  Both detection lifting and view calibration
    # consume this same visibility representation.
    nearest = _nearest_projected_indices(pixel_index, depths, source_indices)
    return (
        source_indices[nearest],
        pixels[nearest],
        depths[nearest],
        pixel_index[nearest],
    )


def _nearest_projected_indices(
    pixel_indices: np.ndarray,
    depths: np.ndarray,
    source_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return one nearest source index for each raster pixel.

    The source index only breaks equal-depth ties.  It is deliberately kept
    separate from the point coordinates so this helper applies equally to
    measured lifting and shared-view depth calibration without inventing a
    world-coordinate validity rule.
    """

    pixels = np.asarray(pixel_indices, dtype=np.int64).reshape(-1)
    values = np.asarray(depths, dtype=np.float64).reshape(-1)
    if pixels.shape != values.shape:
        raise ValueError("projected_pixel_depth_shape_mismatch")
    if not len(pixels):
        return np.zeros(0, dtype=np.int64)
    if source_indices is None:
        source = np.arange(len(pixels), dtype=np.int64)
    else:
        source = np.asarray(source_indices, dtype=np.int64).reshape(-1)
        if source.shape != pixels.shape:
            raise ValueError("projected_source_index_shape_mismatch")
    # ``pixel`` is primary, then depth, then the original source index.  The
    # latter makes equal-depth selection deterministic across repeated points.
    order = np.lexsort((source, values, pixels))
    _, first = np.unique(pixels[order], return_index=True)
    return order[first]


def _mask_membership(mask: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    if not len(pixels):
        return np.zeros(0, dtype=bool)
    height, width = mask.shape
    x = np.rint(pixels[:, 0]).astype(np.int64)
    y = np.rint(pixels[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    result = np.zeros(len(pixels), dtype=bool)
    result[inside] = mask[y[inside], x[inside]] > 0
    return result


def reprojection_evidence(view: View, mask: np.ndarray, records: Mapping,
                          registered_points_map: np.ndarray) -> dict[str, dict]:
    """Project past measured tracks into this image without using model depth."""
    points_map = _valid_points(registered_points_map)
    current_sources, _, current_depths, current_pixels = _project_map_points(
        points_map, view)
    height, width = mask.shape
    current_z = np.full(height * width, np.inf, dtype=np.float64)
    current_z[current_pixels] = current_depths
    current_source_by_pixel = np.full(height * width, -1, dtype=np.int64)
    current_source_by_pixel[current_pixels] = current_sources
    evidence = {}
    for object_id, record in records.items():
        points = _valid_points(record.measured_points)
        if len(points) < 2 or not np.any(np.ptp(points, axis=0) > 0):
            continue
        _, pixels, depths, raster = _project_map_points(points, view)
        if not len(raster):
            continue
        observed = current_z[raster]
        has_return = np.isfinite(observed)
        tolerance = np.maximum(0.03, 0.02 * depths)
        occluded = has_return & (observed < depths - tolerance)
        inside_mask = _mask_membership(mask, pixels)
        depth_consistent = has_return & (np.abs(observed-depths) <= tolerance)
        depth_inconsistent = has_return & ~depth_consistent
        inside = inside_mask & (~has_return | depth_consistent)
        # A nearer return is a foreground occluder; a farther return is a
        # background penetration.  Neither return is support for the
        # historical surface at this same raster pixel.
        conflict = inside_mask & depth_inconsistent
        conflict_raster = np.unique(raster[conflict])
        conflict_sources = current_source_by_pixel[conflict_raster]
        conflict_sources = conflict_sources[conflict_sources >= 0]
        visible_count = int(np.count_nonzero(~depth_inconsistent))
        matched = int(np.count_nonzero(inside))
        evidence[str(object_id)] = {
            'source': 'historical_measured_image_projection',
            'observation_id': view.observation_id, 'view_id': view.view_id,
            'projected_pixels': len(raster), 'visible_pixels': visible_count,
            'occluded_pixels': int(np.count_nonzero(occluded)),
            'depth_inconsistent_pixels': int(np.count_nonzero(depth_inconsistent)),
            'matched_pixels': matched,
            'matched_fraction': matched / visible_count if visible_count else 0.0,
            'matched_current_return_pixels': int(np.count_nonzero(inside & has_return)),
            'matched_no_current_return_pixels': int(np.count_nonzero(inside & ~has_return)),
            'matched_current_depth_consistent_pixels': int(np.count_nonzero(
                inside & has_return & (np.abs(observed-depths) <= tolerance))),
            # These are current registered-return pixels that occupy a mask
            # pixel where the historical surface is nearer/farther-in-depth
            # inconsistent.  ObjectStore may reject only these measured
            # returns; pixels outside this set remain eligible as newly seen
            # object surface.
            'reprojection_conflict_raster_pixels': conflict_raster.tolist(),
            'reprojection_conflict_current_point_indices': conflict_sources.tolist(),
        }
    return evidence


def _foreground_depth_indices(depths: np.ndarray) -> np.ndarray:
    """Keep one supported surface, including when only sparse hits exist."""
    if len(depths) <= 1:
        return np.arange(len(depths))
    order = np.argsort(depths, kind="mergesort")
    sorted_depth = depths[order]
    median = float(np.median(sorted_depth))
    gap = max(0.08, min(0.30, 0.035 * max(median, 1.0)))
    cuts = np.flatnonzero(np.diff(sorted_depth) > gap) + 1
    groups = [group for group in np.split(order, cuts) if len(group)]
    if not groups:
        return order
    maximum_support = max(len(group) for group in groups)
    minimum_viable = min(maximum_support, max(3, int(math.ceil(0.18 * maximum_support))))
    viable = [group for group in groups if len(group) >= minimum_viable]
    # Nearest supported layer wins.  This prevents a broad wall/table layer
    # from overwhelming a smaller foreground object.
    return min(viable, key=lambda group: float(np.median(depths[group])))


def _select_foreground_depth(
    points: np.ndarray, depths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the nearest supported depth layer in a semantic mask."""
    selected = _foreground_depth_indices(depths)
    return points[selected], depths[selected]


def _sample_mask_pixels(mask: np.ndarray, max_pixels: int = _MAX_ESTIMATED_PIXELS) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(mask > 0)
    if len(rows) > max_pixels:
        # Evenly spaced deterministic samples retain the whole mask extent and
        # avoid keeping a large floating point pointmap on the CPU.
        selected = np.linspace(0, len(rows) - 1, max_pixels, dtype=np.int64)
        rows, columns = rows[selected], columns[selected]
    return rows.astype(np.int64), columns.astype(np.int64)


def _depth_scale_from_lidar(
    depth: np.ndarray,
    pixels: np.ndarray,
    lidar_depths: np.ndarray,
) -> tuple[float, bool, int, float | None, str]:
    """Estimate a stable DA3-to-camera-Z scale from visible LiDAR samples."""

    if not len(pixels):
        return 1.0, False, 0, None, "no_visible_lidar_samples"
    height, width = depth.shape
    x = np.rint(pixels[:, 0]).astype(np.int64)
    y = np.rint(pixels[:, 1]).astype(np.int64)
    valid = (
        (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
        & np.isfinite(lidar_depths)
        & (lidar_depths > _MIN_DEPTH_M)
    )
    if not valid.any():
        return 1.0, False, 0, None, "no_valid_lidar_samples"
    predicted = np.asarray(depth[y[valid], x[valid]], dtype=np.float64)
    measured = np.asarray(lidar_depths[valid], dtype=np.float64)
    valid_depth = (
        np.isfinite(predicted)
        & (predicted > _MIN_DEPTH_M)
        & (predicted <= _MAX_DEPTH_M)
    )
    predicted = predicted[valid_depth]
    measured = measured[valid_depth]
    if len(predicted) < 3:
        return 1.0, False, int(len(predicted)), None, "insufficient_lidar_surface_matches"
    ratios = measured / predicted
    ratios = ratios[np.isfinite(ratios) & (ratios >= 0.25) & (ratios <= 4.0)]
    if len(ratios) < 3:
        return 1.0, False, int(len(ratios)), None, "depth_scale_ratios_invalid"
    scale = float(np.median(ratios))
    residual = float(np.median(np.abs(ratios - scale)))
    # A single plane or object surface should have a coherent scale.  If the
    # visible mask mixes unrelated surfaces, retain raw estimated geometry and
    # report that calibration was rejected.
    coherent = residual <= max(0.05, 0.15 * abs(scale))
    if not coherent:
        return scale, False, int(len(ratios)), residual, "lidar_surface_residual_high"
    return scale, True, int(len(ratios)), residual, "calibrated_to_visible_lidar_z"


def calibrate_depth_to_scan(depth_m: np.ndarray, view: View, points_map: np.ndarray) -> tuple[np.ndarray, dict]:
    """Calibrate one shared view depth map using visible registered returns."""
    _, pixels, depths, _ = _project_map_points(_valid_points(points_map), view)
    scale, calibrated, count, residual, status = _depth_scale_from_lidar(depth_m, pixels, depths)
    calibrated_depth = np.asarray(depth_m, dtype=np.float32) * scale
    regions = []
    if len(pixels):
        x, y = np.rint(pixels).astype(int).T
        predicted = calibrated_depth[y, x]
        valid = np.isfinite(predicted) & (predicted > _MIN_DEPTH_M)
        quadrant = (x >= calibrated_depth.shape[1]/2).astype(int) + 2*(y >= calibrated_depth.shape[0]/2)
        for region in range(4):
            selected = valid & (quadrant == region)
            regions.append({'quadrant': region, 'count': int(selected.sum()),
                            'median_axial_residual_m': float(np.median(np.abs(predicted[selected]-depths[selected]))) if selected.any() else None})
    return calibrated_depth, {
        'depth_scale': scale, 'depth_scale_calibrated': calibrated,
        'depth_calibration_matches': count, 'depth_calibration_residual': residual,
        'depth_calibration_status': status, 'depth_calibration_scope': 'shared_view',
        'depth_region_residuals': regions,
    }


def _aabb_from_points(points: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(points) == 0:
        return None, None
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return None, None
    low = np.min(finite, axis=0)
    high = np.max(finite, axis=0)
    center = 0.5 * (low + high)
    # A single spatial position, including repeated identical contributions,
    # gives a useful bearing/center but no measured extent.  Require an actual
    # spatial difference before applying the positive marker floor; contribution
    # count alone must never manufacture an extent.
    bbox = (
        np.maximum(high - low, np.asarray([0.03, 0.03, 0.03], dtype=np.float64))
        .astype(np.float64)
        if np.any(high != low)
        else None
    )
    return center.astype(np.float64), bbox


def _safe_box(box: object, width: int, height: int) -> list[float]:
    try:
        values = [float(value) for value in box]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if len(values) < 4 or not np.isfinite(values[:4]).all():
        return []
    x1, y1, x2, y2 = values[:4]
    x1, x2 = sorted((max(0.0, min(x1, width - 1.0)), max(0.0, min(x2, width - 1.0))))
    y1, y2 = sorted((max(0.0, min(y1, height - 1.0)), max(0.0, min(y2, height - 1.0))))
    return [float(x1), float(y1), float(x2), float(y2)]


def _crop_for_box(image: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
    if len(box) < 4:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in box[:4]]
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width, x2 + 1))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height, y2 + 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.ascontiguousarray(image[y1:y2, x1:x2].copy())


def _appearance_from_attributes(attributes: Mapping[str, Any]) -> np.ndarray | None:
    raw = attributes.get("appearance_descriptor")
    if raw is None:
        raw = attributes.get("appearance_embedding")
    if raw is None:
        return None
    try:
        descriptor = np.asarray(raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not len(descriptor) or not np.isfinite(descriptor).all():
        return None
    norm = float(np.linalg.norm(descriptor))
    if norm <= 1e-8:
        return None
    return (descriptor / norm).astype(np.float32)


def lift_detection(
    detection: Detection,
    view: View,
    registered_points_map: np.ndarray,
    depth_m: np.ndarray | None = None,
    depth_calibration: dict | None = None,
) -> Observation:
    """Lift one perspective detection into map-frame measured/estimated data.

    Registered scan points are projected through the image-time ``View`` pose
    and filtered by a per-pixel z-buffer before the nearest supported mask
    surface is selected.  Optional DA3 depth is only ever written to
    ``estimated_points``.  In particular, registered map points are never
    map-transformed a second time.
    """

    if not isinstance(detection, Detection):
        raise TypeError("detection_must_be_Detection")
    if not isinstance(view, View):
        raise TypeError("view_must_be_View")
    image = np.asarray(view.image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("view_image_must_be_hwc_rgb")
    raw_mask = np.asarray(detection.mask)
    if raw_mask.ndim == 3:
        raw_mask = np.any(raw_mask > 0, axis=-1)
    if raw_mask.ndim != 2 or raw_mask.shape != image.shape[:2]:
        raise ValueError("detection_mask_shape_mismatch")
    mask = np.asarray(raw_mask > 0, dtype=np.uint8)
    points_map = _valid_points(registered_points_map)
    source_indices, pixels, lidar_depths, pixel_indices = _project_map_points(
        points_map, view
    )
    measured_mask = _mask_membership(mask, pixels)

    # Per-pixel z-buffer uses every projected scan point, including points
    # outside the semantic mask, so a masked background point cannot win over a
    # nearer visible surface.
    width = int(image.shape[1])
    height = int(image.shape[0])
    zbuffer = np.full(height * width, np.inf, dtype=np.float64)
    if len(pixel_indices):
        np.minimum.at(zbuffer, pixel_indices, lidar_depths)
    visible = np.zeros(len(pixels), dtype=bool)
    if len(pixels):
        visible = lidar_depths <= zbuffer[pixel_indices] + np.maximum(
            0.03, 0.02 * lidar_depths
        )
    selected = measured_mask & visible
    selected_indices = source_indices[selected]
    selected_depths = lidar_depths[selected]
    measured_points = points_map[selected_indices]
    surface_indices = _foreground_depth_indices(selected_depths)
    measured_points = measured_points[surface_indices]
    selected_depths = selected_depths[surface_indices]
    measured_raster_pixels = pixel_indices[selected][surface_indices]

    depth_scale = 1.0
    depth_scale_calibrated = False
    depth_calibration_matches = 0
    depth_calibration_residual: float | None = None
    depth_calibration_status = "depth_not_provided"
    estimated_points = np.zeros((0, 3), dtype=np.float64)
    if depth_m is not None:
        depth = np.asarray(depth_m, dtype=np.float64)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2 or depth.shape != image.shape[:2]:
            raise ValueError("depth_shape_mismatch")
        # Runtime supplies a single view-calibrated metre map. Per-instance
        # calibration would give zero-LiDAR objects a different scene scale.
        depth_calibration_status = "provided_metric_depth"
        rows, columns = _sample_mask_pixels(mask, _MAX_ESTIMATED_PIXELS)
        if len(rows):
            predicted = depth[rows, columns] * float(depth_scale)
            valid_depth = np.isfinite(predicted) & (predicted > _MIN_DEPTH_M) & (
                predicted <= _MAX_DEPTH_M
            )
            if valid_depth.any():
                focal_x, focal_y = float(view.intrinsics[0, 0]), float(view.intrinsics[1, 1])
                center_x, center_y = float(view.intrinsics[0, 2]), float(view.intrinsics[1, 2])
                z = predicted[valid_depth]
                x = (columns[valid_depth].astype(np.float64) - center_x) / focal_x * z
                y = (rows[valid_depth].astype(np.float64) - center_y) / focal_y * z
                points_view = np.stack((x, y, z), axis=1)
                transform = _as_transform(view.T_map_view, "T_map_view")
                estimated_points = (
                    points_view @ transform[:3, :3].T + transform[:3, 3]
                )
                estimated_points, estimated_z = _select_foreground_depth(
                    estimated_points, z
                )
                # ``estimated_z`` is deliberately unused after selection; the
                # selected point rows already retain their map coordinates.
                del estimated_z

    measured_center, measured_bbox = _aabb_from_points(measured_points)
    estimated_center, estimated_bbox = _aabb_from_points(estimated_points)
    if measured_bbox is not None:
        center, bbox, geometry_source = measured_center, measured_bbox, "measured"
    elif estimated_bbox is not None:
        center, bbox = estimated_center, estimated_bbox
        geometry_source = (
            "estimated_with_sparse_measurement" if len(measured_points) else "estimated"
        )
    elif measured_center is not None:
        center, bbox, geometry_source = measured_center, None, "measured_sparse"
    elif estimated_center is not None:
        center, bbox, geometry_source = estimated_center, None, "estimated_sparse"
    else:
        center, bbox, geometry_source = None, None, "unavailable"

    box = _safe_box(detection.box_2d, width, height)
    attributes = dict(detection.attributes) if isinstance(detection.attributes, Mapping) else {}
    appearance = _appearance_from_attributes(attributes)
    appearance_source = "provided_descriptor" if appearance is not None else "unavailable"
    if appearance is None and np.any(mask):
        # A lightweight masked RGB descriptor is available on every real
        # detection. It is appearance evidence, not a learned CLIP embedding.
        colors = image[mask.astype(bool)].astype(np.int64) // 32
        bins = colors[:, 0]*64 + colors[:, 1]*8 + colors[:, 2]
        histogram = np.bincount(bins, minlength=512).astype(np.float32)
        appearance = np.sqrt(histogram / histogram.sum())
        appearance_source = "masked_rgb_histogram"
    crop = _crop_for_box(image, box)
    geometry_quality: dict[str, Any] = {
        "source": geometry_source,
        "measured_point_count": int(len(measured_points)),
        "estimated_point_count": int(len(estimated_points)),
        "visible_lidar_point_count": int(np.count_nonzero(selected)),
        "projected_lidar_point_count": int(len(pixels)),
        "measured_raster_pixels": measured_raster_pixels.tolist(),
        "bbox_available": bool(bbox is not None),
        "foreground_selection": "nearest_supported_depth_mode",
        "depth_scale": float(depth_scale),
        "depth_scale_calibrated": bool(depth_scale_calibrated),
        "depth_calibration_matches": int(depth_calibration_matches),
        "depth_calibration_residual": (
            float(depth_calibration_residual)
            if depth_calibration_residual is not None
            else None
        ),
        "depth_calibration_status": depth_calibration_status,
        "registered_points_frame": "map",
        "pose_stamp": float(view.stamp),
        "appearance_source": appearance_source,
    }
    if depth_calibration is not None:
        geometry_quality.update(depth_calibration)
    return Observation(
        observation_id=str(detection.observation_id),
        stamp=float(detection.stamp),
        view_id=str(detection.view_id),
        concept=str(detection.concept),
        score=float(detection.score),
        box_2d=box,
        panorama_mask=_project_mask_to_panorama(
            mask, view.map_x, view.map_y, view.panorama_shape
        ),
        camera_position=np.asarray(view.T_map_view[:3, 3], dtype=np.float64).copy(),
        measured_points=np.asarray(measured_points, dtype=np.float64).copy(),
        estimated_points=np.asarray(estimated_points, dtype=np.float64).copy(),
        center=None if center is None else np.asarray(center, dtype=np.float64).copy(),
        bbox=None if bbox is None else np.asarray(bbox, dtype=np.float64).copy(),
        geometry_quality=geometry_quality,
        attributes=attributes,
        appearance_descriptor=appearance,
        crop_rgb=crop,
    )


__all__ = ["make_views", "lift_detection", "calibrate_depth_to_scan"]
