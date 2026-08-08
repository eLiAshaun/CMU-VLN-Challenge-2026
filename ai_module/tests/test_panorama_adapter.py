from __future__ import annotations

import cv2
import numpy as np

from integrations.mast3r.panorama_adapter import (
    project_equirectangular,
    project_mask_to_panorama,
)


def _coordinate_panorama(height: int = 640, width: int = 1920) -> np.ndarray:
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    return np.stack(
        (
            np.broadcast_to(x % 256, (height, width)),
            np.broadcast_to(y % 256, (height, width)),
            np.broadcast_to((x // 256) % 256, (height, width)),
        ),
        axis=-1,
    ).astype(np.uint8)


def test_forward_view_center_and_intrinsics() -> None:
    projected = project_equirectangular(_coordinate_panorama(), yaw_deg=0.0)
    center_y, center_x = 384 // 2, 512 // 2
    assert abs(float(projected.map_x[center_y, center_x]) - 960.0) < 2.0
    assert abs(float(projected.map_y[center_y, center_x]) - 320.0) < 2.0
    assert projected.intrinsics.shape == (3, 3)
    assert np.isclose(projected.intrinsics[0, 0], 256.0)


def test_seam_view_wraps_without_black_border() -> None:
    projected = project_equirectangular(_coordinate_panorama(), yaw_deg=180.0)
    center_y, center_x = 384 // 2, 512 // 2
    seam_x = float(projected.map_x[center_y, center_x])
    assert seam_x < 2.0 or seam_x > 1918.0
    assert projected.image_bgr.shape == (384, 512, 3)


def test_mask_backprojection_uses_saved_ray_map() -> None:
    projected = project_equirectangular(_coordinate_panorama(), yaw_deg=180.0)
    mask = np.zeros((384, 512), dtype=np.uint8)
    cv2.circle(mask, (256, 192), 12, 255, -1)
    panorama_mask = project_mask_to_panorama(
        mask, projected.map_x, projected.map_y, (640, 1920)
    )
    assert panorama_mask.sum() > 0
    seam_support = np.count_nonzero(panorama_mask[:, :20]) + np.count_nonzero(
        panorama_mask[:, -20:]
    )
    assert seam_support > 0
