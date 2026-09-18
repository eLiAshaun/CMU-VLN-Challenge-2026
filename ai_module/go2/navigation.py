"""Nav2 costmap adaptation without inventing free space between sensor returns."""
from __future__ import annotations

import math
import numpy as np


def costmap_points(data, width, height, resolution, T_map_grid, free_cost=0):
    """OccupancyGrid (-1 unknown, 0 free, 1..100 costs) -> classified map points.

    The fourth column is a binary obstacle/unknown flag, not LiDAR intensity.
    Nav2 already inflated this map; the Go2 navigation adapter does not inflate again.
    """
    costs = np.asarray(data, dtype=np.int16).reshape(height, width)
    y, x = np.mgrid[:height, :width]
    points = np.stack(((x.ravel()+0.5)*resolution, (y.ravel()+0.5)*resolution,
                       np.zeros(width*height)), axis=-1)
    points = points @ T_map_grid[:3, :3].T + T_map_grid[:3, 3]
    blocked = ((costs < 0) | (costs > free_cost)).ravel().astype(np.float32)
    return np.column_stack((points, blocked)).astype(np.float32)


def known_free_cells(points, position, resolution=0.25):
    """Conservative fine-to-coarse raster: any non-free sample blocks a coarse cell."""
    points = np.asarray(points)
    if not len(points):
        return set()
    local = points[np.linalg.norm(points[:, :2]-np.asarray(position)[:2], axis=1) <= 12]
    cells = np.rint(local[:, :2]/resolution).astype(int)
    free = set(map(tuple, cells[local[:, 3] == 0]))
    blocked = set(map(tuple, cells[local[:, 3] != 0]))
    return free - blocked


def navigation_factory(config):
    # Import only at runtime: pure projection/costmap tests need no model stack.
    from rebuild.navigation import Navigation

    class Go2Navigation(Navigation):
        def _terrain(self, terrain):
            if terrain is self.last_terrain:
                return
            self.last_terrain = terrain
            self.free_cells = (known_free_cells(terrain, self.position, self.resolution)
                               if self.position is not None else set())

    return Go2Navigation(config)


class HeadingSweep:
    """Real, sequential views from a forward camera; no synthetic panorama."""
    def __init__(self, offsets_deg):
        self.offsets = [math.radians(value) for value in offsets_deg]
        self.reset(0.0)

    def reset(self, yaw):
        self.origin_yaw = yaw
        self.index = 0

    @property
    def complete(self):
        return self.index >= len(self.offsets)

    @property
    def target(self):
        value = self.origin_yaw + self.offsets[self.index]
        return math.atan2(math.sin(value), math.cos(value))

    def consumed(self):
        self.index += 1
