"""Apply Nav2 OccupancyGridUpdate patches to the most recent full grid.

No ROS imports: the message objects supplied by the ROS adapter retain their
origin, resolution and frame. Updates change only their advertised rectangle.
"""
from copy import deepcopy
import numpy as np


class CostmapBuffer:
    def __init__(self):
        self.message = None
        self.version = 0
        self._last_stamp = 0.0

    @staticmethod
    def _stamp(message):
        value = message.header.stamp
        return value.sec + value.nanosec * 1e-9

    def set_full(self, message):
        if len(message.data) != message.info.width * message.info.height:
            raise ValueError('Full costmap dimensions do not match its data')
        self.message = deepcopy(message)
        self._last_stamp = self._stamp(message)
        self.version += 1

    def apply_update(self, update):
        if self.message is None:
            return False, 'waiting_for_full_costmap'
        message = self.message
        if update.header.frame_id.lstrip('/') != message.header.frame_id.lstrip('/'):
            return False, 'costmap_update_frame_mismatch'
        x, y, w, h = int(update.x), int(update.y), int(update.width), int(update.height)
        if (x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > message.info.width
                or y+h > message.info.height or len(update.data) != w*h):
            return False, 'costmap_update_bounds_mismatch'
        stamp = self._stamp(update)
        # Some Nav2 Jazzy releases publish updates with a zero stamp. Those
        # updates still describe the current full-grid layout and must work.
        if stamp > 0 and self._last_stamp > 0 and stamp < self._last_stamp:
            return False, 'older_costmap_update'
        data = np.asarray(message.data, dtype=np.int8).reshape(message.info.height, message.info.width).copy()
        data[y:y+h, x:x+w] = np.asarray(update.data, dtype=np.int8).reshape(h, w)
        message.data = data.ravel().tolist()
        if stamp > 0:
            message.header.stamp = deepcopy(update.header.stamp)
            self._last_stamp = stamp
        self.version += 1
        return True, 'costmap_updated'
