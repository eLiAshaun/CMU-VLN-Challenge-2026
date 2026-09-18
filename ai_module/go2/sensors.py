"""Pinhole RGB-D geometry. No ROS imports; depth is measured camera-Z in metres."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from rebuild.contracts import Frame, View


@dataclass
class RGBDFrame(Frame):
    intrinsics: np.ndarray = field(default_factory=lambda: np.eye(3))
    T_map_camera: np.ndarray = field(default_factory=lambda: np.eye(4))
    metadata: dict = field(default_factory=dict)


def depth_metres(data, width: int, height: int, step: int, encoding: str,
                 bigendian: bool = False, unit_16u: float = 0.001) -> np.ndarray:
    """Decode ROS depth, retaining row stride and byte order (16UC1/mm, 32FC1/m)."""
    kind = {'16UC1': 'u2', '32FC1': 'f4'}.get(encoding.upper())
    if kind is None:
        raise ValueError(f'Unsupported depth encoding: {encoding}')
    dtype = np.dtype(('>' if bigendian else '<') + kind)
    if step < width * dtype.itemsize:
        raise ValueError('Depth row stride is shorter than its pixel data')
    image = np.ndarray((height, width), dtype=dtype, buffer=data,
                       strides=(step, dtype.itemsize)).astype(np.float32)
    if kind == 'u2':
        image *= unit_16u
    image[~np.isfinite(image) | (image <= 0)] = np.nan
    return image


def rgb_image(message) -> np.ndarray:
    encoding = message.encoding.lower()
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}[encoding]
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, :message.width * channels].reshape(message.height, message.width, channels)
    if channels == 1:
        return np.repeat(image, 3, axis=2)
    image = image[:, :, :3]
    return image[:, :, ::-1].copy() if encoding.startswith('bgr') else image.copy()


def rotation_xyzw(quaternion) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('Invalid TF quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def transform_matrix(transform) -> np.ndarray:
    t, q = transform.translation, transform.rotation
    result = np.eye(4)
    result[:3, :3] = rotation_xyzw([q.x, q.y, q.z, q.w])
    result[:3, 3] = [t.x, t.y, t.z]
    return result


def yaw_of(transform: np.ndarray) -> float:
    return math.atan2(transform[1, 0], transform[0, 0])


def angle_difference(a: float, b: float) -> float:
    return math.atan2(math.sin(a-b), math.cos(a-b))


class RGBDProjector:
    """Rectify aligned RGB/depth together and back-project with actual CameraInfo."""
    def __init__(self, min_depth=0.2, max_depth=6.0, stride=2):
        self.min_depth, self.max_depth, self.stride = min_depth, max_depth, stride
        self._key = None
        self._maps = None

    def rectify(self, rgb, depth, K, distortion, model='plumb_bob'):
        if rgb.shape[:2] != depth.shape:
            raise ValueError('Use depth aligned to COLOR at the color image resolution')
        K = np.asarray(K, dtype=float).reshape(3, 3)
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError('CameraInfo has no valid calibration')
        distortion = np.asarray(distortion, dtype=float)
        if not len(distortion) or np.allclose(distortion, 0):
            return rgb.copy(), depth.copy(), K.copy()
        if model not in {'plumb_bob', 'rational_polynomial'}:
            raise ValueError(f'Unsupported distortion model {model}; supply rectified RGB-D')
        h, w = depth.shape
        key = (w, h, *K.ravel(), *distortion)
        if key != self._key:
            self._maps = cv2.initUndistortRectifyMap(K, distortion, np.eye(3), K,
                                                    (w, h), cv2.CV_32FC1)
            self._key = key
        rgb = cv2.remap(rgb, *self._maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        depth = cv2.remap(depth, *self._maps, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=float('nan'))
        return rgb, depth, K.copy()

    def points(self, depth, K, T_map_camera):
        rows, cols = np.mgrid[0:depth.shape[0]:self.stride, 0:depth.shape[1]:self.stride]
        z = depth[rows, cols]
        good = np.isfinite(z) & (z >= self.min_depth) & (z <= self.max_depth)
        pixels = np.stack((cols[good], rows[good], np.ones(good.sum())), axis=-1)
        rays = pixels @ np.linalg.inv(K).T
        points_camera = rays * (z[good] / rays[:, 2])[:, None]
        return (points_camera @ T_map_camera[:3, :3].T + T_map_camera[:3, 3]).astype(np.float32)

    def frame(self, observation_id, stamp, rgb, depth, K, distortion, distortion_model,
              T_map_base, T_map_camera, depth_stamp):
        rgb, depth, K = self.rectify(rgb, depth, K, distortion, distortion_model)
        points = self.points(depth, K, T_map_camera)
        return RGBDFrame(observation_id, stamp, rgb, T_map_base, points, depth_stamp,
                         K, T_map_camera,
                         {'camera_model': 'pinhole', 'depth_source': 'realsense_aligned_depth',
                          'depth_units': 'metres_camera_z', 'depth_stamp': depth_stamp,
                          'rgb_stamp': stamp, 'T_map_camera': T_map_camera.tolist(),
                          'depth_sample_stride': self.stride})


class PinholeObservationAdapter:
    projection = 'rectified_pinhole'
    geometry_source = 'RealSense measured aligned depth; image-time TF in map frame'

    def make_views(self, frame: RGBDFrame):
        h, w = frame.panorama_rgb.shape[:2]
        x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return [View(frame.observation_id, frame.stamp, 'front', frame.panorama_rgb,
                     frame.intrinsics, frame.T_map_camera, x, y, (h, w))]

    def view_metadata(self, view):
        h, w = view.image_rgb.shape[:2]
        K = view.intrinsics
        hfov = math.atan((w-0.5-K[0, 2])/K[0, 0]) - math.atan((-0.5-K[0, 2])/K[0, 0])
        vfov = math.atan((h-0.5-K[1, 2])/K[1, 1]) - math.atan((-0.5-K[1, 2])/K[1, 1])
        return {'horizontal_fov_deg': math.degrees(hfov), 'vertical_fov_deg': math.degrees(vfov)}

    def annotate_observation(self, observation, detection):
        observation.panorama_mask = np.asarray(detection.mask, dtype=np.uint8).copy()
        for old, new in (('visible_lidar_point_count', 'visible_rgbd_point_count'),
                         ('projected_lidar_point_count', 'projected_rgbd_point_count')):
            if old in observation.geometry_quality:
                observation.geometry_quality[new] = observation.geometry_quality[old]
        observation.geometry_quality['legacy_lidar_keys_describe'] = 'measured RGB-D samples'

    def category_image(self, frame, view, detection, config):
        left, top, right, bottom = detection.box_2d
        margin = 0.5 * max(right-left, bottom-top) + 20
        h, w = view.image_rgb.shape[:2]
        x0, y0 = max(0, math.floor(left-margin)), max(0, math.floor(top-margin))
        x1, y1 = min(w, math.ceil(right+margin)), min(h, math.ceil(bottom+margin))
        if x1 <= x0 or y1 <= y0:
            raise ValueError('Detection has no image support')
        image = Image.fromarray(view.image_rgb[y0:y1, x0:x1])
        scale = config['category_crop_size'] / max(image.size)
        size = (max(1, round(image.width*scale)), max(1, round(image.height*scale)))
        image = image.resize(size, Image.Resampling.BICUBIC)
        sx, sy = size[0]/(x1-x0), size[1]/(y1-y0)
        ImageDraw.Draw(image).rectangle(((left-x0)*sx, (top-y0)*sy,
                                        (right-x0)*sx, (bottom-y0)*sy), outline='red', width=2)
        return image, {'source': 'same_pinhole_clamped_context',
                       'view_bounds': [x0, y0, x1, y1], 'output_size': list(size),
                       'image_stamp': frame.stamp}
