"""Isolate the real Runtime methods with fake model/geometry/store dependencies.

These tests execute the modified Runtime source, not a replacement runtime.
They verify adapter selection and measured-depth scheduling, not recognition accuracy.
"""
import ast
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
from rebuild.contracts import Frame, View, Detection, Observation, ObjectRecord, QueryResult, VerifiedObservation
from go2.sensors import RGBDFrame, PinholeObservationAdapter


class FakeModels:
    def __init__(self, config, log_event):
        self.depth_calls = 0
    def compile_task(self, text):
        return {'task_type': 'numerical', 'expression': {'op': 'filter_class', 'class': 'chair'}, 'steps': []}
    def detect(self, view, concepts, queries):
        return [Detection(view.observation_id, view.stamp, view.view_id, 'chair', [0, 0, 3, 3],
                          np.ones((4, 4), dtype=np.uint8), .9)]
    def verify_categories(self, batch):
        return [{'verdict': 'yes'} for _ in batch]
    def depth(self, view):
        self.depth_calls += 1
        return np.ones((4, 4), np.float32)


class FakeStore:
    def __init__(self, config):
        self.records = {}
    def update(self, observations):
        for obs in observations:
            self.records['chair_1'] = ObjectRecord('chair_1', {'chair': .9}, {},
                obs.measured_points, obs.estimated_points, obs.center, obs.bbox,
                obs.geometry_quality, observation_ids=[obs.observation_id])
        return ['chair_1'] * len(observations)


class FakeLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.written = {}
    def event(self, *args, **kwargs):
        pass
    def write(self, path, value):
        self.written[path] = value
    def store_snapshot(self, store):
        return {'count': len(store.records), 'records': {}}


def views(frame, calibration, config):
    x, y = np.meshgrid(np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32))
    return [View(frame.observation_id, frame.stamp, 'cmu_view', frame.panorama_rgb,
                 np.eye(3), np.eye(4), x, y, (4, 4))]


def lift(detection, view, points, *args):
    return Observation(detection.observation_id, detection.stamp, view.view_id, 'chair', .9,
                       detection.box_2d, detection.mask.copy(), np.zeros(3), points,
                       np.empty((0, 3)), np.array([1., 0, 1]), np.ones(3), {})


def runtime_module():
    source = Path(__file__).resolve().parents[2] / 'rebuild/runtime.py'
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level)]
    module = types.ModuleType('go2_isolated_runtime_under_test')
    module.__dict__.update({
        'Frame': Frame, 'ObjectRecord': ObjectRecord, 'VerifiedObservation': VerifiedObservation,
        'ModelRuntime': FakeModels, 'ObjectStore': FakeStore, 'Navigation': lambda config: types.SimpleNamespace(),
        'RunLog': FakeLog, 'make_views': views, 'lift_detection': lift,
        'calibrate_depth_to_scan': lambda depth, view, points: (depth, {}),
        'reprojection_evidence': lambda *args: {},
        'sample_view_region': lambda frame, view, calibration, bounds, size, vfov:
            np.zeros((size[1], size[0], 3), np.uint8),
        'evaluate': lambda *args: QueryResult(),
        'resolve': lambda *args: QueryResult(object_ids=['chair_1'], value=1, complete=True),
        'task_concepts': lambda task: ['chair'], 'task_visual_queries': lambda task: {'chair': 'chair'},
    })
    sys.modules[module.__name__] = module
    exec(compile(tree, str(source), 'exec'), module.__dict__)
    return module


class RuntimeAdapterTest(unittest.TestCase):
    def run_observation(self, use_adapter, depth_enabled=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        (directory / 'calibration.json').write_text('{}')
        config = {'calibration': str(directory / 'calibration.json'), 'panorama_vertical_fov_deg': 120,
                  'horizontal_fov_deg': 105, 'category_crop_size': 32, 'category_batch_size': 2,
                  'depth_min_lidar_points': 12, 'max_pair_verifications_per_observation': 3}
        if depth_enabled is not None:
            config['estimated_depth_enabled'] = depth_enabled
        kwargs = {}
        if use_adapter:
            config['calibration'] = '/no/unity/calibration/on/go2.json'
            kwargs['observation_adapter'] = PinholeObservationAdapter()
            kwargs['navigation_factory'] = lambda config: types.SimpleNamespace(platform='go2')
        runtime = runtime_module().Runtime(config, **kwargs)
        log = FakeLog(directory)
        episode = runtime.begin('How many chairs?', log, 0.)
        frame = RGBDFrame('f', 1., np.zeros((4, 4, 3), np.uint8), np.eye(4),
                          np.array([[0, 0, 1.], [.1, 0, 1.]]), 1.)
        runtime.observe(episode, frame)
        return runtime, episode, log

    def test_go2_uses_adapter_and_no_da3(self):
        runtime, episode, log = self.run_observation(True, False)
        self.assertEqual(runtime.models.depth_calls, 0)
        self.assertEqual(episode.navigation.platform, 'go2')
        self.assertEqual(episode.scene.observations, 1)
        self.assertIn('RealSense', episode.scene.records['chair_1'].geometry_quality['measured_sensor_source'])
        self.assertEqual(log.written['observations/f/views.json'][0]['projection'], 'rectified_pinhole')

    def test_cmu_defaults_preserve_original_path(self):
        runtime, episode, log = self.run_observation(False)
        self.assertEqual(runtime.models.depth_calls, 1)
        self.assertIsNone(runtime.observation_adapter)
        self.assertEqual(log.written['observations/f/views.json'][0]['projection'], 'cropped_equirectangular')
        self.assertNotIn('measured_sensor_source', episode.scene.records['chair_1'].geometry_quality)


if __name__ == '__main__':
    unittest.main()
