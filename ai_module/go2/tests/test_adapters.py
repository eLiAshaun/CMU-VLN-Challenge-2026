"""CPU unit tests; no ROS installation, camera, GPU or robot is required."""
from concurrent.futures import Future
from types import SimpleNamespace as NS
import math
import unittest

import numpy as np
from rebuild.contracts import Detection
from go2.sensors import (depth_metres, RGBDProjector, PinholeObservationAdapter,
                         RGBDFrame, angle_difference, rotation_xyzw)
from go2.navigation import costmap_points, known_free_cells, HeadingSweep
from go2.nav2_transport import Nav2Transport


class SensorsTest(unittest.TestCase):
    def test_millimetres_and_invalid_zero(self):
        raw = np.array([[1000, 2500, 0]], dtype='<u2')
        depth = depth_metres(raw.tobytes(), 3, 1, 6, '16UC1')
        np.testing.assert_allclose(depth[0, :2], [1, 2.5])
        self.assertTrue(np.isnan(depth[0, 2]))

    def test_float_metres(self):
        raw = np.array([[1.25, np.inf, -1, np.nan]], dtype='<f4')
        depth = depth_metres(raw.tobytes(), 4, 1, 16, '32FC1')
        self.assertEqual(depth[0, 0], 1.25)
        self.assertTrue(np.isnan(depth[0, 1:]).all())

    def test_stride_and_big_endian(self):
        raw = np.array([[1000, 2000, 999], [3000, 4000, 999]], dtype='>u2')
        depth = depth_metres(raw.tobytes(), 2, 2, 6, '16UC1', True)
        np.testing.assert_allclose(depth, [[1, 2], [3, 4]])

    def test_colored_depth_rejected(self):
        with self.assertRaises(ValueError):
            depth_metres(bytes(12), 2, 2, 6, 'rgb8')

    def test_axial_depth_and_actual_intrinsics(self):
        K = np.array([[2., 0, 1], [0, 2, 0], [0, 0, 1]])
        points = RGBDProjector(stride=1).points(np.full((1, 3), 2.), K, np.eye(4))
        np.testing.assert_allclose(points, [[-1, 0, 2], [0, 0, 2], [1, 0, 2]])

    def test_full_se3_not_planar(self):
        T = np.eye(4)
        T[:3, :3] = rotation_xyzw([0, math.sin(math.pi/4), 0, math.cos(math.pi/4)])
        T[:3, 3] = [10, 20, 30]
        points = RGBDProjector(stride=1).points(np.array([[2.]]), np.eye(3), T)
        np.testing.assert_allclose(points, [[12, 20, 30]], atol=1e-5)

    def test_invalid_and_out_of_range_are_not_points(self):
        points = RGBDProjector(stride=1).points(np.array([[0., np.nan, .1, 9.]]), np.eye(3), np.eye(4))
        self.assertEqual(points.shape, (0, 3))

    def test_misaligned_resolution_rejected(self):
        with self.assertRaises(ValueError):
            RGBDProjector().rectify(np.zeros((4, 4, 3), np.uint8), np.ones((2, 2)), np.eye(3), [], '')

    def test_rectification_retains_metric_depth(self):
        rgb = np.zeros((10, 10, 3), np.uint8)
        K = np.array([[10., 0, 5], [0, 10, 5], [0, 0, 1]])
        _, depth, new_k = RGBDProjector().rectify(rgb, np.full((10, 10), 2., np.float32),
                                               K, [.1, 0, 0, 0, 0])
        np.testing.assert_allclose(depth[np.isfinite(depth)], 2)
        np.testing.assert_allclose(new_k, K)

    def test_frame_keeps_base_separate_from_optical_pose(self):
        T = np.eye(4)
        T[:3, 3] = [1, 2, 3]
        frame = RGBDProjector(stride=1).frame('f', 7., np.zeros((2, 2, 3), np.uint8),
            np.full((2, 2), 2.), np.eye(3), [], '', np.eye(4), T, 7.01)
        np.testing.assert_array_equal(frame.T_map_sensor, np.eye(4))
        np.testing.assert_array_equal(frame.T_map_camera, T)
        self.assertEqual(frame.metadata['depth_stamp'], 7.01)
        view = PinholeObservationAdapter().make_views(frame)[0]
        np.testing.assert_array_equal(view.T_map_view, T)
        self.assertEqual(view.view_id, 'front')

    def test_pinhole_crop_does_not_wrap(self):
        image = np.zeros((20, 100, 3), np.uint8)
        image[:, 80:, 1] = 255
        frame = RGBDFrame('f', 1., image, np.eye(4), np.empty((0, 3)), 1.)
        adapter = PinholeObservationAdapter()
        view = adapter.make_views(frame)[0]
        det = Detection('f', 1., 'front', 'box', [1, 4, 6, 10], np.ones((20, 100)), .8)
        crop, metadata = adapter.category_image(frame, view, det, {'category_crop_size': 64})
        self.assertEqual(metadata['view_bounds'][0], 0)
        self.assertEqual(np.asarray(crop)[:, :, 1].max(), 0)


class MappingTest(unittest.TestCase):
    def test_unknown_never_free(self):
        points = costmap_points([0, -1, 100], 3, 1, .25, np.eye(4))
        np.testing.assert_array_equal(points[:, 3], [0, 1, 1])
        free = known_free_cells(points, [0, 0, 0])
        self.assertEqual(len(free), 1)

    def test_costmap_origin_rotation(self):
        T = np.eye(4)
        T[:3, :3] = rotation_xyzw([0, 0, math.sin(math.pi/4), math.cos(math.pi/4)])
        T[:3, 3] = [10, 20, 0]
        points = costmap_points([0], 1, 1, 2., T)
        np.testing.assert_allclose(points[0, :3], [9, 21, 0])

    def test_one_obstacle_blocks_coarse_cell(self):
        free = known_free_cells(np.array([[0, 0, 0, 0], [.01, .01, 0, 1]]), [0, 0, 0])
        self.assertEqual(free, set())

    def test_no_free_rays_in_unknown_space(self):
        free = known_free_cells(np.array([[2., 0, 0, 0]]), [0, 0, 0])
        self.assertEqual(free, {(8, 0)})

    def test_heading_sweep_requires_consumed_views(self):
        sweep = HeadingSweep([0, 60, 120, 180, 240, 300])
        sweep.reset(math.radians(350))
        self.assertFalse(sweep.complete)
        sweep.consumed()
        self.assertAlmostEqual(sweep.target, math.radians(50))
        for _ in range(5):
            sweep.consumed()
        self.assertTrue(sweep.complete)

    def test_yaw_wrap(self):
        self.assertAlmostEqual(angle_difference(math.radians(1), math.radians(359)), math.radians(2))


class FakeHandle:
    accepted = True
    def __init__(self):
        self.result, self.cancel_reply = Future(), Future()
        self.cancel_calls = 0
    def get_result_async(self):
        return self.result
    def cancel_goal_async(self):
        self.cancel_calls += 1
        return self.cancel_reply


class FakeClient:
    ready = True
    def __init__(self):
        self.requests = []
    def server_is_ready(self):
        return self.ready
    def send_goal_async(self, goal):
        future = Future()
        self.requests.append((goal, future))
        return future
    def accept(self, index=-1):
        handle = FakeHandle()
        self.requests[index][1].set_result(handle)
        return handle


def goal_factory():
    return NS(pose=NS(header=NS(frame_id='', stamp=None),
                      pose=NS(position=NS(x=0., y=0., z=0.), orientation=NS(z=0., w=1.))))


def action_result(status=4, code=0):
    return NS(status=status, result=NS(error_code=code, error_msg='test'))


class TransportTest(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        clock = NS(now=lambda: NS(nanoseconds=123000000000, to_msg=lambda: None))
        self.transport = Nav2Transport(NS(get_clock=lambda: clock), {'world_frame': 'map'},
                                       self.client, goal_factory)

    def test_nonfinite_goal_is_not_sent(self):
        with self.assertRaises(ValueError):
            self.transport.submit([float('nan'), 0, 0])
        self.assertEqual(len(self.client.requests), 0)

    def test_goal_has_frame_and_heading(self):
        self.transport.submit([1, 2, math.pi/2])
        goal = self.client.requests[0][0]
        self.assertEqual(goal.pose.header.frame_id, 'map')
        self.assertAlmostEqual(goal.pose.pose.orientation.z, math.sqrt(.5))

    def test_acceptance_is_not_success(self):
        self.transport.submit([1, 2, 0])
        handle = self.client.accept()
        self.assertTrue(self.transport.busy)
        self.assertIsNone(self.transport.last_success)
        handle.result.set_result(action_result())
        self.assertEqual(self.transport.last_success, (1, 2, 0))
        self.assertAlmostEqual(self.transport.completed_stamp, 123.)
        self.transport.pump()
        self.assertEqual(len(self.client.requests), 1)

    def test_duplicate_does_not_resend(self):
        self.transport.submit([1, 2, 0])
        self.client.accept()
        self.transport.submit([1, 2, 0])
        self.assertEqual(len(self.client.requests), 1)

    def test_replace_waits_for_cancelled_result(self):
        self.transport.submit([1, 2, 0])
        handle = self.client.accept()
        self.transport.submit([3, 4, 0])
        self.assertEqual(handle.cancel_calls, 1)
        self.assertEqual(len(self.client.requests), 1)
        handle.cancel_reply.set_result(NS(goals_canceling=[1]))
        self.assertEqual(len(self.client.requests), 1)
        handle.result.set_result(action_result(5))
        self.assertEqual(len(self.client.requests), 2)

    def test_cancel_during_acceptance(self):
        self.transport.submit([1, 2, 0])
        self.transport.cancel()
        handle = self.client.accept()
        self.assertEqual(handle.cancel_calls, 1)
        handle.result.set_result(action_result(5))
        self.assertIsNone(self.transport.last_success)
        self.assertIsNone(self.transport.error)

    def test_newest_of_multiple_pending_goals(self):
        self.transport.submit([1, 2, 0])
        self.transport.submit([3, 4, 0])
        self.transport.submit([5, 6, 0])
        handle = self.client.accept()
        handle.result.set_result(action_result(5))
        self.assertEqual(len(self.client.requests), 2)
        self.assertEqual(self.client.requests[1][0].pose.pose.position.x, 5)

    def test_late_cancel_reply_does_not_poison_new_goal(self):
        self.transport.submit([1, 2, 0])
        handle = self.client.accept()
        self.transport.submit([3, 4, 0])
        handle.result.set_result(action_result(5))
        self.client.accept()
        handle.cancel_reply.set_result(NS(goals_canceling=[]))
        self.assertIsNone(self.transport.error)

    def test_goal_rejected(self):
        self.transport.submit([1, 2, 0])
        self.client.requests[0][1].set_result(NS(accepted=False))
        self.assertEqual(self.transport.status, 'rejected')

    def test_server_failure(self):
        self.transport.submit([1, 2, 0])
        handle = self.client.accept()
        handle.result.set_result(action_result(6, 42))
        self.assertIsNotNone(self.transport.error)
        self.assertIsNone(self.transport.last_success)

    def test_nonzero_result_error_not_success(self):
        self.transport.submit([1, 2, 0])
        self.client.accept().result.set_result(action_result(4, 42))
        self.assertEqual(self.transport.status, 'failed')

    def test_server_unavailable_keeps_goal_without_send(self):
        self.client.ready = False
        self.transport.submit([1, 2, 0])
        self.assertEqual(len(self.client.requests), 0)
        self.client.ready = True
        self.transport.pump()
        self.assertEqual(len(self.client.requests), 1)

    def test_denied_cancel_does_not_send_replacement(self):
        self.transport.submit([1, 2, 0])
        handle = self.client.accept()
        self.transport.submit([3, 4, 0])
        handle.cancel_reply.set_result(NS(goals_canceling=[]))
        self.assertIsNotNone(self.transport.error)
        self.assertEqual(len(self.client.requests), 1)


if __name__ == '__main__':
    unittest.main()
