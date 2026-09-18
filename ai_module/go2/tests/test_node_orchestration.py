"""Exercise the actual ROS-node state machine with isolated transport/model fakes."""
import ast
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS
import math
import time
import unittest

import numpy as np
from go2.navigation import HeadingSweep
from go2.sensors import yaw_of, angle_difference


def node_class():
    path = Path(__file__).resolve().parents[1] / 'ros_node.py'
    tree = ast.parse(path.read_text())
    tree.body = [item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == 'Go2Node']
    namespace = {'Node': object, 'np': np, 'time': time, 'math': math,
                 'HeadingSweep': HeadingSweep, 'yaw_of': yaw_of,
                 'angle_difference': angle_difference, 'TransformException': RuntimeError}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['Go2Node']


class NodeOrchestrationTest(unittest.TestCase):
    def setUp(self):
        cls = node_class()
        self.node = cls.__new__(cls)
        n = self.node
        n.config = {'question_time_budget_seconds': 900, 'scan_yaws_deg': [0, 60],
                    'observation_settle_seconds': .3, 'scan_yaw_tolerance_deg': 10.}
        n.started, n.terminal, n.epoch = time.monotonic(), False, 2
        n.future, n.motion_kind, n.sweep = None, None, None
        n.pose = np.eye(4)
        n.update_pose = lambda: None
        n.now_seconds = lambda: 10.
        n.episode = NS()
        n.runtime = NS(observe=lambda *args: None)
        n.events, n.work, n.goals, n.finishes = [], [], [], []
        n.log = NS(event=lambda name, **fields: n.events.append((name, fields)))
        n.capture = lambda: None
        n.waiting = lambda reason: None
        n.finish = lambda reason, **fields: n.finishes.append((reason, fields))
        n.submit_work = lambda *args: n.work.append(args)
        n.transport = NS(pump=lambda: None, busy=False, error=None, status='idle',
                         completed_stamp=9.9, last_success=None,
                         submit=lambda goal: n.goals.append(goal))

    def test_first_view_before_turn(self):
        n = self.node
        frame = object()
        n.capture = lambda: frame
        n._tick()
        self.assertEqual(n.work[0][0], 'observe')
        self.assertIs(n.work[0][-1], frame)
        self.assertEqual(n.goals, [])

    def test_turn_result_logs_kind_then_waits_for_fresh_image(self):
        n = self.node
        n.new_sweep()
        n.sweep.consumed()
        n._tick()
        self.assertEqual(n.motion_kind, 'turn')
        self.assertAlmostEqual(n.goals[0][2], math.pi/3)
        n.transport.busy = True
        n._tick()
        self.assertEqual(n.work, [])
        n.transport.busy, n.transport.status = False, 'succeeded'
        n.transport.last_success = n.goals[0]
        angle = math.pi/3
        n.pose[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        n._tick()
        self.assertIsNone(n.motion_kind)
        self.assertAlmostEqual(n.ready_after, 10.2)
        self.assertEqual(n.events[-1][1]['kind'], 'turn')
        self.assertEqual(n.work, [])

    def test_navigation_failure_finishes_instead_of_observing(self):
        n = self.node
        n.transport.error = 'Nav2 abort'
        n._tick()
        self.assertEqual(n.finishes[0][0], 'navigation_failure')
        self.assertEqual(n.work, [])

    def test_previous_episode_inference_is_not_consumed(self):
        n = self.node
        n.sweep = HeadingSweep([0, 60])
        n.future, n.future_epoch, n.future_kind = Future(), 1, 'observe'
        n.future.set_result(None)
        n.consume_work()
        self.assertIsNone(n.future)
        self.assertEqual(n.sweep.index, 0)


if __name__ == '__main__':
    unittest.main()
