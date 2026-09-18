import unittest
import numpy as np
from types import SimpleNamespace as NS
import test_node_orchestration as orchestration

class IntermediateObservationTest(unittest.TestCase):
    setUp = orchestration.NodeOrchestrationTest.setUp
    def test_remaining_route_gets_one_forward_view(self):
        n=self.node
        n.episode.navigation=NS(route=[np.array([2.,0.])])
        n.motion_kind='move'
        n.transport.status='succeeded'
        n.transport.last_success=(1.,0.,0.)
        n._tick()
        self.assertEqual(n.sweep.offsets,[0.])
        self.assertEqual(n.events[-1][1]['reason'],'intermediate_waypoint')

    def test_route_endpoint_retains_full_sweep(self):
        n=self.node
        n.episode.navigation=NS(route=[])
        n.motion_kind='move'
        n.transport.status='succeeded'
        n.transport.last_success=(1.,0.,0.)
        n._tick()
        self.assertEqual(len(n.sweep.offsets),2)
        self.assertTrue(n.events[-1][1]['full_sweep'])
