from types import SimpleNamespace as NS
import unittest
import numpy as np
from go2.costmap import CostmapBuffer
from go2.packet import packet_timing, timing_valid


def header(stamp=10, frame='map'):
    return NS(frame_id=frame, stamp=NS(sec=int(stamp), nanosec=round((stamp-int(stamp))*1e9)))


def full(width=4, height=3, stamp=10):
    return NS(header=header(stamp), info=NS(width=width, height=height), data=[0]*(width*height))


def update(x=1, y=1, width=2, height=1, values=(100,-1), stamp=11, frame='map'):
    return NS(header=header(stamp,frame), x=x,y=y,width=width,height=height,data=list(values))


class CostmapTest(unittest.TestCase):
    def setUp(self):
        self.buffer=CostmapBuffer()
        self.original=full()
        self.buffer.set_full(self.original)

    def test_incremental_patch_changes_only_its_rectangle(self):
        self.assertTrue(self.buffer.apply_update(update())[0])
        expected=np.zeros((3,4),np.int8);expected[1,1:3]=[100,-1]
        np.testing.assert_array_equal(np.array(self.buffer.message.data).reshape(3,4),expected)
        self.assertEqual(self.buffer.version,2)
        self.assertEqual(self.original.data,[0]*12)

    def test_update_without_full_grid(self):
        self.assertFalse(CostmapBuffer().apply_update(update())[0])

    def test_wrong_frame_does_not_change_map(self):
        self.assertFalse(self.buffer.apply_update(update(frame='odom'))[0])
        self.assertEqual(self.buffer.message.data,[0]*12)

    def test_out_of_bounds(self):
        self.assertFalse(self.buffer.apply_update(update(x=4))[0])

    def test_malformed_patch(self):
        self.assertFalse(self.buffer.apply_update(update(values=(100,)))[0])

    def test_zero_stamp_supported_for_nav2_jazzy(self):
        self.assertTrue(self.buffer.apply_update(update(stamp=0))[0])
        self.assertEqual(self.buffer.message.header.stamp.sec,10)

    def test_older_timestamp_not_applied(self):
        self.assertFalse(self.buffer.apply_update(update(stamp=9))[0])

    def test_new_full_grid_replaces_previous_layout(self):
        self.buffer.apply_update(update())
        self.buffer.set_full(full(2,2,20))
        self.assertEqual(self.buffer.message.data,[0]*4)
        self.assertFalse(self.buffer.apply_update(update())[0])


class PacketTest(unittest.TestCase):
    def test_fresh_rgb_cannot_hide_stale_depth(self):
        messages=[NS(header=header(s)) for s in (10,7,10)]
        self.assertFalse(timing_valid(packet_timing(messages,10.1,1.,.04)))

    def test_each_stream_fresh_but_not_synchronized(self):
        messages=[NS(header=header(s)) for s in (10,9.7,10)]
        result=packet_timing(messages,10.1,1.,.04)
        self.assertTrue(result['all_streams_fresh'])
        self.assertFalse(result['streams_synchronized'])

    def test_valid_sync_packet(self):
        messages=[NS(header=header(s)) for s in (10,10.01,10)]
        self.assertTrue(timing_valid(packet_timing(messages,10.1,1.,.04)))

    def test_future_clock_not_current_observation(self):
        messages=[NS(header=header(15))]*3
        self.assertFalse(timing_valid(packet_timing(messages,10.1,1.,.04)))
