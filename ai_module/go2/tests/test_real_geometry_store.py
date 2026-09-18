"""Real geometry -> real ObjectStore tests with explicitly supplied labels/masks.

These are synthetic geometry regressions, NOT tests of the VLM/detector.
"""
import unittest
import numpy as np
from go2.sensors import RGBDProjector, PinholeObservationAdapter
from rebuild.geometry import lift_detection
from rebuild.contracts import Detection, VerifiedObservation
from rebuild.object_store import ObjectStore


class GeometryStoreTest(unittest.TestCase):
    def observations(self, stamp, camera_x=0.):
        # Two fixed front-facing surfaces. Project world X into the moving camera.
        h,w=48,80
        K=np.array([[40.,0,39.5],[0,40.,23.5],[0,0,1.]])
        rgb=np.full((h,w,3),100,np.uint8)
        depth=np.full((h,w),4.,np.float32)
        boxes=[]
        for world_x in (-1.,1.):
            center=39.5+(world_x-camera_x)*40/2
            box=(round(center-7.5),8,round(center+7.5),38)
            boxes.append(box)
            depth[box[1]:box[3],box[0]:box[2]]=2.
        T=np.eye(4);T[0,3]=camera_x
        adapter=PinholeObservationAdapter()
        frame=RGBDProjector(stride=1).frame(str(stamp),float(stamp),rgb,depth,K,[], '',T,T,float(stamp))
        view=adapter.make_views(frame)[0]
        result=[]
        for box in boxes:
            mask=np.zeros((h,w),np.uint8);mask[box[1]:box[3],box[0]:box[2]]=1
            det=Detection(str(stamp),float(stamp),'front','chair',list(box),mask,.9)
            obs=lift_detection(det,view,frame.registered_points_map)
            adapter.annotate_observation(obs,det)
            result.append(VerifiedObservation(**vars(obs),category_evidence={
                'verdict':'yes','observation_id':str(stamp),'view_id':'front'}))
        return result

    def test_two_similar_instances_remain_two(self):
        store=ObjectStore({})
        ids=store.update(self.observations(1))
        self.assertNotEqual(ids[0],ids[1])
        self.assertEqual(len(store.records),2)

    def test_repeat_view_does_not_repeat_count(self):
        store=ObjectStore({})
        ids=store.update(self.observations(1))
        for i in range(2,6):
            self.assertEqual(store.update(self.observations(i)),ids)
        self.assertEqual(len(store.records),2)

    def test_translated_camera_preserves_world_identity(self):
        store=ObjectStore({})
        ids=store.update(self.observations(1))
        self.assertEqual(store.update(self.observations(2,.2)),ids)
        self.assertEqual(len(store.records),2)
        centers=[record.center[0] for record in store.records.values()]
        np.testing.assert_allclose(sorted(centers),[-1.,1.],atol=.08)
