import json
import sys

import cv2
import numpy as np

from integrations.mast3r.local_object_lift import main


def test_verified_mask_is_lifted_into_local_mast3r_frame(tmp_path, monkeypatch):
    height, width = 4, 5
    points = np.zeros((height, width, 3), dtype=np.float32)
    points[..., 0] = np.arange(width, dtype=np.float32)
    points[..., 1] = np.arange(height, dtype=np.float32)[:, None]
    points[..., 2] = 2.0
    confidence = np.full((height, width), 8.0, dtype=np.float32)
    pointmap_path = tmp_path / "pointmap.npz"
    np.savez_compressed(pointmap_path, points_panorama=points, confidence=confidence)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[1:3, 2:5] = 255
    mask_path = tmp_path / "mask.png"
    assert cv2.imwrite(str(mask_path), mask)
    panorama_mask = np.full((8, 16), 255, dtype=np.uint8)
    panorama_mask_path = tmp_path / "panorama_mask.png"
    assert cv2.imwrite(str(panorama_mask_path), panorama_mask)

    observations_path = tmp_path / "verified.json"
    observations_path.write_text(json.dumps([{
        "observation_id": "obs_1",
        "canonical_class": "pillow",
        "representative_view_id": "view_1",
        "representative_mask_path": str(mask_path),
        "panorama_mask_path": str(panorama_mask_path),
    }]))
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps({
        "frame": "panorama_optical_center",
        "optical_center_group": "same-centre",
        "views": [{"view_id": "view_1", "pointmap_path": str(pointmap_path)}],
    }))
    output = tmp_path / "output"
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({
        "configured": True,
        "source_to_camera": np.eye(4).tolist(),
        "adapter_to_camera_canonical": np.eye(4).tolist(),
    }))
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "frame_id": "map",
        "position_xyz": [1.0, 2.0, 3.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }))
    sensor_scan_path = tmp_path / "sensor_scan.npy"
    registered_scan_path = tmp_path / "registered_scan.npy"
    np.save(sensor_scan_path, np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32))
    np.save(registered_scan_path, points.reshape(-1, 3) + [1.0, 2.0, 3.0])
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps({
        "output_dir": str(output),
        "verified_observations_path": str(observations_path),
        "geometry_manifest_path": str(geometry_path),
        "confidence_threshold": 5.0,
        "max_saved_points_per_observation": 100,
        "calibration_path": str(calibration_path),
        "state_estimation_path": str(state_path),
        "sensor_scan_path": str(sensor_scan_path),
        "registered_scan_path": str(registered_scan_path),
        "panorama_vertical_fov_deg": 120.0,
    }))
    monkeypatch.setattr(sys, "argv", ["local_object_lift", "--request", str(request_path)])

    assert main() == 0
    response = json.loads((output / "worker_response.json").read_text())
    lifted = json.loads((output / "local_3d_observations.json").read_text())
    assert response["lifted_observation_count"] == 1
    assert lifted[0]["point_count"] == 6
    assert lifted[0]["frame"] == "panorama_optical_center"
    assert lifted[0]["world_aligned"] is True
    assert lifted[0]["persistent_instance_authorized"] is False
