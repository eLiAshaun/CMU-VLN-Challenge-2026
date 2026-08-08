import pytest

from tools.live_model_chain import (
    OrderedSummary,
    competition_geometry_gate,
    groundingdino_rescue_required,
    qwen_target_probability,
    target_detection_count,
)


def _request():
    return {
        "question": "How many pillows are in the room?",
        "image_geometry": {"projection": "equirectangular"},
    }


def test_ordered_summary_requires_the_configured_stage_order(tmp_path):
    order = ["raw", "adapter", "mast3r"]
    summary = OrderedSummary(tmp_path, _request(), order)
    summary.record("raw", {"status": "completed"})
    summary.record("adapter", {"status": "completed"})
    summary.record("mast3r", {"status": "blocked"})
    summary.finish()

    assert summary.payload["executed_stage_order"] == order


def test_ordered_summary_rejects_stage_bypass(tmp_path):
    summary = OrderedSummary(tmp_path, _request(), ["raw", "adapter"])

    with pytest.raises(RuntimeError, match="stage_order_violation"):
        summary.record("adapter", {"status": "completed"})


def test_groundingdino_rescue_depends_on_target_hits_only():
    unrelated_hit = {
        "status": "completed",
        "detection_count": 2,
        "counts_by_class": {"chair": 2},
    }
    target_hit = {
        "status": "completed",
        "detection_count": 3,
        "counts_by_class": {"chair": 2, "pillow": 1},
    }

    assert target_detection_count(unrelated_hit, "pillow") == 0
    assert groundingdino_rescue_required(unrelated_hit, "pillow") is True
    assert groundingdino_rescue_required(target_hit, "pillow") is False
    assert groundingdino_rescue_required(target_hit, ["pillow", "chair"]) is False
    assert groundingdino_rescue_required(target_hit, ["pillow", "lamp"]) is True
    assert groundingdino_rescue_required({"status": "failed"}, "pillow") is True


def test_qwen_probability_requires_a_valid_successful_response():
    assert qwen_target_probability({"ok": True, "metadata": {"target_probability": 0.9}}) == 0.9
    assert qwen_target_probability({"ok": False, "metadata": {"target_probability": 0.9}}) is None
    assert qwen_target_probability({"ok": True, "metadata": {}}) is None
    assert qwen_target_probability({"ok": True, "metadata": {"target_probability": 1.1}}) is None


def test_competition_geometry_gate_uses_lidar_pose_and_sync(tmp_path):
    sensor = tmp_path / "sensor.npy"
    registered = tmp_path / "registered.npy"
    state = tmp_path / "state.json"
    sensor.touch()
    registered.touch()
    state.touch()
    payload = {
        "sensor_scan_path": str(sensor),
        "sensor_scan_age_seconds": 0.1,
        "sensor_scan_point_count": 10,
        "sensor_scan_frame": "sensor_at_scan",
        "registered_scan_path": str(registered),
        "registered_scan_age_seconds": 0.1,
        "registered_scan_point_count": 10,
        "registered_scan_frame": "map",
        "state_estimation_path": str(state),
        "state_estimation_age_seconds": 0.02,
        "camera_sensor_scan_offset_seconds": 0.05,
    }
    config = {
        "sensor_scan_required": True,
        "registered_scan_required": True,
        "state_estimation_required": True,
        "max_age_seconds": 1.0,
        "max_camera_lidar_offset_seconds": 0.25,
    }

    assert competition_geometry_gate(payload, config) == (True, [])
    payload["camera_sensor_scan_offset_seconds"] = 0.5
    ready, reasons = competition_geometry_gate(payload, config)
    assert ready is False
    assert reasons == ["camera_sensor_scan_out_of_sync"]
