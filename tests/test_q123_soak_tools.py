"""Tests for the Q1/Q2/Q3 cold-start acceptance and recording tools."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

import cv2
import numpy as np

from demo.live_demo_recorder import main as recorder_main
from demo.live_demo_recorder import render_dashboard
from demo.soak_runner import (
    EXPECTED_RESULTS,
    TASK_QUESTIONS,
    build_summary,
    choose_terminal_status,
    classify_run,
    cold_start_is_verified,
    derive_event_evidence,
    render_run_summary,
    render_summary,
)


def passing_event_evidence() -> dict:
    return {
        "answer_publish_count": 1,
        "reset_verified": True,
        "episode_duration_s": 12.5,
        "forbidden_backend_calls": [],
        "failed_guards": [],
        "probe_events": 0,
    }


class EvidenceDerivationTest(unittest.TestCase):
    def test_soak_contract_matches_authoritative_teacher_entrypoint(self):
        source = (
            Path(__file__).resolve().parents[1] / "demo" / "run_teacher_demo.sh"
        ).read_text(encoding="utf-8")
        for task in ("q1", "q2", "q3"):
            self.assertIn(TASK_QUESTIONS[task], source)
            self.assertIn(f'EXPECTED_RESULT="{EXPECTED_RESULTS[task]}"', source)

    def test_soak_forwards_the_shared_viewer_contract(self):
        source = (
            Path(__file__).resolve().parents[1] / "demo" / "soak_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--viewer-port", str(self.config.viewer_port)', source)
        self.assertIn('command.extend(("--public-url", self.config.public_url))', source)
        self.assertIn('default=ENV_STATUS_URL', source)
        self.assertIn('"--status-url", self.config.status_url', source)

    def test_event_evidence_preserves_guards_and_strict_acceptance_fields(self):
        evidence = derive_event_evidence(
            [
                {
                    "event_type": "episode_reset",
                    "reset_verified": True,
                    "timestamp_wall": 100.0,
                },
                {"event_type": "question_received", "timestamp_wall": 101.0},
                {
                    "event_type": "decision",
                    "action": "PROBE",
                    "failed_guards": ["relation_evidence_insufficient"],
                },
                {
                    "event_type": "answer_published",
                    "timestamp_wall": 109.0,
                },
            ]
        )
        self.assertEqual(evidence["answer_publish_count"], 1)
        self.assertEqual(evidence["probe_events"], 1)
        self.assertEqual(evidence["episode_duration_s"], 8.0)
        self.assertTrue(evidence["reset_verified"])
        self.assertEqual(evidence["failed_guards"], ["relation_evidence_insufficient"])

    def test_forbidden_backend_is_never_hidden_by_success_status(self):
        evidence = passing_event_evidence()
        evidence["forbidden_backend_calls"] = ["oracle"]
        result = classify_run(
            task="q1",
            exit_code=0,
            timed_out=False,
            wall_duration_s=12.5,
            cold_start_verified=True,
            terminal_status={
                "phase": "success",
                "result": "2",
                "result_details": {"kind": "numerical"},
            },
            event_evidence=evidence,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "forbidden_backend_call")

    def test_success_requires_result_type_reset_budget_and_exactly_once(self):
        result = classify_run(
            task="q3",
            exit_code=0,
            timed_out=False,
            wall_duration_s=12.5,
            cold_start_verified=True,
            terminal_status={
                "phase": "success",
                "result": "vase",
                "result_details": {"kind": "object_reference"},
            },
            event_evidence=passing_event_evidence(),
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "passed")

    def test_v2_typed_result_is_used_for_acceptance(self):
        result = classify_run(
            task="q1",
            exit_code=0,
            timed_out=False,
            wall_duration_s=12.5,
            cold_start_verified=True,
            terminal_status={
                "schema_version": "scnav_demo_status_v2",
                "phase": "success",
                "result": {"kind": "numerical", "answer": 2, "label": "pillow"},
            },
            event_evidence=passing_event_evidence(),
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], 2)
        self.assertEqual(result["result_kind"], "numerical")

    def test_latest_terminal_status_wins(self):
        status = choose_terminal_status(
            [
                {"demo_status": {"phase": "probe", "probes_used": 1}},
                {"demo_status": {"phase": "success", "result": "chair"}},
                {"demo_status": {"phase": "ready"}},
            ]
        )
        self.assertEqual(status["phase"], "success")

    def test_previous_cold_start_terminal_status_is_not_reused(self):
        status = choose_terminal_status(
            [
                {
                    "demo_status": {
                        "phase": "success",
                        "question": "old question",
                        "result": "wrong run",
                    }
                },
                {
                    "demo_status": {
                        "phase": "success",
                        "question": "current question",
                        "result": "2",
                    }
                },
            ],
            question="current question",
        )
        self.assertEqual(status["result"], "2")

    def test_same_question_old_viewer_terminal_status_is_not_reused(self):
        question = "How many black pillows are on the sofa?"
        status = choose_terminal_status(
            [
                {
                    "_viewer_container_id": "old",
                    "demo_status": {
                        "phase": "success",
                        "question": question,
                        "result": "99",
                    },
                },
                {
                    "_viewer_container_id": "new",
                    "demo_status": {
                        "phase": "success",
                        "question": question,
                        "result": "2",
                    },
                },
            ],
            question=question,
            viewer_container_id="new",
        )
        self.assertEqual(status["result"], "2")

    def test_cold_start_requires_new_recent_identity_for_every_container(self):
        before = {
            name: {"id": f"old-{name}", "started_at": "2026-07-15T00:00:00Z"}
            for name in (
                "iros2026_ai_module",
                "iros2026_system",
                "iros2026_web_viewer",
            )
        }
        after = {
            name: {"id": f"new-{name}", "started_at": "2026-07-15T01:00:01Z"}
            for name in before
        }
        self.assertTrue(
            cold_start_is_verified(
                before, after, run_started_at="2026-07-15T01:00:00+00:00"
            )
        )
        after["iros2026_system"] = before["iros2026_system"]
        self.assertFalse(
            cold_start_is_verified(
                before, after, run_started_at="2026-07-15T01:00:00+00:00"
            )
        )

    def test_wall_time_over_600_seconds_fails_even_if_episode_is_fast(self):
        result = classify_run(
            task="q1",
            exit_code=0,
            timed_out=False,
            wall_duration_s=600.01,
            cold_start_verified=True,
            terminal_status={
                "phase": "success",
                "result": "2",
                "result_details": {"kind": "numerical"},
            },
            event_evidence=passing_event_evidence(),
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "wall_budget_exceeded")


class SummaryGateTest(unittest.TestCase):
    def test_gate_is_exactly_thirty_of_thirty(self):
        records = [
            {
                "task": task,
                "run": run,
                "success": True,
                "status": "passed",
                "episode_duration_s": 10.0 + run,
                "probes": run % 2,
                "failed_guards": [],
                "forbidden_backend_calls": [],
                "recorder": {"export_complete": True},
            }
            for task in ("q1", "q2", "q3")
            for run in range(1, 11)
        ]
        summary = build_summary(records, tasks=("q1", "q2", "q3"), repeats=10)
        self.assertTrue(summary["gate_30_of_30"])
        self.assertEqual(summary["passed_total"], 30)
        self.assertIn("30/30 gate: **PASS**", render_summary(summary))

        records[-1]["success"] = False
        records[-1]["status"] = "failed"
        summary = build_summary(records, tasks=("q1", "q2", "q3"), repeats=10)
        self.assertFalse(summary["gate_30_of_30"])

    def test_record_all_requires_every_video_export_but_does_not_relabel_results(self):
        records = [
            {
                "task": task,
                "run": run,
                "success": True,
                "status": "passed",
                "episode_duration_s": 2.0,
                "probes": 0,
                "recorder": {"export_complete": run != 10 or task != "q3"},
            }
            for task in ("q1", "q2", "q3")
            for run in range(1, 11)
        ]
        summary = build_summary(
            records,
            tasks=("q1", "q2", "q3"),
            repeats=10,
            record_policy="all",
        )
        self.assertTrue(summary["gate_30_of_30"])
        self.assertFalse(summary["recording_export_gate"])
        self.assertFalse(summary["overall_gate"])

    def test_smaller_debug_matrix_cannot_claim_thirty_run_gate(self):
        records = [
            {
                "task": task,
                "run": 1,
                "success": True,
                "status": "passed",
                "episode_duration_s": 1.0,
                "probes": 0,
            }
            for task in ("q1", "q2", "q3")
        ]
        summary = build_summary(records, tasks=("q1", "q2", "q3"), repeats=1)
        self.assertFalse(summary["gate_30_of_30"])
        self.assertTrue(summary["configured_matrix_gate"])

    def test_per_run_summary_exposes_failure_evidence(self):
        text = render_run_summary(
            {
                "task": "q3",
                "run": 4,
                "status": "failed",
                "exit_code": 1,
                "expected_result": "vase",
                "result": None,
                "duration_s": 30.0,
                "episode_duration_s": None,
                "probes": 2,
                "answer_publish_count": 0,
                "reset_verified": True,
                "failed_guards": ["relation_evidence_insufficient"],
                "forbidden_backend_calls": [],
            }
        )
        self.assertIn("relation_evidence_insufficient", text)
        self.assertIn("never silently removed", text)


class _ViewerHandler(BaseHTTPRequestHandler):
    jpeg: bytes = b""

    def log_message(self, _format, *_args):
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/status.json":
            body = json.dumps(
                {
                    "frame_count": 1,
                    "frame_age_sec": 0.01,
                    "scan_age_sec": 0.01,
                    "state_age_sec": 0.01,
                    "path_points": [[0.0, 0.0], [1.0, 1.0]],
                    "last_state": {
                        "position": {"x": 1.0, "y": 1.0},
                        "yaw": 0.0,
                    },
                    "demo_status": {
                        "phase": "success",
                        "question": "How many black pillows are on the sofa?",
                        "result": "2",
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        elif self.path == "/snapshot_front.jpg":
            body = self.jpeg
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
        else:
            body = b"not found"
            self.send_response(404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class RecorderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = np.full((180, 320, 3), (30, 90, 60), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", source)
        if not ok:
            raise AssertionError("OpenCV JPEG encoder unavailable")
        _ViewerHandler.jpeg = encoded.tobytes()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _ViewerHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_rendered_dashboard_has_requested_fixed_shape(self):
        result = render_dashboard(
            {"frame_age_sec": 0.1, "demo_status": {"phase": "probe"}},
            np.zeros((180, 320, 3), dtype=np.uint8),
            width=640,
            height=360,
        )
        self.assertEqual(result.shape, (360, 640, 3))

    def test_recorder_exports_video_and_refresh_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "viewer.mp4"
            metadata = Path(temp) / "metadata.json"
            base = f"http://127.0.0.1:{self.server.server_port}"
            code = recorder_main(
                [
                    "--status-url",
                    f"{base}/status.json",
                    "--output",
                    str(output),
                    "--metadata",
                    str(metadata),
                    "--width",
                    "320",
                    "--height",
                    "180",
                    "--fps",
                    "5",
                    "--source-refresh-hz",
                    "5",
                    "--source-id",
                    "viewer-test-id",
                    "--expected-question",
                    "How many black pillows are on the sofa?",
                    "--max-duration",
                    "0.45",
                ]
            )
            self.assertEqual(code, 0)
            self.assertGreater(output.stat().st_size, 0)
            capture = cv2.VideoCapture(str(output))
            opened = capture.isOpened()
            decoded, frame = capture.read()
            capture.release()
            self.assertTrue(opened)
            self.assertTrue(decoded)
            self.assertEqual(frame.shape[:2], (180, 320))
            values = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertTrue(values["complete"])
            self.assertTrue(values["question_bound"])
            self.assertEqual(values["source_id"], "viewer-test-id")
            self.assertGreater(values["frames_written"], 0)
            self.assertGreater(values["unique_source_frames"], 0)


if __name__ == "__main__":
    unittest.main()
