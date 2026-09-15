#!/usr/bin/env python3
"""Offline contracts for real-run bundles and their wrapper."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile


SCRIPT = Path(__file__).with_name("real_run_failure_bundle.py")
SPEC = importlib.util.spec_from_file_location("failure_bundle", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def read_manifest(archive_path: Path) -> dict[str, object]:
    with tarfile.open(archive_path) as archive:
        manifest_name = next(
            name for name in archive.getnames()
            if name.endswith("artifact_manifest.json")
        )
        extracted = archive.extractfile(manifest_name)
        assert extracted is not None
        return json.load(extracted)


def collect_offline(root: Path, *, exit_code: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    command = root / "command.json"
    stdout = root / "stdout.log"
    stderr = root / "stderr.log"
    command.write_text(
        json.dumps({"command": ["demo/run_teacher_demo.sh", "q1"]}),
        encoding="utf-8",
    )
    stdout.write_text("demo output\n", encoding="utf-8")
    stderr.write_text("failure output\n", encoding="utf-8")
    output = root / "bundles"
    assert MODULE.main([
        "--offline", "--output-dir", str(output), "--scene", "test_scene",
        "--question", "How many test objects?", "--exit-code", str(exit_code),
        "--command-file", str(command), "--stdout-file", str(stdout),
        "--stderr-file", str(stderr),
    ]) == 0
    archives = list(output.glob("real_run_bundle_test_scene_*.tar.gz"))
    assert len(archives) == 1
    assert re.fullmatch(
        r"real_run_bundle_test_scene_How_many_test_objects_\d{8}T\d{6}\.tar\.gz",
        archives[0].name,
    )
    return archives[0]


def write_wrapper_fixture(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    demo = root / "demo"
    scripts = root / "scripts"
    fake_bin = root / "fake-bin"
    capture = root / "capture"
    demo.mkdir()
    scripts.mkdir()
    fake_bin.mkdir()
    capture.mkdir()
    shutil.copy2(Path(__file__).parents[1] / "demo" / "run_real_question.sh", demo)
    teacher = demo / "run_teacher_demo.sh"
    teacher.write_text(
        "#!/usr/bin/env bash\nprintf 'teacher stdout\\n'\nprintf 'teacher stderr\\n' >&2\nexit \"${TEST_EXIT_CODE:?}\"\n",
        encoding="utf-8",
    )
    teacher.chmod(0o755)
    # The wrapper invokes Python once to make actual_command.json, then once
    # for collection.  Pass the former through and capture the latter while its
    # source logs still exist.
    (fake_bin / "python3").write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "if [[ \"$1\" == \"-\" ]]; then exec \"$REAL_PYTHON\" \"$@\"; fi\n"
        "if [[ \"$1\" == */scripts/real_run_failure_bundle.py ]]; then\n"
        "  printf '%s\\n' \"$@\" >\"$CAPTURE_DIR/collector_args.txt\"\n"
        "  for ((i=1; i <= $#; i++)); do\n"
        "    if [[ \"${!i}\" == \"--stdout-file\" ]]; then j=$((i+1)); cp \"${!j}\" \"$CAPTURE_DIR/stdout.log\"; fi\n"
        "    if [[ \"${!i}\" == \"--stderr-file\" ]]; then j=$((i+1)); cp \"${!j}\" \"$CAPTURE_DIR/stderr.log\"; fi\n"
        "  done\n"
        "  touch \"$CAPTURE_DIR/collector_called\"\n"
        "  exit \"${COLLECTOR_EXIT_CODE:-0}\"\n"
        "fi\n"
        "exec \"$REAL_PYTHON\" \"$@\"\n",
        encoding="utf-8",
    )
    (fake_bin / "python3").chmod(0o755)
    return demo / "run_real_question.sh", fake_bin, capture


def run_wrapper_case(root: Path, *, episode_exit: int, always_bundle: bool,
                     collector_exit: int = 0) -> tuple[subprocess.CompletedProcess[str], Path]:
    wrapper, fake_bin, capture = write_wrapper_fixture(root)
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "CAPTURE_DIR": str(capture),
        "TEST_EXIT_CODE": str(episode_exit),
        "COLLECTOR_EXIT_CODE": str(collector_exit),
    }
    command = [str(wrapper), "q1"]
    if always_bundle:
        command.append("--always-bundle")
    return subprocess.run(command, text=True, capture_output=True, env=environment), capture


def main() -> int:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        structured = root / "structured"
        structured.mkdir()
        (structured / "events.jsonl").write_text(
            "\n".join((
                json.dumps({"event_type": "answer_published", "state": "publish"}),
                json.dumps({"event_type": "shutdown", "reason_code": "question_log_closed"}),
            )) + "\n",
            encoding="utf-8",
        )
        (structured / "07_count_snapshots.jsonl").write_text(
            json.dumps({"available": True, "fallback_reason": ""}) + "\n",
            encoding="utf-8",
        )
        (structured / "candidates.json").write_text(json.dumps({
            "numerical_candidates": [{
                "answer": 2,
                "count_posterior": {"0": 0.1, "2": 0.9},
                "unified_count_snapshot": {
                    "available": True,
                    "map_count": 4,
                    "map_probability": 0.7,
                    "posterior_over_count": {"2": 0.3, "4": 0.7},
                    "global_hypotheses": [
                        {"hypothesis_id": "q0", "weight": 0.7, "track_count": 3},
                        {"hypothesis_id": "q1", "weight": 0.3, "track_count": 4},
                    ],
                    "anchor_posterior": {"anchor-a": 0.5, "anchor-b": 0.5},
                    "per_hypothesis_observed_posterior_by_anchor": [
                        {
                            "anchor-a": {"2": 1.0},
                            "anchor-b": {"2": 1.0},
                        }
                    ],
                    "unseen_count_posterior": {"0": 0.75, "1": 0.25},
                    "unseen_targets_on_known_anchors_posterior": {
                        "0": 0.8,
                        "1": 0.2,
                    },
                    "unseen_anchors_posterior": {"0": 1.0},
                    "targets_on_unseen_anchors_posterior": {"0": 1.0},
                    "compile_quality": {
                        "total_projected_hypotheses": 2,
                        "compiled_weight_mass": 0.7,
                        "residual_weight_mass": 0.3,
                        "posterior_status": "partial_with_residual",
                    },
                    "strict_certificate": {
                        "strict": False,
                        "failed_conditions": ["unseen_mass_open"],
                    },
                },
            }],
        }), encoding="utf-8")
        for name in (
            "00_acquisitions.jsonl",
            "01_canonical_observations.jsonl",
            "03_global_hypotheses.jsonl",
        ):
            (structured / name).write_text("", encoding="utf-8")
        summary = MODULE.pipeline_summary(
            structured,
            {"demo_status": {"phase": "success", "result": {"answer": 0}}},
            exit_code=1,
            stderr_text="task committed an unexpected result",
        )
        assert summary["last_completed_stage"] == "publication"
        assert summary["failed_stage"] == "terminal_result_validation"
        assert summary["fallback_reason"] == ""
        assert summary["answer_published_event_count"] == 1
        assert summary["episode_exit_code"] == 1
        assert summary["episode_outcome"] == "failure"
        assert summary["legacy_map_count"] == 2
        assert summary["unified_map_count"] == 4
        assert summary["unified_strict"] is False
        assert summary["projected_mbm_hypothesis_count"] == 2
        assert summary["query_relevant_tracks_per_projected_hypothesis"] == [3, 4]
        assert summary["anchor_alternative_count"] == 2
        assert summary["observed_expected_count"] == 2.0
        assert summary["unseen_expected_count_by_source"] == {
            "observed_region_detector_miss": 0.2,
            "unseen_anchor_count": 0.0,
            "targets_on_unseen_anchors": 0.0,
            "total_unseen_targets": 0.25,
        }
        assert summary["compiled_weight_mass"] == 0.7
        assert summary["residual_weight_mass"] == 0.3
        assert summary["posterior_status"] == "partial_with_residual"

        success_manifest = read_manifest(collect_offline(root / "success", exit_code=0))
        failure_manifest = read_manifest(collect_offline(root / "failure", exit_code=7))
        assert success_manifest["episode_exit_code"] == 0
        assert success_manifest["episode_outcome"] == "success"
        assert failure_manifest["episode_exit_code"] == 7
        assert failure_manifest["episode_outcome"] == "failure"
        assert failure_manifest["artifacts"]["actual_command"]["status"] == "collected"
        assert failure_manifest["artifacts"]["docker_image_hashes"]["status"] == "missing"

        success, capture = run_wrapper_case(root / "wrapper_success", episode_exit=0, always_bundle=False)
        assert success.returncode == 0
        assert not (capture / "collector_called").exists()

        always_success, capture = run_wrapper_case(root / "wrapper_always_success", episode_exit=0, always_bundle=True)
        assert always_success.returncode == 0
        assert (capture / "collector_called").is_file()
        assert (capture / "stdout.log").read_text(encoding="utf-8") == "teacher stdout\n"
        assert (capture / "stderr.log").read_text(encoding="utf-8") == "teacher stderr\n"
        assert "--exit-code\n0\n" in (capture / "collector_args.txt").read_text(encoding="utf-8")

        failure, capture = run_wrapper_case(root / "wrapper_failure", episode_exit=13, always_bundle=False, collector_exit=2)
        assert failure.returncode == 13
        assert (capture / "collector_called").is_file()
        assert "--exit-code\n13\n" in (capture / "collector_args.txt").read_text(encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
