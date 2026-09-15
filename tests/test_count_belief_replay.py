import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "demo"))

from replay_count_belief import replay_matrix  # noqa: E402


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_replay_uses_evidence_before_ground_truth(tmp_path):
    matrix = tmp_path / "matrix"
    run = matrix / "runs" / "scene_a-q1-01"
    _write(run / "run.json", {"scene": "scene_a"})
    _write(run / "structured_logs" / "decision.json", {
        "numerical_answer": 0,
        "stop_reason": "NO_MORE_INFORMATION_GAIN",
    })
    _write(run / "structured_logs" / "candidates.json", {
        "numerical_candidates": [{
            "answer": 0,
            "count_history": [1, 1],
            "count_posterior": {"0": 1.0},
            "provenance": ["count_belief_materialized"],
            "physical_hypotheses": [],
            "count_ledger": [],
            "unseen_candidate_risk": 1.0,
        }],
    })
    truth = tmp_path / "truth.json"
    _write(truth, {"answers": {"scene_a": 1}})

    report = replay_matrix(matrix, truth)

    assert report["old_exact_count"] == 0
    assert report["new_exact_count"] == 1
    assert report["records"][0]["new_answer"] == 1
    assert report["records"][0]["legacy_pose_linkage"] == (
        "correlated_history_approximation"
    )

