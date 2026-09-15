from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_repository_hygiene_guard_passes_for_tracked_tree():
    subprocess.run(
        ["bash", "scripts/check_repository_hygiene.sh"],
        cwd=ROOT,
        check=True,
    )


def test_hygiene_workflow_invokes_the_guard():
    workflow = (
        ROOT / ".github" / "workflows" / "repository-hygiene.yml"
    ).read_text(encoding="utf-8")
    assert "scripts/check_repository_hygiene.sh" in workflow
