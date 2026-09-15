import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/make_scene_memory_replay_fixture.py"
)
SPEC = importlib.util.spec_from_file_location(
    "make_scene_memory_replay_fixture", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_nearest_entry_is_stable_for_equidistant_messages():
    entries = ((2, 110), (1, 90), (0, 130))
    assert MODULE.nearest_entry(entries, 100) == (1, 90)


def test_bounded_window_is_chronological_and_bounded():
    entries = tuple((index, index * 10) for index in range(10))
    assert MODULE.bounded_window(
        entries, 50, radius_ns=30, maximum=3
    ) == ((4, 40), (5, 50), (6, 60))


def test_selection_helpers_reject_invalid_bounds():
    with pytest.raises(ValueError, match="empty"):
        MODULE.nearest_entry((), 10)
    with pytest.raises(ValueError, match="nonnegative"):
        MODULE.bounded_window(((0, 1),), 1, radius_ns=-1, maximum=1)
