"""Utility functions for validation and geometry operations.

This package provides helper functions for validation, JSON I/O,
and geometric computations.
"""

from .validation import (
    write_json,
    validate_episode_request,
    competition_geometry_gate,
)

from .geometry_utils import union_normalized_boxes

__all__ = [
    # Validation
    "write_json",
    "validate_episode_request",
    "competition_geometry_gate",
    # Geometry
    "union_normalized_boxes",
]
