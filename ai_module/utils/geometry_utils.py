#!/usr/bin/env python3
"""Geometry utility functions for bounding box operations."""

from __future__ import annotations


def union_normalized_boxes(values: list[list[float]]) -> list[float]:
    """Return the smallest normalized image box containing every input box.

    Args:
        values: List of bounding boxes in [x1, y1, x2, y2] format

    Returns:
        Bounding box [x1, y1, x2, y2] that contains all input boxes
    """
    return [
        min(float(value[index]) for value in values)
        if index < 2
        else max(float(value[index]) for value in values)
        for index in range(4)
    ]
