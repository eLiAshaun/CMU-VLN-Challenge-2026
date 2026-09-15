#!/usr/bin/env python3
"""Validate the A3 detector-recall calibration contract.

The A3 contract (docs/closure_program/INTERFACE_CONTRACTS.md) requires that
the priority categories chair/table/vase each have at least one *valid
calibration cell* — a condition cell whose sample_count reaches
``conditioning.minimum_samples_per_cell``.  Without such a cell the runtime
falls back to adjacent cells / category pooling and the conditional recall is
not directly supported.

Inputs accepted:
  - a DetectorRecallCalibration JSON (schema v3): categories.<cat>.conditional_stats[]
  - a detector_recall_records.json (instance-level records with
    ground_truth_conditions): used to preview the cells that a future
    artifact built from these records would contain

Exit code 0 = contract satisfied; 1 = not satisfied (usable as a gate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PRIORITY_CATEGORIES = ("chair", "table", "vase")

RANGE_BIN_EDGES_M = (1.5, 3.0, 5.0, float("inf"))
PIXEL_AREA_BIN_EDGES = (64.0, 256.0, 1024.0, 4096.0, float("inf"))
OCCLUSION_BIN_EDGES = (0.2, 0.5, 0.8, 1.0)


def _bin_index(value: float, edges: tuple) -> int:
    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


def valid_cell_count(
    conditional_stats: list[dict[str, Any]], minimum_samples: int
) -> int:
    """Count cells whose sample_count meets the minimum."""
    return sum(
        1
        for cell in conditional_stats
        if int(cell.get("sample_count", 0)) >= minimum_samples
    )


def _records_to_cell_counts(
    records: list[dict[str, Any]],
) -> dict[str, dict[tuple[int, int, int, bool], int]]:
    """Aggregate instance records into per-category condition-cell counts."""
    cell_counts: dict[str, dict[tuple[int, int, int, bool], int]] = {}
    for record in records:
        category = str(record.get("category", ""))
        conditions = record.get("ground_truth_conditions", {})
        gt_ids = set(record.get("ground_truth_instance_ids", []))
        for object_id, cond in conditions.items():
            if object_id not in gt_ids:
                continue
            key = (
                _bin_index(float(cond["range_m"]), RANGE_BIN_EDGES_M),
                _bin_index(float(cond["projected_pixel_area"]), PIXEL_AREA_BIN_EDGES),
                _bin_index(float(cond["occlusion_fraction"]), OCCLUSION_BIN_EDGES),
                bool(cond.get("truncated", False)),
            )
            bucket = cell_counts.setdefault(category, {})
            bucket[key] = bucket.get(key, 0) + 1
    return cell_counts


def validate_calibration_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a DetectorRecallCalibration payload.

    Returns a report dict with per-category valid-cell counts and the overall
    contract verdict.
    """
    conditioning = payload.get("conditioning", {})
    minimum_samples = int(conditioning.get("minimum_samples_per_cell", 20))
    categories = payload.get("categories", {})

    per_category = {}
    for name in PRIORITY_CATEGORIES:
        stats = categories.get(name)
        if stats is None:
            per_category[name] = {
                "present": False,
                "sample_count": 0,
                "valid_cells": 0,
                "densest_cell_samples": 0,
                "satisfied": False,
            }
            continue
        cells = stats.get("conditional_stats", [])
        valid = valid_cell_count(cells, minimum_samples)
        densest = max((int(c.get("sample_count", 0)) for c in cells), default=0)
        per_category[name] = {
            "present": True,
            "sample_count": int(stats.get("sample_count", 0)),
            "valid_cells": valid,
            "densest_cell_samples": densest,
            "satisfied": valid >= 1,
        }

    satisfied = all(entry["satisfied"] for entry in per_category.values())
    return {
        "contract": "chair/table/vase each have >=1 valid calibration cell",
        "minimum_samples_per_cell": minimum_samples,
        "satisfied": satisfied,
        "categories": per_category,
    }


def validate_records_json(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Preview cell sufficiency from instance-level recall records."""
    minimum_samples = 20
    cell_counts = _records_to_cell_counts(records)
    per_category = {}
    for name in PRIORITY_CATEGORIES:
        cells = cell_counts.get(name, {})
        valid = sum(1 for count in cells.values() if count >= minimum_samples)
        densest = max(cells.values(), default=0)
        per_category[name] = {
            "sample_count": sum(cells.values()),
            "valid_cells": valid,
            "densest_cell_samples": densest,
            "satisfied": valid >= 1,
        }
    satisfied = all(entry["satisfied"] for entry in per_category.values())
    return {
        "contract": "chair/table/vase each have >=1 valid calibration cell",
        "minimum_samples_per_cell": minimum_samples,
        "satisfied": satisfied,
        "categories": per_category,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"Contract: {report['contract']}")
    print(f"minimum_samples_per_cell: {report['minimum_samples_per_cell']}")
    print(f"\n{'类别':<10}{'样本':>8}{'有效cell':>10}{'最密cell':>10}  契约")
    for name, entry in report["categories"].items():
        print(
            f"{name:<10}{entry['sample_count']:>8}{entry['valid_cells']:>10}"
            f"{entry['densest_cell_samples']:>10}  "
            f"{'满足' if entry['satisfied'] else '缺口'}"
        )
    print(f"\n总体: {'满足' if report['satisfied'] else '不满足'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Calibration JSON or records JSON")
    args = parser.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        report = validate_records_json(payload)
    elif "categories" in payload:
        report = validate_calibration_json(payload)
    else:
        print(f"无法识别的输入格式: {args.path}", file=sys.stderr)
        return 2

    _print_report(report)
    return 0 if report["satisfied"] else 1


if __name__ == "__main__":
    sys.exit(main())
