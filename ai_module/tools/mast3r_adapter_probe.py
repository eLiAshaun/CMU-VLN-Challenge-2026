#!/usr/bin/env python3
"""Validate live panorama projection with MASt3R adjacent-view matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mast3r.utils.path_to_dust3r  # noqa: F401
from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.model import AsymmetricMASt3R
from tools.cmu_perspective_baseline import (
    draw_matches,
    matched_ray_angular_errors_deg,
    valid_matches,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    views = list(manifest["views"])
    if len(views) < 2:
        raise ValueError("adapter_probe_requires_at_least_two_views")
    if len({item["optical_center_group"] for item in views}) != 1:
        raise ValueError("adapter_probe_views_do_not_share_one_optical_center")
    args.output.mkdir(parents=True, exist_ok=True)

    images = load_images([item["image_path"] for item in views], size=512, verbose=False)
    for index, image in enumerate(images):
        image["idx"] = index
        image["instance"] = views[index]["view_id"]
    pairs = [(images[index], images[(index + 1) % len(images)]) for index in range(len(images))]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_started = time.perf_counter()
    model = AsymmetricMASt3R.from_pretrained(str(args.weights)).to(args.device).eval()
    model_load_seconds = time.perf_counter() - model_started
    inference_started = time.perf_counter()
    output = inference(pairs, model, args.device, batch_size=1, verbose=False)
    inference_seconds = time.perf_counter() - inference_started

    pointmaps_dir = args.output / "pointmaps"
    pointmaps_dir.mkdir(exist_ok=True)
    geometry_records = []
    for index, view in enumerate(views):
        points_camera = output["pred1"]["pts3d"][index].detach().cpu().numpy()
        confidence_forward = output["pred1"]["conf"][index].detach().cpu().numpy()
        previous_index = (index - 1) % len(views)
        points_from_previous_camera = (
            output["pred2"]["pts3d_in_other_view"][previous_index]
            .detach().cpu().numpy()
        )
        confidence_reverse = (
            output["pred2"]["conf"][previous_index].detach().cpu().numpy()
        )
        expected_shape = (int(view["height"]), int(view["width"]))
        if (
            points_camera.shape[:2] != expected_shape
            or confidence_forward.shape[:2] != expected_shape
            or points_from_previous_camera.shape[:2] != expected_shape
            or confidence_reverse.shape[:2] != expected_shape
        ):
            raise ValueError(f"mast3r_pointmap_shape_changed:{view['view_id']}")
        rotation = np.asarray(view["R_panorama_from_camera"], dtype=np.float32)
        previous_rotation = np.asarray(
            views[previous_index]["R_panorama_from_camera"], dtype=np.float32
        )
        points_forward = points_camera.astype(np.float32) @ rotation.T
        points_reverse = (
            points_from_previous_camera.astype(np.float32) @ previous_rotation.T
        )
        use_reverse = confidence_reverse > confidence_forward
        points_panorama = np.where(use_reverse[..., None], points_reverse, points_forward)
        confidence = np.maximum(confidence_forward, confidence_reverse)
        pointmap_path = pointmaps_dir / f"{view['view_id']}.npz"
        np.savez_compressed(
            pointmap_path,
            points_panorama=points_panorama,
            confidence=confidence.astype(np.float32),
            selected_reverse=use_reverse,
        )
        geometry_records.append({
            "view_id": view["view_id"],
            "pointmap_path": str(pointmap_path.resolve()),
            "width": int(view["width"]),
            "height": int(view["height"]),
            "frame": "panorama_optical_center",
            "world_aligned": False,
            "optical_center_group": view["optical_center_group"],
            "bidirectional_pair_fusion": "per_pixel_max_confidence",
        })
    geometry_manifest_path = args.output / "geometry_manifest.json"
    geometry_manifest_path.write_text(json.dumps({
        "schema_version": "1.0",
        "frame": "panorama_optical_center",
        "world_aligned": False,
        "pointmap_fusion": "two_adjacent_pair_directions_per_pixel_max_confidence",
        "optical_center_group": views[0]["optical_center_group"],
        "views": geometry_records,
    }, indent=2) + "\n", encoding="utf-8")

    records = []
    for index, (first, second) in enumerate(
        zip(views, [*views[1:], views[0]])
    ):
        matches1, matches2 = valid_matches(
            output["pred1"]["desc"][index].detach(),
            output["pred2"]["desc"][index].detach(),
            output["view1"]["true_shape"][index],
            output["view2"]["true_shape"][index],
            args.device,
        )
        intrinsics = np.asarray(first["K"], dtype=np.float64)
        angular_errors = matched_ray_angular_errors_deg(
            matches1,
            matches2,
            np.radians(float(first["yaw_deg"])),
            np.radians(float(second["yaw_deg"])),
            intrinsics,
        )
        record = {
            "view1": first["view_id"],
            "view2": second["view_id"],
            "seam_pair": index == len(views) - 1,
            "valid_matches": int(len(matches1)),
            "matched_ray_error_deg_median": float(np.median(angular_errors)),
            "matched_ray_error_deg_p90": float(np.percentile(angular_errors, 90)),
        }
        records.append(record)
        draw_matches(
            Path(first["image_path"]),
            Path(second["image_path"]),
            matches1,
            matches2,
            args.output / f"pair_{index:02d}_{record['valid_matches']}_matches.jpg",
        )

    counts = np.asarray([item["valid_matches"] for item in records])
    medians = np.asarray([item["matched_ray_error_deg_median"] for item in records])
    p90s = np.asarray([item["matched_ray_error_deg_p90"] for item in records])
    report = {
        "schema_version": "1.0",
        "manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.weights.resolve()),
        "optical_center_group": views[0]["optical_center_group"],
        "views_are_independent_spatial_evidence": False,
        "geometry": {
            "manifest_path": str(geometry_manifest_path.resolve()),
            "frame": "panorama_optical_center",
            "world_aligned": False,
            "view_count": len(geometry_records),
        },
        "runtime": {
            "device": args.device,
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
            "peak_cuda_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
        },
        "results": {
            "pair_count": len(records),
            "valid_matches_min": int(counts.min()),
            "valid_matches_median": float(np.median(counts)),
            "valid_matches_max": int(counts.max()),
            "all_pairs_have_matches": bool(np.all(counts > 0)),
            "matched_ray_error_deg_median_across_pairs": float(np.median(medians)),
            "matched_ray_error_deg_worst_pair_p90": float(p90s.max()),
            "seam_pair_valid_matches": records[-1]["valid_matches"],
        },
        "pairs": records,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
