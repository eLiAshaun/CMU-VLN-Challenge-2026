#!/usr/bin/env python3
"""Validate MASt3R on perspective views projected from CMU-VLN panoramas.

The simulator panorama is a central, cropped equirectangular image.  This tool
projects it to distortion-free pinhole views, compensates the robot yaw when
pairing consecutive frames, runs one MASt3R model instance, and writes a small
evidence bundle with views, match visualizations, and numeric metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mast3r.utils.path_to_dust3r  # noqa: F401
from dust3r.inference import inference
from dust3r.utils.image import load_images
from integrations.mast3r.panorama_adapter import project_equirectangular as project_panorama_view
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def project_equirectangular(
    panorama_bgr: np.ndarray,
    yaw_rad: float,
    width: int = 512,
    height: int = 384,
    horizontal_fov_deg: float = 90.0,
    panorama_vertical_fov_deg: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper around the live calibrated adapter."""
    projected = project_panorama_view(
        panorama_bgr,
        yaw_deg=math.degrees(yaw_rad),
        width=width,
        height=height,
        horizontal_fov_deg=horizontal_fov_deg,
        panorama_vertical_fov_deg=panorama_vertical_fov_deg,
    )
    return projected.image_bgr, projected.intrinsics


def nearest_view_index(angle: float, view_yaws: list[float]) -> int:
    errors = [abs(wrap_pi(candidate - angle)) for candidate in view_yaws]
    return int(np.argmin(errors))


def matched_ray_angular_errors_deg(
    matches1: np.ndarray,
    matches2: np.ndarray,
    yaw1: float,
    yaw2: float,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Angular mismatch after rotating two same-center virtual views to panorama coordinates."""
    if not len(matches1):
        return np.empty(0, dtype=np.float64)

    def rays(matches: np.ndarray, yaw: float) -> np.ndarray:
        x = (matches[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0]
        y = (matches[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1]
        z = np.ones_like(x)
        result = np.stack((x, y, z), axis=-1).astype(np.float64)
        result /= np.linalg.norm(result, axis=-1, keepdims=True)
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return result @ rotation.T

    dots = np.sum(rays(matches1, yaw1) * rays(matches2, yaw2), axis=-1)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def valid_matches(desc1: torch.Tensor, desc2: torch.Tensor, shape1, shape2, device: str):
    matches1, matches2 = fast_reciprocal_NNs(
        desc1,
        desc2,
        subsample_or_initxy1=8,
        device=device,
        dist="dot",
        block_size=2**13,
    )
    h1, w1 = map(int, shape1)
    h2, w2 = map(int, shape2)
    keep = (
        (matches1[:, 0] >= 3)
        & (matches1[:, 0] < w1 - 3)
        & (matches1[:, 1] >= 3)
        & (matches1[:, 1] < h1 - 3)
        & (matches2[:, 0] >= 3)
        & (matches2[:, 0] < w2 - 3)
        & (matches2[:, 1] >= 3)
        & (matches2[:, 1] < h2 - 3)
    )
    return matches1[keep], matches2[keep]


def draw_matches(path1: Path, path2: Path, matches1: np.ndarray, matches2: np.ndarray, output: Path):
    image1 = cv2.imread(str(path1), cv2.IMREAD_COLOR)
    image2 = cv2.imread(str(path2), cv2.IMREAD_COLOR)
    canvas = np.concatenate((image1, image2), axis=1)
    if len(matches1):
        selected = np.linspace(0, len(matches1) - 1, min(80, len(matches1))).round().astype(int)
        rng = np.random.default_rng(0)
        for index in selected:
            color = tuple(int(x) for x in rng.integers(48, 256, size=3))
            p1 = tuple(np.rint(matches1[index]).astype(int))
            p2_array = np.rint(matches2[index]).astype(int)
            p2 = (int(p2_array[0] + image1.shape[1]), int(p2_array[1]))
            cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, p2, 2, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def make_montage(paths: list[Path], output: Path):
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    labeled = []
    for index, image in enumerate(images):
        item = image.copy()
        cv2.putText(item, f"yaw={index * 45:03d} deg", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(item, f"yaw={index * 45:03d} deg", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        labeled.append(item)
    rows = [np.concatenate(labeled[i : i + 4], axis=1) for i in range(0, len(labeled), 4)]
    cv2.imwrite(str(output), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 90])


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    finite = torch.isfinite(tensor)
    values = tensor[finite]
    return {
        "finite_fraction": float(finite.float().mean()),
        "min": float(values.min()) if values.numel() else math.nan,
        "median": float(values.median()) if values.numel() else math.nan,
        "max": float(values.max()) if values.numel() else math.nan,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True, help="Directory containing frame_*.jpg panoramas")
    parser.add_argument("--viewpoints", type=Path, required=True, help="viewpoint_memory.json with pose_map=[x,y,yaw]")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    views_dir = args.output / "views"
    matches_dir = args.output / "matches"
    views_dir.mkdir(exist_ok=True)
    matches_dir.mkdir(exist_ok=True)

    frame_paths = sorted(args.frames.glob("frame_*.jpg"))
    viewpoints = json.loads(args.viewpoints.read_text())
    if len(frame_paths) != len(viewpoints) or len(frame_paths) < 2:
        raise ValueError(f"Need equal frame/viewpoint counts >=2, got {len(frame_paths)} and {len(viewpoints)}")
    base_yaws = [float(item["pose_map"][2]) for item in viewpoints]
    view_yaws = [math.radians(index * 45.0) for index in range(8)]

    projection_start = time.perf_counter()
    view_paths: list[list[Path]] = []
    intrinsics = None
    for frame_index, frame_path in enumerate(frame_paths):
        panorama = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if panorama is None or panorama.shape[:2] != (640, 1920):
            raise ValueError(f"Expected 1920x640 panorama: {frame_path}")
        frame_views = []
        for view_index, view_yaw in enumerate(view_yaws):
            view, intrinsics = project_equirectangular(panorama, view_yaw)
            output_path = views_dir / f"frame_{frame_index + 1:03d}_yaw_{view_index * 45:03d}.png"
            cv2.imwrite(str(output_path), view)
            frame_views.append(output_path)
        view_paths.append(frame_views)
    projection_seconds = time.perf_counter() - projection_start
    make_montage(view_paths[0], args.output / "perspective_views_frame_001.jpg")

    flat_paths = [path for frame in view_paths for path in frame]
    loaded = load_images([str(path) for path in flat_paths], size=512, verbose=False)
    for index, item in enumerate(loaded):
        item["idx"] = index
        item["instance"] = str(flat_paths[index])

    pair_records = []
    pairs = []

    # Adjacent virtual views from one panorama share the exact same center.
    # Their overlap directly tests projection continuity, including the seam.
    for frame_index in range(len(frame_paths)):
        for view_index, local_yaw1 in enumerate(view_yaws):
            view_index2 = (view_index + 1) % len(view_yaws)
            index1 = frame_index * 8 + view_index
            index2 = frame_index * 8 + view_index2
            pairs.append((loaded[index1], loaded[index2]))
            pair_records.append(
                {
                    "pair_kind": "intra_frame_adjacent",
                    "frame1": frame_index + 1,
                    "frame2": frame_index + 1,
                    "view1_yaw_deg": view_index * 45,
                    "view2_yaw_deg": view_index2 * 45,
                    "world_heading_residual_deg": 0.0,
                    "path1": str(flat_paths[index1]),
                    "path2": str(flat_paths[index2]),
                }
            )

    for frame_index in range(len(frame_paths) - 1):
        yaw_delta = wrap_pi(base_yaws[frame_index + 1] - base_yaws[frame_index])
        for view_index, local_yaw1 in enumerate(view_yaws):
            desired_local_yaw2 = wrap_pi(local_yaw1 - yaw_delta)
            view_index2 = nearest_view_index(desired_local_yaw2, view_yaws)
            residual = wrap_pi(view_yaws[view_index2] - desired_local_yaw2)
            index1 = frame_index * 8 + view_index
            index2 = (frame_index + 1) * 8 + view_index2
            pairs.append((loaded[index1], loaded[index2]))
            pair_records.append(
                {
                    "pair_kind": "cross_frame_yaw_compensated",
                    "frame1": frame_index + 1,
                    "frame2": frame_index + 2,
                    "view1_yaw_deg": view_index * 45,
                    "view2_yaw_deg": view_index2 * 45,
                    "robot_yaw_delta_deg": math.degrees(yaw_delta),
                    "world_heading_residual_deg": math.degrees(residual),
                    "path1": str(flat_paths[index1]),
                    "path2": str(flat_paths[index2]),
                }
            )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = AsymmetricMASt3R.from_pretrained(str(args.weights)).to(args.device).eval()
    model_load_seconds = time.perf_counter() - load_start
    inference_start = time.perf_counter()
    output = inference(pairs, model, args.device, batch_size=args.batch_size, verbose=True)
    inference_seconds = time.perf_counter() - inference_start

    matching_start = time.perf_counter()
    all_matches = []
    for pair_index, record in enumerate(pair_records):
        desc1 = output["pred1"]["desc"][pair_index].detach()
        desc2 = output["pred2"]["desc"][pair_index].detach()
        shape1 = output["view1"]["true_shape"][pair_index]
        shape2 = output["view2"]["true_shape"][pair_index]
        matches1, matches2 = valid_matches(desc1, desc2, shape1, shape2, args.device)
        record["valid_matches"] = int(len(matches1))
        if record["pair_kind"] == "intra_frame_adjacent":
            angular_errors = matched_ray_angular_errors_deg(
                matches1,
                matches2,
                math.radians(record["view1_yaw_deg"]),
                math.radians(record["view2_yaw_deg"]),
                intrinsics,
            )
            record["matched_ray_error_deg_median"] = float(np.median(angular_errors))
            record["matched_ray_error_deg_p90"] = float(np.percentile(angular_errors, 90))
        all_matches.append((matches1, matches2))
    matching_seconds = time.perf_counter() - matching_start

    counts = np.array([record["valid_matches"] for record in pair_records])
    selected_pair_indices = []
    for pair_kind in ("intra_frame_adjacent", "cross_frame_yaw_compensated"):
        indices = np.array([index for index, record in enumerate(pair_records) if record["pair_kind"] == pair_kind])
        kind_counts = counts[indices]
        selected_pair_indices.extend(
            [indices[np.argmin(kind_counts)], indices[np.argsort(kind_counts)[len(kind_counts) // 2]], indices[np.argmax(kind_counts)]]
        )
    selected_pair_indices = sorted(set(map(int, selected_pair_indices)))
    for pair_index in selected_pair_indices:
        record = pair_records[pair_index]
        draw_matches(
            Path(record["path1"]),
            Path(record["path2"]),
            all_matches[pair_index][0],
            all_matches[pair_index][1],
            matches_dir / f"{record['pair_kind']}_pair_{pair_index:03d}_{record['valid_matches']}_matches.jpg",
        )

    pred1 = output["pred1"]
    pred2 = output["pred2"]
    result_groups = {}
    for pair_kind in ("intra_frame_adjacent", "cross_frame_yaw_compensated"):
        records = [record for record in pair_records if record["pair_kind"] == pair_kind]
        group_counts = np.array([record["valid_matches"] for record in records])
        group = {
            "pair_count": len(records),
            "valid_matches_min": int(group_counts.min()),
            "valid_matches_median": float(np.median(group_counts)),
            "valid_matches_max": int(group_counts.max()),
            "all_pairs_have_matches": bool(np.all(group_counts > 0)),
        }
        if pair_kind == "intra_frame_adjacent":
            medians = np.array([record["matched_ray_error_deg_median"] for record in records])
            p90s = np.array([record["matched_ray_error_deg_p90"] for record in records])
            group["matched_ray_error_deg_median_across_pairs"] = float(np.median(medians))
            group["matched_ray_error_deg_worst_pair_p90"] = float(np.max(p90s))
        result_groups[pair_kind] = group

    report = {
        "schema_version": 1,
        "input": {
            "frames": [str(path) for path in frame_paths],
            "viewpoints": str(args.viewpoints),
            "panorama_projection": "central cropped equirectangular",
            "panorama_size": [1920, 640],
            "panorama_vertical_fov_deg": 120.0,
            "perspective_size": [512, 384],
            "perspective_horizontal_fov_deg": 90.0,
            "virtual_view_yaws_deg": [index * 45 for index in range(8)],
            "intrinsics": intrinsics.tolist(),
            "pairing": [
                "same-frame adjacent virtual views with 45 degree overlap",
                "consecutive frames, yaw compensated and snapped to nearest 45 degrees",
            ],
        },
        "runtime": {
            "device": args.device,
            "batch_size": args.batch_size,
            "weights": str(args.weights),
            "projection_seconds": projection_seconds,
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
            "matching_seconds": matching_seconds,
            "pair_count": len(pairs),
            "peak_cuda_gib": torch.cuda.max_memory_allocated() / (1024**3),
        },
        "results": {
            "by_pair_kind": result_groups,
            "pred1_conf": tensor_stats(pred1["conf"]),
            "pred2_conf": tensor_stats(pred2["conf"]),
            "pred1_pts3d": tensor_stats(pred1["pts3d"]),
            "pred2_pts3d_in_other_view": tensor_stats(pred2["pts3d_in_other_view"]),
        },
        "pairs": pair_records,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"runtime": report["runtime"], "results": report["results"]}, indent=2))


if __name__ == "__main__":
    main()
