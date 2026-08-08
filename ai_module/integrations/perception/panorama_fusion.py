#!/usr/bin/env python3
"""Back-project perspective masks and merge same-centre duplicate observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import cv2
import numpy as np

from integrations.mast3r.panorama_adapter import project_mask_to_panorama


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _overlap(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(first & second))
    area_first = int(np.count_nonzero(first))
    area_second = int(np.count_nonzero(second))
    union = area_first + area_second - intersection
    iou = intersection / union if union else 0.0
    containment = intersection / min(area_first, area_second) if min(area_first, area_second) else 0.0
    return iou, containment


def _bbox(mask: np.ndarray) -> list[int]:
    rows, columns = np.nonzero(mask)
    return [int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    output = Path(request["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    response_path = output / "worker_response.json"
    response = {"status": "failed", "request_path": str(args.request)}
    try:
        manifest = json.loads(Path(request["manifest_path"]).read_text(encoding="utf-8"))
        view_map = {item["view_id"]: item for item in manifest["views"]}
        detections = json.loads(Path(request["detections_path"]).read_text(encoding="utf-8"))
        pano_width, pano_height = map(int, manifest["source_size"])
        masks_dir = output / "panorama_masks"
        masks_dir.mkdir(exist_ok=True)
        projected = []
        for detection in detections:
            view = view_map[detection["view_id"]]
            mask = cv2.imread(detection["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"sam_mask_unreadable:{detection['detection_id']}")
            map_x = np.load(view["map_x_path"])
            map_y = np.load(view["map_y_path"])
            panorama_mask = project_mask_to_panorama(mask, map_x, map_y, (pano_height, pano_width)) > 0
            if not panorama_mask.any():
                continue
            projected.append((detection, panorama_mask, view))

        clusters = []
        for detection, mask, view in sorted(projected, key=lambda item: -float(item[0]["detector_score"])):
            destination = None
            for cluster in clusters:
                if cluster["canonical_class"] != detection["canonical_class"]:
                    continue
                iou, containment = _overlap(mask, cluster["mask"])
                if iou >= float(request["mask_iou_threshold"]) or containment >= float(request["mask_containment_threshold"]):
                    destination = cluster
                    break
            member = {
                "detection_id": detection["detection_id"],
                "detector": detection.get("detector", "unknown"),
                "view_id": detection["view_id"],
                "detector_score": float(detection["detector_score"]),
                "bbox_xyxy": detection["bbox_xyxy"],
                "mask_path": detection["mask_path"],
                "mask_area_px": int(detection.get("mask_area_px", 0)),
                "sam_score": float(detection.get("sam_score", 0.0)),
                "qwen_grounding_probability": detection.get("qwen_grounding_probability"),
                "qwen_verification_probability": detection.get("qwen_verification_probability"),
                "qwen_target_probability": detection.get("qwen_target_probability"),
            }
            if destination is None:
                clusters.append({
                    "canonical_class": detection["canonical_class"],
                    "mask": mask.copy(),
                    "representative": detection,
                    "representative_view": view,
                    "members": [member],
                })
            else:
                destination["mask"] |= mask
                destination["members"].append(member)

        observations = []
        panorama = cv2.imread(request["panorama_path"], cv2.IMREAD_COLOR)
        overlay = panorama.copy()
        evidence_dir = output / "qwen_evidence"
        evidence_dir.mkdir(exist_ok=True)
        view_prefixes = {
            str(item["view_id"]).split("__", 1)[0]
            for item in manifest["views"]
            if "__" in str(item["view_id"])
        }
        observation_prefix = next(iter(view_prefixes)) if len(view_prefixes) == 1 else "current"
        for index, cluster in enumerate(clusters, start=1):
            observation_id = f"{observation_prefix}__pano_obs_{index:06d}"
            mask_path = masks_dir / f"{observation_id}.png"
            cv2.imwrite(str(mask_path), cluster["mask"].astype(np.uint8) * 255)
            representative = cluster["representative"]
            semantic_weights = []
            semantic_values = []
            for member in cluster["members"]:
                probability = member.get("qwen_target_probability")
                if probability is None:
                    continue
                semantic_values.append(float(probability))
                semantic_weights.append(max(
                    1e-6,
                    float(member.get("detector_score", 0.0))
                    * max(1e-3, float(member.get("sam_score", 0.0))),
                ))
            semantic_probability = (
                float(np.average(semantic_values, weights=semantic_weights))
                if semantic_values else None
            )
            evidence_path = None
            if panorama is not None:
                evidence = panorama.copy()
                tint = evidence.copy()
                tint[cluster["mask"]] = (32, 64, 255)
                evidence = cv2.addWeighted(evidence, 0.72, tint, 0.28, 0)
                x1, y1, x2, y2 = _bbox(cluster["mask"])
                cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 0, 255), 3)
                candidate_path = evidence_dir / f"{observation_id}.png"
                if cv2.imwrite(str(candidate_path), evidence):
                    evidence_path = str(candidate_path)
            observations.append({
                "schema_version": "1.0",
                "observation_id": observation_id,
                "canonical_class": cluster["canonical_class"],
                "panorama_mask_path": str(mask_path),
                "panorama_bbox_xyxy": _bbox(cluster["mask"]),
                "panorama_image_path": str(Path(request["panorama_path"]).resolve()),
                "qwen_evidence_image": evidence_path,
                "representative_view_id": representative["view_id"],
                "representative_view_image": cluster["representative_view"]["image_path"],
                "representative_bbox_xyxy": representative["bbox_xyxy"],
                "representative_detection_id": representative["detection_id"],
                "representative_mask_path": representative["mask_path"],
                "representative_score": float(representative["detector_score"]),
                "member_count": len(cluster["members"]),
                "members": cluster["members"],
                "source_view_ids": list(dict.fromkeys(
                    str(member["view_id"]) for member in cluster["members"]
                )),
                "qwen_target_probability": semantic_probability,
                "optical_center_group": manifest["optical_center_group"],
                "independent_spatial_evidence_count": 1,
                "persistent_instance_authorized": False,
            })
            overlay[cluster["mask"]] = (64, 200, 80)
        if panorama is not None:
            visualization = cv2.addWeighted(panorama, 0.72, overlay, 0.28, 0)
            cv2.imwrite(str(output / "panorama_observations.png"), visualization)
        observations_path = output / "observations.json"
        _write_json(observations_path, observations)
        response = {
            "status": "completed",
            "observations_path": str(observations_path),
            "input_mask_count": len(detections),
            "projected_mask_count": len(projected),
            "fused_observation_count": len(observations),
            "duplicate_count": len(projected) - len(observations),
            "independent_optical_center_count": 1 if observations else 0,
        }
        _write_json(response_path, response)
        return 0
    except Exception as exc:
        response.update({"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        _write_json(response_path, response)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
