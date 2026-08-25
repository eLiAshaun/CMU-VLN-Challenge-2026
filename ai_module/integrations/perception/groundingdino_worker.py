#!/usr/bin/env python3
"""GroundingDINO box-only rescue worker.

SAM deliberately does not run here.  Detector boxes from either YOLO-World or
this rescue stage flow through one shared SAM stage later in the pipeline.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import re
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert


def normalize_label(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", value.lower()).split())


def canonical_for_label(label: str, aliases: dict[str, list[str]]) -> str | None:
    normalized = normalize_label(label)
    matches = []
    for canonical, values in aliases.items():
        for phrase in [canonical, *values]:
            phrase = normalize_label(phrase)
            if normalized == phrase or phrase in normalized or normalized in phrase:
                matches.append((len(phrase), canonical))
    return max(matches)[1] if matches else None


def build_prompt(classes: list[str], aliases: dict[str, list[str]], include_aliases: bool) -> str:
    phrases = []
    for canonical in classes:
        for phrase in [canonical, *(aliases.get(canonical, []) if include_aliases else [])]:
            phrase = normalize_label(phrase)
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return ". ".join(phrases) + "."


def box_iou(first: list[float], second: list[float]) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    low = np.maximum(first[:2], second[:2])
    high = np.minimum(first[2:], second[2:])
    intersection = float(np.prod(np.maximum(high - low, 0.0)))
    union = float(np.prod(first[2:] - first[:2]) + np.prod(second[2:] - second[:2]) - intersection)
    return intersection / union if union > 0 else 0.0


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
        root = Path(__file__).resolve().parents[2]
        source = Path(request.get("source_repo", root / "third_party" / "Grounded-SAM-2")).resolve()
        for import_root in (source, source / "grounding_dino"):
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))
        try:
            from groundingdino.util.inference import load_image, load_model, predict
        except ImportError:
            from grounding_dino.groundingdino.util.inference import load_image, load_model, predict

        device = str(request.get("device", "cuda"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("GroundingDINO requested CUDA but CUDA is unavailable")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model_config = Path(request["groundingdino_config"])
        if request.get("bert_model_path"):
            runtime_config = output / "GroundingDINO_runtime_cfg.py"
            text = model_config.read_text(encoding="utf-8").replace(
                'text_encoder_type = "bert-base-uncased"',
                f'text_encoder_type = {json.dumps(request["bert_model_path"])}',
            )
            runtime_config.write_text(text, encoding="utf-8")
            model_config = runtime_config
        model = load_model(str(model_config), request["groundingdino_checkpoint"], device=device)
        model_load_seconds = time.perf_counter() - started
        aliases = request["class_aliases"]
        classes = list(aliases)
        target = request["target_class"]
        records = []
        timings = {}
        detection_index = 0

        def run_pass(pass_classes: list[str], include_aliases: bool, retry: bool) -> list[dict]:
            nonlocal detection_index
            found = []
            prompt = build_prompt(pass_classes, aliases, include_aliases)
            box_threshold = float(request["retry_box_threshold"] if retry else request["box_threshold"])
            text_threshold = float(request["retry_text_threshold"] if retry else request["text_threshold"])
            for view in request["views"]:
                view_started = time.perf_counter()
                source_image, transformed = load_image(view["image_path"])
                if source_image.shape[:2] != (int(view["height"]), int(view["width"])):
                    raise ValueError(f"perspective_view_size_changed:{view['view_id']}")
                boxes, confidences, labels = predict(
                    model=model,
                    image=transformed,
                    caption=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=device,
                )
                height, width = source_image.shape[:2]
                boxes_xyxy = box_convert(
                    boxes=boxes * torch.tensor([width, height, width, height], device=boxes.device),
                    in_fmt="cxcywh",
                    out_fmt="xyxy",
                ).detach().cpu().numpy()
                provisional = []
                for box, confidence, label in zip(boxes_xyxy, confidences.detach().cpu().tolist(), labels):
                    canonical = canonical_for_label(str(label), aliases)
                    if canonical is None or canonical not in pass_classes:
                        continue
                    detection_index += 1
                    clipped = np.clip(box, [0, 0, 0, 0], [width - 1, height - 1, width, height])
                    provisional.append({
                        "schema_version": "1.0",
                        "detection_id": f"dino_det_{detection_index:06d}",
                        "view_id": view["view_id"],
                        "detector": "groundingdino",
                        "evidence_tier": "rescue",
                        "canonical_class": canonical,
                        "raw_label": str(label),
                        "detector_score": float(confidence),
                        "bbox_xyxy": [float(value) for value in clipped],
                        "retry_pass": retry,
                    })
                kept = []
                for candidate in sorted(provisional, key=lambda item: -item["detector_score"]):
                    if any(box_iou(candidate["bbox_xyxy"], item["bbox_xyxy"]) >= float(request["nms_iou_threshold"]) for item in kept):
                        continue
                    kept.append(candidate)
                    if len(kept) >= int(request["max_detections_per_class_per_view"]):
                        break
                found.extend(kept)
                timings[f"{view['view_id']}_{'retry' if retry else 'initial'}_seconds"] = time.perf_counter() - view_started
            return found

        records.extend(run_pass(classes, False, False))
        retry_used = not any(item["canonical_class"] == target for item in records)
        if retry_used:
            records.extend(run_pass([target], True, True))
        detections_path = output / "detections.json"
        detections_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        response = {
            "status": "completed",
            "detections_path": str(detections_path),
            "detection_count": len(records),
            "counts_by_class": dict(Counter(item["canonical_class"] for item in records)),
            "retry_used": retry_used,
            "model_load_seconds": model_load_seconds,
            "timings": timings,
            "peak_cuda_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
        }
        response_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        response.update({"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        response_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        return 1
    finally:
        if "model" in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
