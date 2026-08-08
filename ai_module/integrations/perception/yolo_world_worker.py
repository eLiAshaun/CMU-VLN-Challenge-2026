from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


YOLO_WORLD_VERSION = "YOLO-World-V2.1-X-stage1-20250205"


def normalize_phrase(value: str) -> str:
    return " ".join(value.strip().lower().split())


def build_prompt_entries(class_aliases: dict[str, list[str]]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for canonical, aliases in class_aliases.items():
        canonical = normalize_phrase(canonical)
        if not canonical:
            raise ValueError("canonical classes must not be empty")
        for phrase in [canonical, *aliases]:
            normalized = normalize_phrase(phrase)
            if normalized and normalized not in seen:
                entries.append({"text": normalized, "canonical_class": canonical})
                seen.add(normalized)
    return entries


def box_iou_xyxy(first: list[float], second: list[float]) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    low = np.maximum(a[:2], b[:2])
    high = np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(high - low, 0.0)))
    area_a = float(np.prod(np.maximum(a[2:] - a[:2], 0.0)))
    area_b = float(np.prod(np.maximum(b[2:] - b[:2], 0.0)))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def suppress_yolo_candidates(
    records: list[dict[str, Any]], iou_threshold: float, limit_per_class: int
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    ordered = sorted(records, key=lambda item: (-item["detector_score"], item["detection_id"]))
    for candidate in ordered:
        same_class = [
            item for item in kept if item["canonical_class"] == candidate["canonical_class"]
        ]
        if any(
            box_iou_xyxy(candidate["bbox_xyxy"], existing["bbox_xyxy"])
            >= iou_threshold
            for existing in same_class
        ):
            continue
        if len(same_class) >= limit_per_class:
            continue
        kept.append(candidate)
    return kept


def annotate(image: np.ndarray, records: list[dict[str, Any]], output: Path) -> None:
    canvas = image.copy()
    for record in records:
        x1, y1, x2, y2 = np.rint(record["bbox_xyxy"]).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (22, 190, 65), 2)
        label = (
            f"{record['detection_id']} {record['canonical_class']} "
            f"{record['detector_score']:.2f}"
        )
        cv2.putText(
            canvas,
            label,
            (max(0, x1), max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (22, 190, 65),
            2,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not save YOLO visualization: {output}")


def _imports():
    import torch
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmdet.apis import init_detector
    from mmdet.utils import get_test_pipeline_cfg

    import yolo_world  # noqa: F401 - registers YOLO-World model components

    return torch, Config, Compose, init_detector, get_test_pipeline_cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    output_root = Path(request["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    response_path = output_root / "worker_response.json"
    response: dict[str, Any] = {"status": "failed", "request_path": str(args.request)}
    try:
        started = time.perf_counter()
        ai_module_root = Path(__file__).resolve().parents[2]
        source_repo = Path(
            request.get(
                "source_repo",
                ai_module_root / "third_party" / "YOLO-World",
            )
        ).resolve()
        for import_root in (source_repo, source_repo / "third_party" / "mmyolo"):
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))
        torch, Config, Compose, init_detector, get_test_pipeline_cfg = _imports()
        checkpoint = Path(request["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"YOLO-World checkpoint missing: {checkpoint}")
        device = request.get("device", "cuda:0")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("YOLO-World requested CUDA but torch.cuda.is_available() is false")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        cfg = Config.fromfile(request["model_config"])
        cfg.model.backbone.text_model.model_name = request["text_model_path"]
        pipeline_cfg = get_test_pipeline_cfg(cfg)
        # A non-"none" palette keeps mmdet from instantiating the config's
        # LVIS validation dataset only to discover visualization metadata.
        # Open-vocabulary inference supplies its own texts.
        model = init_detector(
            cfg, checkpoint=str(checkpoint), device=device, palette="random"
        )
        pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
        test_pipeline = Compose(pipeline_cfg)
        entries = build_prompt_entries(request["class_aliases"])
        texts = [[item["text"]] for item in entries] + [[" "]]
        model.reparameterize(texts)
        model_load_seconds = time.perf_counter() - started

        records: list[dict[str, Any]] = []
        timings: dict[str, float] = {}
        detection_counter = 0
        for view in request["views"]:
            view_started = time.perf_counter()
            image_path = Path(view["image_path"])
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise ValueError(f"Could not read YOLO input image: {image_path}")
            height, width = image_bgr.shape[:2]
            expected_width = view.get("width")
            expected_height = view.get("height")
            if expected_width is not None and expected_height is not None:
                if (width, height) != (int(expected_width), int(expected_height)):
                    raise ValueError(
                        f"Canonical view size changed before YOLO-World: {view['view_id']}"
                    )
            view_dir = output_root / view["view_id"]
            crops_dir = view_dir / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(view_dir / "original.png"), image_bgr)
            image_rgb = image_bgr[:, :, ::-1]
            data_info = {"img": image_rgb, "img_id": 0, "texts": texts}
            transformed = test_pipeline(data_info)
            data_batch = {
                "inputs": transformed["inputs"].unsqueeze(0),
                "data_samples": [transformed["data_samples"]],
            }
            with torch.inference_mode():
                output = model.test_step(data_batch)[0]
            predictions = output.pred_instances.cpu().numpy()
            provisional: list[dict[str, Any]] = []
            for box, label_index, score in zip(
                predictions["bboxes"], predictions["labels"], predictions["scores"]
            ):
                score = float(score)
                label_index = int(label_index)
                if score < float(request["score_threshold"]) or label_index >= len(entries):
                    continue
                box = np.asarray(box, dtype=np.float64)
                box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                detection_counter += 1
                detection_id = f"yolo_det_{detection_counter:06d}"
                entry = entries[label_index]
                provisional.append(
                    {
                        "schema_version": "1.0",
                        "detection_id": detection_id,
                        "view_id": view["view_id"],
                        "detector": "yolo_world",
                        "detector_version": YOLO_WORLD_VERSION,
                        "evidence_tier": "primary",
                        "canonical_class": entry["canonical_class"],
                        "raw_label": entry["text"],
                        "detector_score": score,
                        "bbox_xyxy": [float(value) for value in box],
                        "representative_crop": None,
                    }
                )
            kept = suppress_yolo_candidates(
                provisional,
                float(request["nms_iou_threshold"]),
                int(request["max_detections_per_class_per_view"]),
            )
            for record in kept:
                x1, y1, x2, y2 = record["bbox_xyxy"]
                crop = image_bgr[
                    int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))
                ]
                crop_path = crops_dir / f"{record['detection_id']}.png"
                if crop.size and cv2.imwrite(str(crop_path), crop):
                    record["representative_crop"] = str(crop_path.resolve())
            records.extend(kept)
            annotate(image_bgr, kept, view_dir / "box_visualization.png")
            timings[f"{view['view_id']}_seconds"] = time.perf_counter() - view_started

        detections_path = output_root / "detections.json"
        detections_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        counts = Counter(item["canonical_class"] for item in records)
        response = {
            "status": "completed",
            "detections_path": str(detections_path.resolve()),
            "detection_count": len(records),
            "counts_by_class": dict(counts),
            "prompt_entries": entries,
            "threshold": float(request["score_threshold"]),
            "model": {
                "name": YOLO_WORLD_VERSION,
                "config": request["model_config"],
                "checkpoint": str(checkpoint),
                "text_model_path": request["text_model_path"],
                "device": device,
            },
            "timings": {
                "model_load_seconds": model_load_seconds,
                "total_seconds": time.perf_counter() - started,
                **timings,
            },
            "peak_cuda_gib": (
                float(torch.cuda.max_memory_allocated() / 1024**3)
                if device.startswith("cuda")
                else 0.0
            ),
        }
        response_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        response.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        response_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        return 1
    finally:
        for name in ("model", "test_pipeline"):
            if name in locals():
                del locals()[name]
        gc.collect()
        try:
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
