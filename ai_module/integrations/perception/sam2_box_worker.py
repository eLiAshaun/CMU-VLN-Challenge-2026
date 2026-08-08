#!/usr/bin/env python3
"""Convert detector boxes into mask-backed observations with vendored SAM2."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def _load_sam2(source_repo: Path):
    if str(source_repo) not in sys.path:
        sys.path.insert(0, str(source_repo))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return build_sam2, SAM2ImagePredictor


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _annotate(image: np.ndarray, records: list[dict[str, Any]], output: Path) -> None:
    canvas = image.copy()
    overlay = image.copy()
    palette = [(62, 193, 78), (230, 98, 52), (64, 128, 245), (190, 71, 207)]
    for index, record in enumerate(records):
        mask = cv2.imread(record["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        color = palette[index % len(palette)]
        overlay[mask > 0] = color
        x1, y1, x2, y2 = np.rint(record["bbox_xyxy"]).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{record['detection_id']} {record['canonical_class']}",
            (max(0, x1), max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )
    canvas = cv2.addWeighted(canvas, 0.72, overlay, 0.28, 0)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"could_not_write_visualization:{output}")


class SAM2BoxService:
    """Load SAM2 once and serve any number of box-to-mask requests."""

    def __init__(
        self,
        *,
        sam2_config: str,
        sam2_checkpoint: str,
        device: str = "cuda",
        source_repo: str | Path | None = None,
    ) -> None:
        ai_module_root = Path(__file__).resolve().parents[2]
        source = Path(
            source_repo or ai_module_root / "third_party" / "Grounded-SAM-2"
        ).resolve()
        build_sam2, predictor_type = _load_sam2(source)
        self.device = str(device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SAM2 requested CUDA but torch.cuda.is_available() is false")
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        self.predictor = predictor_type(model)
        self.model_load_seconds = time.perf_counter() - started

    def healthcheck(self) -> dict[str, Any]:
        return {
            "ready": True,
            "device": self.device,
            "model_load_seconds": self.model_load_seconds,
        }

    def segment(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        output_root = Path(request["output_dir"]).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        detections = json.loads(Path(request["detections_path"]).read_text(encoding="utf-8"))
        views = {item["view_id"]: item for item in request["views"]}
        by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for detection in detections:
            if detection["view_id"] not in views:
                raise ValueError(f"unknown_detection_view:{detection['view_id']}")
            by_view[detection["view_id"]].append(detection)

        records: list[dict[str, Any]] = []
        timings: dict[str, float] = {}
        for view_id, candidates in sorted(by_view.items()):
            view_started = time.perf_counter()
            view = views[view_id]
            image_bgr = cv2.imread(str(view["image_path"]), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"could_not_read_image:{view['image_path']}")
            height, width = image_bgr.shape[:2]
            if (width, height) != (int(view["width"]), int(view["height"])):
                raise ValueError(f"canonical_view_size_changed:{view_id}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_rgb)
            boxes = np.asarray([item["bbox_xyxy"] for item in candidates], dtype=np.float32)
            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes,
                    multimask_output=False,
                )
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            if scores.ndim > 1:
                scores = scores.squeeze(-1)
            view_dir = output_root / view_id
            masks_dir = view_dir / "masks"
            crops_dir = view_dir / "crops"
            masks_dir.mkdir(parents=True, exist_ok=True)
            crops_dir.mkdir(parents=True, exist_ok=True)
            view_records: list[dict[str, Any]] = []
            for candidate, mask, sam_score in zip(candidates, masks, scores):
                binary = np.asarray(mask, dtype=bool)
                area = int(binary.sum())
                if area < int(request.get("min_mask_area_px", 40)):
                    continue
                detection_id = str(candidate["detection_id"])
                mask_path = masks_dir / f"{detection_id}.png"
                if not cv2.imwrite(str(mask_path), binary.astype(np.uint8) * 255):
                    raise RuntimeError(f"could_not_write_mask:{mask_path}")
                x1, y1, x2, y2 = np.asarray(candidate["bbox_xyxy"], dtype=float)
                crop = image_bgr[
                    int(max(0, y1)):int(min(height, np.ceil(y2))),
                    int(max(0, x1)):int(min(width, np.ceil(x2))),
                ]
                crop_path = crops_dir / f"{detection_id}.png"
                crop_value = None
                if crop.size and cv2.imwrite(str(crop_path), crop):
                    crop_value = str(crop_path.resolve())
                record = {
                    **candidate,
                    "mask_path": str(mask_path.resolve()),
                    "mask_area_px": area,
                    "sam_score": float(sam_score),
                    "representative_crop": crop_value,
                }
                records.append(record)
                view_records.append(record)
            _annotate(image_bgr, view_records, view_dir / "mask_visualization.png")
            timings[f"{view_id}_seconds"] = time.perf_counter() - view_started

        detections_path = output_root / "detections.json"
        _write_json(detections_path, records)
        return {
            "status": "completed",
            "detections_path": str(detections_path.resolve()),
            "input_detection_count": len(detections),
            "mask_detection_count": len(records),
            "persistent_model": True,
            "timings": {
                "model_load_seconds": self.model_load_seconds,
                "request_seconds": time.perf_counter() - started,
                **timings,
            },
            "peak_cuda_gib": (
                float(torch.cuda.max_memory_allocated() / 1024**3)
                if self.device.startswith("cuda") else 0.0
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    output_root = Path(request["output_dir"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    response_path = output_root / "worker_response.json"
    response: dict[str, Any] = {"status": "failed", "request_path": str(args.request)}
    try:
        service = SAM2BoxService(
            sam2_config=request["sam2_config"],
            sam2_checkpoint=request["sam2_checkpoint"],
            device=str(request.get("device", "cuda")),
            source_repo=request.get("source_repo"),
        )
        response = service.segment(request)
        _write_json(response_path, response)
        return 0
    except Exception as exc:
        response.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(response_path, response)
        return 1
    finally:
        if "service" in locals():
            del service
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
