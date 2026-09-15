#!/usr/bin/env python3
"""Persistent abstract-socket YOLO-World detection service.

Loads the model once at startup and answers ``detect_views`` requests over
an abstract Unix socket.  The one-shot subprocess worker
(yolo_world_worker.py) remains the fallback when this server is disabled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .yolo_world_worker import (
    YOLO_WORLD_VERSION,
    _imports,
    build_prompt_entries,
    suppress_yolo_candidates,
)

YOLO_WORLD_DETECTION_COUNTER_OFFSET = 10_000_000


class YoloWorldService:
    def __init__(
        self,
        *,
        model_config: str,
        checkpoint: str,
        text_model_path: str,
        device: str,
    ) -> None:
        started = time.perf_counter()
        torch, Config, Compose, init_detector, get_test_pipeline_cfg = _imports()
        self.torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("YOLO-World requested CUDA but torch.cuda.is_available() is false")
        cfg = Config.fromfile(model_config)
        cfg.model.backbone.text_model.model_name = text_model_path
        pipeline_cfg = get_test_pipeline_cfg(cfg)
        # A non-"none" palette keeps mmdet from instantiating the config's
        # LVIS validation dataset only to discover visualization metadata.
        self.model = init_detector(
            cfg, checkpoint=str(checkpoint), device=device, palette="random"
        )
        pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
        self.test_pipeline = Compose(pipeline_cfg)
        self.device = device
        self.load_seconds = time.perf_counter() - started
        self._counter = YOLO_WORLD_DETECTION_COUNTER_OFFSET

    def healthcheck(self) -> dict[str, Any]:
        return {
            "name": YOLO_WORLD_VERSION,
            "device": self.device,
            "model_load_seconds": self.load_seconds,
        }

    def detect(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        entries = build_prompt_entries(request["class_aliases"])
        texts = [[item["text"]] for item in entries] + [[" "]]
        self.model.reparameterize(texts)
        records: list[dict[str, Any]] = []
        timings: dict[str, float] = {}
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
            image_rgb = image_bgr[:, :, ::-1]
            data_info = {"img": image_rgb, "img_id": 0, "texts": texts}
            transformed = self.test_pipeline(data_info)
            data_batch = {
                "inputs": transformed["inputs"].unsqueeze(0),
                "data_samples": [transformed["data_samples"]],
            }
            with self.torch.inference_mode():
                output = self.model.test_step(data_batch)[0]
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
                self._counter += 1
                detection_id = f"yolo_det_{self._counter:06d}"
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
            records.extend(kept)
            timings[f"{view['view_id']}_seconds"] = time.perf_counter() - view_started
        from collections import Counter
        counts = Counter(item["canonical_class"] for item in records)
        return {
            "status": "completed",
            "detection_count": len(records),
            "counts_by_class": dict(counts),
            "records": records,
            "threshold": float(request["score_threshold"]),
            "model": {
                "name": YOLO_WORLD_VERSION,
                "device": self.device,
                "model_load_seconds": self.load_seconds,
            },
            "timings": {
                "total_seconds": time.perf_counter() - started,
                **timings,
            },
        }


def serve(
    *,
    model_config: str,
    checkpoint: str,
    text_model_path: str,
    endpoint: str,
    device: str,
) -> None:
    if not endpoint.startswith("@") or len(endpoint) < 2:
        raise ValueError("endpoint_must_be_abstract_unix_socket")
    ai_module_root = Path(__file__).resolve().parents[2]
    source_repo = (ai_module_root / "third_party" / "YOLO-World").resolve()
    for import_root in (source_repo, source_repo / "third_party" / "mmyolo"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    service = YoloWorldService(
        model_config=model_config,
        checkpoint=checkpoint,
        text_model_path=text_model_path,
        device=device,
    )
    listener = Listener("\0" + endpoint[1:], family="AF_UNIX")
    try:
        while True:
            connection = listener.accept()
            request_id = ""
            try:
                payload = json.loads(connection.recv_bytes().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request_must_be_object")
                request_id = str(payload["request_id"])
                operation = str(payload["operation"])
                if operation == "healthcheck":
                    metadata = service.healthcheck()
                elif operation == "detect_views":
                    request = payload.get("parameters")
                    if not isinstance(request, dict):
                        raise ValueError("yolo_parameters_must_be_object")
                    metadata = service.detect(request)
                else:
                    raise ValueError("yolo_operation_invalid")
                response = {"request_id": request_id, "ok": True, "metadata": metadata}
            except Exception as exc:
                response = {
                    "request_id": request_id,
                    "ok": False,
                    "metadata": {"error_detail": str(exc)[:500]},
                    "error_code": type(exc).__name__,
                }
            try:
                connection.send_bytes(json.dumps(
                    response,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8"))
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                connection.close()
    finally:
        listener.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=False)
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--text-model-path", required=False)
    parser.add_argument("--endpoint", required=False)
    parser.add_argument("--device", required=False)
    args = parser.parse_args()

    # Backward compatibility: support CLI arguments
    if args.model_config and args.checkpoint and args.text_model_path:
        model_config = args.model_config
        checkpoint = args.checkpoint
        text_model_path = args.text_model_path
        endpoint = args.endpoint or "@mast3r_yolo"
        device = args.device or "cuda"
    else:
        # New path: load from config
        try:
            from config import load_config
            ai_module_root = Path(__file__).resolve().parents[2]
            config = load_config(
                asset_manifest_path=ai_module_root / "configs" / "model_assets.json"
            )
            model_config = str(ai_module_root / config.yolo_world.model_config_path)
            checkpoint = str(ai_module_root / config.yolo_world.checkpoint_path)
            text_model_path = str(ai_module_root / config.yolo_world.text_model_path)
            endpoint = config.yolo_world.endpoint
            device = config.yolo_world.device
        except ImportError:
            raise RuntimeError(
                "No --model-config provided and config system not available. "
                "Use --model-config, --checkpoint, --text-model-path, --endpoint, --device"
            )

    serve(
        model_config=model_config,
        checkpoint=checkpoint,
        text_model_path=text_model_path,
        endpoint=endpoint,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
