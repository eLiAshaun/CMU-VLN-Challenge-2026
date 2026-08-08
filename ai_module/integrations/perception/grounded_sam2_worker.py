from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert


def _imports():
    try:
        from groundingdino.util.inference import load_image, load_model, predict
    except ImportError:
        from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return load_image, load_model, predict, build_sam2, SAM2ImagePredictor


def normalize_label(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", value.lower()).split())


def canonical_for_label(label: str, class_aliases: dict[str, list[str]]) -> str | None:
    normalized = normalize_label(label)
    matches: list[tuple[int, str]] = []
    for canonical, aliases in class_aliases.items():
        for phrase in [canonical, *aliases]:
            phrase_normalized = normalize_label(phrase)
            if normalized == phrase_normalized or phrase_normalized in normalized or normalized in phrase_normalized:
                matches.append((len(phrase_normalized), canonical))
    return max(matches)[1] if matches else None


def build_prompt(classes: list[str], aliases: dict[str, list[str]], include_aliases: bool) -> str:
    phrases: list[str] = []
    for canonical in classes:
        for phrase in [canonical, *(aliases.get(canonical, []) if include_aliases else [])]:
            phrase = normalize_label(phrase)
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return ". ".join(phrases) + "."


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


def suppress_duplicate_detections(
    values: list[tuple[dict[str, Any], np.ndarray]],
    iou_threshold: float,
    limit: int,
) -> list[tuple[dict[str, Any], np.ndarray]]:
    """Deterministic per-class, per-view NMS before 3D association."""
    ordered = sorted(values, key=lambda item: (-item[0]["detector_score"], item[0]["detection_id"]))
    kept: list[tuple[dict[str, Any], np.ndarray]] = []
    for candidate in ordered:
        if any(box_iou_xyxy(candidate[0]["bbox_xyxy"], existing[0]["bbox_xyxy"]) >= iou_threshold for existing in kept):
            continue
        kept.append(candidate)
        if len(kept) >= limit:
            break
    return kept


def annotate(image: np.ndarray, records: list[dict[str, Any]], masks: list[np.ndarray], output: Path) -> None:
    canvas = image.copy()
    overlay = image.copy()
    palette = [(62, 193, 78), (230, 98, 52), (64, 128, 245), (190, 71, 207)]
    for index, (record, mask) in enumerate(zip(records, masks)):
        color = palette[index % len(palette)]
        overlay[mask.astype(bool)] = color
        x1, y1, x2, y2 = np.rint(record["bbox_xyxy"]).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        caption = f"{record['detection_id']} {record['canonical_class']} {record['detector_score']:.2f}"
        cv2.putText(canvas, caption, (max(0, x1), max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
    canvas = cv2.addWeighted(canvas, 0.72, overlay, 0.28, 0)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not save visualization: {output}")


def annotate_boxes(image: np.ndarray, records: list[dict[str, Any]], output: Path) -> None:
    canvas = image.copy()
    for record in records:
        x1, y1, x2, y2 = np.rint(record["bbox_xyxy"]).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (62, 193, 78), 2)
        cv2.putText(canvas, f"{record['detection_id']} {record['canonical_class']}", (max(0, x1), max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (62, 193, 78), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not save box visualization: {output}")


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
        ai_module_root = Path(__file__).resolve().parents[2]
        source_repo = Path(
            request.get(
                "source_repo",
                ai_module_root / "third_party" / "Grounded-SAM-2",
            )
        ).resolve()
        for import_root in (source_repo, source_repo / "grounding_dino"):
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))
        load_image, load_model, predict, build_sam2, SAM2ImagePredictor = _imports()
        device = request.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        timings: dict[str, float] = {}
        started = time.perf_counter()
        grounding_config = Path(request["groundingdino_config"])
        if request.get("bert_model_path"):
            runtime_config = output_root / "GroundingDINO_runtime_cfg.py"
            config_text = grounding_config.read_text(encoding="utf-8")
            config_text = config_text.replace(
                'text_encoder_type = "bert-base-uncased"',
                f'text_encoder_type = {json.dumps(request["bert_model_path"])}',
            )
            runtime_config.write_text(config_text, encoding="utf-8")
            grounding_config = runtime_config
        grounding_model = load_model(
            model_config_path=str(grounding_config),
            model_checkpoint_path=request["groundingdino_checkpoint"],
            device=device,
        )
        sam2_model = build_sam2(request["sam2_config"], request["sam2_checkpoint"], device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        timings["model_load_seconds"] = time.perf_counter() - started
        class_aliases = request["class_aliases"]
        classes = list(class_aliases)
        target_class = request["target_class"]
        records: list[dict[str, Any]] = []
        detection_counter = 0

        def run_pass(pass_classes: list[str], include_aliases: bool, retry_pass: bool) -> list[dict[str, Any]]:
            nonlocal detection_counter
            pass_records: list[dict[str, Any]] = []
            prompt = build_prompt(pass_classes, class_aliases, include_aliases)
            box_threshold = float(request["retry_box_threshold"] if retry_pass else request["box_threshold"])
            text_threshold = float(request["retry_text_threshold"] if retry_pass else request["text_threshold"])
            for view in request["views"]:
                per_view_started = time.perf_counter()
                view_id = view["view_id"]
                view_dir = output_root / view_id
                masks_dir = view_dir / "masks"
                crops_dir = view_dir / "crops"
                masks_dir.mkdir(parents=True, exist_ok=True)
                crops_dir.mkdir(parents=True, exist_ok=True)
                image_source, transformed = load_image(view["image_path"])
                if image_source.shape[:2] != (int(view["height"]), int(view["width"])):
                    raise ValueError(f"Canonical view size changed before perception: {view_id}")
                if not retry_pass:
                    cv2.imwrite(str(view_dir / "original.png"), cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR))
                predictor.set_image(image_source)
                boxes, confidences, labels = predict(
                    model=grounding_model,
                    image=transformed,
                    caption=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    device=device,
                )
                if len(boxes) == 0:
                    timings[f"{view_id}_{'retry' if retry_pass else 'initial'}_seconds"] = time.perf_counter() - per_view_started
                    continue
                h, w = image_source.shape[:2]
                boxes_xyxy = box_convert(boxes=boxes * torch.tensor([w, h, w, h], device=boxes.device), in_fmt="cxcywh", out_fmt="xyxy").detach().cpu().numpy()
                with torch.inference_mode():
                    masks, sam_scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=boxes_xyxy,
                        multimask_output=bool(request.get("sam_multimask", False)),
                    )
                if masks.ndim == 4:
                    if bool(request.get("sam_multimask", False)):
                        best = np.argmax(sam_scores, axis=1)
                        masks = masks[np.arange(len(masks)), best]
                    else:
                        masks = masks.squeeze(1)
                provisional: list[tuple[dict[str, Any], np.ndarray]] = []
                for box, confidence, raw_label, mask in zip(boxes_xyxy, confidences.detach().cpu().tolist(), labels, masks):
                    canonical = canonical_for_label(str(raw_label), class_aliases)
                    if canonical is None or canonical not in pass_classes:
                        continue
                    binary = np.asarray(mask, dtype=bool)
                    area = int(binary.sum())
                    if area < int(request["min_mask_area_px"]):
                        continue
                    detection_counter += 1
                    detection_id = f"det_{detection_counter:06d}"
                    mask_path = masks_dir / f"{detection_id}.png"
                    cv2.imwrite(str(mask_path), binary.astype(np.uint8) * 255)
                    x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [w - 1, h - 1, w, h])
                    crop = image_source[int(max(0, y1)):int(min(h, y2)), int(max(0, x1)):int(min(w, x2))]
                    crop_path = crops_dir / f"{detection_id}.png"
                    if crop.size:
                        cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    record = {
                        "detection_id": detection_id,
                        "view_id": view_id,
                        "canonical_class": canonical,
                        "raw_label": str(raw_label),
                        "detector_score": float(confidence),
                        "class_scores": {canonical: float(confidence)},
                        "bbox_xyxy": [float(value) for value in box],
                        "mask_path": str(mask_path.resolve()),
                        "mask_area_px": area,
                        "retry_pass": retry_pass,
                        "representative_crop": str(crop_path.resolve()) if crop.size else None,
                    }
                    provisional.append((record, binary))
                by_class: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = {}
                for record, mask in provisional:
                    by_class.setdefault(record["canonical_class"], []).append((record, mask))
                kept: list[tuple[dict[str, Any], np.ndarray]] = []
                for values in by_class.values():
                    kept.extend(
                        suppress_duplicate_detections(
                            values,
                            float(request.get("nms_iou_threshold", 0.65)),
                            int(request["max_detections_per_class_per_view"]),
                        )
                    )
                if kept:
                    annotate_boxes(
                        cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR),
                        [item[0] for item in kept],
                        view_dir / ("retry_box_visualization.png" if retry_pass else "box_visualization.png"),
                    )
                    annotate(
                        cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR),
                        [item[0] for item in kept],
                        [item[1] for item in kept],
                        view_dir / ("retry_visualization.png" if retry_pass else "detection_mask_visualization.png"),
                    )
                pass_records.extend([item[0] for item in kept])
                timings[f"{view_id}_{'retry' if retry_pass else 'initial'}_seconds"] = time.perf_counter() - per_view_started
            return pass_records

        records.extend(run_pass(classes, include_aliases=False, retry_pass=False))
        retry_used = not any(item["canonical_class"] == target_class for item in records)
        if retry_used:
            records.extend(run_pass([target_class], include_aliases=True, retry_pass=True))
        counts = Counter(item["canonical_class"] for item in records)
        detection_path = output_root / "detections.json"
        detection_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        response = {
            "status": "completed",
            "detections_path": str(detection_path.resolve()),
            "detection_count": len(records),
            "counts_by_class": dict(counts),
            "retry_used": retry_used,
            "target_detected": counts[target_class] > 0,
            "prompt_initial": build_prompt(classes, class_aliases, False),
            "prompt_retry": build_prompt([target_class], class_aliases, True) if retry_used else None,
            "timings": timings,
            "peak_cuda_gib": float(torch.cuda.max_memory_allocated() / 1024**3) if device.startswith("cuda") else 0.0,
        }
        response_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        response.update({"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        response_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        return 1
    finally:
        for name in ("grounding_model", "sam2_model", "predictor"):
            if name in locals():
                del locals()[name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
