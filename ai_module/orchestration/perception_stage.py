"""Task-conditioned perception for the lean CMU-VLN production chain.

The normal path is deliberately small:

    one 360 panorama -> eight perspective views -> one Qwen3-VL grounding
    batch -> one SAM2 segmentation batch -> bounded Qwen verification batches.

YOLO-World is optional proposal supplementation.  The chain never requires a
YOLO worker to produce its first proposal set and never performs one-model-call
per candidate unless a view batch itself must be split.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from integrations.mast3r.panorama_adapter import write_perspective_bundle
from integrations.model_socket_client import socket_request

logger = logging.getLogger(__name__)


_CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    "picture": ("photo", "painting", "framed picture", "wall art"),
    "photo": ("picture", "framed photo"),
    "television": ("tv", "screen"),
    "television cabinet": ("tv cabinet", "media cabinet", "tv stand"),
    "potted plant": ("plant", "indoor plant"),
    "sofa": ("couch",),
    "pillow": ("cushion",),
    "refrigerator": ("fridge", "refridgerator"),
    "trash can": ("bin", "waste bin"),
    "nightstand": ("bedside table",),
}


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _entity_lookup(task_ir: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(value.get("id", "")): dict(value)
        for value in task_ir.get("entities", ())
        if str(value.get("id", "")).strip()
    }


def extract_task_concepts(
    task_ir: Mapping[str, Any],
    entity_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build only the vocabulary that can change the current answer."""
    entities = _entity_lookup(task_ir)
    selected_ids = [str(value) for value in (entity_ids or ()) if str(value) in entities]
    selected = [entities[value] for value in selected_ids] if selected_ids else list(entities.values())

    classes: list[str] = []
    entity_aliases: dict[str, list[str]] = defaultdict(list)
    attributes: dict[str, dict[str, Any]] = {}
    for entity in selected:
        class_name = str(entity.get("class_name", "")).strip().lower()
        if not class_name:
            continue
        classes.append(class_name)
        entity_aliases[class_name].extend(
            str(alias).strip().lower()
            for alias in entity.get("aliases", ())
            if str(alias).strip()
        )
        if isinstance(entity.get("attributes"), Mapping):
            attributes[class_name] = dict(entity["attributes"])

    for class_name in task_ir.get("required_classes", ()):
        normalized = str(class_name).strip().lower()
        if normalized:
            classes.append(normalized)

    classes = _ordered_unique(classes)
    plan_by_class = {
        str(value.get("primary_class", "")).strip().lower(): dict(value)
        for value in task_ir.get("grounding_plan", {}).get("entities", ())
        if isinstance(value, Mapping)
        and str(value.get("primary_class", "")).strip()
    }
    concepts: list[dict[str, Any]] = []
    for class_name in classes:
        planned = plan_by_class.get(class_name, {})
        aliases = _ordered_unique([
            *entity_aliases.get(class_name, ()),
            *_CLASS_ALIASES.get(class_name, ()),
            *(
                str(value).strip().lower()
                for value in planned.get("detector_aliases", ())
                if str(value).strip()
            ),
        ])
        attr = attributes.get(class_name, {})
        attr_text = ", ".join(
            f"{key}={value}"
            for key, value in attr.items()
            if value not in (None, "", [], {})
        )
        definition = str(
            planned.get("visual_definition", "")
        ).strip() or (
            class_name if not attr_text else f"{class_name} with {attr_text}"
        )
        concepts.append({
            "class_name": class_name,
            "aliases": aliases,
            "visual_definition": definition,
            # Context is an attention hint for candidate recall. Relation
            # membership remains the exact-ID verifier's responsibility.
            "supporting_context": [
                dict(value)
                for value in planned.get("supporting_context", ())
                if isinstance(value, Mapping)
            ],
            "hard_negatives": _ordered_unique([
                str(value).strip().lower()
                for value in planned.get("hard_negatives", ())
                if str(value).strip()
            ]),
            "complete_instance_box": True,
        })
    return concepts


def _qwen_payload(
    *,
    acquisition_id: str,
    operation: str,
    parameters: Mapping[str, Any],
    input_handles: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the complete request envelope required by the Qwen worker."""
    return {
        "acquisition_id": str(acquisition_id),
        "backend": "qwen3vl",
        "operation": str(operation),
        "input_handles": [str(value) for value in input_handles],
        "parameters": dict(parameters),
    }


def _remaining(deadline_monotonic: float) -> float:
    return max(0.0, float(deadline_monotonic) - time.monotonic())


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _qwen_ground(
    *,
    acquisition_id: str,
    views: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    question: str,
    endpoint: str,
    timeout: float,
    deadline_monotonic: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    """Qwen grounding with fail-open semantics.

    Returns:
        (records, metadata, error_warning)
        - records: list of detections (empty if failed)
        - metadata: response metadata
        - error_warning: None on success, error description on failure
    """
    payload_views = [
        {
            "view_id": str(view["view_id"]),
            "image_path": str(view["image_path"]),
            "concepts": concepts,
            "scene_context": question,
        }
        for view in views[:8]
    ]

    try:
        response = socket_request(
            endpoint,
            _qwen_payload(
                acquisition_id=acquisition_id,
                operation="ground_objects_batch",
                input_handles=[value["image_path"] for value in payload_views],
                parameters={"views": payload_views},
            ),
            max(1.0, float(timeout)),
            deadline_monotonic=deadline_monotonic,
        )
        metadata = dict(response.get("metadata", {}))
    except Exception as exc:
        error_type = type(exc).__name__
        error_msg = str(exc)[:200]
        return [], {}, f"qwen_grounding_failed:{error_type}:{error_msg}"

    by_view = {str(view["view_id"]): view for view in views}
    records: list[dict[str, Any]] = []
    counter = 0
    for view_result in metadata.get("views", ()):
        view_id = str(view_result.get("view_id", ""))
        view = by_view.get(view_id)
        if view is None:
            continue
        width, height = int(view["width"]), int(view["height"])
        for proposal in view_result.get("proposals", ()):
            normalized = proposal.get("bbox_xyxy_normalized")
            if not isinstance(normalized, (list, tuple)) or len(normalized) != 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in normalized]
            x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
            y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
            if x2 - x1 < 0.005 or y2 - y1 < 0.005:
                continue
            class_name = str(proposal.get("class_name", "")).strip().lower()
            if not class_name:
                continue
            counter += 1
            records.append({
                "schema_version": "1.0",
                "detection_id": f"qwen_det_{counter:06d}",
                "view_id": view_id,
                "canonical_class": class_name,
                "raw_label": class_name,
                "bbox_xyxy": [x1 * width, y1 * height, x2 * width, y2 * height],
                "bbox_xyxy_normalized": [x1, y1, x2, y2],
                "width": width,
                "height": height,
                "detector": "qwen3vl",
                "detector_score": 0.70,
                "proposal_status": "grounded",
                "rationale_tags": list(proposal.get("rationale_tags", ())),
            })
    return records, metadata, None


def _optional_yolo(
    *,
    views: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    endpoint: str,
    timeout: float,
    threshold: float,
) -> tuple[list[dict[str, Any]], str | None]:
    aliases = {
        str(value["class_name"]): [str(alias) for alias in value.get("aliases", ())]
        for value in concepts
    }
    view_lookup = {str(value["view_id"]): value for value in views}
    try:
        response = socket_request(
            endpoint,
            {
                "operation": "detect_views",
                "parameters": {
                    "class_aliases": aliases,
                    "views": [
                        {
                            "view_id": str(value["view_id"]),
                            "image_path": str(value["image_path"]),
                            "width": int(value["width"]),
                            "height": int(value["height"]),
                        }
                        for value in views
                    ],
                    "score_threshold": float(threshold),
                    "nms_iou_threshold": 0.65,
                    "max_detections_per_class_per_view": 20,
                },
            },
            max(1.0, float(timeout)),
        )
        normalized_records: list[dict[str, Any]] = []
        for raw in response.get("metadata", {}).get("records", ()):
            record = dict(raw)
            view = view_lookup.get(str(record.get("view_id", "")))
            bbox = record.get("bbox_xyxy")
            if view is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            width, height = int(view["width"]), int(view["height"])
            x1, y1, x2, y2 = [float(value) for value in bbox]
            record.update({
                "width": width,
                "height": height,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xyxy_normalized": [
                    x1 / max(1.0, width),
                    y1 / max(1.0, height),
                    x2 / max(1.0, width),
                    y2 / max(1.0, height),
                ],
            })
            normalized_records.append(record)
        return normalized_records, None
    except Exception as exc:  # optional worker absence is not a chain failure
        return [], f"{type(exc).__name__}:{str(exc)[:160]}"


def _merge_optional_supplement(
    primary: list[dict[str, Any]],
    supplement: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = list(primary)
    for candidate in supplement:
        if any(
            str(existing.get("view_id")) == str(candidate.get("view_id"))
            and str(existing.get("canonical_class")) == str(candidate.get("canonical_class"))
            and _box_iou(
                existing.get("bbox_xyxy", (0, 0, 0, 0)),
                candidate.get("bbox_xyxy", (0, 0, 0, 0)),
            ) >= 0.85
            for existing in result
        ):
            continue
        candidate = dict(candidate)
        candidate["proposal_status"] = "supplemental"
        result.append(candidate)
    return result


def _segment_all_views(
    *,
    views: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    endpoint: str,
    output_dir: Path,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not detections:
        return [], {"input_detection_count": 0, "mask_detection_count": 0}
    stage_dir = output_dir / "03_sam2_masks"
    stage_dir.mkdir(parents=True, exist_ok=True)
    detections_path = stage_dir / "detections_input.json"
    detections_path.write_text(json.dumps(detections, indent=2) + "\n", encoding="utf-8")
    response = socket_request(
        endpoint,
        {
            "operation": "segment_boxes",
            "parameters": {
                "output_dir": str(stage_dir),
                "detections_path": str(detections_path),
                "views": [
                    {
                        "view_id": str(value["view_id"]),
                        "image_path": str(value["image_path"]),
                        "width": int(value["width"]),
                        "height": int(value["height"]),
                    }
                    for value in views
                ],
                "min_mask_area_px": 40,
            },
        },
        max(1.0, float(timeout)),
    )
    metadata = dict(response.get("metadata", {}))
    path = Path(str(metadata.get("detections_path", "")))
    if not path.is_file():
        raise RuntimeError("sam2_detections_output_missing")
    return [dict(value) for value in json.loads(path.read_text(encoding="utf-8"))], metadata


def _verify_by_view(
    *,
    acquisition_id: str,
    detections: list[dict[str, Any]],
    views: list[dict[str, Any]],
    concepts: list[dict[str, Any]],
    endpoint: str,
    deadline_monotonic: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    view_lookup = {str(value["view_id"]): value for value in views}
    concept_lookup = {
        str(value["class_name"]): value for value in concepts
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        grouped[str(detection.get("view_id", ""))].append(detection)

    output: list[dict[str, Any]] = []
    warnings: list[str] = []
    for view_id, items in grouped.items():
        view = view_lookup.get(view_id)
        if view is None:
            continue
        def verification_priority(item: Mapping[str, Any]) -> tuple[bool, bool]:
            bbox = item.get("bbox_xyxy_normalized")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                width = float(item.get("width") or view["width"])
                height = float(item.get("height") or view["height"])
                x1, y1, x2, y2 = [
                    float(value) for value in item["bbox_xyxy"]
                ]
                bbox = [x1 / width, y1 / height, x2 / width, y2 / height]
            boundary_truncated = bool(
                float(bbox[0]) <= 0.0
                or float(bbox[1]) <= 0.0
                or float(bbox[2]) >= 1.0
                or float(bbox[3]) >= 1.0
            )
            supplemental = str(item.get("proposal_status", "")) == "supplemental"
            # The verifier is semantic authority for the proposals most
            # likely to be detector fragments.  Schedule those before full,
            # grounded boxes so a finite evidence window cannot leave only
            # the structurally riskiest candidates on detector priors.
            return (not boundary_truncated, not supplemental)

        items = sorted(items, key=verification_priority)
        for start in range(0, len(items), 8):
            batch = items[start:start + 8]
            candidates = []
            for item in batch:
                bbox = item.get("bbox_xyxy_normalized")
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    width = float(item.get("width") or view["width"])
                    height = float(item.get("height") or view["height"])
                    x1, y1, x2, y2 = [float(value) for value in item["bbox_xyxy"]]
                    bbox = [x1 / width, y1 / height, x2 / width, y2 / height]
                concept = str(item.get("canonical_class", ""))
                candidates.append({
                    "candidate_bbox": [float(value) for value in bbox],
                    "query_concept": concept,
                    "visual_definition": str(
                        concept_lookup.get(concept, {}).get(
                            "visual_definition", concept
                        )
                    ).strip(),
                    "hard_negatives": list(
                        concept_lookup.get(concept, {}).get(
                            "hard_negatives", ()
                        )
                    )[:5],
                })

            verifications: dict[int, dict[str, Any]] = {}
            remaining = _remaining(deadline_monotonic)
            if remaining >= 5.0:
                try:
                    response = socket_request(
                        endpoint,
                        _qwen_payload(
                            acquisition_id=acquisition_id,
                            operation="verify_object_batch",
                            input_handles=[str(view["image_path"])],
                            parameters={
                                "image_path": str(view["image_path"]),
                                "candidates": candidates,
                            },
                        ),
                        max(1.0, min(remaining - 1.0, 45.0)),
                        deadline_monotonic=deadline_monotonic,
                    )
                    for row in response.get("metadata", {}).get("verifications", ()):
                        try:
                            index = int(row.get("index", -1))
                            normalized_index = index - 1 if 1 <= index <= len(batch) else index
                            if 0 <= normalized_index < len(batch):
                                verifications[normalized_index] = dict(row)
                        except (TypeError, ValueError):
                            continue
                except Exception as exc:
                    error_type = type(exc).__name__
                    # Qwen verification failure is not fatal - use detector priors
                    warnings.append(
                        f"verify_object_batch:{view_id}:{error_type}:{str(exc)[:120]}"
                    )
            else:
                warnings.append(f"verify_object_batch:{view_id}:deadline_reserve")

            for index, item in enumerate(batch):
                row = verifications.get(index)
                enriched = dict(item)
                if row is None:
                    semantic_probability = max(
                        0.45,
                        min(
                            0.78,
                            0.35
                            + 0.25 * float(item.get("detector_score", 0.5))
                            + 0.20 * float(item.get("sam_score", 0.5)),
                        ),
                    )
                    enriched.update({
                        "qwen_candidate_verdict": "unavailable",
                        "qwen_verified": False,
                        "semantic_probability": semantic_probability,
                    })
                else:
                    probability = max(
                        0.0,
                        min(1.0, float(row.get("target_probability", 0.0))),
                    )
                    confusers = {
                        str(name): max(0.0, min(1.0, float(value)))
                        for name, value in dict(
                            row.get("confuser_probabilities", {})
                        ).items()
                        if isinstance(value, (int, float))
                        and str(name).strip().lower() != concept.lower()
                    }
                    verdict = (
                        "confuser_wins"
                        if confusers and max(confusers.values()) >= probability
                        else "target_wins"
                        if probability > 1.0 - probability
                        else "inconclusive"
                    )
                    enriched.update({
                        "qwen_target_probability": probability,
                        "qwen_anchor_probability": float(row.get("anchor_probability", 0.0)),
                        "qwen_confuser_probabilities": confusers,
                        "qwen_rationale_tags": list(
                            row.get("rationale_tags", ())
                        ),
                        "qwen_candidate_verdict": verdict,
                        "qwen_verified": verdict == "target_wins",
                        "semantic_probability": probability,
                    })
                normalized_box = candidates[index]["candidate_bbox"]
                view_boundary_truncated = bool(
                    normalized_box[0] <= 0.0
                    or normalized_box[1] <= 0.0
                    or normalized_box[2] >= 1.0
                    or normalized_box[3] >= 1.0
                )
                # A box clipped by the perspective-view boundary does not
                # contain the complete visual object.  Preserve it for
                # high-recall discovery, but do not let a positive judgment
                # over that partial crop become semantic authority.  The
                # overlapping panorama views or a later acquisition can
                # still verify the candidate from a complete observation.
                if (
                    view_boundary_truncated
                    and enriched.get("qwen_candidate_verdict") == "target_wins"
                ):
                    enriched["qwen_candidate_verdict"] = "inconclusive"
                    enriched["qwen_verified"] = False
                    enriched["qwen_rationale_tags"] = list(dict.fromkeys([
                        *enriched.get("qwen_rationale_tags", ()),
                        "view_boundary_truncated",
                    ]))
                enriched["semantic_view_boundary_truncated"] = (
                    view_boundary_truncated
                )
                output.append(enriched)
    return output, warnings


def execute_perception_pipeline(
    *,
    panorama_path: Path,
    task_ir: Mapping[str, Any],
    scene_observation_entity_ids: Sequence[str],
    worker_endpoints: Mapping[str, str],
    output_dir: Path,
    timeout: float,
    acquisition_id: str = "acquisition",
    detection_threshold: float = 0.20,
    perspective_width: int = 1024,
    perspective_height: int = 768,
    perspective_yaws_deg: Sequence[float] = (0, 45, 90, 135, 180, 225, 270, 315),
    perception_mode: str = "initial_full",
    skip_semantic_verification: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    total_budget = max(1.0, float(timeout))
    deadline = started + total_budget
    panorama = cv2.imread(str(panorama_path), cv2.IMREAD_COLOR)
    if panorama is None:
        raise ValueError(f"panorama_unreadable:{panorama_path}")
    concepts = extract_task_concepts(task_ir, scene_observation_entity_ids)
    if not concepts:
        raise ValueError("task_conditioned_vocabulary_empty")

    view_dir = output_dir / "01_perspective_views"
    bundle = write_perspective_bundle(
        panorama,
        view_dir,
        yaws_deg=perspective_yaws_deg,
        pitches_deg=(0.0,),
        width=int(perspective_width),
        height=int(perspective_height),
        horizontal_fov_deg=90.0,
        panorama_vertical_fov_deg=120.0,
        optical_center_group=str(acquisition_id),
        view_id_prefix=str(acquisition_id),
        write_remap_arrays=True,
    )
    views = [dict(value) for value in bundle["views"]]
    views_manifest_path = Path(str(bundle["manifest_path"]))

    normalized_mode = str(perception_mode or "initial_full")
    grounding_fraction = 0.72 if normalized_mode == "emergency_targeted" else 0.60
    grounding_budget = max(5.0, min(120.0, total_budget * grounding_fraction))
    deadline_monotonic = deadline

    primary, grounding_metadata, qwen_error = _qwen_ground(
        acquisition_id=acquisition_id,
        views=views,
        concepts=concepts,
        question=str(task_ir.get("original_question", "")),
        endpoint=str(worker_endpoints["qwen"]),
        timeout=min(grounding_budget, max(1.0, _remaining(deadline))),
        deadline_monotonic=deadline_monotonic,
    )

    grounding_warnings: list[str] = []
    if qwen_error:
        grounding_warnings.append(qwen_error)

    for view_result in grounding_metadata.get("views", ()):
        if not bool(view_result.get("ok", True)):
            grounding_warnings.append(
                f"qwen_grounding_unrecoverable:{view_result.get('view_id', '')}"
            )
        for event in view_result.get("recovery_events", ()):
            grounding_warnings.append(
                f"qwen_grounding_recovered:{view_result.get('view_id', '')}:{event}"
            )

    # Qwen and YOLO form a recall-oriented proposal union.  Either detector can
    # be absent for one acquisition; the available proposals still enter
    # SAM2/SceneMemory and the graph/identity layers remove false positives.
    yolo_proposals, yolo_warning = _optional_yolo(
        views=views,
        concepts=concepts,
        endpoint=str(worker_endpoints.get("yolo", "@mast3r_yolo")),
        timeout=max(1.0, min(_remaining(deadline) * 0.30, 45.0)),
        threshold=detection_threshold,
    )
    if yolo_warning:
        grounding_warnings.append(f"yolo_supplement_unavailable:{yolo_warning}")

    proposals = _merge_optional_supplement(primary, yolo_proposals)
    proposals_path = output_dir / "02_task_proposals.json"
    proposals_path.write_text(json.dumps(proposals, indent=2) + "\n", encoding="utf-8")
    if not proposals:
        return {
            "status": "blocked",
            "perception_mode": normalized_mode,
            "reason": "no_recall_ensemble_proposals",
            "detections": [],
            "views": views,
            "concepts": concepts,
            "grounding_metadata": grounding_metadata,
            "proposals_path": str(proposals_path.resolve()),
            "views_manifest_path": str(views_manifest_path.resolve()),
            "warnings": [value for value in (*grounding_warnings, yolo_warning) if value],
            "elapsed_seconds": time.monotonic() - started,
        }

    remaining_before_sam = _remaining(deadline)
    if remaining_before_sam < 3.0:
        return {
            "status": "blocked",
            "perception_mode": normalized_mode,
            "reason": "deadline_before_sam2",
            "detections": [],
            "views": views,
            "concepts": concepts,
            "grounding_metadata": grounding_metadata,
            "proposals_path": str(proposals_path.resolve()),
            "views_manifest_path": str(views_manifest_path.resolve()),
            "warnings": [value for value in (*grounding_warnings, yolo_warning) if value],
            "elapsed_seconds": time.monotonic() - started,
        }

    masked, sam_metadata = _segment_all_views(
        views=views,
        detections=proposals,
        endpoint=str(worker_endpoints["sam2"]),
        output_dir=output_dir,
        timeout=max(1.0, min(remaining_before_sam - 1.0, 75.0)),
    )
    if not masked:
        return {
            "status": "system_failure",
            "perception_mode": normalized_mode,
            "reason": "sam2_no_valid_masks",
            "detections": [],
            "views": views,
            "concepts": concepts,
            "grounding_metadata": grounding_metadata,
            "sam_metadata": sam_metadata,
            "proposals_path": str(proposals_path.resolve()),
            "views_manifest_path": str(views_manifest_path.resolve()),
            "warnings": [value for value in (*grounding_warnings, yolo_warning) if value],
            "elapsed_seconds": time.monotonic() - started,
        }

    if skip_semantic_verification or _remaining(deadline) < 8.0:
        verified = []
        for item in masked:
            enriched = dict(item)
            detector_score = float(enriched.get("detector_score", 0.5))
            sam_score = float(enriched.get("sam_score", 0.5))
            enriched.update({
                "qwen_candidate_verdict": "grounding_only",
                "qwen_verified": False,
                "semantic_probability": max(
                    0.50, min(0.78, 0.38 + 0.22 * detector_score + 0.18 * sam_score)
                ),
            })
            verified.append(enriched)
        verify_warnings = [f"semantic_verification_skipped:{normalized_mode}"]
    else:
        verified, verify_warnings = _verify_by_view(
            acquisition_id=acquisition_id,
            detections=masked,
            views=views,
            concepts=concepts,
            endpoint=str(worker_endpoints["qwen"]),
            deadline_monotonic=deadline,
        )
    counts = Counter(str(value.get("canonical_class", "")) for value in verified)
    verified_path = output_dir / "04_verified_detections.json"
    verified_path.write_text(json.dumps(verified, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "completed",
        "perception_mode": normalized_mode,
        "detections": verified,
        "views": views,
        "concepts": concepts,
        "counts_by_class": dict(counts),
        "proposal_sources": {
            "qwen3vl": len(primary),
            "yolo_world": len(yolo_proposals),
            "union": len(proposals),
            "policy": "recall_first_graph_precision",
        },
        "grounding_metadata": grounding_metadata,
        "sam_metadata": sam_metadata,
        "proposals_path": str(proposals_path.resolve()),
        "verified_detections_path": str(verified_path.resolve()),
        "views_manifest_path": str(views_manifest_path.resolve()),
        "warnings": [value for value in (*grounding_warnings, yolo_warning, *verify_warnings) if value],
        "elapsed_seconds": time.monotonic() - started,
    }
