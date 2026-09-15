"""Local model adapters for the rebuilt perception and geometry chain.

The production perception path is GroundingDINO-Tiny box grounding followed
by SAM2.1 box refinement. Qwen is used for TaskIR compilation, unary
attribute reads, and small image-pair relation checks. DA3METRIC-LARGE is
called on demand for sparse depth. All checkpoint settings are local paths;
runtime model loading never resolves a Hub id or personal cache.

The runtime lock serializes model requests from the ROS worker and protects the
mutable SAM2.1 image predictor. Returned masks, boxes, and scores are copied
to CPU before the predictor can be reused.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .contracts import Detection, View


DEFAULT_CONFIG: dict[str, Any] = {
    "device": "cuda",
    "perception_backend": "grounding_dino_sam21",
    "qwen_max_pixels": 262_144,
    "qwen_parse_tokens": 1_536,
    "qwen_verify_tokens": 192,
    "category_batch_size": 4,
    "grounding_dino_box_threshold": 0.25,
    "grounding_dino_text_threshold": 0.25,
    "grounding_dino_concept_aliases": {
        "photo": "picture frame",
        "tv cabinet": "tv stand",
    },
    "sam21_config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "da3_process_res": 504,
    "da3_process_res_method": "upper_bound_resize",
}


class ModelRuntime:
    """Lazy local adapters for GroundingDINO-Tiny, SAM2.1, Qwen3-VL, and DA3."""

    def __init__(
        self,
        config: dict[str, Any],
        log_event: Callable[..., Any] | None = None,
    ) -> None:
        self.config = {**DEFAULT_CONFIG, **dict(config)}
        self.log_event = log_event
        self.device = str(self.config["device"])
        self._lock = threading.RLock()

        self._sam21_model: Any | None = None
        self._sam21_predictor: Any | None = None
        self._sam21_view_key: tuple[Any, ...] | None = None

        self._grounding_dino_model: Any | None = None
        self._grounding_dino_processor: Any | None = None
        self._grounding_dino_device: Any | None = None

        self._qwen_model: Any | None = None
        self._qwen_processor: Any | None = None
        self._qwen_device: Any | None = None
        self._compiler_tokenizer_data: Any | None = None

        self._da3_model: Any | None = None
        self._active_auxiliary_group: str | None = None
        self._phase_switch_sequence = 0
        self._category_batch_sequence = 0
        self._loaded_models: set[str] = set()
        self.last_depth_metadata: dict[str, Any] = {}
        self.last_resource_event: dict[str, Any] = {}


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile_task(self, question: str) -> dict[str, Any]:
        """Construct task semantics, then translate only detector vocabulary."""
        from .language_frontend import parse_question
        from .task_ir import parse_task_ir
        started = time.perf_counter()
        with self._lock:
            self._begin_request()
            input_shape = None
            try:
                task = parse_task_ir(json.dumps(parse_question(question)))
                task['instruction'] = question.strip()
                concepts = task['concepts']
                # The model sees category names only. It cannot change the
                # parsed goal, relationship, selected side, or action order.
                schema = {
                    'title': 'detector_vocabulary_v1', 'type': 'object',
                    'properties': {name: {'type': 'string', 'minLength': 1, 'maxLength': 48}
                                   for name in concepts},
                    'required': concepts, 'additionalProperties': False,
                }
                prompt = (
                    'Map each intrinsic semantic object category to a short physical '
                    'category suitable for an open-vocabulary object detector. '
                    'Keep the same physical object and use ordinary space-separated '
                    'English nouns. Remove appearance or functional modifiers when a '
                    'broader detector category is appropriate. Correct ordinary spelling '
                    'in the detector label. If no shorter physical category is suitable, '
                    'keep the full category. Return exactly one JSON object mapping each '
                    'provided category to its detector label, with no explanation. Categories: '
                    + json.dumps(concepts)
                )
                raw, input_shape = self._qwen_generate(
                    prompt, image=None, max_new_tokens=int(self.config['qwen_parse_tokens']),
                    json_schema=schema,
                )
                vocabulary = json.loads(raw)
                if set(vocabulary) != set(concepts) or any(
                        not isinstance(value, str) or not value.strip()
                        for value in vocabulary.values()):
                    raise ValueError('detector vocabulary does not cover the parsed categories')
                def apply_vocabulary(value):
                    if isinstance(value, dict):
                        if value.get('op') == 'filter_class':
                            value['visual_class'] = vocabulary[value['class']].strip().lower()
                        for child in value.values():
                            apply_vocabulary(child)
                    elif isinstance(value, list):
                        for child in value:
                            apply_vocabulary(child)
                apply_vocabulary(task)
                self._emit_resource('visual_vocabulary_generated', raw_response=raw,
                                    semantic_program_source='compositional_english_parser')
                self._emit_resource('task_ir_generated', raw_response=json.dumps(task),
                                    semantic_program_source='compositional_english_parser')
                return task
            finally:
                self._end_request(operation='compile_task', started=started, input_shape=input_shape)

    def detect(
        self,
        view: View,
        concepts: list[str],
        visual_queries: dict[str, str],
    ) -> list[Detection]:
        """Ground semantic concepts using their independent visual queries."""

        image = _validate_rgb_image(view.image_rgb, name="view.image_rgb")
        prompts = _normalise_concepts(concepts)
        if not prompts:
            return []
        if not isinstance(visual_queries, Mapping):
            raise TypeError("visual_queries must map semantic concepts to visual strings")
        visual_by_concept: dict[str, str] = {}
        for key, value in visual_queries.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("visual_queries keys and values must be strings")
            visual_by_concept[key.strip().casefold()] = value.strip().lower().rstrip(".").strip()
        missing = [concept for concept in prompts if concept.casefold() not in visual_by_concept]
        if missing:
            raise ValueError(
                "visual_queries is missing requested semantic concepts: "
                + ", ".join(repr(concept) for concept in missing)
            )
        if any(not visual_by_concept[concept.casefold()] for concept in prompts):
            raise ValueError("visual_queries values must be non-empty")
        backend = str(
            self.config.get("perception_backend", "grounding_dino_sam21")
        ).strip().lower()
        if backend != "grounding_dino_sam21":
            raise ValueError(
                "perception_backend must be the explicit production route "
                "'grounding_dino_sam21'"
            )

        started = time.perf_counter()
        with self._lock:
            self._begin_request()
            input_shape: tuple[int, ...] | None = None
            prompt_batch_shapes: list[list[int]] = []
            processed_prompt_batches = 0
            prompt_batch_size = 1
            try:
                self._activate_auxiliary_group("grounding_sam21")
                processor = self._ensure_grounding_dino()
                aliases = self.config.get("grounding_dino_concept_aliases", {})
                if not isinstance(aliases, Mapping):
                    raise TypeError(
                        "config['grounding_dino_concept_aliases'] must be a mapping"
                    )
                normalized_aliases: dict[str, str] = {}
                for key, value in aliases.items():
                    if not isinstance(key, str) or not isinstance(value, str):
                        raise TypeError(
                            "grounding_dino_concept_aliases keys and values must be strings"
                        )
                    alias = value.strip().lower().rstrip(".").strip()
                    if not alias:
                        raise ValueError(
                            "grounding_dino_concept_aliases values must be non-empty"
                        )
                    normalized_aliases[key.strip().casefold()] = alias
                semantic_queries = [
                    visual_by_concept[concept.casefold()] for concept in prompts
                ]
                detector_queries = [
                    normalized_aliases.get(query, query)
                    for query in semantic_queries
                ]
                prompt_texts = [query + "." for query in detector_queries]
                from PIL import Image

                model = self._grounding_dino_model
                if model is None:
                    raise RuntimeError("GroundingDINO model was not initialized")
                import torch

                boxes: list[list[float]] = []
                concepts_for_boxes: list[str] = []
                detector_scores: list[float] = []
                returned_labels: list[str | None] = []
                visual_queries_for_boxes: list[str] = []
                detector_queries_for_boxes: list[str] = []
                prompt_texts_for_boxes: list[str] = []
                prompt_batch_indices: list[int] = []
                prompt_item_indices: list[int] = []
                seen_boxes: set[tuple[str, tuple[float, ...]]] = set()
                H, W = image.shape[:2]
                for batch_start in range(0, len(prompts), prompt_batch_size):
                    batch_index = batch_start // prompt_batch_size
                    batch_concepts = prompts[batch_start : batch_start + prompt_batch_size]
                    batch_visual_queries = semantic_queries[
                        batch_start : batch_start + prompt_batch_size
                    ]
                    batch_detector_queries = detector_queries[
                        batch_start : batch_start + prompt_batch_size
                    ]
                    batch_prompt_texts = prompt_texts[
                        batch_start : batch_start + prompt_batch_size
                    ]
                    inputs = processor(
                        images=[Image.fromarray(image, mode="RGB") for _ in batch_concepts],
                        text=batch_prompt_texts,
                        return_tensors="pt",
                        padding=True,
                    )
                    batch_input_shape = _batch_input_shape(inputs)
                    input_shape = batch_input_shape
                    if batch_input_shape is not None:
                        prompt_batch_shapes.append(list(batch_input_shape))
                    inputs = _move_batch_to_device(inputs, self._grounding_dino_device)
                    with torch.inference_mode(), self._model_autocast():
                        outputs = model(**inputs)
                    target_sizes = [(int(H), int(W)) for _ in batch_concepts]
                    postprocessed = processor.post_process_grounded_object_detection(
                        outputs,
                        input_ids=inputs["input_ids"],
                        threshold=float(self.config["grounding_dino_box_threshold"]),
                        text_threshold=float(self.config["grounding_dino_text_threshold"]),
                        target_sizes=target_sizes,
                    )
                    if len(postprocessed) != len(batch_concepts):
                        raise RuntimeError(
                            "GroundingDINO batch result count disagrees with prompt count: "
                            f"prompts={len(batch_concepts)} results={len(postprocessed)}"
                        )
                    raw_labels_for_batch: list[list[str | None]] = []
                    batch_proposal_count = 0
                    for item_index, (concept, visual_query, detector_query, prompt_text, result) in enumerate(
                        zip(
                            batch_concepts,
                            batch_visual_queries,
                            batch_detector_queries,
                            batch_prompt_texts,
                            postprocessed,
                        )
                    ):
                        if not isinstance(result, Mapping):
                            raise RuntimeError(
                                "GroundingDINO postprocess returned an invalid batch result"
                            )
                        boxes_value = _copy_cpu_array(
                            result.get("boxes"), name="GroundingDINO boxes"
                        )
                        scores_value = _copy_cpu_array(
                            result.get("scores"), name="GroundingDINO scores"
                        )
                        if boxes_value is None or scores_value is None:
                            raise RuntimeError(
                                "GroundingDINO postprocess must return boxes and scores"
                            )
                        boxes_array = np.asarray(boxes_value, dtype=np.float32)
                        scores_array = np.asarray(scores_value, dtype=np.float32).reshape(-1)
                        if boxes_array.ndim != 2 or boxes_array.shape[1] != 4:
                            raise RuntimeError(
                                "GroundingDINO boxes must have shape (N,4), got "
                                f"{boxes_array.shape}"
                            )
                        if len(boxes_array) != len(scores_array):
                            raise RuntimeError(
                                "GroundingDINO result lengths disagree: "
                                f"boxes={len(boxes_array)} scores={len(scores_array)}"
                            )
                        labels_value = result.get("text_labels")
                        if labels_value is None:
                            raw_labels = [None] * len(boxes_array)
                        elif isinstance(labels_value, (str, bytes, bytearray)):
                            raw_labels = [str(labels_value)]
                        else:
                            try:
                                raw_labels = [str(label) for label in list(labels_value)]
                            except TypeError:
                                raw_labels = [str(labels_value)]
                        raw_labels_for_batch.append(raw_labels)
                        batch_proposal_count += len(boxes_array)
                        for proposal_index, box_value in enumerate(boxes_array):
                            box = np.asarray(box_value, dtype=np.float32)
                            score = float(scores_array[proposal_index])
                            if not np.all(np.isfinite(box)) or not np.isfinite(score):
                                raise RuntimeError(
                                    "GroundingDINO result contains non-finite values"
                                )
                            if box[2] <= box[0] or box[3] <= box[1]:
                                raise RuntimeError(
                                    "GroundingDINO result is not xyxy: "
                                    f"{box.tolist()}"
                                )
                            pixel_box = [float(value) for value in box.tolist()]
                            dedup_key = (concept.casefold(), tuple(pixel_box))
                            if dedup_key in seen_boxes:
                                continue
                            seen_boxes.add(dedup_key)
                            boxes.append(pixel_box)
                            concepts_for_boxes.append(concept)
                            detector_scores.append(score)
                            returned_labels.append(
                                raw_labels[proposal_index]
                                if proposal_index < len(raw_labels)
                                else None
                            )
                            visual_queries_for_boxes.append(visual_query)
                            detector_queries_for_boxes.append(detector_query)
                            prompt_texts_for_boxes.append(prompt_text)
                            prompt_batch_indices.append(batch_index)
                            prompt_item_indices.append(item_index)
                    self._emit_resource(
                        "grounding_output",
                        observation_id=view.observation_id,
                        view_id=view.view_id,
                        batch_index=batch_index,
                        batch_size=len(batch_concepts),
                        requested_semantic_concepts=list(batch_concepts),
                        visual_queries=list(batch_visual_queries),
                        detector_queries=list(batch_detector_queries),
                        prompt_texts=list(batch_prompt_texts),
                        raw_decoded_labels=raw_labels_for_batch,
                        input_shape=(
                            list(batch_input_shape)
                            if batch_input_shape is not None
                            else None
                        ),
                        proposal_count=batch_proposal_count,
                    )
                    processed_prompt_batches += 1

                if not boxes:
                    return []
                predictor = self._ensure_sam21()
                view_key = (
                    view.observation_id,
                    view.view_id,
                    float(view.stamp),
                    id(view.image_rgb),
                )
                if self._sam21_view_key != view_key:
                    with self._model_autocast():
                        predictor.set_image(image)
                    self._sam21_view_key = view_key

                # SAM2.1 accepts absolute xyxy boxes when normalize_coords=True.
                with self._model_autocast():
                    masks, sam_scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=np.asarray(boxes, dtype=np.float32),
                        multimask_output=False,
                        normalize_coords=True,
                    )
                # Copy all mutable predictor outputs before another image/query
                # can mutate its cached features.
                masks = _copy_cpu_array(masks, name="SAM2.1 masks")
                sam_scores = _copy_cpu_array(sam_scores, name="SAM2.1 scores")
                if masks is None or sam_scores is None:
                    raise RuntimeError("SAM2.1 returned no masks or scores")
                masks = _normalise_masks(masks, image.shape[:2])
                sam_scores = np.asarray(sam_scores, dtype=np.float32).reshape(-1)
                if len(masks) != len(boxes) or len(sam_scores) != len(boxes):
                    raise RuntimeError(
                        "SAM2.1 result lengths disagree: "
                        f"boxes={len(boxes)} masks={len(masks)} scores={len(sam_scores)}"
                    )
                return [
                    Detection(
                        observation_id=view.observation_id,
                        stamp=float(view.stamp),
                        view_id=view.view_id,
                        concept=concept,
                        box_2d=[float(value) for value in box],
                        mask=np.asarray(mask, dtype=bool).copy(),
                        score=float(detector_score),
                        attributes={
                            "source": "grounding-dino-tiny+sam2.1",
                            "detector_score": float(detector_score),
                            "sam_score": float(sam_score),
                            "sam_quality_score": float(sam_score),
                            "sam_mask_area_px": int(np.count_nonzero(mask)),
                            "sam_mask_area_fraction": float(
                                np.count_nonzero(mask) / float(H * W)
                            ),
                            "requested_semantic_concept": concept,
                            "visual_query": visual_query,
                            "detector_query": detector_query,
                            "grounding_label": raw_decoded_label,
                            "raw_decoded_label": raw_decoded_label,
                            "grounding_prompt": prompt_text,
                            "grounding_prompt_batch_index": int(prompt_batch_index),
                            "grounding_prompt_item_index": int(prompt_item_index),
                        },
                    )
                    for concept, box, detector_score, sam_score, visual_query, detector_query, raw_decoded_label, prompt_text, prompt_batch_index, prompt_item_index, mask in zip(
                        concepts_for_boxes,
                        boxes,
                        detector_scores,
                        sam_scores,
                        visual_queries_for_boxes,
                        detector_queries_for_boxes,
                        returned_labels,
                        prompt_texts_for_boxes,
                        prompt_batch_indices,
                        prompt_item_indices,
                        masks,
                    )
                ]
            finally:
                self._end_request(
                    operation="detect",
                    started=started,
                    input_shape=input_shape,
                    auxiliary_group=self._active_auxiliary_group,
                    prompt_batch_size=prompt_batch_size,
                    prompt_batch_count=processed_prompt_batches,
                    prompt_batch_shapes=prompt_batch_shapes,
                )

    def depth(self, view: View) -> np.ndarray:
        """Return DA3METRIC-LARGE depth in metres at ``view`` resolution."""

        image = _validate_rgb_image(view.image_rgb, name="view.image_rgb")
        K = _validate_intrinsics(view.intrinsics)
        started = time.perf_counter()
        with self._lock:
            self._begin_request()
            input_shape: tuple[int, ...] | None = None
            try:
                self._activate_auxiliary_group("da3")
                model = self._ensure_da3()
                process_res = int(self.config["da3_process_res"])
                process_res_method = str(self.config["da3_process_res_method"])
                prediction = model.inference(
                    [image],
                    intrinsics=np.asarray([K], dtype=np.float32),
                    process_res=process_res,
                    process_res_method=process_res_method,
                    use_ray_pose=False,
                    ref_view_strategy="first",
                    export_dir=None,
                    export_format="mini_npz",
                )
                raw_depth = np.asarray(prediction.depth)
                if raw_depth.ndim == 3:
                    raw_depth = raw_depth[0]
                if raw_depth.ndim != 2:
                    raise RuntimeError(
                        "DA3 prediction.depth must have shape (N,H,W) or (H,W), "
                        f"got {raw_depth.shape}"
                    )
                raw_depth = np.asarray(raw_depth, dtype=np.float32)
                if not np.all(np.isfinite(raw_depth)):
                    raise RuntimeError("DA3 returned non-finite canonical depth")
                out_h, out_w = (int(raw_depth.shape[0]), int(raw_depth.shape[1]))
                input_shape = (1, 1, 3, out_h, out_w)

                # DA3METRIC-LARGE emits canonical depth. Its official conversion
                # uses average focal length in the processed pixel coordinates.
                scale_x = out_w / float(image.shape[1])
                scale_y = out_h / float(image.shape[0])
                focal = 0.5 * (float(K[0, 0]) * scale_x + float(K[1, 1]) * scale_y)
                metric_depth = raw_depth * (focal / 300.0)
                metric_depth = _resize_float_depth(
                    metric_depth,
                    target_hw=(int(image.shape[0]), int(image.shape[1])),
                )
                if metric_depth.shape != image.shape[:2]:
                    raise RuntimeError(
                        "DA3 depth adapter returned the wrong resolution: "
                        f"{metric_depth.shape} != {image.shape[:2]}"
                    )
                if not np.all(np.isfinite(metric_depth)):
                    raise RuntimeError("DA3 metric conversion returned non-finite depth")
                self.last_depth_metadata = {
                    "model": "DA3METRIC-LARGE",
                    "view_id": view.view_id,
                    "canonical": True,
                    "conversion": "focal * net_output / 300",
                    "focal_pixels_processed": focal,
                    "input_shape": list(input_shape),
                    "output_resolution": [out_h, out_w],
                    "returned_resolution": [int(image.shape[0]), int(image.shape[1])],
                    "process_res": process_res,
                    "process_res_method": process_res_method,
                }
                return np.asarray(metric_depth, dtype=np.float32)
            finally:
                self._end_request(
                    operation="depth",
                    started=started,
                    input_shape=input_shape,
                    auxiliary_group=self._active_auxiliary_group,
                )

    def verify_pair(
        self,
        crop_rgb: np.ndarray,
        subject: str,
        relation: str,
        anchor: str,
    ) -> dict[str, Any]:
        """Ask Qwen one constrained local relation question."""

        image = _validate_rgb_image(crop_rgb, name="crop_rgb")
        for value, name in ((subject, "subject"), (relation, "relation"), (anchor, "anchor")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        prompt = (
            "Decide only the stated spatial relation in this image. Return one "
            "JSON object and no prose with this schema: "
            '{"verdict":"yes|no|unknown","attributes":{}}. '
            f"Subject: {subject.strip()}. Relation: {relation.strip()}. "
            f"Anchor: {anchor.strip()}. Use unknown when the crop lacks enough evidence."
        )
        return self._verify_crop(image, prompt, "verify_pair")

    def verify_category(self, crop_rgb: np.ndarray, concept: str) -> dict[str, Any]:
        return self.verify_categories([(crop_rgb, concept)])[0]

    def verify_categories(self, instances: list[tuple[np.ndarray, str]]) -> list[dict[str, Any]]:
        """Identify the surface owner without task labels, then match its type."""
        if not isinstance(instances, list):
            raise TypeError("instances must be a list of (crop_rgb, concept) pairs")
        batch_limit = int(self.config.get("category_batch_size", 4))
        if batch_limit <= 0:
            raise ValueError("config['category_batch_size'] must be positive")
        if len(instances) > batch_limit:
            raise ValueError(
                "verify_categories received more instances than the configured "
                f"batch boundary: {len(instances)} > {batch_limit}"
            )
        prompts: list[str] = []
        images: list[np.ndarray] = []
        for index, item in enumerate(instances):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError(
                    f"instances[{index}] must be a (crop_rgb, concept) pair"
                )
            crop, concept = item
            if not isinstance(concept, str) or not concept.strip():
                raise ValueError(
                    f"instances[{index}][1] concept must be a non-empty string"
                )
            images.append(_validate_rgb_image(crop, name="category_crop"))
            prompts.append(
                "Identify the complete physical artifact whose surface is marked by the red rectangle. "
                "The rectangle is an annotation and may cover only part of the artifact. Use the visible "
                "surrounding structure to identify the whole artifact. For a mounted display, name what is "
                "physically displayed and its medium together with its mount; do not reduce it to the mount "
                "alone. Do not name a separate background object or the things depicted inside an image. "
                "Furniture, object, shape, color and material alone are not specific object types. If no "
                "specific everyday type can be identified, use unknown. "
                "Return JSON only: {\"identity_status\":\"identified|unknown\","
                "\"physical_category\":\"complete artifact type including its visible function or display medium\","
                "\"visible_evidence\":\"one brief sentence\"}."
            )
        if not prompts:
            return []
        started = time.perf_counter()
        with self._lock:
            self._category_batch_sequence += 1
            invocation_order = self._category_batch_sequence
            self._begin_request()
            input_shape: tuple[int, ...] | None = None
            output_count = 0
            identity_ms = membership_ms = 0.0
            membership_count = 0
            try:
                phase_started = time.perf_counter()
                texts, input_shape = self._qwen_generate_batch(
                    prompts, images, max_new_tokens=int(self.config['qwen_verify_tokens']))
                identities = [self._physical_identity_result(text) for text in texts]
                identity_ms = (time.perf_counter() - phase_started) * 1000
                indices = [index for index, identity in enumerate(identities)
                           if identity['identity_status'] == 'identified']
                membership = [None] * len(identities)
                membership_count = len(indices)
                if indices:
                    from .category_semantics import MEMBERSHIP_SCHEMA, membership_prompt
                    match_prompts = [membership_prompt(identities[index], instances[index][1])
                                     for index in indices]
                    phase_started = time.perf_counter()
                    matches, _ = self._qwen_generate_batch(
                        match_prompts, [images[index] for index in indices],
                        max_new_tokens=int(self.config['qwen_verify_tokens']),
                        json_schema=MEMBERSHIP_SCHEMA)
                    membership_ms = (time.perf_counter() - phase_started) * 1000
                    for index, match in zip(indices, matches):
                        membership[index] = match
                results = [self._category_verification_result(identity, match)
                           for identity, match in zip(identities, membership)]
                output_count = len(results)
                return results
            finally:
                self._end_request(
                    operation='verify_categories',
                    started=started,
                    input_shape=input_shape,
                    batch_size=len(images),
                    image_shapes=[list(image.shape) for image in images],
                    invocation_order=invocation_order,
                    output_count=output_count,
                    identity_stage_ms=round(identity_ms, 3),
                    membership_stage_ms=round(membership_ms, 3),
                    membership_batch_size=membership_count,
                    identity_prompt_contains_task_category=False,
                )

    @staticmethod
    def _verification_result(raw_text: str) -> dict[str, Any]:
        parsed, error = _parse_json_object(raw_text)
        verdict = str(parsed.get('verdict', 'unknown')).strip().lower() if parsed else 'unknown'
        if parsed is None or verdict not in {'yes', 'no', 'unknown'}:
            return {'verdict': 'unknown', 'attributes': {'source': 'qwen3-vl-4b',
                    'parse_error': error or 'invalid_verdict', 'raw_response': raw_text[:500]}}
        attributes = parsed.get('attributes', {})
        if not isinstance(attributes, dict):
            attributes = {'model_attributes': attributes}
        return {'verdict': verdict, 'attributes': {**attributes, 'source': 'qwen3-vl-4b'}}

    @staticmethod
    def _physical_identity_result(raw_text: str) -> dict[str, Any]:
        parsed, error = _parse_json_object(raw_text)
        parsed = parsed or {}
        category = parsed.get('physical_category')
        category = category.strip().lower() if isinstance(category, str) else 'unknown'
        identified = parsed.get('identity_status') == 'identified' and category not in {'', 'unknown'}
        visible = parsed.get('visible_evidence', '')
        return {'identity_status': 'identified' if identified else 'unknown',
                'physical_category': category if identified else 'unknown',
                'visible_evidence': visible if isinstance(visible, str) else '',
                'parse_error': error, 'raw_response': raw_text[:1200]}

    @staticmethod
    def _category_verification_result(identity: dict, membership: str | None) -> dict[str, Any]:
        parsed, error = _parse_json_object(membership) if membership is not None else (None, None)
        parsed = parsed or {}
        verdict = parsed.get('verdict', 'unknown')
        if identity['identity_status'] != 'identified' or verdict not in {'yes', 'no', 'unknown'}:
            verdict = 'unknown'
        return {'verdict': verdict, 'attributes': {
            'source': 'qwen3-vl-4b',
            'identity_status': identity['identity_status'],
            'observed_category': identity['physical_category'],
            'visible_identity_evidence': identity['visible_evidence'],
            'identity_source': 'class_blind_current_crop',
            'identity_raw_response': identity['raw_response'],
            'identity_parse_error': identity['parse_error'],
            'membership_source': 'current_image_and_independent_artifact_owner',
            'membership_raw_response': membership,
            'membership_parse_error': error,
        }}

    def _verify_crop(self, image: np.ndarray, prompt: str, operation: str) -> dict[str, Any]:
        started = time.perf_counter()
        with self._lock:
            self._begin_request()
            input_shape: tuple[int, ...] | None = None
            try:
                raw_text, input_shape = self._qwen_generate(
                    prompt,
                    image=image,
                    max_new_tokens=int(self.config["qwen_verify_tokens"]),
                )
                return self._verification_result(raw_text)
            finally:
                self._end_request(
                    operation=operation,
                    started=started,
                    input_shape=input_shape,
                )

    def verify_attributes(
        self,
        crop_rgb: np.ndarray,
        concept: str,
        attributes: list[str],
    ) -> dict[str, Any]:
        """Read requested unary attributes from one bounded object crop.

        Qwen receives only the crop and a closed attribute vocabulary. Every
        requested key is returned; a missing, malformed, or unobserved value
        is represented by ``None`` so callers do not turn absent evidence into
        a guessed attribute.
        """

        image = _validate_rgb_image(crop_rgb, name="crop_rgb")
        if not isinstance(concept, str) or not concept.strip():
            raise ValueError("concept must be a non-empty string")
        if not isinstance(attributes, list):
            raise TypeError("attributes must be a list of strings")
        requested: list[str] = []
        seen: set[str] = set()
        for attribute in attributes:
            if not isinstance(attribute, str) or not attribute.strip():
                raise ValueError("attributes must contain non-empty strings")
            name = attribute.strip()
            if name not in seen:
                requested.append(name)
                seen.add(name)
        if not requested:
            return {}

        schema = json.dumps({name: "observed value or null" for name in requested})
        prompt = (
            f"Inspect the {concept.strip()} in this crop. Return exactly one JSON "
            "object and no prose. Use exactly these keys and set a value to null "
            f"when it is not visibly supported: {schema}"
        )
        started = time.perf_counter()
        with self._lock:
            self._begin_request()
            input_shape: tuple[int, ...] | None = None
            try:
                raw_text, input_shape = self._qwen_generate(
                    prompt,
                    image=image,
                    max_new_tokens=int(self.config["qwen_verify_tokens"]),
                )
                parsed, _ = _parse_json_object(raw_text)
                if parsed is None:
                    return {name: None for name in requested}
                observed = parsed.get("attributes", parsed)
                if not isinstance(observed, Mapping):
                    return {name: None for name in requested}
                observed_by_name = {
                    str(key).casefold(): value for key, value in observed.items()
                }
                return {
                    name: observed_by_name.get(name.casefold())
                    for name in requested
                }
            finally:
                self._end_request(
                    operation="verify_attributes",
                    started=started,
                    input_shape=input_shape,
                )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _activate_auxiliary_group(self, group: str) -> None:
        """Keep exactly one auxiliary model group resident beside Qwen."""

        if group not in {"grounding_sam21", "da3"}:
            raise ValueError(f"unsupported auxiliary model group: {group!r}")
        previous = self._active_auxiliary_group
        if previous == group:
            return

        started = time.perf_counter()
        moved_to_cpu: list[str] = []
        moved_to_device: list[str] = []
        sam_image_cache_cleared = False

        if group == "da3":
            if self._sam21_predictor is not None or self._sam21_view_key is not None:
                sam_image_cache_cleared = self._clear_sam21_image_cache()
            self._move_auxiliary_model(
                self._grounding_dino_model,
                "grounding-dino-tiny",
                "cpu",
                moved_to_cpu,
            )
            self._move_auxiliary_model(
                self._sam21_model,
                "sam2.1",
                "cpu",
                moved_to_cpu,
            )
            self._move_auxiliary_model(
                self._da3_model,
                "da3metric-large",
                self.device,
                moved_to_device,
            )
        else:
            self._move_auxiliary_model(
                self._da3_model,
                "da3metric-large",
                "cpu",
                moved_to_cpu,
            )
            self._move_auxiliary_model(
                self._grounding_dino_model,
                "grounding-dino-tiny",
                self.device,
                moved_to_device,
            )
            self._move_auxiliary_model(
                self._sam21_model,
                "sam2.1",
                self.device,
                moved_to_device,
            )

        self._active_auxiliary_group = group
        if self._grounding_dino_model is not None:
            self._grounding_dino_device = _module_device(
                self._grounding_dino_model,
                fallback=self.device,
            )
        cache_released = self._release_cuda_cache()
        self._phase_switch_sequence += 1
        self._emit_resource(
            "phase_group_switch",
            from_group=previous,
            to_group=group,
            switch_sequence=self._phase_switch_sequence,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            moved_to_cpu=moved_to_cpu,
            moved_to_device=moved_to_device,
            sam_image_cache_cleared=sam_image_cache_cleared,
            cuda_cache_released=cache_released,
            models_loaded_once=sorted(self._loaded_models),
        )

    def _move_auxiliary_model(
        self,
        model: Any | None,
        name: str,
        target: str,
        moved: list[str],
    ) -> None:
        if model is None:
            return
        import torch

        target_device = torch.device(target)
        current = _module_device(model, fallback=target)
        try:
            current_device = torch.device(current)
        except Exception:
            current_device = None
        if (
            current_device is not None
            and current_device.type == target_device.type
            and (
                target_device.index is None
                or current_device.index == target_device.index
            )
        ):
            return
        model.to(target_device)
        moved.append(name)

    def _clear_sam21_image_cache(self) -> bool:
        predictor = self._sam21_predictor
        had_state = self._sam21_view_key is not None
        if predictor is not None:
            reset_predictor = getattr(predictor, "reset_predictor", None)
            if not callable(reset_predictor):
                raise RuntimeError(
                    "SAM2.1 predictor does not expose reset_predictor()"
                )
            reset_predictor()
            had_state = True
        self._sam21_view_key = None
        return had_state

    def _release_cuda_cache(self) -> bool:
        if not self.device.startswith("cuda"):
            return False
        import torch

        if not torch.cuda.is_available():
            return False
        torch.cuda.empty_cache()
        return True

    def _ensure_grounding_dino(self) -> Any:
        if (
            self._grounding_dino_model is not None
            and self._grounding_dino_processor is not None
        ):
            return self._grounding_dino_processor
        checkpoint = _require_local_path(self.config, "grounding_dino_checkpoint")
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except Exception as exc:
            raise RuntimeError(
                "GroundingDINO dependencies are unavailable; install transformers"
            ) from exc
        try:
            processor = AutoProcessor.from_pretrained(
                str(checkpoint),
                local_files_only=True,
            )
            detector_config = AutoConfig.from_pretrained(
                str(checkpoint),
                local_files_only=True,
            )
            # The native Transformers implementation otherwise attempts to
            # compile the optional custom CUDA kernels during model creation.
            detector_config.disable_custom_kernels = True
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                str(checkpoint),
                config=detector_config,
                local_files_only=True,
                dtype=torch.float32,
            )
            model = model.to(self.device)
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                "GroundingDINO-Tiny could not be loaded from local checkpoint: "
                f"{checkpoint}"
            ) from exc
        self._grounding_dino_model = model
        self._grounding_dino_processor = processor
        self._grounding_dino_device = _module_device(model, fallback=self.device)
        self._loaded_models.add("grounding-dino-tiny")
        self._emit_resource(
            "cold_start",
            model="grounding-dino-tiny",
            checkpoint=str(checkpoint),
            weight_dtype="float32",
            disable_custom_kernels=True,
            load_once=True,
        )
        return processor

    def _ensure_sam21(self) -> Any:
        if self._sam21_model is not None and self._sam21_predictor is not None:
            return self._sam21_predictor
        try:
            checkpoint = _require_local_path(self.config, "sam21_checkpoint")
        except (ValueError, FileNotFoundError) as exc:
            raise RuntimeError(f"SAM2.1 checkpoint is unavailable: {exc}") from exc
        config_name = self._sam21_config_name()
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise RuntimeError(
                "SAM2.1 is unavailable; install the official sam2 package in the AI image"
            ) from exc
        try:
            with self._cuda_device_context():
                model = build_sam2(config_name, str(checkpoint), device=self.device)
            predictor = SAM2ImagePredictor(model)
        except Exception as exc:
            raise RuntimeError(
                "SAM2.1 checkpoint/config could not be loaded: "
                f"checkpoint={checkpoint}, config={config_name}"
            ) from exc
        self._sam21_model = model
        self._sam21_predictor = predictor
        self._loaded_models.add("sam2.1")
        self._emit_resource(
            "cold_start",
            model="sam2.1",
            checkpoint=str(checkpoint),
            config=config_name,
            load_once=True,
        )
        return predictor

    def _ensure_qwen(self) -> Any:
        if self._qwen_model is not None and self._qwen_processor is not None:
            return self._qwen_model
        checkpoint = _require_local_path(self.config, "qwen_checkpoint")
        try:
            import torch
            from transformers import (
                AutoProcessor,
                Qwen3VLForConditionalGeneration,
            )
        except Exception as exc:
            raise RuntimeError(
                "Qwen3-VL dependencies are unavailable; install transformers and "
                "accelerate"
            ) from exc
        device_map = self.config.get("qwen_device_map", "auto")
        try:
            processor = AutoProcessor.from_pretrained(
                str(checkpoint),
                local_files_only=True,
                max_pixels=int(self.config["qwen_max_pixels"]),
            )
            processor.tokenizer.padding_side = 'left'
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(checkpoint),
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map=device_map,
            )
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                "Qwen3-VL-4B BF16 could not be loaded from local checkpoint: "
                f"{checkpoint}"
            ) from exc
        self._qwen_model = model
        self._qwen_processor = processor
        self._qwen_device = _module_device(model, fallback=self.device)
        self._loaded_models.add("qwen3-vl-4b")
        self._emit_resource(
            "cold_start",
            model="qwen3-vl-4b",
            checkpoint=str(checkpoint),
            quantization="none",
            dtype="bfloat16",
            device_map=device_map,
            load_once=True,
        )
        return model

    def _ensure_da3(self) -> Any:
        if self._da3_model is not None:
            return self._da3_model
        checkpoint = _require_local_path(self.config, "da3_checkpoint")
        try:
            from depth_anything_3.api import DepthAnything3
        except Exception as exc:
            raise RuntimeError(
                "Depth Anything 3 is unavailable; install the prepared source tree"
            ) from exc
        try:
            model = DepthAnything3.from_pretrained(
                str(checkpoint),
                local_files_only=True,
            )
            model = model.to(self.device)
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                "DA3METRIC-LARGE could not be loaded from local checkpoint: "
                f"{checkpoint}"
            ) from exc
        self._da3_model = model
        self._loaded_models.add("da3metric-large")
        self._emit_resource(
            "cold_start",
            model="da3metric-large",
            checkpoint=str(checkpoint),
            output="canonical_depth_scaled_with_processed_focal",
            load_once=True,
        )
        return model

    def _qwen_generate(
        self,
        prompt: Any,
        *,
        image: np.ndarray | None,
        max_new_tokens: int,
        json_schema: Mapping[str, Any] | None = None,
    ) -> tuple[str, tuple[int, ...] | None]:
        decoded, shape = self._qwen_generate_batch(
            [prompt],
            [image],
            max_new_tokens=max_new_tokens,
            json_schema=json_schema,
        )
        return decoded[0], shape

    def _qwen_generate_batch(
        self,
        prompts: list,
        images: list[np.ndarray | None],
        *,
        max_new_tokens: int,
        json_schema: Mapping[str, Any] | None = None,
    ) -> tuple[list[str], tuple[int, ...] | None]:
        if not isinstance(prompts, list) or not isinstance(images, list):
            raise TypeError("Qwen batch prompts and images must both be lists")
        if len(prompts) != len(images):
            raise ValueError(
                "Qwen batch prompts and images must have one-to-one lengths: "
                f"prompts={len(prompts)} images={len(images)}"
            )
        if not prompts:
            raise ValueError("Qwen batch must contain at least one prompt")
        model = self._ensure_qwen()
        processor = self._qwen_processor
        if processor is None:
            raise RuntimeError("Qwen processor was not initialized")
        messages = [_as_messages(prompt, image=image) for prompt, image in zip(prompts, images)]
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            input_shape = _batch_input_shape(inputs)
            inputs = _move_batch_to_device(inputs, self._qwen_device)
            import torch

            generation_kwargs: dict[str, Any] = {}
            if json_schema is not None:
                if not isinstance(json_schema, Mapping):
                    raise TypeError("json_schema must be a mapping")
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import (
                    build_token_enforcer_tokenizer_data,
                    build_transformers_prefix_allowed_tokens_fn,
                )

                # Build the parser for this request, while retaining the
                # tokenizer prefix tree across TaskIR and category schemas.
                # The parser carries the schema state and must not be reused
                # between independent generations or batch rows.
                parser = JsonSchemaParser(dict(json_schema))
                if self._compiler_tokenizer_data is None:
                    self._compiler_tokenizer_data = (
                        build_token_enforcer_tokenizer_data(processor.tokenizer)
                    )
                    self._emit_resource("compiler_tokenizer_ready", load_once=True)
                generation_kwargs["prefix_allowed_tokens_fn"] = (
                    build_transformers_prefix_allowed_tokens_fn(
                        self._compiler_tokenizer_data,
                        parser,
                    )
                )
                self._emit_resource(
                    "compiler_grammar_enabled",
                    schema=_json_schema_label(json_schema),
                )

            with torch.inference_mode(), self._model_autocast():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=False,
                    **generation_kwargs,
                )
            if len(generated_ids) != len(prompts):
                raise RuntimeError(
                    "Qwen3-VL returned an incomplete generated batch: "
                    f"expected={len(prompts)} got={len(generated_ids)}"
                )
            input_ids = inputs["input_ids"]
            trimmed = [
                output_ids[len(input_ids_row) :].detach().to("cpu")
                for input_ids_row, output_ids in zip(input_ids, generated_ids)
            ]
            decoded = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:
            raise RuntimeError("Qwen3-VL generation failed") from exc
        if len(decoded) != len(prompts):
            raise RuntimeError("Qwen3-VL returned an incomplete batch")
        return [str(text) for text in decoded], input_shape

    def _begin_request(self) -> None:
        try:
            import torch

            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        except Exception:
            pass

    def _end_request(
        self,
        *,
        operation: str,
        started: float,
        input_shape: Sequence[int] | None,
        **fields: Any,
    ) -> None:
        self._emit_resource(
            "request",
            operation=operation,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            input_shape=list(input_shape) if input_shape is not None else None,
            **fields,
        )

    def _emit_resource(self, event: str, **fields: Any) -> None:
        snapshot = _resource_snapshot(self.device)
        payload = {
            "phase": event,
            "loaded_models": sorted(self._loaded_models),
            **snapshot,
            **fields,
        }
        self.last_resource_event = {"event": event, **payload}
        if self.log_event is not None:
            try:
                self.log_event(event, **payload)
            except Exception:
                pass

    def _sam21_config_name(self) -> str:
        configured = self.config.get(
            "sam21_config", "configs/sam2.1/sam2.1_hiera_b+.yaml"
        )
        if not isinstance(configured, (str, os.PathLike)) or not str(configured):
            raise ValueError("config['sam21_config'] must name a SAM2.1 config")
        normalized = str(configured).replace("\\", "/")
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        marker = "/configs/"
        if marker in normalized:
            return "configs/" + normalized.split(marker, 1)[1].lstrip("/")
        return normalized

    @contextlib.contextmanager
    def _cuda_device_context(self) -> Iterable[None]:
        import torch

        if self.device.startswith("cuda") and torch.cuda.is_available():
            with torch.cuda.device(torch.device(self.device)):
                yield
            return
        yield

    @contextlib.contextmanager
    def _model_autocast(self, dtype: Any | None = None) -> Iterable[None]:
        import torch

        dtype = torch.bfloat16 if dtype is None else dtype
        if (
            self.device.startswith("cuda")
            and torch.cuda.is_available()
            and dtype in (torch.float16, torch.bfloat16)
        ):
            with torch.autocast(device_type="cuda", dtype=dtype):
                yield
            return
        yield


def _require_local_path(config: Mapping[str, Any], key: str) -> Path:
    value = config.get(key)
    if value is None or not isinstance(value, (str, os.PathLike)):
        raise ValueError(
            f"config[{key!r}] must be an existing local checkpoint directory/file"
        )
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Local model asset for {key} does not exist: {path}. "
            "Run prepare_assets.py before starting the runtime."
        )
    return path


def _validate_rgb_image(image: Any, *, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4) or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{name} must be a non-empty HxWx3/4 RGB array, got {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and np.isfinite(array).all() and float(array.max()) <= 1.0:
            array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _validate_intrinsics(intrinsics: Any) -> np.ndarray:
    K = np.asarray(intrinsics, dtype=np.float32)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError(f"view.intrinsics must be a finite 3x3 matrix, got {K.shape}")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("view.intrinsics must have positive focal lengths")
    return K.copy()


def _normalise_concepts(concepts: Sequence[str]) -> list[str]:
    if not isinstance(concepts, (list, tuple)):
        raise TypeError("concepts must be a list of strings")
    result: list[str] = []
    seen: set[str] = set()
    for concept in concepts:
        if not isinstance(concept, str) or not concept.strip():
            raise ValueError("concepts must contain non-empty strings")
        normalized = concept.strip()
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _copy_cpu_array(value: Any, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().to("cpu")
        if getattr(value, "is_floating_point", lambda: False)():
            value = value.float()
        return value.numpy().copy()
    return np.asarray(value).copy()


def _normalise_masks(masks: np.ndarray, hw: tuple[int, int]) -> list[np.ndarray]:
    array = np.asarray(masks)
    H, W = hw
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1:] != (H, W):
        raise RuntimeError(
            f"SAM2.1 masks must be (N,{H},{W}) or (N,1,{H},{W}), got {array.shape}"
        )
    return [np.asarray(mask, dtype=bool).copy() for mask in array]


def _resize_float_depth(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if tuple(depth.shape) == tuple(target_hw):
        return np.asarray(depth, dtype=np.float32).copy()
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(np.asarray(depth, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
    return resized[0, 0].numpy().astype(np.float32, copy=True)


def _as_messages(prompt: Any, *, image: np.ndarray | None) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        content: list[dict[str, Any]] = []
        if image is not None:
            from PIL import Image

            content.append({"type": "image", "image": Image.fromarray(image, mode="RGB")})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]
    if isinstance(prompt, Mapping):
        messages = [dict(prompt)]
    elif isinstance(prompt, Sequence) and not isinstance(prompt, (bytes, bytearray, str)):
        messages = [dict(message) for message in prompt]
    else:
        raise TypeError(
            "prompt must be a string or chat messages, got "
            f"{type(prompt).__name__}"
        )
    if image is None:
        return messages
    from PIL import Image

    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            message["content"] = [
                {"type": "image", "image": Image.fromarray(image, mode="RGB")},
                *list(content),
            ]
            return messages
    return [
        *messages,
        {
            "role": "user",
            "content": [{"type": "image", "image": Image.fromarray(image, mode="RGB")}],
        },
    ]


def _parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].lstrip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, None
    return None, "response did not contain a valid JSON object"


def _module_device(module: Any, *, fallback: str) -> Any:
    try:
        import torch

        for parameter in module.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        for buffer in module.buffers():
            if buffer.device.type != "meta":
                return buffer.device
        return torch.device(fallback)
    except Exception:
        return fallback


def _move_batch_to_device(batch: Any, device: Any, dtype: Any | None = None) -> Any:
    if hasattr(batch, "to"):
        try:
            return batch.to(device, dtype) if dtype is not None else batch.to(device)
        except TypeError:
            # Older BatchFeature implementations accept only the device. The
            # mapping path below still casts floating tensors explicitly.
            pass
    if hasattr(batch, "items"):
        values = {}
        for key, value in batch.items():
            if hasattr(value, "to"):
                if dtype is not None and getattr(value, "is_floating_point", lambda: False)():
                    values[key] = value.to(device=device, dtype=dtype)
                else:
                    values[key] = value.to(device)
            else:
                values[key] = value
        try:
            return batch.__class__(values)
        except Exception:
            return values
    raise TypeError(f"processor returned unsupported batch type {type(batch).__name__}")


def _batch_input_shape(batch: Any) -> tuple[int, ...] | None:
    try:
        for key in ("pixel_values", "input_ids"):
            value = batch.get(key)
            if value is not None and hasattr(value, "shape"):
                return tuple(int(item) for item in value.shape)
    except Exception:
        pass
    return None


def _resource_snapshot(device: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "cuda_peak_bytes": None,
        "cuda_current_bytes": None,
        "cuda_reserved_bytes": None,
    }
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch_device = torch.device(device)
            snapshot.update(
                {
                    "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(torch_device)),
                    "cuda_current_bytes": int(torch.cuda.memory_allocated(torch_device)),
                    "cuda_reserved_bytes": int(torch.cuda.memory_reserved(torch_device)),
                }
            )
    except Exception:
        pass
    try:
        import psutil

        snapshot["rss_bytes"] = int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        snapshot["rss_bytes"] = None
    return snapshot


def _json_schema_label(schema: Mapping[str, Any]) -> str:
    """Return a stable diagnostic label without inspecting schema contents."""

    for key in ("$id", "title", "schema_version"):
        value = schema.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "json_schema"


__all__ = ["DEFAULT_CONFIG", "ModelRuntime"]
