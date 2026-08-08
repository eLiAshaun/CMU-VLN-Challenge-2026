"""Pinned, lazy local Qwen3-VL grounding and candidate verification.

One request verifies one detector candidate from an object crop plus a wider
context crop.  It may also report how many query instances are contained by
that exact detector box so a group proposal is not mistaken for one physical
object.  This is local proposal structure, never a scene-level answer.  Model
weights are loaded only when the first request arrives so the worker can be
scheduled independently from the ROS vision process.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

from .semantic_verifier import SemanticVerification
from .relation_types import (
    ObjectGroundedRelationResult,
)


def _normalized_bbox(value) -> tuple[float, float, float, float]:
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        raise ValueError("candidate_bbox_must_have_four_numbers") from None
    if not (
        0.0 <= x1 < x2 <= 1.0
        and 0.0 <= y1 < y2 <= 1.0
    ):
        raise ValueError("candidate_bbox_must_be_normalized_xyxy")
    return x1, y1, x2, y2


def _candidate_images_from_pil(source, normalized_bbox):
    """Return object and context PIL crops with a context-only box overlay."""
    from PIL import Image, ImageDraw

    if not isinstance(source, Image.Image):
        raise TypeError("qwen3vl_source_must_be_pil_image")
    source = source.convert("RGB")
    width, height = source.size
    x1, y1, x2, y2 = _normalized_bbox(normalized_bbox)
    box = (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )
    box_width = max(2, box[2] - box[0])
    box_height = max(2, box[3] - box[1])

    object_pad_x = max(3, int(round(0.25 * box_width)))
    object_pad_y = max(3, int(round(0.25 * box_height)))
    object_bounds = (
        max(0, box[0] - object_pad_x),
        max(0, box[1] - object_pad_y),
        min(width, box[2] + object_pad_x),
        min(height, box[3] + object_pad_y),
    )

    context_pad_x = max(32, int(round(3.0 * box_width)))
    context_pad_y = max(32, int(round(3.0 * box_height)))
    context_bounds = (
        max(0, box[0] - context_pad_x),
        max(0, box[1] - context_pad_y),
        min(width, box[2] + context_pad_x),
        min(height, box[3] + context_pad_y),
    )
    object_crop = source.crop(object_bounds)
    context_crop = source.crop(context_bounds)
    context_box = (
        box[0] - context_bounds[0],
        box[1] - context_bounds[1],
        box[2] - context_bounds[0],
        box[3] - context_bounds[1],
    )
    draw = ImageDraw.Draw(context_crop)
    stroke = max(2, int(round(min(context_crop.size) / 120.0)))
    draw.rectangle(context_box, outline=(255, 32, 32), width=stroke)
    return object_crop, context_crop


def _candidate_images(image_path: str, normalized_bbox):
    from PIL import Image

    return _candidate_images_from_pil(
        Image.open(str(image_path)), normalized_bbox
    )


def _request_source_image(parameters):
    from PIL import Image

    image_path = str(parameters.get("image_path", "")).strip()
    if image_path:
        if not Path(image_path).is_file():
            raise ValueError("qwen3vl_image_path_missing")
        return Image.open(image_path).convert("RGB")

    shared = parameters.get("shared_memory")
    if not isinstance(shared, Mapping):
        raise ValueError("qwen3vl_image_input_missing")
    required = {"name", "shape", "dtype"}
    if set(shared) != required:
        raise ValueError("qwen3vl_shared_memory_schema_invalid")
    from .shared_memory_io import SharedArrayStore

    image = SharedArrayStore().get(
        str(shared["name"]),
        tuple(int(value) for value in shared["shape"]),
        str(shared["dtype"]),
        unlink=False,
    )
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("qwen3vl_shared_image_must_be_rgb")
    if str(image.dtype) != "uint8":
        raise ValueError("qwen3vl_shared_image_must_be_uint8")
    return Image.fromarray(image, mode="RGB")


def _verification_prompt(
    *,
    operation: str,
    query_concept: str,
    anchor_concept: str,
    hard_negatives: Sequence[str],
) -> str:
    if operation == "verify_anchor":
        negatives = ", ".join(
            str(item).strip()
            for item in hard_negatives
            if str(item).strip()
        ) or "none supplied"
        return (
            "You are a calibrated visual relation-anchor verifier. "
            "Image 1 is a tight crop of one detector-proposed physical "
            "anchor. Image 2 is wider context; the red rectangle encloses "
            "that exact anchor. Candidate-and-relation concept: "
            f"{query_concept!r}. Required context group: "
            f"{anchor_concept!r}. Explicit hard negatives: {negatives}. "
            "Inspect the wider context, including the wall area above and "
            "behind the red rectangle. target_probability is confidence "
            "that the red-rectangle object has the requested physical class. "
            "anchor_probability is confidence that this exact object, not a "
            "different nearby object, satisfies the requested spatial "
            "relation to the required context group. It must be low when the "
            "context is absent, only one isolated context member is visible, "
            "or the group belongs to a different wall/object. "
            "contained_instance_count is the number from 0 through 10 of "
            "visually distinct required-context members supporting that "
            "relation; count_confidence is confidence in that group count. "
            "Do not answer the scene-wide numerical question. Return exactly "
            "one JSON object and no markdown with exactly these keys: "
            '{"target_probability":0.0,"anchor_probability":0.0,'
            '"confuser_probabilities":{},"rationale_tags":[],'
            '"contained_instance_count":0,"count_confidence":0.0}. '
            "Probabilities must be calibrated numbers in [0,1]."
        )
    if operation == "count_on_anchor":
        negatives = ", ".join(
            str(item).strip()
            for item in hard_negatives
            if str(item).strip()
        ) or "none supplied"
        return (
            "You are a calibrated anchor-local visual inventory verifier. "
            "Image 1 is a crop of the local inventory region around one "
            "detector-proposed support object. Image 2 is wider context; "
            "the red rectangle covers that support and the usable surface "
            "immediately above it. Query object concept: "
            f"{query_concept!r}. Support concept: {anchor_concept!r}. "
            f"Explicit hard negatives for the query object: {negatives}. "
            "Count visually distinct query-object instances that are on, "
            "attached to, or immediately supported by that one support "
            "object inside the red ROI. The support may occupy the lower "
            "part of the rectangle; do not require its whole body to fill "
            "the ROI. Trace physical bodies, not screen "
            "content: two back-to-back monitor bodies are two instances even "
            "when their outlines overlap or form one V shape. Count a rear "
            "housing as an instance only when its separate frame, casing, "
            "stand, or mount is visibly distinguishable. Exclude objects on "
            "other support surfaces and "
            "exclude every hard negative. Do not infer hidden objects, do "
            "not use a typical-layout prior, and do not answer any scene-wide "
            "question. contained_instance_count is this measured local count "
            "from 0 through 10. count_confidence is confidence in the local "
            "count. target_probability is confidence that at least one query "
            "object is present in the ROI; anchor_probability is confidence "
            "that the named support is the red-ROI support. "
            "Return exactly one JSON object and no markdown with exactly "
            'these keys: {"target_probability":0.0,'
            '"anchor_probability":0.0,"confuser_probabilities":{},'
            '"rationale_tags":[],"contained_instance_count":0,'
            '"count_confidence":0.0}. Probabilities must be calibrated '
            "numbers in [0,1]."
        )
    role = "target object" if operation == "verify_object" else "anchor object"
    negatives = ", ".join(
        str(item).strip() for item in hard_negatives if str(item).strip()
    ) or "none supplied"
    return (
        "You are a calibrated visual candidate verifier. "
        "Image 1 is a tight detector-candidate crop. Image 2 is a wider "
        "context crop; the exact candidate is enclosed by the red rectangle. "
        f"Verify only that one {role}. Query concept: {query_concept!r}. "
        f"Anchor concept: {anchor_concept!r}. Explicit hard negatives: "
        f"{negatives}. Treat the target and every hard negative as mutually "
        "exclusive visual classes. A semantically related hard negative is "
        "not the target. Use visible shape, mounting, text, material, and "
        "requested attributes; do not accept a candidate merely because the "
        "query words appear in the prompt. For every hard negative that is "
        "visually supported with probability above 0.5, include that exact "
        "supplied name as a key in confuser_probabilities. Omit every "
        "unsupported or <=0.5 hard negative; never emit zero-valued confuser "
        "keys, and emit at most the three strongest confusers. "
        "contained_instance_count is the number from 0 through 10 of visually "
        "distinct query-concept instances whose visible body or frame is at "
        "least half inside the red candidate rectangle. Use 1 for one object, "
        "and use more than 1 only when this exact detector candidate is an "
        "instance group. count_confidence is confidence in that local count. "
        "Do not count outside the red rectangle and do not answer the scene "
        "question. Treat blur or insufficient pixels as uncertainty, not "
        "false. "
        "Return exactly one JSON object and no markdown with exactly these "
        'keys: {"target_probability":0.0,"anchor_probability":0.0,'
        '"confuser_probabilities":{},"rationale_tags":[],'
        '"contained_instance_count":1,"count_confidence":0.0}. '
        "Probabilities must be calibrated numbers in [0,1]."
    )


def _strict_semantic_json(text: str) -> SemanticVerification:
    """Extract the first complete JSON object and validate the strict schema.

    Quantized instruction models occasionally prepend a sentence or emit a
    second example object even when asked for JSON-only output.  The previous
    greedy ``{.*}`` capture joined both objects into invalid JSON and discarded
    an otherwise valid calibrated verdict.  ``raw_decode`` preserves strict
    schema validation while tolerating harmless surrounding text.
    """
    value = str(text)
    decoder = json.JSONDecoder()
    saw_object_start = False
    last_schema_error: Exception | None = None
    for index, character in enumerate(value):
        if character != "{":
            continue
        saw_object_start = True
        try:
            payload, _end = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        try:
            return SemanticVerification.from_json(payload)
        except (TypeError, ValueError) as exc:
            last_schema_error = exc
    if last_schema_error is not None:
        raise ValueError("qwen3vl_json_schema_invalid") from last_schema_error
    if saw_object_start:
        raise ValueError("qwen3vl_json_invalid")
    raise ValueError("qwen3vl_json_object_missing")


def _grounding_prompt(
    concepts: Sequence[Mapping[str, object]],
    *,
    view_id: str,
    scene_context: str,
    maximum: int,
) -> str:
    specification = []
    for concept in concepts:
        class_name = str(concept["class_name"])
        aliases = ", ".join(str(value) for value in concept.get("aliases", ())) or "none"
        context = json.dumps(
            list(concept.get("supporting_context", ())),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        negatives = ", ".join(
            str(value) for value in concept.get("hard_negatives", ())
        ) or "none"
        specification.append(
            f"- label {class_name!r}; aliases: {aliases}; required supporting "
            f"context: {context}; exclude: {negatives}"
        )
    return (
        "Locate every visible physical instance that belongs to the categories "
        "listed below. Report bbox coordinates in Qwen3-VL's native JSON format: "
        "a JSON list of objects like "
        '[{"bbox_2d":[x1,y1,x2,y2],"label":"category"}]. '
        "bbox_2d coordinates are integer relative coordinates from 0 to 1000. "
        "Use exactly one of the supplied label names in each label field. Return [] "
        "when none is visible. Do not output confidence scores, prose, markdown, "
        "duplicate boxes, screen content, reflections, or inferred hidden objects. "
        "A segmentation model will refine the coarse boxes. Apply the required "
        "supporting context and exclusions when choosing instances. Return at most "
        f"{maximum} boxes.\n" + "\n".join(specification) +
        f"\nQuestion context: {scene_context}\nView id: {view_id}"
    )


def _strict_grounding_json(
    text: str,
    label_to_class: Mapping[str, str],
    *,
    maximum: int,
) -> list[dict[str, object]]:
    # Qwen occasionally emits a stray quote immediately after a numeric box
    # coordinate (for example ``"bottom":0.51"``).  Normalize that lexical
    # artifact without altering any coordinate value or semantic field.
    text = re.sub(r'(?<=\d)"(?=\s*[,}])', "", str(text))
    decoder = json.JSONDecoder()
    last_error: Exception | None = None
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
            if isinstance(payload, list):
                raw_proposals = payload
            elif isinstance(payload, Mapping) and set(payload) == {"proposals"}:
                raw_proposals = payload["proposals"]
            else:
                raise ValueError("grounding_output_schema_invalid")
            if not isinstance(raw_proposals, list):
                raise ValueError("grounding_proposals_not_list")
            proposals = []
            for raw in raw_proposals[:maximum]:
                if not isinstance(raw, Mapping) or set(raw) not in ({
                    "label", "bbox_2d",
                }, {
                    "class_name", "bbox", "semantic_probability", "rationale_tags",
                }, {
                    "class_name", "box", "semantic_probability", "rationale_tags",
                }):
                    raise ValueError("grounding_proposal_schema_invalid")
                emitted_label = str(raw.get("label", raw.get("class_name", ""))).strip().lower()
                class_name = label_to_class.get(emitted_label)
                if class_name is None:
                    raise ValueError("grounding_class_outside_task")
                if "bbox_2d" in raw:
                    bbox_1000 = [float(value) for value in raw["bbox_2d"]]
                    bbox = list(_normalized_bbox([value / 1000.0 for value in bbox_1000]))
                elif "bbox" in raw:
                    bbox = list(_normalized_bbox(raw["bbox"]))
                else:
                    box = raw["box"]
                    if not isinstance(box, Mapping) or set(box) != {
                        "left", "top", "right", "bottom"
                    }:
                        raise ValueError("grounding_box_schema_invalid")
                    bbox = list(_normalized_bbox([
                        box["left"], box["top"], box["right"], box["bottom"]
                    ]))
                if "semantic_probability" in raw:
                    probability = float(raw["semantic_probability"])
                    if not 0.0 <= probability <= 1.0:
                        raise ValueError("grounding_probability_invalid")
                    if probability == 0.0:
                        continue
                    tags = raw["rationale_tags"]
                    if not isinstance(tags, list):
                        raise ValueError("grounding_rationale_tags_invalid")
                else:
                    probability = 1.0
                    tags = ["qwen3vl_native_2d_grounding"]
                proposals.append({
                    "class_name": class_name,
                    "bbox_xyxy_normalized": bbox,
                    "semantic_probability": probability,
                    "rationale_tags": [str(value) for value in tags],
                })
            return proposals
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = exc
    raise ValueError("qwen3vl_grounding_json_invalid") from last_error


def _relation_images(parameters):
    values = parameters.get("shared_memories")
    if not isinstance(values, list) or not 1 <= len(values) <= 3:
        raise ValueError("qwen3vl_relation_image_contract_invalid")
    from PIL import Image
    from .shared_memory_io import SharedArrayStore

    output = []
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {
            "role", "name", "shape", "dtype", "preprocessed_once"
        }:
            raise ValueError("qwen3vl_relation_shared_memory_schema_invalid")
        if value["preprocessed_once"] is not True:
            raise ValueError("qwen3vl_relation_image_not_preprocessed")
        image = SharedArrayStore().get(
            str(value["name"]),
            tuple(int(item) for item in value["shape"]),
            str(value["dtype"]),
            unlink=False,
        )
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or str(image.dtype) != "uint8"
            or image.shape[0] % 32
            or image.shape[1] % 32
        ):
            raise ValueError("qwen3vl_relation_image_profile_invalid")
        # No crop or resize occurs in the worker.  The processor receives the
        # one client-prepared, patch-aligned image.
        output.append((
            str(value["role"]),
            Image.fromarray(image, mode="RGB"),
        ))
    return tuple(output)


def _relation_prompt(request: Mapping[str, object]) -> str:
    subject_id = int(request["subject_id"])
    object_ids = [int(value) for value in request["object_ids"]]
    predicate = str(request["predicate"]).upper()
    output_example = json.dumps(
        {
            "subject_id": subject_id,
            "predicate": predicate,
            "object_ids": object_ids,
            "state": "uncertain",
            "confidence": 0.0,
            "reason_code": "insufficient_visible_evidence",
            "visible_subject": False,
            "visible_objects": [False for _ in object_ids],
            "jointly_observable": False,
            "occlusion": "severe",
        },
        separators=(",", ":"),
    )
    return (
        "You are an object-ID-grounded visual relation verifier. "
        "Use only the supplied crops and joint context. Never invent, merge, "
        "or substitute IDs. Treat missing pixels, occlusion, or absent joint "
        "visibility as uncertain. Verify this exact tuple: "
        f"subject_id={subject_id}, predicate={predicate}, "
        f"object_ids={object_ids}. Choose state as exactly supported, "
        "refuted, or uncertain; choose occlusion as exactly none, partial, "
        "or severe. Return exactly one JSON object, no prose and no markdown, "
        "with exactly these keys and the same IDs. This valid uncertain "
        f"default illustrates the schema: {output_example}. "
        "confidence is confidence in the chosen state, not a scene answer."
    )


def _strict_relation_json(
    text: str,
    expected: Mapping[str, object],
) -> ObjectGroundedRelationResult:
    decoder = json.JSONDecoder()
    last_error = None
    for index, character in enumerate(str(text)):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(str(text)[index:])
            result = ObjectGroundedRelationResult.from_json(payload)
            if (
                result.subject_id != int(expected["subject_id"])
                or result.predicate != str(expected["predicate"]).upper()
                or result.object_ids != tuple(
                    int(value) for value in expected["object_ids"]
                )
            ):
                raise ValueError("qwen_relation_object_id_mismatch")
            return result
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = exc
    raise ValueError("qwen3vl_relation_json_invalid") from last_error


class LocalQwen3VLImplementation:
    """Lazy, memory-bounded Qwen3-VL callable for ``Qwen3VLBackend``.

    The detector/SAM process occupies most of a 32 GB competition GPU.  A
    second full-BF16 copy of Qwen3-VL can therefore be killed by the driver
    after the worker socket has already passed a superficial liveness check.
    INT8 keeps the verifier resident alongside the official vision process
    without changing its candidate-only contract.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cuda",
        max_new_tokens: int = 128,
        max_pixels: int = 512 * 512,
        quantization: str = "int8",
    ):
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_dir():
            raise ValueError("qwen3vl_checkpoint_directory_missing")
        self.checkpoint_path = str(checkpoint)
        self.device = str(device)
        self.max_new_tokens = max(32, int(max_new_tokens))
        self.max_pixels = max(224 * 224, int(max_pixels))
        quantization = str(quantization).strip().lower()
        if quantization not in {"int8", "int4", "bf16"}:
            raise ValueError("qwen3vl_quantization_must_be_int8_int4_or_bf16")
        self.quantization = quantization
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3VLForConditionalGeneration,
        )

        self._processor = AutoProcessor.from_pretrained(
            self.checkpoint_path,
            local_files_only=True,
            max_pixels=self.max_pixels,
        )
        compute_dtype = (
            torch.float16 if self.quantization in {"int8", "int4"}
            else torch.bfloat16
        )
        model_kwargs = {
            "local_files_only": True,
            "dtype": compute_dtype,
            "attn_implementation": "sdpa",
        }
        if self.quantization == "int8":
            model_kwargs.update({
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                "device_map": {"": self.device},
            })
        elif self.quantization == "int4":
            model_kwargs.update({
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                ),
                "device_map": {"": self.device},
            })
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.checkpoint_path,
            **model_kwargs,
        )
        if self.quantization == "bf16":
            self._model = self._model.to(self.device)
        self._model.eval()

    def healthcheck(self) -> Mapping[str, object]:
        """Load all weights so launch-time health proves runtime residency."""
        self._ensure_loaded()
        return {
            "ready": True,
            "device": str(self._model.device),
            "quantization": self.quantization,
            "max_pixels": self.max_pixels,
            "checkpoint_path": self.checkpoint_path,
        }

    def __call__(self, request) -> Mapping[str, object]:
        parameters = dict(getattr(request, "parameters", {}) or {})
        operation = str(getattr(request, "operation", ""))
        if operation == "verify_relation":
            return self._verify_relation(parameters)
        if operation == "ground_objects":
            return self._ground_objects(parameters)
        if operation == "ground_objects_batch":
            return self._ground_objects_batch(parameters)
        if operation == "rank_candidates":
            return self._rank_candidates(parameters)
        bbox = _normalized_bbox(parameters.get("candidate_bbox", ()))
        query_concept = str(parameters.get("query_concept", "")).strip()
        if not query_concept:
            raise ValueError("qwen3vl_query_concept_missing")
        anchor_concept = str(parameters.get("anchor_concept", "")).strip()
        hard_negatives = tuple(parameters.get("hard_negatives", ()) or ())

        self._ensure_loaded()
        source_image = _request_source_image(parameters)
        object_crop, context_crop = _candidate_images_from_pil(
            source_image, bbox
        )
        prompt = _verification_prompt(
            operation=operation,
            query_concept=query_concept,
            anchor_concept=anchor_concept,
            hard_negatives=hard_negatives,
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": object_crop},
                {"type": "image", "image": context_crop},
                {"type": "text", "text": prompt},
            ],
        }]
        model_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        import torch
        with torch.inference_mode():
            generated = self._model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prompt_length = int(model_inputs["input_ids"].shape[1])
        decoded = self._processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        try:
            verification = _strict_semantic_json(decoded)
        except ValueError as exc:
            # The worker log is local and contains model output only (never
            # credentials or source image bytes).  Keep a bounded diagnostic
            # so a failed semantic verdict is distinguishable from transport
            # failure instead of silently appearing as generic "unknown".
            print(json.dumps({
                "event": "qwen3vl_invalid_output",
                "error": str(exc),
                "decoded": decoded[:500],
            }, ensure_ascii=False), file=sys.stderr, flush=True)
            raise
        return {
            "target_probability": verification.target_probability,
            "anchor_probability": verification.anchor_probability,
            "confuser_probabilities": dict(
                verification.confuser_probabilities
            ),
            "rationale_tags": list(verification.rationale_tags),
            "contained_instance_count": (
                verification.contained_instance_count
            ),
            "count_confidence": verification.count_confidence,
            "backend": "qwen3vl_local_candidate_verifier",
            "checkpoint_path": self.checkpoint_path,
            "quantization": self.quantization,
        }

    def _ground_objects(self, parameters: Mapping[str, object]) -> Mapping[str, object]:
        view_id = str(parameters.get("view_id", "")).strip()
        concepts = parameters.get("concepts")
        if not view_id:
            raise ValueError("qwen3vl_grounding_view_id_missing")
        if not isinstance(concepts, list) or not concepts:
            raise ValueError("qwen3vl_grounding_concepts_missing")
        normalized_concepts = []
        label_to_class = {}
        for raw in concepts:
            if not isinstance(raw, Mapping):
                raise ValueError("qwen3vl_grounding_concept_invalid")
            class_name = str(raw.get("class_name", "")).strip().lower()
            if not class_name:
                raise ValueError("qwen3vl_grounding_class_missing")
            label_to_class[class_name] = class_name
            for alias in raw.get("aliases", ()):
                label_to_class[str(alias).strip().lower()] = class_name
            normalized_concepts.append({**dict(raw), "class_name": class_name})
        maximum = max(1, min(100, int(parameters.get("max_proposals", 40))))

        self._ensure_loaded()
        source_image = _request_source_image(parameters)
        prompt = _grounding_prompt(
            normalized_concepts,
            view_id=view_id,
            scene_context=str(parameters.get("scene_context", "")),
            maximum=maximum,
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": source_image},
                {"type": "text", "text": prompt},
            ],
        }]
        model_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        import torch
        with torch.inference_mode():
            generated = self._model.generate(
                **model_inputs,
                max_new_tokens=max(self.max_new_tokens, 2048),
                do_sample=False,
            )
        prompt_length = int(model_inputs["input_ids"].shape[1])
        decoded = self._processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        # The model occasionally emits a truncated JSON array when it runs
        # out of tokens.  Repair the common pattern of a missing closing "]".
        decoded_stripped = str(decoded).rstrip()
        if decoded_stripped.endswith(",") or (
            decoded_stripped.count("[") > decoded_stripped.count("]")
        ):
            decoded_stripped = re.sub(r',\s*$', '', decoded_stripped)
            if decoded_stripped.count("[") > decoded_stripped.count("]"):
                decoded_stripped += "]"
        try:
            proposals = _strict_grounding_json(
                decoded,
                label_to_class,
                maximum=maximum,
            )
        except ValueError as exc:
            print(json.dumps({
                "event": "qwen3vl_invalid_grounding_output",
                "error": str(exc),
                "decoded": decoded[:1000],
            }, ensure_ascii=False), file=sys.stderr, flush=True)
            raise
        return {
            "view_id": view_id,
            "proposals": proposals,
            "proposal_count": len(proposals),
            "backend": "qwen3vl_query_conditioned_grounder",
            "checkpoint_path": self.checkpoint_path,
            "quantization": self.quantization,
        }

    def _ground_objects_batch(
        self, parameters: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Ground objects across multiple views in a single GPU forward pass.

        Accepts ``views``: a list of {view_id, image_path, concepts,
        scene_context, max_proposals}.  All images are encoded together so the
        vision backbone runs once per batch instead of once per view.
        """
        views = parameters.get("views")
        if not isinstance(views, list) or not views:
            raise ValueError("ground_objects_batch_views_missing")
        if len(views) > 4:
            raise ValueError("ground_objects_batch_max_four_views")

        self._ensure_loaded()
        from PIL import Image

        per_view: list[dict] = []
        for v in views:
            view_id = str(v.get("view_id", "")).strip()
            concepts = v.get("concepts")
            if not view_id or not isinstance(concepts, list) or not concepts:
                raise ValueError("ground_objects_batch_view_incomplete")
            label_to_class: dict[str, str] = {}
            normalized = []
            for raw in concepts:
                if not isinstance(raw, dict):
                    raise ValueError("ground_objects_batch_concept_invalid")
                cn = str(raw.get("class_name", "")).strip().lower()
                if not cn:
                    raise ValueError("ground_objects_batch_class_missing")
                label_to_class[cn] = cn
                for alias in raw.get("aliases", ()):
                    label_to_class[str(alias).strip().lower()] = cn
                normalized.append({**dict(raw), "class_name": cn})
            maximum = max(1, min(100, int(v.get("max_proposals", 12))))
            prompt = _grounding_prompt(
                normalized,
                view_id=view_id,
                scene_context=str(v.get("scene_context", "")),
                maximum=maximum,
            )
            image = Image.open(str(v["image_path"])).convert("RGB")
            per_view.append({
                "view_id": view_id,
                "label_to_class": label_to_class,
                "maximum": maximum,
                "image": image,
                "prompt": prompt,
            })

        # Build a multi-image message with per-view markers so the model
        # returns a JSON object keyed by view_id.
        view_ids = [pv["view_id"] for pv in per_view]
        content_parts: list[dict] = []
        for pv in per_view:
            content_parts.append({"type": "image", "image": pv["image"]})
            content_parts.append({"type": "text", "text": pv["prompt"]})
        # Append a formatting instruction so responses are grouped by view.
        content_parts.append({"type": "text", "text": (
            "Return exactly one JSON object whose keys are the view ids "
            + json.dumps(view_ids, separators=(",", ":"))
            + " and whose values are the proposal lists for each view. "
            "No prose, no markdown."
        )})
        messages = [{"role": "user", "content": content_parts}]

        model_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        import torch

        with torch.inference_mode():
            generated = self._model.generate(
                **model_inputs,
                max_new_tokens=max(self.max_new_tokens * len(views), 2048),
                do_sample=False,
            )
        prompt_length = int(model_inputs["input_ids"].shape[1])
        decoded = self._processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        results: list[dict] = []
        # Parse the response: prefer a view_id-keyed object, fall back to a
        # flat list assigned by label match.
        parsed_by_view: dict[str, list] = {}
        decoder = json.JSONDecoder()
        raw_text = str(decoded)
        for idx, ch in enumerate(raw_text):
            if ch != "{":
                continue
            try:
                payload, _end = decoder.raw_decode(raw_text[idx:])
                if isinstance(payload, dict):
                    # Check if this is a view_id-keyed object.
                    view_keys = [k for k in payload if k in {pv["view_id"] for pv in per_view}]
                    if len(view_keys) >= len(per_view) * 0.5:
                        parsed_by_view = {
                            k: (
                                _strict_grounding_json(
                                    json.dumps(v, separators=(",", ":")),
                                    per_view[i]["label_to_class"]
                                    if i < len(per_view) else {},
                                    maximum=per_view[i]["maximum"] if i < len(per_view) else 12,
                                )
                                if isinstance(v, list) else []
                            )
                            for i, (k, v) in enumerate(payload.items())
                            if k in {pv["view_id"] for pv in per_view}
                        }
                        break
                    # Single flat object — try fallback parsing below.
                elif isinstance(payload, list) and not parsed_by_view:
                    # Flat list of proposals; assign by class-name match below.
                    all_flat = payload
            except (json.JSONDecodeError, ValueError):
                continue

        for i, pv in enumerate(per_view):
            view_proposals: list[dict] = []
            if pv["view_id"] in parsed_by_view:
                view_proposals = parsed_by_view[pv["view_id"]]
            elif isinstance(parsed_by_view, dict) and not parsed_by_view:
                # Fallback: parse the full text as one flat list and filter.
                try:
                    all_proposals = _strict_grounding_json(
                        raw_text,
                        pv["label_to_class"],
                        maximum=pv["maximum"],
                    )
                    view_proposals = all_proposals
                except ValueError:
                    pass
            results.append({
                "view_id": pv["view_id"],
                "proposals": view_proposals,
                "proposal_count": len(view_proposals),
                "backend": "qwen3vl_batch_multiview_grounder",
                "checkpoint_path": self.checkpoint_path,
                "quantization": self.quantization,
            })

        return {
            "views": results,
            "backend": "qwen3vl_batch_multiview_grounder",
            "checkpoint_path": self.checkpoint_path,
            "quantization": self.quantization,
        }

    def _rank_candidates(
        self, parameters: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Given a scene image and labelled candidates, return the index
        of the candidate that best matches the question description."""
        question = str(parameters.get("question", "")).strip()
        candidates = parameters.get("candidates")
        if not question or not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError("rank_candidates_needs_question_and_2plus_candidates")

        self._ensure_loaded()
        from PIL import Image, ImageDraw

        source = Image.open(str(parameters["image_path"])).convert("RGB")
        width, height = source.size
        draw = ImageDraw.Draw(source)

        candidate_descriptions = []
        for i, cand in enumerate(candidates):
            label = chr(ord("A") + i)
            bbox = cand.get("bbox_xyxy")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50), width=3)
                draw.text((x1 + 4, y1 + 4), label, fill=(255, 50, 50))
            cls = str(cand.get("class_name", ""))
            ctx = str(cand.get("context", ""))
            candidate_descriptions.append(
                f"Candidate {label}: {cls}"
                + (f" ({ctx})" if ctx else "")
            )

        prompt = (
            f"Question: {question}\n\n"
            + "\n".join(candidate_descriptions)
            + "\n\n"
            "The image shows the scene with candidates labelled A,B,C... "
            "in red boxes.  Which candidate (exactly one letter) best "
            "matches what the question asks for?  "
            "Consider the full spatial context described in the question. "
            "Return exactly one JSON object with key \"best_candidate\" "
            "(the letter) and \"confidence\" (0-1).  No prose, no markdown."
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": source},
                {"type": "text", "text": prompt},
            ],
        }]
        model_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        import torch
        with torch.inference_mode():
            generated = self._model.generate(
                **model_inputs,
                max_new_tokens=64,
                do_sample=False,
            )
        prompt_length = int(model_inputs["input_ids"].shape[1])
        decoded = self._processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse the response
        import json as _json, re as _re
        decoder = _json.JSONDecoder()
        best_idx = 0
        confidence = 0.5
        for idx, ch in enumerate(str(decoded)):
            if ch != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(str(decoded)[idx:])
                letter = str(payload.get("best_candidate", "A")).strip().upper()
                if len(letter) == 1 and "A" <= letter <= "Z":
                    best_idx = ord(letter) - ord("A")
                confidence = float(payload.get("confidence", 0.5))
                break
            except (_json.JSONDecodeError, ValueError, KeyError):
                continue

        best_idx = max(0, min(best_idx, len(candidates) - 1))
        return {
            "best_candidate_index": int(best_idx),
            "confidence": float(max(0.0, min(1.0, confidence))),
            "backend": "qwen3vl_candidate_ranker",
            "checkpoint_path": self.checkpoint_path,
            "quantization": self.quantization,
        }

    def _verify_relation(
        self, parameters: Mapping[str, object]
    ) -> Mapping[str, object]:
        grounded = parameters.get("grounded_relation_request")
        if not isinstance(grounded, Mapping):
            raise ValueError("qwen3vl_grounded_relation_request_missing")
        required = {
            "request_id",
            "episode_id",
            "query_node_id",
            "world_snapshot_version",
            "acquisition_id",
            "subject_id",
            "subject_instance_version",
            "object_ids",
            "object_instance_versions",
            "predicate",
            "subject_bbox_or_mask",
            "object_bboxes_or_masks",
            "camera_pose",
            "evidence_provenance",
        }
        if set(grounded) != required:
            raise ValueError("qwen3vl_grounded_relation_schema_invalid")
        images = _relation_images(parameters)
        self._ensure_loaded()
        messages = [{
            "role": "user",
            "content": [
                *(
                    {"type": "image", "image": image}
                    for _, image in images
                ),
                {"type": "text", "text": _relation_prompt(grounded)},
            ],
        }]
        model_inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        import torch
        with torch.inference_mode():
            generated = self._model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prompt_length = int(model_inputs["input_ids"].shape[1])
        decoded = self._processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        try:
            result = _strict_relation_json(decoded, grounded)
        except ValueError as exc:
            print(json.dumps({
                "event": "qwen3vl_relation_invalid_output",
                "error": str(exc),
                "decoded": decoded[:500],
                "subject_id": grounded["subject_id"],
                "predicate": grounded["predicate"],
                "object_ids": grounded["object_ids"],
            }, ensure_ascii=False), file=sys.stderr, flush=True)
            raise
        return {
            "subject_id": result.subject_id,
            "predicate": result.predicate,
            "object_ids": list(result.object_ids),
            "state": result.state,
            "confidence": result.confidence,
            "visible_subject": result.visible_subject,
            "visible_objects": list(result.visible_objects),
            "jointly_observable": result.jointly_observable,
            "occlusion": result.occlusion,
            "reason_code": result.reason_code,
            "backend": "qwen3vl_local_object_grounded_relation_verifier",
            "checkpoint_path": self.checkpoint_path,
            "quantization": self.quantization,
        }
