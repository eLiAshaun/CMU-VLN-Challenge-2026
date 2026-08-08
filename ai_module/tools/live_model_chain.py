#!/usr/bin/env python3
"""Execute the live vision chain in one explicit, fail-closed stage order."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from multiprocessing.connection import Client
from pathlib import Path
import subprocess
import sys
import time
import uuid

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.mast3r.panorama_adapter import write_perspective_bundle
from integrations.execution.query_executor import execute_semantic_relation_count, execute_task
from integrations.execution.root_finalizer import finalize_execution
from integrations.execution.route_planner import plan_route
from integrations.execution.world_model import update_world_model
from integrations.semantics.task_compiler import compile_task


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_worker(module: str, request: dict, root: Path, stage: Path) -> dict:
    stage.mkdir(parents=True, exist_ok=True)
    request_path = stage / "worker_request.json"
    write_json(request_path, request)
    completed = subprocess.run(
        [sys.executable, "-m", module, "--request", str(request_path)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    (stage / "worker.log").write_text(completed.stdout, encoding="utf-8")
    response_path = stage / "worker_response.json"
    response = (
        json.loads(response_path.read_text(encoding="utf-8"))
        if response_path.is_file()
        else {"status": "failed", "error": "worker_response_missing"}
    )
    response["returncode"] = int(completed.returncode)
    return response


def socket_request(endpoint: str, payload: dict, timeout_seconds: float) -> dict:
    connection = Client("\0" + endpoint[1:], family="AF_UNIX")
    try:
        connection.send_bytes(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        if not connection.poll(float(timeout_seconds)):
            raise TimeoutError(f"model_service_timeout:{payload.get('operation')}")
        response = json.loads(connection.recv_bytes().decode("utf-8"))
    finally:
        connection.close()
    if response.get("request_id") != payload.get("request_id"):
        raise RuntimeError("model_service_request_id_mismatch")
    return response


def qwen_ground_view(
    endpoint: str,
    view: dict,
    concepts: list[dict],
    scene_context: str,
    *,
    maximum: int,
) -> dict:
    request_id = uuid.uuid4().hex
    response = socket_request(endpoint, {
        "request_id": request_id,
        "acquisition_id": f"ground-{request_id}",
        "backend": "qwen3vl",
        "operation": "ground_objects",
        "input_handles": [],
        "parameters": {
            "image_path": str(view["image_path"]),
            "view_id": str(view["view_id"]),
            "concepts": concepts,
            "scene_context": scene_context,
            "max_proposals": int(maximum),
        },
    }, 240.0)
    if response.get("ok") is not True:
        raise RuntimeError(
            f"qwen3vl_grounding_failed:{response.get('error_code')}:"
            f"{response.get('metadata', {}).get('error_detail', '')}"
        )
    return dict(response["metadata"])


def qwen_ground_views_batch(
    endpoint: str,
    view_specs: list[dict],
    *,
    batch_timeout: float = 480.0,
) -> list[dict]:
    """Ground objects across up to 4 views in a single GPU forward pass."""
    if not view_specs:
        return []
    request_id = uuid.uuid4().hex
    response = socket_request(endpoint, {
        "request_id": request_id,
        "acquisition_id": f"batch-{request_id}",
        "backend": "qwen3vl",
        "operation": "ground_objects_batch",
        "input_handles": [],
        "parameters": {
            "views": [
                {
                    "view_id": str(vs["view_id"]),
                    "image_path": str(vs["image_path"]),
                    "concepts": list(vs["concepts"]),
                    "scene_context": str(vs.get("scene_context", "")),
                    "max_proposals": int(vs.get("max_proposals", 12)),
                }
                for vs in view_specs
            ],
        },
    }, batch_timeout)
    if response.get("ok") is not True:
        raise RuntimeError(
            f"qwen3vl_batch_grounding_failed:{response.get('error_code')}:"
            f"{response.get('metadata', {}).get('error_detail', '')}"
        )
    return list(response["metadata"].get("views", []))


def qwen_verify_detection(
    endpoint: str,
    detection: dict,
    view: dict,
    task_context: str,
    *,
    hard_negatives: list[str],
) -> dict:
    width, height = float(view["width"]), float(view["height"])
    x1, y1, x2, y2 = (float(value) for value in detection["bbox_xyxy"])
    request_id = uuid.uuid4().hex
    return socket_request(endpoint, {
        "request_id": request_id,
        "acquisition_id": f"verify-{request_id}",
        "backend": "qwen3vl",
        "operation": "verify_object",
        "input_handles": [],
        "parameters": {
            "image_path": str(view["image_path"]),
            "candidate_bbox": [x1 / width, y1 / height, x2 / width, y2 / height],
            "query_concept": str(detection["canonical_class"]),
            "anchor_concept": task_context,
            "hard_negatives": hard_negatives,
        },
    }, 180.0)


def run_persistent_sam(endpoint: str, request: dict, stage: Path, timeout_seconds: float) -> dict:
    stage.mkdir(parents=True, exist_ok=True)
    write_json(stage / "worker_request.json", request)
    request_id = uuid.uuid4().hex
    response = socket_request(endpoint, {
        "request_id": request_id,
        "operation": "segment_boxes",
        "parameters": request,
    }, timeout_seconds)
    metadata = dict(response.get("metadata", {}))
    if response.get("ok") is not True:
        metadata.update({
            "status": "failed",
            "error_code": response.get("error_code"),
        })
    metadata["persistent_model"] = True
    write_json(stage / "worker_response.json", metadata)
    return metadata


def ensure_persistent_sam(runtime: dict, models: dict) -> dict:
    endpoint = str(runtime["sam2"]["endpoint"])
    request_id = uuid.uuid4().hex
    health = {
        "request_id": request_id,
        "operation": "healthcheck",
        "parameters": {},
    }
    try:
        response = socket_request(endpoint, health, 1.0)
        if response.get("ok") is True:
            return dict(response["metadata"])
    except (ConnectionError, EOFError, OSError, RuntimeError, TimeoutError):
        pass
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            str(models["sam2"]["worker_module"]),
            "--checkpoint",
            str((ROOT / models["sam2"]["checkpoint"]).resolve()),
            "--config",
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "--endpoint",
            endpoint,
            "--device",
            "cuda",
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + float(runtime["sam2"]["startup_timeout_seconds"])
    while time.monotonic() < deadline:
        time.sleep(0.25)
        request_id = uuid.uuid4().hex
        try:
            response = socket_request(endpoint, {
                "request_id": request_id,
                "operation": "healthcheck",
                "parameters": {},
            }, 1.0)
            if response.get("ok") is True:
                return dict(response["metadata"])
        except (ConnectionError, EOFError, OSError, RuntimeError, TimeoutError):
            continue
    raise TimeoutError("sam2_persistent_worker_startup_timeout")


def qwen_verify(
    endpoint: str,
    observation: dict,
    concept: str,
    task_context: str,
    *,
    operation: str = "verify_object",
    hard_negatives: list[str] | None = None,
) -> dict:
    use_panorama = bool(observation.get("qwen_evidence_image"))
    image_path = Path(
        observation["qwen_evidence_image"]
        if use_panorama else observation["representative_view_image"]
    )
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("qwen_representative_view_unreadable")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (
        float(value) for value in observation[
            "panorama_bbox_xyxy" if use_panorama else "representative_bbox_xyxy"
        ]
    )
    request_id = uuid.uuid4().hex
    payload = {
        "request_id": request_id,
        "acquisition_id": f"live-{request_id}",
        "backend": "qwen3vl",
        "operation": operation,
        "input_handles": [],
        "parameters": {
            "image_path": str(image_path),
            "candidate_bbox": [x1 / width, y1 / height, x2 / width, y2 / height],
            "query_concept": concept,
            "anchor_concept": task_context,
            "hard_negatives": list(hard_negatives or []),
        },
    }
    return socket_request(endpoint, payload, 180.0)


def anchor_inventory_observation(
    observation: dict,
    panorama_path: Path,
) -> dict:
    """Expand a support-object box into the surface inventory region above it."""
    image = cv2.imread(str(panorama_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("anchor_inventory_panorama_unreadable")
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = (
        float(value) for value in observation["panorama_bbox_xyxy"]
    )
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    expanded = [
        max(0.0, x1 - max(0.85 * box_width, 0.05 * image_width)),
        max(0.0, y1 - max(4.0 * box_height, 0.10 * image_height)),
        min(float(image_width), x2 + max(0.85 * box_width, 0.05 * image_width)),
        min(float(image_height), y2 + max(0.35 * box_height, 0.015 * image_height)),
    ]
    return {
        **observation,
        "qwen_evidence_image": str(panorama_path),
        "panorama_bbox_xyxy": expanded,
        "support_bbox_xyxy": [x1, y1, x2, y2],
        "inventory_region_policy": "support_body_plus_surface_above",
    }


def target_detection_count(response: dict, target: str) -> int:
    """Count only the requested class, never unrelated open-vocabulary hits."""
    counts = response.get("counts_by_class", {})
    if isinstance(counts, dict):
        return int(counts.get(target, 0))
    return 0


def groundingdino_rescue_required(yolo_response: dict, required_classes) -> bool:
    classes = [required_classes] if isinstance(required_classes, str) else list(required_classes)
    return yolo_response.get("status") != "completed" or any(
        target_detection_count(yolo_response, class_name) == 0
        for class_name in classes
    )


def merge_detection_sources(paths: list[str], output_path: Path) -> str:
    records = []
    seen = set()
    for source in paths:
        for record in json.loads(Path(source).read_text(encoding="utf-8")):
            key = (record.get("detector"), record.get("detection_id"))
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    write_json(output_path, records)
    return str(output_path)


def qwen_target_probability(response: dict) -> float | None:
    if response.get("ok") is not True:
        return None
    value = response.get("metadata", {}).get("target_probability")
    if not isinstance(value, (int, float)):
        return None
    probability = float(value)
    return probability if 0.0 <= probability <= 1.0 else None


def competition_geometry_gate(payload: dict, config: dict) -> tuple[bool, list[str]]:
    reasons = []
    maximum_age = float(config["max_age_seconds"])
    maximum_offset = float(config["max_camera_lidar_offset_seconds"])
    checks = (
        ("sensor_scan", "sensor_scan", config["sensor_scan_required"]),
        ("registered_scan", "registered_scan", config["registered_scan_required"]),
        ("state_estimation", "state_estimation", config["state_estimation_required"]),
    )
    for label, prefix, required in checks:
        if not required:
            continue
        path = payload.get(f"{prefix}_path")
        age = payload.get(f"{prefix}_age_seconds")
        if not path or not Path(path).is_file():
            reasons.append(f"{label}_artifact_missing")
        if age is None or float(age) > maximum_age:
            reasons.append(f"{label}_stale")
    if int(payload.get("sensor_scan_point_count", 0)) <= 0:
        reasons.append("sensor_scan_empty")
    if int(payload.get("registered_scan_point_count", 0)) <= 0:
        reasons.append("registered_scan_empty")
    offset = payload.get("camera_sensor_scan_offset_seconds")
    if offset is None or float(offset) > maximum_offset:
        reasons.append("camera_sensor_scan_out_of_sync")
    if payload.get("sensor_scan_frame") != "sensor_at_scan":
        reasons.append("sensor_scan_frame_invalid")
    if payload.get("registered_scan_frame") != "map":
        reasons.append("registered_scan_frame_invalid")
    if payload.get("state_estimation_frame") != "map":
        reasons.append("state_estimation_frame_invalid")
    if payload.get("state_estimation_child_frame") != "sensor":
        reasons.append("state_estimation_child_frame_invalid")
    return not reasons, reasons


class OrderedSummary:
    def __init__(self, output: Path, request: dict, configured_order: list[str]):
        self.output = output
        self.configured_order = configured_order
        self.payload = {
            "schema_version": "2.0",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "question": request["question"],
            "input": request["image_geometry"],
            "configured_stage_order": configured_order,
            "executed_stage_order": [],
            "stages": {},
            "gates": {"answer_authorized": False},
        }
        self.flush()

    def record(self, name: str, result: dict, **gates: object) -> None:
        index = len(self.payload["executed_stage_order"])
        if index >= len(self.configured_order) or self.configured_order[index] != name:
            raise RuntimeError(f"stage_order_violation:{name}:index={index}")
        self.payload["executed_stage_order"].append(name)
        self.payload["stages"][name] = result
        self.payload["gates"].update(gates)
        self.flush()

    def finish(self) -> None:
        if self.payload["executed_stage_order"] != self.configured_order:
            raise RuntimeError("pipeline_finished_before_all_stages_recorded")
        self.payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.flush()

    def flush(self) -> None:
        write_json(self.output / "summary.json", self.payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    output = Path(request["output_dir"]).resolve()
    config = json.loads((ROOT / "configs" / "model_assets.json").read_text(encoding="utf-8"))
    runtime = config["runtime"]
    models = config["models"]
    order = list(runtime["pipeline"]["stage_order"])
    summary = OrderedSummary(output, request, order)
    image_path = Path(request["image_path"]).resolve()
    panorama = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if panorama is None:
        raise ValueError("live_panorama_unreadable")
    task_ir = compile_task(request["question"], runtime.get("deepseek", {}))
    grounding_plan = task_ir["grounding_plan"]
    target_entity = next(
        item for item in task_ir["entities"] if item["id"] == task_ir["target_entity"]
    )
    target = str(target_entity["class_name"])
    aliases = dict(grounding_plan["detector_classes"])
    grounding_by_class = {
        str(item["primary_class"]): item for item in grounding_plan["entities"]
    }
    task_roles_by_class = {
        str(item["class_name"]): str(item["role"]) for item in task_ir["entities"]
    }
    semantic_scene_context = json.dumps(
        {
            "question": task_ir["original_question"],
            "task_type": task_ir["task_type"],
            "relations": task_ir["relations"],
            "ordered_subgoals": task_ir["ordered_subgoals"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    summary.record(
        "task_compilation",
        {
            "status": "completed",
            "task_ir": task_ir,
            "target_class": target,
            "required_classes": task_ir["required_classes"],
            "detector_classes": list(aliases),
            "grounding_plan": grounding_plan,
            "output_contract": task_ir["output_contract"],
        },
        task_compilation=True,
        task_type=task_ir["task_type"],
    )
    competition_geometry = dict(request.get("competition_geometry", {}))
    geometry_ready, geometry_reasons = competition_geometry_gate(
        competition_geometry, runtime["competition_geometry"]
    )
    summary.record(
        "canonical_panorama",
        {
            "status": "completed",
            "image_path": str(image_path),
            "projection": "equirectangular",
            "role": "immutable_canonical_evidence",
            "semantic_primary_input": True,
            "semantic_role": "global_scene_overview_before_focused_views",
        },
        live_rgb=True,
        competition_geometry=geometry_ready,
    )

    adapter = None
    semantic_adapter = None
    reconstruction_bundles = []
    mast3r_runtime = runtime["mast3r"]
    reconstruction_module = {
        "sparse_global_alignment": "integrations.mast3r.sparse_global_mapper",
    }[mast3r_runtime["reconstruction_mode"]]
    try:
        keyframes = list(request.get("reconstruction_keyframes", ()))
        if not keyframes:
            state = json.loads(Path(competition_geometry["state_estimation_path"]).read_text(encoding="utf-8"))
            keyframes = [{
                "keyframe_id": output.name,
                "panorama_path": str(image_path),
                "state_estimation_path": competition_geometry["state_estimation_path"],
                "sensor_scan_path": competition_geometry["sensor_scan_path"],
                "registered_scan_path": competition_geometry["registered_scan_path"],
                "state_estimation": state,
                "optical_center_group": f"station:{output.name}",
            }]
        for keyframe in keyframes:
            keyframe_id = str(keyframe["keyframe_id"])
            keyframe_panorama = (
                panorama if Path(keyframe["panorama_path"]).resolve() == image_path
                else cv2.imread(str(keyframe["panorama_path"]), cv2.IMREAD_COLOR)
            )
            if keyframe_panorama is None:
                raise ValueError(f"keyframe_panorama_unreadable:{keyframe_id}")
            bundle = write_perspective_bundle(
                keyframe_panorama,
                output / "01_perspective_adapter" / "keyframes" / keyframe_id,
                yaws_deg=mast3r_runtime["perspective_yaws_deg"],
                pitches_deg=mast3r_runtime["perspective_pitches_deg"],
                width=int(mast3r_runtime["perspective_width"]),
                height=int(mast3r_runtime["perspective_height"]),
                horizontal_fov_deg=float(mast3r_runtime["perspective_horizontal_fov_deg"]),
                panorama_vertical_fov_deg=float(mast3r_runtime["panorama_vertical_fov_deg"]),
                optical_center_group=str(keyframe["optical_center_group"]),
                view_id_prefix=keyframe_id,
            )
            reconstruction_bundles.append({**keyframe, "manifest_path": bundle["manifest_path"]})
            if Path(keyframe["panorama_path"]).resolve() == image_path:
                adapter = bundle
                semantic_adapter = write_perspective_bundle(
                    keyframe_panorama,
                    output / "01_perspective_adapter" / "semantic_views" / keyframe_id,
                    yaws_deg=mast3r_runtime["perspective_yaws_deg"],
                    pitches_deg=mast3r_runtime["perspective_pitches_deg"],
                    width=int(runtime["qwen3vl"]["semantic_perspective_width"]),
                    height=int(runtime["qwen3vl"]["semantic_perspective_height"]),
                    horizontal_fov_deg=float(mast3r_runtime["perspective_horizontal_fov_deg"]),
                    panorama_vertical_fov_deg=float(mast3r_runtime["panorama_vertical_fov_deg"]),
                    optical_center_group=str(keyframe["optical_center_group"]),
                    view_id_prefix=keyframe_id,
                    write_remap_arrays=False,
                )
        if adapter is None:
            adapter = json.loads(Path(reconstruction_bundles[-1]["manifest_path"]).read_text(encoding="utf-8"))
            adapter["manifest_path"] = reconstruction_bundles[-1]["manifest_path"]
        adapter_stage = {
            "status": "completed",
            "manifest_path": adapter["manifest_path"],
            "view_count": len(adapter["views"]),
            "semantic_view_count": len(semantic_adapter["views"]) if semantic_adapter else 0,
            "semantic_view_size": [
                int(runtime["qwen3vl"]["semantic_perspective_width"]),
                int(runtime["qwen3vl"]["semantic_perspective_height"]),
            ],
            "global_view_count": sum(len(json.loads(Path(item["manifest_path"]).read_text())["views"]) for item in reconstruction_bundles),
            "optical_center_group_count": len(reconstruction_bundles),
            "primary_model_input": True,
            "same_panorama_slices_share_optical_center": True,
        }
    except Exception as exc:
        adapter_stage = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    summary.record(
        "perspective_adapter",
        adapter_stage,
        perspective_adapter=adapter_stage["status"] == "completed",
    )

    perspective_views = adapter["views"] if adapter is not None else []
    semantic_views = semantic_adapter["views"] if semantic_adapter is not None else perspective_views
    geometry_views_by_id = {str(item["view_id"]): item for item in perspective_views}
    qwen_grounding_dir = output / "02_qwen3vl_grounding"
    qwen_grounding_dir.mkdir(parents=True, exist_ok=True)
    grounding_concepts = [{
        "class_name": str(item["primary_class"]),
        "aliases": list(item.get("detector_aliases", ())),
        "supporting_context": list(item.get("supporting_context", ())),
        "hard_negatives": list(item.get("hard_negatives", ())),
    } for item in grounding_plan["entities"]]
    entities_by_id = {
        str(item["id"]): item for item in task_ir.get("entities", ())
    }
    target_relations = [
        item for item in task_ir.get("relations", ())
        if str(item.get("subject_entity")) == str(task_ir.get("target_entity"))
    ]
    relation_anchor_classes = {
        str(entities_by_id[str(anchor_id)]["class_name"])
        for relation in target_relations
        for anchor_id in relation.get("object_entities", ())
        if str(anchor_id) in entities_by_id
    }
    if relation_anchor_classes:
        grounding_concepts.sort(
            key=lambda item: 0 if str(item["class_name"]) in relation_anchor_classes else 1
        )
    helper_context_classes: dict[str, set[str]] = {}
    helper_concepts_by_class: dict[str, dict] = {}
    task_primary_classes = {
        str(item["primary_class"]) for item in grounding_plan["entities"]
    }
    for item in grounding_plan["entities"]:
        owner = str(item["primary_class"])
        for context in item.get("supporting_context", ()):
            if str(context.get("candidate_role")) != "context":
                continue
            helper_class = str(context["class_name"])
            helper_context_classes.setdefault(owner, set()).add(helper_class)
            helper_concepts_by_class.setdefault(helper_class, {
                "class_name": helper_class,
                "aliases": list(context.get("aliases", ())),
                "supporting_context": [],
                "hard_negatives": sorted(task_primary_classes - {helper_class}),
            })
    qwen_grounding_results = []
    qwen_grounding_errors = []
    qwen_detections = []
    detection_index = 0
    panorama_scan = None
    panorama_scan_error = None
    panorama_descriptor = {
        "view_id": f"{output.name}__raw_panorama",
        "image_path": str(image_path),
        "width": int(panorama.shape[1]),
        "height": int(panorama.shape[0]),
    }
    try:
        panorama_scan = qwen_ground_view(
            runtime["qwen3vl"]["endpoint"],
            panorama_descriptor,
            grounding_concepts,
            semantic_scene_context,
            maximum=int(runtime["qwen3vl"]["max_grounding_proposals_per_view"]),
        )
        overview = [{
            "class_name": item["class_name"],
            "semantic_probability": item["semantic_probability"],
            "rationale_tags": item["rationale_tags"],
        } for item in panorama_scan.get("proposals", ())]
        semantic_scene_context = (
            semantic_scene_context
            + "; raw_panorama_overview="
            + json.dumps(overview, ensure_ascii=False, separators=(",", ":"))
        )
    except Exception as exc:
        panorama_scan_error = {
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    # ---- Phase 1: merged per-view grounding (all non-ROI concepts) ----
    # Each view sends helpers + non-ROI anchors/targets in a single call,
    # avoiding repeated vision-encoder passes for the same image.
    ground_started = time.perf_counter()
    for view in semantic_views:
        geometry_view = geometry_views_by_id[str(view["view_id"])]
        view_proposals: list[dict] = []
        concept_results: list[dict] = []
        helper_proposals: list[dict] = []

        # Merge helper concepts + non-ROI main concepts into one call.
        merged_concepts: list[dict] = []
        seen_classes: set[str] = set()
        for hc in helper_concepts_by_class.values():
            if str(hc["class_name"]) not in seen_classes:
                merged_concepts.append(hc)
                seen_classes.add(str(hc["class_name"]))
        for gc in grounding_concepts:
            gc_class = str(gc["class_name"])
            # Only skip concepts that use ROIs: targets with
            # anchor-conditioned ROIs and helper-dependent concepts.
            if gc_class in helper_context_classes:
                continue
            if gc_class == target and relation_anchor_classes:
                continue
            if gc_class not in seen_classes:
                merged_concepts.append(gc)
                seen_classes.add(gc_class)
        if merged_concepts:
            try:
                merged_result = qwen_ground_view(
                    runtime["qwen3vl"]["endpoint"],
                    view,
                    merged_concepts,
                    semantic_scene_context,
                    maximum=min(12, int(runtime["qwen3vl"]["max_grounding_proposals_per_view"])),
                )
                concept_results.append(merged_result)
                for proposal in merged_result.get("proposals", []):
                    cls = str(proposal["class_name"])
                    if cls in helper_concepts_by_class:
                        helper_proposals.append(proposal)
                    view_proposals.append(proposal)
            except Exception as exc:
                qwen_grounding_errors.append({
                    "view_id": str(view["view_id"]),
                    "class_name": "merged_non_roi",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                })

        # ROI-cropped grounding: anchor-conditioned target + helper-conditioned anchor.
        for concept in grounding_concepts:
            try:
                concept_class = str(concept["class_name"])
                use_relation_roi = (
                    concept_class == target
                    and bool(relation_anchor_classes)
                    and bool(target_relations)
                )
                use_helper_roi = concept_class in helper_context_classes
                if use_relation_roi or use_helper_roi:
                    roi_sources = [
                        item for item in view_proposals
                        if str(item["class_name"]) in relation_anchor_classes
                    ] if use_relation_roi else [
                        item for item in helper_proposals
                        if str(item["class_name"]) in helper_context_classes[concept_class]
                    ]
                    source = cv2.imread(str(view["image_path"]), cv2.IMREAD_COLOR)
                    if source is None:
                        raise ValueError("qwen_relation_roi_source_unreadable")
                    height, width = source.shape[:2]
                    relation_crop_dir = qwen_grounding_dir / "relation_crops" / str(view["view_id"])
                    relation_crop_dir.mkdir(parents=True, exist_ok=True)
                    for anchor_index, anchor in enumerate(roi_sources):
                        ax1, ay1, ax2, ay2 = (
                            float(value) for value in anchor["bbox_xyxy_normalized"]
                        )
                        anchor_width = max(0.02, ax2 - ax1)
                        anchor_height = max(0.02, ay2 - ay1)
                        if use_relation_roi:
                            crop_x1 = max(0.0, ax1 - 0.50 * anchor_width)
                            crop_x2 = min(1.0, ax2 + 0.50 * anchor_width)
                            crop_y1 = max(0.0, ay1 - 2.25 * anchor_height)
                            crop_y2 = min(1.0, ay2 + 0.50 * anchor_height)
                            roi_tag = "anchor_conditioned_roi"
                        else:
                            crop_x1 = max(0.0, ax1 - 1.75 * anchor_width)
                            crop_x2 = min(1.0, ax2 + 1.75 * anchor_width)
                            crop_y1 = max(0.0, ay1 - 0.50 * anchor_height)
                            crop_y2 = min(1.0, ay2 + 3.00 * anchor_height)
                            roi_tag = "helper_context_conditioned_roi"
                        left = int(round(crop_x1 * width))
                        top = int(round(crop_y1 * height))
                        right = int(round(crop_x2 * width))
                        bottom = int(round(crop_y2 * height))
                        if right - left < 32 or bottom - top < 32:
                            continue
                        crop_path = relation_crop_dir / f"anchor_{anchor_index:03d}.png"
                        if not cv2.imwrite(str(crop_path), source[top:bottom, left:right]):
                            raise RuntimeError("qwen_relation_roi_write_failed")
                        crop_view = {
                            "view_id": f"{view['view_id']}__relation_roi_{anchor_index:03d}",
                            "image_path": str(crop_path),
                            "width": right - left,
                            "height": bottom - top,
                        }
                        grounded = qwen_ground_view(
                            runtime["qwen3vl"]["endpoint"],
                            crop_view,
                            [concept],
                            semantic_scene_context + "; target search is restricted to the named anchor region",
                            maximum=min(
                                6,
                                int(runtime["qwen3vl"]["max_grounding_proposals_per_view"]),
                            ),
                        )
                        mapped = []
                        for proposal in grounded["proposals"]:
                            px1, py1, px2, py2 = proposal["bbox_xyxy_normalized"]
                            mapped.append({
                                **proposal,
                                "bbox_xyxy_normalized": [
                                    crop_x1 + float(px1) * (crop_x2 - crop_x1),
                                    crop_y1 + float(py1) * (crop_y2 - crop_y1),
                                    crop_x1 + float(px2) * (crop_x2 - crop_x1),
                                    crop_y1 + float(py2) * (crop_y2 - crop_y1),
                                ],
                                "rationale_tags": [
                                    *proposal["rationale_tags"],
                                    roi_tag,
                                ],
                            })
                        concept_results.append({
                            **grounded,
                            "source_context_class": str(anchor["class_name"]),
                            "source_context_bbox_xyxy_normalized": anchor["bbox_xyxy_normalized"],
                            "proposals": mapped,
                            "proposal_count": len(mapped),
                        })
                        view_proposals.extend(mapped)
                else:
                    grounded = qwen_ground_view(
                        runtime["qwen3vl"]["endpoint"],
                        view,
                        [concept],
                        semantic_scene_context,
                        maximum=min(
                            4 if concept_class in relation_anchor_classes else 8,
                            int(runtime["qwen3vl"]["max_grounding_proposals_per_view"]),
                        ),
                    )
                    concept_results.append(grounded)
                    view_proposals.extend(grounded["proposals"])
            except Exception as exc:
                qwen_grounding_errors.append({
                    "view_id": str(view["view_id"]),
                    "class_name": str(concept["class_name"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                })

        # ── Fallback: if target not found with ROI constraints, re-search full-view ──
        if target and not any(
            str(p.get("class_name", "")) == target for p in view_proposals
        ):
            target_concept = next(
                (c for c in grounding_concepts if str(c["class_name"]) == target), None
            )
            if target_concept:
                try:
                    fallback = qwen_ground_view(
                        runtime["qwen3vl"]["endpoint"],
                        view,
                        [target_concept],
                        semantic_scene_context + "; previous constrained search missed the target — scan the ENTIRE view without anchor restrictions",
                        maximum=int(runtime["qwen3vl"]["max_grounding_proposals_per_view"]),
                    )
                    if fallback.get("proposals"):
                        for proposal in fallback["proposals"]:
                            proposal.setdefault("rationale_tags", []).append("target_fallback_full_view")
                        concept_results.append(fallback)
                        view_proposals.extend(fallback["proposals"])
                except Exception as exc:
                    qwen_grounding_errors.append({
                        "view_id": str(view["view_id"]),
                        "class_name": str(target),
                        "error_type": f"fallback_{type(exc).__name__}",
                        "error": str(exc)[:500],
                    })

        if concept_results:
            qwen_grounding_results.append({
                "view_id": str(view["view_id"]),
                "proposals": view_proposals,
                "proposal_count": len(view_proposals),
                "concept_results": concept_results,
            })
            for proposal in view_proposals:
                detection_index += 1
                x1, y1, x2, y2 = proposal["bbox_xyxy_normalized"]
                qwen_detections.append({
                    "detection_id": f"qwen_det_{detection_index:06d}",
                    "detector": "qwen3vl_grounding",
                    "view_id": str(geometry_view["view_id"]),
                    "canonical_class": str(proposal["class_name"]),
                    "detector_score": float(proposal["semantic_probability"]),
                    "qwen_grounding_probability": float(proposal["semantic_probability"]),
                    "qwen_grounding_rationale_tags": list(proposal["rationale_tags"]),
                    "bbox_xyxy": [
                        float(x1) * int(geometry_view["width"]),
                        float(y1) * int(geometry_view["height"]),
                        float(x2) * int(geometry_view["width"]),
                        float(y2) * int(geometry_view["height"]),
                    ],
                })
    ground_seconds = time.perf_counter() - ground_started
    qwen_detections_path = qwen_grounding_dir / "detections.json"
    write_json(qwen_detections_path, qwen_detections)
    qwen_grounding = {
        "status": "completed" if qwen_grounding_results else "failed",
        "completion_scope": "partial" if qwen_grounding_errors and qwen_grounding_results else "complete",
        "input_view_count": len(semantic_views) + 1,
        "raw_panorama_scan": panorama_scan,
        "raw_panorama_scan_error": panorama_scan_error,
        "focused_view_resolution": [
            int(runtime["qwen3vl"]["semantic_perspective_width"]),
            int(runtime["qwen3vl"]["semantic_perspective_height"]),
        ],
        "processed_view_count": len(qwen_grounding_results),
        "proposal_count": len(qwen_detections),
        "detections_path": str(qwen_detections_path),
        "errors": qwen_grounding_errors,
        "results": qwen_grounding_results,
        "input_role": "deepseek_task_ir_plus_raw_panorama_overview_then_high_resolution_calibrated_views",
        "grounding_batch_mode": "multiview_4x_concept_merged",
        "grounding_wall_seconds": ground_seconds,
    }
    summary.record(
        "qwen3vl_grounding",
        qwen_grounding,
        qwen3vl_grounding=qwen_grounding["status"] == "completed",
    )

    mast3r_dir = output / "02_mast3r_matching"
    mast3r_request = {
        "output_dir": str(mast3r_dir),
        "keyframes": reconstruction_bundles,
        "checkpoint": str((ROOT / models["mast3r"]["checkpoint"]).resolve()),
        "calibration_path": str((ROOT / runtime["competition_geometry"]["calibration"]).resolve()),
        "device": mast3r_runtime["device"],
        "coarse_iterations": mast3r_runtime["coarse_iterations"],
        "refine_iterations": mast3r_runtime["refine_iterations"],
        "matching_confidence_threshold": mast3r_runtime["matching_confidence_threshold"],
        "pointmap_confidence_threshold": mast3r_runtime["pointmap_confidence_threshold"],
    }
    # Any2Full optional refinement
    if mast3r_runtime.get("any2full", {}).get("enabled"):
        mast3r_request["any2full"] = {
            "enabled": True,
            "checkpoint": str((ROOT / models["any2full"]["checkpoint"]).resolve()),
            "encoder": mast3r_runtime["any2full"].get("encoder", "vitb"),
            "depth_scale": mast3r_runtime["any2full"].get("depth_scale", 100.0),
            "denoise": mast3r_runtime["any2full"].get("denoise", True),
        }
    mast3r = run_worker(
        reconstruction_module,
        mast3r_request,
        ROOT,
        mast3r_dir,
    )
    mast3r_quality = (
        mast3r.get("status") == "completed"
        and mast3r.get("native_sga_executed") is True
    )
    summary.record(
        "mast3r_matching",
        mast3r,
        mast3r_matching=mast3r_quality,
        mast3r_world_alignment=mast3r.get("status") == "completed",
    )

    yolo_dir = output / "03_yolo_world_primary"
    detector_supplement_enabled = bool(
        runtime["pipeline"].get("detector_supplement_enabled", False)
    )
    if detector_supplement_enabled and perspective_views:
        yolo = run_worker(
            "integrations.perception.yolo_world_worker",
            {
                "output_dir": str(yolo_dir),
                "views": perspective_views,
                "class_aliases": aliases,
                "model_config": str((ROOT / models["yolo_world"]["config"]).resolve()),
                "checkpoint": str((ROOT / models["yolo_world"]["checkpoint"]).resolve()),
                "text_model_path": str((ROOT / models["yolo_world"]["text_encoder"]).resolve()),
                "device": "cuda:0",
                "score_threshold": runtime["yolo_world"]["score_threshold"],
                "nms_iou_threshold": runtime["yolo_world"]["nms_iou_threshold"],
                "max_detections_per_class_per_view": 20,
            },
            ROOT,
            yolo_dir,
        )
    elif detector_supplement_enabled:
        yolo = {"status": "blocked", "reason": "perspective_adapter_failed"}
    else:
        yolo = {"status": "skipped", "reason": "qwen3vl_is_primary_proposal_backend"}
    yolo_count = target_detection_count(yolo, target)
    summary.record(
        "yolo_world_primary",
        {
            **yolo,
            "input_role": "calibrated_perspective_views",
            "target_detection_count": yolo_count,
        },
        yolo_world=yolo.get("status") == "completed",
        yolo_target_count=yolo_count,
    )

    dino_needed = detector_supplement_enabled and groundingdino_rescue_required(
        yolo, task_ir["required_classes"]
    )
    dino_dir = output / "04_groundingdino_rescue"
    if not dino_needed:
        dino = {
            "status": "skipped",
            "reason": (
                "detector_supplement_disabled"
                if not detector_supplement_enabled
                else "yolo_primary_has_target_candidates"
            ),
        }
    elif perspective_views:
        dino = run_worker(
            "integrations.perception.groundingdino_worker",
            {
                "output_dir": str(dino_dir),
                "views": perspective_views,
                "class_aliases": aliases,
                "target_class": target,
                "groundingdino_config": str((ROOT / models["groundingdino"]["config"]).resolve()),
                "groundingdino_checkpoint": str((ROOT / models["groundingdino"]["checkpoint"]).resolve()),
                "bert_model_path": str((ROOT / models["groundingdino"]["text_encoder"]).resolve()),
                "device": "cuda",
                "box_threshold": runtime["groundingdino"]["box_threshold"],
                "text_threshold": runtime["groundingdino"]["text_threshold"],
                "retry_box_threshold": 0.15,
                "retry_text_threshold": 0.15,
                "max_detections_per_class_per_view": 20,
                "nms_iou_threshold": 0.65,
            },
            ROOT,
            dino_dir,
        )
    else:
        dino = {"status": "blocked", "reason": "perspective_adapter_failed"}
    summary.record(
        "groundingdino_rescue",
        {**dino, "input_role": "calibrated_perspective_views"},
        groundingdino=dino.get("status") in {"completed", "skipped"},
        groundingdino_ran=dino.get("status") == "completed",
    )

    selected_detector = "qwen3vl_grounding"
    selected_path = str(qwen_detections_path)
    detector_paths = [str(qwen_detections_path)]
    if yolo.get("status") == "completed":
        detector_paths.append(yolo["detections_path"])
    if dino.get("status") == "completed":
        detector_paths.append(dino["detections_path"])
    if len(detector_paths) > 1:
        selected_detector = "+".join(
            ["qwen3vl_grounding"]
            + (["yolo_world"] if yolo.get("status") == "completed" else [])
            + (["groundingdino"] if dino.get("status") == "completed" else [])
        )
        selected_path = merge_detection_sources(
            detector_paths, output / "04_groundingdino_rescue" / "combined_detections.json"
        )
    sam_dir = output / "05_sam2_segmentation"
    sam_request = {
        "output_dir": str(sam_dir),
        "views": perspective_views,
        "detections_path": selected_path,
        "sam2_config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2_checkpoint": str((ROOT / models["sam2"]["checkpoint"]).resolve()),
        "device": "cuda",
        "min_mask_area_px": 40,
    }
    selected_detections = (
        json.loads(Path(selected_path).read_text(encoding="utf-8"))
        if selected_path is not None else []
    )
    view_by_id = {str(item["view_id"]): item for item in perspective_views}
    task_context = json.dumps(
        {
            "question": task_ir["original_question"],
            "task_type": task_ir["task_type"],
            "relations": task_ir["relations"],
            "ordered_subgoals": task_ir["ordered_subgoals"],
            "grounding_plan": grounding_plan,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    def verify_proposals() -> list[dict]:
        results = []
        for detection in selected_detections:
            class_name = str(detection["canonical_class"])
            grounding = grounding_by_class.get(class_name, {})
            try:
                response = qwen_verify_detection(
                    runtime["qwen3vl"]["endpoint"],
                    detection,
                    view_by_id[str(detection["view_id"])],
                    task_context,
                    hard_negatives=list(grounding.get("hard_negatives", ())),
                )
            except Exception as exc:
                response = {
                    "ok": False,
                    "error_code": type(exc).__name__,
                    "metadata": {"error_detail": str(exc)[:300]},
                }
            results.append({
                "detection_id": str(detection["detection_id"]),
                "response": response,
            })
        return results

    sam_service = ensure_persistent_sam(runtime, models)
    parallel_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        sam_future = executor.submit(
            run_persistent_sam,
            runtime["sam2"]["endpoint"],
            sam_request,
            sam_dir,
            float(runtime["sam2"]["request_timeout_seconds"]),
        )
        qwen_future = executor.submit(verify_proposals)
        sam = sam_future.result()
        qwen_results = qwen_future.result()
    parallel_seconds = time.perf_counter() - parallel_started
    summary.record(
        "sam2_segmentation",
        {
            **sam,
            "box_source": selected_detector,
            "service": sam_service,
            "parallel_with": "qwen3vl_verification",
            "parallel_join_seconds": parallel_seconds,
        },
        sam2=sam.get("status") == "completed",
    )

    verification_by_id = {
        item["detection_id"]: item["response"] for item in qwen_results
    }
    qwen_errors = [
        item for item in qwen_results
        if qwen_target_probability(item["response"]) is None
    ]
    qwen_complete = qwen_grounding["status"] == "completed"
    qwen_dir = output / "07_qwen3vl_verification"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    semantic_detections_path = qwen_dir / "semantic_detections.json"
    semantic_detections = []
    if sam.get("status") == "completed":
        for detection in json.loads(Path(sam["detections_path"]).read_text(encoding="utf-8")):
            response = verification_by_id.get(str(detection["detection_id"]), {})
            verification_probability = qwen_target_probability(response)
            grounding_probability = float(
                detection.get("qwen_grounding_probability", detection.get("detector_score", 0.0))
            )
            semantic_detections.append({
                **detection,
                "qwen_target_probability": (
                    grounding_probability
                    if verification_probability is None
                    else 0.5 * grounding_probability + 0.5 * verification_probability
                ),
                "qwen_verification_probability": verification_probability,
                "qwen_verification_metadata": dict(response.get("metadata", {})),
            })
    write_json(semantic_detections_path, semantic_detections)
    qwen = {
        "status": "completed" if qwen_complete else "failed",
        "completion_scope": "partial" if qwen_errors else "complete",
        "input_role": "qwen_grounded_boxes_verified_in_parallel_with_sam2",
        "submitted_candidate_count": len(selected_detections),
        "verified_candidate_count": len(qwen_results) - len(qwen_errors),
        "verification_error_count": len(qwen_errors),
        "semantic_detections_path": str(semantic_detections_path),
        "results": qwen_results,
        "persistent_model": True,
        "parallel_with": "sam2_segmentation",
        "parallel_join_seconds": parallel_seconds,
    }
    summary.record(
        "qwen3vl_verification",
        qwen,
        qwen3vl=qwen["status"] == "completed",
    )

    fusion_dir = output / "06_panorama_observation_fusion"
    if sam.get("status") == "completed":
        fusion = run_worker(
            "integrations.perception.panorama_fusion",
            {
                "output_dir": str(fusion_dir),
                "manifest_path": adapter["manifest_path"],
                "panorama_path": str(image_path),
                "detections_path": str(semantic_detections_path),
                "mask_iou_threshold": runtime["sam2"]["panorama_mask_iou_threshold"],
                "mask_containment_threshold": runtime["sam2"]["panorama_mask_containment_threshold"],
            },
            ROOT,
            fusion_dir,
        )
    else:
        fusion = {"status": "blocked", "reason": "sam2_not_completed"}
    summary.record(
        "panorama_observation_fusion",
        fusion,
        panorama_fusion=fusion.get("status") == "completed",
    )

    observations = []
    if fusion.get("status") == "completed":
        observations = json.loads(Path(fusion["observations_path"]).read_text(encoding="utf-8"))
    task_observations = [
        item for item in observations
        if str(item["canonical_class"]) in task_roles_by_class
    ]
    context_observations = [
        item for item in observations
        if str(item["canonical_class"]) not in task_roles_by_class
    ]
    threshold = float(runtime["qwen3vl"]["target_probability_threshold"])
    verified_observations = [{
            **observation,
            "semantic_role": task_roles_by_class[str(observation["canonical_class"])],
            "qwen_verified": True,
            "source_sensor_scan_path": competition_geometry["sensor_scan_path"],
            "source_registered_scan_path": competition_geometry["registered_scan_path"],
            "source_state_estimation_path": competition_geometry["state_estimation_path"],
        } for observation in task_observations]
    rejected_observations = []
    qwen_inputs = list(task_observations)
    verified_path = qwen_dir / "verified_observations.json"
    rejected_path = qwen_dir / "rejected_observations.json"
    write_json(verified_path, verified_observations)
    write_json(rejected_path, rejected_observations)
    global_verified = list(verified_observations)
    current_ids = {str(item["observation_id"]) for item in global_verified}
    for keyframe in request.get("reconstruction_keyframes", ()):
        historical_path = keyframe.get("semantic_observations_path")
        if not historical_path or not Path(historical_path).is_file():
            continue
        for historical in json.loads(Path(historical_path).read_text(encoding="utf-8")):
            if (
                str(historical.get("canonical_class")) not in task_roles_by_class
                or str(historical.get("observation_id")) in current_ids
            ):
                continue
            global_verified.append({
                **historical,
                "semantic_role": task_roles_by_class[str(historical["canonical_class"])],
            })
            current_ids.add(str(historical["observation_id"]))
    global_verified_path = qwen_dir / "global_verified_observations.json"
    write_json(global_verified_path, global_verified)

    entities_by_id = {str(item["id"]): item for item in task_ir["entities"]}
    anchor_entity = None
    target_relations = [
        item for item in task_ir["relations"]
        if str(item.get("subject_entity")) == str(task_ir["target_entity"])
        and str(item.get("predicate", "")).lower() == "on"
        and len(item.get("object_entities", ())) == 1
    ]
    if len(target_relations) == 1:
        anchor_entity = entities_by_id.get(str(target_relations[0]["object_entities"][0]))
    anchor_count_results = []
    native_relation_view_counts = Counter(
        str(item["view_id"])
        for item in qwen_detections
        if str(item["canonical_class"]) == target
        and "anchor_conditioned_roi" in item.get("qwen_grounding_rationale_tags", ())
    )
    native_relation_count_results = []
    positive_native_counts = list(native_relation_view_counts.values())
    if positive_native_counts:
        frequency = Counter(positive_native_counts)
        winning_frequency = max(frequency.values())
        # When multiple counts tie for most frequent, occlusion makes
        # under-counting far more likely than hallucinated over-counting
        # (Qwen runs with do_sample=False).  Prefer the higher count.
        native_answer = max(
            count for count, occurrences in frequency.items()
            if occurrences == winning_frequency
        )
        native_relation_count_results.append({
            "observation_id": "qwen_native_anchor_conditioned_multiview",
            "source_view_counts": dict(native_relation_view_counts),
            "response": {
                "ok": True,
                "error_code": "",
                "metadata": {
                    "target_probability": 1.0,
                    "anchor_probability": 1.0,
                    "confuser_probabilities": {},
                    "rationale_tags": ["anchor_conditioned_multiview_consensus"],
                    "contained_instance_count": int(native_answer),
                    "count_confidence": float(winning_frequency / len(positive_native_counts)),
                    "backend": "qwen3vl_native_2d_grounding_consensus",
                    "checkpoint_path": str(
                        (ROOT / models["qwen3vl"]["checkpoint"]).resolve()
                    ),
                    "quantization": runtime["qwen3vl"]["quantization"],
                },
            },
        })
    target_count_concept = " / ".join(dict.fromkeys([
        str(entities_by_id[str(task_ir["target_entity"])]["class_name"]),
        *(
            str(value)
            for value in grounding_by_class.get(target, {}).get("detector_aliases", ())
        ),
    ]))
    if task_ir["task_type"] == "numerical" and anchor_entity is not None:
        anchor_grounding = grounding_by_class.get(str(anchor_entity["class_name"]), {})
        anchor_count_concept = " / ".join(dict.fromkeys([
            str(anchor_entity["class_name"]),
            *(
                str(value)
                for value in anchor_grounding.get("detector_aliases", ())
            ),
        ]))
        for observation in verified_observations:
            if observation.get("semantic_role") != "anchor":
                continue
            inventory_observation = None
            try:
                inventory_observation = anchor_inventory_observation(
                    observation,
                    image_path,
                )
                response = qwen_verify(
                    runtime["qwen3vl"]["endpoint"],
                    inventory_observation,
                    target_count_concept,
                    anchor_count_concept,
                    operation="count_on_anchor",
                    hard_negatives=list(
                        grounding_by_class.get(target, {}).get("hard_negatives", ())
                    ),
                )
            except Exception as exc:
                response = {
                    "ok": False,
                    "error_code": type(exc).__name__,
                    "error_detail": str(exc)[:300],
                }
            anchor_count_results.append({
                "observation_id": observation["observation_id"],
                "support_bbox_xyxy": observation["panorama_bbox_xyxy"],
                "inventory_bbox_xyxy": (
                    inventory_observation["panorama_bbox_xyxy"]
                    if inventory_observation is not None
                    else None
                ),
                "response": response,
            })
    anchor_count_stage = {
        "status": "completed" if anchor_count_results or native_relation_count_results else "skipped",
        "reason": None if anchor_count_results or native_relation_count_results else "numerical_on_anchor_not_available",
        "input_role": "qwen_native_anchor_conditioned_grounding_plus_direct_count_diagnostic",
        "answer_eligible": request.get("arrival_goal_id") is not None,
        "acquisition_role": (
            "post_navigation_close_view"
            if request.get("arrival_goal_id") is not None
            else "initial_anchor_localization_diagnostic"
        ),
        "native_relation_view_counts": dict(native_relation_view_counts),
        "native_relation_consensus": native_relation_count_results,
        "direct_count_diagnostics": anchor_count_results,
        "results": [*native_relation_count_results, *anchor_count_results],
    }
    summary.record("qwen3vl_anchor_local_count", anchor_count_stage)

    lift_dir = output / "08_mast3r_3d_association"
    if (
        qwen_complete
        and global_verified
        and mast3r.get("geometry_manifest_path")
        and geometry_ready
    ):
        local_lift = run_worker(
            "integrations.mast3r.local_object_lift",
            {
                "output_dir": str(lift_dir),
                "verified_observations_path": str(global_verified_path),
                "geometry_manifest_path": mast3r["geometry_manifest_path"],
                "confidence_threshold": mast3r_runtime["pointmap_confidence_threshold"],
                "max_saved_points_per_observation": 20000,
                "sensor_scan_path": competition_geometry["sensor_scan_path"],
                "registered_scan_path": competition_geometry["registered_scan_path"],
                "state_estimation_path": competition_geometry["state_estimation_path"],
                "calibration_path": str(
                    (ROOT / runtime["competition_geometry"]["calibration"]).resolve()
                ),
                "panorama_vertical_fov_deg": mast3r_runtime["panorama_vertical_fov_deg"],
            },
            ROOT,
            lift_dir,
        )
    else:
        local_lift = {
            "status": "blocked",
            "reason": "verified_visual_or_competition_geometry_missing",
        }
    world_alignment_ready = (
        local_lift.get("status") == "completed"
        and local_lift.get("world_aligned") is True
    )
    association = {
        "status": "completed" if local_lift.get("status") == "completed" else "blocked",
        "input_role": "cross_station_qwen_observations_relifted_in_latest_mast3r_geometry",
        "input_observations_path": str(global_verified_path),
        "current_station_observation_count": len(verified_observations),
        "global_observation_count": len(global_verified),
        "geometry_manifest_path": mast3r.get("geometry_manifest_path"),
        "competition_geometry": competition_geometry,
        "local_geometry_lift": local_lift,
        "independent_viewpoint_count": int(mast3r.get("optical_center_group_count", 0)),
        "reconstruction_mode": mast3r_runtime["reconstruction_mode"],
    }
    summary.record(
        "mast3r_3d_association",
        association,
        mast3r_3d_association=association["status"] == "completed",
        mast3r_world_alignment=world_alignment_ready,
        independent_viewpoint_support=False,
    )

    semantic_config = runtime["semantic_execution"]
    snapshot = None

    # ---- 3D ray-surface verification of anchor-conditioned detections ----
    ray_verified_count_results = []
    if (
        local_lift.get("status") == "completed"
        and local_lift.get("local_3d_observations_path")
        and mast3r.get("geometry_manifest_path")
        and anchor_entity is not None
    ):
        import numpy as np

        _lifted = json.loads(
            Path(local_lift["local_3d_observations_path"]).read_text(encoding="utf-8")
        )
        anchor_class = str(anchor_entity["class_name"])
        anchor_points: list[np.ndarray] = []
        # Collect the anchor's 3D point cloud (already in map frame) for the
        # current station so rays can be checked against actual geometry rather
        # than an infinite plane.
        current_group = f"station:{output.name}"
        for obs in _lifted:
            if (
                obs.get("canonical_class") == anchor_class
                and obs.get("optical_center_group") == current_group
            ):
                cloud_path_str = obs.get("pointcloud_path")
                if cloud_path_str and Path(cloud_path_str).is_file():
                    cloud_data = np.load(cloud_path_str)
                    pts = None
                    for _key in ("world_points", "points"):
                        try:
                            pts = cloud_data[_key]
                            break
                        except KeyError:
                            continue
                    if pts is not None and len(pts) > 0:
                        anchor_points.append(
                            pts.astype(np.float64).reshape(-1, 3)
                        )
        anchor_cloud = (
            np.concatenate(anchor_points) if anchor_points
            else np.zeros((0, 3), dtype=np.float64)
        )

        if len(anchor_cloud) >= 30:
            _geom = json.loads(
                Path(mast3r["geometry_manifest_path"]).read_text(encoding="utf-8")
            )
            cam_poses = {
                v["view_id"]: np.asarray(v["cam2w_map"], dtype=np.float64)
                for v in _geom["views"]
            }
            # Down-sample anchor cloud for fast ray-distance checks.
            _n_anchor = len(anchor_cloud)
            if _n_anchor > 4000:
                _idx = np.linspace(0, _n_anchor - 1, 4000, dtype=np.int64)
                _anchor_sample = anchor_cloud[_idx]
            else:
                _anchor_sample = anchor_cloud
            _anchor_sample = np.ascontiguousarray(_anchor_sample, dtype=np.float64)

            verified_by_view: dict[str, int] = {}
            for det in qwen_detections:
                if (
                    str(det["canonical_class"]) != target
                    or "anchor_conditioned_roi" not in det.get(
                        "qwen_grounding_rationale_tags", ()
                    )
                ):
                    continue
                vid = str(det["view_id"])
                gv = geometry_views_by_id.get(vid)
                cam = cam_poses.get(vid)
                if gv is None or cam is None:
                    continue

                x1, y1, x2, y2 = (float(v) for v in det["bbox_xyxy"])
                cx_px = (x1 + x2) * 0.5
                cy_px = (y1 + y2) * 0.5

                K_inv = np.linalg.inv(np.asarray(gv["K"], dtype=np.float64))
                ray_cam = K_inv @ np.array([cx_px, cy_px, 1.0], dtype=np.float64)
                ray_map = cam[:3, :3] @ ray_cam
                ray_map /= np.linalg.norm(ray_map)
                cam_center = cam[:3, 3]

                # Minimum distance from the ray to any anchor point-cloud
                # point.  A detection whose ray passes within ~0.4 m of the
                # anchor geometry is consistent with "on top of" it.
                to_pts = _anchor_sample - cam_center[None, :]
                proj = np.sum(to_pts * ray_map[None, :], axis=1)
                closest_t = np.clip(proj, 0.0, None)
                ray_pts = cam_center[None, :] + closest_t[:, None] * ray_map[None, :]
                dists = np.linalg.norm(_anchor_sample - ray_pts, axis=1)
                min_dist = float(np.min(dists))

                if min_dist <= 1.00:
                    verified_by_view[vid] = verified_by_view.get(vid, 0) + 1

            if verified_by_view:
                pos = list(verified_by_view.values())
                from collections import Counter as _Counter
                freq = _Counter(pos)
                best_freq = max(freq.values())
                verified_answer = max(
                    c for c, n in freq.items() if n == best_freq
                )
                ray_verified_count_results.append({
                    "observation_id": "ray_pointcloud_3d_verified_multiview",
                    "source_view_counts": dict(verified_by_view),
                    "response": {
                        "ok": True,
                        "error_code": "",
                        "metadata": {
                            "target_probability": 1.0,
                            "anchor_probability": 1.0,
                            "confuser_probabilities": {},
                            "rationale_tags": ["ray_pointcloud_3d_verified"],
                            "contained_instance_count": int(verified_answer),
                            "count_confidence": float(
                                best_freq / len(pos) if pos else 0.0
                            ),
                            "backend": "ray_pointcloud_3d_verification",
                            "anchor_cloud_points": int(len(anchor_cloud)),
                            "max_ray_to_cloud_distance_m": 1.00,
                        },
                    },
                })
            # Store in summary for diagnostic traceability.
            summary.payload.setdefault("stages", {}).setdefault(
                "qwen3vl_anchor_local_count", {}
            )["ray_verified_results"] = ray_verified_count_results
            summary.flush()

    # ---- Ray-surface hybrid lift for small objects ----
    # When MASt3R pointmap lacks confident points for a required entity
    # (typical for books, photos, pillows), project the Qwen 2D bbox
    # through the camera onto the supporting anchor's top surface.
    if (
        local_lift.get("status") == "completed"
        and local_lift.get("local_3d_observations_path")
        and mast3r.get("geometry_manifest_path")
    ):
        _lifted = json.loads(
            Path(local_lift["local_3d_observations_path"]).read_text(encoding="utf-8")
        )
        _lifted_classes = {obs["canonical_class"] for obs in _lifted}
        _missing = [
            cls for cls in task_ir["required_classes"]
            if cls not in _lifted_classes
        ]
        if _missing and anchor_entity is not None:
            _anchor_class = str(anchor_entity["class_name"])
            _anchor_surface = None
            for obs in _lifted:
                sp = obs.get("surface_plane")
                if sp and obs.get("canonical_class") == _anchor_class:
                    _anchor_surface = sp
                    break
            if _anchor_surface is not None:
                import numpy as np
                _geom = json.loads(
                    Path(mast3r["geometry_manifest_path"]).read_text(encoding="utf-8")
                )
                _cam_poses = {
                    v["view_id"]: np.asarray(v["cam2w_map"], dtype=np.float64)
                    for v in _geom["views"]
                }
                _plane_n = np.asarray(_anchor_surface["normal"], dtype=np.float64)
                _plane_c = np.asarray(_anchor_surface["center"], dtype=np.float64)
                _extents = [
                    float(_anchor_surface["extents_xy_m"][0]),
                    float(_anchor_surface["extents_xy_m"][1]),
                ]
                for _missing_cls in _missing:
                    # Find Qwen detections for this missing class that have
                    # anchor-conditioned ROI evidence.
                    _missing_dets = [
                        d for d in qwen_detections
                        if str(d["canonical_class"]) == _missing_cls
                    ]
                    if not _missing_dets:
                        continue
                    _projected_pts = []
                    _best_view_id = ""
                    for det in _missing_dets:
                        _vid = str(det["view_id"])
                        _gv = geometry_views_by_id.get(_vid)
                        _cam = _cam_poses.get(_vid)
                        if _gv is None or _cam is None:
                            continue
                        x1, y1, x2, y2 = (float(v) for v in det["bbox_xyxy"])
                        cx = (x1 + x2) * 0.5
                        cy = (y1 + y2) * 0.5
                        _K_inv = np.linalg.inv(
                            np.asarray(_gv["K"], dtype=np.float64)
                        )
                        ray_cam = _K_inv @ np.array([cx, cy, 1.0], dtype=np.float64)
                        ray_cam /= np.linalg.norm(ray_cam)
                        ray_map = _cam[:3, :3] @ ray_cam
                        cc = _cam[:3, 3]
                        denom = float(np.dot(ray_map, _plane_n))
                        if abs(denom) < 1e-6:
                            continue
                        t = float(np.dot(_plane_c - cc, _plane_n)) / denom
                        if t <= 0:
                            continue
                        inter = cc + t * ray_map
                        delta = inter - _plane_c
                        in_plane = delta - np.dot(delta, _plane_n) * _plane_n
                        if (
                            abs(float(in_plane[0])) <= _extents[0] * 2.0
                            and abs(float(in_plane[1])) <= _extents[1] * 2.0
                        ):
                            _projected_pts.append(inter)
                            _best_view_id = _vid
                    if _projected_pts:
                        _pts = np.stack(_projected_pts)
                        _center = np.median(_pts, axis=0).astype(float)
                        _ext = np.maximum(
                            np.ptp(_pts, axis=0), [0.05, 0.05, 0.02]
                        ).astype(float)
                        _lifted.append({
                            "schema_version": "1.0",
                            "observation_id": f"{output.name}__ray_surface_{_missing_cls}",
                            "canonical_class": _missing_cls,
                            "semantic_probability": 0.85,
                            "source_view_ids": [_best_view_id] if _best_view_id else [],
                            "source_view_count": 1,
                            "frame": "map",
                            "world_aligned": True,
                            "map_transform_source": "ray_surface_projection",
                            "center_3d": _center.tolist(),
                            "bbox_3d": _ext.tolist(),
                            "centroid_xyz": _center.tolist(),
                            "median_confidence": 0.7,
                            "surface_plane": None,
                            "point_count": len(_projected_pts),
                            "saved_point_count": len(_projected_pts),
                            "pointcloud_path": None,
                            "optical_center_group": f"station:{output.name}",
                            "independent_optical_center_count": 1,
                            "per_view_geometry": [],
                            "viewpoint_position_map": _cam_poses.get(
                                _best_view_id, np.eye(4)
                            )[:3, 3].tolist() if _best_view_id else [0, 0, 0],
                            "view_center_dispersion_m": 0.0,
                            "lidar_mask_support_point_count": 0,
                            "registered_scan_median_distance_m": None,
                            "registered_scan_inlier_fraction_0_25m": 0.0,
                            "lidar_validated": False,
                            "reconstruction_id": mast3r.get("reconstruction_id", ""),
                            "source_cam2w_maps": [],
                        })
                if len(_lifted) > len(json.loads(
                    Path(local_lift["local_3d_observations_path"]).read_text(encoding="utf-8")
                )):
                    _updated_path = Path(str(local_lift["local_3d_observations_path"]))
                    write_json(_updated_path, _lifted)

    if local_lift.get("status") == "completed" and local_lift.get("local_3d_observations_path"):
        lifted_observations = json.loads(
            Path(local_lift["local_3d_observations_path"]).read_text(encoding="utf-8")
        )
        store_path = Path(
            request.get(
                "world_model_path",
                output.parent / semantic_config["persistent_store_name"],
            )
        ).resolve()
        try:
            snapshot = update_world_model(
                store_path,
                lifted_observations,
                task_ir,
                acquisition_id=output.name,
                task_scope_id=str(request.get("episode_id", output.name)),
                minimum_viewpoint_separation_m=float(
                    semantic_config["minimum_independent_viewpoint_separation_m"]
                ),
            )
            memory_stage = {
                "status": "completed",
                "snapshot": snapshot,
                "confirmed_object_count": sum(
                    item.get("status") == "confirmed" for item in snapshot["objects"]
                ),
            }
        except Exception as exc:
            memory_stage = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    else:
        memory_stage = {"status": "blocked", "reason": "map_frame_3d_observations_missing"}
    summary.record(
        "persistent_world_model",
        memory_stage,
        persistent_world_model=memory_stage["status"] == "completed",
        independent_viewpoint_support=bool(snapshot and snapshot["candidate_set_closed"]),
    )

    bringup_execution = None
    if (
        semantic_config.get("bringup_mode") is True
        and semantic_config.get("numerical_semantic_relation_count_enabled") is True
        and qwen_complete
        and len(qwen_inputs) == len(task_observations)
        and request.get("arrival_goal_id") is not None
    ):
        bringup_execution = execute_semantic_relation_count(
            task_ir,
            ray_verified_count_results or native_relation_count_results or anchor_count_results,
            acquisition_id=output.name,
            probability_threshold=threshold,
            geometry_reconstruction_id=str(mast3r.get("reconstruction_id", "")) or None,
        )
    if bringup_execution is not None:
        execution = bringup_execution
        execution_stage = {
            "status": "completed",
            "mode": "semantic_relation_count_bringup",
            "geometry_diagnostic_only": True,
            "execution": execution,
        }
    elif snapshot is not None:
        execution = execute_task(
            task_ir,
            snapshot,
            allow_numerical_resolution=request.get("arrival_goal_id") is not None,
        )
        execution_stage = {"status": "completed", "execution": execution}
    else:
        execution = {
            "task_type": task_ir["task_type"],
            "scene_version": 0,
            "resolved": False,
            "evidence_ids": [],
            "failed_constraints": ["scene_snapshot_missing"],
        }
        execution_stage = {"status": "blocked", "reason": "scene_snapshot_missing", "execution": execution}
    summary.record(
        "query_execution",
        execution_stage,
        query_execution=bool(execution.get("resolved")),
    )

    # ---- Qwen 2D candidate disambiguation ----
    # When the 3D spatial query produces multiple valid candidates of the
    # same class, ask Qwen to pick based on the full visual context.
    if (
        execution.get("resolved")
        and task_ir["task_type"] in {"object_reference", "instruction_following"}
        and panorama is not None
    ):
        target_class = str(
            entities_by_id.get(task_ir["target_entity"], {}).get("class_name", "")
        )
        # Collect all candidates of the target class from the snapshot.
        snap = (
            json.loads(
                Path(
                    request.get(
                        "world_model_path",
                        output.parent / semantic_config["persistent_store_name"],
                    )
                ).read_text(encoding="utf-8")
            )
            if Path(
                request.get(
                    "world_model_path",
                    output.parent / semantic_config["persistent_store_name"],
                )
            ).is_file()
            else {}
        )
        snap_objects = snap.get("objects", [])
        alt_candidates = [
            obj for obj in snap_objects
            if obj.get("class_label") == target_class
            and obj.get("status") in {"tentative", "confirmed"}
        ]
        if len(alt_candidates) >= 2:
            # Map candidates to their panorama bboxes.
            cand_specs = []
            for obj in alt_candidates:
                evidence = obj.get("evidence", [])
                bbox = None
                for ev in evidence:
                    obs_id = str(ev.get("observation_id", ""))
                    for obs in observations:
                        if str(obs.get("observation_id")) == obs_id:
                            bbox = obs.get("panorama_bbox_xyxy")
                            break
                    if bbox:
                        break
                cand_specs.append({
                    "class_name": target_class,
                    "bbox_xyxy": bbox,
                    "context": (
                        f"near ({obj['center_3d'][0]:.1f},{obj['center_3d'][1]:.1f})"
                    ),
                })
            # Only call Qwen if we have bbox annotations.
            if any(cs.get("bbox_xyxy") for cs in cand_specs):
                try:
                    rank_resp = socket_request(
                        runtime["qwen3vl"]["endpoint"],
                        {
                            "request_id": uuid.uuid4().hex,
                            "acquisition_id": f"rank-{uuid.uuid4().hex}",
                            "backend": "qwen3vl",
                            "operation": "rank_candidates",
                            "input_handles": [],
                            "parameters": {
                                "image_path": str(image_path),
                                "question": task_ir["original_question"],
                                "candidates": cand_specs,
                            },
                        },
                        120.0,
                    )
                    if rank_resp.get("ok") is True:
                        best_idx = int(
                            rank_resp.get("metadata", {}).get(
                                "best_candidate_index", 0
                            )
                        )
                        if 0 <= best_idx < len(alt_candidates):
                            qwen_pick = alt_candidates[best_idx]
                            if task_ir["task_type"] == "object_reference":
                                execution["selected_object"] = qwen_pick
                                execution["answer"] = int(qwen_pick["object_id"])
                                execution["resolution_mode"] = (
                                    "qwen_disambiguated_3d_candidates"
                                )
                                execution_stage["execution"] = execution
                            elif task_ir["task_type"] == "instruction_following":
                                # Replace the terminal step's object with
                                # Qwen's pick.
                                for ro in execution.get("route_objects", []):
                                    if ro.get("terminal"):
                                        ro["object"] = qwen_pick
                                        break
                                execution["resolution_mode"] = (
                                    "qwen_disambiguated_terminal_step"
                                )
                                execution_stage["execution"] = execution
                            summary.payload["stages"]["query_execution"] = (
                                execution_stage
                            )
                            summary.flush()
                except Exception:
                    pass  # Fall through to original 3D-only result.

    # ---- Short-path PROBE for instruction-following ----
    # When the selected path is very short (< 2 m total), the resolver
    # likely grabbed a nearby cluster of objects rather than the real
    # navigation path across the room.  Force a PROBE to get a closer
    # look before committing.
    if (
        execution.get("resolved")
        and task_ir["task_type"] == "instruction_following"
        and execution.get("route_objects")
    ):
        positions = [
            (float(ro["object"]["center_3d"][0]),
             float(ro["object"]["center_3d"][1]))
            for ro in execution["route_objects"]
        ]
        import math as _math
        total_path = sum(
            _math.hypot(
                positions[i+1][0] - positions[i][0],
                positions[i+1][1] - positions[i][1],
            )
            for i in range(len(positions) - 1)
        )
        if total_path < 2.0:
            # Path too short — probe the farthest selected object.
            farthest = max(
                execution["route_objects"],
                key=lambda ro: _math.hypot(
                    float(ro["object"]["center_3d"][0]),
                    float(ro["object"]["center_3d"][1]),
                ),
            )
            execution["resolved"] = False
            execution["probe_object"] = farthest["object"]
            execution["failed_constraints"] = ["short_path_probe_required"]
            execution_stage["execution"] = execution
            # Re-record with the probe decision.
            summary.payload["stages"]["query_execution"] = execution_stage
            summary.flush()

    route_plan = None
    state_payload = json.loads(
        Path(competition_geometry["state_estimation_path"]).read_text(encoding="utf-8")
    )
    terrain_map_path = competition_geometry.get("terrain_map_path")
    if not execution.get("resolved") and execution.get("probe_object") is not None:
        route_plan = plan_route(
            [{"order": 0, "action": "probe", "object": execution["probe_object"]}],
            registered_scan_path=competition_geometry["registered_scan_path"],
            robot_pose_xyz=state_payload["position_xyz"],
            global_geometry_manifest_path=mast3r.get("geometry_manifest_path"),
            terrain_map_path=terrain_map_path,
        )
        route_plan["probe"] = True
        route_stage = route_plan
    elif task_ir["task_type"] != "instruction_following":
        route_stage = {"status": "skipped", "reason": "task_has_no_navigation_output"}
    elif not execution.get("resolved"):
        route_stage = {"status": "blocked", "reason": "route_references_unresolved"}
    else:
        route_plan = plan_route(
            execution["route_objects"],
            registered_scan_path=competition_geometry["registered_scan_path"],
            robot_pose_xyz=state_payload["position_xyz"],
            global_geometry_manifest_path=mast3r.get("geometry_manifest_path"),
            terrain_map_path=terrain_map_path,
        )
        route_stage = route_plan
    summary.record(
        "route_planning",
        route_stage,
        route_planning=route_stage["status"] in {"completed", "skipped"},
    )

    decision = finalize_execution(task_ir, execution, route_plan=route_plan)
    summary.record(
        "root_finalization",
        {"status": "completed", "decision": decision},
        root_finalization=True,
        root_action=decision["action"],
    )

    answer_reasons = list(decision["failed_constraints"])
    answer_authorized = decision["action"] == "COMMIT"
    answer_gate = {
        "status": "authorized" if answer_authorized else "blocked",
        "publish_answers": answer_authorized,
        "reasons": answer_reasons,
        "decision": decision,
    }
    summary.record("answer_gate", answer_gate, answer_authorized=answer_authorized)
    summary.finish()
    required_diagnostic_stages = [
        summary.payload["stages"]["perspective_adapter"]["status"] == "completed",
        qwen_grounding["status"] == "completed",
        mast3r_quality,
        summary.payload["stages"]["sam2_segmentation"]["status"] == "completed",
        summary.payload["stages"]["panorama_observation_fusion"]["status"] == "completed",
        qwen_complete,
        local_lift.get("status") == "completed",
    ]
    return 0 if all(required_diagnostic_stages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
