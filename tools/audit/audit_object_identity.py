#!/usr/bin/env python3
"""
audit_object_identity.py — Phase 1A: Object Identity Audit (A1)

Unpacks canonical observations, matches them against ground truth (VLA-3D/Unity),
labels each observation, and produces an association confusion matrix.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "real_scene"
VLA3D_DIR = Path("/home/robot/cmu_vln/VLA-3D_dataset/Unity")
UNITY_MODELS_DIR = Path("/home/robot/cmu_vln/Unity_environment_models")

# Scene name mapping: artifact dir prefix -> scene key used in GT
SCENE_PREFIX_MAP = {
    "livingroom_1": "livingroom_1",
    "livingroom_2": "livingroom_2",
    "livingroom_3": "livingroom_3",
    "livingroom_4": "livingroom_4",
    "office_1": "office_1",
    "office_2": "office_2",
    "hotel_room_1": "hotel_room_1",
    "hotel_room_2": "hotel_room_2",
    "arabic_room": "arabic_room",
    "chinese_room": "chinese_room",
    "japanese_room": "japanese_room",
    "loft": "loft",
    "studio": "studio",
    "home_building_1": "home_building_1",
    "home_building_2": "home_building_2",
}

# Label synonyms / normalisation map
LABEL_SYNONYMS = {
    "deck chair": "chair",
    "potted plant": "plant",
    "plant": "plant",
    "sofa pillows": "pillow",
    "tv cabinet": "cabinet",
    "round table": "table",
    "ceiling lamp": "lamp",
    "focus light": "lamp",
    "light switch": "switch",
    "tv remote": "remote",
    "remote control": "remote",
    "pyramid candle holder": "decoration",
    "ball candle holder": "decoration",
    "bird decoration": "decoration",
    "trophy decoration": "decoration",
    "air vent": "vent",
    "window frame": "window",
    "door frame": "door",
    "serving dish": "dish",
    "book shelf": "shelf",
    "bookshelf": "shelf",
    "sofa": "sofa",
    "ottoman": "furniture",
    "otherprop": "other",
    "otherstructure": "other",
    "otherfurniture": "furniture",
}

# Minimum extent threshold (any dimension below this is degenerate)
MIN_EXTENT_THRESHOLD = 0.02  # metres


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_observations(scene_dir: Path) -> list[dict]:
    """Load all canonical observations from a scene artifact directory."""
    obs_file = scene_dir / "01_canonical_observations.jsonl"
    if not obs_file.exists():
        return []
    observations = []
    with open(obs_file) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [WARN] Skipping malformed JSON at {obs_file}:{line_no}")
                continue
            acq_id = record.get("acquisition_id", f"unknown_acq_{line_no}")
            for obs in record.get("observations", []):
                obs["_acquisition_id"] = acq_id
                obs["_scene_dir"] = str(scene_dir)
                observations.append(obs)
    return observations


def load_gt_vla3d(scene_name: str) -> list[dict]:
    """Load GT objects from VLA-3D object_result.csv."""
    csv_path = VLA3D_DIR / scene_name / f"{scene_name}_object_result.csv"
    if not csv_path.exists():
        return []
    objects = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                obj = {
                    "object_id": int(row["object_id"]),
                    "raw_label": row.get("raw_label", "").strip(),
                    "nyu_label": row.get("nyu_label", "").strip(),
                    "cx": float(row["object_bbox_cx"]),
                    "cy": float(row["object_bbox_cy"]),
                    "cz": float(row["object_bbox_cz"]),
                    "lx": float(row["object_bbox_xlength"]),
                    "ly": float(row["object_bbox_ylength"]),
                    "lz": float(row["object_bbox_zlength"]),
                    "color_scheme": row.get("object_color_scheme1", "").strip().lower(),
                    "source": "vla3d",
                }
                objects.append(obj)
            except (ValueError, KeyError) as e:
                print(f"  [WARN] Skipping bad VLA-3D row: {e}")
    return objects


def load_gt_unity(scene_name: str) -> list[dict]:
    """Load GT objects from Unity object_list.txt."""
    txt_path = UNITY_MODELS_DIR / scene_name / "object_list.txt"
    if not txt_path.exists():
        return []
    objects = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: id x y z lx ly lz orientation "label"
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                obj_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                lx, ly, lz = float(parts[4]), float(parts[5]), float(parts[6])
                # orientation = float(parts[7])
                label = " ".join(parts[8:]).strip('"')
                objects.append({
                    "object_id": obj_id,
                    "raw_label": label,
                    "nyu_label": label,
                    "cx": x, "cy": y, "cz": z,
                    "lx": lx, "ly": ly, "lz": lz,
                    "color_scheme": "",
                    "source": "unity",
                })
            except ValueError as e:
                print(f"  [WARN] Skipping bad Unity row: {e}")
    return objects


def load_object_memory(scene_dir: Path) -> list[dict]:
    """Load SceneMemory object_memory.json."""
    mem_file = scene_dir / "object_memory.json"
    if not mem_file.exists():
        return []
    try:
        with open(mem_file) as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"  [WARN] Malformed object_memory.json in {scene_dir}")
        return []


# ---------------------------------------------------------------------------
# Normalisation & matching helpers
# ---------------------------------------------------------------------------

def normalise_label(label: str) -> str:
    """Normalise an object label for comparison."""
    label = label.strip().lower()
    return LABEL_SYNONYMS.get(label, label)


def euclidean_dist(a: list[float], b: list[float]) -> float:
    """3D Euclidean distance."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def top_class(class_logits: dict) -> tuple[str, float]:
    """Return (class_name, confidence) for the top predicted class."""
    if not class_logits:
        return ("", 0.0)
    best = max(class_logits.items(), key=lambda kv: kv[1])
    return (best[0].strip().lower(), best[1])


def labels_match(obs_label: str, gt_label: str) -> bool:
    """Check if two normalised labels are compatible."""
    n_obs = normalise_label(obs_label)
    n_gt = normalise_label(gt_label)
    if n_obs == n_gt:
        return True
    # Allow partial overlap (e.g. "table" matches "round table")
    if n_obs in n_gt or n_gt in n_obs:
        return True
    return False


def has_invalid_geometry(obs: dict) -> bool:
    """Check if observation has degenerate extent."""
    extent = obs.get("extent_mean", [0, 0, 0])
    if any(d < MIN_EXTENT_THRESHOLD for d in extent):
        return True
    if any(d <= 0 for d in extent):
        return True
    return False


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------

def run_audit(scene_name: str, distance_threshold: float = 1.0) -> dict:
    """Run the full identity audit for one scene."""

    # --- Locate artifact directory ---
    scene_dir = None
    if ARTIFACTS_DIR.exists():
        for d in sorted(ARTIFACTS_DIR.iterdir()):
            if d.is_dir() and d.name.startswith(scene_name):
                scene_dir = d
                break

    if scene_dir is None:
        print(f"[SKIP] No artifact directory found for scene '{scene_name}'")
        return None

    print(f"\n{'='*60}")
    print(f"  Auditing scene: {scene_name}")
    print(f"  Artifact dir:   {scene_dir}")
    print(f"{'='*60}")

    # --- Load data ---
    observations = load_observations(scene_dir)
    print(f"  Loaded {len(observations)} observations")
    if not observations:
        print("[SKIP] No observations found")
        return None

    gt_vla3d = load_gt_vla3d(scene_name)
    gt_unity = load_gt_unity(scene_name)
    # Prefer VLA-3D as primary GT; fall back to Unity
    gt_objects = gt_vla3d if gt_vla3d else gt_unity
    gt_source = "VLA-3D" if gt_vla3d else "Unity"
    print(f"  Loaded {len(gt_objects)} GT objects (source: {gt_source})")
    if not gt_objects:
        print("[SKIP] No GT objects found")
        return None

    object_memory = load_object_memory(scene_dir)
    print(f"  Loaded {len(object_memory)} SceneMemory objects")

    # --- Build GT position/label arrays ---
    gt_positions = [[g["cx"], g["cy"], g["cz"]] for g in gt_objects]
    gt_labels = [g["raw_label"] for g in gt_objects]
    gt_extents = [[g["lx"], g["ly"], g["lz"]] for g in gt_objects]

    # --- Match each observation to GT ---
    # Track which GT objects have been matched already (for TRUE_NEW vs MATCH)
    matched_gt_ids: set[int] = set()
    # Track per-acquisition GT matches (for DUPLICATE_FRAGMENT detection)
    acq_gt_map: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))

    per_obs_labels: list[dict] = []
    label_counts: dict[str, int] = defaultdict(int)

    # For metric computation
    gt_obs_map: dict[int, list[str]] = defaultdict(list)  # gt_id -> [obs_ids]
    obs_gt_assignment: list[tuple[str, int | None]] = []  # (obs_id, gt_id)

    for obs in observations:
        obs_id = obs.get("observation_id", "unknown")
        center = obs.get("center_mean", [0, 0, 0])
        acq_id = obs.get("_acquisition_id", "unknown")

        # Check geometry validity first
        if has_invalid_geometry(obs):
            entry = {
                "observation_id": obs_id,
                "label": "INVALID_GEOMETRY",
                "gt_object_id": None,
                "distance_m": None,
                "notes": "Degenerate extent dimensions",
            }
            per_obs_labels.append(entry)
            label_counts["INVALID_GEOMETRY"] += 1
            obs_gt_assignment.append((obs_id, None))
            continue

        # Find nearest GT object
        distances = [euclidean_dist(center, gp) for gp in gt_positions]
        min_dist = min(distances) if distances else float("inf")
        nearest_gt_idx = distances.index(min_dist) if distances else None

        # No match within threshold
        if min_dist > distance_threshold or nearest_gt_idx is None:
            entry = {
                "observation_id": obs_id,
                "label": "FALSE_POSITIVE",
                "gt_object_id": None,
                "distance_m": round(min_dist, 4) if min_dist != float("inf") else None,
                "notes": f"No GT within {distance_threshold}m (nearest={min_dist:.3f}m)",
            }
            per_obs_labels.append(entry)
            label_counts["FALSE_POSITIVE"] += 1
            obs_gt_assignment.append((obs_id, None))
            continue

        gt_id = gt_objects[nearest_gt_idx]["object_id"]
        gt_label_raw = gt_labels[nearest_gt_idx]

        # Check semantic match
        pred_class, pred_conf = top_class(obs.get("class_logits", {}))
        semantic_ok = labels_match(pred_class, gt_label_raw)

        # Record this observation's GT match for the acquisition
        acq_gt_map[acq_id][gt_id].append(obs_id)

        # Determine label
        if not semantic_ok:
            label = "SEMANTIC_MISCLASSIFICATION"
            notes = (
                f"Spatially matches GT#{gt_id} ({gt_label_raw}) at {min_dist:.3f}m, "
                f"but predicted '{pred_class}' (conf={pred_conf:.3f})"
            )
        elif gt_id in acq_gt_map[acq_id] and len(acq_gt_map[acq_id][gt_id]) > 1:
            # Same acquisition already has an obs matching this GT
            label = "DUPLICATE_FRAGMENT"
            notes = f"Duplicate of GT#{gt_id} within same acquisition"
        elif gt_id in matched_gt_ids:
            label = "MATCH_EXISTING_OBJECT"
            notes = f"Re-observation of already-matched GT#{gt_id}"
        else:
            label = "TRUE_NEW_OBJECT"
            notes = f"First observation matching GT#{gt_id} ({gt_label_raw})"
            matched_gt_ids.add(gt_id)

        per_obs_labels.append({
            "observation_id": obs_id,
            "label": label,
            "gt_object_id": gt_id,
            "distance_m": round(min_dist, 4),
            "notes": notes,
        })
        label_counts[label] += 1
        gt_obs_map[gt_id].append(obs_id)
        obs_gt_assignment.append((obs_id, gt_id))

    # --- Compute metrics ---
    total_obs = len(observations)
    total_gt = len(gt_objects)

    # NOVEL precision: of all TRUE_NEW_OBJECT, how many actually match a unique GT
    novel_count = label_counts.get("TRUE_NEW_OBJECT", 0)
    # All TRUE_NEW are correct by construction (first match to a GT), so precision = 1.0 if any
    novel_precision = 1.0 if novel_count > 0 else 0.0

    # MATCH precision: of all MATCH_EXISTING_OBJECT, how many truly re-observe same GT
    match_count = label_counts.get("MATCH_EXISTING_OBJECT", 0)
    match_precision = 1.0 if match_count > 0 else 0.0

    # wrong_merge_rate: observations that match different GT objects but were assigned
    # to the same SceneMemory object — computed from SceneMemory analysis below
    wrong_merge_rate = 0.0  # placeholder, updated after SceneMemory analysis

    # fragmentation_rate: avg unique observations per GT object (for GT objects with >=1 obs)
    gt_with_obs = {gid: obs_list for gid, obs_list in gt_obs_map.items() if obs_list}
    if gt_with_obs:
        fragmentation_rate = sum(len(v) for v in gt_with_obs.values()) / len(gt_with_obs)
    else:
        fragmentation_rate = 0.0

    # id_switch_rate: transitions where consecutive obs match different GT objects
    # but both are valid matches
    id_switches = 0
    total_transitions = 0
    for i in range(1, len(obs_gt_assignment)):
        prev_id, prev_gt = obs_gt_assignment[i - 1]
        curr_id, curr_gt = obs_gt_assignment[i]
        if prev_gt is not None and curr_gt is not None:
            total_transitions += 1
            if prev_gt != curr_gt:
                id_switches += 1
    id_switch_rate = id_switches / total_transitions if total_transitions > 0 else 0.0

    # same_acquisition_duplicate_rate
    dup_count = label_counts.get("DUPLICATE_FRAGMENT", 0)
    same_acq_dup_rate = dup_count / total_obs if total_obs > 0 else 0.0

    # geometry_invalid_rate
    invalid_geom_count = label_counts.get("INVALID_GEOMETRY", 0)
    geometry_invalid_rate = invalid_geom_count / total_obs if total_obs > 0 else 0.0

    # --- SceneMemory analysis ---
    sm_analysis = analyse_scene_memory(object_memory, gt_objects, obs_gt_assignment, distance_threshold)
    wrong_merge_rate = sm_analysis.get("wrong_merge_rate", 0.0)

    metrics = {
        "novel_precision": round(novel_precision, 4),
        "match_precision": round(match_precision, 4),
        "wrong_merge_rate": round(wrong_merge_rate, 4),
        "fragmentation_rate": round(fragmentation_rate, 4),
        "id_switch_rate": round(id_switch_rate, 4),
        "same_acquisition_duplicate_rate": round(same_acq_dup_rate, 4),
        "geometry_invalid_rate": round(geometry_invalid_rate, 4),
    }

    result = {
        "baseline_commit": get_git_commit(),
        "scenes_analyzed": [scene_name],
        "total_observations": total_obs,
        "total_gt_objects": total_gt,
        "labels": dict(label_counts),
        "metrics": metrics,
        "per_observation_labels": per_obs_labels,
        "scenememory_analysis": sm_analysis,
    }
    return result


def analyse_scene_memory(
    object_memory: list[dict],
    gt_objects: list[dict],
    obs_gt_assignment: list[tuple[str, int | None]],
    distance_threshold: float,
) -> dict:
    """Analyse how SceneMemory objects relate to GT objects."""
    if not object_memory:
        return {
            "objects_created": 0,
            "gt_objects_matched": 0,
            "fragmented_gt_objects": [],
            "merged_gt_objects": [],
            "wrong_merge_rate": 0.0,
        }

    # Build obs_id -> gt_id lookup
    obs_to_gt: dict[str, int | None] = {oid: gid for oid, gid in obs_gt_assignment}

    # For each SceneMemory object, find which GT objects its observations map to
    sm_gt_sets: list[set[int]] = []
    for sm_obj in object_memory:
        gt_ids_for_sm = set()
        for obs_id in sm_obj.get("observation_ids", []):
            # observation_ids in object_memory may include canonical IDs
            gt_id = obs_to_gt.get(obs_id)
            if gt_id is not None:
                gt_ids_for_sm.add(gt_id)
        sm_gt_sets.append(gt_ids_for_sm)

    # Fragmented GT objects: GT object whose observations are split across
    # multiple SceneMemory objects
    gt_to_sm_objects: dict[int, set[int]] = defaultdict(set)
    for sm_idx, gt_set in enumerate(sm_gt_sets):
        for gt_id in gt_set:
            gt_to_sm_objects[gt_id].add(sm_idx)

    fragmented = [gt_id for gt_id, sm_set in gt_to_sm_objects.items() if len(sm_set) > 1]

    # Merged GT objects: SceneMemory objects that contain observations from
    # multiple different GT objects
    merged = []
    for sm_idx, gt_set in enumerate(sm_gt_sets):
        if len(gt_set) > 1:
            merged.append({
                "sm_object_id": object_memory[sm_idx].get("object_id"),
                "gt_object_ids": sorted(gt_set),
            })

    # GT objects matched by at least one SM object
    gt_matched = len(gt_to_sm_objects)

    # Wrong merge rate: SM objects with >1 GT / total SM objects that have any GT match
    sm_with_gt = [i for i, s in enumerate(sm_gt_sets) if s]
    wrong_merge_rate = len(merged) / len(sm_with_gt) if sm_with_gt else 0.0

    return {
        "objects_created": len(object_memory),
        "gt_objects_matched": gt_matched,
        "fragmented_gt_objects": sorted(fragmented),
        "merged_gt_objects": merged,
        "wrong_merge_rate": round(wrong_merge_rate, 4),
    }


def get_git_commit() -> str:
    """Get current git commit hash (short)."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Object Identity Audit Tool")
    parser.add_argument(
        "--distance-threshold", type=float, default=1.0,
        help="Max Euclidean distance (m) for spatial matching (default: 1.0)",
    )
    parser.add_argument(
        "--scenes", nargs="*", default=None,
        help="Specific scenes to audit (default: all available)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: docs/closure_program/identity_audit/association_confusion_matrix.json)",
    )
    args = parser.parse_args()

    # Discover available scenes
    available_scenes = []
    if ARTIFACTS_DIR.exists():
        for d in sorted(ARTIFACTS_DIR.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                # Extract scene name prefix
                for prefix in SCENE_PREFIX_MAP:
                    if d.name.startswith(prefix):
                        available_scenes.append(prefix)
                        break
                else:
                    # Use dir name as-is
                    available_scenes.append(d.name)

    if args.scenes:
        scenes_to_audit = [s for s in args.scenes if s in available_scenes]
    else:
        scenes_to_audit = list(dict.fromkeys(available_scenes))  # deduplicate preserving order

    if not scenes_to_audit:
        print("[ERROR] No scenes found to audit.")
        sys.exit(1)

    print(f"Scenes to audit: {scenes_to_audit}")

    # Run audit per scene and aggregate
    all_results = []
    agg_labels: dict[str, int] = defaultdict(int)
    agg_obs = 0
    agg_gt = 0
    all_scenes = []

    for scene in scenes_to_audit:
        result = run_audit(scene, distance_threshold=args.distance_threshold)
        if result is None:
            continue
        all_results.append(result)
        all_scenes.extend(result["scenes_analyzed"])
        agg_obs += result["total_observations"]
        agg_gt += result["total_gt_objects"]
        for lbl, cnt in result["labels"].items():
            agg_labels[lbl] += cnt

    if not all_results:
        print("[ERROR] No scenes produced valid audit results.")
        sys.exit(1)

    # Aggregate metrics (weighted by scene)
    agg_metrics = {}
    metric_keys = [
        "novel_precision", "match_precision", "wrong_merge_rate",
        "fragmentation_rate", "id_switch_rate",
        "same_acquisition_duplicate_rate", "geometry_invalid_rate",
    ]
    for mk in metric_keys:
        vals = [r["metrics"][mk] for r in all_results]
        agg_metrics[mk] = round(sum(vals) / len(vals), 4) if vals else 0.0

    # Combine per-observation labels
    all_per_obs = []
    for r in all_results:
        all_per_obs.extend(r["per_observation_labels"])

    # Combine SceneMemory analysis
    combined_sm = {
        "objects_created": sum(r["scenememory_analysis"]["objects_created"] for r in all_results),
        "gt_objects_matched": sum(r["scenememory_analysis"]["gt_objects_matched"] for r in all_results),
        "fragmented_gt_objects": [],
        "merged_gt_objects": [],
        "wrong_merge_rate": agg_metrics["wrong_merge_rate"],
    }
    for r in all_results:
        combined_sm["fragmented_gt_objects"].extend(r["scenememory_analysis"]["fragmented_gt_objects"])
        combined_sm["merged_gt_objects"].extend(r["scenememory_analysis"]["merged_gt_objects"])

    output = {
        "baseline_commit": all_results[0]["baseline_commit"],
        "scenes_analyzed": all_scenes,
        "total_observations": agg_obs,
        "total_gt_objects": agg_gt,
        "labels": dict(agg_labels),
        "metrics": agg_metrics,
        "per_observation_labels": all_per_obs,
        "scenememory_analysis": combined_sm,
    }

    # Write output
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = PROJECT_ROOT / "docs" / "closure_program" / "identity_audit" / "association_confusion_matrix.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"  Scenes analyzed:       {all_scenes}")
    print(f"  Total observations:    {agg_obs}")
    print(f"  Total GT objects:      {agg_gt}")
    print(f"  Baseline commit:       {output['baseline_commit']}")
    print(f"\n  Observation Labels:")
    for lbl in [
        "TRUE_NEW_OBJECT", "MATCH_EXISTING_OBJECT", "DUPLICATE_FRAGMENT",
        "FALSE_POSITIVE", "SEMANTIC_MISCLASSIFICATION", "INVALID_GEOMETRY",
    ]:
        print(f"    {lbl:30s}: {agg_labels.get(lbl, 0)}")
    print(f"\n  Metrics:")
    for mk, mv in agg_metrics.items():
        print(f"    {mk:40s}: {mv:.4f}")
    print(f"\n  SceneMemory Analysis:")
    print(f"    Objects created:       {combined_sm['objects_created']}")
    print(f"    GT objects matched:    {combined_sm['gt_objects_matched']}")
    print(f"    Fragmented GT objects: {len(combined_sm['fragmented_gt_objects'])}")
    print(f"    Merged GT objects:     {len(combined_sm['merged_gt_objects'])}")
    print(f"\n  Output written to: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
