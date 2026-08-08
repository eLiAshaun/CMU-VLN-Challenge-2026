# Reusable model assets

This MASt3R-centered `ai_module` contains the live perception, persistent
object identity, semantic execution, navigation lifecycle, and official ROS
output adapters. The former SC-NAV runtime is not part of this chain.

## Layout

- `third_party/Grounded-SAM-2/`: GroundingDINO and SAM2 source without nested
  Git metadata.
- `third_party/YOLO-World/`: YOLO-World and its MMYOLO source without nested
  Git metadata.
- `integrations/perception/`: file-protocol GroundingDINO/SAM2 and YOLO-World
  workers. Each worker discovers the vendored source tree by default.
- `integrations/qwen3vl/`: the validated candidate-verification worker reduced
  to its independent dependency closure; it no longer imports `scnav_vln`.
- `integrations/execution/`: persistent map-frame object IDs, relation and
  count/unique execution, registered-scan standoff planning, the sole root
  decision authority, and the ROS-free navigation lifecycle.
- `integrations/ros/`: live competition acquisition plus validated adapters for
  `/numerical_response`, `/selected_object_marker`,
  `/way_point_with_heading`, and `/stop`.
- `checkpoints/`: local model weights and offline text encoders.
- `vendor/wheels/`: cached MMEngine/MMDetection/MMCV and related offline wheels.
- `configs/model_assets.json`: canonical relative paths and source revisions.

Weights are deliberately ignored by Git. Source, integration code, configs,
licenses, and source-revision metadata remain visible to the parent repository.

## Qwen3-VL worker

From the `ai_module` directory:

```bash
python3 -m integrations.qwen3vl.worker_server \
  --checkpoint checkpoints/qwen3vl/Qwen3-VL-8B-Instruct \
  --endpoint @mast3r_qwen3vl \
  --quantization int8 \
  --max-pixels 1048576
```

Probe a running worker:

```bash
python3 -m integrations.qwen3vl.worker_probe \
  --endpoint @mast3r_qwen3vl \
  --timeout 120
```

The worker exposes query-conditioned `ground_objects` in addition to candidate
and relation verification. Grounding first reads the complete panorama for
scene context, then follows the TaskIR-derived helper-context -> anchor -> target
order on higher-resolution perspectives. It uses Qwen's native integer
`bbox_2d` coordinates in the 0..1000 range. SAM2 refines those boxes and MASt3R
remains responsible for physical instance identity and map-frame geometry.
After a navigation probe, consistent anchor-conditioned target boxes across
adjacent views are the primary numerical evidence for support-relation questions;
the direct local-number prompt remains diagnostic only.

## Perception workers

`yolo_world_worker.py`, `groundingdino_worker.py`, and `sam2_box_worker.py`
consume a JSON request via `--request`. GroundingDINO is box-only so both
detectors feed the same SAM2 stage. Model paths should come from
`configs/model_assets.json`; callers must write outputs outside the source and
checkpoint directories.

Qwen3-VL is the active query-conditioned proposal backend. YOLO-World and
GroundingDINO remain installed but disabled. SAM2 turns Qwen boxes into masks.
Start its persistent service with:

```bash
python3 -m integrations.perception.sam2_worker_server \
  --checkpoint checkpoints/sam2/sam2.1_hiera_base_plus.pt \
  --config configs/sam2.1/sam2.1_hiera_b+.yaml \
  --endpoint @mast3r_sam2
```

After Qwen produces boxes, SAM segmentation and Qwen candidate verification
run concurrently. Their 2D outputs remain observations; MASt3R geometry owns
cross-view and cross-station physical instance association.

## Live ordered chain

`tools/live_model_chain.py` enforces the stage order in
`configs/model_assets.json`:

1. compile the question to TaskIR with DeepSeek or the local parser;
2. preserve the raw equirectangular panorama and give it directly to Qwen for
   global semantic context;
3. generate 1024x768 Qwen views and matching 512x384 MASt3R/SAM views from one
   optical centre, with identical view IDs and projection parameters;
4. run query-conditioned Qwen grounding, then native MASt3R SGA over the
   cross-station pair graph;
5. keep YOLO-World and GroundingDINO disabled; run SAM2 from Qwen boxes and
   Qwen candidate verification concurrently;
6. reproject masks to the panorama and remove overlapping-slice duplicates;
7. for numerical `ON(target, anchor)` tasks, expand the support ROI over the
   complete support surface and count native Qwen target boxes in each adjacent
   anchor-conditioned view;
8. combine Qwen-verified masks, MASt3R pointmaps, `/sensor_scan`,
   `/registered_scan`, calibrated camera-LiDAR extrinsics, and
   `/state_estimation` into map-frame 3D observations;
9. associate map-frame observations into the session world model; tentative
   instances are usable and closure remains uncertainty metadata;
10. execute relations, count/object selection, and ordered route references;
11. navigate toward the lifted anchor before authorizing a numerical answer,
   accept the official navigation stack's reachable projected stop after a
   real motion baseline, and reacquire the same question from the new station;
12. let the root finalizer emit `COMMIT`, `PROBE`, `SAFE_REJECT`, or
   `SYSTEM_FAILURE`;
13. publish the official ROS output for `COMMIT`; after every
    probe or instruction waypoint, require a fresh full-chain acquisition
    before advancing.

The perspective cuts share one optical centre. They improve model input and
allow panorama deduplication, but never count as independent spatial evidence.
Every stage records `completed`, `skipped`, or `blocked` in `summary.json`; a
stage cannot be bypassed without raising a stage-order violation.

The current configuration deliberately enables `semantic_execution.bringup_mode`.
In this mode, a positive anchor-local numerical inventory may publish without
MASt3R world alignment, camera-LiDAR synchronization, persistent-object
confirmation, or a second viewpoint. Those stages still execute when inputs
are available and remain visible as diagnostics, but they are not numerical
answer gates. The stricter policy can be restored after the complete task
chain works end to end; it must not be reintroduced as scattered implicit
checks.

Map-frame geometry is explicit: `local_object_lift.py` keeps the original
MASt3R centre as `local_centroid_xyz`, while `center_3d`, `centroid_xyz`, and
`bbox_3d` are computed after the state-estimation/extrinsic transform.  A
single panorama and its perspective cuts can create only tentative objects.

`/camera/depth` is intentionally neither subscribed nor required. It is not an
allowed evaluation input. The competition geometry gate accepts only the live
RGB, LiDAR, and pose topics listed by the organizer.

## Licensing boundary

The copied upstream trees retain their own license files. In particular,
YOLO-World is GPL-3.0; review distribution obligations before publishing a
combined deliverable.
