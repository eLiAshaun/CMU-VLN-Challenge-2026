# CMU AI module research rebuild

Continue development on `post-competition-scene-memory`, in [`rebuild/`](rebuild/). Configuration is [`configs/rebuild.json`](configs/rebuild.json); the sole entry is [`docker/start_live_chain.sh`](docker/start_live_chain.sh). The selected chain is **GroundingDINO-Tiny + SAM2.1, Qwen3-VL-4B BF16 without quantization, sensor geometry with DA3, and CPU ObjectStore**. It does not require SAM3 access.

Qwen remains GPU-resident in BF16. GroundingDINO-Tiny plus SAM2.1 and DA3 use mutually exclusive auxiliary GPU groups; each model is initialized once, and phase changes move the inactive group to CPU before returning the required group to the configured device. DINO runs one prompt per image batch. Completed RTX 5090 runs with this schedule measured an AI process peak around 11.2 GiB; see the report for each run's task result and resources.

The compositional English frontend owns TaskIR structure, relation direction and route order. Qwen translates only intrinsic object names into detector vocabulary and verifies current image evidence. It cannot rewrite the executable task. All 75 public questions produced a valid program, the expected output family and nonempty visual queries in the actual-model diagnostic; this does not establish semantic accuracy or scene success. BF16 is retained after the earlier quantized compiler experiments. The target 16 GB Laptop's timing and shared resource behavior remain unverified.

The default local research tag `docker_ai_module:post-competition-rebuild` now points to the V7 build tested in the final four rounds. Navigation improved, but the reference box scored only 0.074947/2; this is a development baseline, not evidence of high accuracy or real-robot readiness.

See [architecture](ARCHITECTURE_FINAL.md), [model preparation](MODEL_ASSETS.md), and the repository's [development and evidence report](REBUILD_STATUS.md). The report and test artifacts are not bundled into the runtime image.

From the repository root, after preparing the four model directories:

```bash
docker build -f ai_module/docker/Dockerfile -t docker_ai_module:post-competition-rebuild ai_module
export CMU_AI_UID=$(id -u) CMU_AI_GID=$(id -g)
export SCNAV_UNITY_SCENE_DIR=/home/robot/cmu_vln/Unity_environment_models/livingroom_3/environment
docker-compose -p cmu_rebuild -f docker/compose_gpu_headless.yml -f docker/compose_scene_override.yml -f ai_module/docker/compose.rebuild.yml up -d --no-build
docker exec cmu_rebuild_ai bash -c 'source /opt/ros/jazzy/setup.bash; python3 -m rebuild.publish_question "How many photos are on the TV cabinet?"'
```

Publish promptly after startup: the first question's 600-second budget includes cold startup. Follow the official per-question container restart protocol for subsequent cases. Repeated identical question text preserves the current episode. Confirm the actual system scene mount in every acceptance report.

The existing system-side bridge forwards the official Pose2D output to FAR. The AI does not publish FAR's internal goal topic or modify its planner. Local semantic arrival distances and the remaining-time reserve are explicit in the runtime configuration; they are not claims about the private evaluator.

All model loads are local and offline. The image directly imports installed SAM2.1 and DA3 packages and does not search historical source folders or personal caches. `.dockerignore` restricts build input to the active source/configuration and four selected asset directories. The inherited base remains `elias1012/cmu-vln-2026:latest`.

The supplied extrinsic calibration is the prior simulation calibration. A real robot needs its own calibration via `CMU_AI_CALIBRATION`; the AI does not subscribe to TF. Observations use image-time pose interpolation and registered points already in map coordinates.

Run records are under `runs/rebuild/<episode>/`. `python3 -m rebuild.output_probe` independently records the actual ROS messages and trajectory. `python3 -m rebuild.resource_monitor --output runs/rebuild/resources.jsonl --duration 600` samples whole-board/process GPU usage and host memory. Current development hardware is RTX 5090 32 GB; its results do not establish 16 GB Laptop acceptance.

AI Fast DDS uses UDPv4 explicitly. The inspected environment could discover ROS topics with shared-memory transport but did not deliver data across containers; UDP delivered the real image/scan/odometry streams. No extra input topic is introduced.

The frozen `submission/official`, recovery snapshots, root `docker/`, official system and other repositories are outside this development change.

The superseded MASt3R/DUSt3R, old detector services, old orchestration and their demo/training/build entrypoints have been removed from this checkout. The derived AI image also removes the base image's old AI checkout before copying this implementation. Historical reports, recorded data and local model assets remain available on the host; they are not executable alternatives to `rebuild/`. Git history retains the deleted tracked source.

Perception commits one complete observation at a time. ROS decisions use a detached CPU scene after geometry and visual checks finish. Final outputs, object snapshots and the last consumed image stamp refer to that same scene; a pending observation cannot change a published answer. Crop artifacts receive immutable filenames so later updates cannot replace earlier evidence.

Detector proposals are verified per current image region before association and fusion. `VerifiedObservation` separates that evidence from raw detections. Unconfirmed regions may guide another observation, but cannot modify confirmed object geometry or become an answer/action target. This replaces the former per-track category cache that could let a new bad proposal inherit an old positive label.

Category crops retain context across artificial perspective edges by sampling the same original panorama. The verifier names a complete physical artifact without the task category, then matches its visual evidence to that category. Historical measured points projected into the current image can preserve identity when monocular position drifts; conflicting current geometry is recorded separately from identity and is not fused into the track.

Observed geometric envelopes are independent of sampled point storage. They preserve previously seen surfaces, while the query layer keeps incomplete entity extent uncertain. The current assumptions are static scene objects and the official map/pose convention; the local Unity results do not establish operation under real localization resets, moving furniture or target-laptop timing.

Research publication and the flattened V7 runtime image are documented in [the 2026-09-16 release record](RELEASE_20260916.md). The release image retains the tested filesystem, models, environment and ROS entrypoint, while discarding hidden deleted lower-layer files.
