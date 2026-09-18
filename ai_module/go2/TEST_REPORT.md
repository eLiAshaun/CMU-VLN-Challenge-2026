# Go2 compatibility validation — 2026-09-18

## Current result

**53 CPU tests passed, 9 cross-process ROS Jazzy interface tests passed, and the actual Go2 deployment Docker image built successfully.** The 53 CPU tests also passed inside that deployment image. RealSense ROS launch arguments were expanded with the installed driver and all 14 arguments used by the supplied camera script were present.

These are software/interface results, not learned-perception accuracy or Go2 hardware acceptance.

Evidence:

- ROS/CPU workflow: https://github.com/eLiAshaun/CMU-VLN-Challenge-2026/actions/runs/35405512410
- Deployment image workflow: https://github.com/eLiAshaun/CMU-VLN-Challenge-2026/actions/runs/35405588095
- ROS-tested implementation: `b22493ea65a3eabc5655049f89e362f37fcbcfcc`.
- Image-tested snapshot: `5e9c08cabe97e4cb40ed726c5b99cb4f6b68865a` (same AI code, adds image validation workflow).
- Original research base: `post-competition-scene-memory` / `cdfeb52771084b79d816a871c4801788dba24196`.

## Executed checks

| Check | Environment | Result and scope |
|---|---|---|
| CPU regression suite | Local Linux/Python 3.13; Ubuntu 24.04/ROS Jazzy test container | 53 passed |
| Python compilation and real ROS imports | Installed ROS Jazzy dependencies | Passed |
| Shell syntax, Compose parsing, production Dockerfile build checks | GitHub Actions Ubuntu 24.04 | Passed |
| DDS RGB/depth/CameraInfo reception, image-time TF and projection | Separate sensor-fixture and AI processes using real ROS | Passed |
| Full costmap plus OccupancyGridUpdate changes the AI terrain | Actual ROS message publication/subscription | Passed |
| NavigateToPose success, abort, cancellation and replacement | Actual ROS action client/server protocol | Passed |
| Stale image handling and preflight positive/old-depth-negative cases | Actual ROS publishers and a separate preflight process | Passed |
| Published research image pull | `elias1012/cmu-vln-2026:research-20260916-v7` | Passed |
| Actual `Dockerfile.go2` build | Published research base plus this branch's code/dependencies | Passed |
| CPU regression and imports inside deployment image | amd64; PyTorch 2.11.0+cu128; CUDA device unavailable | 53 passed; imports passed |
| RealSense driver installation and supplied script `--show-args` | ROS Jazzy, no camera attached | Passed; 14 configured argument names found |

The built Go2 image was 24,582,282,861 bytes uncompressed according to Docker inspect. This is not its network download size. It was built as a temporary CI image; **no new Docker Hub tag was pushed** and no existing registry tag was changed.

## What the tests actually exercise

The CPU suite includes the previous 36 isolated tests plus 12 costmap/timestamp tests, three real geometry/ObjectStore regressions, and two observation-scheduling tests.

The three geometry/ObjectStore tests execute the real RGB-D projection, `lift_detection` and `ObjectStore`, using synthetic depth and explicitly supplied object labels/masks. They check separate same-category instances, repeated-view deduplication, and identity preservation after camera translation. They do not run GroundingDINO, SAM2.1 or Qwen, and cannot establish recognition accuracy.

The nine ROS tests are in `tests/ros/test_live_interfaces.py`. Their fixture publishes synthetic RGB-D, calibration, TF and costmaps from another process. Navigation uses the real `nav2_msgs/NavigateToPose` interface with an intentionally simple action server. That server acknowledges, completes, aborts or cancels goals; it does **not** execute a Nav2 planner/controller or move a robot. Communication stays in an isolated localhost ROS domain, so this is not a cross-computer wireless-network test.

The ROS node and read-only preflight use their actual implementations. Model loading is lazy and no language task is issued to the real model worker in these interface tests.

## Problems fixed in this iteration

- Receive and apply incremental Nav2 costmap rectangles instead of retaining only the initial full map. Recompute terrain using a map revision counter. Unknown cells stay unknown/blocked.
- Use the same RGB/depth/CameraInfo timing contract in preflight and capture. Fresh RGB cannot hide stale or unsynchronized depth.
- Take a new forward view at intermediate route bends; retain full heading sweeps at initial positions and route endpoints. This reduces redundant full-sweep calls by construction; navigation latency and semantic accuracy have not been measured on hardware.

## Test-development failures and remaining warnings

The first expanded ROS run passed eight tests but failed the ninth because its new preflight DDS participant had only two seconds to discover publishers. The negative test was corrected to allow discovery and to require that all three streams were actually received before asserting stale-depth rejection. The final nine-test run passed. The fixture also now avoids calling `rclpy.shutdown()` twice after SIGTERM.

The ROS test log includes repeated logger-name warnings while sequential test cases recreate nodes with the same name. The passing run is not claimed to be warning-free.

## Not executed / not claimed

- CUDA model inference, GPU peak VRAM or target-device latency.
- Physical D435i/D455 acquisition, mounting calibration or actual Go2 movement.
- A real Nav2 planner/controller or Unity closed-loop mission with learned perception.
- Cross-host DDS/wireless behavior, physical emergency stopping or signal-to-hardware stopping.
- Jetson/ARM, Humble or other deployment variants.
- Improved object recognition, semantic closure, bounding-box IoU, instruction success or long-term memory.

The camera launch check confirms driver/package/argument availability only. A real device still needs supported stream profiles, USB access, correct installation TF and a working Go2 localization/navigation stack.

## Reproduction

```bash
PYTHONPATH=ai_module python3 -m unittest discover -s ai_module/go2/tests -v
python3 -m compileall -q ai_module/go2 ai_module/rebuild
bash -n ai_module/docker/start_go2.sh ai_module/docker/start_realsense_d435i.sh

# In the ROS Jazzy test environment, isolated from physical robots:
source /opt/ros/jazzy/setup.bash
ROS_DOMAIN_ID=197 ROS_LOCALHOST_ONLY=1 PYTHONPATH=ai_module \
  python3 ai_module/go2/tests/ros/test_live_interfaces.py -v

docker build -f ai_module/docker/Dockerfile.go2 -t cmu-vln-go2:compatible ai_module
```

The repository's two Go2 validation workflows contain the complete environment setup and commands. Test logs and source snapshots are attached to the linked workflow runs.
