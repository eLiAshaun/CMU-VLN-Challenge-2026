# Go2 compatibility test report

Date: 2026-09-18 UTC (2026-09-17 in New York).
Base: `post-competition-scene-memory` / `cdfeb52771084b79d816a871c4801788dba24196`.
Environment: Linux, Python 3.13.5, CPU only.

## Executed

- `PYTHONPATH=ai_module python3 -m unittest discover -s ai_module/go2/tests -v`: **32 passed**.
- `python3 -m compileall -q ai_module/go2 ai_module/rebuild/runtime.py`: passed.
- `bash -n ai_module/docker/start_go2.sh ai_module/docker/start_realsense_d435i.sh`: passed.
- JSON profile and Compose YAML parsing: passed.

The sensor and mapping tests exercise actual NumPy/OpenCV code. Nav2 asynchronous
lifecycle tests use injected fake action clients/goal handles. Runtime tests execute
the modified Runtime source with fake model, geometry and ObjectStore dependencies
to isolate adapter routing. They test the pinhole path, no-DA3 measured-depth mode,
and preservation of the default CMU path; they do not test real detector quality.

## Not executed / not claimed

No ROS DDS transport or ROS launch validation, no RealSense hardware, no Go2,
no CUDA/VLM inference, no Unity closed-loop regression, no Docker build/pull/push,
and no Jetson/ARM deployment were available in this environment.
The existing competition/research accuracy limitations remain unchanged.

A first CPU test used exact equality on floating-point timestamps and failed on
123.00000000000001 versus 123.0. It was corrected to numerical approximate equality;
the final 32-test run passed. The transport success/cancellation logic was also
covered by goal replacement and late-cancellation-response tests.

## Required deployment validation

Run the read-only `python3 -m go2.preflight` in the actual ROS environment first.
Then test with an operator present: a single Nav2 goal, a heading scan, an object
reference and an instruction task. Inspect actual TF trajectory, stop/cancel
behavior, image-time transforms, GPU peak and semantic outputs. Do not interpret
this report as real-robot acceptance or as a solution to semantic closure.
