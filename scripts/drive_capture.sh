#!/bin/bash
# 移动机器人并采集校准帧：每次前进 ~0.8m（8s@0.5m/s），停 4s 供同步采集，共 6 次
source /opt/ros/jazzy/setup.bash
OUT="$1"
SCENE="$2"
rm -rf "$OUT"; mkdir -p "$OUT"
python3 /tmp/capture_unity_calibration_frames.py --output "$OUT" --scene-id "$SCENE" &
CAP_PID=$!
sleep 6
for i in 1 2 3 4 5 6; do
  echo "--- move $i: 前进 8s ---"
  timeout 9 ros2 topic pub /cmd_vel geometry_msgs/msg/TwistStamped "{twist: {linear: {x: 0.5}}}" -r 20 >/dev/null 2>&1 &
  PUB_PID=$!
  sleep 8
  kill $PUB_PID 2>/dev/null
  timeout 2 ros2 topic pub /cmd_vel geometry_msgs/msg/TwistStamped "{twist: {linear: {x: 0.0}}}" -r 20 >/dev/null 2>&1 &
  SPID=$!
  sleep 4
  kill $SPID 2>/dev/null
done
sleep 3
kill $CAP_PID 2>/dev/null
sleep 2
echo "=== 采集结果 ==="
ls "$OUT" | head -30
python3 -c "import json; m=json.load(open('$OUT/frames_manifest.json')); print('帧数:', len(m['frames']))"
