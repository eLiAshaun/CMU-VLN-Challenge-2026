#!/usr/bin/env bash
set -e
AI_ROOT="${GO2_AI_ROOT:-/home/docker/ai_module}"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
source "/opt/ros/$ROS_DISTRO/setup.bash"
export PYTHONPATH="$AI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$AI_ROOT"
exec python3 -m go2.ros_node --config "${GO2_AI_CONFIG:-$AI_ROOT/configs/go2_d435i.json}" "$@"
