#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
set -u
AI_ROOT="${CMU_AI_ROOT:-/home/docker/ai_module}"
export PYTHONPATH="$AI_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTDDS_BUILTIN_TRANSPORTS=UDPv4
cd "$AI_ROOT"
exec python3 -m rebuild.ros_node --config "${CMU_AI_CONFIG:-$AI_ROOT/configs/rebuild.json}"
