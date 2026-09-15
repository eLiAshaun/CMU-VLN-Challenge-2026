#!/usr/bin/env bash
# system_simulation_headless.sh — Headless simulation startup.
#
# This is a thin wrapper over the official simulation flow.  The official
# autonomy_stack_mecanum_wheel_platform/system_simulation.sh does:
#   ./Model.x86_64 & sleep 3 ros2 launch ... rviz
#
# Headless differences:
#   - Virtual display (Xvfb by default; NVIDIA Xorg when HEADLESS_BACKEND=nvidia-xorg)
#   - No RViz
#   - Unity log written to /tmp/unity_player.log
#
# The ROS-TCP endpoint is started before Unity.  The official Unity-first
# sequence can fill Unity's outgoing image queue before a receiver exists,
# producing repeated "Encoding error: not enough capacity" failures.
set -eo pipefail

SCRIPT_DIR="/home/docker/autonomy_stack_mecanum_wheel_platform"
DISPLAY_ID="${HEADLESS_DISPLAY:-:99}"
WIDTH="${HEADLESS_WIDTH:-1280}"
HEIGHT="${HEADLESS_HEIGHT:-720}"
DEPTH="${HEADLESS_DEPTH:-24}"
UNITY_DELAY="${UNITY_STARTUP_DELAY:-5}"
HEADLESS_BACKEND="${HEADLESS_BACKEND:-xvfb}"

export DISPLAY="$DISPLAY_ID"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-docker}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export COLCON_TRACE="${COLCON_TRACE:-}"

mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

DISPLAY_PID=""
UNITY_PID=""
ROS_PID=""
PLANNER_BRIDGE_PID=""
GLOBAL_PLANNER_BACKEND="${SCNAV_GLOBAL_PLANNER_BACKEND:-far}"

# FAR is the only supported planner for the CMU-VLN runtime.  Reject an
# explicit legacy backend instead of silently launching the base-only stack.
case "$GLOBAL_PLANNER_BACKEND" in
  far|far_planner) ;;
  *)
    echo "FAR is mandatory: unsupported SCNAV_GLOBAL_PLANNER_BACKEND=$GLOBAL_PLANNER_BACKEND" >&2
    exit 2
    ;;
esac

cleanup() {
  if [ -n "$ROS_PID" ]; then
    kill "$ROS_PID" 2>/dev/null || true
  fi
  if [ -n "$PLANNER_BRIDGE_PID" ]; then
    kill "$PLANNER_BRIDGE_PID" 2>/dev/null || true
  fi
  if [ -n "$UNITY_PID" ]; then
    kill "$UNITY_PID" 2>/dev/null || true
  fi
  if [ -n "$DISPLAY_PID" ]; then
    kill "$DISPLAY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

start_xvfb() {
  echo "Starting Xvfb on $DISPLAY_ID (${WIDTH}x${HEIGHT})"
  # Docker can leave the X socket/lock behind after an ungraceful restart.
  # With no live server those files make Xvfb exit immediately, after which
  # Unity starts against a 0x0 display and never publishes camera frames.
  local display_number="${DISPLAY_ID#:}"
  rm -f "/tmp/.X11-unix/X${display_number}" "/tmp/.X${display_number}-lock"
  Xvfb "$DISPLAY_ID" \
    -screen 0 "${WIDTH}x${HEIGHT}x${DEPTH}" \
    -ac +extension GLX +render -noreset \
    > /tmp/xvfb.log 2>&1 &
  DISPLAY_PID="$!"
  sleep 2
  if ! kill -0 "$DISPLAY_PID" 2>/dev/null || \
     ! DISPLAY="$DISPLAY_ID" glxinfo -B \
       > /tmp/glxinfo_xvfb.log 2>&1; then
    echo "Xvfb failed to provide a usable GLX display on $DISPLAY_ID." >&2
    cat /tmp/xvfb.log >&2 || true
    cat /tmp/glxinfo_xvfb.log >&2 || true
    return 1
  fi
  echo "Using Xvfb headless renderer"
  grep -E "OpenGL vendor|OpenGL renderer" /tmp/glxinfo_xvfb.log \
    || true
}

start_nvidia_xorg() {
  local XORG_CONF="${HEADLESS_XORG_CONF:-/etc/X11/xorg_nvidia_headless.conf}"
  local privilege=()
  if [ "$(id -u)" != "0" ]; then
    if ! command -v sudo >/dev/null 2>&1 || ! sudo -n true; then
      echo "NVIDIA Xorg requires passwordless sudo; falling back to Xvfb." >&2
      return 1
    fi
    privilege=(sudo -n)
  fi
  if ! command -v Xorg >/dev/null 2>&1 || [ ! -f "$XORG_CONF" ]; then
    echo "Xorg or $XORG_CONF missing; falling back to Xvfb." >&2
    return 1
  fi

  export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
  unset LIBGL_ALWAYS_SOFTWARE

  local display_number="${DISPLAY_ID#:}"
  "${privilege[@]}" rm -f \
    "/tmp/.X11-unix/X${display_number}" "/tmp/.X${display_number}-lock"

  "${privilege[@]}" Xorg "$DISPLAY_ID" \
    -config "$XORG_CONF" \
    -noreset \
    +extension GLX +extension RANDR +extension RENDER \
    -logfile /tmp/xorg_nvidia.log \
    -nolisten tcp -ac -novtswitch -sharevts \
    > /tmp/xorg_nvidia_stdout.log 2>&1 &
  DISPLAY_PID="$!"
  sleep 5

  if ! kill -0 "$DISPLAY_PID" 2>/dev/null || \
     ! glxinfo -B > /tmp/glxinfo_headless.log 2>&1 || \
     ! grep -q "OpenGL renderer string: NVIDIA" /tmp/glxinfo_headless.log; then
    echo "NVIDIA Xorg check failed; falling back to Xvfb." >&2
    kill "$DISPLAY_PID" 2>/dev/null || true
    # Xorg can retain its UNIX socket briefly after SIGTERM. Starting Xvfb
    # before the server has actually exited makes Xvfb fail with "server
    # already running", after which Unity attaches to a nonexistent display
    # and publishes no camera or scan data.
    wait "$DISPLAY_PID" 2>/dev/null || true
    DISPLAY_PID=""
    return 1
  fi

  echo "Using NVIDIA Xorg headless renderer"
  grep -E "OpenGL vendor|OpenGL renderer" /tmp/glxinfo_headless.log
  return 0
}

if ! pgrep -f "(Xorg|Xvfb) $DISPLAY_ID" >/dev/null 2>&1; then
  if [ "$HEADLESS_BACKEND" = "nvidia-xorg" ]; then
    start_nvidia_xorg || start_xvfb
  else
    start_xvfb
  fi
fi

# ---------------------------------------------------------------------------
# Robust startup flow:
#   1. ROS2 endpoint  2. startup grace  3. Unity
# ---------------------------------------------------------------------------

cd "$SCRIPT_DIR"
source ./install/setup.bash

# 1. Start ROS2 first so the TCP endpoint is listening before Unity publishes.
case "$GLOBAL_PLANNER_BACKEND" in
  far|far_planner)
    GLOBAL_PLANNER_BACKEND="far"
    python3 /usr/local/bin/waypoint_far_bridge.py &
    PLANNER_BRIDGE_PID="$!"
    ros2 launch vehicle_simulator system_simulation_with_route_planner.launch &
    ;;
  *)
    echo "Unsupported SCNAV_GLOBAL_PLANNER_BACKEND=$GLOBAL_PLANNER_BACKEND" >&2
    exit 2
    ;;
esac
ROS_PID="$!"
echo "Semantic global planner backend: $GLOBAL_PLANNER_BACKEND"

# 2. Give the launch graph and TCP endpoint a bounded startup grace period.
sleep "$UNITY_DELAY"

# 3. Start Unity only after a receiver exists.
./src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64 \
  -screen-width "$WIDTH" \
  -screen-height "$HEIGHT" \
  -screen-fullscreen 0 \
  -force-glcore \
  -logFile /tmp/unity_player.log \
  > /tmp/unity_headless.log 2>&1 &
UNITY_PID="$!"

wait "$ROS_PID"
