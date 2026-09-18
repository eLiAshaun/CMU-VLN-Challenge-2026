#!/usr/bin/env bash
# Run on the robot/companion host, not inside the AI container.
set -e
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
exec ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=camera camera_name:=camera \
    enable_color:=true enable_depth:=true \
    enable_sync:=true align_depth.enable:=true \
    rgb_camera.color_profile:=640x480x30 depth_module.depth_profile:=640x480x30 \
    rgb_camera.color_format:=RGB8 depth_module.depth_format:=Z16 \
    colorizer.enable:=false hole_filling_filter.enable:=false \
    temporal_filter.enable:=false pointcloud.enable:=false "$@"
