FROM zhangjicmu/ubuntu24_ros:cmu_vla_challenge_simulation

USER root

# Headless display: Xvfb (software fallback) + NVIDIA Xorg (GPU accelerated).
# The NVIDIA X11 driver module is mounted at runtime via the compose volume:
#   /usr/lib/x86_64-linux-gnu/nvidia:/usr/lib/x86_64-linux-gnu/nvidia:ro
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      xserver-xorg-core \
      xvfb \
      xauth \
      x11-xserver-utils \
      mesa-utils \
      libgl1 \
      libgl1-mesa-dri \
      libglx-mesa0 && \
    rm -rf /var/lib/apt/lists/*

# Create mount-point for host NVIDIA Xorg driver module.
# At runtime mount /usr/lib/x86_64-linux-gnu/nvidia from the host.
RUN mkdir -p /usr/lib/x86_64-linux-gnu/nvidia

RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

COPY xorg_nvidia_headless.conf /etc/X11/xorg_nvidia_headless.conf
COPY system_simulation_headless.sh /usr/local/bin/system_simulation_headless.sh
COPY waypoint_far_bridge.py /usr/local/bin/waypoint_far_bridge.py
COPY ros_tcp_endpoint_multitype.patch /tmp/ros_tcp_endpoint_multitype.patch
RUN echo "093024feb53c937d07b382c7c0d613c22715372513c58452df19ab564b239278  /tmp/ros_tcp_endpoint_multitype.patch" \
      | sha256sum -c - && \
    patch -d /home/docker/autonomy_stack_mecanum_wheel_platform/src/utilities/ROS-TCP-Endpoint \
      -p1 < /tmp/ros_tcp_endpoint_multitype.patch && \
    rm /tmp/ros_tcp_endpoint_multitype.patch
RUN chmod +x \
  /usr/local/bin/system_simulation_headless.sh \
  /usr/local/bin/waypoint_far_bridge.py

# Official sample bags publish livox_ros_driver2/msg/CustomMsg. The simulation
# base image contains pinned source but not the SDK or installed ROS 2 type.
RUN cmake \
      -S /home/docker/autonomy_stack_mecanum_wheel_platform/src/utilities/livox_ros_driver2/Livox-SDK2 \
      -B /tmp/livox-sdk2-build \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/livox-sdk2-build --parallel 2 && \
    cmake --install /tmp/livox-sdk2-build && \
    ldconfig && \
    rm -rf /tmp/livox-sdk2-build
RUN /bin/bash -lc \
      "source /opt/ros/jazzy/setup.bash && \
       cd /home/docker/autonomy_stack_mecanum_wheel_platform && \
       colcon build --packages-select livox_ros_driver2 \
         --cmake-args -DROS_EDITION=ROS2"

# Fix: Unity Mono looks for libdl.so in its own directory but only
# libdl.so.2 exists system-wide. Without this symlink, Unity throws
# DllNotFoundException which prevents ROS-TCP-Connector scripts from
# loading properly.
RUN ln -sf /usr/lib/x86_64-linux-gnu/libdl.so.2 \
  /home/docker/autonomy_stack_mecanum_wheel_platform/src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model_Data/MonoBleedingEdge/x86_64/libdl.so

USER docker
