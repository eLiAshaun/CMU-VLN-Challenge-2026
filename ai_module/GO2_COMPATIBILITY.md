# Go2 兼容版：D435i RGB-D + Nav2

本分支：`go2-compatible`。基于 `post-competition-scene-memory` 的研究版提交
`cdfeb52771084b79d816a871c4801788dba24196`，不是旧的 `official-submission-20260916`。
默认相机选择 **RealSense D435i**，使用官方 ROS 2 驱动的彩色图、对齐彩色图的深度和 CameraInfo；
D455 等输出相同接口的型号也可使用，内参从设备读取，不使用固定 D435i 焦距。
这是一份**已实现、完成隔离 CPU 单元测试的硬件接口适配版本**，尚未经过 ROS/GPU/Go2 实机验证。

## 适配了什么

- 共用 `rebuild.Runtime`、GroundingDINO-Tiny、SAM2.1、Qwen3-VL-4B BF16、ObjectStore 和关系求解。
- 新增真正的 pinhole RGB-D 入口，不把前置 RGB 填充成全景，也不沿用 Unity 外参。
- 同步 RGB、aligned depth 和 color CameraInfo。支持 16UC1 毫米、32FC1 米、行跨度和大小端；
  彩色和对齐深度一起去畸变。按图像时刻的 `map <- camera_color_optical_frame` 完整 SE(3) 回投。
- 物体的 `measured_points` 来自 RealSense 实测深度，DA3 默认不加载、不调用。
  Go2 LiDAR 仍交给你已有的定位/避障/Nav2；本适配没有把 `/utlidar/cloud` 冒充 `/registered_scan`。
- 实际旋转获得六个朝向的观测。每次到达新的高层 waypoint 后再次扫描；先停稳，再取新图，
  不是把过去图像拼起来当作实时 360°。覆盖仍可能受遮挡限制，不代表整个候选集合完整。
- 语义 waypoint 通过 Nav2 `NavigateToPose` action 执行，处理接受、完成、拒绝、失败和取消。
  新任务、超时和关闭会取消未完成动作；语义步骤继续根据实际 TF 轨迹推进。
- 使用 Nav2 OccupancyGrid costmap 提供可行走信息，不将未知格子当作自由空间，不重复膨胀。
- 原 `rebuild.ros_node` 和 CMU 启动脚本不变。`Runtime` 新增可选观察/导航工厂参数，缺省仍走原链。

**这不是裸 Go2 的完整 SLAM/运动控制发行版。** 它接在已经能够接收 Nav2 目标的 Go2 导航系统上。
没有新增关节控制、SDK 私有命令编号或另一套 SLAM。初次上机必须保留操作员急停，先在空旷平地低速验证。
自动取消不是硬件急停，也不构成物理安全保证。

## 部署前提：三项必须真实存在

1. **已有 Go2 Nav2 栈**：`/navigate_to_pose` action 正常工作；底层 SDK/速度桥、运动控制和传感器避障由该栈负责。
2. **地图和时间**：`map -> odom -> base_link` 定位 TF，以及该相机实际安装的 `base_link -> camera_link`。
   RealSense 驱动提供相机内部光学 TF，但不知道相机装在狗背上哪里。不能用零位姿替代安装标定。
   多电脑的时钟也需要对齐；在同一 ROS 时钟体系中运行，不能混用仿真 `/clock` 和实际设备时钟。
3. **驱动和算力**：RealSense USB 驱动运行在采集主机；AI 可放在同机或有网络连接的 NVIDIA GPU 伴随电脑上。
   当前 Docker 基于研究发布的 **x86-64 / ROS 2 Jazzy** 镜像。同一块 GPU 不要同时启动 CMU AI 和 Go2 AI 两套模型进程。不​​要把它直接用于 Jetson ARM，
   也不要假设 Jazzy Nav2 action 能无配置地跨发行版连接 Humble。使用同一 ROS 发行版；
   ARM/JetPack 或 Humble 需要对应环境构建和单独验证。Go2 不同型号的 SDK 开放程度应按实际设备确认。

## 默认接口

| 信息 | 默认接口 |
|---|---|
| RGB | `/camera/camera/color/image_raw` |
| 对齐深度 | `/camera/camera/aligned_depth_to_color/image_raw` |
| 内参 | `/camera/camera/color/camera_info` |
| 位姿 | TF：`map`、`base_link`、图像实际 optical frame |
| 已膨胀成本地图 | `/global_costmap/costmap`，`nav_msgs/OccupancyGrid` |
| 英文任务 | `/go2_ai/task`，`std_msgs/String` |
| 导航动作 | `/navigate_to_pose` |
| 数值结果 | `/go2_ai/numerical_response` |
| 物体框 | `/go2_ai/selected_object_marker` |
| 状态/终态 | `/go2_ai/status`，JSON 字符串 |

这些参数在 `configs/go2_d435i.json` 中。base frame、camera namespace、costmap QoS 或 Nav2 action
名称与现场不同时，在该文件修改。Nav2 的目标位置/朝向容差应与本配置匹配：建议其 XY 容差
不超过 0.25 m、yaw 不超过 0.15 rad；AI 的中间点到达距离为 0.35 m。这里不是修改原厂安全配置的授权。

## 启动

在仓库根目录操作。先启动并独立验证你已有的 Go2 定位/导航，再启动 RealSense：

```bash
# 已装好 ROS Jazzy 与 realsense2_camera 的传感器主机
bash ai_module/docker/start_realsense_d435i.sh
```

脚本启用 `align_depth.enable` 和 `enable_sync`，默认 RGB/Depth 都为 640×480×30。
应按设备实际支持的 profile 调整；检查驱动输出分辨率，最终 K 来自 CameraInfo。
不要启用把深度转成彩色图的 colorizer。本版本不依赖相机 IMU；LIO 使用已有导航栈的配置。

构建 AI 适配镜像（大模型直接继承研究发布镜像，不从个人缓存下载）：

```bash
export CMU_AI_UID=$(id -u) CMU_AI_GID=$(id -g)
mkdir -p ai_module/runs/go2
export ROS_DOMAIN_ID=0  # 与同机/网络上的相机和 Nav2 设置一致

docker compose -f ai_module/docker/compose.go2.yml build
```

Dockerfile 使用 `elias1012/cmu-vln-2026:research-20260916-v7`，因为仓库发布记录说明该标签包含
这条新链。用户原来的 `official-20260916` 标签、`latest` 和正式分支都不被覆盖。
`Dockerfile.go2.dockerignore` 单独控制新镜像的构建上下文，避免重新传输本地 checkpoints。
**本次提交没有实际构建或推送 Go2 Docker 镜像；以上是可复现的构建入口。**

先做只读联通检查，不加载模型、不让狗移动：

```bash
docker compose -f ai_module/docker/compose.go2.yml run --rm \
  --entrypoint /bin/bash go2_ai -lc \
  'source /opt/ros/jazzy/setup.bash; python3 -m go2.preflight --timeout 30'
```

接口检查通过且操作员就位后启动 AI：

```bash
docker compose -f ai_module/docker/compose.go2.yml up

# 另一个终端，使用与你现场物体匹配的英文指令
ros2 topic pub --once /go2_ai/task std_msgs/msg/String \
  "{data: 'Take the path near the window to the fridge.'}"
ros2 topic echo /go2_ai/status
```

不使用 Docker、但已有匹配依赖和模型目录时，可在 `ai_module/` 中运行：

```bash
source /opt/ros/jazzy/setup.bash
python3 -m go2.ros_node --config configs/go2_d435i.json
```

启动不会自动发任务。收到任务后才会开始规划/转向。识别类 `Find ...` 输出物体真实包围盒，
**不会把包围盒中心直接当成可达地面终点**；需要实际行走请发 instruction 类型任务。
活动任务期间重复同一文本被忽略；终止后再发同一文本会开启新一轮。

## 验证范围与已知限制

测试命令：

```bash
PYTHONPATH=ai_module python3 -m unittest discover -s ai_module/go2/tests -v
python3 -m compileall -q ai_module/go2 ai_module/rebuild/runtime.py
bash -n ai_module/docker/start_go2.sh ai_module/docker/start_realsense_d435i.sh
```

CPU 测试覆盖米/毫米、字节序/行跨度、真实 K、SE(3)、无深度、去畸变、前置图像裁剪、
未知成本格、实际朝向序列、Nav2 异步取消/重发状态，以及共用 Runtime 的适配分支。
Nav2 action 测试使用模拟服务对象；Runtime 测试用模拟模型和存储隔离 GPU 依赖，
**不等于 ROS DDS 联通、真实 VLM 识别或实机闭环通过**。详情见 `go2/TEST_REPORT.md`。

原研究版识别、身份关联、关系闭合、计数和包围盒不稳定的问题仍然存在，这次不宣称修好。
数值任务沿用预算结束时输出当前集合计数的策略；状态和 summary 会同时记录 `evidence_complete`。
这个字段不是全房间候选集合完整性的证明。单次六向扫描也不等于全局探索完成。
按物体关系避让依旧通过中间目标和轨迹检查实现，Nav2 自己重新规划可能违反语义约束，仍须实测。

默认只用 D435i 有效实测深度；玻璃、反光、弱纹理或测距范围之外可能没有有效点。
本版本不把这些空洞填成可信几何。相机深度也有测量误差，不等于完整物体真实外形。
每个新任务仍然创建新的 ObjectStore，没有增加跨任务长期记忆、搬动家具处理或 SLAM 重定位修正。
四足全 SE(3) 感知不等于已经支持上下楼梯、多层规划或腿部落足规划。

## 依据

- 实际新链及发布记录：本仓库 `ai_module/README.md`、`RELEASE_20260916.md`。
- RealSense 官方 ROS 2 驱动：https://github.com/realsenseai/realsense-ros
- RealSense RGB-D 接口：https://dev.realsenseai.com/docs/ros2-wrapper/
- Nav2 Jazzy NavigateToPose：https://api.nav2.org/actions/jazzy/navigatetopose.html
- Unitree 官方 ROS 2 支持：https://github.com/unitreerobotics/unitree_ros2
