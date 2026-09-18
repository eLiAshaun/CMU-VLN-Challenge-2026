# Go2 兼容版

RealSense D435i RGB-D + 已有 Go2 Nav2 导航栈的研究适配。

部署入口：[`../GO2_COMPATIBILITY.md`](../GO2_COMPATIBILITY.md)。
测试记录与 CI 链接：[`TEST_REPORT.md`](TEST_REPORT.md)。

2026-09-18：53 项 CPU 回归、9 项跨进程 ROS Jazzy 接口测试通过；实际 Go2 Docker 镜像构建、镜像内导入及 53 项 CPU 测试通过；已安装 RealSense 驱动的启动参数检查通过。

ROS 测试使用合成传感器与测试 action 服务端，不运行真实导航控制器。没有 CUDA 推理、相机实测或 Go2 硬件验收；也没有推送新 Docker Hub 标签。

共用原重构模型、ObjectStore 和关系求解，不声称修复识别或语义闭合。新增完整/增量 costmap 接收、统一 RGB-D 时间检查和中间 waypoint 的轻量前向观察。
