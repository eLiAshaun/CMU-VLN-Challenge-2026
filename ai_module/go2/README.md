# Go2 兼容版

RealSense D435i RGB-D + 已有 Go2 Nav2 导航栈的研究适配。

部署入口：[`../GO2_COMPATIBILITY.md`](../GO2_COMPATIBILITY.md)。
测试记录：[`TEST_REPORT.md`](TEST_REPORT.md)。

36 项隔离 CPU 单元测试通过；ROS、GPU、Docker 和 Go2 实机尚未验证。
共用原重构模型、ObjectStore 和关系求解，不声称修复识别或语义闭合。
