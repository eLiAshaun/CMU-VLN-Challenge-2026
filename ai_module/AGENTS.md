# ai_module 项目限制

## 真实场景验收

- 对当前模型链路的正确性、效果、回归或可用性作出结论前，必须运行真实 `livingroom_3` 的 Q1 链路。
- 标准问题固定为：`How many photos are on the TV cabinet?`。优先使用已经接到当前模型链路的 `demo/run_real_question.sh q1`；若该脚本仍启动已迁出的旧链路，必须通过当前在线节点的 `/challenge_question` 入口发出同一问题，不得为了使用旧脚本而绕过当前实现。
- 必须从运行时容器、场景挂载或等价在线证据确认场景确为 `livingroom_3`，不得仅根据脚本默认值推定场景。
- 验收必须由真实 `livingroom_3` 环境经 `/challenge_question` 发题，并使用在线 ROS 输入，包括 `/camera/image`、`/state_estimation`、`/sensor_scan` 和 `/registered_scan`；结果必须来自本次在线运行新生成的 run 目录。
- 手工编辑请求 JSON、复制旧 run 产物、手工挑选关键帧、合成数据、单独调用模型以及离线/手工回放都只能用于诊断，不得表述为真实场景 Q1 测试或验收结果。
- 不得以 `Find the TV.`、其他问题、其他场景或预录全景替代 `livingroom_3/Q1` 验收。
- 每次报告真实 Q1 结果时，必须同时给出实际命令、场景、完整问题、run 目录、根决策、数值答案、失败阶段以及是否向 ROS 发布答案。
- 除非用户明确要求，不运行 pytest、单元测试或类似的 `N passed` 测试；这些测试也不得替代真实 `livingroom_3/Q1` 验收。
