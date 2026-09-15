# 研究版收尾记录（2026-09-15—16）

开发目录 `/home/robot/cmu_vln/CMU-VLN-Challenge-2026`，分支 `post-competition-scene-memory`。有效源码为 `ai_module/rebuild/`，架构以 `ARCHITECTURE_FINAL.md` 为准。SAM3 无权限，当前链路使用 GroundingDINO-Tiny、SAM2.1 base+、Qwen3-VL-4B BF16、注册 LiDAR、按需 DA3METRIC-LARGE 与 CPU ObjectStore。

## 结论与验证范围

**尚不能宣称三种题型高正确率或多场景稳定。** 用户要求节省 token 并尽快收尾，已停止所有子 agent，把最后一版验证收缩为四轮：标准计数、两道暴露问题的导航、一道指代。扩大前的原版十题结果为计数 1/4 正确、指代 0/3 输出、导航 0/3 完成，本地代理总分 1.797357/28；此前少量成功回合不足以代表泛化表现。

最终候选镜像为 `docker_ai_module:post-competition-root-repair-v7`，构建日志 `runs/rebuild/build_root_repair_v7.log`。下列四轮使用同一候选镜像，每题独立重启 system/AI，实际场景由容器挂载确认。AI 只挂载 runs，评分器和参考答案在主机侧，未作为模型输入。三轮后续清单在 `evaluation/expanded_20260915/manifest_closure_v7.json`。

<!-- CLOSURE_RESULTS_BEGIN -->
| 场景 / 题目 | Run | 根决策 | 代理分 | 耗时 | ROS / 失败阶段 |
|---|---|---|---|---|---|
| livingroom_3 / q1 | `20260915T154946_360563Z` | `numerical_output` | 1.000000/1 | 530.069 s | 已发布；`null` |
| hotel_room_1 / q4 | `20260915T155843_636901Z` | `instruction_trajectory_complete` | 3.694356/6 | 67.041 s | 已发布；`null` |
| livingroom_4 / q4 | `20260915T160013_833587Z` | `instruction_trajectory_complete` | 2.345752/6 | 57.234 s | 已发布；`null` |
| hotel_room_1 / q2 | `20260915T160134_080627Z` | `object_reference_output` | 0.074947/2 | 530.035 s | 已发布；`null` |

标准 Q1 答案 **2**，与公开答案一致；`failure_stage=null`，`/numerical_response` 已发布，独立订阅收到 2。两道导航均完成两个有序步骤；hotel_room_1 终点误差 0.557 m，livingroom_4 为 0.538 m。相较 V5 的 0.243252/6 与 0.918395/6，分别提高到 3.694356/6 与 2.345752/6；仍不能视为高分。

指代已发布 Marker，但仅 0.074947/2，IoU=0.037474，**指代质量未达标**。四轮全部在 600 s 内产生规定的运行终态，失败阶段均为 `null`；这仅证明执行闭环，不表示四题全部答对。完整字段和独立 ROS 接收记录见 `evaluation/expanded_20260915/closure_summary_v7.json`。

四轮完整问题：
- livingroom_3/q1: How many photos are on the TV cabinet?
- hotel_room_1/q4: Go to the bedside table closest to the window and stop at the chair closest to the TV.
- livingroom_4/q4: Go near the chair closest to the bookcase and stop at the table with the flowers on it.
- hotel_room_1/q2: Find the bedside table farthest from the window.

<!-- CLOSURE_RESULTS_END -->

导航分数来自只读的 `challenge_evaluator/challenge_eval.py`，输入为独立 ROS 订阅收到的实际 `/state_estimation` 轨迹；不是 commanded waypoints，也不是私有官方分。指代采用三维 AABB IoU × 2；数值采用整数精确匹配。内部 `instruction_trajectory_complete` 只说明本地运动判据满足，不能代替评分。

## 标准 Q1 复现与证据

实际场景 `livingroom_3`；完整问题 `How many photos are on the TV cabinet?`；run `runs/rebuild/20260915T154946_360563Z/`。

实际发题命令：

```bash
docker exec cmu_rebuild_ai /bin/bash -c "source /opt/ros/jazzy/setup.bash; python3 -m rebuild.publish_question 'How many photos are on the TV cabinet?'"
```

实际场景挂载 `/home/robot/cmu_vln/Unity_environment_models/livingroom_3/environment` → `/home/docker/autonomy_stack_mecanum_wheel_platform/src/base_autonomy/vehicle_simulator/mesh/unity/environment`，详见 run 内 `container_provenance.json`。本次使用在线相机、图像时刻位姿、传感器与注册扫描，由 `/challenge_question` 入口接受问题。根决策、数值、失败阶段、ROS 发布与独立接收均以最终表格及该 run 的 `summary.json`、独立 probe 为准。

## 已落地的根因修改

- **任务结构与模型自由生成解耦**：组合式英语前端解释关系方向、修饰语归属、比较和有序动作；Qwen 只翻译开放词汇类别为检测词，不能改任务目标或步骤。AST 唯一决定输出类型。75 道公开题的实际模型诊断中，结构解析、输出题型、非空视觉查询均 75/75；不代表 75 道语义或场景答对。
- **类别核验依赖当前图像**：先独立识别完整实物，再看同一图像判断所求类别，替换文本标签之间的子类推断。305 个捕获区域的对照中，错误的 desk→bedside table、console→bookcase 正例被拒绝，同时保留真实对应家具；此为诊断，不代替现场正确率。
- **查询可执行性与覆盖完整性分开**：单调关系查询的已证实正例可驱动动作；比较、排除与计数仍保留完整性信息。比较使用对象到观测参考物集合的最近距离，避免多个锚点导致无动作。BETWEEN 两个角色、返回对象和扁平候选接口统一。
- **几何和身份分开维护**：每像素只取最近一次注册扫描返回；历史表面重投影维持身份，冲突的新测量不进入旧物体；同视角分离的掩膜不因粗体素相交合并；有界点集保留轴向极值。实测与单目估计范围分别保存，估计不确定性不能扩大到达区域。
- **空间关系依据物理证据**：ON 使用观测支撑表面加独立图像关系；ABOVE/BELOW 同时检查垂直位移与横向分离，不能只凭高度。
- **导航绑定与重规划统一**：从已证实目标集合选可行接近路径，目标身份、路线与完成条件一同绑定；失效或停滞后重规划，保留已完成步骤。普通目标交给原 FAR，局部不完整地形不再否定全局可达性；显式避让仍使用语义区域。进度取实际位姿。
- **全链路 deadline**：600 秒包含冷启动、编译、感知和导航。超时记录真实阶段，隔离迟到模型结果，已成功终态不会被超时覆盖。运行中的 GPU 调用尚不能强制中断。

## 资源和未解决边界

<!-- CLOSURE_RESOURCES_BEGIN -->
四轮 AI 进程显存采样峰值 **11436 MiB（11.17 GiB）**；整卡峰值 12703 MiB（含其他进程）；CUDA 分配峰值 10701.8 MiB；AI host RSS 峰值 6667.5 MiB。按每个 run 的实际 AI PID 和时间窗口筛选；资源原始记录为 `runs/rebuild/resources_expanded_validation.jsonl`，未把其他进程显存算作 AI 进程用量。
<!-- CLOSURE_RESOURCES_END -->

计划中的 AI 约 11–12 GB 是设计目标，当前仅有 RTX 5090 32 GB 实测；不等于 16 GB Laptop 的算力、共享资源或冷拉镜像验收。默认 600 秒总预算、70 秒尾部预留；数值与指代通常约 530 秒输出。

仍需明确保留的问题：开放语言的歧义修饰语；真实凳子被模型解释为边桌的类别混淆；未知参考物和未观测候选；局部可见包围盒与完整实体差异；复杂避让、跨房间关系、动态物体、定位重置；目标笔记本与真实机器人闭环。当前实机 rosbag 缺少完整的相机、注册点、位姿和标定输入，不能宣称实机回放通过。没有增加场景/题目/答案/坐标词典，也没有用类别别名掩盖失败。

本次指代的具体缺口：最终选择 `object_000003`，只有 1 次观测、0 个实测点、512 个估计点，输出范围约 0.399 × 0.367 × 0.292 m。同一区域还保留了更完整的实测候选；最终排序中，实测表面间距和估计位置间距参与同一比较，局部身份与几何不一致仍可能影响胜者。当前修复未解决这一问题。不能用提高置信门槛、类别别名或手工改答案把本轮失败改写为通过。

## 保留证据与清理

原版十题：`evaluation/expanded_20260915/baseline_final_summary.json`。V5 三轮：`partial_v5_assessed.json`；其中 hotel_room_1 导航内部完成但仅 0.243252/6，livingroom_4 导航超时且仅 0.918395/6，均没有当作成功。失败的 V1–V4 自由或约束生成程序实验已移出活动源码；捕获诊断只作为历史证据。

已移除 1860 个确定被替代的跟踪文件，包括旧 MASt3R/DUSt3R、检测服务、编排和旧训练/演示/构建入口。镜像清空基底旧 AI 目录，仅复制当前源码、配置和选定资产。冻结提交、回退源码、根 docker、底盘/FAR、评分器、驱动和其他仓库未修改。没有执行哈希校验、pytest 或单元测试套件；只做必要源码检查、模型诊断及真实 Unity/ROS 检验。验证收尾时尚未创建提交、合并、推送或外部比赛提交；随后用户已授权发布研究版本，见 `RELEASE_20260916.md`。

<!-- CLOSURE_STATE_BEGIN -->
最后四轮已结束，测试容器 `cmu_rebuild_ai`、`cmu_rebuild_system` 和本次资源采样进程均已停止；全部子 agent 已停止。默认研究标签 `docker_ai_module:post-competition-rebuild` 已更新为完成这四轮的 V7 构建，作为后续开发基线，**不代表竞赛高分版本或实机验收通过**。验证收尾时源码、必要评估摘要及文档已在本地暂存；随后用户已授权上传，发布信息见 `RELEASE_20260916.md`。
<!-- CLOSURE_STATE_END -->
