# 真实场景扩展验证与最后回归

最终状态以 [`../../REBUILD_STATUS.md`](../../REBUILD_STATUS.md) 为准。最后候选为 `docker_ai_module:post-competition-root-repair-v7`，构建日志 `../../runs/rebuild/build_root_repair_v7.log`。用户要求停止子 agent、节省 token、尽快收尾，因此取消十二题扩展，保留已启动标准 Q1 及 `manifest_closure_v7.json` 的两道导航、一道指代。未执行的清单不能计入完成数或正确率。

最终四轮汇总保存在 `closure_summary_v7.json`（完成后生成），包含题目、run、实际输出、失败阶段、代理评分、导航步骤、资源与独立 ROS 接收记录。每个 label 的 `score.json`、`capture.json`、`report.md` 保留详细证据。原始输入和容器挂载记录在 `../../runs/rebuild/<run>/`。

该目录不挂载进 AI，参考答案只由主机侧评分器在回合后读取。评分器使用只读的 `challenge_evaluator/challenge_eval.py`：数值精确匹配，指代三维 AABB IoU × 2，导航依据独立接收到的实际 `/state_estimation` 轨迹计算公开参考代理分。内部完成、发出 waypoint 和私有官方得分是不同证据，不能相互替代。

原版十题完整结果见 `baseline_final_summary.json`：计数 1/4 正确、指代 0/3 输出、导航 0/3 完成，代理总分 1.797357/28。原进度文件中的早期评分导入错误已补评，最终应读取该汇总及逐题 score。原版构建为 `../../runs/rebuild/build_support_relation.log`；其历史镜像名称即便随后用于新的研究构建，也不能改变旧 run 的出处。

V5 只实际跑了三轮，见 `partial_v5_assessed.json`。hotel_room_1 导航内部完成却仅得 0.243252/6；livingroom_4 导航超时，得 0.918395/6。它们推动了当前图像类别核验、参考物集合距离排序、正例与覆盖分离、导航目标集合绑定等修改。

V1–V4 是未采纳的程序生成实验；捕获区域对照和 75 题语言诊断不能充当新场景验收。当前组合语言前端的 75/75 只证明解析、输出题型及非空检测词接口，不证明整套题目语义或运行正确率。

所有源码改动限于 ai_module；冻结提交、底盘/FAR、根 Docker 配置与评分器未改动。未做哈希校验或单元测试套件。随后用户授权发布研究版本，发布目标和额外复测见 `../../RELEASE_20260916.md`。16 GB Laptop、实机、私有场景和正式高分仍未验证。
