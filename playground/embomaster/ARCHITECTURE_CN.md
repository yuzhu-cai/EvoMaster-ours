# EmboMaster 架构说明（基于 EvoMaster）

## 1. 文档目标
本文档从高层和实现两条线说明 EmboMaster：

1. EmboMaster 如何复用 EvoMaster 现有基础能力。
2. EmboMaster 在数据、计算基建、工具调用三个层面的工作方式。
3. EmboMaster 如何“独立维护 workspace（尤其是数据存储）”。

---

## 2. 一句话定位
EmboMaster 是一个面向 ML 实验迭代的自动执行框架：  
它复用 EvoMaster 的 Agent/Tool/Session 基座，在上层增加多轮实验编排、K8S 执行和工作区隔离能力。

---

## 3. 总体架构

```text
run.py
  -> 自动注册并加载 embomaster playground
  -> BasePlayground 生命周期（setup/run/cleanup）
  -> EmboMasterPlayground（工具策略、双 Agent、K8S 服务）
  -> EmboMasterExp（多轮实验编排）
      -> coding agent 改代码
      -> debug_test 快速验证
      -> K8SExperimentRunner 提交/监控作业
      -> feedback agent 生成下一轮建议
      -> metric 驱动下一轮
```

---

## 4. EmboMaster 如何复用 EvoMaster

## 4.1 启动与注册能力（复用）
1. `run.py` 自动导入 `playground/*/core/playground.py`，触发注册。
2. EmboMaster 用 `@register_playground("embomaster")` 接入注册表。
3. `get_playground_class` 负责按名字实例化 playground。

## 4.2 Playground 生命周期（复用 + 扩展）
1. EmboMaster 继承 `BasePlayground`，保留标准生命周期：`setup -> run -> cleanup`。
2. 在 `setup()` 中扩展：
   - 注册自定义工具 `debug_test`
   - 应用工具白名单/禁用策略
   - 创建 `coding/feedback` 双 Agent
   - 创建并注入 `K8SExperimentRunner`

## 4.3 Agent 执行内核（复用）
1. 任务执行循环、tool-calling、finish 终止等机制仍使用 EvoMaster `Agent.run()`。
2. EmboMaster 不改内核协议，只通过 prompt 参数和工具集控制行为。
3. 轨迹记录继续复用 EvoMaster 的增量写入机制（`trajectory.jsonl`）。

## 4.4 Tool 与 Session（复用 + 增量）
1. 复用默认工具注册中心（editor/bash/finish/think）。
2. 新增 `debug_test`，但执行仍走统一 Session 抽象。
3. `debug_test` 支持两种路径：
   - 本地 session 执行
   - K8S debug pod 执行（失败可按配置回退本地）

## 4.5 Exp 编排层（EmboMaster 主要增量）
1. `EmboMasterExp` 继承 `BaseExp`，把“单次任务”扩展成“多轮迭代实验”。
2. 每轮典型流程：
   - 准备本轮 workspace
   - coding agent 改代码
   - 可选提交 K8S job 并抓取日志
   - 解析 metric
   - feedback agent 输出下一轮建议
3. 用 `best_metric` 与配置策略控制父 workspace 选择和停止条件。

---

## 5. High-Level 三层工作模型

## 5.1 数据层（Data Plane）
1. 输入数据：PRD/任务描述、历史反馈、日志、指标。
2. 状态数据：每轮 workspace、轨迹、实验结果。
3. 输出数据：变更后的代码、submission、metric 与反馈。
4. 关键机制：每轮隔离 + 指标回流，形成“数据驱动的持续迭代”。

## 5.2 计算基建层（Compute / Infra Plane）
1. 本地计算：快速编辑和短验证（低延迟）。
2. K8S 计算：标准化训练/评估（可调度、可观测）。
3. Debug Pod：在集群环境里做快速诊断，减少“本地可跑/线上失败”的偏差。
4. Manifest patch：运行前自动注入本轮 codebase、submission 与大目录挂载。

## 5.3 工具调用层（Tool Plane）
1. Agent 通过标准工具协议执行动作，而不是直接“隐式改代码”。
2. `debug_test` 把“代码修改”与“验证反馈”连接成闭环。
3. Tool policy 可配置，控制哪些工具可见、哪些工具禁用。

---

## 6. 独立维护 Workspace（重点：数据存储）

## 6.1 设计目标
1. 轮次隔离：每轮有独立代码副本，避免互相污染。
2. 存储可控：大数据目录不做全量复制，降低磁盘和 IO 成本。
3. 执行一致：本地与 K8S 都能看到统一目录结构。

## 6.2 目录形态
每轮会在 `session_dir/round_workspaces/{short_id}` 下生成独立 codebase：

```text
round_workspaces/{short_id}/
  codebase_{short_id}/
    ... code + small files copy ...
    submission/
    large_dirs.json
  codebase -> codebase_{short_id}   # 便捷 symlink
```

## 6.3 本轮 workspace 的准备逻辑
每轮运行前会构建 `workspace_context`，核心步骤如下：

1. 生成唯一 `workspace_id`。
2. 按策略选择父 workspace：
   - `previous`：基于上一轮
   - `best`：基于当前最好轮次
   - `none`：不继承
3. 尝试从父 workspace 复制（快路径）。
4. 如果没有父 workspace，则从 `source_codebase_dir` 初始化。
5. 清理 `eval_result`/`run_results` 等污染目录。
6. 创建本轮 `submission` 目录。

## 6.4 数据存储策略：小文件复制 + 大目录挂载
EmboMaster 把目录分成两类：

1. 小文件/小目录：直接复制到本轮 codebase。
2. 大目录（超阈值且命中关键词，如 `data/assets/ckpt/checkpoint`）：
   - 本轮 codebase 内只创建占位目录
   - 记录真实源路径到 `large_dirs.json`
   - 在 K8S 运行时通过 `hostPath` 挂载回容器

收益是“可隔离 + 低复制 + 保结构”。

## 6.5 Copy Plan 缓存机制
为避免每轮递归扫描全仓库，支持 copy plan 缓存：

1. 首次扫描生成 `.embomaster_copy_plan.json`。
2. 后续轮次命中缓存，直接复用 large-dir 规划。
3. 配置可强制重建缓存（用于目录结构大变化后）。

## 6.6 K8S 如何使用本轮 workspace 数据
提交 K8S Job 前会 patch manifest：

1. 挂载 `workspace_codebase_path` 到容器 codebase 目录。
2. 挂载本轮 `submission` 目录。
3. 将 `large_dirs.json` 中每个条目挂载到对应相对路径。

这样容器内路径连续，训练代码无需感知“这部分是复制、那部分是挂载”。

## 6.7 关键配置项（建议重点审查）
1. `workspace_isolation.enabled`
2. `workspace_isolation.session_dir`
3. `workspace_isolation.source_codebase_dir`
4. `workspace_isolation.parent_strategy`
5. `workspace_isolation.size_threshold_mb`
6. `workspace_isolation.copy_plan_cache_enabled`
7. `workspace_isolation.copy_plan_cache_file`
8. `workspace_isolation.copy_plan_rebuild`
9. `workspace_isolation.submission_subdir`
10. `k8s_runner.codebase_mount_path`
11. `k8s_runner.enable_submission_mount`
12. `k8s_runner.enable_large_dir_mounts`

---

## 7. 端到端执行视角（Round 级）
1. 读取任务与配置。
2. 构建当前轮 workspace。
3. coding agent 在当前轮 codebase 中修改代码。
4. `debug_test` 做快速验证（本地或 debug pod）。
5. 提交 K8S Job（可选）并采集日志。
6. 抽取 metric，更新 best round。
7. feedback agent 生成下一轮建议。
8. 到达轮次上限、时间上限或停止条件后结束。

---

## 8. 总结
EmboMaster 的核心不是替代 EvoMaster，而是“站在 EvoMaster 的标准执行内核上做实验工程化增强”：

1. 通过 workspace 隔离保证迭代可控。
2. 通过 K8S 执行保证实验可落地。
3. 通过工具调用和指标反馈保证闭环可持续优化。

