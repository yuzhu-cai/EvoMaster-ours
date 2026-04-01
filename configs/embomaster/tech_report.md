# EmboMaster 技术报告

## 1. 文档目的

本文档面向论文写作，总结当前代码仓库中 `embomaster` 的真实实现结构、模块职责、执行链路与实验工程机制。文档以当前仓库 `/data/zixing/code/EvoMaster-ours` 中的实现为准，重点回答以下问题：

1. EmboMaster 在 EvoMaster 框架中的位置是什么。
2. 它如何把大模型代码代理、工作区隔离、Kubernetes 训练评测和多轮反馈闭环组合成统一系统。
3. 当前仓库支持哪些任务族、配置方式和实验产物。
4. 论文中可以如何抽象和表述该系统。

本文档不复述配置中的敏感凭据，只描述其技术作用。

---

## 2. 一句话定位

EmboMaster 是构建在 EvoMaster 通用 Agent 基座之上的面向 embodied ML 研发的自动化实验系统。它不是单轮代码代理，而是一个带有“代码修改 - 快速验证 - 集群训练评测 - 指标回流 - 下一轮继承选择”闭环的多轮实验编排器。

从系统层面看，EmboMaster 同时具备以下三种身份：

1. 它是一个 `playground`，通过 `@register_playground("embomaster")` 挂接进 EvoMaster。
2. 它是一个 `Exp` 编排器，用多轮 round 替代普通 agent 的单次任务执行。
3. 它是一个实验工程壳层，把 workspace 隔离、K8s 作业提交、日志抽取、评测结果校验、反馈代理建议组合到一起。

---

## 3. 当前代码位置与目录结构

当前实现主要分布在以下目录：

```text
run.py
evomaster/
configs/embomaster/
playground/embomaster/
```

其中 `playground/embomaster` 是核心实现目录，`configs/embomaster` 是实验配置与 K8s 模板目录。

### 3.1 `playground/embomaster` 的核心结构

```text
playground/embomaster/
├── core/
│   ├── playground.py
│   ├── exp.py
│   ├── services/
│   │   └── k8s_experiment_runner.py
│   ├── tools/
│   │   ├── debug_test.py
│   │   └── video_descriptor.py
│   └── utils/
│       └── workspace_isolation.py
├── prompts/
│   ├── coding_system_prompt.txt
│   ├── coding_user_prompt.txt
│   ├── feedback_system_prompt.txt
│   └── feedback_user_prompt.txt
├── PRDs/
├── scripts/
├── test/
├── README_CN.md
└── ARCHITECTURE_CN.md
```

### 3.2 `configs/embomaster` 的当前结构

当前配置目录不只包含一个默认配置，而是已经演化成面向多任务、多基准的实验模板集合，主要包括：

1. 通用配置：`config.yaml`、`config.steps1.yaml`
2. RoboTwin 任务配置：
   - `config_robotwin_adjust_bottle_dsv32.yaml`
   - `config_robotwin_adjust_bottle_dsv32_e2e_once.yaml`
   - `config_robotwin_adjust_bottle_dsv32_e2e_10s.yaml`
   - `config_robotwin_place_phone_stand_dsv32_e2e_10s.yaml`
   - `config_robotwin_pick_banana_pi05_e2e.yaml`
3. K8s 模板目录：`k8s_template/`
4. MCP 搜索配置：`mcp_x_master_search.json`

从 PRD 和 K8s 模板可以看出，当前 EmboMaster 已覆盖或尝试覆盖以下任务族：

1. RoboTwin
2. MetaWorld
3. Robomimic
4. D4RL
5. 一部分 sim2real/robotwin 任务

这说明 EmboMaster 当前并不是单一 benchmark 的脚本，而是面向多 embodied benchmark 的统一实验外壳。

---

## 4. 与 EvoMaster 主框架的关系

### 4.1 启动入口

顶层入口仍然是 `run.py`。其职责是：

1. 自动导入所有 playground，触发注册。
2. 解析 `--agent`、`--config`、`--task`、`--run-dir` 等命令行参数。
3. 为对应 playground 构建运行目录。
4. 调用 `playground.run(...)` 执行任务。

对 `embomaster` 而言，`run.py` 还有一层专用逻辑：默认 `run_dir` 不再只是时间戳目录，而是按如下层级组织：

```text
playground/embomaster/workspaces/{simulator}/{task}/{model}/{date}/{experiment_name}_{HHMMSS}
```

该分层来自以下三类配置推断：

1. `workspace_isolation.source_codebase_dir` 提取 simulator 名称
2. `k8s_runner.manifest_env.TASK_NAME` 提取任务名
3. `agents.coding.llm` 对应的模型配置提取模型名

因此，EmboMaster 的 run 目录天然编码了“模拟器 / 任务 / 模型 / 日期”四元组，方便后续管理与论文实验追踪。

### 4.2 复用的 EvoMaster 基座能力

EmboMaster 并没有重写 EvoMaster 的所有底层机制，而是复用了以下基座：

1. `BasePlayground`：生命周期管理、配置装载、Session 初始化、Agent 创建、run_dir 管理
2. `BaseExp`：实验对象抽象和结果存储接口
3. `Agent` 与 tool-calling 执行框架
4. `LocalSession`/`DockerSession` 一类 Session 抽象
5. 工具注册中心与 MCP 工具集成
6. 轨迹记录与结果保存机制

EmboMaster 的创新点主要不在底层 agent 协议，而在实验编排层和工程系统层。

---

## 5. 系统总体架构

### 5.1 逻辑分层

从实现上，可以把当前 EmboMaster 划分为五层：

1. 启动与注册层
   - `run.py`
   - `register_playground("embomaster")`
2. Playground 装配层
   - `EmboMasterPlayground`
   - 负责 Session、双 Agent、自定义工具、K8s 服务装配
3. Round 编排层
   - `EmboMasterExp`
   - 负责多轮执行、父工作区选择、指标比较、反馈回流
4. 基础设施服务层
   - `K8SExperimentRunner`
   - `workspace_isolation`
5. 工具与观察层
   - `debug_test`
   - `video-descriptor`
   - MCP 搜索工具

### 5.2 高层执行图

```text
run.py
  -> EmboMasterPlayground.setup()
      -> 初始化 Session
      -> 创建 coding agent
      -> 创建 feedback agent
      -> 注册 debug_test / video-descriptor
      -> 应用 tool policy
      -> 创建 K8SExperimentRunner
  -> EmboMasterExp.run()
      -> for each round:
           1. 选择父 workspace
           2. 构建本轮隔离 codebase
           3. coding agent 改代码
           4. 可选 debug_test 快速验证
           5. 提交 K8s job
           6. 采集日志与 metric
           7. 校验实验产物
           8. feedback agent 生成下一轮建议
           9. 更新 best round
```

---

## 6. Playground 层设计

`EmboMasterPlayground` 是 EmboMaster 的装配中心，而不是执行算法本身。其主要职责如下。

### 6.1 创建双 Agent 结构

当前实现默认支持两个 agent：

1. `coding_agent`
   - 主执行代理
   - 负责读任务、改代码、调用工具、结束 round
2. `feedback_agent`
   - 反馈代理
   - 负责读取上一轮状态、日志、指标和候选摘要，生成下一轮建议

这意味着 EmboMaster 当前采用的是“执行代理 + 反思代理”的双代理结构，而不是单 Agent 自反思。

### 6.2 自定义工具注入

Playground 在 `setup()` 中动态注册两个自定义工具：

1. `debug_test`
   - 面向短命令验证
   - 适合 `pytest`、小脚本、快速训练前检查
2. `video-descriptor`
   - 面向机器人操作视频理解
   - 调用兼容 OpenAI 接口的视频多模态模型

工具注册完成后，系统会应用 `tool_policy` 对工具做白名单/黑名单裁剪。这一点很关键：EmboMaster 不是把所有工具都直接暴露给 agent，而是通过配置把 agent 行为空间收缩到实验需要的工具集合。

### 6.3 K8s 服务装配

如果 `k8s_runner.enabled=true`，或者 `debug_test` 被配置为优先走 debug pod，则 Playground 会创建 `K8SExperimentRunner`，并注入给工具层和 Exp 层共同使用。

这使得 K8s 在系统中有两个用途：

1. 正式训练/评测作业执行
2. debug pod 环境中的短命令诊断

---

## 7. EmboMasterExp：核心实验编排器

`EmboMasterExp` 是系统最核心的逻辑。与普通 `BaseExp` 的单轮 Agent 执行不同，它实现了一个明确的多轮实验闭环。

### 7.1 Round 主循环

每次任务执行会进入一个 `for round in 1..max_rounds` 循环，受以下配置控制：

1. `experiment.steps`
2. `experiment.time_limit_seconds`
3. `experiment.stop_on_job_failed`

每轮都会生成一个 `round_result`，并在全局维护：

1. `round_results`
2. `best_metric`
3. `best_round`
4. `previous_workspace_id`
5. `previous_feedback`

### 7.2 Coding round 的输入增强

在每一轮调用 coding agent 之前，系统会将以下上下文显式注入 prompt 变量：

1. `round_index`
2. `max_rounds`
3. `feedback_for_next_round`
4. `best_metric`
5. `workspace_id`
6. `parent_workspace_id`
7. `workspace_codebase_path`
8. `workspace_source_type`
9. `workspace_large_dirs_count`

这表明当前 EmboMaster 已经不是“无状态代码代理”，而是显式感知迭代轮次和代码继承来源的状态化代理。

### 7.3 Session 工作目录切换

在调用 coding agent 前，系统会把 agent session 的 `workspace_path` 暂时切换到当前 round 的隔离 codebase 路径。调用结束后恢复原值。

这一机制的意义是：

1. agent 感知到的工作目录始终是本轮 codebase
2. 多轮之间不会直接在同一目录上互相污染
3. 工具执行与代码编辑天然落在 round 级隔离环境中

### 7.4 K8s 回路

若启用 `k8s_runner`，每一轮 coding 完成后会：

1. 基于模板生成唯一 job name
2. patch manifest 中的 env 和挂载
3. 提交 K8s job
4. 轮询等待作业完成或超时
5. 拉取 job 日志
6. 从日志中抽取 metric

当前默认 metric 抽取机制支持：

1. 用户自定义正则 `experiment.metric_pattern`
2. 若未命中，则回退匹配若干常见字段，如 `success rate`、`validation score`、`metric`

### 7.5 结果校验与无效轮过滤

EmboMaster 不把日志里抽出的 metric 视为绝对可信。当前实现存在一个额外的 `result_validation` 阶段，用文件系统产物验证本轮结果是否有效。

可校验的内容包括：

1. 是否存在 `eval_result/{task}` 目录
2. 是否存在结果文本，如 `_result.txt`
3. 是否存在 checkpoint
4. 是否存在 `dataset_stats.pkl`

如果校验失败，系统会：

1. 把本轮 `result_valid` 设为 `false`
2. 根据配置覆盖 `k8s_status`
3. 将 `metric_valid=false`
4. 可选地清空 `metric_value`

这一步对论文很重要，因为它说明系统不是只相信日志字符串，而是对实验产物做了一层结构化一致性检查。

### 7.6 Feedback round

若存在 feedback agent，则每一轮结束后还会执行一个反馈轮。反馈代理读取的上下文包括：

1. 本轮 K8s 状态
2. 本轮 metric
3. K8s 日志尾部
4. 当前 best round 摘要
5. 当前 last round 摘要
6. 最近若干轮摘要
7. 父工作区选择约束

反馈代理输出两类信息：

1. 面向下一轮修改的自然语言建议
2. 一个机器可解析的父工作区推荐块

推荐块格式如下：

```text
## Parent Recommendation
Choice: best|last|none
Confidence: 0.00-1.00
Reason: ...
```

系统随后解析该块，为下一轮 `advisor` 模式下的 workspace 继承提供依据。

---

## 8. Workspace Isolation：每轮独立代码空间机制

这是 EmboMaster 当前最有工程特色的模块之一。

### 8.1 设计目标

该模块同时要满足三个目标：

1. 每轮代码可隔离，避免 round 之间互相污染
2. 大型数据、权重、资产目录不能被全量复制
3. K8s 容器内仍需看到连续一致的目录结构

### 8.2 基本目录形态

以任务 workspace 内的 session 目录为根，系统会创建：

```text
.embomaster_session/
└── round_workspaces/
    └── {workspace_id-or-short-name}/
        ├── codebase_{workspace_id-or-short-name}/
        └── codebase -> codebase_{workspace...}
```

其中：

1. `codebase_*` 是真实目录
2. `codebase` 是便捷 symlink
3. 每个 round 对应一个独立 codebase

### 8.3 工作区来源类型

当前实现中，round codebase 可以来自四种来源：

1. `existing`
   - 该 round 目录已经存在，直接复用
2. `parent`
   - 从父 round 的 codebase 增量复制
3. `original`
   - 从原始 `source_codebase_dir` 初始化
4. `empty` 或 `fallback`
   - 没有可用源目录时退化为空目录或共享路径

### 8.4 小文件复制 + 大目录挂载

EmboMaster 并不复制整个源代码仓库，而是使用“复制小文件、跳过大目录”的策略。

识别大目录的规则由以下因素共同决定：

1. 目录大小超过阈值 `size_threshold_mb`
2. 目录名命中关键词：
   - `assets`
   - `data`
   - `ckpt`
   - `checkpoint`

对于这类大目录，系统不会复制内容，而是：

1. 在目标 codebase 中创建占位目录
2. 在 `large_dirs.json` 中记录真实源路径和相对路径
3. 在 K8s 作业运行时通过 `hostPath` 将其重新挂载回容器中的对应位置

这个设计是 EmboMaster 当前工程可扩展性的关键，因为 embodied 任务往往伴随大数据集、模型权重和资源文件。

### 8.5 Copy Plan Cache

为了避免每轮都全量扫描原始代码库，系统引入 copy plan cache，默认文件名为：

```text
.embomaster_copy_plan.json
```

其缓存内容包含：

1. source root
2. size threshold
3. large_dirs 列表
4. cache 版本号

命中缓存时，系统可直接复用目录规划；目录变化大时可以强制重建。

### 8.6 排除策略

当前 workspace 复制过程中会显式跳过一批高噪声或不适合复制的目录与文件，例如：

1. `eval_result`
2. `run_results`
3. `__pycache__`
4. `.git`
5. `.venv`、`venv`
6. 各类 checkpoint 文件后缀
7. 若干本地 cache 和临时 uv 目录

这说明当前系统不仅仅做“粗粒度目录复制”，而是已经针对 ML 工程仓库做了较细的拷贝过滤优化。

### 8.7 局部本地工作目录

当前实现中，`policy/ACT/act_ckpt` 被列为 `LOCAL_WORKSPACE_DIR_PREFIXES`。这意味着即使该目录可能与大目录相关，系统仍会在每轮工作区中确保它以本地目录形式存在，以适配训练过程写入。

这反映了当前实现兼顾了“只读大目录挂载”和“局部训练产物本地落盘”的需求。

---

## 9. Parent Selection：父工作区继承策略

EmboMaster 当前支持三类父 workspace 策略：

1. `previous`
   - 永远继承上一轮
2. `best`
   - 永远继承当前 best round
3. `advisor`
   - 由反馈代理推荐 `best`、`last` 或 `none`

在 `advisor` 模式下，系统不会盲信反馈代理，而是还会检查约束：

1. `last` 是否有 metric
2. `last` 是否有有效 eval artifact
3. `last` 的 K8s 状态是否命中拒绝集合，如 `failed`、`timeout`

如果反馈推荐不合法，系统会退回到配置中的 `fallback_strategy`。

论文角度看，这一设计使 EmboMaster 具备一种轻量级“搜索树剪枝”特征：它并不是简单线性迭代，而是允许在最近轮与历史最佳轮之间切换继承分支。

---

## 10. K8SExperimentRunner：集群实验执行服务

K8S runner 是 EmboMaster 的第二个关键工程模块。

### 10.1 服务职责

它负责把 round 级实验从“本地代码修改结果”桥接到“集群训练评测 job”，包括：

1. 根据模板准备 manifest
2. 注入环境变量
3. patch workspace/codebase/submission/large_dir 挂载
4. 提交 job
5. 轮询 job 状态
6. 拉取日志
7. 按需清理 job

### 10.2 Manifest patch 机制

在提交 job 前，runner 会对模板做动态改写，核心挂载包括：

1. `workspace_codebase_path` -> 容器中的 codebase 挂载路径
2. `submission_dir` -> 容器中的 submission 路径
3. `large_dirs.json` 中记录的每个大目录 -> 容器内 codebase 下对应相对路径

此外，当前实现还支持一个兼容性挂载：

1. `enable_assets_alias_mount=true`
2. 将 codebase 下的 `assets` 再别名挂载到容器的 `/workspace/assets`

这个设计显然是为兼容 RoboTwin 一类任务中路径写死到 `/workspace/assets/...` 的历史代码。

### 10.3 作业生命周期管理

当前 runner 的基本作业周期是：

1. `kubectl apply -f manifest`
2. 周期性轮询 `kubectl get job -o json`
3. 成功或失败后结束等待
4. `kubectl logs job/{job_name}`
5. 根据配置清理 job

超时后可选自动删除 job，避免资源持续占用。

### 10.4 Debug Pod

除了正式 Job，runner 还支持 debug pod 模式，供 `debug_test` 使用。其能力包括：

1. 自动创建或复用一个长驻 pod
2. 将 host workspace 挂载到容器中的 `/workspace`
3. 在 pod 中执行短 shell 命令
4. 可按 workspace host path 生成稳定 pod 名
5. 可固定到本机 node，减少调试时路径不一致问题

这一设计的价值在于减少“本地能跑、集群不能跑”的环境偏差。

---

## 11. 工具层设计

### 11.1 `debug_test`

`debug_test` 是 EmboMaster 中最重要的动作工具之一。其特点是：

1. 参数简单：`command`、`timeout`、`working_dir`、`env_init`
2. 默认运行在当前 round 的 workspace 内
3. 可在本地 session 执行
4. 也可切换到 K8s debug pod 执行
5. 若 pod 执行失败，可按配置回退本地

这使得 agent 可以在正式提交训练任务前进行低成本检查。

### 11.2 `video-descriptor`

该工具面向机器人评测视频的诊断。其流程是：

1. 接收视频路径和分析 prompt
2. 将视频编码为 data URL
3. 调用兼容 OpenAI Chat Completions 的多模态模型
4. 返回中文诊断与改进建议

当前工具特别适合 embodied task，因为很多失败无法仅从日志判断，还需要从行为视频中观察轨迹偏差、末端姿态误差和任务接触失败等问题。

### 11.3 MCP 搜索工具

在当前 `config_robotwin_pick_banana_pi05_e2e.yaml` 中，coding agent 还启用了：

1. `xmaster_search_web_search`
2. `xmaster_search_web_parse`

说明当前 EmboMaster 已具备外部检索增强能力，可在某些任务中查询网络资料或外部页面。

---

## 12. Prompt 与 Agent 行为设计

### 12.1 Coding Agent Prompt

当前 coding system prompt 对 agent 施加了几个硬约束：

1. 负责修改当前 workspace 中的代码
2. 必须使用 `str_replace_editor`、`debug_test` 等工具
3. 不允许声称成功而不做验证
4. 结束前必须调用 `finish`

user prompt 则显式注入了轮次、workspace 路径、best metric 与上一轮反馈。这说明当前 prompt 设计已经把实验状态信息嵌入到模型输入中，而不是只给一个 PRD。

### 12.2 Feedback Agent Prompt

feedback agent 的 system prompt 要求其：

1. 读取 K8s 状态、metric、日志尾部
2. 如果存在评测视频，则优先调用 `video-descriptor`
3. 输出不超过 5 条建议
4. 明确给出下一轮应继承的 parent workspace 选择

因此，feedback agent 不是泛泛地“写总结”，而是承担一个轻量 experiment advisor 的角色。

---

## 13. 当前配置体系与实例

### 13.1 通用配置块

当前 EmboMaster 配置大体由以下 section 组成：

1. `llm`
2. `agents`
3. `session`
4. `debug_test`
5. `video_descriptor`
6. `tool_policy`
7. `experiment`
8. `workspace_isolation`
9. `parent_selection`
10. `result_validation`
11. `k8s_runner`
12. `mcp`
13. `logging`
14. `run_dir_base/workspace/results_dir`

这说明它已经是一套完整实验系统配置，而不是单纯的 LLM 调参文件。

### 13.2 当前活跃实例：`config_robotwin_pick_banana_pi05_e2e.yaml`

从当前用户打开的配置看，系统正在围绕 `pick_banana` 的 Pi0.5/OpenPI 任务运行端到端闭环。该配置体现出几个特征：

1. coding 与 feedback 共用同一主模型配置
2. coding agent 可使用 `debug_test`、`video-descriptor` 和外部搜索工具
3. feedback agent 主要做结果分析，不承担正式代码修改
4. 每轮最多 10 次迭代，时间上限 24 小时
5. `workspace_isolation.parent_strategy=advisor`
6. 启用 K8s 训练评测和 artifact 校验
7. 任务 metric 通过日志中 `FINAL_METRIC=...` 抽取

换言之，当前的 pick_banana 配置已经代表了一套完整的“多轮自动优化”工作流，而不是一次性脚本。

### 13.3 Session 配置

当前实例采用本地 session：

1. `session.type=local`
2. `working_dir=./playground/embomaster/workspaces`
3. 通过 symlink 将外部 RoboTwin 代码库映射为 `codebase`

这说明 EmboMaster 当前并不强制要求源代码必须位于 EvoMaster 仓库内部。它可以把外部真实训练仓库作为被操作对象，再由 workspace isolation 为每轮复制/挂载出隔离版本。

---

## 14. 输入、状态与输出数据

### 14.1 主要输入

当前系统的主要输入包括：

1. PRD 或任务描述 Markdown
2. 源代码仓库路径
3. LLM 配置
4. K8s 模板与环境变量
5. 评测 metric 抽取规则
6. workspace 继承与校验策略

### 14.2 中间状态

EmboMaster 在运行中维护的关键状态包括：

1. `round_results`
2. `best_round`
3. `best_metric`
4. `previous_feedback`
5. `workspace_context`
6. `large_dirs.json`
7. copy plan cache

### 14.3 主要输出

系统输出既包括语言层输出，也包括工程产物：

1. 修改后的 round 级 codebase
2. `submission/` 目录
3. K8s job 日志
4. `eval_result/`
5. checkpoint
6. trajectory 文件
7. 每轮的 round result 字典

因此，EmboMaster 输出的是一个“可追溯实验状态空间”，不是单一文本答案。

---

## 15. 轨迹与监控

当前仓库为 EmboMaster 提供了专用监控脚本：

1. `scripts/traj_monitor_server.py`
2. `scripts/traj_monitor_app.js`

其监控对象不仅包括普通 agent 轨迹，还包括：

1. 不同 round 的 step 记录
2. `debug_test` 调用
3. pod/job 状态
4. K8s 日志
5. 每轮的 K8s status 与 metric

这表明 EmboMaster 已经具有一定实验可观测性，而不是“黑盒自动改代码”。

---

## 16. 当前系统的工程特征总结

从代码实现看，当前 EmboMaster 具备以下鲜明特征。

### 16.1 它是“状态化”的

系统把 round、best metric、parent workspace、artifact 校验结果等状态显式暴露给 agent。相比传统 stateless code agent，这使得它更接近一个实验搜索系统。

### 16.2 它是“面向真实 ML 仓库”的

大量逻辑都围绕真实训练代码库展开，例如：

1. 大目录挂载
2. checkpoint 与结果目录过滤
3. K8s manifest patch
4. dataset/statistics/ckpt 校验

说明它不是泛化的代码助手，而是高度贴近 ML 实验工程的系统。

### 16.3 它是“多观察源反馈”的

当前反馈源不只来自文本日志，还来自：

1. metric
2. 文件系统产物
3. 视频诊断
4. 最近若干轮历史摘要

这使其在 embodied 任务上比纯日志驱动系统更有表现力。

### 16.4 它是“半搜索式”的

通过 `best/last/none` 的 parent 选择，系统已经具备某种轻量分支恢复能力。虽然还不是显式树搜索，但已明显超出简单线性 self-refinement。

---

## 17. 当前局限与论文中应如实说明的点

从当前实现看，也存在一些应在论文中诚实描述的限制。

### 17.1 反馈仍是启发式

feedback agent 提供的是自然语言建议和一个离散 parent recommendation，并没有显式学习到稳定策略，也没有独立 value model。

### 17.2 metric 抽取依赖日志格式

尽管已有 fallback regex，但 metric 仍主要来自日志字符串匹配，鲁棒性受训练脚本输出格式影响。

### 17.3 result validation 仍是任务相关规则集合

当前 artifact 校验使用了一组通用但仍偏工程经验的文件规则，尚不是完全 benchmark-agnostic 的 evaluator。

### 17.4 workspace 继承是近似分支搜索

系统目前只在 `best`、`last`、`none` 三者间切换，没有构建更系统的多分支并行搜索或回溯调度。

### 17.5 K8s 与路径配置仍较重

当前系统对 hostPath、容器路径、外部代码库位置有较强依赖，因此迁移到新集群或新代码仓库时仍需工程对齐。

---

## 18. 论文写作建议：如何表述 EmboMaster

如果面向论文，可将当前 EmboMaster 抽象为以下系统描述。

### 18.1 系统定义

可将 EmboMaster 定义为：

> 一个面向 embodied ML 实验自动化的 LLM-driven iterative experimentation system。系统通过 round-based workspace isolation、cluster-backed training/evaluation、artifact-aware result validation，以及 feedback-guided parent workspace selection 来持续优化策略代码。

### 18.2 建议的系统模块图

论文图中建议画出以下模块：

1. Task/PRD Input
2. Coding Agent
3. Workspace Isolation Manager
4. Debug/Test Tools
5. K8s Experiment Runner
6. Metric & Artifact Validator
7. Feedback Agent
8. Parent Workspace Selector
9. Round Repository / Experiment Memory

### 18.3 可以强调的创新点

建议重点强调以下三点：

1. 面向 embodied ML 仓库的 round-level isolated workspace 机制
2. 结合 cluster execution 与 artifact validation 的自动实验闭环
3. 通过 feedback-guided parent selection 在“最近轮”和“最优轮”之间切换继承路径

### 18.4 适合的术语

文中建议优先使用以下术语：

1. iterative experimentation
2. round-based optimization
3. isolated workspace
4. artifact-aware validation
5. cluster-backed evaluation
6. feedback-guided inheritance
7. embodied ML code optimization

---

## 19. 结论

当前代码仓库中的 EmboMaster 已经不是一个简单的“让大模型改代码”的 playground，而是一个较完整的 embodied ML 自动实验系统。其核心价值不在单个 prompt，而在以下系统级组合：

1. 用 EvoMaster 基座承接标准 Agent 生命周期
2. 用多轮 `Exp` 编排形成持续优化闭环
3. 用 workspace isolation 解决 round 之间的代码与数据隔离问题
4. 用 K8s runner 将本地代码修改映射到真实训练评测环境
5. 用 artifact validation 和 feedback agent 提高指标反馈的可信度与可用性

如果从论文角度概括，EmboMaster 的本质可以表述为：

> 一个面向 embodied policy optimization 的、具备状态感知、多源反馈、工程可落地的 LLM 自动实验代理系统。

