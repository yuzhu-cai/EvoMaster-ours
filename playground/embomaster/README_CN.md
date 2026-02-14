# EmboMaster Playground（最小骨架）

该 playground 提供三个基础能力：

1. `debug_test` 自定义工具（Agent 层快速验证）
2. `EmboMasterPlayground`（编排层入口）
3. `K8SExperimentRunner`（Exp 层实验执行服务）

## 目录结构

```text
playground/embomaster/
├── core/
│   ├── playground.py
│   ├── exp.py
│   ├── services/
│   │   └── k8s_experiment_runner.py
│   ├── utils/
│   │   └── workspace_isolation.py
│   └── tools/
│       └── debug_test.py
└── prompts/
    ├── coding_system_prompt.txt
    └── coding_user_prompt.txt
```

## 运行

```bash
python run.py --agent embomaster --config configs/embomaster/config.yaml --task "your task"
```

RoboTwin-Adjust-Bottle-DSV32 映射配置：

```bash
python run.py --agent embomaster \
  --config configs/embomaster/config_robotwin_adjust_bottle_dsv32.yaml \
  --task /data/agents/openhands-ml-master/PRDs/robotwin_task_2.md
```

## 配置说明

- `debug_test.enabled`: 是否注册 `debug_test` 工具
- `debug_test.use_k8s_debug_pod`: `debug_test` 是否优先在 K8S debug pod 中执行
- `debug_test.k8s_fallback_to_local`: K8S debug pod 失败时是否自动回退本地执行
- `workspace_isolation.enabled`: 是否启用“每轮独立工作空间 + 大目录挂载”逻辑
- `k8s_runner.enabled`: 是否在 Exp 阶段调用 K8S job 执行
- `k8s_runner.manifest_path/job_name_prefix`: 启用 K8S 时必填
- `k8s_runner.debug_pod.*`: debug pod 的镜像、名称前缀、挂载路径等参数

## 术语迁移

为对齐 EvoMaster 框架语义，embomaster 已统一使用 `workspace/round` 概念：

| 旧术语 | 新术语 |
|---|---|
| `node_workspace` | `workspace_isolation` |
| `node_id` | `workspace_id` |
| `parent_node_id` | `parent_workspace_id` |
| `node_codebase_path` | `workspace_codebase_path` |
| `node_context` | `workspace_context` |
| `nodes/{id}/codebase_*` | `round_workspaces/{id}/codebase_*` |

## Dry-Run 校验

在不提交真实 K8S Job 的情况下，可通过 `prepare_manifest` 校验挂载改写逻辑：

- codebase hostPath 是否指向本轮 `workspace_codebase_path`
- submission hostPath 是否指向本轮 `submission_dir`
- large-dir 挂载是否正确拼接到 `{codebase_mount_path}/{rel}`
