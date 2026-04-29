# Codex CLI 批量运行器

这个目录提供了一个本机 Codex CLI 运行脚本，输出结构对齐
`playground/openclaw/batch_openclaw_runner.py`。

脚本会为每个任务启动一个新的 `codex exec --json` 进程，在任务自己的
`workspace` 中运行，保存 Codex 原始事件流，并从中提取简洁轨迹和最终回答。

## 当前主机 Codex 配置

当前测试使用的是本机已安装的 Codex CLI 和 `~/.codex/config.toml` 配置：

- Codex CLI 版本：`codex-cli 0.125.0`
- 默认模型：`gpt-5.5`
- 推理强度：`xhigh`
- 上下文窗口：`272000`
- 最大输出 token：`128000`
- Web search：开启，`tools_web_search = true`
- Plan tool：开启，`include_plan_tool = true`
- Apply patch tool：开启，`include_apply_patch_tool = true`

脚本默认会覆盖部分非交互运行配置：

- 默认 sandbox：`workspace-write`
- 默认 approval policy：`never`，通过 `-c approval_policy="never"` 传给 `codex exec`
- prompt 通过 stdin 传给 `codex exec -`，不会直接放在命令行参数里

可以通过 `--model`、`--profile`、`--sandbox`、`--approval-policy` 和 `--config`
覆盖默认行为。

## 输入任务文件

`--tasks` 需要传入一个 JSON 数组。数组中的每一项可以是字符串，也可以是对象。

```json
[
  {
    "id": "task_001",
    "prompt": "请输出一句 hello from codex。"
  },
  {
    "id": "task_002",
    "description": "检查当前 workspace，并总结你看到的内容。"
  }
]
```

支持的单任务字段包括：

- `id` / `instance_id`
- `prompt` / `description`
- `workspace_root`
- `codex_timeout_sec`
- `codex_system_prompt`
- `codex_model`
- `codex_profile`
- `images`

## 运行示例

单任务：

```bash
python playground/codex/batch_codex_runner.py \
  --task "只输出 hello。" \
  --output-dir runs/codex_single_001
```

批量任务：

```bash
python playground/codex/batch_codex_runner.py \
  --tasks playground/codex/tasks.example.json \
  --output-dir runs/codex_batch_001 \
  --parallel 2
```

如果省略 `--output-dir`，脚本会自动创建：

```text
runs/codex_batch_<timestamp>
```

## 输出结构

在 `--output-dir` 下会生成：

- `summary.json`：总体成功/失败统计
- `results.jsonl`：每个任务一行的结果记录
- `<task_id>/workspace/`：该任务默认的 Codex 工作目录
- `<task_id>/codex.stdout.jsonl`：原始 `codex exec --json` 事件流
- `<task_id>/codex.stderr.log`：Codex stderr 日志
- `<task_id>/dialogs.json`：提取后的对话式中间轨迹
- `<task_id>/trajectory.json`：EvoMaster 风格的轨迹对象
- `<task_id>/meta.json`：运行元数据
- `<task_id>/final_answer.txt`：最后一个 assistant 回答
- `<task_id>/result.json`：单任务结果摘要或错误信息

目录示例：

```text
runs/codex_batch_001/
  summary.json
  results.jsonl
  task_001/
    workspace/
    codex.stdout.jsonl
    codex.stderr.log
    dialogs.json
    trajectory.json
    meta.json
    final_answer.txt
    result.json
```

## 说明

- 脚本使用 `codex exec`，不是交互式 TUI。
- 默认 sandbox 是 `workspace-write`，因为每个任务都会有自己的 workspace。
- 默认 approval policy 通过 `-c approval_policy="never"` 传入，适合非交互执行。
- prompt 会通过 stdin 传给 `codex exec -`，不会直接放在命令行参数里。
