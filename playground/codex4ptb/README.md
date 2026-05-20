# Codex for PostTrainBench

这个目录提供一个 **native fallback** 版 PostTrainBench + Codex runner。
它不依赖 Apptainer、FUSE、HTCondor，适合当前这种不能修改容器启动权限的环境。

默认约定：

| 项目 | 默认位置 |
|---|---|
| PostTrainBench checkout | `/data/yuzhu/Devs/PostTrainBench` |
| venv / HF cache / pip/uv/vLLM/Triton cache | `playground/codex4ptb/local_state/` |
| 运行结果 | 仓库根目录 `runs/codex4ptb_<timestamp>/` |
| HTTP/HTTPS/ALL proxy | `http://127.0.0.1:7890` |

## 1. 自检

```bash
cd /data/yuzhu/Devs/EvoMaster-ours

python playground/codex4ptb/codex4ptb_runner.py doctor
```

## 2. 构建 native Python 环境

如果基础镜像缺少 venv 或 Python 头文件，先安装系统包：

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3.10-venv python3.10-dev build-essential
```

这一步会在 `playground/codex4ptb/local_state/base_venv` 下创建 base venv，并安装
vLLM、Transformers、TRL、PEFT、inspect-ai、inspect_evals 等依赖。

```bash
python playground/codex4ptb/codex4ptb_runner.py setup-env
```

如果 `python3.10` 不存在，可以指定当前 Python：

```bash
python playground/codex4ptb/codex4ptb_runner.py setup-env --python python3
```

## 3. 下载基础模型 cache

默认下载 PostTrainBench 的 4 个 base models 到本目录的 HF cache，并跳过
SmolLM3 仓库里不参与 vLLM 评测的 ONNX 大文件：

```bash
python playground/codex4ptb/codex4ptb_runner.py download-cache
```

如果没有 Gemma 的 Hugging Face gated repo 权限，先下载 3 个公开模型：

```bash
python playground/codex4ptb/codex4ptb_runner.py download-cache \
  --model Qwen/Qwen3-4B-Base \
  --model Qwen/Qwen3-1.7B-Base \
  --model HuggingFaceTB/SmolLM3-3B-Base
```

Gemma 需要已在 Hugging Face 接受模型条款，并提供有权限的 token：

```bash
HF_TOKEN=hf_xxx python playground/codex4ptb/codex4ptb_runner.py download-cache \
  --model google/gemma-3-4b-pt
```

只下载一个模型：

```bash
python playground/codex4ptb/codex4ptb_runner.py download-cache \
  --model Qwen/Qwen3-1.7B-Base
```

## 4. 跑一个 Codex + PTB 单任务

可以先 dry-run，只生成 run 目录、任务文件和 prompt，不调用 Codex：

```bash
python playground/codex4ptb/codex4ptb_runner.py run \
  --eval gsm8k \
  --model-to-train Qwen/Qwen3-1.7B-Base \
  --codex-model gpt-5.5 \
  --hours 1 \
  --dry-run
```

先用 GSM8K + Qwen3-1.7B-Base 小样本验证：

```bash
python playground/codex4ptb/codex4ptb_runner.py run \
  --eval gsm8k \
  --model-to-train Qwen/Qwen3-1.7B-Base \
  --codex-model gpt-5.5 \
  --hours 1 \
  --eval-limit 20 \
  --eval-baseline
```

输出目录类似：

```text
runs/codex4ptb_20260518_123456/
  prompt.txt
  run_config.json
  summary.json
  venv_agent/
  task/
  home/
  tmp/
  codex.stdout.jsonl
  codex.stderr.log
  codex_meta.json
  dialogs.json
  trajectory.json
  final_answer.txt
  metrics.json
  baseline_metrics.json
```

## 注意

- 这是 native fallback，不是官方 Apptainer 环境；结果适合本机开发/对比，
  不应直接当作官方 PostTrainBench leaderboard 结果。
- runner 会把当前 `~/.codex` 复制到每个 run 的独立 `home/.codex`，并将
  `HOME` 指向该目录，避免不同任务共享 Codex 会话/配置写入。
- Codex 默认启用 `codex --search`，并给子进程设置
  `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY=http://127.0.0.1:7890`。
- 每个 run 默认创建独立 `venv_agent/`。这个 venv 通过 `.pth` 读取
  `local_state/base_venv` 的已安装依赖，但新的 `pip install` 会写到
  `venv_agent/`，避免污染后续任务。
- agent 的工作目录是 run 目录下的 `task/`，最终模型应写入
  `task/final_model/`。
- 最终评估使用 `local_state/base_venv` 直接运行 PostTrainBench 原始
  `evaluate.py`，不复用 agent 的 `venv_agent/`。
