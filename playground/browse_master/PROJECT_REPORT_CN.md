# Browse-Master 项目报告

> 本报告基于当前仓库中的实际代码实现整理，重点参考 `playground/browse_master` 及其依赖的 `evomaster` 框架代码，而不是只看 README 的预期描述。

## 1. 项目定位

`playground/browse_master` 是 EvoMaster 框架中的一个网页搜索智能体 playground，目标是解决 BrowseComp 这类需要联网检索、跨页面拼接事实、再输出短答案的问题。

从当前代码来看，Browse-Master 的核心思路是：

- 用 EvoMaster 提供的通用 Agent 执行框架承载任务；
- 给 Agent 注册两个浏览相关工具：`google_search` 和 `web_fetch`；
- 通过提示词强约束 Agent 必须先搜索、再抓取网页、最后只输出简短答案；
- 通过 `scripts/` 中的一组脚本完成批量运行、结果合并、LLM 判分和统计。

一句话概括：它不是从零开始写的独立搜索系统，而是一个“基于 EvoMaster 框架、专门面向 BrowseComp 的搜索型 Agent 实现”。

## 2. 当前实现的一个关键结论

虽然 `playground/browse_master/README_CN.md` 把它描述成一个 `Planner + Executor` 的双智能体系统，但当前代码实际运行的版本是单智能体版本。

也就是说，当前仓库里的 Browse-Master：

- 只有一个 `search_agent`；
- 没有代码层面的 Planner/Executor 分工；
- 没有多 Agent 之间的消息回路；
- 本质上是“一个带搜索工具的 ReAct 风格 Agent”。

这点是理解整个项目时最重要的事实，因为文档描述和代码实现已经出现了明显偏差。

## 3. 它在 EvoMaster 框架里的位置

Browse-Master 并不重新实现 Agent 基座，而是复用 EvoMaster 的标准组件：

- `run.py`：统一入口，负责加载 playground、解析任务、设置 `run_dir`；
- `evomaster/core/playground.py`：负责配置加载、session 初始化、agent 创建、日志与轨迹保存；
- `evomaster/agent/agent.py`：负责真正的 Agent 循环，包括 LLM 调用、tool calling、上下文压缩、finish 判定；
- `evomaster/core/exp.py`：负责实验结果封装与最终答案提取；
- `playground/browse_master/*`：只补充与 BrowseComp 相关的任务解析、提示词、搜索工具和批量评测脚本。

因此，Browse-Master 的实现非常“薄”：

- 框架层负责通用能力；
- playground 层只负责领域特化。

这也是 EvoMaster 设计思路的典型体现：一个新 Agent 不需要重写底座，只需补充 prompt、tool 和少量 workflow 代码。

## 4. 目录结构与职责

当前 `playground/browse_master` 可以按下面理解：

| 路径 | 作用 |
| --- | --- |
| `core/playground.py` | playground 入口，注册 Agent、加载数据集、注册工具、发起实验 |
| `core/exp.py` | 将问题包装成 `TaskInstance` 并执行，提取最终答案 |
| `prompts/system_prompt.txt` | 约束 Agent 的搜索策略与最终输出格式 |
| `prompts/user_prompt.txt` | 将问题文本注入用户提示词 |
| `tools/google_search.py` | 基于 Serper 的 Google 检索工具 |
| `tools/web_fetch.py` | 基于 Jina Reader 的网页抓取工具 |
| `scripts/run_batch.py` | 批量并行跑数据集任务 |
| `scripts/merge.py` | 合并标准答案与模型答案 |
| `scripts/eval.py` | 用 LLM 评判答案语义是否一致 |
| `scripts/summarize.py` | 汇总准确率 |
| `scripts/run_browse.sh` | 预期的一键评测脚本 |
| `test/browsecomp_decrypted.json` | 主数据集样例，当前代码直接依赖它 |
| `test/browsecomp-zh_task.json` | 中文任务样例，但当前主流程没有直接接入它 |

## 5. 核心执行链路

当前 Browse-Master 的实际执行链路如下：

```text
run.py
  -> BrowseMasterPlayground
    -> setup session / tools / agent
    -> BrowseMasterExp
      -> Agent.run()
        -> LLM 思考
        -> google_search / web_fetch / think / finish
        -> 轨迹保存
      -> 提取最终答案
```

更具体地说：

### 5.1 入口

运行命令通常是：

```bash
python run.py --agent browse_master --config configs/browse_master/config.yaml --task "..."
```

`run.py` 会自动导入 `playground/*/core/playground.py`，通过 `@register_playground("browse_master")` 找到 `BrowseMasterPlayground`。

### 5.2 Playground 层

`playground/browse_master/core/playground.py` 做了几件事：

1. 默认读取 `configs/browse_master` 下的配置；
2. 预声明一个 agent 槽位：`search_agent`；
3. 启动时加载 `test/browsecomp_decrypted.json`，并按 `id` 建索引；
4. 在基类默认工具之外，再注册 `google_search` 与 `web_fetch`；
5. 在运行任务前，把这两个自定义工具补进 agent 的 `enabled_tool_names`；
6. 支持两种任务输入：
   - 原始问题文本；
   - `dataset:42` 或 `id:42` 这种引用数据集编号的写法。

这个类并不负责复杂推理，只负责把 BrowseComp 任务接到 EvoMaster 的通用执行管线上。

### 5.3 Exp 层

`playground/browse_master/core/exp.py` 非常直接：

- 将问题包装成 `TaskInstance(task_type="search")`；
- 调用 `self.agent.run(task)`；
- 结束后用 `BaseExp._extract_agent_response()` 提取最终答案；
- 将 `trajectory`、`agent_answer`、`ground_truth` 一起写入结果结构。

这里说明 Browse-Master 本质上仍然是“标准 EvoMaster Agent + 一个轻量实验封装”。

### 5.4 Agent 层

真正的智能体执行逻辑在 `evomaster/agent/agent.py`：

- 按 `max_turns` 进入循环；
- 每轮把当前 `Dialog` 发给 LLM；
- 若 LLM 产生 tool call，则调用对应工具；
- 工具结果回填到对话中，继续下一轮；
- 调用 `finish` 时结束；
- 所有步骤会写进 trajectory。

这套流程是通用的，Browse-Master 只是把“科学实验”替换成了“网页检索问答”。

## 6. Prompt 设计

Browse-Master 的 prompt 非常关键，因为它把通用 Agent 收敛成了搜索型 Agent。

### 6.1 system prompt

`prompts/system_prompt.txt` 做了三件事：

- 指定可用的核心工具是 `google_search`、`web_fetch`、`finish`、`think`；
- 指定推荐工作流：先计划，再搜，再抓网页，再汇总；
- 强约束 `finish.message` 必须只包含最终答案，不能带解释。

这个“只输出短答案”的约束非常适合 BrowseComp，因为数据集通常要求一个名字、日期、数字或短语。

### 6.2 user prompt

`prompts/user_prompt.txt` 很简单，只是把 `{description}` 注入：

- 用户任务本身几乎不再加工；
- 复杂性主要由 system prompt 和工具能力承担。

## 7. 工具设计

Browse-Master 的搜索能力来自两个自定义工具。

### 7.1 `google_search`

`tools/google_search.py` 的特点：

- 使用 `SERPER_KEY_ID` 环境变量调用 Serper；
- 支持一次提交多个 query；
- 会根据 query 是否含中文自动切换中英文搜索地域参数：
  - 中文：`China / gl=cn / hl=zh-cn`
  - 非中文：`United States / gl=us / hl=en`
- 返回的是“标题 + 链接 + snippet”的原始检索结果，不做深度总结。

这说明它更像“候选网页召回器”，而不是答案生成器。

### 7.2 `web_fetch`

`tools/web_fetch.py` 的特点：

- 使用 `https://r.jina.ai/<url>` 获取网页正文；
- 支持一次抓取多个 URL，并发上限为 5；
- 目标是根据 `goal` 抽取对当前问题有用的证据和摘要；
- 预留了一个 `EXTRACTOR_PROMPT`，理论上可用 LLM 对网页正文做二次抽取。

但当前实现里有一个很重要的现实情况：

- `WebFetchTool.__init__()` 里 `self._llm = None`；
- 当前 Browse-Master 代码没有把 agent 的 LLM 注入给这个工具；
- 所以 `_extract_with_llm()` 会走降级分支，直接返回原始网页内容，而不是结构化摘要。

也就是说，当前 `web_fetch` 的实际行为更接近：

- “抓网页正文并截断”

而不是：

- “抓网页正文并用额外 LLM 抽取关键信息”。

### 7.3 一个容易忽略的点：实际暴露给 Agent 的工具比 prompt 里写的更多

配置里 `builtin: ["*"]`，而 EvoMaster 的内置工具包括：

- `execute_bash`
- `str_replace_editor`
- `think`
- `finish`

Browse-Master 再额外追加：

- `google_search`
- `web_fetch`

所以从框架实际暴露的工具集合来看，当前 `search_agent` 不只是“搜索 Agent”，它还具备执行 shell 和编辑文件的能力。只是 system prompt 没有重点强调这两项能力。

## 8. 数据集与任务组织

当前主数据集是：

- `playground/browse_master/test/browsecomp_decrypted.json`

我本地查看到它包含 1266 条记录，每条记录至少有：

- `id`
- `question`
- `answer`
- `canary`

Browse-Master 对这个数据集的支持是原生的：

- `core/playground.py` 启动时直接加载；
- 支持手工输入 `dataset:<id>` 读取指定题目；
- 评测脚本默认也围绕这个 JSON 展开。

另外仓库还放了一个：

- `playground/browse_master/test/browsecomp-zh_task.json`

但这个文件的字段是 `prompt` / `Answer` / `Topic`，与主流程里期待的 `question` / `answer` 并不一致，所以它目前更像样例数据，而不是已接入的正式输入格式。

## 9. 配置方式

Browse-Master 当前有两套配置：

### 9.1 `configs/browse_master/config.yaml`

- 默认 LLM 是 `deepseek`；
- `search` agent 最多 80 轮；
- context 上限 128000；
- session 使用 `local`。

### 9.2 `configs/browse_master/config_gpt.yaml`

- `search` agent 改用 `openai`；
- 最多 150 轮；
- context 上限 30000；
- 添加了 `reasoning_effort: medium`。

需要注意：

- 手工运行 `run.py` 时，如果不显式指定配置，一般走 `config.yaml`；
- 批量脚本 `run_batch.py` 里却硬编码使用 `config_gpt.yaml`。

所以“单题手工跑”和“批量评测跑”默认不是同一模型配置。

## 10. 自动评测流水线

Browse-Master 的 `scripts/` 目录实现了一套评测流水线。

### 10.1 `run_batch.py`

职责：

- 解析 `0-9,15,20-25` 这种 `id` 范围；
- 对每个 `id` 取出数据集中的 `question`；
- 并行调用：

```bash
python run.py --agent browse_master --config configs/browse_master/config_gpt.yaml --task <question> --run-dir <task_path>
```

- 为每个样本创建一个目录：`task_XXXX`；
- 尝试提取最终答案并写入 `solution.txt`。

### 10.2 `merge.py`

职责：

- 把数据集答案 `answer` 和每个任务目录下的 `solution.txt` 合并成 `merge.jsonl`。

### 10.3 `eval.py`

职责：

- 读取 `merge.jsonl`；
- 调用外部 LLM 判断 `answer` 与 `solution` 是否语义等价；
- 给每条记录追加 `score` 字段，取值 0/1。

### 10.4 `summarize.py`

职责：

- 统计 `score == 1` 的个数；
- 计算准确率；
- 写出结果 JSON。

### 10.5 `run_browse.sh`

原本设计成一键串起完整流程：

```text
run_batch -> merge -> eval -> summarize
```

并通过环境变量控制：

- `IDS`
- `RUN_NAME`
- `RUN_WORKERS`
- `EVAL_WORKERS`
- `DATA_JSON`

## 11. 从框架视角看，这个实现的优点

### 11.1 复用度高

Browse-Master 没有重新造 Agent、Session、Tool Registry、Trajectory 这些轮子，而是直接复用了 EvoMaster 的：

- 配置系统；
- prompt 加载；
- 工具注册；
- 轨迹记录；
- 上下文压缩；
- 本地运行环境。

这使得 playground 代码量很小，维护成本也低。

### 11.2 面向 BrowseComp 的约束比较明确

它没有追求“开放域聊天”，而是非常明确地把 Agent 收敛到：

- 多轮检索；
- 网页证据读取；
- 最后输出短答案。

这种强约束对 benchmark 型任务是合理的。

### 11.3 评测脚本齐全

即使还有一些实现细节问题，`run_batch.py / merge.py / eval.py / summarize.py` 这套结构已经把“跑样本 -> 落盘 -> 判分 -> 汇总”闭环搭好了。

## 12. 当前实现中的不一致与局限

这是我在理解整个项目后认为最值得注意的部分。

### 12.1 README 和代码不一致

当前 `README_CN.md` 说的是双 Agent 架构，但代码实际上是单 Agent。

连 README 里提到的 prompt 文件名：

- `planner_prefix.txt`
- `planner_user.txt`
- `executor_prefix.txt`
- `executor_user.txt`

在当前目录里都不存在。

结论是：文档更像旧设计或论文叙述，当前仓库实现则是简化后的单 Agent 版本。

### 12.2 `web_fetch` 的“LLM 抽取”能力当前没有真正接通

代码虽然预留了网页抽取 prompt 和 JSON 输出协议，但由于没有注入 LLM，实际运行时会退化为返回原始网页文本。

这会带来两个影响：

- Agent 需要自己在长文本中继续找答案；
- token 开销和推理负担更大。

### 12.3 `web_fetch` 依赖本地固定代理

抓取网页时写死了：

- `http://127.0.0.1:7890`
- `https://127.0.0.1:7890`

这意味着当前实现隐含假设本机有一个本地代理服务；否则网页抓取可能失败。这个依赖没有进入配置文件，移植性较弱。

### 12.4 `run_batch.py` 的轨迹读取路径和实际输出路径不一致

`run.py` 在单任务模式下会把任务 ID 固定成 `task_0`，因此 trajectory 实际会落在类似：

```text
task_XXXX/trajectories/task_0/trajectory.json
```

但 `run_batch.py` 读取的是：

```text
task_XXXX/trajectories/trajectory.json
```

所以它在很多情况下读不到轨迹文件，导致自己写出的 `solution.txt` 可能为空。后面的 `extract_browse_solutions.py` 实际上成了补救步骤。

### 12.5 `run_browse.sh` 目前没有真正串完完整评测

脚本里：

- `merge.py`
- `eval.py`
- `summarize.py`

这三个步骤都被注释掉了。

所以现在的 `run_browse.sh` 实际只执行：

- 批量跑题；
- 从日志里抽答案。

并不会自动产出最终准确率。

### 12.6 子集评测会被整份数据集稀释

`merge.py` 当前是按整个数据集遍历，不是按 `IDS` 子集遍历。

这意味着如果你只跑了 10 道题：

- `merge.jsonl` 仍然会包含全部数据集条目；
- 其余未跑题目的 `solution` 会是空字符串；
- 后续 `eval.py` 和 `summarize.py` 会把这些空答案也纳入分母。

因此，当前这套脚本对“只评测一个子集”的统计结果是不准确的。

### 12.7 Agent 实际权限比“搜索”更大

由于内置工具全部开启，Browse-Master 实际不只是浏览网页，它还可以：

- 执行 bash；
- 编辑文件。

这会让它比一个纯搜索 Agent 更强，但也意味着：

- 行为边界更宽；
- 复现实验时更难界定“到底是靠搜索能力还是靠额外 shell 能力解题”。

## 13. 总体评价

从代码角度看，Browse-Master 是一个“工程上很轻、但落点很明确”的实现：

- 它借助 EvoMaster 框架快速搭出了一个可运行的搜索 Agent；
- 它已经有完整的单题执行和批量评测雏形；
- 它适合当作 BrowseComp 类任务的研究原型；
- 但它当前仍然处于“原型已成、细节待收口”的阶段。

如果把它当成研究 playground，它已经足够清晰；
如果把它当成稳定评测系统，当前还存在一些文档、工具注入和批处理逻辑上的缺口。

## 14. 我给这个项目的建议理解

如果你之后要继续在这个目录上做工作，我建议把它理解成三层：

### 第一层：框架层

EvoMaster 提供通用 Agent 基建：

- config
- session
- agent loop
- tools
- trajectory

### 第二层：任务适配层

Browse-Master 自己补充：

- BrowseComp 数据读取；
- 搜索/抓网页工具；
- 搜索型 prompt；
- 最终答案抽取。

### 第三层：实验评测层

`scripts/` 负责把“单题 Agent”扩展成“可批量 benchmark 的实验流程”。

这样看，这个项目的本质不是一个复杂的多智能体系统，而是一个“依托 EvoMaster 的、可批量评测的网页检索 Agent 实验实现”。

## 15. 后续最值得优先改进的点

如果要继续完善，我认为优先级最高的几项是：

1. 统一文档与代码，明确当前到底是单 Agent 还是双 Agent；
2. 给 `web_fetch` 正式注入 LLM，让网页抽取真正工作；
3. 把代理地址、搜索后端等外部依赖移入配置；
4. 修复 `run_batch.py` 的 trajectory 路径问题；
5. 让 `merge/eval/summarize` 支持只统计指定 `IDS`；
6. 明确是否真的要向 Browse-Master 暴露 `execute_bash` 和 `str_replace_editor`。

---

如果只用一句话总结当前代码状态：

**Browse-Master 现在是一个建立在 EvoMaster 通用 Agent 框架上的单智能体网页搜索原型，核心链路已经打通，但文档与评测流水线仍有明显的“研究代码”特征。**
