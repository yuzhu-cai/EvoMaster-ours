# Browse-Master Playground

## 概述

Browse-Master Playground 实现两个Agent的工作流

- **Planner** 将任务划分为多个子任务，与Executor交互，生成最终答案
- **Executor** 利用工具搜索子任务，返回阶段性的答案到Planner

## 工作流程

```

                     ┌──────────┐                            
           ┌─────────│  Planner │─────最终答案                            
           |         └────┬─────┘
           |              ▼                           
           |            子任务
           |              |
           |              ▼
           |         ┌──────────┐                            
           |         │ Executor │                            
           |         └────┬─────┘
           |             答案
           |              |
           └──────────────┘
```

## 快速开始

### 1. 配置

编辑 `configs/browse_master/config.yaml`：

```yaml
# ============================================
# Multi-Agent Configuration
# ============================================
# In the multi-agent system, each Agent has independent configuration

agents:
  
  planner:
    llm: "openai"
    max_turns: 10
    tools:
        builtin: []

    context:
      max_tokens: 4096
      truncation_strategy: "latest_half"
      preserve_system_messages: true
      preserve_recent_turns: 5

    # Prompt configuration (relative to playground/browse_master/)
    system_prompt_file: "prompts/planner_prefix.txt"
    user_prompt_file: "prompts/planner_user.txt"

  executor:
    llm: "openai"
    max_turns: 50
    tools:
        builtin: ["*"]     
        mcp: "mcp_config.json"

    context:
      max_tokens: 4096
      truncation_strategy: "latest_half"
      preserve_system_messages: true
      preserve_recent_turns: 5

    # Prompt configuration (relative to playground/browse_master/)
    system_prompt_file: "prompts/executor_prefix.txt"
    user_prompt_file: "prompts/executor_user.txt"


```

### 2. 运行

```bash
python run.py --agent browse_master --config configs/browse_master/config.yaml --task <task description>
```