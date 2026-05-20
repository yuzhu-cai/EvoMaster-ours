# WebMaster

WebMaster 是一个面向 BrowseComp 的网页搜索 playground，实现
`web_master_playground_architecture.md` 里定义的版本 A：Flash-Searcher 风格的
DAG 并行搜索 agent。

当前实现沿用 EvoMaster 现有 playground 模式：

- `WebMasterPlayground(BasePlayground)` 负责配置、数据集加载、agent 声明和网页搜索工具注册。
- `FlashSearchExp(BaseExp)` 负责单题运行流程：规划 DAG -> 并行执行搜索节点 -> 汇总最终答案。
- 网页搜索工具从 `playground/web_master/tools` 加载。

运行示例：

```bash
python run.py --agent web_master --task "dataset:0"
python run.py --agent web_master --task "Who ...?"
```

默认配置：

```text
configs/web_master/config.yaml
```
