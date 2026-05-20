# WebMaster

WebMaster is a BrowseComp playground that implements the version-A architecture from
`web_master_playground_architecture.md`: a Flash-Searcher style DAG-parallel web search
agent.

The playground follows the existing EvoMaster pattern:

- `WebMasterPlayground(BasePlayground)` wires config, dataset loading, agents, and browse tools.
- `FlashSearchExp(BaseExp)` runs one BrowseComp task as plan -> parallel search nodes -> final answer.
- Browse tools are loaded from `playground/web_master/tools`.

Run examples:

```bash
python run.py --agent web_master --task "dataset:0"
python run.py --agent web_master --task "Who ...?"
```

Default config:

```text
configs/web_master/config.yaml
```
