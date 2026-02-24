# PHY Master Playground

PHY Master implements a five-agent workflow on top of EvoMaster:

1. **Clarifier**: analyzes the problem and creates initial subtasks.
   Clarifier can retrieve stage templates via the `workflow-retrieval` skill.
2. **Supervisor**: schedules next subtasks from critic feedback.
3. **Theoretician**: solves each selected subtask.
4. **Critic**: scores quality and proposes refinements.
5. **Summarizer**: reports the best path found by MCTS-style search.

## Run

```bash
python run.py --agent phy_master --config configs/phy_master/config.yaml --task "your physics problem"
```

After each run, MCTS visualization is exported to:

`runs/{agent}_{timestamp}/visualization.html`

## Search Loop

- Selection: UCB-style node selection.
- Expansion: supervisor/critic proposed subtasks become child nodes.
- Evaluation: critic score (0-1) is normalized to reward.
- Backpropagation: reward is propagated to ancestors.
- Finalization: summarizer agent explains the best path.

## Tuning

Use `phy_mcts` in `configs/phy_master/config.yaml`:

- `max_rounds`
- `max_depth`
- `max_children_per_node`
- `beam_width` (per depth, only top-K reward nodes stay expandable)
- `exploration_constant`
