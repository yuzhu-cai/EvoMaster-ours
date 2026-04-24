# **PRD: Improving Metaworld MT10 Multi-Task PPO Training Performance**

## **Role**

You are an RL training engineer working on MetaWorld MT10 PPO in EmboMaster.

You are given an incomplete reinforcement learning training script with TODOs based on an existing codebase. Your task is to complete PPOConfig and model setup so MT10 training converges stably and reaches reasonable multi-task performance.

You are not implementing PPO from scratch. PPO algorithm, rollout logic, logging, and training loop already exist in the codebase. Your responsibility is algorithm configuration and model setup only.

## **Validated Runner Baseline**

Assume runner infrastructure is already correct (image, mount, cache, orchestration).

- training/evaluation are runner-managed
- environment identity remains MetaWorld MT10
- budget scale remains at the original `total_steps` regime

If a run fails due to environment/infrastructure issues, report clearly, but do not convert this task into infra work.

## **Primary Target**

Unless runner explicitly overrides, keep the default target:

- task family: `metaworld_mt10`
- training selector: `--task_config mt10`
- evaluation call (actual template behavior):
  `bash eval.sh "/workspace/submission" "${WORKSPACE_ID:-${HOSTNAME:-metaworld-job}}"`

## **Ground Rules (Agent Contract)**

- Environment must remain unchanged: do not modify MetaWorld env code or success criteria.
- Training budget is fixed: do not increase total training scale (`total_steps` level).
- Modification scope is restricted: edit only approved source files.
- Keep PPO line unchanged: do not replace with another algorithm/framework.
- Keep hyperparameter changes moderate: avoid large disruptive jumps.

## **Codebase Path**

The full codebase is already in workspace and current working directory is project root.

## **Related Code**

You may modify **only** this file:

1. `./examples/multi_task/ppo_mt10.py`

Default expectation:

- Most successful solutions should only update PPOConfig and model setup in this file.
- Do not create new files outside allowed scope.

## **Tool Usage Instructions**

Follow this workflow, consistent with the new PRD style:

1. Inspect current code in the allowed file first.
2. Apply direct, minimal edits only within the approved scope.
3. Re-open and verify the modified file content after edits.
4. Let the runner/system execute training and evaluation automatically.

Do not turn this task into infrastructure work, and do not rewrite unrelated modules.

## **Recommended Workflow**

### **Step 1: Inspect Before Editing**

Read `./examples/multi_task/ppo_mt10.py` and confirm:

- current PPOConfig defaults
- policy/value network settings
- rollout horizon, batch/update cadence, and advantage-related settings
- likely bottleneck (instability, under-exploration, multi-task interference)

### **Step 2: Apply Targeted Improvements**

Good directions:

- tune PPOConfig for stability and sample efficiency
- modestly improve capacity or regularization for MT10 generalization
- reduce variance/task imbalance without increasing budget

Bad directions:

- large unbounded hyperparameter swings
- environment or success-criteria modifications
- changing task identity away from MT10
- rewriting PPO loop/training framework

### **Step 3: Verify Changes**

Re-open modified file and ensure:

- intended edits are present
- no unrelated changes were introduced
- syntax/config structure remains valid

## **Train/Eval Contract Reference**

Keep this runner contract aligned:

- train side includes `--task_config mt10` (or `${TASK_CONFIG}` with default `mt10`)
- eval side follows actual template call:
  `bash eval.sh "/workspace/submission" "${WORKSPACE_ID:-${HOSTNAME:-metaworld-job}}"`
- typical chain in template:

```bash
timeout "${TRAIN_TIMEOUT:-7200}" bash train.sh --task_config "${TASK_CONFIG}" || true
bash eval.sh "/workspace/submission" "${WORKSPACE_ID:-${HOSTNAME:-metaworld-job}}"
```

## **Task Characteristics**

MT10 is a multi-task control benchmark where optimization stability and cross-task balance are both critical.

Typical failure modes:

- unstable early updates and noisy returns
- overfitting to easier task subsets
- insufficient exploration on harder tasks
- later-stage regressions on previously improving tasks

## **Optimization Priority**

Use this order:

1. verify MT10 train/eval argument contract alignment
2. stabilize optimization dynamics (LR/clip/advantage/update schedule)
3. improve multi-task robustness with small config/model refinements
4. avoid broad structural rewrites unless there is a concrete blocking bug

## **Deliverables**

After modification, report:

1. changed files
2. exact modified fields/sections
3. rationale for each change
4. expected impact on stability/performance
5. remaining risks or assumptions
