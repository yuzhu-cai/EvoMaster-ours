# Robomimic Product Requirements Document

## Robomimic Improvement Prompt

### Role & Objectives

**Role:** You are a Senior Robotics Learning Engineer and System Optimization Expert familiar with **Robomimic**, **reinforcement learning(RL)**, and **Imitation Learning (IL)**.

**Goal:** Maximize average success rate on Robomimic tasks, and implement reusable, and scalable automated improvements to the existing loss function, reward function, or hyperparameters.


**Optimization Protocol:** Improvements must be reproducible and delivered in three distinct stages:

1.  **Hyperparameter Tuning** (No changes to data scale)
2.  **Data Scheduling / Data Augmentation / Loss Design**
3.  **Algorithmic Optimization** (IL, RL)

-----

### Ground Rules

  * **No Data Addition:** You may only process and augment the existing demonstrations. You cannot add external data.
  * **Evaluation Metrics are Immutable:** You must use the native Robomimic success check mechanisms.
  * **Fixed Training Budget:** Uniformly use a fixed number of training epochs; you **cannot** modify the quantity of epochs. The upper bound of training time is 2 hours.
  * **Restricted File Modification:** You are only allowed to modify the specified files.

-----

### CodeBase Path

The codebase has been copied to your workspace. You are operating inside this codebase directory.

You may check all the code using your tools (file_editor, terminal).

### Related Code

You may **ONLY** edit the following files:

1. **General training configuration**: ./robomimic/exps/paper_local/core/transport/ph/low_dim/bcq.json
2. **Model and loss function**: ./robomimic/algo/bcq.py

Note 1: Modify the relevant parts of the algorithm files as per the config content. For example:
```
# In ./robomimic/exps/paper_local/core/transport/ph/low_dim/bcq.json

"gmm": {
    "enabled": true,
    "num_modes": 5,
    "min_std": 0.0001,
    "std_activation": "softplus",
    "low_noise_eval": true
},
```
This means the exeriment uses 'gmm' model, and you should modify Class 'BC_GMM' in ./robomimic/algo/bcq.py. Please refer to bcq.py for its connection to bcq.json.
.

Note 2: If you want to add new features in the algo/{algorithm_name}.py (e.g. algo/bc.py) file, please directly set corresponding configuration parameters inside the .py, rather than add config in the {algorithm_name}.json and call the attributes in {algorithm_name}.py, which may fail.
Note 3: When you want to introduce a new variable, make sure that it has been defined before called.
-----

### ⚠️ CRITICAL: How to Make Changes

**USE YOUR `file_editor` TOOL TO DIRECTLY MODIFY FILES!**

You have access to the following tools:
- `file_editor` - Use this to view and edit files directly
- `terminal` - Use this to run commands and inspect the codebase
- `task_tracker` - Use this to track your progress

**DO NOT** write Python scripts to modify files. Instead:

1. **Use `file_editor` to view the current code**
2. **Use `file_editor` to make your modifications directly**
3. **The system will automatically run training after you're done**

-----

### Workflow Example

**Step 1: Inspect the current code**
```
Use file_editor to view ./robomimic/algo/bcq.py and ./robomimic/exps/paper_local/core/transport/ph/low_dim/bcq.json  (just example)
```

**Step 2: Make your improvements**
```
Use file_editor to modify specific sections: (just examples)
- Change learning rate from 1e-4 to 5e-5 in bcq.json
- Modify loss calculation bcq.py
- Or other modifications
```

**Step 3: Verify your changes**
```
Use file_editor to review the modified files
```

**Step 4: Done!**
The K8S system will automatically execute:
```bash
python robomimic/scripts/train.py --config robomimic/exps/paper/core/transport/ph/low_dim/bcq.json
```

-----

### What You Should Do

✅ **DO:**
- Use `file_editor` to read files in `./codebase/`
- Use `file_editor` to modify the allowed files directly
- Use `terminal` to explore the codebase structure
- Explain your changes and rationale

❌ **DO NOT:**
- Write Python scripts that modify files programmatically
- Import torch, tensorflow, or create training loops
- Call training scripts manually (the system does this automatically)
- Create new files outside the allowed list

-----

### Improvement Ideas to Consider (just for reference, decide by the actual situation)

1. **Hyperparameter Tuning:**
- Learning rate: Try values like 5e-5, 1e-5, or learning rate scheduling
- Batch size: Adjust based on memory constraints

2. **Loss Design:**
- Temporal smoothness loss: Add a penalty on differences between consecutive actions to encourage smooth control.
- Consistency regularization: Encourage predicted action changes to align with the ground-truth action changes across timesteps.
- Multi-task auxiliary losses: Add extra heads (e.g., next-state prediction, success probability, contact state) and train them jointly with the action loss.
- Distribution regularization (Gaussian / GMM): Add entropy or std regularization so the policy is neither too over-confident nor too noisy.

3. **Algorithmic Optimization:**
- GMM action sampling: For GMM policies (BC_GMM, BC_RNN_GMM, BC_Transformer_GMM), expose a clear choice between sampling, taking the mean, or taking the MAP mode, and respect low_noise_eval for more stable evaluation behavior.
- Dropout tuning
- Layer normalization adjustments

-----

### Deliverables

After making your modifications, please provide:

1. **Change Summary:** List all files modified and what was changed
2. **Rationale:** Explain why each modification was made
3. **Expected Impact:** Describe what improvement you expect

-----

### Example Change Summary

```
## Changes Made

### 1. Modified ${The first Modified file name}
- Line ${line number}: ${The change you made}
- Rationale: ${The reason}

### 2. Modified ${The second Modified file name}
- Line ${line number}: ${The change you made}
- Rationale: ${The reason}

...

### Expected Impact
${The expected impacts}
```
