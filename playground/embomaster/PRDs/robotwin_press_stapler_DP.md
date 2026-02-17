# RoboTwin press_stapler Imitation Learning PRD Docs

## RoboTwin + DP (Diffusion Policy) Automated Improvement Prompt

### Role & Objectives

**Role:** You are a Senior Robotics Learning Engineer and System Optimization Expert familiar with **RoboTwin**, **DP (Diffusion Policy)**, and **Imitation Learning (IL)**.

**Goal:** Maximize the success rates of the `press_stapler` tasks on the RoboTwin platform, and implement structured, reusable, and scalable automated improvements to the existing DP code. This task leverages a large-scale dataset of 1000 expert demonstrations.

**Optimization Protocol:** Improvements must be reproducible and delivered in three distinct stages:

1.  **Hyperparameter Tuning** (No changes to data scale)
2.  **Data Scheduling / Data Augmentation / Loss Design**
3.  **Algorithmic Optimization** (DP specific: diffusion scheduler, architecture, conditioning)

-----

### Ground Rules

  * **No Data Addition:** You may only process and augment the existing 1000 demonstrations. You cannot add external data.
  * **Evaluation Metrics are Immutable:** You must use the native RoboTwin success check mechanisms.
  * **Fixed Training Budget:** Uniformly use a fixed number of training epochs; you **cannot** modify the quantity of epochs. The training time upperbound is 4 hours for this large-scale dataset.
  * **Restricted File Modification:** You are only allowed to modify the specified files.

-----

### CodeBase Path

The codebase is located at `/data/agents/openhands-ml-master/embodied-benchmarks-code-repos/robotwin/`.

You may check all the code using your tools (read_file, grep, codebase_search).

### Related Code

You may **ONLY** edit the following files:

1. **Policy**: `./policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py`
2. **Training workspace**: `./policy/DP/diffusion_policy/workspace/robotworkspace.py`
3. **Config file**: `./policy/DP/diffusion_policy/config/robot_dp_14.yaml`
4. **Dataset**: `./policy/DP/diffusion_policy/dataset/robot_image_dataset.py`

-----

### ⚠️ CRITICAL: How to Make Changes

**YOU MUST USE THE PROVIDED TOOLS TO DIRECTLY MODIFY FILES!**

#### Tool Usage Principles:
1. **View first**: Use `read_file` to read the file before attempting any edits.
2. **Exact match**: When using `search_replace`, the `old_string` must match the file content **exactly**, including all whitespace, indentation, and newlines.
3. **Direct Edit**: **DO NOT** write Python scripts to modify files. Use the `search_replace` or `write` tools directly.

-----

### Workflow Example

**Step 1: Inspect the current code**
```
Read ./policy/DP/diffusion_policy/config/robot_dp_14.yaml
Read ./policy/DP/diffusion_policy/workspace/robotworkspace.py
```

**Step 2: Make your improvements**
```
Modify the allowed files based on your optimization strategy.
```

**Step 3: Verify your changes**
```
Check the modified files.
```

**Step 4: Execute Pipeline**
The K8S system will execute:
```bash
cd /workspace/RoboTwin/policy/DP
# Training with 1000 episodes
bash train.sh press_stapler demo_clean_1000 1000 0 14 0
# Evaluation
bash eval.sh press_stapler demo_clean_1000 demo_clean_1000 1000 0 0 30
```

-----

### Improvement Ideas to Consider

1. **Hyperparameter Tuning (for 1000 episodes):**
   - Learning rate: Try values like 5e-5, 1e-5, or schedule adjustments for larger data.
   - Batch size: Consider increasing batch size (e.g., 256) for more stable gradients with large data.
   - Horizon: Adjust prediction horizon to better capture the "press" sequence.

2. **Data Augmentation (Precision-focused):**
   - Image augmentation: Random crop, color jitter to handle lighting variations.
   - Action smoothing: Low-pass filtering for the 14-dim action space to reduce noise.
   - State noise: Inject small Gaussian noise into joint states to prevent overfitting to the 1000 trajectories.

3. **Loss Design & Training Strategy:**
   - EMA tuning: Fine-tune EMA parameters for the large-scale dataset.
   - Checkpoint frequency: Keep `checkpoint_every: 30` for regular monitoring.
   - LR scheduler: Use cosine decay or warmup for better convergence.

4. **Algorithmic Optimization (DP-specific):**
   - **Diffusion architecture:**
     - Adjust `down_dims` or `kernel_size` for more complex stapler interactions.
     - Change `diffusion_step_embed_dim`.
   - **Noise scheduler:**
     - Try different schedules (linear vs squaredcos).
     - Tune `num_inference_steps` for better precision during the "press" phase.

-----

### Task Specifics: `press_stapler`
- Requires high-precision vertical alignment and force application.
- Uses a 14-dim action space (Dual arm setup).
- Target: Reach the stapler, align the gripper/end-effector, and press down successfully.

-----

### Deliverables

After making your modifications, please provide:

1. **Change Summary:** List all files modified and what was changed.
2. **Rationale:** Explain why each modification was made (e.g., "Increased batch size to 256 to stabilize training on 1000 episodes").
3. **Expected Impact:** Describe what improvement you expect in success rate or convergence.

