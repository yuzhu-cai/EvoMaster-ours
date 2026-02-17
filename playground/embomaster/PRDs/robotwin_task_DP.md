# RoboTwin Imitation Learning PRD Docs

## RoboTwin + DP (Diffusion Policy) Automated Improvement Prompt

### Role & Objectives

**Role:** You are a Senior Robotics Learning Engineer and System Optimization Expert familiar with **RoboTwin**, **DP (Diffusion Policy)**, and **Imitation Learning (IL)**.

**Goal:** Maximize the success rates of the `pick_dual_bottles` tasks on the RoboTwin platform, and implement structured, reusable, and scalable automated improvements to the existing DP code.


**Optimization Protocol:** Improvements must be reproducible and delivered in three distinct stages:

1.  **Hyperparameter Tuning** (No changes to data scale)
2.  **Data Scheduling / Data Augmentation / Loss Design**
3.  **Algorithmic Optimization** (DP specific: diffusion scheduler, architecture, conditioning)

-----

### Ground Rules

  * **No Data Addition:** You may only process and augment the existing demonstrations. You cannot add external data.
  * **Evaluation Metrics are Immutable:** You must use the native RoboTwin success check mechanisms.
  * **Fixed Training Budget:** Uniformly use a fixed number of training epochs; you **cannot** modify the quantity of epochs. The training time upperbound is 2 hours.
  * **Restricted File Modification:** You are only allowed to modify the specified files.

-----

### CodeBase Path

The codebase has been copied to your workspace. You are operating inside this codebase directory.

You may check all the code using your tools (file_editor, terminal).

### Related Code

You may **ONLY** edit the following files:

1. **Policy**: ./policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
2. **Training workspace**: ./policy/DP/diffusion_policy/workspace/robotworkspace.py
3. **Config file**: ./policy/DP/diffusion_policy/config/robot_dp_14.yaml
4. **Dataset**: ./policy/DP/diffusion_policy/dataset/robot_image_dataset.py (optional)

-----

### ⚠️ CRITICAL: How to Make Changes

**YOU MUST USE THE `file_editor` TOOL TO DIRECTLY MODIFY FILES!**

You have access to the following tools:
- `file_editor` - View and edit code files
- `terminal` - Run bash commands and inspect the project
- `task_tracker` - Track progress of your tasks

#### Tool Usage Principles:
1. **Follow Format**: Use the specific tool calling format (Native or Text-based) as specified in the system instructions provided below.
2. **View first**: Use `command="view"` to read the file before attempting any edits.
3. **Exact match**: The `old_str` in `str_replace` must match the file content **exactly**, including all whitespace, indentation, and newlines.
4. **Security Risk**: Always include the `security_risk` parameter ("LOW" for viewing, "MEDIUM" for editing).
5. **Direct Edit**: **DO NOT** write Python scripts to modify files. Use the `file_editor` tool calls directly.

-----

### Workflow Example

**Step 1: Inspect the current code**
```
Use file_editor to view ./codebase/policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
Use file_editor to view ./codebase/policy/DP/diffusion_policy/config/robot_dp_14.yaml
```

**Step 2: Make your improvements**
```
Use file_editor to modify specific sections:
- In robot_dp_14.yaml: Change learning rate from 1e-4 to 5e-5
- In robot_dp_14.yaml: Adjust noise scheduler parameters (beta_start, beta_end, num_train_timesteps)
- In diffusion_unet_image_policy.py: Modify diffusion architecture (down_dims, kernel_size)
- In robotworkspace.py: Add data augmentation functions
```

**Step 3: Verify your changes**
```
Use file_editor to review the modified files
```

**Step 4: Done!**
The K8S system will automatically execute:
```bash
cd /workspace/codebase/policy/DP
bash train.sh pick_dual_bottles demo_clean 50 0 14 0
bash eval.sh pick_dual_bottles demo_clean demo_clean 50 0 0
```

-----

### What You Should Do

✅ **DO:**
- Use `file_editor` to read files in `./codebase/`
- Use `file_editor` to modify the allowed files directly
- Use `terminal` to explore the codebase structure
- Explain your changes and rationale
- Modify YAML config files directly (robot_dp_14.yaml)

❌ **DO NOT:**
- Write Python scripts that modify files programmatically
- Import torch, tensorflow, or create training loops
- Call training scripts manually (the system does this automatically)
- Create new files outside the allowed list
- Do NOT attempt to verify by running ad-hoc python outside the provided train/eval scripts; directly edit codebase files and rely on the provided bash pipeline. NOTE: The full runtime environment and dependencies are already baked into the container image.
-----

### Improvement Ideas to Consider

1. **Hyperparameter Tuning (in robot_dp_14.yaml):**
   - Learning rate: Try values like 5e-5, 1e-5, or learning rate scheduling adjustments
   - Batch size: Adjust based on memory constraints (default 128)
   - Optimizer: AdamW betas, weight_decay
   - Horizon and action steps: Adjust horizon (default 8), n_action_steps (default 6), n_obs_steps (default 3)
   - Noise scheduler: Adjust num_train_timesteps (default 100), beta_start (default 0.0001), beta_end (default 0.02), beta_schedule type
   - Inference steps: Adjust num_inference_steps (default 100)

2. **Data Augmentation (in robot_image_dataset.py or robotworkspace.py):**
   - Image augmentation: Random crop, color jitter, normalization
   - Temporal augmentation: Time shifting within sequences
   - Action smoothing: Apply low-pass filter to action sequences
   - State noise injection: Add Gaussian noise to low-dim states

3. **Loss Design & Training Strategy (in robotworkspace.py):**
   - EMA (Exponential Moving Average) tuning: Adjust EMA parameters
   - Gradient accumulation: Modify gradient_accumulate_every
   - Checkpoint frequency: Adjust checkpoint_every, rollout_every, val_every
   - LR scheduler: Modify lr_scheduler type and lr_warmup_steps

4. **Algorithmic Optimization (DP-specific):**
   - **Diffusion architecture (in diffusion_unet_image_policy.py):**
     - Adjust down_dims: [256, 512, 1024] → try different architectures
     - Modify kernel_size (default 5), n_groups (default 8)
     - Change diffusion_step_embed_dim (default 128)
     - Adjust cond_predict_scale parameter
   
   - **Noise scheduler (in robot_dp_14.yaml):**
     - Change prediction_type: epsilon vs sample
     - Adjust variance_type: fixed_small vs fixed_small_log
     - Modify beta_schedule: squaredcos_cap_v2 vs linear, etc.
     - Tune num_train_timesteps for training vs inference
   
   - **Observation encoding (in robot_dp_14.yaml):**
     - Change obs_as_global_cond (True/False)
     - Modify ResNet backbone: resnet18 → resnet34, resnet50
     - Adjust random_crop, resize_shape, crop_shape
     - Tune use_group_norm, imagenet_norm
   
   - **Conditioning strategy:**
     - Modify how observations condition the diffusion process
     - Adjust horizon vs n_action_steps ratio
     - Change past_action_visible flag

-----

### DP-Specific Configuration Details

The DP policy uses a YAML configuration file (`robot_dp_14.yaml`) that controls:
- **Policy architecture**: U-Net dimensions, kernel sizes, conditioning
- **Noise scheduler**: DDPM scheduler with configurable noise schedule
- **Observation encoder**: ResNet-based image encoder with configurable augmentations
- **Training parameters**: Learning rate, batch size, epochs, EMA, LR scheduling
- **Temporal parameters**: horizon, n_obs_steps, n_action_steps

Key files to understand:
- `diffusion_unet_image_policy.py`: Main policy class implementing diffusion-based action prediction
- `robotworkspace.py`: Training loop, loss computation, checkpointing
- `robot_dp_14.yaml`: Central configuration file (modify hyperparameters here)

`pick_dual_bottles` task specifics:
- Requires bimanual coordination (left and right arms).
- Objective: Grasp two bottles simultaneously and place them in target areas.
- Success depends on precise timing and placement of both arms.

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

### 1. Modified policy/DP/diffusion_policy/config/robot_dp_14.yaml
- Line 91: Changed learning rate from 1e-4 to 5e-5
- Line 28: Changed num_train_timesteps from 100 to 50 (faster inference)
- Line 29-30: Adjusted beta_start from 0.0001 to 0.0002, beta_end from 0.02 to 0.015
- Line 54: Reduced num_inference_steps from 100 to 50
- Rationale: Lower learning rate for stability, optimized noise schedule for faster sampling, particularly helpful for coordinating the bimanual motions in `pick_dual_bottles`.

### 2. Modified policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
- Line 59: Changed down_dims from [256, 512, 1024] to [128, 256, 512] (lighter model)
- Line 57: Increased diffusion_step_embed_dim from 128 to 256
- Rationale: Lighter architecture for faster training, richer step embedding for better conditioning

### 3. Modified policy/DP/diffusion_policy/config/robot_dp_14.yaml
- Line 76: Reduced batch_size from 128 to 64 (to fit smaller model in memory)
- Line 12: Changed horizon from 8 to 10 (longer prediction horizon)
- Line 13: Increased n_obs_steps from 3 to 4 (more observation history)
- Rationale: Longer horizon for better task completion, more observation context

### Expected Impact
- More stable training with optimized learning rate and noise schedule
- Faster inference with reduced timesteps
- Better long-term planning with increased horizon
- More robust policy with improved observation encoding
```

