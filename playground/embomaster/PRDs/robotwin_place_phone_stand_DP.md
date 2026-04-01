# RoboTwin place_phone_stand Imitation Learning PRD Docs

## RoboTwin + DP (Diffusion Policy) Automated Improvement Prompt

### Role & Objectives

**Role:** You are a Senior Robotics Learning Engineer and System Optimization Expert familiar with **RoboTwin**, **DP (Diffusion Policy)**, and **Imitation Learning (IL)**.

**Goal:** Maximize the success rates of the `place_phone_stand` tasks on the RoboTwin platform, and implement structured, reusable, and scalable automated improvements to the existing DP code. This task leverages a large-scale dataset of 1000 expert demonstrations.

**Optimization Protocol:** Improvements must be reproducible and delivered in three distinct stages:

1. **Hyperparameter Tuning** (no changes to data scale)
2. **Data Scheduling / Data Augmentation / Loss Design**
3. **Algorithmic Optimization** (DP-specific)

-----

### Ground Rules

* **No Data Addition:** You may only process and augment the existing 1000 demonstrations. You cannot add external data.
* **Evaluation Metrics are Immutable:** You must use the native RoboTwin success check mechanisms.
* **Fixed Training Budget:** Uniformly use a fixed number of training epochs; you **cannot** modify the quantity of epochs. The training time upperbound is 4 hours for this large-scale dataset.
* **Restricted File Modification:** You are only allowed to modify the specified files.

-----

### Technical Robustness & Common Pitfalls

To avoid common execution failures, you **must** adhere to the following technical constraints:

1. **Diffusion Parameter Consistency:**
   - `num_inference_steps` **MUST** be less than or equal to `train_timesteps`.
   - Never hardcode an inference step count that exceeds the training timesteps.
2. **U-Net Architecture Alignment:**
   - If you modify `down_dims` or `kernel_size`, ensure tensor shapes remain aligned at skip connections.
   - Prefer keeping `horizon` as a power of 2 (for example 16 or 32) to reduce upsampling mismatch risk.
3. **Data Augmentation Safety:**
   - In `CropRandomizer`, `crop_height` and `crop_width` **MUST** be strictly smaller than the input image dimensions.
4. **Valid Model References:**
   - Only use standard models that actually exist in the current dependency set.
   - Do **not** reference unsupported backbones or unavailable imports.
5. **Time Limit Awareness:**
   - With 1000 episodes, training is computationally intensive. Avoid overly complex architecture changes or very slow augmentation that may exceed the 4-hour timeout.

-----

### CodeBase Path

The codebase has been copied to your workspace. You are operating inside this codebase directory.

You may inspect the code using your available tools such as `file_editor` and `terminal`.

### Related Code

You may **ONLY** edit the following files:

1. **Policy:** `./policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py`
2. **Training workspace:** `./policy/DP/diffusion_policy/workspace/robotworkspace.py`
3. **Config file:** `./policy/DP/diffusion_policy/config/robot_dp_14.yaml`
4. **Dataset:** `./policy/DP/diffusion_policy/dataset/robot_image_dataset.py`
5. **Env code (optional):** `./envs/place_phone_stand.py`

-----

### How to Make Changes

**USE YOUR `file_editor` TOOL TO DIRECTLY MODIFY FILES.**

You have access to the following tools:
- `file_editor` - Use this to view and edit files directly
- `terminal` - Use this to inspect the codebase structure and confirm assumptions
- `task_tracker` - Use this to track progress and milestones

**DO NOT** write Python scripts to modify files programmatically.

Instead:

1. **Use `file_editor` to read the allowed files first**
2. **Use `file_editor` to make focused edits directly**
3. **Use `terminal` only for inspection and context gathering**
4. **Do not manually launch training or evaluation scripts; rely on the provided pipeline**

-----

### Workflow Example

**Step 1: Inspect the current code**
```text
Use file_editor to view:
- ./policy/DP/diffusion_policy/config/robot_dp_14.yaml
- ./policy/DP/diffusion_policy/workspace/robotworkspace.py
- ./policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py
```

**Step 2: Make your improvements**
```text
Modify only the allowed files based on your optimization strategy.
Keep diffusion parameters internally consistent.
Keep image crop sizes and tensor shapes valid.
```

**Step 3: Verify your changes**
```text
Use file_editor to review the modified files.
Use terminal to inspect nearby code paths if needed.
Record milestones with task_tracker.
```

**Step 4: Done**

The system will automatically execute the standard `place_phone_stand` DP training and evaluation pipeline in the RoboTwin workspace after your edits are complete. Do not invoke the train or eval scripts manually.

-----

### What You Should Do

**DO:**
- Use `file_editor` to read files in the workspace codebase
- Modify only the allowed files directly
- Use `terminal` to inspect structure, configs, and nearby call sites
- Keep tensor shapes, diffusion steps, and image preprocessing settings coherent
- Explain your changes and rationale clearly

**DO NOT:**
- Write Python scripts that modify files programmatically
- Call training scripts manually
- Create new files outside the allowed list unless absolutely necessary
- Introduce unsupported model names, invalid crop sizes, or inconsistent diffusion settings
- Assume outdated absolute paths or external codebase locations still apply

-----

### Improvement Ideas to Consider

1. **Hyperparameter Tuning:**
   - Adjust learning rate, optimizer settings, or scheduling for more stable convergence on the 1000-demo dataset.
   - Rebalance batch size and accumulation-related settings if memory pressure is high.
2. **Data Augmentation:**
   - Add moderate image augmentation to improve robustness across camera viewpoints and object variants.
   - Inject small state noise or temporal perturbation only if it remains consistent with task dynamics.
3. **Loss Design:**
   - Add temporal smoothness or consistency regularization if the policy produces unstable action sequences.
   - Reweight losses only if the change is simple, defensible, and does not destabilize training.
4. **Algorithmic Optimization (DP-specific):**
   - Tune `down_dims`, `kernel_size`, or diffusion embedding dimensions carefully.
   - Adjust the diffusion scheduler and `num_inference_steps` under the constraint `num_inference_steps <= train_timesteps`.

-----

### Task Specifics: `place_phone_stand`

- **Complexity:** Single-arm placement task requiring precise phone-to-stand alignment.
- **Action Space:** 14-dim.
- **Key Challenges:**
  - The robot must align the phone with the stand accurately enough for a valid placement.
  - Camera observations must be used to infer pose and alignment from multiple viewpoints.
  - The task contains object variation across multiple phone and stand instances.

### GPU Memory Constraints

- **GPU VRAM:** Limited to about 22GB.
- **Current Risk:** Training may fail with CUDA OOM if batch size, image processing, and model footprint are too aggressive.
- **Practical Direction:** Prefer changes that improve stability or efficiency without substantially increasing memory usage.

-----

### Deliverables

After making your modifications, please provide:

1. **Change Summary:** List all files modified and what changed.
2. **Rationale:** Explain why each modification was made.
3. **Expected Impact:** Describe what improvement you expect in stability, convergence, or success rate.
