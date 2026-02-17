# RoboTwin put_object_cabinet Imitation Learning PRD Docs

## RoboTwin + DP (Diffusion Policy) Automated Improvement Prompt

### Role & Objectives

**Role:** You are a Senior Robotics Learning Engineer and System Optimization Expert familiar with **RoboTwin**, **DP (Diffusion Policy)**, and **Imitation Learning (IL)**.

**Goal:** Maximize the success rates of the `put_object_cabinet` tasks on the RoboTwin platform, and implement structured, reusable, and scalable automated improvements to the existing DP code. This task leverages a large-scale dataset of 1000 expert demonstrations.

### Ground Rules

  * **No Data Addition:** You may only process and augment the existing 1000 demonstrations. You cannot add external data.
  * **Evaluation Metrics are Immutable:** You must use the native RoboTwin success check mechanisms.
  * **Fixed Training Budget:** Uniformly use a fixed number of training epochs; you **cannot** modify the quantity of epochs. The training time upperbound is 4 hours for this large-scale dataset.
  * **Restricted File Modification:** You are only allowed to modify the specified files.

-----

### ⚠️ CRITICAL: Technical Robustness & Common Pitfalls

To avoid common execution failures, you **must** adhere to the following technical constraints:

1.  **Diffusion Parameter Consistency**: 
    - `num_inference_steps` **MUST** be less than or equal to `train_timesteps` (typically 100 or 1000). Never hardcode an inference step count that exceeds the training timesteps.
2.  **U-Net Architecture Alignment**:
    - If you modify `down_dims` or `kernel_size`, ensure the Tensor shapes align at skip connections. 
    - **Recommendation**: Keep the `horizon` (prediction horizon) as a power of 2 (e.g., 16, 32) to avoid dimension mismatches (like 8 vs 7) during upsampling.
3.  **Data Augmentation Safety**:
    - In `CropRandomizer`, the `crop_height` and `crop_width` **MUST** be strictly smaller than the input image dimensions (e.g., if input is 84x84, crop should be 76x76, NOT 84x84 or larger).
4.  **Valid Model References**:
    - Only use standard models from `torchvision.models` (e.g., `resnet18`, `resnet34`). Do **NOT** attempt to use non-existent models like `resnet10`.
5.  **Time Limit Awareness**:
    - With 1000 episodes, training is computationally intensive. Avoid overly complex architectures or extremely slow data augmentation that might exceed the **4-hour** timeout.

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

**Step 2.5: Update progress with `task_tracker`**

Use `task_tracker` to record key milestones (e.g., inspected files, applied edits, debug test passed) so progress is visible during execution.

**Step 3: ⚠️ CRITICAL - Test your changes with `debug_test` tool**

After making code modifications, you **MUST** use `debug_test` to verify your changes before submitting for full training. You must confirm the training script reaches the actual training stage (not just data loading).

#### Purpose
`debug_test` runs your command in the same K8S environment (GPU, conda, dependencies) as full training, so you can verify the training loop actually runs.

#### Required dataset
- Use **only** `put_object_cabinet-demo_clean-10.zarr` (10 trajectories). It is fast and matches the real data format.
- **Do not** use `demo_clean 1000`, and **do not** create your own test data.

#### Required debug sequence
1. **Import sanity check:**
```python
debug_test(
    command='python -c "from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy; print(\'Import OK\')"',
    timeout=30,
    working_dir="policy/DP"
)
```
2. **Mandatory training check (must reach training stage):**
```python
debug_test(
    command="bash train.sh put_object_cabinet demo_clean 10 0 14 0",
    timeout=180,
    working_dir="policy/DP"
)
```
**Pass criteria:**
- Shows training progress (e.g., `Training epoch 0:`)
- Runs at least one training step (not just data loading)

#### Optional deeper check
```python
debug_test(
    command="bash train.sh put_object_cabinet demo_clean 50 0 14 0",
    timeout=600,
    working_dir="policy/DP"
)
```

#### Wrong uses (do not do these)
- Large datasets (`demo_clean 1000`)
- Custom test data
- Treating timeout as success

**Step 4: Execute Full Training Pipeline**
After your changes pass the debug tests, the K8S system will automatically execute:
```bash
cd /workspace/RoboTwin/policy/DP
# Training with 1000 episodes
bash train.sh put_object_cabinet demo_clean 1000 1000 0 14 0
# Evaluation
bash eval.sh put_object_cabinet demo_clean demo_clean 1000 0 0 30
```
4. **Algorithmic Optimization (DP-specific):**
   - **Diffusion architecture:**
     - Adjust `down_dims` or `kernel_size` for complex bimanual coordination. **Ensure shape alignment at skip connections.**
     - Change `diffusion_step_embed_dim`.
   - **Noise scheduler:**
     - Try different schedules (linear vs squaredcos).
     - Tune `num_inference_steps` to balance precision and speed. **Constraint: `num_inference_steps <= train_timesteps`.**

-----

### Task Specifics: `put_object_cabinet`
- **Complexity**: Bimanual coordination task requiring opening a drawer and placing an object inside.
- **Action Space**: 14-dim (Dual arm setup).
- **Key Challenges**:
  - Simultaneous control of two arms: one holding the drawer handle, the other holding the object.
  - Multi-stage sequence: Grasp both -> Pull drawer -> Lift object -> Place object.
  - Object diversity: 10 different types of objects are used.

### ⚠️ CRITICAL: GPU Memory Constraints
- **GPU VRAM**: Limited to ~22GB (RTX 4090)
- **Current Issue**: Training fails with `CUDA out of memory` errors
- **Root Cause**: Default batch_size=128 + model size (~220M parameters) + image encoder (ResNet18) exceeds available VRAM


-----

### Deliverables

After making your modifications, please provide:

1. **Change Summary**: List all files modified and what was changed.
2. **Rationale**: Explain why each modification was made.
3. **Expected Impact**: Describe what improvement you expect in success rate or convergence.
