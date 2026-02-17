# RoboTwin place_phone_stand Imitation Learning PRD Docs

## Role & Objectives

**Role:** You are a Senior Robotics Learning Engineer and System Optimization Expert familiar with **RoboTwin**, **Robotics Diffusion Transformer (RDT)**, and **Imitation Learning (IL)**.

**Goal:** Maximize the success rates of the `place phone stand` tasks on the RoboTwin platform, and implement structured, reusable, and scalable automated improvements to the existing code. This task leverages a large-scale dataset of 1000 expert demonstrations.

### Ground Rules

  * **No Data Addition:** You may only process and augment the existing 1000 demonstrations. You cannot add external data.
  * **Evaluation Metrics are Immutable:** You must use the native RoboTwin success check mechanisms.
  * **Fixed Training Budget:** Uniformly use a fixed number of training epochs; you **cannot** modify the quantity of epochs. The training time upperbound is 4 hours for this large-scale dataset.
  * **Restricted File Modification:** You are only allowed to modify the specified files.  * **Immutable Execution Script:** The training and evaluation workflow is controlled by `policy/RDT/train_and_eval_rdt_place_phone_stand.sh`. This script is **READ-ONLY** and must not be modified.



## CRITICAL: Technical Robustness & Common Pitfalls

To avoid common execution failures, you **must** adhere to the following technical constraints:

1. **Time Limit Awareness**:
   With 1000 episodes, training is computationally intensive. Avoid overly complex architectures or extremely slow data augmentation that might exceed the **4-hour** timeout.

2. **Finetuning Instructions: MANDATORY LoRA Implementation**

   **Core Strategy: Parameter-Efficient Fine-Tuning (PEFT) via LoRA**
   The current codebase likely supports full fine-tuning or simple freezing. You are **REQUIRED** to modify the training script to implement Low-Rank Adaptation (LoRA). **Do not assume LoRA is already implemented.**

   **Implementation Requirements (You MUST write code for this):**

   * **Import `peft`:** In `train.py` (or the model loading script), import `get_peft_model` and `LoraConfig`.
   * **Target Modules:** You must inspect the RDT model structure (using `print(model)`) to identify the correct Linear layer names (e.g., `q_proj`, `k_proj`, `v_proj`, `out_proj`, or `linear`). **Do not guess; verify layer names first.**
   * **Wrap the Model:**
     * Create a `LoraConfig` (Rank $r=16$ or $32$, $\alpha=32$, Dropout $0.05$).
     * Apply `model = get_peft_model(model, peft_config)` **before** passing parameters to the optimizer.
     * Ensure the optimizer only optimizes `model.parameters()` (which will now only include LoRA weights and un-frozen layers).
   * **Inference Compatibility:** You **must** ensure the model loading logic (e.g., in a `Policy` class or `eval.sh` workflow) correctly loads the LoRA adapters for inference. Verify if `model.load_state_dict` or a specialized PEFT loading method is required.
   * **Config Synchronization:** If you add LoRA parameters (e.g., `lora_rank`) to the `.yml` config file, you must ensure the Python script actually reads these keys.

   * **Memory Optimization:**
     * Enable **Gradient Checkpointing** on the RDT backbone if VRAM is tight (~22GB limit).
     * Use a higher learning rate for adapters ($1e-4$ to $5e-4$) compared to full fine-tuning.



## CodeBase Path

The codebase is located at `/data/agents/openhands-ml-master/embodied-benchmarks-code-repos/robotwin/`.

You may check all the code using your tools (read_file, grep, codebase_search).

### Related Code

You may **ONLY** edit the following files or files in following path:

1. **Policy**: `./policy/DP/RDT/models` (or wherever the model definition/loading logic resides)
2. **Config file**: `./policy/RDT/model_config/rdt_place_phone.yml`
3. **Dataset**: `./policy/RDT/training_data/rdt_place_phone`
4. **Training Script**: `./policy/RDT/train/train.py` (Crucial for injecting LoRA logic)



## CRITICAL: How to Make Changes

**YOU MUST USE THE PROVIDED TOOLS TO DIRECTLY MODIFY FILES!**

#### Tool Usage Principles:

1. **View first**: Use `read_file` to read the file before attempting any edits.
2. **Exact match**: When using `search_replace`, the `old_string` must match the file content **exactly**, including all whitespace, indentation, and newlines.
3. **Direct Edit**: **DO NOT** write Python scripts to modify files. Use the `search_replace` or `write` tools directly.



## Workflow Example

**Step 1: Inspect Code & Model Structure**

1. 

   Read ./policy/RDT/model_config/rdt_place_phone.yml Read ./policy/RDT/train/train.py

   ```
   *Crucial:* Run a quick python snippet to print the model architecture. You need this to know which layers to target for LoRA (e.g., is it `model.transformer.blocks[0].attn.q_proj`?).
   
   **Step 2: Implement LoRA Logic & Inference Adaptation**
   
   1.  Modify `rdt_place_phone.yml` to include LoRA params (rank, alpha).
   2.  Modify `train.py` to:
       * Load these params, wrap model with `get_peft_model`.
       * **Crucial:** Ensure weights are saved in a format that includes the LoRA adapter (e.g., `model.save_pretrained`).
   3.  **Iterate on Inference Code (e.g., `agilex_model.py`):**
       * Update the loading logic to detect if the checkpoint is a LoRA adapter.
       * Use `PeftModel.from_pretrained` or equivalent to load weights. **Failure to do this will result in weight mismatch errors during evaluation.**
   
   **Step 2.5: Update progress & Verify Code Alignment**
   
   Use `task_tracker` to record milestones. Ensure that for every change in `train.py` (saving logic), there is a corresponding change in the inference scripts (loading logic).
   
   **Step 3: ⚠️ CRITICAL - Test your changes with `debug_test` tool**
   
   You **MUST** use `debug_test` to verify your changes.
   
   - Use **only** `place_phone_stand-demo_clean-10.zarr` (10 trajectories).
   - **Do not** use `demo_clean_1000`.
   
   #### Required debug sequence
   
   1. **Import sanity check:**
   ```python
   debug_test(
       command='python -c "import peft; from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy; print(\'Import OK\')"',
       timeout=30,
       working_dir="policy/RDT"
   )
   ```

   1. **LoRA Verification (MUST PASS):** Verify that the model is actually using LoRA (small number of trainable params).

   Python

   ```
   debug_test(
       # NOTE: You may need to adjust the import path to match the actual train.py structure
       command='python -c "from train import model; print(f\'Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\')"',
       timeout=60,
       working_dir="policy/RDT"
   )
   ```

   1. **Mandatory training check:**

   Python

   ```
   debug_test(
       command="bash finetune.sh rdt_place_phone",
       timeout=180,
       working_dir="policy/RDT"
   )
   ```

   1. **Inference Verification (MUST PASS):** Verify that the evaluation script successfully loads the model with LoRA adapters and can run a forward pass without OOM or weight mismatch errors.

   Python

   ```
   debug_test(
       command="python -c \"import torch; from train import model; x = torch.randn(1, ...); y = model(x); print('Inference OK')\"",
       timeout=60,
       working_dir="policy/RDT"
   )
   ```

   **Step 4: Execute Full Training Pipeline** After your changes pass the debug tests, the K8S system will automatically execute:

   Bash

   ```
   cd /workspace/RoboTwin/policy/DP
   # Training with 1000 episodes
   bash finetune.sh rdt_place_phone
   # Evaluation
   bash eval.sh place_phone_stand demo_clean_1000 demo_clean_1000 1000 0 0 30
   ```

   1. **Algorithmic Optimization (RDT-LoRA specific):**
      - **Architecture (Transformer-based):**
        - **Do NOT** look for "down_dims" or "kernel_size" (these are UNet concepts). RDT is a Diffusion Transformer (DiT).
        - The Backbone is frozen. Focus on optimizing the **Adapter** (LoRA) and **Solver**.
      - **LoRA Hyperparameters:**
        - Tune `lora_alpha`: A higher alpha (e.g., 2x rank) often stabilizes training.
        - Tune `lora_dropout`: Increase if you observe overfitting on the validation set.
      - **Inference Reliability:**
        - Ensure that `eval.sh` (which calls the inference script) is using the weights from the LoRA adapters, not just the base model.
      - **Noise scheduler:**
        - Tune `num_inference_steps` to balance precision and speed.
        - Constraint: `num_inference_steps <= train_timesteps`.

   ------

   ### Task Specifics: `place_phone_stand`

   - The task and environment is defined in `envs/place_phone_stand.py`.

   ### ⚠️ CRITICAL: GPU Memory Constraints

   - **GPU VRAM**: Limited to ~22GB (8 * RTX 4090)
   - **Strategy**:
     - Since RDT is large, you **MUST** use LoRA.
     - If OOM occurs, enable **Gradient Checkpointing** in your LoRA implementation or reduce `batch_size` in the YAML config (and proportionally increase `gradient_accumulation_steps` if available).

   ------

   ### Deliverables

   After making your modifications, please provide:

   1. **Change Summary**: List all files modified. **Explicitly confirm that inference scripts (e.g., agilex_model.py) have been updated to support LoRA loading.**
   2. **LoRA Details**: Specifically state the Rank, Alpha, and Target Modules used.
   3. **Cross-Verification**: Confirm that the saved adapter from training can be successfully loaded by the evaluation script without weight mismatches.