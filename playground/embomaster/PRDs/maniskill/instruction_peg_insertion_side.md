# Reward Module Design Brief (ManiSkill3: PegInsertionSide-v1)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves picking up an orange-white peg and inserting the orange end into a box with a hole.

### Environment Facts
- Class: PegInsertionSideEnv
- Robot description: PandaWristCam, a robotic arm with a wrist camera.
- Episode length: 100 steps
- Randomization:
  - Peg half length is randomized between 0.085 and 0.125 meters. Box half length is the same value.
  - Peg radius/half-width is randomized between 0.015 and 0.025 meters. Box hole's radius is the same value + 0.003m of clearance.
  - Peg is laid flat on the table with its xy position and z-axis rotation randomized.
  - Box is laid flat on the table with its xy position and z-axis rotation randomized.
- Goal & Success (already computed in `evaluate()`):
  - Success is achieved when the white end of the peg is within 0.015m of the center of the box.
- Useful tensors each step (batch size = B):
  - `self.peg_head_pos`: Position of the peg head.
  - `self.peg_head_pose`: Pose of the peg head.
  - `self.box_hole_pose`: Pose of the box hole.
  - `self.goal_pose`: Target pose for the peg head relative to the box hole.
  - `self.peg_half_sizes`: Dimensions of the peg.
  - `self.box_hole_radii`: Radius of the box hole.

### Interfaces You Must Implement
Inside the class `PegInsertionSideEnv`, check the following methods:
```python
def compute_dense_reward(obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
    ## Return a 1D torch tensor of shape [B] with the dense reward for each env.

def compute_normalized_dense_reward(obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
    ## Return the dense reward normalized to [0, 1] with a consistent max.
```
If a method body is unimplemented (e.g., 'pass'), then add and complete its implementation according to this instruction.
If the method already has valid content, then optimize the existing implementation according to this instruction.

#### Non-Negotiable Constraints
- Vectorized over batch `B` (no Python loops).
- Torch-only; device-safe (use tensors on `self.device`).
- Numerically robust (avoid div-by-zero, clamp where needed).
- Deterministic given inputs.
- Keep a single scalar constant `MAX_REWARD` for normalization.

### Reward Design Goals
1. Encourage smooth progression from approaching the peg, aligning the gripper, grasping the peg, orienting it towards the hole, and finally inserting it.
2. Avoid shortcuts such as oscillating near the peg or box without meaningful progress.
3. Ensure the reward is smooth enough for policy gradient methods, with no discontinuous spikes except for final success.
4. Reward should increase as the peg gets closer to being correctly inserted, with a significant reward boost upon successful insertion.

### Deliverables
- Implementations of the two methods.
- A brief design note (docstring or comment) explaining your reward rationale (how it encourages approach → align → grasp → orient → insert).
- The property test above passes without editing it.

---
### Property-Based Evaluation (must pass)
Write the reward so these properties are true; the exact numbers are up to you.
1. Ordering
  - `reward(success=True) > reward(any non-success state)` by a clear margin.
2. Normalization
  - `compute_normalized_dense_reward outputs` ∈ `[0, 1]`; at success it returns exactly 1.0.
3. Boundedness
  - `compute_dense_reward` is bounded above by a constant `MAX_REWARD` and below by a finite value (no `inf/-inf` or NaNs).
4. Progression
  - Reward should smoothly increase as the peg transitions from being approached, grasped, oriented, and inserted.
5. Robustness
  - Reward should be robust to variations in peg and box positions due to randomization.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.

### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/peg_insertion_side.py
2.  ./examples/baselines/ppo/ppo_fast.py
