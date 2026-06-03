# Reward Module Design Brief (ManiSkill3: StackCube-v4)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves stacking three cubes (A=red, B=green, C=blue) on the same layer (table height plane) aligned along the world X direction in a 1x3 horizontal configuration, with adjacent cubes being "edge-touching" (center distance ≈ 2*half_x). The final state should be stable, with the robot not grasping any cube.

### Environment Facts
- Class: `StackCubeEnvV4`
- Robot description: Supports robots "panda_wristcam", "panda", and "fetch".
- Episode length: 50 steps
- Randomization:
  - Initial positions of cubes are randomized within a specified region.
  - Randomized initial robot joint positions with noise.
- Goal & Success (already computed in `evaluate()`):
  - Cubes must be on the same layer (minimum z-plane).
  - Cubes must be aligned along the Y-axis.
  - Adjacent cubes must be edge-touching along the X-axis.
  - All cubes must be static and not grasped by the robot.
- Useful tensors each step (batch size = B):
  - `tcp_pos`: Position of the robot's tool center point.
  - `A_pos`, `B_pos`, `C_pos`: Positions of cubes A, B, and C.
  - `half_x`: Half size of the cube along the X-axis.
  - `info`: Dictionary containing evaluation results such as `same_layer`, `y_aligned`, `touch_left_mid`, `touch_mid_right`, `success`, etc.

### Interfaces You Must Implement
Inside the class `StackCubeEnvV4`, check the following methods:
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
1. Avoid shortcuts (e.g., vibrating near the handle forever, or opening then oscillating).
2. Be smooth enough for policy gradient methods (no discontinuous spikes except final success).
3. Encourage the robot to approach cubes, align them, open the gripper, and stabilize the cubes.
4. Reward should progressively increase as the robot achieves sub-goals leading to the final success state.

### Deliverables
- Implementations of the two methods.
- A brief design note (docstring or comment) explaining your reward rationale (how it encourages approach → align → open → stabilize).
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
4. Gradual increase in reward as cubes approach the desired configuration, ensuring smooth learning progression.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/stack_cube_v4.py
2.  ./examples/baselines/ppo/ppo_fast.py
