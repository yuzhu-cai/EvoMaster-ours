# Reward Module Design Brief (ManiSkill3: StackCube-v7)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves stacking four cubes (A, B, C, D) vertically on a table. The cubes must be aligned and stacked in the order: A on B, B on C, C on D, with D resting on the table. The robot must not grasp any cube at the end of the task.

### Environment Facts
- Class: `StackCubeEnvV7`
- Robot description: The environment supports three types of robots: "panda_wristcam", "panda", and "fetch". These robots are used to manipulate the cubes.
- Episode length: 50 steps
- Randomization:
  - Initial positions of the cubes are randomized within a specified region on the table.
  - Initial joint positions of the robot are randomized with a noise level of 0.02.
- Goal & Success (already computed in `evaluate()`):
  - The cubes must be stacked in the correct order (A on B, B on C, C on D).
  - All cubes must be aligned in the XY plane.
  - Cube D must be on the table.
  - All cubes must be static, and the robot must not be grasping any cube.
- Useful tensors each step (batch size = B):
  - `tcp`: The position of the robot's tool center point.
  - `A`, `B`, `C`, `D`: The positions of cubes A, B, C, and D, respectively.
  - `half_z`: Half the height of a cube.
  - `is_A_grasped`, `is_B_grasped`, `is_C_grasped`, `is_D_grasped`: Boolean tensors indicating if a cube is grasped.
  - `not_grasp_any`: Boolean tensor indicating if no cube is grasped.
  - `success`: Boolean tensor indicating if the task is successfully completed.

### Interfaces You Must Implement
Inside the class `StackCubeEnvV7`, check the following methods:
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
3. Encourage progressive stacking: reward should increase as cubes are stacked correctly.
4. Penalize misalignment in the XY plane to encourage proper alignment.
5. Reward stability and non-grasping as the task progresses.

### Deliverables
- Implementations of the two methods.
- A brief design note (docstring or comment) explaining your reward rationale (how it encourages approach → align → stack → stabilize).
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
4. Progressiveness
  - Reward should progressively increase as more cubes are stacked correctly.
5. Alignment
  - Reward should penalize misalignment in the XY plane.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/stack_cube_v7.py
2.  ./examples/baselines/ppo/ppo_fast.py
