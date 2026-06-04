# Reward Module Design Brief (ManiSkill3: HumanoidStackCube-v7)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves stacking four cubes (A, B, C, D) vertically on a table. The cubes must be aligned in the XY plane and stacked in the order A on B, B on C, and C on D, with D resting on the table. The cubes must remain static, and the robot should not be grasping any cube at the end of the task.

### Environment Facts
- Class: `HumanoidStackCubeEnvV7`
- Robot description: The environment uses a robot model `unitree_g1_simplified_upper_body_with_head_camera`, which includes an upper body with a head camera.
- Episode length: 50 steps
- Randomization:
  - Initial positions of the cubes are randomized within a specified region on the table.
  - Random quaternions are applied to the cubes to introduce variability in their initial orientations.
- Goal & Success (already computed in `evaluate()`):
  - The cubes must be aligned in the XY plane within a tolerance.
  - The cubes must be stacked in the correct order with specified Z tolerances.
  - All cubes must be static, and the robot must not be grasping any cube.
- Useful tensors each step (batch size = B):
  - `tcp_pos`: Position of the robot's tool center point.
  - `A_pos`, `B_pos`, `C_pos`, `D_pos`: Positions of cubes A, B, C, and D.
  - `cube_half_size`: Half size of the cubes.
  - `is_A_static`, `is_B_static`, `is_C_static`, `is_D_static`: Boolean tensors indicating if each cube is static.
  - `is_A_grasped`, `is_B_grasped`, `is_C_grasped`, `is_D_grasped`: Boolean tensors indicating if each cube is grasped by the robot.

### Interfaces You Must Implement
Add/complete these methods inside `HumanoidStackCubeEnvV7`:
```python
def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
    ## Return a 1D torch tensor of shape [B] with the dense reward for each env.

def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
    ## Return the dense reward normalized to [0, 1] with a consistent max.
```

#### Non-Negotiable Constraints
- Vectorized over batch `B` (no Python loops).
- Torch-only; device-safe (use tensors on `self.device`).
- Numerically robust (avoid div-by-zero, clamp where needed).
- Deterministic given inputs.
- Keep a single scalar constant `MAX_REWARD` for normalization.

### Reward Design Goals
1. Avoid shortcuts (e.g., vibrating near the handle forever, or opening then oscillating).
2. Be smooth enough for policy gradient methods (no discontinuous spikes except final success).
3. Encourage progressive stacking: approach → align → stack → stabilize.
4. Penalize unnecessary movements or grasping actions.
5. Reward should increase as the task progresses towards completion.

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
4. Progressiveness
  - Reward should reflect the progress towards the goal, with intermediate rewards for partial stacking and alignment.
5. Stability
  - Reward should penalize instability or unnecessary grasping actions.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/stack_cube_v7.py
2.  ./examples/baselines/ppo/ppo_fast.py
