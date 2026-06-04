# Reward Module Design Brief (ManiSkill3: HumanoidStackCube-v3)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task.
The task involves manipulating three cubes (A=red, B=green, C=blue) such that they are all on the "first layer" (on the table surface, not stacked), aligned along the world X direction (Y coordinates are approximately the same), and at least one pair of cubes is "edge-touching" in the X direction (center distance ≈ 2 * half_x). The final state must be stable, and the robot should release all cubes (not grasp any cube).

### Environment Facts
- Class: `HumanoidStackCubeEnvV3`
- Robot description: A simplified upper body of the Unitree G1 robot with a head camera.
- Episode length: 50 steps
- Randomization:
  - Initial XY positions of the three cubes are randomized within the table area, with random orientations around the Z-axis.
  - Sampling avoids initial overlaps.
- Goal & Success (already computed in `evaluate()`):
  - All cubes are on the same "lowest z plane" within a tolerance `z_tol`.
  - Cubes are aligned along the Y-axis within a tolerance `y_tol`.
  - At least one pair of cubes is edge-touching in the X direction within a tolerance `x_tol`.
  - All cubes are static (linear velocity < 0.01 m/s, angular velocity < 0.5 rad/s).
  - The robot is not grasping any cube.
- Useful tensors each step (batch size = B):
  - `self.cubeA.pose.p`, `self.cubeB.pose.p`, `self.cubeC.pose.p`: Positions of cubes A, B, and C.
  - `self.agent.right_tcp.pose.p`: Position of the robot's tool center point (TCP).
  - `self.cube_half_size`: Half size of the cubes.
  - `info["same_layer"]`, `info["y_aligned"]`, `info["pair_touch_any"]`: Boolean tensors indicating success conditions.

### Interfaces You Must Implement
Add/complete these methods inside `HumanoidStackCubeEnvV3`:
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
3. Encourage progressive achievement of task goals: approach cubes, align them, ensure they are on the same layer, and stabilize them without grasping.
4. Provide clear incentives for achieving each sub-goal, with the highest reward for complete success.

### Deliverables
- Implementations of the two methods.
- A brief design note (docstring or comment) explaining your reward rationale (how it encourages approach → align → stabilize → release).

---
### Property-Based Evaluation (must pass)
Write the reward so these properties are true; the exact numbers are up to you.
1. Ordering
  - `reward(success=True) > reward(any non-success state)` by a clear margin.
2. Normalization
  - `compute_normalized_dense_reward outputs` ∈ `[0, 1]`; at success it returns exactly 1.0.
3. Boundedness
  - `compute_dense_reward` is bounded above by a constant `MAX_REWARD` and below by a finite value (no `inf/-inf` or NaNs).
4. Incremental Progress
  - Reward should increase as the agent makes progress towards the goal, even if the final success is not achieved.
5. Stability Incentive
  - Encourage stability by rewarding static states more than dynamic ones.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/stack_cube_v3.py
2.  ./examples/baselines/ppo/ppo_fast.py
