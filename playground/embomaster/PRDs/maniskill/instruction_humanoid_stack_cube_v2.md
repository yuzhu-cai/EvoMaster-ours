# Reward Module Design Brief (ManiSkill3: HumanoidStackCube-v2)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves stacking three cubes in order such that cube A is on cube B, and cube B is on cube C, with the stack remaining stable and cubes A and B not being grasped by the robot.

### Environment Facts
- Class: HumanoidStackCubeEnvV2
- Robot description: UnitreeG1UpperBodyWithHeadCamera, a humanoid robot with a head camera.
- Episode length: 50 steps
- Randomization:
  - All cubes have their z-axis rotation randomized.
  - All cubes have their xy positions randomized on top of the table scene without initial collision.
- Goal & Success (already computed in `evaluate()`):
  - Cube A is on top of cube B within tolerance.
  - Cube B is on top of cube C within tolerance.
  - Cubes A, B, and C are static (linear velocity < 0.01 m/s, angular velocity < 0.5 rad/s).
  - Cubes A and B are not being grasped by the robot.
- Useful tensors each step (batch size = B):
  - `self.cubeA.pose.p`: Position of cube A.
  - `self.cubeB.pose.p`: Position of cube B.
  - `self.cubeC.pose.p`: Position of cube C.
  - `self.agent.right_tcp.pose.p`: Position of the robot's right hand TCP.
  - `self.cubeA.linear_velocity`: Linear velocity of cube A.
  - `self.cubeA.angular_velocity`: Angular velocity of cube A.
  - `self.cubeB.linear_velocity`: Linear velocity of cube B.
  - `self.cubeB.angular_velocity`: Angular velocity of cube B.
  - `self.agent.right_hand_dist_to_open_grasp()`: Distance to open grasp of the robot's right hand.

### Interfaces You Must Implement
Add/complete these methods inside `HumanoidStackCubeEnvV2`:
```python
def compute_dense_reward(obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
    ## Return a 1D torch tensor of shape [B] with the dense reward for each env.

def compute_normalized_dense_reward(obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
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
3. Encourage a clear progression from approaching the cubes, aligning them, opening the gripper, and stabilizing the stack.
4. Reward should incentivize the robot to ungrasp cubes A and B once they are correctly stacked and stable.

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
4. Reward should smoothly increase as the cubes are correctly aligned and stacked, with additional reward for ungrasping and stabilizing.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/stack_cube_v2.py
2.  ./examples/baselines/ppo/ppo_fast.py
