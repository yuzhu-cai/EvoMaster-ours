# Reward Module Design Brief (ManiSkill3: HumanoidStackCube-v5)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task.
The task involves stacking four cubes (A=red, B=green, C=blue, D=yellow) in a specific configuration across two layers on a kitchen counter. The goal is to have cubes C and D aligned on the first layer, and cubes A and B stacked on top of C and D respectively, forming the second layer. The cubes should be aligned in rows and columns with minimal spacing, remain static, and not be grasped by the robot.

### Environment Facts
- Class: HumanoidStackCubeEnvV5
- Robot description: UnitreeG1UpperBodyWithHeadCamera, a humanoid robot with a simplified upper body and head camera.
- Episode length: 50 steps
- Randomization:
  - Initial cube positions are randomized within a specified region on the kitchen counter to avoid overlap.
  - Robot's initial joint positions are subject to noise.
- Goal & Success (already computed in `evaluate()`):
  - Success is achieved when cubes C and D are on the same plane, cubes A and B are correctly stacked on C and D, all cubes are aligned in rows and columns, spacing between cubes is minimal, cubes are static, and none are grasped by the robot.
- Useful tensors each step (batch size = B):
  - `self.cubeA.pose.p`, `self.cubeB.pose.p`, `self.cubeC.pose.p`, `self.cubeD.pose.p`: Positions of cubes A, B, C, and D.
  - `self.agent.right_tcp.pose.p`: Position of the robot's end effector.
  - `self.cube_half_size`: Half-size of the cubes used for alignment and spacing calculations.
  - `info`: A dictionary containing boolean tensors indicating various success conditions.

### Interfaces You Must Implement
Add/complete these methods inside `HumanoidStackCubeEnvV5`:
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
1. Encourage the robot to approach the cubes, align them correctly, stack them, and ensure they remain static without being grasped.
2. Avoid shortcuts such as maintaining proximity without achieving the stacking goal.
3. Ensure smooth reward transitions to facilitate policy gradient methods, with a significant reward spike upon achieving success.
4. Incorporate penalties for misalignment and instability to guide the robot towards the desired configuration.

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
4. Consistency
  - Reward values should consistently reflect the degree of task completion, penalizing misalignment and instability proportionally.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/stack_cube_v5.py
2.  ./examples/baselines/ppo/ppo_fast.py
