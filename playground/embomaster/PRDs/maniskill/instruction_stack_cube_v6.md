# Reward Module Design Brief (ManiSkill3: StackCube-v6)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves stacking four colored cubes (red, green, blue, yellow) on a table in a specific configuration where all cubes are aligned along the world X direction in a 1x4 horizontal line, with their Y coordinates nearly identical. The cubes must be adjacent with minimal gaps between them, and the final state should be stable with the robot not grasping any cube.

### Environment Facts
- Class: `StackCubeEnvV6`
- Robot description: Supports robots "panda_wristcam", "panda", and "fetch". The robot is initialized with a small positional noise.
- Episode length: Maximum of 50 steps per episode.
- Randomization:
  - Initial cube positions are randomized within a specified region on the table to avoid overlap.
  - Initial cube orientations are randomized around the Z-axis.
- Goal & Success (already computed in `evaluate()`):
  - All cubes must be on the same layer (table surface).
  - Cubes must be aligned in the Y direction.
  - Adjacent cubes must be touching with minimal gaps.
  - All cubes must be static (not moving).
  - The robot must not be grasping any cube.
- Useful tensors each step (batch size = B):
  - `self.cube_half_size`: Tensor representing half the size of the cubes.
  - `self.agent.tcp.pose.p`: Position of the robot's tool center point.
  - `self.cubeA.pose.p`, `self.cubeB.pose.p`, `self.cubeC.pose.p`, `self.cubeD.pose.p`: Positions of the cubes.
  - `self.cubeA.linear_velocity`, `self.cubeB.linear_velocity`, `self.cubeC.linear_velocity`, `self.cubeD.linear_velocity`: Linear velocities of the cubes.
  - `self.cubeA.angular_velocity`, `self.cubeB.angular_velocity`, `self.cubeC.angular_velocity`, `self.cubeD.angular_velocity`: Angular velocities of the cubes.

### Interfaces You Must Implement
Inside the class `StackCubeEnvV6`, check the following methods:
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
3. Encourage sequential task completion: approach cubes → align cubes → release grip → stabilize cubes.
4. Penalize excessive movement or instability of cubes.
5. Reward should be shaped to guide the robot towards achieving the task goals progressively.

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
4. Progression
  - Reward should increase as the robot progresses towards achieving the task goals (approach, align, release, stabilize).

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/stack_cube_v6.py
2.  ./examples/baselines/ppo/ppo_fast.py
