# Reward Module Design Brief (ManiSkill3: StackCube-v1)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves picking up a red cube and stacking it on top of a green cube, then releasing the red cube without it falling.

### Environment Facts
- Class: `StackCubeEnv`
- Robot description: The environment supports robots such as "panda_wristcam", "panda", and "fetch", which are capable of manipulation tasks.
- Episode length: 50 steps
- Randomization:
  - Both cubes have their z-axis rotation randomized.
  - Both cubes have their xy positions on the table scene randomized, ensuring no collision between them.
- Goal & Success (already computed in `evaluate()`):
  - The red cube must be on top of the green cube within half of the cube size.
  - The red cube must be static.
  - The red cube must not be grasped by the robot.
- Useful tensors each step (batch size = B):
  - `self.cubeA.pose.p`: Position of the red cube.
  - `self.cubeB.pose.p`: Position of the green cube.
  - `self.cube_half_size`: Half size of the cubes.
  - `self.agent.tcp.pose.p`: Position of the robot's tool center point.
  - `info["is_cubeA_grasped"]`: Boolean indicating if the red cube is grasped.
  - `info["is_cubeA_on_cubeB"]`: Boolean indicating if the red cube is on the green cube.
  - `info["is_cubeA_static"]`: Boolean indicating if the red cube is static.
  - `info["success"]`: Boolean indicating if the task is successfully completed.

### Interfaces You Must Implement
Inside the class `StackCubeEnv`, check the following methods:
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
3. Encourage a sequence of actions: approach → grasp → stack → release → stabilize.
4. Ensure the reward incentivizes the robot to maintain the cube's stability after stacking.

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
  - Reward values should consistently reflect the task progress, with higher rewards as the robot approaches the goal state.
5. Stability Incentive
  - Reward should increase as the cube becomes more stable post-release, discouraging any movement after stacking.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/stack_cube.py
2.  ./examples/baselines/ppo/ppo_fast.py
