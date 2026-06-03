# Reward Module Design Brief (ManiSkill3: StackCube-v2)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves stacking three cubes in order so that cube A is on cube B, cube B is on cube C, and releasing the cubes with the stack remaining stable.

### Environment Facts
- Class: `StackCubeEnvV2`
- Robot description: The environment supports three types of robots: Panda with wrist camera, Panda, and Fetch.
- Episode length: Maximum of 50 steps per episode.
- Randomization:
  - All cubes have their z-axis rotation randomized.
  - All cubes have their xy positions randomized on top of the table scene without initial collision.
- Goal & Success (already computed in `evaluate()`):
  - Cube A must be on top of cube B within a specified tolerance.
  - Cube B must be on top of cube C within a specified tolerance.
  - Cubes A, B, and C must be static (linear velocity < 0.01 m/s, angular velocity < 0.5 rad/s).
  - Cubes A and B must not be grasped by the robot.
- Useful tensors each step (batch size = B):
  - `self.cubeA.pose.p`, `self.cubeB.pose.p`, `self.cubeC.pose.p`: Positions of cubes A, B, and C.
  - `self.agent.tcp.pose.p`: Position of the robot's tool center point.
  - `self.cube_half_size`: Half size of the cubes.
  - `info["is_A_on_B"]`, `info["is_B_on_C"]`, `info["success"]`: Boolean flags indicating stacking status and success.
  - `info["is_A_grasped"]`, `info["is_B_grasped"]`: Boolean flags indicating grasp status.

### Interfaces You Must Implement
Inside the class `StackCubeEnvV2`, check the following methods:
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
3. Encourage the robot to approach the cubes, align them correctly, release them, and ensure stability.
4. Provide intermediate rewards for partial completion of the stacking task to guide the robot towards the final goal.

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
  - The reward should consistently encourage stacking behavior across different random initializations.
5. Stability
  - The reward should penalize any instability in the stack after release.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/stack_cube_v2.py
2.  ./examples/baselines/ppo/ppo_fast.py
