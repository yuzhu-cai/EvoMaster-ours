# Reward Module Design Brief (ManiSkill3: StackCube-v5)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves stacking four cubes (A, B, C, D) in a specific configuration on a table. The cubes must be aligned in two layers: C and D on the bottom layer, and A and B on the top layer, with specific alignment and spacing requirements.

### Environment Facts
- Class: `StackCubeEnvV5`
- Robot description: The environment supports robots such as "panda_wristcam", "panda", and "fetch". The robot is positioned to the left side of the table.
- Episode length: 50 steps
- Randomization:
  - Initial positions of the cubes are randomized within a specified region on the table to avoid overlap.
  - Initial robot joint positions have a noise factor of 0.02.
- Goal & Success (already computed in `evaluate()`):
  - The goal is to achieve a specific configuration where cubes C and D are aligned on the bottom layer, and cubes A and B are stacked on top of C and D, respectively. The cubes must be aligned in both rows and columns, be stable, and not be grasped by the robot.
- Useful tensors each step (batch size = B):
  - `self.cubeA.pose.p`, `self.cubeB.pose.p`, `self.cubeC.pose.p`, `self.cubeD.pose.p`: Positions of the cubes.
  - `self.agent.tcp.pose.p`: Position of the robot's tool center point.
  - `self.cube_half_size`: Half-size of the cubes, used for alignment and spacing calculations.
  - `info`: Dictionary containing evaluation results such as alignment and stability checks.

### Interfaces You Must Implement
Inside the class `StackCubeEnvV5`, check the following methods:
```python
def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
    ## Return a 1D torch tensor of shape [B] with the dense reward for each env.

def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
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
3. Encourage progressive task completion: approach cubes, align them, stack them, stabilize, and release.
4. Penalize instability and grasping when the task is geometrically correct.

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
  - Rewards should consistently reflect the degree of task completion, with higher rewards for configurations closer to the goal state.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/stack_cube_v5.py
2.  ./examples/baselines/ppo/ppo_fast.py
