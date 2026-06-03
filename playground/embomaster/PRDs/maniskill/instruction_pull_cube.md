# Reward Module Design Brief (ManiSkill3: PullCube-v1)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves pulling a cube onto a target region marked by a red and white target.

### Environment Facts
- Class: PullCubeEnv
- Robot description: Supports two types of robots, Panda and Fetch.
- Episode length: Maximum of 50 steps per episode.
- Randomization:
  - The cube's xy position is randomized on top of a table within the region [0.1, 0.1] x [-0.1, -0.1].
  - The target goal region is fixed relative to the cube's initial position, offset by [-0.1 + goal_radius, 0].
- Goal & Success (already computed in `evaluate()`):
  - Success is achieved when the cube's xy position is within a goal_radius (default 0.1) of the target's xy position by Euclidean distance.
- Useful tensors each step (batch size = B):
  - `self.obj.pose.p`: Position tensor of the cube.
  - `self.goal_region.pose.p`: Position tensor of the target region.
  - `self.agent.tcp.pose.p`: Position tensor of the robot's tool center point (TCP).
  - `self.device`: Device tensor for computation.

### Interfaces You Must Implement
Inside the class `PullCubeEnv`, check the following methods:
```python
def compute_dense_reward(obs: Any, action: Array, info: Dict) -> torch.Tensor:
    ## Return a 1D torch tensor of shape [B] with the dense reward for each env.

def compute_normalized_dense_reward(obs: Any, action: Array, info: Dict) -> torch.Tensor:
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
3. Encourage a clear sequence of actions: approach the cube, align the TCP behind it, pull towards the target, and stabilize the cube within the goal region.
4. Reward should progressively increase as the cube gets closer to the target region, with a significant reward boost upon success.

### Deliverables
- Implementations of the two methods.
- A brief design note (docstring or comment) explaining your reward rationale (how it encourages approach → align → pull → stabilize).
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
4. Smoothness
  - Reward should smoothly increase as the cube approaches the target, avoiding abrupt changes except for the success condition.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/pull_cube.py
2.  ./examples/baselines/ppo/ppo_fast.py
