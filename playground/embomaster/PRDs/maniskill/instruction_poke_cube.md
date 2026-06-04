# Reward Module Design Brief (ManiSkill3: PokeCube-v1)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves using a robot to poke a red cube with a peg and push it to a target goal position.

### Environment Facts
- Class: PokeCubeEnv
- Robot description: The environment supports two types of robots, Panda and Fetch, which are used to manipulate a peg to interact with a cube.
- Episode length: 50 steps
- Randomization:
  - The peg's xy position is randomized within the region [0.1, 0.1] x [-0.1, -0.1] on a table.
  - The cube's x-coordinate is fixed relative to the peg, and its y-coordinate is randomized within [-0.1, 0.1].
  - The cube's z-axis rotation is randomized within [-π/6, π/6].
  - The target goal region is fixed relative to the cube's initial position.
- Goal & Success (already computed in `evaluate()`):
  - The cube's xy position must be within the goal_radius (default 0.05) of the target's xy position by Euclidean distance.
  - The robot must be static.
- Useful tensors each step (batch size = B):
  - `tcp_pos`: The position of the robot's tool center point.
  - `tgt_tcp_pose`: The pose of the peg.
  - `tcp_to_peg_dist`: Distance from the tool center point to the peg.
  - `angle_diff`: The angular difference between the peg and the cube.
  - `head_to_cube_dist`: Distance from the peg head to the cube.
  - `cube_to_goal_dist`: Distance from the cube to the goal region.
  - `is_peg_grasped`: Boolean indicating if the peg is grasped.
  - `is_peg_cube_fit`: Boolean indicating if the peg and cube are aligned and close.
  - `is_cube_placed`: Boolean indicating if the cube is placed within the goal region.
  - `static_reward`: Reward for the robot being static.

### Interfaces You Must Implement
Inside the class `PokeCubeEnv`, check the following methods:
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
3. Encourage a sequence of actions: approach the peg, align the peg with the cube, push the cube towards the goal, and stabilize the robot.
4. Provide intermediate rewards for reaching, aligning, and placing actions to guide the agent progressively towards success.

### Deliverables
- Implementations of the two methods.
- A brief design note (docstring or comment) explaining your reward rationale (how it encourages approach → align → push → stabilize).
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
  - Rewards should increase as the agent progresses from reaching the peg, aligning with the cube, pushing the cube, and finally stabilizing at the goal.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/poke_cube.py
2.  ./examples/baselines/ppo/ppo_fast.py
