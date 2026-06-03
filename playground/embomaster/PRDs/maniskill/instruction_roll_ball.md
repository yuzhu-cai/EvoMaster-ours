# Reward Module Design Brief (ManiSkill3: RollBall-v1)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves pushing and rolling a ball to a designated goal region on a table.

### Environment Facts
- Class: RollBallEnv
- Robot description: The environment supports the "panda" robot, which is used to manipulate the ball.
- Episode length: 80 steps
- Randomization:
  - The ball's initial xy position is randomized within the region [0.2, 0.5] x [-0.4, 0.7] on the table.
  - The target goal region's position is randomized within the region [-0.4, -0.7] x [0.2, -0.9] on the table.
- Goal & Success (already computed in `evaluate()`):
  - The task is considered successful if the ball's xy position is within a `goal_radius` (default 0.1) of the target's xy position, measured by Euclidean distance.
- Useful tensors each step (batch size = B):
  - `self.ball.pose.p`: The current position of the ball.
  - `self.goal_region.pose.p`: The position of the goal region.
  - `self.reached_status`: A tensor indicating whether the robot's TCP is close enough to the ball to influence it.
  - `self.agent.tcp.pose.p`: The position of the robot's tool center point (TCP).

### Interfaces You Must Implement
Inside the class `RollBallEnv`, check the following methods:
```python
def compute_dense_reward(self, obs: Any, action: Array, info: Dict) -> torch.Tensor:
    ## Return a 1D torch tensor of shape [B] with the dense reward for each env.

def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict) -> torch.Tensor:
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
1. Avoid shortcuts (e.g., vibrating near the ball forever, or rolling then oscillating).
2. Be smooth enough for policy gradient methods (no discontinuous spikes except final success).
3. Encourage continuous progress towards the goal, with higher rewards as the ball gets closer to the target.
4. Penalize large deviations from the optimal path to the goal to ensure efficient task completion.

### Deliverables
- Implementations of the two methods.
- A brief design note (docstring or comment) explaining your reward rationale (how it encourages approach → align → roll → stabilize).
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
  - Reward should increase as the ball gets closer to the goal, with a smooth gradient to facilitate learning.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/roll_ball.py
2.  ./examples/baselines/ppo/ppo_fast.py
