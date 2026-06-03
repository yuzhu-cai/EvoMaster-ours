# Reward Module Design Brief (ManiSkill3: TwoRobotPickCube-v1)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves two robots working collaboratively to pick up a red cube and lift it to a goal location. The left robot can reach the cube but not the goal, while the right robot can reach the goal but not the cube, necessitating cooperation between the two robots.

### Environment Facts
- Class: TwoRobotPickCube
- Robot description: Two Panda robots equipped with wrist cameras, working together to manipulate a cube.
- Episode length: 100 steps
- Randomization:
  - The cube's z-axis rotation is randomized.
  - The cube's xy position on the table is randomized to be within reach of the left robot but not the right.
  - The target goal position is randomized to be within reach of the right robot but not the left.
- Goal & Success (already computed in `evaluate()`):
  - The task is successful when the red cube is at the goal location, and the right robot arm is static.
- Useful tensors each step (batch size = B):
  - `self.cube.pose.p`: Position of the cube.
  - `self.goal_site.pose.p`: Position of the goal site.
  - `self.left_agent.tcp.pose.p`: Position of the left robot's tool center point (TCP).
  - `self.right_agent.tcp.pose.p`: Position of the right robot's TCP.
  - `self.left_init_qpos`: Initial joint positions of the left robot.
  - `self.right_agent.robot.get_qvel()`: Velocity of the right robot.
  - `self.left_agent.robot.get_qvel()`: Velocity of the left robot.

### Interfaces You Must Implement
Inside the class `TwoRobotPickCube`, check the following methods:
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
3. Encourage a clear sequence of actions: approach the cube, align for grasping, grasp the cube, transport it to the goal, and stabilize at the goal.
4. Penalize unnecessary movements or oscillations to encourage efficient task completion.

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
4. Progressiveness
  - Reward should progressively increase as the cube gets closer to the goal and the robots perform actions leading towards task completion.
5. Stability
  - Encourage stability by rewarding static states when the cube is at the goal.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/two_robot_pick_cube.py
2.  ./examples/baselines/ppo/ppo_fast.py
