# Reward Module Design Brief (ManiSkill3: TransportCube-v1)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves a robot finding a red cube on one table, transporting it to another table, and placing it within a designated area.

### Environment Facts
- Class: `TransportCubeEnv`
- Robot description: The environment supports multiple robots, including Panda, Fetch, XArm6Robotiq, SO100, and WidowXAI.
- Episode length: 100 steps
- Randomization:
  - The cube's xy position is randomized on the source table within the region [0.1, 0.1] x [-0.1, -0.1].
  - The cube's z-axis rotation is randomized to a random angle.
  - The target table position is fixed, but the cube placement area is randomized.
- Goal & Success (already computed in `evaluate()`):
  - The cube must be placed on the target table within the designated area.
  - The robot must be static (q velocity < 0.2).
- Useful tensors each step (batch size = B):
  - `self.cube.pose.p`: Position of the cube.
  - `self.agent.tcp_pose.p`: Position of the robot's tool center point (TCP).
  - `self.target_zone.pose.p`: Position of the target placement zone.
  - `info["is_grasped"]`: Boolean indicating if the cube is grasped.
  - `info["is_obj_placed"]`: Boolean indicating if the cube is placed in the target area.
  - `info["success"]`: Boolean indicating if the task is successfully completed.

### Interfaces You Must Implement
Inside the class `TransportCubeEnv`, check the following methods:
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
3. Encourage a clear progression: approach the cube, grasp it, transport it, and place it accurately.
4. Penalize unnecessary movements or oscillations to promote efficient task completion.

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
  - Reward should increase as the robot progresses through the stages of the task: approaching, grasping, transporting, and placing the cube.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/transport_cube.py
2.  ./examples/baselines/ppo/ppo_fast.py
