# Reward Module Design Brief (ManiSkill3: PickSingleYCB-v1)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves picking up a random object sampled from the YCB dataset and moving it to a random goal position.

### Environment Facts
- Class: PickSingleYCBEnv
- Robot description: Supports Panda, PandaWristCam, and Fetch robots.
- Episode length: Maximum of 50 steps per episode.
- Randomization:
  - The object's xy position is randomized within the region [0.1, 0.1] x [-0.1, -0.1] on a table.
  - The object's z-axis rotation is randomized.
  - The object geometry is randomized by sampling any YCB object.
- Goal & Success (already computed in `evaluate()`):
  - The object must be within a euclidean distance of 0.025 from the goal position.
  - The robot must be static with q velocity < 0.2.
- Useful tensors each step (batch size = B):
  - `self.obj.pose.p`: Position of the object.
  - `self.agent.tcp.pose.p`: Position of the robot's tool center point (TCP).
  - `self.goal_site.pose.p`: Position of the goal site.
  - `info["is_grasped"]`: Boolean indicating if the object is grasped.
  - `info["is_obj_placed"]`: Boolean indicating if the object is placed at the goal.
  - `info["success"]`: Boolean indicating successful task completion.

### Interfaces You Must Implement
Inside the class `PickSingleYCBEnv`, check the following methods:
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
3. Encourage a sequence of actions: approach → grasp → transport → place → stabilize.
4. Reward should increase as the object gets closer to the goal and the robot remains static.

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
4. Smoothness
  - Reward changes smoothly with changes in object position and robot state, avoiding abrupt jumps except at success.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/pick_single_ycb.py
2.  ./examples/baselines/ppo/ppo_fast.py
