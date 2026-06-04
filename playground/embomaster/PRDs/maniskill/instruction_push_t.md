# Reward Module Design Brief (ManiSkill3: PushT-v1)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. In this task, the robot needs to precisely push the T-shaped block into the target region and move the end-effector to the end-zone, which terminates the episode.

### Environment Facts
- Class: PushTEnv
- Robot description: PandaStick robot, capable of precise manipulation tasks.
- Episode length: 100 steps
- Randomization:
  - 3D T block initial position on table within [-1,1] x [-1,2] relative to the goal T position.
  - 3D T block initial z rotation within [0,2π].
- Goal & Success (already computed in `evaluate()`):
  - The T block must cover 90% of the 2D goal T's area for success.
  - The end-effector must reach the end-zone for episode termination.
- Useful tensors each step (batch size = B):
  - `self.tee.pose.q`: Quaternion representing the T block's rotation.
  - `self.tee.pose.p`: Position of the T block.
  - `self.goal_tee.pose.p`: Position of the goal T.
  - `self.agent.tcp.pose.p`: Position of the robot's end-effector.
  - `self.intersection_thresh`: Threshold for intersection area success.
  - `self.device`: Device for tensor operations.

### Interfaces You Must Implement
Inside the class `PushTEnv`, check the following methods:
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
3. Encourage precise alignment of the T block with the goal region.
4. Reward proximity of the end-effector to the T block to promote effective pushing strategies.
5. Ensure that the reward scales appropriately with the degree of success in aligning the T block.

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
4. Consistency
  - Reward should consistently increase as the T block approaches the goal region and aligns correctly.
5. Robustness
  - Reward should be robust to minor perturbations in position and orientation, ensuring smooth learning dynamics.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/push_t.py
2.  ./examples/baselines/ppo/ppo_fast.py
