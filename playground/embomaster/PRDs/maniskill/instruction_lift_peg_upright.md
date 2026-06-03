# Reward Module Design Brief (ManiSkill3: LiftPegUpright-v1)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves moving a peg laying flat on a table to an upright position on the table.

### Environment Facts
- Class: LiftPegUprightEnv
- Robot description: The environment supports two types of robots, Panda and Fetch, which are used to manipulate the peg.
- Episode length: 50 steps
- Randomization:
  - The peg's xy position is randomized within the region [0.1, 0.1] x [-0.1, -0.1] on the table, and it is initially placed flat along its length.
- Goal & Success (already computed in `evaluate()`):
  - The peg's y euler angle must be within 0.08 radians of π/2, and its z position must be within 0.005 of its half-length (0.12) to be considered upright and successful.
- Useful tensors each step (batch size = B):
  - `self.peg.pose.q`: Quaternion representing the peg's orientation.
  - `self.peg.pose.p`: Position vector of the peg.
  - `self.agent.tcp.pose.p`: Position vector of the robot's tool center point (TCP).
  - `self.device`: Device on which tensors are allocated.

### Interfaces You Must Implement
Inside the class `LiftPegUprightEnv`, check the following methods:
```python
def compute_dense_reward(obs: Any, action: Array, info: Dict) -> torch.Tensor:
    ## Return a 1D torch tensor of shape [B] with the dense reward for each env.

def compute_normalized_dense_reward(obs: Any, action: Array, info: Dict) -> torch.Tensor:
    ## Return the dense reward normalized to [0, 1] with a consistent max.
```
If a method body is unimplemented (e.g., 'pass'), then add and complete its implementation according to this instruction. If the method already has valid content, then optimize the existing implementation according to this instruction.

#### Non-Negotiable Constraints
- Vectorized over batch `B` (no Python loops).
- Torch-only; device-safe (use tensors on `self.device`).
- Numerically robust (avoid div-by-zero, clamp where needed).
- Deterministic given inputs.
- Keep a single scalar constant `MAX_REWARD` for normalization.

### Reward Design Goals
1. Avoid shortcuts (e.g., vibrating near the peg forever, or lifting then oscillating).
2. Be smooth enough for policy gradient methods (no discontinuous spikes except final success).
3. Encourage a sequence of actions: approach → align → lift → stabilize.
4. Reward should motivate reaching and gripping the peg initially, aligning it vertically, and maintaining its upright position.

### Deliverables
- Implementations of the two methods.
- A brief design note (docstring or comment) explaining your reward rationale (how it encourages approach → align → lift → stabilize).
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
  - Reward should consistently increase as the peg approaches the upright position and decrease if it moves away from this position.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/lift_peg_upright.py
2.  ./examples/baselines/ppo/ppo_fast.py
