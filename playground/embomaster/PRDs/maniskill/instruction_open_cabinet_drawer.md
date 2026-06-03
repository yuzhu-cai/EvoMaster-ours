# Reward Module Design Brief (ManiSkill3: OpenCabinetDrawer-v1)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves using the Fetch mobile manipulation robot to approach a target cabinet and open a specified drawer.

### Environment Facts
- Class: OpenCabinetDrawerEnv
- Robot description: Fetch mobile manipulation robot
- Episode length: 100 steps
- Randomization:
  - Robot initialized 1.6 to 1.8 meters away from the cabinet, facing it.
  - Base orientation randomized by -9 to 9 degrees.
  - Cabinet and drawer selection randomized from available options.
- Goal & Success (already computed in `evaluate()`):
  - The drawer is open at least 90% of the way, and the angular/linear velocities of the drawer link are small.
- Useful tensors each step (batch size = B):
  - `self.agent.tcp.pose.p`: Position of the robot's tool center point.
  - `info["handle_link_pos"]`: Position of the handle link.
  - `self.handle_link.joint.qpos`: Current position of the joint controlling the drawer.
  - `self.target_qpos`: Target position for the joint to consider the drawer open.
  - `info["success"]`: Boolean tensor indicating if the success condition is met.

### Interfaces You Must Implement
Inside the class `OpenCabinetDrawerEnv`, check the following methods:
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
3. Encourage a clear progression: approach → align → open → stabilize.
4. Penalize excessive velocities to promote stability after opening.

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
  - Reward should increase as the robot approaches the handle, aligns, opens the drawer, and stabilizes.
5. Stability
  - Penalize high angular/linear velocities to ensure the drawer remains stable after opening.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/mobile_manipulation/open_cabinet_drawer.py
2.  ./examples/baselines/ppo/ppo_fast.py
