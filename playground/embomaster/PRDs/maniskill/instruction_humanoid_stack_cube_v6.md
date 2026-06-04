# Reward Module Design Brief (ManiSkill3: HumanoidStackCube-v6)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task. The task involves stacking four colored cubes (A=red, B=green, C=blue, D=yellow) on the same layer (table surface) in a 1x4 horizontal alignment along the world X direction. The cubes must be adjacent with their centers nearly touching, and the final state should be stable with the robot not grasping any cube.

### Environment Facts
- Class: `HumanoidStackCubeEnvV6`
- Robot description: `unitree_g1_simplified_upper_body_with_head_camera`, a humanoid robot with an upper body and head camera.
- Episode length: 50 steps
- Randomization:
  - Initial cube positions are randomized within a specified region on the table to avoid overlap.
  - Randomized initial robot joint positions with noise.
- Goal & Success (already computed in `evaluate()`):
  - Cubes are on the same layer (Z coordinates close to the lowest Z plane).
  - Cubes are horizontally aligned (Y coordinates close to each other).
  - Adjacent cubes are touching (X distances close to twice the half size of a cube).
  - All cubes are static (not moving).
  - Robot is not grasping any cube.
- Useful tensors each step (batch size = B):
  - `self.cubeA.pose.p`, `self.cubeB.pose.p`, `self.cubeC.pose.p`, `self.cubeD.pose.p`: Positions of cubes A, B, C, and D.
  - `self.agent.right_tcp.pose.p`: Position of the robot's right hand TCP.
  - `self.cube_half_size`: Half size of the cubes.
  - `info["same_layer"]`, `info["y_aligned"]`, `info["all_touch"]`, `info["success"]`: Boolean tensors indicating success conditions.

### Interfaces You Must Implement
Add/complete these methods inside `HumanoidStackCubeEnvV6`:
```python
def compute_dense_reward(obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
    ## Return a 1D torch tensor of shape [B] with the dense reward for each env.

def compute_normalized_dense_reward(obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
    ## Return the dense reward normalized to [0, 1] with a consistent max.
```

#### Non-Negotiable Constraints
- Vectorized over batch `B` (no Python loops).
- Torch-only; device-safe (use tensors on `self.device`).
- Numerically robust (avoid div-by-zero, clamp where needed).
- Deterministic given inputs.
- Keep a single scalar constant `MAX_REWARD` for normalization.

### Reward Design Goals
1. Avoid shortcuts (e.g., vibrating near the cubes forever, or stacking then oscillating).
2. Be smooth enough for policy gradient methods (no discontinuous spikes except final success).
3. Encourage the robot to approach cubes, align them horizontally, ensure they are touching, and stabilize them without grasping.
4. Reward should progressively increase as the robot achieves sub-goals leading to the final success state.

### Deliverables
- Implementations of the two methods.
- A brief design note (docstring or comment) explaining your reward rationale (how it encourages approach → align → touch → stabilize).
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
  - Reward should increase as cubes get closer to the target positions and orientations, encouraging continuous improvement towards the goal.

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/stack_cube_v6.py
2.  ./examples/baselines/ppo/ppo_fast.py
