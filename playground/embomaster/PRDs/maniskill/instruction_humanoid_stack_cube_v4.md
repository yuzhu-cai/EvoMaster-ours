# Reward Module Design Brief (ManiSkill3: HumanoidStackCube-v4)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task involving stacking cubes with a humanoid robot.

### Environment Facts
- Class: `HumanoidStackCubeEnvV4`
- Robot description: The environment uses a `UnitreeG1UpperBodyWithHeadCamera` robot, which is a humanoid robot with a head camera and a simplified upper body.
- Episode length: 50 steps per episode.
- Randomization:
  - Initial positions of the cubes are randomized within a specified region using a uniform placement sampler.
  - Initial robot joint positions are subject to noise (`robot_init_qpos_noise=0.02`).
- Goal & Success (already computed in `evaluate()`):
  - The task is to align three cubes (red, green, blue) on the same horizontal plane (first layer) along the X-axis, with minimal Y-axis deviation and ensuring that adjacent cubes are "touching" (center distance approximately equals twice the half-width of a cube).
  - Success is achieved when the cubes are aligned, touching, stable, and not grasped by the robot.
- Useful tensors each step (batch size = B):
  - `tcp_pos`: Position of the robot's tool center point.
  - `A_pos`, `B_pos`, `C_pos`: Positions of cubes A, B, and C.
  - `z_stack`: Stack of Z positions for the cubes.
  - `y_min`, `y_max`: Minimum and maximum Y positions of the cubes.
  - `dA`, `dB`, `dC`: Distances from the TCP to each cube.
  - `vA`, `vB`, `vC`: Linear velocities of the cubes.
  - `wA`, `wB`, `wC`: Angular velocities of the cubes.

### Interfaces You Must Implement
Add/complete these methods inside `HumanoidStackCubeEnvV4`:
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
1. Avoid shortcuts (e.g., vibrating near the cubes forever, or aligning then oscillating).
2. Be smooth enough for policy gradient methods (no discontinuous spikes except final success).
3. Encourage progressive alignment and stabilization of cubes.
4. Reward should incentivize the robot to release the cubes once they are aligned and stable.

### Deliverables
- Implementations of the two methods.
- A brief design note (docstring or comment) explaining your reward rationale (how it encourages approach → align → stabilize → release).
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
  - Reward should increase as cubes get closer to the desired configuration (aligned, touching, stable).

### Guardrails / Non-Goals
- No external packages beyond PyTorch.
- No prints/logging.
- No heuristic resets or environment side-effects from the reward.
- Keep compute cost low; single-pass, vectorized operations only.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/stack_cube_v4.py
2.  ./examples/baselines/ppo/ppo_fast.py
