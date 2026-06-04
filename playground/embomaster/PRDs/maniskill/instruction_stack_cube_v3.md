# Reward Module Design Brief (ManiSkill3: StackCube-v3)

## Project
Implement the **reward module** for a ManiSkill3 mobile manipulation task.
The task involves manipulating three cubes (A=red, B=green, C=blue) such that they are all on the "first layer" (on the table, not stacked), aligned along the world X direction (Y coordinates are approximately equal), and at least one pair of cubes is "edge-connected" in the X direction (center distance ≈ 2 * half_x). The final state must be stable, and the robot must release all cubes (not grasp any cube).

### Environment Facts
- Class: StackCubeEnvV3
- Robot description: The environment supports three types of robots: "panda_wristcam", "panda", and "fetch". The robot is positioned on the left side of the table.
- Episode length: 50 steps
- Randomization:
  - Initial XY positions of the cubes are randomized within the table area, with random orientations around the Z-axis. The sampling avoids initial overlaps.
- Goal & Success (already computed in `evaluate()`):
  - All cubes are on the same "lowest Z plane" with a tolerance of z_tol.
  - The Y coordinates of the cubes are aligned within a tolerance of y_tol.
  - At least one pair of cubes is edge-connected in the X direction with a center distance ≈ 2 * half_x and Y coordinates within y_tol.
  - All cubes are static (linear velocity < 0.01 m/s, angular velocity < 0.5 rad/s).
  - The robot is not grasping any cube.
- Useful tensors each step (batch size B):
  - `self.cubeA.pose.p`, `self.cubeB.pose.p`, `self.cubeC.pose.p`: Positions of cubes A, B, and C.
  - `self.cube_half_size`: Half size of the cubes.
  - `self.agent.tcp.pose.p`: Position of the robot's tool center point (TCP).

### Environment-Specific Methods
Inside the class `StackCubeEnvV3`, check the following methods:
```python
def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
    ## Return a 1D torch tensor of shape [B] with the dense reward value for each environment instance.

def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
    ## Return the dense reward normalized to [0, 1] with a consistent max.
```

#### Non-Negotiable Constraints
- Vectorized over batch `B` (no Python loops).
- Torch-only; device-safe (use tensors on `self.device`).
- Numerically robust (avoid div-by-zero, clamp where needed).
- Deterministic given inputs.
- Keep a single scalar constant `MAX_REWARD` for normalization.

### Reward Design Goals
1. Encourage the robot to approach the cubes by minimizing the distance to the nearest cube.
2. Encourage alignment of the cubes along the Y-axis by minimizing the range of Y coordinates.
3. Encourage all cubes to be on the same layer by minimizing the deviation of their Z coordinates from the lowest Z plane.
4. Encourage at least one pair of cubes to be edge-connected by minimizing the gap between their X coordinates while ensuring their Y coordinates are aligned.
5. Reward the robot for not grasping any cubes and for all cubes being static.
6. Provide a significant success reward when all success conditions are met.

### Deliverables
- Implementations of the two methods.
- A brief design note (docstring or comment) explaining your reward rationale (how it encourages approach → align → open → stabilize).
- The property test above passes without editing it.

### Property-Based Evaluation (must pass)
1. Ordering
  - `compute_dense_reward` should increase as the cubes become more aligned, more on the same layer, and more edge-connected.
  - The reward should be higher when the robot is not grasping any cube and all cubes are static.
2. Normalization
  - `compute_normalized_dense_reward` should return values in the range `[0, 1]`, with a value of exactly 1.0 when the success condition is met.
3. Boundedness
  - `compute_dense_reward` should be bounded above by a constant `MAX_REWARD` and should not produce infinite or NaN values.
4. Consistency
  - The reward should be consistent across different runs given the same inputs.
5. Robustness
  - The reward calculation should handle edge cases, such as when cubes are initially overlapping, without errors or unexpected behavior.
### Related Code
You may **ONLY** edit the following files:
1.  ./mani_skill/envs/tasks/tabletop/stack_cube_v3.py
2.  ./examples/baselines/ppo/ppo_fast.py
