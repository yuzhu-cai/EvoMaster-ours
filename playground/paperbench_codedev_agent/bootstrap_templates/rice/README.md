# RICE Reproduction: Explanation-Guided RL Refinement

This repository is a clean-room implementation of the main code-development components for **RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation**.  It focuses on the paper's central mechanism: use a step-level explanation method to find critical visited states, mix those states with the default initial-state distribution, and continue PPO refinement with an RND exploration bonus.

## Paper-To-Code Map

- **StateMask-style explanation and the RICE mask objective**: `rice_reproduce/mask_network.py` implements a binary `MaskNetwork`, the `PerturbedPolicy`, and PPO training with the paper's alpha blinding bonus `R' = R + alpha * a_m`.
- **Ours rollout generation**: `OptimizedStateMaskExplanation` in `rice_reproduce/mask_network.py` loads or wraps a trained optimized mask and generates perturbed rollouts without retraining, including autonomous-driving/CAGE/selfish-mining use through the environment factory.
- **Original StateMask baseline**: `rice_reproduce/statemask.py` implements the original objective `J(theta)=|eta(pi)-eta(pi_bar)|`, a primal-dual mask-budget update, inference-time rollout generation, and training-time/cumulative-reward logging.
- **Critical-state selection**: `rice_reproduce/critical_states.py` implements top-window scoring, `select_critical_state`, random-explanation selection, and the fidelity score formula from the addendum.
- **Mixed initial-state distribution**: `rice_reproduce/refinement.py` implements Algorithm 2 with probability `p_reset_to_critical` for resetting to an explanation-selected state and probability `1-p` for the default reset distribution.
- **RND exploration bonus**: `rice_reproduce/rnd.py` implements fixed target and trainable predictor networks, normalized prediction-error bonuses, and predictor updates.
- **PPO refinement**: `rice_reproduce/policies.py` and `rice_reproduce/refinement.py` implement actor-critic distributions, clipped PPO policy/value losses, entropy terms, GAE, and RND-augmented rewards.
- **Pretraining**: `rice_reproduce/pretraining.py` and `scripts/run_pretrain.py` implement PPO pretraining for MuJoCo, SelfishMining, CAGE/network-defense, and autonomous-driving policies while recording cumulative reward and training time.
- **Baselines**: `rice_reproduce/baselines/refining.py` covers lower-learning-rate PPO fine-tuning, StateMask-R, RICE with random explanation, and optimized-StateMask explanation wiring for PPO/JSRL; `rice_reproduce/baselines/jsrl.py` implements JSRL with a separate exploration policy initialized from the guide policy (`pi_e <- pi_g`).
- **Environments**: `rice_reproduce/envs/factory.py` and `rice_reproduce/envs/sparse_mujoco.py` implement dense/sparse MuJoCo setup, Walker/HalfCheetah observation normalization, SelfishMining, CAGE Challenge 2/network defense, and MetaDrive Macro-v1/autonomous-driving adapters.
- **SAC-to-PPO transfer**: `rice_reproduce/sac_gail.py` provides SAC pretraining for dense MuJoCo, teacher dataset collection, supervised distillation, and a GAIL discriminator/adversarial imitation loop for the SAC warm-start experiment.
- **Experiments**: `rice_reproduce/experiments/configs.py`, `rice_reproduce/experiments/experiment_ii.py`, `scripts/run_table1.py`, `scripts/run_fidelity.py`, `scripts/run_refine.py`, and `scripts/run_hyperparams.py` materialize Experiments I-V from the main text and appendix details.

## Main Experiment Coverage

### Experiment I: Fidelity and Efficiency

`MaskTrainer` learns the mask network through PPO on perturbed rollouts. `evaluate_fidelity` applies the addendum protocol: choose a highest-average-importance sliding window for K in `{0.1, 0.2, 0.3, 0.4}`, randomize actions in that segment, let the target policy finish the episode, and compute `log(d/d_max) - log(l/L)`.

### Experiment II: Refining Effectiveness

`RICERefiner` implements the paper's core refining loop: a trajectory is sampled from the current policy, the mask network ranks states, the refiner resets either to the critical state or the default distribution according to `p`, and PPO optimizes rewards augmented by `lambda * RND`.

### Experiment III: Explanation Quality

`random_importance` and `critical_from_method` let the same refiner use random, StateMask, or RICE mask explanations so the downstream effect of explanation quality can be compared.

### Experiment IV: SAC Agent Refinement

`sac_gail.py` supports collecting actions from a SAC teacher and distilling them into the PPO-compatible policy used by RICE, matching the paper's Hopper SAC experiment structure.

### Experiment V: Hyper-parameters

`configs/table1.yaml`, `configs/rice_hopper.yaml`, and `scripts/run_hyperparams.py` expose the paper's grids for `p`, `lambda`, and `alpha`, including Table 3 defaults.

## Quick Checks

```bash
python -m rice_reproduce.cli describe
python -m rice_reproduce.cli smoke --out outputs/smoke.json
python scripts/run_pretrain.py --env-id SelfishMining-v0 --updates 1 --out outputs/pretrain.json
pytest -q
```

## Running A Refinement Job

```bash
python scripts/run_refine.py --config configs/rice_hopper.yaml --out outputs/hopper_refine.json
```

The command expects the corresponding Gymnasium/MuJoCo dependencies.  The code keeps external simulator adapters separate so the algorithmic components can be tested without heavyweight environments.

## Notes On Scope

The implementation follows the addendum's scope: malware-mutation experiments and appendix-only sparse Walker2d sensitivity runs are represented as adapters/configuration hooks, while the main reproducible code emphasizes the explanation method, RICE refiner, dense MuJoCo tasks, selected sparse MuJoCo tasks, SAC transfer, and real-world environment interfaces.
