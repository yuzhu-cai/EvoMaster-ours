"""RICE clean-room reproduction package."""

from .critical_states import CriticalState, select_critical_state, sliding_window_scores
from .mask_network import MaskNetwork, MaskTrainer, OptimizedStateMaskExplanation, PerturbedPolicy
from .refinement import RICERefiner, RefinementConfig
from .rnd import RNDModel, RNDReward
from .statemask import OriginalStateMaskExplanation, StateMaskPrimalDualTrainer, statemask_objective

__all__ = [
    "CriticalState",
    "select_critical_state",
    "sliding_window_scores",
    "MaskNetwork",
    "MaskTrainer",
    "OptimizedStateMaskExplanation",
    "PerturbedPolicy",
    "RICERefiner",
    "RefinementConfig",
    "RNDModel",
    "RNDReward",
    "OriginalStateMaskExplanation",
    "StateMaskPrimalDualTrainer",
    "statemask_objective",
]
