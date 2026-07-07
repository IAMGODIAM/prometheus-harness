"""J-Lens Engine — Jacobian Lens fitting, readout, decomposition, interventions.

Public API re-exported from submodules:
- lens: JacobianLens, LensConfig
- fitting: fit_jacobian_lens, jacobian_for_prompt, JLensFitter, FitProgress, ActivationRecorder
- jspace: JSpaceDecomposer, JSpaceResult
- interventions: SteeringIntervention, AblationIntervention, ConceptSwapIntervention, InterventionHook
- watchlist: Watchlist, WatchlistScores
- model_adapter: ModelAdapter, ModelInfo
"""

from prometheus.jlens.lens import JacobianLens, LensConfig
from prometheus.jlens.fitting import (
    ActivationRecorder,
    FitProgress,
    JLensFitter,
    fit_jacobian_lens,
    jacobian_for_prompt,
    get_final_norm,
    get_model_layers,
    get_unembed,
)
from prometheus.jlens.jspace import JSpaceDecomposer, JSpaceResult
from prometheus.jlens.interventions import (
    AblationIntervention,
    ConceptSwapIntervention,
    Intervention,
    InterventionHook,
    InterventionType,
    SteeringIntervention,
    ablate,
    concept_swap,
    steer,
)
from prometheus.jlens.watchlist import Watchlist, WatchlistScores
from prometheus.jlens.model_adapter import ModelAdapter, ModelInfo

__all__ = [
    "JacobianLens",
    "LensConfig",
    "ActivationRecorder",
    "FitProgress",
    "JLensFitter",
    "fit_jacobian_lens",
    "jacobian_for_prompt",
    "get_final_norm",
    "get_model_layers",
    "get_unembed",
    "JSpaceDecomposer",
    "JSpaceResult",
    "AblationIntervention",
    "ConceptSwapIntervention",
    "Intervention",
    "InterventionHook",
    "InterventionType",
    "SteeringIntervention",
    "ablate",
    "concept_swap",
    "steer",
    "Watchlist",
    "WatchlistScores",
    "ModelAdapter",
    "ModelInfo",
]
