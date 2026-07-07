"""J-Lens Interventions — Steering, ablation, and concept-swap in lens coordinates.

Three intervention forms from the paper:
1. Steering: h ← h + α·v_t (directional injection)
2. Ablation: project out J-space component or negate specific directions
3. Concept-swap: h_patched = h + V(σ(c) − c) where V=[v_s, v_t], c = V†h, σ swaps
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn


class InterventionType(str, Enum):
    STEER = "steer"
    ABLATE = "ablate"
    CONCEPT_SWAP = "concept_swap"


@dataclass
class Intervention(ABC):
    """Base class for J-lens interventions."""

    layers: list[int]
    positions: list[int] | None = None  # None = all positions

    @abstractmethod
    def apply(self, h: torch.Tensor, layer: int, position: int) -> torch.Tensor:
        """Apply the intervention to an activation.

        Args:
            h: Activation tensor of shape (d_model,) or (seq_len, d_model).
            layer: Current layer index.
            position: Current position index.

        Returns:
            Modified activation tensor.
        """
        ...

    @property
    @abstractmethod
    def intervention_type(self) -> InterventionType:
        ...


@dataclass
class SteeringIntervention(Intervention):
    """Directional steering: h ← h + α·v_t

    Injects a J-lens direction into the activation at specified layers/positions.
    """

    direction: torch.Tensor = field(default_factory=lambda: torch.zeros(1))
    alpha: float = 1.0

    @property
    def intervention_type(self) -> InterventionType:
        return InterventionType.STEER

    def apply(self, h: torch.Tensor, layer: int, position: int) -> torch.Tensor:
        if layer not in self.layers:
            return h
        if self.positions is not None and position not in self.positions:
            return h

        direction = self.direction.to(h.device, dtype=h.dtype)
        if h.dim() == 1:
            return h + self.alpha * direction
        else:
            # h is (seq_len, d_model), apply at specific position
            h = h.clone()
            h[position] = h[position] + self.alpha * direction
            return h


@dataclass
class AblationIntervention(Intervention):
    """Ablation: project out specific J-lens directions or the full J-space component.

    Two modes:
    - Direction ablation: project out specific J-lens vectors
    - J-space ablation: zero the top-k most strongly activated J-lens projections
    """

    directions: list[torch.Tensor] = field(default_factory=list)
    alpha: float = -1.0  # Negative alpha for ablation

    @property
    def intervention_type(self) -> InterventionType:
        return InterventionType.ABLATE

    def apply(self, h: torch.Tensor, layer: int, position: int) -> torch.Tensor:
        if layer not in self.layers:
            return h
        if self.positions is not None and position not in self.positions:
            return h

        h_modified = h.clone()
        target = h_modified[position] if h.dim() > 1 else h_modified

        for direction in self.directions:
            d = direction.to(target.device, dtype=target.dtype)
            d_norm = d / (d.norm() + 1e-8)
            # Project out the component along this direction
            projection = torch.dot(target, d_norm) * d_norm
            target = target - projection  # Full ablation
            # Or with alpha: target = target + self.alpha * projection

        if h.dim() > 1:
            h_modified[position] = target
        else:
            h_modified = target

        return h_modified


@dataclass
class ConceptSwapIntervention(Intervention):
    """Concept-swap in lens coordinates: h_patched = h + V(σ(c) − c)

    Given source token s and target token t:
    - V = [v_s | v_t] ∈ ℝ^{d_model × 2}
    - c = V† h ∈ ℝ² (pseudoinverse projection)
    - σ swaps the two entries of c (optionally scaled by α)
    - The component of h orthogonal to span{v_s, v_t} is left untouched.
    """

    source_direction: torch.Tensor = field(default_factory=lambda: torch.zeros(1))
    target_direction: torch.Tensor = field(default_factory=lambda: torch.zeros(1))
    alpha: float = 1.0

    @property
    def intervention_type(self) -> InterventionType:
        return InterventionType.CONCEPT_SWAP

    def apply(self, h: torch.Tensor, layer: int, position: int) -> torch.Tensor:
        if layer not in self.layers:
            return h
        if self.positions is not None and position not in self.positions:
            return h

        h_modified = h.clone()
        target_h = h_modified[position] if h.dim() > 1 else h_modified

        v_s = self.source_direction.to(target_h.device, dtype=target_h.dtype)
        v_t = self.target_direction.to(target_h.device, dtype=target_h.dtype)

        # V = [v_s, v_t] as columns: shape (d_model, 2)
        V = torch.stack([v_s, v_t], dim=1)

        # c = V† h (pseudoinverse projection into 2D subspace)
        # V† = (V^T V)^{-1} V^T
        VtV = V.T @ V  # (2, 2)
        try:
            VtV_inv = torch.linalg.inv(VtV + 1e-8 * torch.eye(2, device=target_h.device))
        except torch.linalg.LinAlgError:
            return h  # Degenerate case, skip intervention

        V_pinv = VtV_inv @ V.T  # (2, d_model)
        c = V_pinv @ target_h  # (2,)

        # σ swaps the two entries, scaled by α
        sigma_c = torch.tensor([c[1], c[0]], device=target_h.device, dtype=target_h.dtype)
        sigma_c = c + self.alpha * (sigma_c - c)

        # h_patched = h + V(σ(c) − c)
        delta = V @ (sigma_c - c)  # (d_model,)
        patched = target_h + delta

        if h.dim() > 1:
            h_modified[position] = patched
        else:
            h_modified = patched

        return h_modified


def steer(
    h: torch.Tensor,
    direction: torch.Tensor,
    alpha: float = 1.0,
    layer: int = 0,
    position: int = 0,
) -> torch.Tensor:
    """Apply steering intervention: h ← h + α·v_t.

    Convenience function for single-shot steering.
    """
    intervention = SteeringIntervention(
        layers=[layer],
        positions=[position],
        direction=direction,
        alpha=alpha,
    )
    return intervention.apply(h, layer, position)


def ablate(
    h: torch.Tensor,
    directions: list[torch.Tensor],
    layer: int = 0,
    position: int = 0,
) -> torch.Tensor:
    """Apply ablation intervention: project out specified directions.

    Convenience function for single-shot ablation.
    """
    intervention = AblationIntervention(
        layers=[layer],
        positions=[position],
        directions=directions,
    )
    return intervention.apply(h, layer, position)


def concept_swap(
    h: torch.Tensor,
    source_direction: torch.Tensor,
    target_direction: torch.Tensor,
    alpha: float = 1.0,
    layer: int = 0,
    position: int = 0,
) -> torch.Tensor:
    """Apply concept-swap intervention: h_patched = h + V(σ(c) − c).

    Convenience function for single-shot concept swap.
    """
    intervention = ConceptSwapIntervention(
        layers=[layer],
        positions=[position],
        source_direction=source_direction,
        target_direction=target_direction,
        alpha=alpha,
    )
    return intervention.apply(h, layer, position)


class InterventionHook:
    """Runtime hook that applies interventions during model forward pass.

    Registers as a forward hook on transformer layers and applies
    configured interventions at the appropriate layers and positions.
    """

    def __init__(self, interventions: list[Intervention]) -> None:
        self.interventions = interventions
        self._hooks: list[Any] = []

    def register(self, model: nn.Module, layers: list[nn.Module]) -> None:
        """Register forward hooks on model layers."""
        for layer_idx, layer_module in enumerate(layers):
            handle = layer_module.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output

            # Apply all interventions for this layer
            for intervention in self.interventions:
                if layer_idx in intervention.layers:
                    seq_len = act.shape[1] if act.dim() == 3 else 1
                    for pos in range(seq_len):
                        if intervention.positions is None or pos in intervention.positions:
                            if act.dim() == 3:
                                act[0, pos] = intervention.apply(
                                    act[0], layer_idx, pos
                                )[pos]
                            else:
                                act = intervention.apply(act, layer_idx, pos)

            if isinstance(output, tuple):
                return (act,) + output[1:]
            return act

        return hook_fn

    def remove(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()
