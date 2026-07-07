"""Jacobian Lens — core lens object: config, readout, persistence, merging.

The J-lens transports a residual-stream vector at any source layer into the
final-layer basis using an average input-output Jacobian, then decodes with
the model's own unembedding:

    lens(h_l) = softmax(W_U · norm(J_l · h_l))

where ``J_l = E[∂h_final,t' / ∂h_l,t]`` is a single ``d_model × d_model``
matrix per source layer, estimated over a corpus of pretraining-like prompts
(see ``prometheus.jlens.fitting``). When ``J_l = I`` this reduces to the
classical logit lens.

J-lens *vectors* are the rows of ``W_U · J_l`` — one residual-stream
direction per vocabulary token, forming an overcomplete frame
(``|V| > d_model``). They are the dictionary atoms for J-space decomposition
(``prometheus.jlens.jspace``) and the directions used by interventions
(``prometheus.jlens.interventions``).

This module was reconstructed from the method guide (docs/jlens-method-guide.md)
and the consumer contract exercised by fitting, the MCP server, the CLI, and
the test suite.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

LENS_FORMAT_VERSION = 1


@dataclass
class LensConfig:
    """Configuration identifying a fitted Jacobian Lens.

    The identity-bearing fields (model, dimensions, layer selection, and the
    estimator hyperparameters that change the fitted matrices) feed the
    deterministic ``lens_id()``. Two lenses fitted with the same config are
    interchangeable and mergeable.
    """

    model_id: str
    d_model: int
    n_layers: int
    source_layers: list[int]
    target_layer: int
    skip_first_n_positions: int = 16
    max_seq_len: int = 128
    dim_batch: int = 8
    n_prompts: int = 100

    def lens_id(self) -> str:
        """Deterministic identifier derived from identity-bearing config fields."""
        payload = json.dumps(
            {
                "model_id": self.model_id,
                "d_model": self.d_model,
                "n_layers": self.n_layers,
                "source_layers": [int(x) for x in self.source_layers],
                "target_layer": int(self.target_layer),
                "skip_first_n_positions": int(self.skip_first_n_positions),
                "max_seq_len": int(self.max_seq_len),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"jlens-{digest}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LensConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class JacobianLens:
    """A fitted Jacobian Lens: one ``d_model × d_model`` matrix per source layer.

    Args:
        config: The :class:`LensConfig` describing how the lens was fitted.
        matrices: Mapping ``layer_index -> J_l`` tensor of shape
            ``(d_model, d_model)`` where ``J_l[out, in]`` maps source-layer
            residual coordinates to final-layer coordinates.
        metadata: Optional free-form provenance (n_prompts_processed,
            fit duration, device, corpus name, ...).
    """

    def __init__(
        self,
        config: LensConfig,
        matrices: dict[int, torch.Tensor],
        metadata: dict[str, Any] | None = None,
    ):
        self.config = config
        self.matrices: dict[int, torch.Tensor] = {int(k): v for k, v in matrices.items()}
        self.metadata: dict[str, Any] = dict(metadata) if metadata else {}

        for layer, J in self.matrices.items():
            if J.shape != (config.d_model, config.d_model):
                raise ValueError(
                    f"Matrix for layer {layer} has shape {tuple(J.shape)}, "
                    f"expected ({config.d_model}, {config.d_model})"
                )

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    @property
    def lens_id(self) -> str:
        """Deterministic lens identifier (same config -> same id)."""
        return self.config.lens_id()

    @property
    def layers(self) -> list[int]:
        """Source layers this lens covers, sorted ascending."""
        return sorted(self.matrices.keys())

    # ------------------------------------------------------------------ #
    # Readout
    # ------------------------------------------------------------------ #

    def apply(
        self,
        h: torch.Tensor,
        layer: int,
        norm_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        unembed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the J-lens readout at a source layer.

        Computes ``t = J_l · h`` (the transported final-layer residual), then
        optionally applies the model's final normalisation and unembedding:
        ``logits = W_U · norm(J_l · h)``.

        Args:
            h: Residual-stream activations of shape ``(..., d_model)``.
            layer: Source layer index; must be one of the fitted layers.
            norm_fn: The model's own final norm (LayerNorm/RMSNorm) or ``None``.
            unembed: Unembedding weight ``W_U`` of shape ``(vocab, d_model)``,
                or ``None`` to return the transported hidden state.

        Returns:
            Logits of shape ``(..., vocab)`` if ``unembed`` is given, else the
            transported hidden state of shape ``(..., d_model)``.
        """
        layer = int(layer)
        if layer not in self.matrices:
            raise ValueError(
                f"Layer {layer} not in lens (available layers: {self.layers})"
            )

        J = self.matrices[layer]
        # Compute on the activation's device, promoting to the J dtype (fp32)
        # for numerical stability of the readout.
        J_local = J.to(device=h.device)
        t = h.to(dtype=J_local.dtype) @ J_local.T  # (..., d_model)

        if norm_fn is not None:
            norm_dtype = None
            if isinstance(norm_fn, torch.nn.Module):
                try:
                    norm_dtype = next(norm_fn.parameters()).dtype
                except StopIteration:
                    norm_dtype = None
            if norm_dtype is not None:
                t = norm_fn(t.to(norm_dtype))
            else:
                t = norm_fn(t)

        if unembed is None:
            return t

        W = unembed.detach().to(device=t.device)
        return t.to(W.dtype) @ W.T  # (..., vocab)

    def top_k_tokens(
        self,
        h: torch.Tensor,
        layer: int,
        unembed: torch.Tensor,
        k: int = 10,
        norm_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k vocabulary tokens the activation is disposed to make the model say.

        Returns:
            ``(values, indices)`` each of shape ``(..., k)``.
        """
        logits = self.apply(h, layer, norm_fn=norm_fn, unembed=unembed)
        k = min(k, logits.shape[-1])
        return logits.topk(k, dim=-1)

    def get_jlens_vectors(
        self,
        layer: int,
        unembed: torch.Tensor,
    ) -> torch.Tensor:
        """J-lens vectors at a layer: rows of ``W_U · J_l``.

        One residual-stream direction per vocabulary token — the overcomplete
        dictionary used for J-space decomposition and per-token probes.

        Returns:
            Tensor of shape ``(vocab, d_model)``.
        """
        layer = int(layer)
        if layer not in self.matrices:
            raise ValueError(
                f"Layer {layer} not in lens (available layers: {self.layers})"
            )
        J = self.matrices[layer]
        W = unembed.detach().to(device=J.device, dtype=J.dtype)
        return W @ J  # (vocab, d_model)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Serialize the lens (config + matrices + metadata) to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": LENS_FORMAT_VERSION,
            "lens_id": self.lens_id,
            "config": self.config.to_dict(),
            "matrices": {int(k): v.detach().cpu() for k, v in self.matrices.items()},
            "metadata": self.metadata,
        }
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        tmp.rename(path)  # atomic
        logger.info("Saved J-lens %s to %s", self.lens_id, path)

    @classmethod
    def load(cls, path: str | Path) -> "JacobianLens":
        """Load a lens previously written by :meth:`save`."""
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        config = LensConfig.from_dict(data["config"])
        matrices = {int(k): v for k, v in data["matrices"].items()}
        return cls(config=config, matrices=matrices, metadata=data.get("metadata", {}))

    # Alias matching the reference repo's API.
    from_pretrained = load

    # ------------------------------------------------------------------ #
    # Merging (data-parallel fitting across disjoint prompt shards)
    # ------------------------------------------------------------------ #

    @classmethod
    def merge(cls, lenses: list["JacobianLens"]) -> "JacobianLens":
        """Combine shard lenses by weighted mean.

        Weights come from ``metadata['n_prompts_processed']`` when present
        (so shards with more prompts count proportionally), defaulting to 1,
        which reduces to a simple average for equal shards.
        """
        if not lenses:
            raise ValueError("Cannot merge an empty list of lenses")

        first = lenses[0]
        ref_id = first.lens_id
        for lens in lenses[1:]:
            if lens.lens_id != ref_id:
                raise ValueError(
                    f"Cannot merge lenses with different configs: "
                    f"{lens.lens_id} != {ref_id}"
                )
            if set(lens.matrices.keys()) != set(first.matrices.keys()):
                raise ValueError("Cannot merge lenses covering different layers")

        weights = [float(l.metadata.get("n_prompts_processed", 1) or 1) for l in lenses]
        total = sum(weights)

        merged_matrices: dict[int, torch.Tensor] = {}
        for layer in first.matrices:
            acc = torch.zeros_like(first.matrices[layer], dtype=torch.float32)
            for lens, w in zip(lenses, weights):
                acc += lens.matrices[layer].to(torch.float32) * w
            merged_matrices[layer] = acc / total

        merged_meta: dict[str, Any] = {
            "merged_from": len(lenses),
            "n_prompts_processed": int(
                sum(l.metadata.get("n_prompts_processed", 0) for l in lenses)
            ),
        }
        return cls(config=first.config, matrices=merged_matrices, metadata=merged_meta)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"JacobianLens(lens_id={self.lens_id!r}, model={self.config.model_id!r}, "
            f"layers={self.layers}, d_model={self.config.d_model})"
        )
