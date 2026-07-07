"""Model Adapter — Model-agnostic interface for J-Lens operations.

Supports any HuggingFace-compatible decoder model. Follows the Hermes Agent
pattern of model-agnostic tool calling with adapter-based abstraction.

Supported architectures:
- Llama (Llama-3.2-1B/3B, etc.)
- Qwen (Qwen-2.5-1.5B/3B/7B, Qwen3-*)
- GPT-2 (all sizes)
- Gemma (Gemma-2/3)
- Mistral
- Pythia / GPT-NeoX
- DeepSeek
- Any HuggingFace CausalLM with standard architecture
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from prometheus.jlens.fitting import get_final_norm, get_model_layers, get_unembed

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model for J-Lens operations."""

    model_id: str
    architecture: str
    d_model: int
    n_layers: int
    vocab_size: int
    norm_type: str  # "rmsnorm" or "layernorm"
    dtype: torch.dtype
    device: torch.device


class ModelAdapter:
    """Model-agnostic adapter for J-Lens operations.

    Wraps a HuggingFace model and provides uniform access to:
    - Transformer layers
    - Final normalization
    - Unembedding matrix
    - Activation extraction
    - Forward pass with hooks
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: torch.device | str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        # Extract model components
        self._layers = get_model_layers(model)
        self._final_norm = get_final_norm(model)
        self._lm_head = get_unembed(model)

        # Model info
        config = model.config
        self.info = ModelInfo(
            model_id=getattr(config, "_name_or_path", "unknown"),
            architecture=type(model).__name__,
            d_model=config.hidden_size,
            n_layers=len(self._layers),
            vocab_size=config.vocab_size,
            norm_type=self._detect_norm_type(),
            dtype=next(model.parameters()).dtype,
            device=self.device,
        )

    def _detect_norm_type(self) -> str:
        """Detect whether the model uses RMSNorm or LayerNorm."""
        if self._final_norm is None:
            return "unknown"
        norm_type = type(self._final_norm).__name__.lower()
        if "rms" in norm_type:
            return "rmsnorm"
        elif "layer" in norm_type:
            return "layernorm"
        return "unknown"

    @property
    def layers(self) -> list[nn.Module]:
        return self._layers

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    @property
    def d_model(self) -> int:
        return self.info.d_model

    @property
    def vocab_size(self) -> int:
        return self.info.vocab_size

    @property
    def final_norm(self) -> nn.Module | None:
        return self._final_norm

    @property
    def unembed_weight(self) -> torch.Tensor:
        """Get the unembedding weight matrix W_U of shape (vocab_size, d_model)."""
        if self._lm_head is None:
            raise ValueError("Model does not have an identifiable lm_head.")
        return self._lm_head.weight.data

    def tokenize(
        self,
        text: str,
        max_length: int = 128,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Tokenize text with the model's tokenizer."""
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            max_length=max_length,
            truncation=True,
            padding=False,
        )

    def decode_tokens(self, token_ids: torch.Tensor | list[int]) -> list[str]:
        """Decode token IDs to strings."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return [self.tokenizer.decode([tid]) for tid in token_ids]

    def get_activations(
        self,
        input_ids: torch.Tensor,
        layers: list[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Run forward pass and capture activations at specified layers.

        Args:
            input_ids: Token IDs of shape (1, seq_len).
            layers: Layer indices to capture. Defaults to all.

        Returns:
            Dict mapping layer index to activation tensor (1, seq_len, d_model).
        """
        if layers is None:
            layers = list(range(self.n_layers))

        activations: dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                activations[layer_idx] = act.detach().cpu()
            return hook_fn

        for layer_idx in layers:
            h = self._layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        try:
            with torch.no_grad():
                self.model(input_ids.to(self.device))
        finally:
            for h in hooks:
                h.remove()

        return activations

    def apply_lens_readout(
        self,
        transformed_h: torch.Tensor,
    ) -> torch.Tensor:
        """Apply final norm + unembedding to get logits from J-lens transformed activation.

        Args:
            transformed_h: Tensor of shape (..., d_model) after J_ℓ multiplication.

        Returns:
            Logits of shape (..., vocab_size).
        """
        h = transformed_h.to(self.device, dtype=self.info.dtype)

        if self._final_norm is not None:
            h = self._final_norm(h)

        # Unembed
        W_U = self.unembed_weight.to(h.device, dtype=h.dtype)
        logits = torch.matmul(h, W_U.T)

        return logits

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
    ) -> ModelAdapter:
        """Load a model from HuggingFace and create an adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            device: Target device.
            dtype: Model dtype (default: auto-detect).
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes).
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes).
            trust_remote_code: Allow custom model code.

        Returns:
            Configured ModelAdapter instance.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {model_name_or_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            else:
                model_kwargs["torch_dtype"] = "auto"

        if not (load_in_4bit or load_in_8bit):
            model_kwargs["device_map"] = device

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        if load_in_4bit or load_in_8bit:
            actual_device = next(model.parameters()).device
        else:
            actual_device = torch.device(device)

        return cls(model=model, tokenizer=tokenizer, device=actual_device)
