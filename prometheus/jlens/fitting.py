"""J-Lens Fitting — Estimate the averaged Jacobian via reverse-mode VJP.

Implements the estimator from Anthropic's reference code:
- For each output dimension, inject a one-hot cotangent at every valid target position
  at once and backprop.
- The gradient at source position p is Σ_{p' ≥ p} ∂h_final[p']/∂h_l[p].
- Take the mean over source positions p.
- Positions with index < SKIP_FIRST_N_POSITIONS are excluded as attention sinks.
- Prompts are truncated to max_seq_len = 128.

Cost per prompt: 1 forward + ⌈d_model / dim_batch⌉ backwards.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn

from prometheus.jlens.lens import JacobianLens, LensConfig

logger = logging.getLogger(__name__)

SKIP_FIRST_N_POSITIONS = 16
MAX_SEQ_LEN = 128
DIM_BATCH = 8


@dataclass
class FitProgress:
    """Progress report during fitting."""

    prompt_idx: int
    total_prompts: int
    elapsed_seconds: float
    estimated_remaining_seconds: float


class ActivationRecorder:
    """Hook-based activation recorder for capturing residual stream states.

    Registers forward hooks on specified layers to capture activations during
    the forward pass. Compatible with any HuggingFace-style transformer.
    """

    def __init__(self) -> None:
        self.activations: dict[str, torch.Tensor] = {}
        self._hooks: list[Any] = []

    def register(self, module: nn.Module, name: str) -> None:
        """Register a forward hook on a module to capture its output."""

        def hook_fn(mod: nn.Module, input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                self.activations[name] = output[0]
            else:
                self.activations[name] = output

        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def clear(self) -> None:
        """Clear recorded activations."""
        self.activations.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        self.remove_hooks()


def get_model_layers(model: nn.Module) -> list[nn.Module]:
    """Extract the transformer layer modules from a HuggingFace model.

    Supports common architectures: LlamaForCausalLM, GPT2LMHeadModel,
    Qwen2ForCausalLM, GemmaForCausalLM, MistralForCausalLM, etc.
    """
    # Try common attribute paths
    for attr_path in [
        "model.layers",       # Llama, Qwen, Mistral, Gemma
        "transformer.h",      # GPT-2, GPT-Neo
        "gpt_neox.layers",    # GPT-NeoX, Pythia
        "model.decoder.layers",  # OPT
    ]:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue

    raise ValueError(
        f"Cannot find transformer layers in model of type {type(model).__name__}. "
        "Please provide a custom layer extraction function."
    )


def get_final_norm(model: nn.Module) -> nn.Module | None:
    """Extract the final normalization layer from a HuggingFace model."""
    for attr_path in [
        "model.norm",          # Llama, Qwen, Mistral, Gemma
        "transformer.ln_f",    # GPT-2
        "gpt_neox.final_layer_norm",  # Pythia
        "model.decoder.final_layer_norm",  # OPT
    ]:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def get_unembed(model: nn.Module) -> nn.Module | None:
    """Extract the unembedding (lm_head) from a HuggingFace model."""
    for attr in ["lm_head", "embed_out", "output"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    return None


def jacobian_for_prompt(
    model: nn.Module,
    input_ids: torch.Tensor,
    source_layers: list[int],
    target_layer: int,
    d_model: int,
    dim_batch: int = DIM_BATCH,
    skip_first_n: int = SKIP_FIRST_N_POSITIONS,
    device: torch.device | str = "cuda",
) -> dict[int, torch.Tensor]:
    """Compute the averaged Jacobian for a single prompt.

    Reverse-mode VJP with structured cotangents, following the reference
    design: expand the input to a batch of ``dim_batch`` copies, run a single
    forward pass, then run ``ceil(d_model / dim_batch)`` backward passes. In
    backward pass ``b``, copy ``i`` of the batch carries a one-hot cotangent
    on output dimension ``b * dim_batch + i``, summed over all valid target
    positions. Because the model is causal, the gradient at source position
    ``p`` is automatically the sum over target positions ``p' >= p``:

        grad[p] = sum_{p' >= p} d h_final[p'] / d h_l[p]

    Rows of the Jacobian are the per-dimension gradients averaged over valid
    source positions ``[skip_first_n, seq_len - 1)`` (attention sinks and the
    final position are excluded).

    Args:
        model: HuggingFace causal LM model.
        input_ids: Token IDs of shape (1, seq_len), already truncated.
        source_layers: Layer indices to compute Jacobians for.
        target_layer: Target layer index (typically n_layers - 1); the
            captured activation is the pre-final-norm residual stream.
        d_model: Model hidden dimension.
        dim_batch: Output dimensions per backward pass (= batch size).
        skip_first_n: Initial positions to exclude (attention sinks).
        device: Computation device.

    Returns:
        Dict mapping layer index -> Jacobian (d_model, d_model), CPU fp32,
        with J[out, in] = mean_p sum_{p' >= p} d h_final[p', out] / d h_l[p, in].
    """
    model.eval()
    seq_len = input_ids.shape[1]

    # Valid source positions: [skip_first_n, seq_len - 1)
    # (the last position has no future target)
    valid_src_start = skip_first_n
    valid_src_end = seq_len - 1
    n_valid_src = valid_src_end - valid_src_start

    jacobians: dict[int, torch.Tensor] = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers
    }

    if n_valid_src <= 0:
        logger.warning(
            f"No valid source positions (seq_len={seq_len}, skip={skip_first_n}). "
            "Returning zero matrices."
        )
        return jacobians

    layers = get_model_layers(model)

    # Record source and target activations WITHOUT detaching, so the autograd
    # graph from every source layer through the target stays intact.
    source_activations: dict[int, torch.Tensor] = {}
    target_activation: list[torch.Tensor] = []

    def make_source_hook(layer_idx: int):
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            act = output[0] if isinstance(output, tuple) else output
            source_activations[layer_idx] = act
        return hook

    def target_hook(module: nn.Module, inputs: Any, output: Any) -> None:
        act = output[0] if isinstance(output, tuple) else output
        target_activation.append(act)

    handles = [
        layers[layer_idx].register_forward_hook(make_source_hook(layer_idx))
        for layer_idx in source_layers
    ]
    if target_layer not in source_layers:
        handles.append(layers[target_layer].register_forward_hook(target_hook))

    n_backward_batches = (d_model + dim_batch - 1) // dim_batch

    try:
        # One forward pass over dim_batch identical copies of the prompt.
        batch_ids = input_ids.to(device).repeat(dim_batch, 1)
        with torch.enable_grad():
            _ = model(batch_ids)

        if target_layer in source_layers:
            target_act = source_activations[target_layer]
        else:
            if not target_activation:
                raise RuntimeError(
                    f"Target layer {target_layer} activation was not captured."
                )
            target_act = target_activation[0]  # (dim_batch, seq_len, d_model)

        grad_inputs = [source_activations[layer_idx] for layer_idx in source_layers]

        for batch_idx in range(n_backward_batches):
            dim_start = batch_idx * dim_batch
            dims = list(range(dim_start, min(dim_start + dim_batch, d_model)))

            # Copy i carries a one-hot cotangent on dimension dims[i],
            # injected at every valid target position at once.
            cotangent = torch.zeros_like(target_act)
            for i, dim_idx in enumerate(dims):
                cotangent[i, valid_src_start:seq_len, dim_idx] = 1.0

            grads = torch.autograd.grad(
                outputs=target_act,
                inputs=grad_inputs,
                grad_outputs=cotangent,
                retain_graph=(batch_idx < n_backward_batches - 1),
                allow_unused=True,
            )

            for layer_idx, grad in zip(source_layers, grads):
                if grad is None:
                    continue
                # grad: (dim_batch, seq_len, d_model). Row for dims[i] is
                # copy i's gradient, averaged over valid source positions.
                grad_valid = grad[: len(dims), valid_src_start:valid_src_end, :]
                rows = grad_valid.mean(dim=1)  # (n_dims, d_model)
                jacobians[layer_idx][dims[0] : dims[0] + len(dims)] = (
                    rows.detach().float().cpu()
                )
    finally:
        for h in handles:
            h.remove()
        source_activations.clear()
        target_activation.clear()

    return jacobians


def _jacobian_for_prompt_hookbased(
    model: nn.Module,
    input_ids: torch.Tensor,
    source_layers: list[int],
    target_layer: int,
    d_model: int,
    dim_batch: int = DIM_BATCH,
    skip_first_n: int = SKIP_FIRST_N_POSITIONS,
    device: torch.device | str = "cuda",
) -> dict[int, torch.Tensor]:
    """Backward-compatible alias for :func:`jacobian_for_prompt`."""
    return jacobian_for_prompt(
        model=model,
        input_ids=input_ids,
        source_layers=source_layers,
        target_layer=target_layer,
        d_model=d_model,
        dim_batch=dim_batch,
        skip_first_n=skip_first_n,
        device=device,
    )



def fit_jacobian_lens(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str] | Iterator[str],
    source_layers: list[int] | None = None,
    target_layer: int | None = None,
    n_prompts: int = 100,
    dim_batch: int = DIM_BATCH,
    skip_first_n: int = SKIP_FIRST_N_POSITIONS,
    max_seq_len: int = MAX_SEQ_LEN,
    device: torch.device | str = "cuda",
    checkpoint_dir: str | Path | None = None,
    progress_callback: Callable[[FitProgress], None] | None = None,
) -> JacobianLens:
    """Fit a Jacobian Lens by averaging over a corpus of prompts.

    Args:
        model: HuggingFace causal LM model.
        tokenizer: The model's tokenizer.
        prompts: Iterator or list of text prompts (will be truncated to max_seq_len).
        source_layers: Layers to compute Jacobians for. Defaults to all interior layers.
        target_layer: Target layer index. Defaults to n_layers - 1.
        n_prompts: Number of prompts to average over.
        dim_batch: Batch size for backward passes (output dimensions per pass).
        skip_first_n: Positions to skip at start (attention sinks).
        max_seq_len: Maximum sequence length for prompts.
        device: Computation device.
        checkpoint_dir: If set, save intermediate results for resumption.
        progress_callback: Optional callback for progress reporting.

    Returns:
        Fitted JacobianLens instance.
    """
    layers = get_model_layers(model)
    n_layers = len(layers)
    d_model = model.config.hidden_size

    if source_layers is None:
        source_layers = list(range(n_layers))
    if target_layer is None:
        target_layer = n_layers - 1

    config = LensConfig(
        model_id=getattr(model.config, "_name_or_path", "unknown"),
        d_model=d_model,
        n_layers=n_layers,
        source_layers=source_layers,
        target_layer=target_layer,
        skip_first_n_positions=skip_first_n,
        max_seq_len=max_seq_len,
        dim_batch=dim_batch,
        n_prompts=n_prompts,
    )

    # Accumulators
    jacobian_sum: dict[int, torch.Tensor] = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers
    }

    start_time = time.time()
    prompts_iter = iter(prompts)
    processed = 0

    for i in range(n_prompts):
        try:
            prompt = next(prompts_iter)
        except StopIteration:
            logger.warning(f"Ran out of prompts at {i}/{n_prompts}. Using {i} prompts.")
            break

        # Tokenize
        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_seq_len,
            truncation=True,
            padding=False,
        )
        input_ids = tokens["input_ids"]

        if input_ids.shape[1] < skip_first_n + 2:
            logger.debug(f"Prompt {i} too short ({input_ids.shape[1]} tokens), skipping.")
            continue

        # Compute Jacobian for this prompt
        prompt_jacobians = _jacobian_for_prompt_hookbased(
            model=model,
            input_ids=input_ids,
            source_layers=source_layers,
            target_layer=target_layer,
            d_model=d_model,
            dim_batch=dim_batch,
            skip_first_n=skip_first_n,
            device=device,
        )

        # Accumulate
        for layer_idx, J in prompt_jacobians.items():
            jacobian_sum[layer_idx] += J

        processed += 1

        # Progress callback
        if progress_callback:
            elapsed = time.time() - start_time
            rate = elapsed / processed if processed > 0 else 0
            remaining = rate * (n_prompts - processed)
            progress_callback(
                FitProgress(
                    prompt_idx=processed,
                    total_prompts=n_prompts,
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=remaining,
                )
            )

        # Checkpoint
        if checkpoint_dir and processed % 10 == 0:
            _save_checkpoint(jacobian_sum, processed, config, Path(checkpoint_dir))

    # Average
    if processed > 0:
        matrices = {layer: J / processed for layer, J in jacobian_sum.items()}
    else:
        matrices = jacobian_sum

    elapsed = time.time() - start_time
    lens = JacobianLens(
        config=config,
        matrices=matrices,
        metadata={
            "n_prompts_processed": processed,
            "fit_duration_seconds": elapsed,
            "device": str(device),
        },
    )

    logger.info(
        f"J-lens fitted: {processed} prompts, {len(source_layers)} layers, "
        f"{elapsed:.1f}s total"
    )

    return lens


def _save_checkpoint(
    jacobian_sum: dict[int, torch.Tensor],
    n_processed: int,
    config: LensConfig,
    checkpoint_dir: Path,
) -> None:
    """Save a fitting checkpoint for resumption."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "jacobian_sum": jacobian_sum,
        "n_processed": n_processed,
        "config": config,
    }
    tmp_path = checkpoint_dir / "checkpoint.tmp"
    final_path = checkpoint_dir / "checkpoint.pt"
    torch.save(checkpoint, tmp_path)
    tmp_path.rename(final_path)  # Atomic rename


# --------------------------------------------------------------------------- #
# High-level fitter (CLI-facing wrapper around fit_jacobian_lens)
# --------------------------------------------------------------------------- #

# A small built-in pretraining-like corpus used when the caller supplies no
# prompts. Adequate for smoke-fitting and algorithmic debugging; for real
# experiments pass 50-200 prompts sampled from The Pile / C4 / SlimPajama.
DEFAULT_FIT_PROMPTS: list[str] = [
    "The history of the printing press begins in the fifteenth century, when movable type first allowed books to be produced at scale and literacy spread rapidly across Europe. Monasteries that had spent generations copying manuscripts by hand suddenly found their scriptoriums obsolete, and a new trade of printers, typesetters, and booksellers emerged in cities along the major trade routes.",
    "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen. The light-dependent reactions occur in the thylakoid membranes, where chlorophyll absorbs photons and drives the synthesis of ATP and NADPH, which then power the Calvin cycle in the stroma.",
    "In the quarterly earnings call, the chief financial officer explained that revenue growth had slowed due to currency headwinds in international markets, but that operating margins improved as the company completed its transition to a subscription-based model and reduced its dependence on one-time license sales.",
    "The recipe calls for two cups of all-purpose flour, a teaspoon of baking soda, half a teaspoon of salt, and a stick of unsalted butter softened to room temperature. Cream the butter with both sugars until light and fluffy, then beat in the eggs one at a time before folding in the dry ingredients.",
    "When the expedition reached the ridge, the weather turned suddenly, and the climbers were forced to establish an emergency bivouac below the summit pyramid. Wind speeds exceeded eighty kilometers per hour through the night, and by morning two members of the team showed early signs of frostbite.",
    "The city council voted on Tuesday to approve the new transit plan, which includes dedicated bus lanes on the main corridor, a redesigned downtown interchange, and a pilot program for on-demand shuttles in neighborhoods that lack fixed-route service. Opponents argued the plan underfunds road maintenance.",
    "A hash table stores key-value pairs and offers average-case constant-time lookup by computing an index from the key. Collisions, where two keys map to the same bucket, are handled either by chaining, in which each bucket holds a linked list, or by open addressing, in which the table probes for the next free slot.",
    "The novel opens in a coastal village where the narrator, a retired schoolteacher, spends her mornings cataloguing the letters her late husband left behind. Each letter reveals a fragment of a life she thought she knew, and the book's central tension grows from the widening gap between memory and record.",
    "Under contract law, an offer becomes binding when it is accepted without modification and supported by consideration. If the offeree proposes new terms, the original offer is extinguished and replaced by a counteroffer, which the original offeror is free to accept or reject.",
    "Astronomers announced the discovery of an exoplanet orbiting within the habitable zone of a nearby red dwarf star. Transit photometry suggests a radius about one and a half times that of Earth, and follow-up spectroscopy will search the planet's atmosphere for water vapor and other biosignature gases.",
    "The manufacturing process begins with cold-rolled steel coils that are cleaned, cut to length, and stamped into panels. Robotic welders join the panels into subassemblies, which travel by overhead conveyor to the paint shop, where they receive a corrosion-resistant primer before the color coats are applied.",
    "Immunologists have long studied how memory B cells enable a faster and stronger antibody response upon reinfection. After the primary response subsides, a small population of these cells persists in lymphoid tissue, ready to differentiate into plasma cells within days of encountering the same antigen.",
]


class JLensFitter:
    """High-level fitter used by the CLI: wraps :func:`fit_jacobian_lens`.

    Args:
        adapter: A ``ModelAdapter`` wrapping the target model + tokenizer.
        n_samples: Number of prompts to average over.
        batch_size: Output dimensions per backward pass (``dim_batch``).
        prompts: Optional prompt corpus; defaults to a small built-in
            pretraining-like set (cycled to ``n_samples``).
        source_layers / target_layer / checkpoint_dir / progress_callback:
            Forwarded to :func:`fit_jacobian_lens`.
    """

    def __init__(
        self,
        adapter: Any,
        n_samples: int = 100,
        batch_size: int = DIM_BATCH,
        prompts: list[str] | None = None,
        source_layers: list[int] | None = None,
        target_layer: int | None = None,
        checkpoint_dir: str | Path | None = None,
        progress_callback: Callable[[FitProgress], None] | None = None,
    ):
        self.adapter = adapter
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.prompts = prompts
        self.source_layers = source_layers
        self.target_layer = target_layer
        self.checkpoint_dir = checkpoint_dir
        self.progress_callback = progress_callback

    def _prompt_iter(self) -> Iterator[str]:
        corpus = self.prompts if self.prompts else DEFAULT_FIT_PROMPTS
        i = 0
        while True:
            yield corpus[i % len(corpus)]
            i += 1

    def fit_sync(self) -> JacobianLens:
        """Blocking fit."""
        device = getattr(self.adapter, "device", "cpu")
        return fit_jacobian_lens(
            model=self.adapter.model,
            tokenizer=self.adapter.tokenizer,
            prompts=self._prompt_iter(),
            source_layers=self.source_layers,
            target_layer=self.target_layer,
            n_prompts=self.n_samples,
            dim_batch=self.batch_size,
            device=device,
            checkpoint_dir=self.checkpoint_dir,
            progress_callback=self.progress_callback,
        )

    async def fit(self) -> JacobianLens:
        """Async fit — runs the blocking estimator in a worker thread."""
        import asyncio

        return await asyncio.to_thread(self.fit_sync)
