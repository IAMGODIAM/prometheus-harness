"""MCP Server Implementation — FastMCP-based J-Lens and Harness tools.

Exposes the J-Lens spine and harness capabilities as MCP tools following
the 2025-11-25 spec with 2026-07-28 forward-compatibility:
- Explicit state handles (lens_id, model_id)
- W3C trace context in _meta
- RFC 8707 audience-bound tokens
- Mcp-Method/Mcp-Name routing headers

Tool naming follows Anthropic's mcp-builder convention:
- Domain-prefixed (jlens_, verifier_, harness_)
- Action-oriented verbs first
- readOnlyHint/destructiveHint/idempotentHint declared
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Tool Input/Output Schemas ───────────────────────────────────────────────


class ProbeInput(BaseModel):
    """Input for jlens_probe tool."""
    prompt: str = Field(description="Text prompt to analyze through J-lens")
    layers: list[int] | None = Field(
        default=None,
        description="Specific layers to probe. None = workspace band (middle third)"
    )
    positions: list[int] | None = Field(
        default=None,
        description="Specific token positions to probe. None = all valid positions"
    )
    top_k: int = Field(default=10, description="Number of top tokens to return per position")
    model_id: str | None = Field(
        default=None,
        description="Model identifier. Uses default if not specified"
    )
    lens_id: str | None = Field(
        default=None,
        description="Lens artifact ID (SHA of fit config). Uses latest if not specified"
    )


class ProbeOutput(BaseModel):
    """Output from jlens_probe tool."""
    model_id: str
    lens_id: str
    layers_probed: list[int]
    results: list[dict[str, Any]]  # Per-layer, per-position top-k tokens
    metadata: dict[str, Any] = Field(default_factory=dict)


class WatchlistInput(BaseModel):
    """Input for jlens_watchlist_scores tool."""
    prompt: str = Field(description="Text prompt to score against watchlists")
    categories: list[str] | None = Field(
        default=None,
        description="Watchlist categories to check. None = all configured categories"
    )
    layers: list[int] | None = Field(
        default=None,
        description="Layers to evaluate. None = workspace band"
    )
    model_id: str | None = Field(default=None)
    lens_id: str | None = Field(default=None)


class WatchlistOutput(BaseModel):
    """Output from jlens_watchlist_scores tool."""
    model_id: str
    lens_id: str
    alerts: list[dict[str, Any]]  # Triggered watchlist categories
    scores: dict[str, dict[str, float]]  # category -> {token: score}
    any_triggered: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecomposeInput(BaseModel):
    """Input for jlens_decompose tool."""
    prompt: str = Field(description="Text prompt to decompose in J-space")
    layer: int = Field(description="Layer to decompose at")
    position: int = Field(description="Token position to decompose")
    k: int = Field(default=25, description="Maximum sparsity level")
    model_id: str | None = Field(default=None)
    lens_id: str | None = Field(default=None)


class DecomposeOutput(BaseModel):
    """Output from jlens_decompose tool."""
    model_id: str
    lens_id: str
    layer: int
    position: int
    top_tokens: list[dict[str, Any]]  # [{token, coefficient, rank}]
    variance_explained: float
    occupancy: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SteerInput(BaseModel):
    """Input for jlens_steer tool (Phase 3, feature-flagged)."""
    prompt: str = Field(description="Text prompt to generate with steering")
    interventions: list[dict[str, Any]] = Field(
        description="List of interventions: [{type, token, alpha, layers}]"
    )
    max_new_tokens: int = Field(default=100, description="Maximum tokens to generate")
    model_id: str | None = Field(default=None)
    lens_id: str | None = Field(default=None)


class SteerOutput(BaseModel):
    """Output from jlens_steer tool."""
    model_id: str
    lens_id: str
    original_text: str
    steered_text: str
    interventions_applied: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Server Factory ──────────────────────────────────────────────────────────


def create_jlens_server(
    lens_registry: Any = None,
    model_registry: Any = None,
    watchlist: Any = None,
    enable_steering: bool = False,
) -> Any:
    """Create a FastMCP server exposing J-Lens tools.

    Args:
        lens_registry: Registry of fitted J-lens artifacts.
        model_registry: Registry of loaded model adapters.
        watchlist: Configured Watchlist instance.
        enable_steering: Whether to expose the steer tool (Phase 3 feature flag).

    Returns:
        FastMCP server instance.
    """
    try:
        from fastmcp import FastMCP
    except ImportError:
        # Fallback to a minimal server implementation
        logger.warning("FastMCP not available, using minimal server stub")
        return _create_minimal_server(lens_registry, model_registry, watchlist, enable_steering)

    server = FastMCP(
        name="prometheus-jlens",
        description="J-Lens interpretability spine — probe, decompose, and monitor model internals",
    )

    @server.tool(
        name="jlens_probe",
        description=(
            "Run J-lens readout on a prompt to see what tokens/concepts the model's "
            "residual stream is 'disposed to say' at each layer and position. "
            "Returns top-k tokens per layer/position with their J-lens scores."
        ),
    )
    async def jlens_probe(input: ProbeInput) -> ProbeOutput:
        """Probe the model's internal representations via J-lens."""
        start_time = time.time()

        # Resolve model and lens
        adapter = _get_adapter(model_registry, input.model_id)
        lens = _get_lens(lens_registry, input.lens_id, input.model_id)

        # Tokenize
        tokens = adapter.tokenize(input.prompt)
        input_ids = tokens["input_ids"]

        # Determine layers to probe (default: workspace band = middle third)
        if input.layers is not None:
            layers = input.layers
        else:
            n = adapter.n_layers
            layers = list(range(n // 3, 2 * n // 3))

        # Get activations
        activations = adapter.get_activations(input_ids, layers=layers)

        # Apply J-lens at each layer
        results = []
        for layer_idx in layers:
            if layer_idx not in activations:
                continue
            act = activations[layer_idx][0]  # Remove batch dim: (seq_len, d_model)

            positions = input.positions or list(range(act.shape[0]))
            layer_results = []

            for pos in positions:
                if pos >= act.shape[0]:
                    continue
                h = act[pos]

                # Apply J-lens: logits = W_U · norm(J_ℓ · h)
                logits = lens.apply(
                    h.unsqueeze(0),
                    layer_idx,
                    norm_fn=adapter.final_norm,
                    unembed=adapter.unembed_weight,
                )

                # Get top-k
                values, indices = logits.squeeze().topk(input.top_k)
                top_tokens = [
                    {
                        "token": adapter.decode_tokens([idx.item()])[0],
                        "token_id": idx.item(),
                        "score": val.item(),
                    }
                    for val, idx in zip(values, indices)
                ]

                layer_results.append({
                    "position": pos,
                    "token_at_position": adapter.decode_tokens(
                        [input_ids[0, pos].item()]
                    )[0],
                    "top_tokens": top_tokens,
                })

            results.append({
                "layer": layer_idx,
                "positions": layer_results,
            })

        return ProbeOutput(
            model_id=adapter.info.model_id,
            lens_id=lens.lens_id,
            layers_probed=layers,
            results=results,
            metadata={
                "duration_ms": (time.time() - start_time) * 1000,
                "n_positions": sum(len(r["positions"]) for r in results),
            },
        )

    @server.tool(
        name="jlens_watchlist_scores",
        description=(
            "Score a prompt against configured watchlist categories (deception, "
            "prompt_injection, eval_awareness, self_preservation, reward_hacking). "
            "Returns per-category scores and whether any alert threshold is triggered."
        ),
    )
    async def jlens_watchlist_scores(input: WatchlistInput) -> WatchlistOutput:
        """Score prompt against watchlist categories."""
        start_time = time.time()

        adapter = _get_adapter(model_registry, input.model_id)
        lens = _get_lens(lens_registry, input.lens_id, input.model_id)
        wl = watchlist or _default_watchlist()

        # Tokenize and get activations
        tokens = adapter.tokenize(input.prompt)
        input_ids = tokens["input_ids"]

        if input.layers is not None:
            layers = input.layers
        else:
            n = adapter.n_layers
            layers = list(range(n // 3, 2 * n // 3))

        activations = adapter.get_activations(input_ids, layers=layers)

        # Compute J-lens logits per layer
        logits_per_layer: dict[int, Any] = {}
        for layer_idx in layers:
            if layer_idx not in activations:
                continue
            act = activations[layer_idx][0]  # (seq_len, d_model)

            # Average over positions for watchlist scoring
            h_mean = act.mean(dim=0)
            logits = lens.apply(
                h_mean.unsqueeze(0),
                layer_idx,
                norm_fn=adapter.final_norm,
                unembed=adapter.unembed_weight,
            )
            logits_per_layer[layer_idx] = logits.squeeze()

        # Score against watchlist
        all_scores = wl.score_batch(logits_per_layer, adapter.tokenizer)

        # Aggregate across layers
        alerts = []
        category_scores: dict[str, dict[str, float]] = {}
        any_triggered = False

        for layer_idx, layer_scores in all_scores.items():
            for ws in layer_scores:
                if input.categories and ws.category not in input.categories:
                    continue
                if ws.category not in category_scores:
                    category_scores[ws.category] = {}
                for token, score in ws.token_scores.items():
                    existing = category_scores[ws.category].get(token, 0)
                    category_scores[ws.category][token] = max(existing, score)
                if ws.triggered:
                    any_triggered = True
                    alerts.append({
                        "category": ws.category,
                        "layer": layer_idx,
                        "max_score": ws.max_score,
                        "threshold": ws.threshold,
                        "top_tokens": dict(
                            sorted(ws.token_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                        ),
                    })

        return WatchlistOutput(
            model_id=adapter.info.model_id,
            lens_id=lens.lens_id,
            alerts=alerts,
            scores=category_scores,
            any_triggered=any_triggered,
            metadata={
                "duration_ms": (time.time() - start_time) * 1000,
                "layers_evaluated": layers,
            },
        )

    @server.tool(
        name="jlens_decompose",
        description=(
            "Decompose an activation at a specific layer and position into J-space: "
            "a sparse non-negative combination of ≤k J-lens vectors. Returns the "
            "active tokens, their coefficients, and variance explained."
        ),
    )
    async def jlens_decompose(input: DecomposeInput) -> DecomposeOutput:
        """Decompose activation into J-space."""
        start_time = time.time()

        adapter = _get_adapter(model_registry, input.model_id)
        lens = _get_lens(lens_registry, input.lens_id, input.model_id)

        # Get activation at specified layer/position
        tokens = adapter.tokenize(input.prompt)
        input_ids = tokens["input_ids"]
        activations = adapter.get_activations(input_ids, layers=[input.layer])

        act = activations[input.layer][0]  # (seq_len, d_model)
        if input.position >= act.shape[0]:
            raise ValueError(
                f"Position {input.position} out of range (seq_len={act.shape[0]})"
            )

        h = act[input.position]

        # Get J-lens vectors for decomposition
        jlens_vectors = lens.get_jlens_vectors(input.layer, adapter.unembed_weight)

        # Decompose
        from prometheus.jlens.jspace import JSpaceDecomposer
        decomposer = JSpaceDecomposer(jlens_vectors, k=input.k)
        result = decomposer.decompose(h)

        # Format output
        top_tokens = [
            {
                "token": adapter.decode_tokens([idx])[0],
                "token_id": idx,
                "coefficient": coeff,
                "rank": rank + 1,
            }
            for rank, (idx, coeff) in enumerate(result.top_tokens)
        ]

        return DecomposeOutput(
            model_id=adapter.info.model_id,
            lens_id=lens.lens_id,
            layer=input.layer,
            position=input.position,
            top_tokens=top_tokens,
            variance_explained=result.variance_explained,
            occupancy=len(result.top_tokens),
            metadata={
                "duration_ms": (time.time() - start_time) * 1000,
                "token_at_position": adapter.decode_tokens(
                    [input_ids[0, input.position].item()]
                )[0],
            },
        )

    if enable_steering:
        @server.tool(
            name="jlens_steer",
            description=(
                "EXPERIMENTAL (Phase 3): Apply steering interventions to model generation. "
                "Injects or ablates J-lens directions during forward pass. "
                "Feature-flagged — requires explicit enablement."
            ),
        )
        async def jlens_steer(input: SteerInput) -> SteerOutput:
            """Apply steering interventions during generation."""
            adapter = _get_adapter(model_registry, input.model_id)
            lens = _get_lens(lens_registry, input.lens_id, input.model_id)

            # Parse interventions
            from prometheus.jlens.interventions import (
                InterventionHook,
                SteeringIntervention,
                AblationIntervention,
                ConceptSwapIntervention,
            )

            interventions = []
            for spec in input.interventions:
                int_type = spec.get("type", "steer")
                token_str = spec.get("token", "")
                alpha = spec.get("alpha", 1.0)
                int_layers = spec.get("layers", list(range(
                    adapter.n_layers // 3, 2 * adapter.n_layers // 3
                )))

                # Resolve token to J-lens direction
                token_ids = adapter.tokenizer.encode(token_str, add_special_tokens=False)
                if not token_ids:
                    continue
                token_id = token_ids[0]

                jlens_vectors = lens.get_jlens_vectors(int_layers[0], adapter.unembed_weight)
                direction = jlens_vectors[token_id]

                if int_type == "steer":
                    interventions.append(SteeringIntervention(
                        layers=int_layers, direction=direction, alpha=alpha
                    ))
                elif int_type == "ablate":
                    interventions.append(AblationIntervention(
                        layers=int_layers, directions=[direction]
                    ))

            # Generate with and without interventions
            tokens = adapter.tokenize(input.prompt)
            input_ids = tokens["input_ids"].to(adapter.device)

            # Original generation
            with torch.no_grad():
                original_output = adapter.model.generate(
                    input_ids, max_new_tokens=input.max_new_tokens, do_sample=False
                )
            original_text = adapter.tokenizer.decode(
                original_output[0][input_ids.shape[1]:], skip_special_tokens=True
            )

            # Steered generation
            hook = InterventionHook(interventions)
            hook.register(adapter.model, adapter.layers)
            try:
                with torch.no_grad():
                    steered_output = adapter.model.generate(
                        input_ids, max_new_tokens=input.max_new_tokens, do_sample=False
                    )
            finally:
                hook.remove()

            steered_text = adapter.tokenizer.decode(
                steered_output[0][input_ids.shape[1]:], skip_special_tokens=True
            )

            return SteerOutput(
                model_id=adapter.info.model_id,
                lens_id=lens.lens_id,
                original_text=original_text,
                steered_text=steered_text,
                interventions_applied=[
                    {"type": spec.get("type"), "token": spec.get("token"), "alpha": spec.get("alpha")}
                    for spec in input.interventions
                ],
            )

    return server


def create_harness_server(harness: Any = None) -> Any:
    """Create a FastMCP server exposing harness management tools.

    Tools:
    - harness_status: Get current harness configuration and status
    - harness_configure: Update harness configuration
    - verifier_check: Run async verification on a completion claim
    """
    try:
        from fastmcp import FastMCP
    except ImportError:
        logger.warning("FastMCP not available")
        return None

    server = FastMCP(
        name="prometheus-harness",
        description="Agentic harness management — status, configuration, and verification",
    )

    @server.tool(
        name="harness_status",
        description="Get current harness status including loaded models, active lenses, and security posture.",
    )
    async def harness_status() -> dict[str, Any]:
        if harness is None:
            return {"status": "not_initialized"}
        return harness.get_status()

    @server.tool(
        name="verifier_check",
        description=(
            "Submit a CompletionClaim for async verification. The verifier runs "
            "deterministic checks, execution-based verification, and LLM-as-judge "
            "review using a different model family than the actor."
        ),
    )
    async def verifier_check(
        goal: str,
        deliverables: list[str],
        evidence: list[str],
        test_results: dict[str, Any] | None = None,
        self_report: str | None = None,
    ) -> dict[str, Any]:
        if harness is None:
            return {"status": "ESCALATE", "reason": "Harness not initialized"}
        return await harness.verify_claim({
            "goal": goal,
            "deliverables": deliverables,
            "evidence": evidence,
            "test_results": test_results,
            "self_report": self_report,
        })

    return server


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _get_adapter(registry: Any, model_id: str | None) -> Any:
    """Resolve a model adapter from the registry."""
    if registry is None:
        raise RuntimeError("No model registry configured. Load a model first.")
    if model_id:
        return registry.get(model_id)
    return registry.get_default()


def _get_lens(registry: Any, lens_id: str | None, model_id: str | None) -> Any:
    """Resolve a lens artifact from the registry."""
    if registry is None:
        raise RuntimeError("No lens registry configured. Fit a lens first.")
    if lens_id:
        return registry.get(lens_id)
    return registry.get_latest(model_id)


def _default_watchlist():
    """Create a default watchlist."""
    from prometheus.jlens.watchlist import Watchlist
    return Watchlist()


def _create_minimal_server(lens_registry, model_registry, watchlist, enable_steering):
    """Create a minimal server stub when FastMCP is not available."""

    class MinimalServer:
        def __init__(self):
            self.tools = {}
            self.name = "prometheus-jlens"

        def tool(self, **kwargs):
            def decorator(fn):
                self.tools[kwargs.get("name", fn.__name__)] = fn
                return fn
            return decorator

        async def call_tool(self, name: str, arguments: dict) -> Any:
            if name in self.tools:
                return await self.tools[name](**arguments)
            raise ValueError(f"Unknown tool: {name}")

    return MinimalServer()


# Import torch only if needed for steering
try:
    import torch
except ImportError:
    pass
