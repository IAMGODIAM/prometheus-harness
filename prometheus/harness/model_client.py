"""Model client — model-agnostic LLM provider abstraction (RIG addition).

The code review found that the harness shipped no concrete LLM client, so the
agent loop could not run. This module provides one.

Providers:
  * OpenAICompatibleProvider — talks to any OpenAI-compatible /chat/completions
    endpoint. Default target is NVIDIA's hosted API (integrate.api.nvidia.com);
    swap to a local NIM by changing a single env var (PROMETHEUS_LLM_BASE_URL).
  * MockProvider — deterministic, offline, no network/keys. Drives a
    plan -> tool-call -> completion-claim sequence so the loop can be smoke-tested.

API keys are NEVER hardcoded — they are read from the environment at call time.

Response contract (matches AgentLoop._call_llm / _parse_tool_calls /
_parse_completion in orchestrator.py):

    {
      "content": str,
      "tool_calls": [ {"id": str, "name": str, "arguments": dict}, ... ],
      "finish_reason": "tool_calls" | "stop" | ...,
      "usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int},
    }

Subagent Tier/Effort Abstraction (P0 - War Room Replit Integration):
  * SubagentTier: SMALL, STANDARD, LARGE — cost/capability steps
  * EffortLevel: LOW, MEDIUM, HIGH, XHIGH — reasoning effort within tier
  * TierRegistry: maps (tier, effort) -> provider config (model, params, cost)
  * ProviderFactory: builds provider from tier/effort at dispatch time

API keys are NEVER hardcoded — they are read from the environment at call time.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default endpoints. NVIDIA hosted is the launch target; local NIM is the swap.
NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
LOCAL_NIM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_ID = "meta/llama-3.1-8b-instruct"  # override with PROMETHEUS_LLM_MODEL


class SubagentTier(str, Enum):
    """Subagent compute tier — cost/capability step."""
    SMALL = "small"          # Fast, cheap: mechanical renames, simple reads
    STANDARD = "standard"    # Balanced: typical implementation tasks
    LARGE = "large"          # High capability: hypothesis generation, complex refactors


class EffortLevel(str, Enum):
    """Reasoning effort within a tier."""
    LOW = "low"              # Minimal reasoning, fast
    MEDIUM = "medium"        # Balanced reasoning
    HIGH = "high"            # Deep reasoning
    XHIGH = "xhigh"          # Maximum reasoning (cache-preserving on GPT-6 family)


@dataclass(frozen=True)
class TierConfig:
    """Configuration for a (tier, effort) combination."""
    tier: SubagentTier
    effort: EffortLevel
    model_id: str
    temperature: float
    max_tokens: int
    timeout: float
    # Cost tracking (USD per 1K tokens, approximate)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    # Capability metadata
    description: str = ""
    suitable_for: list[str] = field(default_factory=list)


# Tier Registry — maps (tier, effort) to provider configs.
# Can be overridden via PROMETHEUS_TIER_REGISTRY_JSON env var (JSON string).
TIER_REGISTRY: dict[tuple[SubagentTier, EffortLevel], TierConfig] = {
    # SMALL tier
    (SubagentTier.SMALL, EffortLevel.LOW): TierConfig(
        tier=SubagentTier.SMALL, effort=EffortLevel.LOW,
        model_id="meta/llama-3.1-8b-instruct",
        temperature=0.1, max_tokens=512, timeout=30.0,
        cost_per_1k_input=0.0001, cost_per_1k_output=0.0002,
        description="Fast mechanical tasks",
        suitable_for=["rename", "simple_read", "format", "grep"],
    ),
    (SubagentTier.SMALL, EffortLevel.MEDIUM): TierConfig(
        tier=SubagentTier.SMALL, effort=EffortLevel.MEDIUM,
        model_id="meta/llama-3.1-8b-instruct",
        temperature=0.2, max_tokens=1024, timeout=45.0,
        cost_per_1k_input=0.0001, cost_per_1k_output=0.0002,
        description="Standard mechanical tasks",
        suitable_for=["mechanical_rename", "simple_extraction", "boilerplate"],
    ),

    # STANDARD tier
    (SubagentTier.STANDARD, EffortLevel.LOW): TierConfig(
        tier=SubagentTier.STANDARD, effort=EffortLevel.LOW,
        model_id="meta/llama-3.1-70b-instruct",
        temperature=0.2, max_tokens=1024, timeout=60.0,
        cost_per_1k_input=0.0005, cost_per_1k_output=0.001,
        description="Balanced implementation",
        suitable_for=["feature_impl", "bug_fix", "test_generation"],
    ),
    (SubagentTier.STANDARD, EffortLevel.MEDIUM): TierConfig(
        tier=SubagentTier.STANDARD, effort=EffortLevel.MEDIUM,
        model_id="meta/llama-3.1-70b-instruct",
        temperature=0.3, max_tokens=2048, timeout=90.0,
        cost_per_1k_input=0.0005, cost_per_1k_output=0.001,
        description="Standard implementation with reasoning",
        suitable_for=["typical_task", "refactor", "integration"],
    ),
    (SubagentTier.STANDARD, EffortLevel.HIGH): TierConfig(
        tier=SubagentTier.STANDARD, effort=EffortLevel.HIGH,
        model_id="meta/llama-3.1-70b-instruct",
        temperature=0.4, max_tokens=4096, timeout=120.0,
        cost_per_1k_input=0.0005, cost_per_1k_output=0.001,
        description="Deep reasoning for standard tasks",
        suitable_for=["stubborn_bug", "design_decision", "architecture"],
    ),

    # LARGE tier
    (SubagentTier.LARGE, EffortLevel.MEDIUM): TierConfig(
        tier=SubagentTier.LARGE, effort=EffortLevel.MEDIUM,
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        temperature=0.3, max_tokens=4096, timeout=120.0,
        cost_per_1k_input=0.002, cost_per_1k_output=0.005,
        description="High-capability hypothesis generation",
        suitable_for=["complex_hypothesis", "multi_file_refactor", "root_cause"],
    ),
    (SubagentTier.LARGE, EffortLevel.HIGH): TierConfig(
        tier=SubagentTier.LARGE, effort=EffortLevel.HIGH,
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        temperature=0.4, max_tokens=8192, timeout=180.0,
        cost_per_1k_input=0.002, cost_per_1k_output=0.005,
        description="Deep analysis for hard problems",
        suitable_for=["stubborn_bug_hypothesis", "architectural_decision", "novel_algorithm"],
    ),
    (SubagentTier.LARGE, EffortLevel.XHIGH): TierConfig(
        tier=SubagentTier.LARGE, effort=EffortLevel.XHIGH,
        model_id="nvidia/nemotron-3-ultra-550b-a55b",
        temperature=0.5, max_tokens=16384, timeout=300.0,
        cost_per_1k_input=0.002, cost_per_1k_output=0.005,
        description="Maximum reasoning (cache-preserving on GPT-6 family)",
        suitable_for=["novel_research", "theoretical_proof", "system_design"],
    ),
}


def _load_tier_registry_from_env() -> dict[tuple[SubagentTier, EffortLevel], TierConfig]:
    """Load tier registry from PROMETHEUS_TIER_REGISTRY_JSON env var if present."""
    import json
    registry_json = os.environ.get("PROMETHEUS_TIER_REGISTRY_JSON")
    if not registry_json:
        return TIER_REGISTRY
    try:
        data = json.loads(registry_json)
        registry = {}
        for key_str, config_dict in data.items():
            tier_str, effort_str = key_str.split("|")
            tier = SubagentTier(tier_str)
            effort = EffortLevel(effort_str)
            registry[(tier, effort)] = TierConfig(**config_dict)
        return registry
    except Exception as e:
        logger.warning(f"Failed to parse PROMETHEUS_TIER_REGISTRY_JSON: {e}; using defaults")
        return TIER_REGISTRY


def get_tier_config(tier: SubagentTier, effort: EffortLevel) -> TierConfig:
    """Get TierConfig for (tier, effort), with fallback to closest available."""
    registry = _load_tier_registry_from_env()
    key = (tier, effort)
    if key in registry:
        return registry[key]
    # Fallback: same tier, closest effort
    for e in EffortLevel:
        if (tier, e) in registry:
            logger.warning(f"Tier config ({tier}, {effort}) not found; falling back to ({tier}, {e})")
            return registry[(tier, e)]
    # Fallback: any config for this tier
    for (t, e), cfg in registry.items():
        if t == tier:
            logger.warning(f"Tier config ({tier}, {effort}) not found; falling back to ({t}, {e})")
            return cfg
    # Ultimate fallback
    return TierConfig(
        tier=tier, effort=effort, model_id=DEFAULT_MODEL_ID,
        temperature=0.2, max_tokens=1024, timeout=60.0,
    )


class ModelProvider(ABC):
    """Abstract, model-agnostic provider. Mirrors a ModelProvider-style pattern:
    one uniform ``complete()`` regardless of backend, with a ``model_id`` attribute
    the verifier reads to record which model produced/checked a result."""

    model_id: str = "unknown"

    # Full OpenAI-style tool schemas, keyed by tool name. The harness context only
    # carries tool *names*; the runner injects real schemas here so hosted models
    # can emit well-formed tool calls.
    tool_specs: dict[str, dict[str, Any]]

    def set_tool_specs(self, specs: dict[str, dict[str, Any]]) -> None:
        self.tool_specs = dict(specs or {})

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Return a response dict following the contract in the module docstring."""
        raise NotImplementedError


def _normalize_tool_calls(raw_tool_calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool_calls into the harness shape {id,name,arguments}."""
    out: list[dict[str, Any]] = []
    for i, tc in enumerate(raw_tool_calls or []):
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = fn.get("name", "")
        args_raw = fn.get("arguments", {})
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
        else:
            args = args_raw or {}
        out.append({"id": tc.get("id", f"call_{i}"), "name": name, "arguments": args})
    return out


class OpenAICompatibleProvider(ModelProvider):
    """OpenAI-compatible chat-completions client (NVIDIA hosted or local NIM).

    ``httpx`` (already a core dependency) is imported lazily so that importing
    this module and using MockProvider require no third-party packages.
    """

    def __init__(
        self,
        base_url: str = NVIDIA_DEFAULT_BASE_URL,
        model_id: str = DEFAULT_MODEL_ID,
        api_key: str | None = None,
        api_key_env: str = "NVIDIA_API_KEY",
        timeout: float = 60.0,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key_env = api_key_env
        # Key is read from the environment (never hardcoded). An explicitly passed
        # api_key is supported for tests but callers should prefer the env var.
        self._api_key = api_key or os.environ.get(api_key_env)
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_specs = {}

    def _openai_tools(self, tools: list[Any] | None) -> list[dict[str, Any]] | None:
        """Build OpenAI ``tools`` from the context's tool names + injected specs."""
        if not tools:
            return None
        specs: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):  # already a full spec
                specs.append(t)
                continue
            name = str(t)
            spec = self.tool_specs.get(name)
            if spec:
                specs.append(spec)
            else:  # name-only fallback (harness limitation: no schema tracked)
                specs.append({
                    "type": "function",
                    "function": {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}},
                })
        return specs

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        if not self._api_key:
            raise RuntimeError(
                f"No API key found. Set ${self.api_key_env} in the environment "
                f"(e.g. export {self.api_key_env}=nvapi-...). Keys are never hardcoded."
            )

        import httpx  # lazy import — keeps MockProvider dependency-free

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        openai_tools = self._openai_tools(tools)
        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        usage = data.get("usage", {}) or {}
        return {
            "content": msg.get("content") or "",
            "tool_calls": _normalize_tool_calls(msg.get("tool_calls")),
            "finish_reason": choice.get("finish_reason", "stop"),
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }


class MockProvider(ModelProvider):
    """Deterministic offline provider for smoke tests. No network, no keys.

    Turn 1: emit a plan + one tool call (exercises tool dispatch + gates + authz).
    Turn 2: emit a structured CompletionClaim (exercises the verifier).
    """

    model_id = "mock/deterministic-v1"

    def __init__(self, tool_name: str | None = None):
        self._turn = 0
        self._forced_tool = tool_name
        self.tool_specs = {}

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        self._turn += 1
        if self._turn == 1:
            tool = self._forced_tool or (str(tools[0]) if tools else "echo")
            return {
                "content": "Plan: (1) call a safe read-only tool, (2) verify, (3) finish.",
                "tool_calls": [
                    {"id": "call_1", "name": tool, "arguments": {"text": "hello from Prometheus"}}
                ],
                "finish_reason": "tool_calls",
                "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            }
        claim = {
            "goal": "Demonstrate an end-to-end agent run (plan -> act -> verify -> gate).",
            "deliverables": ["Called a safe tool via the gated tool router"],
            "evidence": ["Tool executed under authz + dual-LLM gate", "Budget not exhausted"],
            "acceptance_criteria": ["A tool call was authorized and executed"],
            "self_report": "Completed the demonstration task within budget and safety policy.",
        }
        return {
            "content": "CompletionClaim " + json.dumps(claim),
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {"input_tokens": 30, "output_tokens": 60, "total_tokens": 90},
        }


class ProviderFactory:
    """Factory for building providers from tier/effort or env."""

    def __init__(self):
        self._provider_cache: dict[tuple[SubagentTier, EffortLevel], ModelProvider] = {}

    def build_from_tier_effort(
        self,
        tier: SubagentTier,
        effort: EffortLevel,
        api_key_env: str = "NVIDIA_API_KEY",
    ) -> ModelProvider:
        """Build provider from tier/effort config."""
        cache_key = (tier, effort)
        if cache_key in self._provider_cache:
            return self._provider_cache[cache_key]

        config = get_tier_config(tier, effort)
        provider = OpenAICompatibleProvider(
            base_url=os.environ.get("PROMETHEUS_LLM_BASE_URL", NVIDIA_DEFAULT_BASE_URL),
            model_id=config.model_id,
            api_key_env=api_key_env,
            timeout=config.timeout,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self._provider_cache[cache_key] = provider
        return provider

    def build_from_env(self) -> ModelProvider:
        """Construct a provider from environment variables (legacy)."""
        provider = os.environ.get("PROMETHEUS_LLM_PROVIDER", "openai").lower()
        if provider == "mock":
            return MockProvider()
        base_url = os.environ.get("PROMETHEUS_LLM_BASE_URL", NVIDIA_DEFAULT_BASE_URL)
        model_id = os.environ.get("PROMETHEUS_LLM_MODEL", DEFAULT_MODEL_ID)
        api_key_env = os.environ.get("PROMETHEUS_API_KEY_ENV", "NVIDIA_API_KEY")
        return OpenAICompatibleProvider(
            base_url=base_url, model_id=model_id, api_key_env=api_key_env
        )


def build_provider_from_env() -> ModelProvider:
    """Legacy entry point — delegates to ProviderFactory."""
    return ProviderFactory().build_from_env()


# Export key symbols
__all__ = [
    "ModelProvider",
    "OpenAICompatibleProvider",
    "MockProvider",
    "ProviderFactory",
    "SubagentTier",
    "EffortLevel",
    "TierConfig",
    "get_tier_config",
    "build_provider_from_env",
    "NVIDIA_DEFAULT_BASE_URL",
    "LOCAL_NIM_BASE_URL",
    "DEFAULT_MODEL_ID",
]
