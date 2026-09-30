"""Warm Subagent Pool — P0 War Room Replit Integration.

Implements reusable subagents with cache-aware TTL persistence.
Subagents stay warm across kinds and tiers; core loop can return to
already-briefed subagents instead of starting over.

Key design (from Replit "Free the Models"):
- No single sidekick kept alive for the session
- Any number of subagents stay warm across kinds and tiers
- Core loop picks which to wake
- Longer cache lifetime on newer models keeps re-brief cost down
- Cache key strategy: (tier, effort, specialization, context_hash)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from prometheus.harness.model_client import (
    SubagentTier,
    EffortLevel,
    ModelProvider,
    ProviderFactory,
    get_tier_config,
)
from prometheus.harness.types import (
    AgentContext,
    AgentRole,
    HookType,
    Hook,
    HookResult,
    ToolResult,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from prometheus.harness.orchestrator import AgentLoop

logger = logging.getLogger(__name__)


@dataclass
class SubagentSpec:
    """Specification for a subagent — used as pool key."""
    tier: SubagentTier
    effort: EffortLevel
    specialization: str  # "general", "explorer", "reviewer", "tester", "designer"
    context_hash: str    # Hash of briefing context for cache matching

    def __hash__(self):
        return hash((self.tier, self.effort, self.specialization, self.context_hash))

    def __eq__(self, other):
        if not isinstance(other, SubagentSpec):
            return False
        return (
            self.tier == other.tier
            and self.effort == other.effort
            and self.specialization == other.specialization
            and self.context_hash == other.context_hash
        )


@dataclass
class PooledSubagent:
    """A warm subagent in the pool."""
    spec: "SubagentSpec"
    agent: "AgentLoop"
    provider: Any  # ModelProvider
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    use_count: int = 0
    briefing: str = ""  # The briefing context that warmed this subagent

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used_at


class SubagentPool:
    """Cache-aware pool of warm subagents with TTL eviction.

    Features:
    - LRU eviction with TTL
    - Cache key = (tier, effort, specialization, context_hash)
    - Warm return: reuse subagent if briefing context matches (hash)
    - TTL configurable per tier (larger tiers = longer TTL)
    - Max pool size configurable
    - Statistics tracking for war room review
    """

    # Default TTL by tier (seconds) — larger tiers stay warm longer
    DEFAULT_TTL_BY_TIER = {
        "small": 300,       # 5 min
        "standard": 600,    # 10 min
        "large": 1800,      # 30 min
    }

    def __init__(
        self,
        max_size: int = 20,
        ttl_by_tier: dict[str, int] | None = None,
        provider_factory: Any = None,
        tool_router: Any = None,
        authz_engine: Any = None,
        jlens_gate: Any = None,
        dual_llm_gate: Any = None,
        tracer: Any = None,
        logger: Any = None,
    ):
        self.max_size = max_size
        self.ttl_by_tier = ttl_by_tier or self.DEFAULT_TTL_BY_TIER
        self.provider_factory = provider_factory or ProviderFactory()
        self.tool_router = tool_router
        self.authz_engine = authz_engine
        self.jlens_gate = jlens_gate
        self.dual_llm_gate = dual_llm_gate
        self.tracer = tracer
        self.logger = logger

        # Pool: OrderedDict for LRU — key = SubagentSpec, value = PooledSubagent
        self._pool: "OrderedDict[SubagentSpec, PooledSubagent]" = OrderedDict()
        self._lock = asyncio.Lock()

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "creations": 0,
            "expired": 0,
        }

    def _make_context_hash(self, briefing: str, max_chars: int = 2000) -> str:
        """Create a hash of the briefing context for cache matching."""
        # Truncate to avoid huge hashes; use first + last for context boundaries
        if len(briefing) <= max_chars:
            content = briefing
        else:
            content = briefing[:max_chars//2] + "..." + briefing[-max_chars//2:]
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def get_or_create(
        self,
        tier: SubagentTier,
        effort: EffortLevel,
        specialization: str,
        briefing: str,
        tools: list[str] | None = None,
        permission_mode: Any = None,
        parent_id: str | None = None,
        max_tool_calls: int = 50,
        max_runtime_seconds: float = 300.0,
    ) -> tuple["AgentLoop", bool]:
        """Get a warm subagent or create a new one.

        Returns:
            (AgentLoop, is_warm_return) — agent and whether it was a cache hit
        """
        config = get_tier_config(tier, effort)
        context_hash = self._make_context_hash(briefing)
        spec = SubagentSpec(
            tier=tier,
            effort=effort,
            specialization=specialization,
            context_hash=context_hash,
        )

        async with self._lock:
            # Check for warm return (exact context match)
            if spec in self._pool:
                pooled = self._pool[spec]
                # Move to end (LRU)
                self._pool.move_to_end(spec)
                pooled.last_used_at = time.time()
                pooled.use_count += 1
                self.stats["hits"] += 1
                logger.info(f"Subagent pool HIT: {spec.specialization} ({tier}/{effort}) use_count={pooled.use_count}")
                return pooled.agent, True

            # Cache miss — need to create
            self.stats["misses"] += 1

            # Evict if at capacity
            await self._evict_if_needed()

            # Create new subagent
            pooled = await self._create_subagent(
                spec=spec,
                briefing=briefing,
                tools=tools,
                permission_mode=permission_mode,
                parent_id=parent_id,
                max_tool_calls=max_tool_calls,
                max_runtime_seconds=max_runtime_seconds,
            )

            self._pool[spec] = pooled
            self.stats["creations"] += 1
            logger.info(f"Subagent pool CREATE: {spec.specialization} ({tier}/{effort})")
            return pooled.agent, False

    async def _create_subagent(
        self,
        spec: SubagentSpec,
        briefing: str,
        tools: list[str] | None = None,
        permission_mode: Any = None,
        parent_id: str | None = None,
        max_tool_calls: int = 50,
        max_runtime_seconds: float = 300.0,
    ) -> "PooledSubagent":
        """Create a new warmed subagent with the given briefing."""
        # Build provider for this tier/effort
        provider = self.provider_factory.build_from_tier_effort(spec.tier, spec.effort)

        # Build agent context
        from prometheus.harness.types import AgentContext, AgentRole, HookType, Hook, HookResult

        context = AgentContext(
            role=AgentRole.ACTOR,
            parent_id=parent_id,
            tools_available=tools or [],
            permission_mode=permission_mode,
            max_tool_calls=max_tool_calls,
            max_runtime_seconds=max_runtime_seconds,
        )

        # Combine hooks
        hooks = {ht: [] for ht in HookType}

        # Add dual-LLM taint gate if available
        if self.dual_llm_gate:
            hooks[HookType.PRE_TOOL_USE].append(self.dual_llm_gate.check)

        # Add J-lens gate if available
        if self.jlens_gate:
            hooks[HookType.PRE_TOOL_USE].append(self.jlens_gate.check)

        # Add authz check
        if self.authz_engine:
            hooks[HookType.PRE_TOOL_USE].append(self.authz_engine.check_permission)

        # Import AgentLoop here to avoid circular import
        from prometheus.harness.orchestrator import AgentLoop

        # Create agent loop
        agent = AgentLoop(
            context=context,
            llm_client=provider,
            tool_router=self.tool_router,
            hooks=hooks,
            tracer=self.tracer,
            logger=self.logger,
        )

        # Warm the subagent with the briefing
        # This pre-loads the context so the first actual task runs faster
        if briefing:
            # Prepend briefing as system context
            context.messages.insert(0, {
                "role": "system",
                "content": f"[SUBAGENT BRIEFING — {spec.specialization.upper()}]\n{briefing}\n\nYou are a {spec.specialization} subagent. Tier: {spec.tier.value}, Effort: {spec.effort.value}. Await task."
            })

        return PooledSubagent(
            spec=spec,
            agent=agent,
            provider=provider,
            briefing=briefing,
        )

    async def _evict_if_needed(self) -> None:
        """Evict expired or LRU entries if pool at capacity."""
        now = time.time()

        # First pass: evict expired
        expired_keys = []
        for spec, pooled in self._pool.items():
            ttl = self.ttl_by_tier.get(pooled.spec.tier.value, 600)
            if pooled.idle_seconds > ttl:
                expired_keys.append(spec)

        for key in expired_keys:
            del self._pool[key]
            self.stats["expired"] += 1
            self.stats["evictions"] += 1
            logger.debug(f"Subagent pool EVICT (expired): {key}")

        # Second pass: LRU eviction if still over capacity
        while len(self._pool) >= self.max_size:
            oldest_spec, _ = self._pool.popitem(last=False)  # FIFO = LRU
            self.stats["evictions"] += 1
            logger.debug(f"Subagent pool EVICT (LRU): {oldest_spec}")

    async def warm_subagent(
        self,
        tier: SubagentTier,
        effort: EffortLevel,
        specialization: str,
        briefing: str,
        tools: list[str] | None = None,
    ) -> "AgentLoop":
        """Explicitly warm a subagent without executing a task.

        Useful for pre-warming known specialists before they're needed.
        """
        agent, _ = await self.get_or_create(
            tier=tier,
            effort=effort,
            specialization=specialization,
            briefing=briefing,
            tools=tools,
        )
        return agent

    async def shutdown(self) -> None:
        """Shutdown all pooled subagents."""
        async with self._lock:
            for pooled in self._pool.values():
                pooled.agent.stop()
            self._pool.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics for war room review."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            **self.stats,
            "current_size": len(self._pool),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "pool_contents": [
                {
                    "tier": p.spec.tier.value,
                    "effort": p.spec.effort.value,
                    "specialization": p.spec.specialization,
                    "use_count": p.use_count,
                    "age_seconds": p.age_seconds,
                    "idle_seconds": p.idle_seconds,
                }
                for p in self._pool.values()
            ],
        }


# Global pool instance (initialized by orchestrator)
_global_pool: Optional["SubagentPool"] = None


def init_global_pool(**kwargs) -> "SubagentPool":
    """Initialize the global subagent pool."""
    global _global_pool
    _global_pool = SubagentPool(**kwargs)
    return _global_pool


def get_global_pool() -> Optional["SubagentPool"]:
    """Get the global subagent pool."""
    return _global_pool


def shutdown_global_pool() -> None:
    """Shutdown the global pool (async)."""
    global _global_pool
    if _global_pool:
        # Note: caller must await this in async context
        _global_pool = None