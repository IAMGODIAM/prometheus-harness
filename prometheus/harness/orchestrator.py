"""Orchestrator — Agent loop with hook system and subagent context isolation.

Implements the 2026 consensus agent-loop architecture drawn from Claude Agent SDK,
DeerFlow 2.0, OpenAI Agents SDK, and Hermes Agent patterns:
- Hook system: PreToolUse, PostToolUse, PreCompact, UserPromptSubmit, SessionStart/End
- Permission modes: default, acceptEdits, plan, bypassPermissions, auto
- Subagent context isolation
- Programmatic tool calling (code-mode for token reduction)
- Context folding/compaction with durable anchors

P0 WAR ROOM REPLIT INTEGRATION:
- Subagent pool with warm returns (subagent_pool.py)
- Tier/effort-based provider factory (model_client.py)
- Delegation-aware J-Lens gate (jlens_gate.py)
- Composable delegation: specialists, tiers, warm return, dynamic effort
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Optional

from prometheus.harness.model_client import (
    SubagentTier,
    EffortLevel,
    ProviderFactory,
)
from prometheus.harness.subagent_pool import (
    SubagentPool,
    SubagentSpec,
    init_global_pool,
    get_global_pool,
)
from prometheus.harness.types import (
    HookType,
    PermissionMode,
    AgentRole,
    HookResult,
    AgentContext,
    ToolResult,
    Hook,
)

logger = logging.getLogger(__name__)


class EffortEscalationHook:
    """Hook that monitors task trajectory and escalates effort level mid-turn.

    Tracks tool call patterns, failure rates, and context complexity to
    dynamically increase reasoning effort when the agent is struggling.
    """

    def __init__(
        self,
        failure_threshold: float = 0.3,
        complexity_threshold: int = 2000,
        max_escalations: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.complexity_threshold = complexity_threshold
        self.max_escalations = max_escalations
        self._escalation_count: dict[str, int] = {}

    async def __call__(self, data: dict[str, Any]) -> HookResult:
        """Evaluate whether to escalate effort based on context."""
        context = data.get("context")
        if not context:
            return HookResult(allow=True)

        agent_id = context.agent_id
        current_effort = context.subagent_effort
        current_tier = context.subagent_tier

        # Count recent failures
        tool_calls = context.tool_calls_made
        recent_results = data.get("recent_results", [])
        if recent_results:
            failures = sum(1 for r in recent_results[-5:] if not r.get("success", True))
            failure_rate = failures / len(recent_results[-5:])
        else:
            failure_rate = 0.0

        # Check complexity (token count)
        token_estimate = data.get("token_count", 0)

        # Check if we should escalate
        should_escalate = False
        new_effort = current_effort
        new_tier = current_tier

        if failure_rate >= self.failure_threshold:
            should_escalate = True
        elif token_estimate >= self.complexity_threshold:
            should_escalate = True

        if should_escalate:
            escalation_key = f"{agent_id}:{current_effort}:{current_tier}"
            current_escalations = self._escalation_count.get(escalation_key, 0)

            if current_escalations < self.max_escalations:
                # Escalate effort level
                if current_effort == EffortLevel.LOW:
                    new_effort = EffortLevel.MEDIUM
                elif current_effort == EffortLevel.MEDIUM:
                    new_effort = EffortLevel.HIGH

                # Also escalate tier if at HIGH effort
                if new_effort == EffortLevel.HIGH and current_tier == SubagentTier.STANDARD:
                    new_tier = SubagentTier.LARGE

                self._escalation_count[escalation_key] = current_escalations + 1

                return HookResult(
                    allow=True,
                    modified_args={
                        "escalate_effort": True,
                        "new_effort": new_effort,
                        "new_tier": new_tier,
                        "escalation_reason": f"failure_rate={failure_rate:.2f}, tokens={token_estimate}",
                    },
                    escalate=True,
                )

        return HookResult(allow=True)


class AgentLoop:
    """Core agent loop implementing the think-act-observe cycle.

    The loop:
    1. Receive user input or continuation signal
    2. Run PreCompact hook (context management)
    3. Call LLM with current context
    4. Parse tool calls from response
    5. For each tool call:
       a. Run PreToolUse hooks (permission check, J-lens gate, taint check)
       b. Execute tool via MCP
       c. Run PostToolUse hooks (result validation, observability)
    6. If completion claimed, route to verifier
    7. Loop until done or budget exhausted
    """

    def __init__(
        self,
        context: AgentContext,
        llm_client: Any = None,
        tool_router: Any = None,
        hooks: dict[HookType, list[Hook]] | None = None,
        tracer: Any = None,
        logger: Any = None,
        provider_factory: Any = None,
    ):
        self.context = context
        self.llm_client = llm_client
        self.tool_router = tool_router
        self.hooks: dict[HookType, list[Hook]] = hooks or {ht: [] for ht in HookType}
        # RIG: optional observability, no-op when None (preserves existing tests).
        self.tracer = tracer
        self.logger = logger
        self._running = False
        self._paused = False
        self._provider_factory = provider_factory

        # Effort escalation state
        self._effort_escalation_hook: EffortEscalationHook | None = None
        for hook_list in self.hooks.values():
            for hook in hook_list:
                if isinstance(hook, EffortEscalationHook):
                    self._effort_escalation_hook = hook
                    break

    def register_hook(self, hook_type: HookType, hook: Hook) -> None:
        """Register a hook for a specific lifecycle event."""
        if hook_type not in self.hooks:
            self.hooks[hook_type] = []
        self.hooks[hook_type].append(hook)

    async def _run_hooks(self, hook_type: HookType, data: dict[str, Any]) -> HookResult:
        """Run all hooks for a given type. Deny overrides allow (deny-first invariant)."""
        combined = HookResult(allow=True)

        for hook in self.hooks.get(hook_type, []):
            try:
                result = await hook(data)
                # Deny-first: any deny overrides all allows
                if not result.allow:
                    combined.allow = False
                    combined.reason = result.reason
                    combined.escalate = result.escalate
                    break  # Short-circuit on deny
                if result.modified_args:
                    combined.modified_args = result.modified_args
            except Exception as e:
                logger.error(f"Hook error ({hook_type}): {e}")
                # Hook errors default to deny for safety
                combined.allow = False
                combined.reason = f"Hook error: {e}"

        return combined

    async def run(self, initial_message: str | None = None) -> dict[str, Any]:
        """Execute the agent loop until completion or budget exhaustion.

        Args:
            initial_message: Optional initial user message to start the loop.

        Returns:
            Final result dict with status, output, and metadata.
        """
        self._running = True
        if self.tracer:
            self.tracer.new_trace()

        # Session start hook
        await self._run_hooks(HookType.SESSION_START, {
            "agent_id": self.context.agent_id,
            "role": self.context.role,
        })

        if initial_message:
            self.context.messages.append({
                "role": "user",
                "content": initial_message,
            })

        result = {"status": "incomplete", "output": None, "tool_calls": []}

        while self._running and not self.context.budget_exhausted:
            if self._paused:
                await asyncio.sleep(0.1)
                continue

            try:
                # Pre-compact hook (context management)
                compact_result = await self._run_hooks(HookType.PRE_COMPACT, {
                    "messages": self.context.messages,
                    "token_count": self._estimate_tokens(),
                })

                # Effort escalation hook (pre-LLM call)
                if self._effort_escalation_hook:
                    recent_results = [tc for tc in result.get("tool_calls", [])]
                    escalation_data = {
                        "context": self.context,
                        "recent_results": recent_results,
                        "token_count": self._estimate_tokens(),
                    }
                    escalation_result = await self._effort_escalation_hook(escalation_data)
                    if escalation_result.escalate and escalation_result.modified_args:
                        # Apply effort escalation
                        new_effort = escalation_result.modified_args.get("new_effort")
                        new_tier = escalation_result.modified_args.get("new_tier")
                        if new_effort:
                            self.context.subagent_effort = new_effort
                            # Rebuild LLM client with new effort
                            if self._provider_factory:
                                self.llm_client = self._provider_factory.build_from_tier_effort(
                                    self.context.subagent_tier, new_effort
                                )
                        if new_tier:
                            self.context.subagent_tier = new_tier
                        if self.logger:
                            self.logger.info(
                                f"Effort escalated: {escalation_result.modified_args.get('escalation_reason')}"
                            )

                # Call LLM
                response = await self._call_llm()
                if response is None:
                    break

                # RIG: record the LLM generation span when tracing is wired
                if self.tracer and isinstance(response, dict):
                    _usage = response.get("usage", {}) or {}
                    self.tracer.record_llm_call(
                        model=getattr(self.llm_client, "model_id", "unknown"),
                        input_tokens=_usage.get("input_tokens", 0),
                        output_tokens=_usage.get("output_tokens", 0),
                        duration_ms=0.0,
                    )

                # Parse response for tool calls or completion
                tool_calls = self._parse_tool_calls(response)
                completion_claim = self._parse_completion(response)

                if completion_claim:
                    # Route to verifier
                    result["status"] = "completion_claimed"
                    result["output"] = completion_claim
                    break

                if not tool_calls:
                    # No tool calls, no completion — add assistant message and continue
                    self.context.messages.append({
                        "role": "assistant",
                        "content": response.get("content", ""),
                    })
                    # Check if this is a natural end
                    if self._is_natural_end(response):
                        result["status"] = "complete"
                        result["output"] = response.get("content", "")
                        break
                    continue

                # Execute tool calls
                for tool_call in tool_calls:
                    # PreToolUse hook — include delegation context for J-Lens gate
                    pre_data = {
                        "tool_name": tool_call.get("name"),
                        "arguments": tool_call.get("arguments", {}),
                        "context": self.context,
                    }
                    # Add delegation context for J-Lens gate (P0)
                    if self.context.subagent_tier:
                        pre_data["delegation_context"] = True
                        pre_data["subagent_tier"] = self.context.subagent_tier.value
                        if self.context.subagent_effort:
                            pre_data["subagent_effort"] = self.context.subagent_effort.value

                    pre_result = await self._run_hooks(HookType.PRE_TOOL_USE, pre_data)

                    if not pre_result.allow:
                        # Tool call denied by a PreToolUse gate (dual-LLM / J-lens / authz)
                        if self.logger:
                            self.logger.authz_decision(
                                tool_call.get("name", ""), "deny", pre_result.reason or ""
                            )
                        tool_result = ToolResult(
                            tool_name=tool_call.get("name", ""),
                            success=False,
                            error=f"Denied: {pre_result.reason}",
                        )
                        if pre_result.escalate:
                            result["status"] = "escalated"
                            result["output"] = pre_result.reason
                            self._running = False
                            break
                    else:
                        # Execute tool (traced + logged when observability is wired)
                        if self.tracer:
                            from prometheus.observability.tracing import SpanKind
                            with self.tracer.span(
                                f"tool.{tool_call.get('name', '')}", SpanKind.TOOL_CALL
                            ):
                                tool_result = await self._execute_tool(tool_call)
                        else:
                            tool_result = await self._execute_tool(tool_call)
                        if self.logger:
                            self.logger.tool_call(
                                tool_call.get("name", ""),
                                tool_result.duration_ms,
                                tool_result.success,
                            )

                    # PostToolUse hook
                    await self._run_hooks(HookType.POST_TOOL_USE, {
                        "tool_name": tool_call.get("name"),
                        "result": tool_result,
                        "context": self.context,
                    })

                    # Add to context
                    self.context.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "name": tool_call.get("name"),
                        "content": str(tool_result.output if tool_result.success else tool_result.error),
                    })

                    self.context.tool_calls_made += 1
                    result["tool_calls"].append({
                        "name": tool_call.get("name"),
                        "success": tool_result.success,
                    })

            except Exception as e:
                logger.error(f"Agent loop error: {e}")
                result["status"] = "error"
                result["output"] = str(e)
                break

        if self.context.budget_exhausted:
            result["status"] = "budget_exhausted"

        # Session end hook
        await self._run_hooks(HookType.SESSION_END, {
            "agent_id": self.context.agent_id,
            "result": result,
        })

        self._running = False
        return result

    async def _call_llm(self) -> dict[str, Any] | None:
        """Call the LLM with current context."""
        if self.llm_client is None:
            logger.error("No LLM client configured")
            return None

        try:
            response = await self.llm_client.complete(
                messages=self.context.messages,
                tools=self.context.tools_available,
            )
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    async def _execute_tool(self, tool_call: dict[str, Any]) -> ToolResult:
        """Execute a tool call via the tool router (MCP)."""
        start = time.time()
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        if self.tool_router is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error="No tool router configured",
            )

        try:
            output = await self.tool_router.call(tool_name, arguments)
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=output,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _parse_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse tool calls from LLM response (model-agnostic)."""
        return response.get("tool_calls", [])

    def _parse_completion(self, response: dict[str, Any]) -> dict[str, Any] | None:
        """Check if response contains a structured CompletionClaim."""
        content = response.get("content", "")
        if isinstance(content, str) and "CompletionClaim" in content:
            # Parse structured claim
            try:
                import json
                # Look for JSON block
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    claim = json.loads(content[start:end])
                    if "goal" in claim and "deliverables" in claim:
                        return claim
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _is_natural_end(self, response: dict[str, Any]) -> bool:
        """Check if the response indicates natural completion."""
        return response.get("finish_reason") == "stop" and not response.get("tool_calls")

    def _estimate_tokens(self) -> int:
        """Rough token estimate for context management."""
        total = 0
        for msg in self.context.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4  # Rough estimate
        return total

    def pause(self) -> None:
        """Pause the agent loop."""
        self._paused = True

    def resume(self) -> None:
        """Resume the agent loop."""
        self._paused = False

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False


class Orchestrator:
    """Top-level orchestrator managing multiple agent loops and subagents.

    Implements the orchestrator-workers pattern with:
    - Subagent context isolation (>90% improvement on complex tasks)
    - Programmatic tool calling for token reduction
    - Durable context anchors for rewind
    - Kill switch and per-session pause

    P0 WAR ROOM REPLIT INTEGRATION:
    - Subagent pool with warm returns (subagent_pool.py)
    - Tier/effort-based provider factory (model_client.py)
    - Delegation-aware J-Lens gate (jlens_gate.py)
    - Composable delegation: specialists, tiers, warm return, dynamic effort
    """

    def __init__(
        self,
        llm_client: Any = None,
        tool_router: Any = None,
        verifier: Any = None,
        authz_engine: Any = None,
        jlens_gate: Any = None,
        dual_llm_gate: Any = None,
        tracer: Any = None,
        logger: Any = None,
        # P0: Subagent pool configuration
        subagent_pool_max_size: int = 20,
        subagent_pool_ttl_by_tier: dict[str, int] | None = None,
    ):
        self.llm_client = llm_client
        self.tool_router = tool_router
        self.verifier = verifier
        self.authz_engine = authz_engine
        self.jlens_gate = jlens_gate
        self.dual_llm_gate = dual_llm_gate
        self.tracer = tracer
        self.obs_logger = logger
        self.agents: dict[str, AgentLoop] = {}
        self._global_hooks: dict[HookType, list[Hook]] = {ht: [] for ht in HookType}
        self._kill_switch = False

        # P0: Provider factory for tier/effort-based delegation
        # If the top-level client is MockProvider, force mock for all pool builds
        # so --mock harness runs never require NVIDIA_API_KEY.
        from prometheus.harness.model_client import MockProvider as _MockProvider
        force_mock = isinstance(llm_client, _MockProvider)
        self.provider_factory = ProviderFactory(force_mock=force_mock)

        # Store the base llm_client for fallback
        self._base_llm_client = llm_client

        # P0: Warm subagent pool
        self.subagent_pool = SubagentPool(
            max_size=subagent_pool_max_size,
            ttl_by_tier=subagent_pool_ttl_by_tier,
            provider_factory=self.provider_factory,
            tool_router=tool_router,
            authz_engine=authz_engine,
            jlens_gate=jlens_gate,
            dual_llm_gate=dual_llm_gate,
            tracer=tracer,
            logger=logger,
        )

        # Initialize global pool for cross-orchestrator access
        init_global_pool(
            max_size=subagent_pool_max_size,
            ttl_by_tier=subagent_pool_ttl_by_tier,
            provider_factory=self.provider_factory,
            tool_router=tool_router,
            authz_engine=authz_engine,
            jlens_gate=jlens_gate,
            dual_llm_gate=dual_llm_gate,
            tracer=tracer,
            logger=logger,
        )

    def register_global_hook(self, hook_type: HookType, hook: Hook) -> None:
        """Register a hook that applies to all agents."""
        self._global_hooks[hook_type].append(hook)

    async def spawn_agent(
        self,
        role: AgentRole = AgentRole.ACTOR,
        tools: list[str] | None = None,
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        parent_id: str | None = None,
        max_tool_calls: int = 50,
        max_runtime_seconds: float = 300.0,
        # P0: Delegation parameters
        subagent_tier: Optional[SubagentTier] = None,
        subagent_effort: Optional[EffortLevel] = None,
        specialization: str = "general",
        briefing: str = "",
        use_pool: bool = True,
    ) -> AgentLoop:
        """Spawn a new agent with isolated context.

        Args:
            role: Agent role (actor, verifier, planner, explorer, reviewer, tester, designer)
            tools: Available tools for this agent.
            permission_mode: Permission mode for tool access.
            parent_id: Parent agent ID for context inheritance.
            max_tool_calls: Maximum tool calls before budget exhaustion.
            max_runtime_seconds: Maximum runtime.
            subagent_tier: Compute tier for this subagent (P0).
            subagent_effort: Reasoning effort level (P0).
            specialization: Specialist type (explorer, reviewer, tester, designer, general).
            briefing: Briefing context for warm pool matching.
            use_pool: Whether to use warm subagent pool (P0).

        Returns:
            Configured AgentLoop instance.
        """
        # Coerce string tier/effort (CLI / harness may pass bare strings)
        if isinstance(subagent_tier, str):
            subagent_tier = SubagentTier(subagent_tier)
        if isinstance(subagent_effort, str):
            subagent_effort = EffortLevel(subagent_effort)

        # Determine tier/effort from role if not specified
        if subagent_tier is None:
            if role in (AgentRole.EXPLORER, AgentRole.REVIEWER, AgentRole.TESTER):
                subagent_tier = SubagentTier.STANDARD
            elif role == AgentRole.DESIGNER:
                subagent_tier = SubagentTier.LARGE
            else:
                subagent_tier = SubagentTier.STANDARD

        if subagent_effort is None:
            if role in (AgentRole.EXPLORER, AgentRole.REVIEWER):
                subagent_effort = EffortLevel.LOW
            elif role == AgentRole.TESTER:
                subagent_effort = EffortLevel.MEDIUM
            elif role == AgentRole.DESIGNER:
                subagent_effort = EffortLevel.HIGH
            else:
                subagent_effort = EffortLevel.MEDIUM

        # Try to get warm subagent from pool
        if use_pool:
            agent, is_warm = await self.subagent_pool.get_or_create(
                tier=subagent_tier,
                effort=subagent_effort,
                specialization=specialization,
                briefing=briefing,
                tools=tools,
                permission_mode=permission_mode,
                parent_id=parent_id,
                max_tool_calls=max_tool_calls,
                max_runtime_seconds=max_runtime_seconds,
            )
            if is_warm:
                logger.info(f"Warm subagent return: {specialization} ({subagent_tier}/{subagent_effort})")
                # Update context with new parameters
                agent.context.role = role
                agent.context.parent_id = parent_id
                agent.context.tools_available = tools or []
                agent.context.permission_mode = permission_mode
                agent.context.max_tool_calls = max_tool_calls
                agent.context.max_runtime_seconds = max_runtime_seconds
                agent.context.subagent_tier = subagent_tier
                agent.context.subagent_effort = subagent_effort
                agent.context.specialization = specialization
                agent.context.briefing = briefing
                return agent

        # Cold start — create new agent (fallback or pool disabled)
        context = AgentContext(
            role=role,
            parent_id=parent_id,
            tools_available=tools or [],
            permission_mode=permission_mode,
            max_tool_calls=max_tool_calls,
            max_runtime_seconds=max_runtime_seconds,
            subagent_tier=subagent_tier,
            subagent_effort=subagent_effort,
            specialization=specialization,
            briefing=briefing,
        )

        # Combine global hooks with any role-specific hooks
        hooks = {ht: list(self._global_hooks[ht]) for ht in HookType}

        # RIG: Add dual-LLM taint gate as PreToolUse hook if available (deny-first)
        if self.dual_llm_gate and role == AgentRole.ACTOR:
            hooks[HookType.PRE_TOOL_USE].append(self.dual_llm_gate.check)

        # Add J-lens gate as PreToolUse hook if available
        if self.jlens_gate and role == AgentRole.ACTOR:
            hooks[HookType.PRE_TOOL_USE].append(self.jlens_gate.check)

        # Add authz check as PreToolUse hook
        if self.authz_engine:
            hooks[HookType.PRE_TOOL_USE].append(self.authz_engine.check_permission)

        # Use provided base llm_client if available (e.g., MockProvider in tests),
        # otherwise build provider from tier/effort
        provider = self._base_llm_client or self.provider_factory.build_from_tier_effort(subagent_tier, subagent_effort)

        # Add effort escalation hook
        escalation_hook = EffortEscalationHook()
        hooks[HookType.PRE_COMPACT].append(escalation_hook)

        agent = AgentLoop(
            context=context,
            llm_client=provider,
            tool_router=self.tool_router,
            hooks=hooks,
            tracer=self.tracer,
            logger=self.obs_logger,
            provider_factory=self.provider_factory,
        )

        self.agents[context.agent_id] = agent
        return agent

    async def run_task(
        self,
        task: str,
        tools: list[str] | None = None,
        verify: bool = True,
        max_tool_calls: int = 50,
        max_runtime_seconds: float = 300.0,
        # P0: Delegation parameters for top-level task
        subagent_tier: Optional[SubagentTier] = None,
        subagent_effort: Optional[EffortLevel] = None,
        # Budget parameters
        max_cost_usd_cents: int = 1000,  # $10 default
    ) -> dict[str, Any]:
        """Run a complete task with optional verification.

        Args:
            task: Task description / user message.
            tools: Available tools.
            verify: Whether to run async verification on completion.
            max_tool_calls: Maximum tool calls before budget exhaustion.
            max_runtime_seconds: Maximum runtime.
            subagent_tier: Compute tier for primary agent (P0).
            subagent_effort: Reasoning effort for primary agent (P0).
            max_cost_usd_cents: Maximum cost in USD cents for this task.

        Returns:
            Task result with verification status and cost tracking.
        """
        if self._kill_switch:
            return {"status": "killed", "output": "Global kill switch activated"}

        # Set up delegation budget for the primary agent
        if self.authz_engine:
            # We'll set the budget after spawning the agent
            pass

        # Spawn actor agent with tier/effort
        actor = await self.spawn_agent(
            role=AgentRole.ACTOR,
            tools=tools,
            max_tool_calls=max_tool_calls,
            max_runtime_seconds=max_runtime_seconds,
            subagent_tier=subagent_tier,
            subagent_effort=subagent_effort,
            specialization="general",
            briefing=task,  # Top-level task is the briefing
        )

        # Set up delegation budget for this agent
        if self.authz_engine:
            self.authz_engine.set_delegation_budget_limit(actor.context.agent_id, "max_cost_usd_cents", max_cost_usd_cents)
            self.authz_engine.set_delegation_budget_limit(actor.context.agent_id, "max_tool_calls", max_tool_calls)

        # Run the agent loop
        result = await actor.run(initial_message=task)

        # Add budget status to result
        if self.authz_engine:
            result["budget_status"] = self.authz_engine.get_delegation_budget_status(actor.context.agent_id)

        # Verify if completion claimed and verification enabled
        if verify and result["status"] == "completion_claimed" and self.verifier:
            verdict = await self.verifier.verify(result["output"])
            result["verification"] = verdict

            if verdict.get("status") == "REVISE":
                # Retry with feedback
                retry_msg = f"Verification failed. Issues: {verdict.get('failures', [])}. Please revise."
                result = await actor.run(initial_message=retry_msg)
                if self.authz_engine:
                    result["budget_status"] = self.authz_engine.get_delegation_budget_status(actor.context.agent_id)

        return result

    # P0: Delegation helpers
    async def delegate_to_specialist(
        self,
        task: str,
        specialist: AgentRole,
        briefing: str,
        tools: list[str] | None = None,
        tier: Optional[SubagentTier] = None,
        effort: Optional[EffortLevel] = None,
        verify: bool = True,
        max_cost_usd_cents: int = 500,  # $5 default for specialists
    ) -> dict[str, Any]:
        """Delegate a task to a specialist subagent (P0 - War Room).

        Args:
            task: Task description.
            specialist: Specialist role (EXPLORER, REVIEWER, TESTER, DESIGNER).
            briefing: Detailed briefing for the specialist.
            tools: Tools available to the specialist. If None, uses specialist-specific allowlist.
            tier: Override default tier for this specialist.
            effort: Override default effort for this specialist.
            verify: Whether to verify the result.
            max_cost_usd_cents: Maximum cost in USD cents for this delegation.

        Returns:
            Specialist result with verification and budget status.
        """
        # Default tier/effort per specialist
        defaults = {
            AgentRole.EXPLORER: (SubagentTier.STANDARD, EffortLevel.LOW),
            AgentRole.REVIEWER: (SubagentTier.STANDARD, EffortLevel.LOW),
            AgentRole.TESTER: (SubagentTier.STANDARD, EffortLevel.MEDIUM),
            AgentRole.DESIGNER: (SubagentTier.LARGE, EffortLevel.HIGH),
        }
        default_tier, default_effort = defaults.get(specialist, (SubagentTier.STANDARD, EffortLevel.MEDIUM))

        # If no tools specified, use specialist-specific allowlist
        if tools is None:
            from prometheus.harness.specialists.config import (
                SpecialistType,
                TOOL_ALLOWLISTS,
            )
            specialist_type_map = {
                AgentRole.EXPLORER: SpecialistType.EXPLORER,
                AgentRole.REVIEWER: SpecialistType.REVIEWER,
                AgentRole.TESTER: SpecialistType.TESTER,
                AgentRole.DESIGNER: SpecialistType.DESIGNER,
            }
            specialist_type = specialist_type_map.get(specialist)
            if specialist_type:
                tools = TOOL_ALLOWLISTS.get(specialist_type, [])

        specialist_agent = await self.spawn_agent(
            role=specialist,
            tools=tools,
            briefing=briefing,
            subagent_tier=tier or default_tier,
            subagent_effort=effort or default_effort,
            specialization=specialist.value,
        )

        # Set up delegation budget for this specialist
        if self.authz_engine:
            self.authz_engine.set_delegation_budget_limit(specialist_agent.context.agent_id, "max_cost_usd_cents", max_cost_usd_cents)
            self.authz_engine.set_delegation_budget_limit(specialist_agent.context.agent_id, "max_tool_calls", 30)  # Lower limit for specialists

        result = await specialist_agent.run(initial_message=task)

        # Add budget status to result
        if self.authz_engine:
            result["budget_status"] = self.authz_engine.get_delegation_budget_status(specialist_agent.context.agent_id)

        if verify and result["status"] == "completion_claimed" and self.verifier:
            verdict = await self.verifier.verify(result["output"])
            result["verification"] = verdict

        return result

    async def run_delegated_task_graph(
        self,
        tasks: list[dict[str, Any]],
        # Each task: {"task": str, "specialist": AgentRole, "briefing": str, "tools": [], "depends_on": []}
    ) -> dict[str, Any]:
        """Run a DAG of delegated specialist tasks (P1 - War Room).

        Future: dependency resolution, parallel execution, result passing.
        """
        # Placeholder for P1 implementation
        raise NotImplementedError("Delegated task graph coming in P1")

    def kill(self) -> None:
        """Activate global kill switch — stops all agents."""
        self._kill_switch = True
        for agent in self.agents.values():
            agent.stop()

    async def shutdown(self) -> None:
        """Shutdown orchestrator and subagent pool."""
        self.kill()
        await self.subagent_pool.shutdown()
        # Also shutdown global pool
        pool = get_global_pool()
        if pool:
            await pool.shutdown()

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status."""
        pool_stats = self.subagent_pool.get_stats()
        return {
            "active_agents": len([a for a in self.agents.values() if a._running]),
            "total_agents_spawned": len(self.agents),
            "kill_switch": self._kill_switch,
            "has_verifier": self.verifier is not None,
            "has_jlens_gate": self.jlens_gate is not None,
            "has_dual_llm_gate": self.dual_llm_gate is not None,
            "has_authz": self.authz_engine is not None,
            "subagent_pool": pool_stats,
        }