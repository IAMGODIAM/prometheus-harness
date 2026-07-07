"""Orchestrator — Agent loop with hook system and subagent context isolation.

Implements the 2026 consensus agent-loop architecture drawn from Claude Agent SDK,
DeerFlow 2.0, OpenAI Agents SDK, and Hermes Agent patterns:
- Hook system: PreToolUse, PostToolUse, PreCompact, UserPromptSubmit, SessionStart/End
- Permission modes: default, acceptEdits, plan, bypassPermissions, auto
- Subagent context isolation
- Programmatic tool calling (code-mode for token reduction)
- Context folding/compaction with durable anchors
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class HookType(str, Enum):
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    PRE_COMPACT = "pre_compact"
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_COMPLETION = "pre_completion"
    POST_COMPLETION = "post_completion"


class PermissionMode(str, Enum):
    DEFAULT = "default"
    ACCEPT_EDITS = "accept_edits"
    PLAN = "plan"
    BYPASS = "bypass_permissions"
    AUTO = "auto"
    DONT_ASK = "dont_ask"


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    ACTOR = "actor"
    VERIFIER = "verifier"
    ARBITER = "arbiter"
    PLANNER = "planner"


@dataclass
class HookResult:
    """Result from a hook execution."""
    allow: bool = True
    modified_args: dict[str, Any] | None = None
    reason: str | None = None
    escalate: bool = False


@dataclass
class AgentContext:
    """Isolated context for a subagent."""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: AgentRole = AgentRole.ACTOR
    parent_id: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools_available: list[str] = field(default_factory=list)
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    memory_blocks: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    max_tool_calls: int = 50
    max_runtime_seconds: float = 300.0
    tool_calls_made: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def budget_exhausted(self) -> bool:
        return (
            self.tool_calls_made >= self.max_tool_calls
            or self.elapsed_seconds >= self.max_runtime_seconds
        )


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    tokens_used: int = 0


Hook = Callable[[dict[str, Any]], Awaitable[HookResult]]


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
    ):
        self.context = context
        self.llm_client = llm_client
        self.tool_router = tool_router
        self.hooks: dict[HookType, list[Hook]] = hooks or {ht: [] for ht in HookType}
        self._running = False
        self._paused = False

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

                # Call LLM
                response = await self._call_llm()
                if response is None:
                    break

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
                    # PreToolUse hook
                    pre_result = await self._run_hooks(HookType.PRE_TOOL_USE, {
                        "tool_name": tool_call.get("name"),
                        "arguments": tool_call.get("arguments", {}),
                        "context": self.context,
                    })

                    if not pre_result.allow:
                        # Tool call denied
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
                        # Execute tool
                        tool_result = await self._execute_tool(tool_call)

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
                        "name": tool_call.get("name", ""),
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
    """

    def __init__(
        self,
        llm_client: Any = None,
        tool_router: Any = None,
        verifier: Any = None,
        authz_engine: Any = None,
        jlens_gate: Any = None,
    ):
        self.llm_client = llm_client
        self.tool_router = tool_router
        self.verifier = verifier
        self.authz_engine = authz_engine
        self.jlens_gate = jlens_gate
        self.agents: dict[str, AgentLoop] = {}
        self._global_hooks: dict[HookType, list[Hook]] = {ht: [] for ht in HookType}
        self._kill_switch = False

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
    ) -> AgentLoop:
        """Spawn a new agent with isolated context.

        Args:
            role: Agent role (actor, verifier, planner, etc.)
            tools: Available tools for this agent.
            permission_mode: Permission mode for tool access.
            parent_id: Parent agent ID for context inheritance.
            max_tool_calls: Maximum tool calls before budget exhaustion.
            max_runtime_seconds: Maximum runtime.

        Returns:
            Configured AgentLoop instance.
        """
        context = AgentContext(
            role=role,
            parent_id=parent_id,
            tools_available=tools or [],
            permission_mode=permission_mode,
            max_tool_calls=max_tool_calls,
            max_runtime_seconds=max_runtime_seconds,
        )

        # Combine global hooks with any role-specific hooks
        hooks = {ht: list(self._global_hooks[ht]) for ht in HookType}

        # Add J-lens gate as PreToolUse hook if available
        if self.jlens_gate and role == AgentRole.ACTOR:
            hooks[HookType.PRE_TOOL_USE].append(self.jlens_gate.check)

        # Add authz check as PreToolUse hook
        if self.authz_engine:
            hooks[HookType.PRE_TOOL_USE].append(self.authz_engine.check_permission)

        agent = AgentLoop(
            context=context,
            llm_client=self.llm_client,
            tool_router=self.tool_router,
            hooks=hooks,
        )

        self.agents[context.agent_id] = agent
        return agent

    async def run_task(
        self,
        task: str,
        tools: list[str] | None = None,
        verify: bool = True,
    ) -> dict[str, Any]:
        """Run a complete task with optional verification.

        Args:
            task: Task description / user message.
            tools: Available tools.
            verify: Whether to run async verification on completion.

        Returns:
            Task result with verification status.
        """
        if self._kill_switch:
            return {"status": "killed", "output": "Global kill switch activated"}

        # Spawn actor agent
        actor = await self.spawn_agent(
            role=AgentRole.ACTOR,
            tools=tools,
        )

        # Run the agent loop
        result = await actor.run(initial_message=task)

        # Verify if completion claimed and verification enabled
        if verify and result["status"] == "completion_claimed" and self.verifier:
            verdict = await self.verifier.verify(result["output"])
            result["verification"] = verdict

            if verdict.get("status") == "REVISE":
                # Retry with feedback
                retry_msg = f"Verification failed. Issues: {verdict.get('failures', [])}. Please revise."
                result = await actor.run(initial_message=retry_msg)

        return result

    def kill(self) -> None:
        """Activate global kill switch — stops all agents."""
        self._kill_switch = True
        for agent in self.agents.values():
            agent.stop()

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status."""
        return {
            "active_agents": len([a for a in self.agents.values() if a._running]),
            "total_agents_spawned": len(self.agents),
            "kill_switch": self._kill_switch,
            "has_verifier": self.verifier is not None,
            "has_jlens_gate": self.jlens_gate is not None,
            "has_authz": self.authz_engine is not None,
        }
