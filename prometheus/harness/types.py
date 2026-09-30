"""Shared types for Prometheus harness — avoids circular imports."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Optional

from prometheus.harness.model_client import SubagentTier, EffortLevel


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
    # Delegation specialists (P0 - War Room)
    EXPLORER = "explorer"
    REVIEWER = "reviewer"
    TESTER = "tester"
    DESIGNER = "designer"


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

    # Delegation metadata (P0)
    subagent_tier: Optional[SubagentTier] = None
    subagent_effort: Optional[EffortLevel] = None
    specialization: str = "general"
    briefing: str = ""

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

# Forward reference for AgentLoop (defined in orchestrator.py)
AgentLoop = Any