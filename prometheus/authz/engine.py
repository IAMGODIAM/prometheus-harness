"""Authorization Engine — Cedar-inspired policy evaluation.

Cedar-style PARC model with deny-by-default semantics:
- permit/forbid policies with conditions
- Deny overrides allow (a broad allow cannot override a narrow deny)
- Sub-millisecond evaluation for inline tool-call authorization
- Integration with J-lens concept scores as policy conditions
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PolicyEffect(str, Enum):
    PERMIT = "permit"
    FORBID = "forbid"


class ToolCategory(str, Enum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    EXTERNAL_VISIBLE = "external_visible"
    IRREVERSIBLE = "irreversible"


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"  # Requires human confirmation


@dataclass
class Policy:
    """A single authorization policy (Cedar-inspired).

    Attributes:
        id: Unique policy identifier.
        effect: permit or forbid.
        principal: Who is making the request (agent_id, role, etc.)
        action: What action is being performed (tool name, category).
        resource: What resource is being accessed.
        conditions: Additional conditions that must be met.
        jlens_conditions: J-lens concept score conditions.
        priority: Higher priority policies are evaluated first.
    """
    id: str
    effect: PolicyEffect
    principal: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)
    resource: dict[str, Any] = field(default_factory=dict)
    conditions: dict[str, Any] = field(default_factory=dict)
    jlens_conditions: dict[str, float] = field(default_factory=dict)
    priority: int = 0
    description: str = ""

    def matches(self, request: AuthzRequest) -> bool:
        """Check if this policy matches the given request."""
        # Check principal
        if self.principal:
            for key, value in self.principal.items():
                if request.principal.get(key) != value:
                    if value != "*":
                        return False

        # Check action
        if self.action:
            action_name = self.action.get("name", "*")
            action_category = self.action.get("category", "*")

            if action_name != "*" and action_name != request.action_name:
                return False
            if action_category != "*" and action_category != request.action_category:
                return False

        # Check resource
        if self.resource:
            for key, value in self.resource.items():
                if request.resource.get(key) != value:
                    if value != "*":
                        return False

        # Check J-lens conditions
        if self.jlens_conditions and request.jlens_scores:
            for concept, threshold in self.jlens_conditions.items():
                score = request.jlens_scores.get(concept, 0.0)
                # Forbid policies trigger when score EXCEEDS threshold
                # Permit policies require score BELOW threshold
                if self.effect == PolicyEffect.FORBID and score < threshold:
                    return False  # Score below threshold, forbid doesn't apply
                if self.effect == PolicyEffect.PERMIT and score > threshold:
                    return False  # Score above threshold, permit doesn't apply

        # Check custom conditions
        if self.conditions:
            for key, expected in self.conditions.items():
                actual = request.context.get(key)
                if actual != expected:
                    return False

        return True


@dataclass
class AuthzRequest:
    """Authorization request to evaluate."""
    principal: dict[str, Any]  # {agent_id, role, user_id, spiffe_id}
    action_name: str           # Tool name
    action_category: str = ""  # Tool category (read_only, irreversible, etc.)
    resource: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    jlens_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class AuthzDecision:
    """Result of policy evaluation."""
    decision: Decision
    reason: str = ""
    matching_policies: list[str] = field(default_factory=list)
    evaluation_time_ms: float = 0.0


class AuthzEngine:
    """Cedar-style authorization engine with deny-first semantics.

    Evaluation order:
    1. Collect all matching policies
    2. If ANY forbid policy matches → DENY (deny-first invariant)
    3. If at least one permit policy matches → ALLOW
    4. Otherwise → DENY (deny-by-default)

    Integration points:
    - PreToolUse hook in the orchestrator
    - J-lens concept scores as policy conditions
    - Taint labels for information flow
    - Budget enforcement
    """

    def __init__(self):
        self.policies: list[Policy] = []
        self._tool_categories: dict[str, ToolCategory] = {}
        self._budgets: dict[str, dict[str, int]] = {}  # agent_id -> {counter: value}
        self._budget_limits: dict[str, int] = {
            "max_tool_calls": 100,
            "max_writes": 20,
            "max_external_calls": 10,
            "max_irreversible": 3,
        }

    def add_policy(self, policy: Policy) -> None:
        """Add a policy to the engine."""
        self.policies.append(policy)
        # Keep sorted by priority (higher first)
        self.policies.sort(key=lambda p: p.priority, reverse=True)

    def remove_policy(self, policy_id: str) -> None:
        """Remove a policy by ID."""
        self.policies = [p for p in self.policies if p.id != policy_id]

    def classify_tool(self, tool_name: str, category: ToolCategory) -> None:
        """Classify a tool into a category for policy matching."""
        self._tool_categories[tool_name] = category

    def set_budget_limit(self, limit_name: str, value: int) -> None:
        """Set a budget limit."""
        self._budget_limits[limit_name] = value

    def evaluate(self, request: AuthzRequest) -> AuthzDecision:
        """Evaluate an authorization request against all policies.

        Implements deny-first semantics:
        - Any matching forbid → DENY
        - At least one matching permit → ALLOW
        - No matches → DENY (deny-by-default)
        """
        start_time = time.time()

        # Enrich request with tool category if not provided
        if not request.action_category and request.action_name in self._tool_categories:
            request.action_category = self._tool_categories[request.action_name].value

        # Check budget limits
        budget_decision = self._check_budget(request)
        if budget_decision:
            return budget_decision

        # Evaluate policies
        matching_permits: list[str] = []
        matching_forbids: list[str] = []

        for policy in self.policies:
            if policy.matches(request):
                if policy.effect == PolicyEffect.FORBID:
                    matching_forbids.append(policy.id)
                else:
                    matching_permits.append(policy.id)

        elapsed = (time.time() - start_time) * 1000

        # Deny-first: any forbid overrides all permits
        if matching_forbids:
            forbid_policy = next(p for p in self.policies if p.id == matching_forbids[0])
            return AuthzDecision(
                decision=Decision.DENY,
                reason=forbid_policy.description or f"Denied by policy: {matching_forbids[0]}",
                matching_policies=matching_forbids,
                evaluation_time_ms=elapsed,
            )

        # At least one permit → allow
        if matching_permits:
            return AuthzDecision(
                decision=Decision.ALLOW,
                reason=f"Permitted by: {matching_permits[0]}",
                matching_policies=matching_permits,
                evaluation_time_ms=elapsed,
            )

        # No matches → deny by default
        return AuthzDecision(
            decision=Decision.DENY,
            reason="No matching permit policy (deny-by-default)",
            matching_policies=[],
            evaluation_time_ms=elapsed,
        )

    async def check_permission(self, data: dict[str, Any]) -> Any:
        """Hook-compatible permission check for orchestrator integration.

        Called as a PreToolUse hook. Converts hook data to AuthzRequest
        and evaluates against policies.
        """
        from prometheus.harness.orchestrator import HookResult

        context = data.get("context")
        tool_name = data.get("tool_name", "")

        request = AuthzRequest(
            principal={
                "agent_id": context.agent_id if context else "unknown",
                "role": context.role.value if context else "unknown",
            },
            action_name=tool_name,
            context=data.get("arguments", {}),
            jlens_scores=data.get("jlens_scores", {}),
        )

        decision = self.evaluate(request)

        if decision.decision == Decision.DENY:
            return HookResult(
                allow=False,
                reason=decision.reason,
                escalate=False,
            )
        elif decision.decision == Decision.ASK:
            return HookResult(
                allow=False,
                reason=f"Human confirmation required: {decision.reason}",
                escalate=True,
            )

        # Track budget
        self._increment_budget(request)

        return HookResult(allow=True)

    def _check_budget(self, request: AuthzRequest) -> AuthzDecision | None:
        """Check if the request would exceed budget limits."""
        agent_id = request.principal.get("agent_id", "default")

        if agent_id not in self._budgets:
            self._budgets[agent_id] = {k: 0 for k in self._budget_limits}

        budgets = self._budgets[agent_id]

        # Check total tool calls
        if budgets.get("max_tool_calls", 0) >= self._budget_limits["max_tool_calls"]:
            return AuthzDecision(
                decision=Decision.DENY,
                reason="Tool call budget exhausted",
            )

        # Check category-specific budgets
        category = self._tool_categories.get(request.action_name)
        if category == ToolCategory.IRREVERSIBLE:
            if budgets.get("max_irreversible", 0) >= self._budget_limits["max_irreversible"]:
                return AuthzDecision(
                    decision=Decision.ASK,
                    reason="Irreversible action budget exhausted — requires human approval",
                )
        elif category == ToolCategory.EXTERNAL_VISIBLE:
            if budgets.get("max_external_calls", 0) >= self._budget_limits["max_external_calls"]:
                return AuthzDecision(
                    decision=Decision.ASK,
                    reason="External call budget exhausted — requires human approval",
                )

        return None

    def _increment_budget(self, request: AuthzRequest) -> None:
        """Increment budget counters after a permitted action."""
        agent_id = request.principal.get("agent_id", "default")
        if agent_id not in self._budgets:
            self._budgets[agent_id] = {k: 0 for k in self._budget_limits}

        self._budgets[agent_id]["max_tool_calls"] = (
            self._budgets[agent_id].get("max_tool_calls", 0) + 1
        )

        category = self._tool_categories.get(request.action_name)
        if category == ToolCategory.IRREVERSIBLE:
            self._budgets[agent_id]["max_irreversible"] = (
                self._budgets[agent_id].get("max_irreversible", 0) + 1
            )
        elif category == ToolCategory.EXTERNAL_VISIBLE:
            self._budgets[agent_id]["max_external_calls"] = (
                self._budgets[agent_id].get("max_external_calls", 0) + 1
            )

    def load_default_policies(self) -> None:
        """Load sensible default policies for the harness.

        Default posture: deny-by-default with permits for read-only tools
        and J-lens-gated permits for higher-risk actions.
        """
        # Permit read-only tools for all agents
        self.add_policy(Policy(
            id="default_permit_readonly",
            effect=PolicyEffect.PERMIT,
            action={"category": "read_only"},
            description="Allow all read-only tool calls",
            priority=10,
        ))

        # Permit reversible tools for actor agents
        self.add_policy(Policy(
            id="default_permit_reversible",
            effect=PolicyEffect.PERMIT,
            principal={"role": "actor"},
            action={"category": "reversible"},
            description="Allow reversible actions for actor agents",
            priority=10,
        ))

        # Forbid external-visible tools when deception score is high
        self.add_policy(Policy(
            id="jlens_gate_deception",
            effect=PolicyEffect.FORBID,
            action={"category": "external_visible"},
            jlens_conditions={"deception": 0.15},
            description="Block external actions when deception concepts detected in J-space",
            priority=100,
        ))

        # Forbid network tools when prompt injection detected
        self.add_policy(Policy(
            id="jlens_gate_injection",
            effect=PolicyEffect.FORBID,
            action={"category": "external_visible"},
            jlens_conditions={"prompt_injection": 0.10},
            description="Block external actions when prompt injection detected in J-space",
            priority=100,
        ))

        # Require human approval for irreversible actions
        self.add_policy(Policy(
            id="default_ask_irreversible",
            effect=PolicyEffect.FORBID,
            action={"category": "irreversible"},
            description="Irreversible actions require explicit human approval",
            priority=50,
        ))
