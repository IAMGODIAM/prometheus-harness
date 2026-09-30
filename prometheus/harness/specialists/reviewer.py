"""ReviewerAgent — Analysis specialist for code review, security scanning, and diff analysis."""

from __future__ import annotations

import logging
from typing import Any

from prometheus.harness.orchestrator import (
    Orchestrator,
    AgentRole,
    AgentLoop,
    AgentContext,
    PermissionMode,
    SubagentTier,
    EffortLevel,
)
from prometheus.harness.specialists.config import (
    SpecialistType,
    SpecialistConfig,
    TOOL_ALLOWLISTS,
    get_specialist_config,
)

logger = logging.getLogger(__name__)


class ReviewerAgent:
    """Analysis specialist agent for code review, security scanning, and diff analysis.

    Capabilities:
    - Code review and diff analysis
    - Security vulnerability scanning
    - Static code analysis
    - Dependency analysis
    - J-Lens mechanistic interpretability
    - Code quality assessment
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        config: SpecialistConfig | None = None,
    ):
        """Initialize the ReviewerAgent.

        Args:
            orchestrator: The parent Orchestrator instance for spawning agents.
            config: Optional SpecialistConfig override. If None, uses default REVIEWER config.
        """
        self.orchestrator = orchestrator
        self.config = config or get_specialist_config(SpecialistType.REVIEWER)
        self._agent: AgentLoop | None = None

    @property
    def specialist_type(self) -> SpecialistType:
        return SpecialistType.REVIEWER

    @property
    def available_tools(self) -> list[str]:
        return self.config.tools

    async def spawn(self, briefing: str = "", parent_id: str | None = None) -> AgentLoop:
        """Spawn the reviewer agent with isolated context.

        Args:
            briefing: Detailed briefing/context for the review task.
            parent_id: Optional parent agent ID for context inheritance.

        Returns:
            The spawned AgentLoop instance.
        """
        tier_map = {
            "small": SubagentTier.SMALL,
            "standard": SubagentTier.STANDARD,
            "large": SubagentTier.LARGE,
        }
        effort_map = {
            "low": EffortLevel.LOW,
            "medium": EffortLevel.MEDIUM,
            "high": EffortLevel.HIGH,
            "xhigh": EffortLevel.XHIGH,
        }

        self._agent = await self.orchestrator.spawn_agent(
            role=AgentRole.REVIEWER,
            tools=self.config.tools,
            permission_mode=PermissionMode.DEFAULT,
            parent_id=parent_id,
            max_tool_calls=self.config.max_tool_calls,
            max_runtime_seconds=self.config.max_runtime_seconds,
            subagent_tier=tier_map.get(self.config.tier, SubagentTier.STANDARD),
            subagent_effort=effort_map.get(self.config.effort, EffortLevel.MEDIUM),
            specialization=self.config.specialist_type.value,
            briefing=briefing,
            use_pool=True,
        )

        logger.info(
            f"Spawned ReviewerAgent: {self._agent.context.agent_id} "
            f"(tier={self.config.tier}, effort={self.config.effort}, "
            f"tools={len(self.config.tools)})"
        )

        return self._agent

    async def run(self, task: str, briefing: str = "") -> dict[str, Any]:
        """Run a review/analysis task.

        Args:
            task: The review task description.
            briefing: Additional context/briefing for the agent.

        Returns:
            Task result with status, output, and metadata.
        """
        if self._agent is None:
            await self.spawn(briefing=briefing)

        assert self._agent is not None
        result = await self._agent.run(initial_message=task)

        logger.info(
            f"ReviewerAgent {self._agent.context.agent_id} completed: "
            f"status={result.get('status')}, tool_calls={len(result.get('tool_calls', []))}"
        )

        return result

    async def review_code(
        self,
        target: str,
        briefing: str = "Review the code for quality, security, and best practices.",
    ) -> dict[str, Any]:
        """Review code for quality, security, and best practices.

        Args:
            target: Path to file/directory or GitHub PR reference to review.
            briefing: Optional additional briefing context.

        Returns:
            Review result with findings and recommendations.
        """
        task = f"""
Review the code at: {target}

Use the available analysis tools:
- read_file: Read the target files
- search_files: Find related code patterns
- github_code_review: Perform structured code review (if GitHub PR)
- diff_analysis: Analyze changes in diffs
- security_scan: Scan for security vulnerabilities
- analyze_code: Analyze code structure and patterns
- analyze_dependencies: Check dependency risks
- static_analysis: Run static analysis checks
- jlens_probe: Probe for mechanistic interpretability signals
- jlens_decompose: Decompose specific activations
- jlens_watchlist_scores: Score against security/quality watchlists

Provide a comprehensive review with:
1. Summary of findings (critical, major, minor)
2. Specific file:line references for each issue
3. Security vulnerability assessment
4. Code quality and best practice recommendations
5. Suggested fixes or improvements
"""
        return await self.run(task, briefing)

    async def analyze_diff(
        self,
        diff_content: str,
        briefing: str = "Analyze the diff for potential issues.",
    ) -> dict[str, Any]:
        """Analyze a code diff for potential issues.

        Args:
            diff_content: The diff content to analyze.
            briefing: Optional additional briefing context.

        Returns:
            Diff analysis result with findings.
        """
        task = f"""
Analyze the following diff for potential issues:

{diff_content}

Use the available tools:
- diff_analysis: Structured diff analysis
- security_scan: Check for security issues in changes
- analyze_code: Analyze changed code patterns
- jlens_probe: Probe changes for interpretability signals

Provide analysis covering:
1. Change summary and scope
2. Potential bugs or logic errors
3. Security implications
4. Performance considerations
5. Breaking changes or API impacts
6. Test coverage adequacy
"""
        return await self.run(task, briefing)

    async def security_audit(
        self,
        target: str,
        briefing: str = "Perform a security audit of the target.",
    ) -> dict[str, Any]:
        """Perform a security audit of the target.

        Args:
            target: Path to file/directory to audit.
            briefing: Optional additional briefing context.

        Returns:
            Security audit result with vulnerabilities and recommendations.
        """
        task = f"""
Perform a security audit of: {target}

Use the available tools:
- read_file: Read target files
- search_files: Find security-relevant patterns
- security_scan: Automated security vulnerability scanning
- analyze_dependencies: Check for vulnerable dependencies
- static_analysis: Run static analysis for security issues
- jlens_watchlist_scores: Score against security watchlists

Provide a security audit report with:
1. Vulnerability summary (critical, high, medium, low)
2. Specific findings with file:line references
3. CVE references where applicable
4. Exploitability assessment
5. Remediation recommendations
6. Compliance considerations
"""
        return await self.run(task, briefing)

    def get_context(self) -> AgentContext | None:
        """Get the agent's context if spawned."""
        return self._agent.context if self._agent else None

    def is_spawned(self) -> bool:
        """Check if the agent has been spawned."""
        return self._agent is not None