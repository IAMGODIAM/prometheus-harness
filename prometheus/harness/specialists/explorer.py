"""ExplorerAgent — Read-only specialist for exploration, search, and analysis.

The ExplorerAgent is a specialist subagent with read-only tool access:
- web_search, web_extract: Web search and content extraction
- search_files, read_file: Codebase search and file reading
- jlens_probe, jlens_decompose, jlens_watchlist_scores: Mechanistic interpretability
- sovereign_search, sovereign_knowledge_search: Sovereign search mesh
- skill_view, skills_list: Skill discovery
- analyze_code, analyze_architecture: Code analysis tools

No write, execute, or modification tools are available.
"""

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


class ExplorerAgent:
    """Read-only exploration specialist agent.

    The ExplorerAgent handles:
    - Web search and content extraction
    - Codebase search and file reading
    - Mechanistic interpretability (J-Lens)
    - Sovereign search mesh queries
    - Skill discovery and analysis
    - Code and architecture analysis

    All tools are read-only — no file writes, no command execution, no modifications.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        config: SpecialistConfig | None = None,
    ):
        """Initialize the ExplorerAgent.

        Args:
            orchestrator: The parent Orchestrator instance for spawning agents.
            config: Optional SpecialistConfig override. If None, uses default EXPLORER config.
        """
        self.orchestrator = orchestrator
        self.config = config or get_specialist_config(SpecialistType.EXPLORER)
        self._agent: AgentLoop | None = None

    @property
    def specialist_type(self) -> SpecialistType:
        return SpecialistType.EXPLORER

    @property
    def available_tools(self) -> list[str]:
        return self.config.tools

    async def spawn(self, briefing: str = "", parent_id: str | None = None) -> AgentLoop:
        """Spawn the explorer agent with isolated context.

        Args:
            briefing: Detailed briefing/context for the exploration task.
            parent_id: Optional parent agent ID for context inheritance.

        Returns:
            The spawned AgentLoop instance.
        """
        # Map string tier/effort to enums
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
            role=AgentRole.EXPLORER,
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
            f"Spawned ExplorerAgent: {self._agent.context.agent_id} "
            f"(tier={self.config.tier}, effort={self.config.effort}, "
            f"tools={len(self.config.tools)})"
        )

        return self._agent

    async def run(self, task: str, briefing: str = "") -> dict[str, Any]:
        """Run an exploration task.

        Args:
            task: The exploration task description.
            briefing: Additional context/briefing for the agent.

        Returns:
            Task result with status, output, and metadata.
        """
        if self._agent is None:
            await self.spawn(briefing=briefing)

        assert self._agent is not None
        result = await self._agent.run(initial_message=task)

        logger.info(
            f"ExplorerAgent {self._agent.context.agent_id} completed: "
            f"status={result.get('status')}, tool_calls={len(result.get('tool_calls', []))}"
        )

        return result

    async def explore_codebase(
        self,
        query: str,
        briefing: str = "Explore the codebase to find relevant files and patterns.",
    ) -> dict[str, Any]:
        """Explore the codebase for a specific query.

        Args:
            query: The exploration query (e.g., "find all authentication handlers").
            briefing: Optional additional briefing context.

        Returns:
            Exploration result with findings.
        """
        task = f"""
Explore the codebase for: {query}

Use the available read-only tools:
- search_files: Search for patterns in files
- read_file: Read relevant files
- analyze_code: Analyze code structure and patterns
- analyze_architecture: Understand architectural patterns

Provide a summary of findings with file paths and relevant code snippets.
"""
        return await self.run(task, briefing)

    async def search_web(
        self,
        query: str,
        briefing: str = "Search the web for relevant information.",
    ) -> dict[str, Any]:
        """Search the web for information.

        Args:
            query: The search query.
            briefing: Optional additional briefing context.

        Returns:
            Search result with extracted content.
        """
        task = f"""
Search the web for: {query}

Use the available tools:
- web_search: Search for relevant pages
- web_extract: Extract content from relevant pages

Provide a summary of findings with sources and key information.
"""
        return await self.run(task, briefing)

    async def analyze_with_jlens(
        self,
        prompt: str,
        briefing: str = "Analyze the prompt using J-Lens mechanistic interpretability.",
    ) -> dict[str, Any]:
        """Analyze a prompt using J-Lens tools.

        Args:
            prompt: The prompt to analyze.
            briefing: Optional additional briefing context.

        Returns:
            J-Lens analysis result.
        """
        task = f"""
Analyze the following prompt using J-Lens mechanistic interpretability tools:
{prompt}

Use the available tools:
- jlens_probe: Probe the prompt for token/concept activations
- jlens_decompose: Decompose activations at specific layers/positions
- jlens_watchlist_scores: Score against configured watchlist categories

Provide the analysis results with scores and interpretations.
"""
        return await self.run(task, briefing)

    async def search_skills(
        self,
        query: str,
        briefing: str = "Search for relevant skills in the Hermes skill library.",
    ) -> dict[str, Any]:
        """Search for relevant skills.

        Args:
            query: The skill search query.
            briefing: Optional additional briefing context.

        Returns:
            Skill search results.
        """
        task = f"""
Search for skills related to: {query}

Use the available tools:
- skills_list: List available skills (optionally filtered by category)
- skill_view: View full skill content including references/templates/scripts

Provide a summary of relevant skills with their descriptions and use cases.
"""
        return await self.run(task, briefing)

    def get_context(self) -> AgentContext | None:
        """Get the agent's context if spawned."""
        return self._agent.context if self._agent else None

    def is_spawned(self) -> bool:
        """Check if the agent has been spawned."""
        return self._agent is not None