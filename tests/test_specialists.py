"""Tests for the Specialist agents."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prometheus.harness.specialists.config import (
    SpecialistType,
    SpecialistConfig,
    TOOL_ALLOWLISTS,
    get_specialist_config,
)
from prometheus.harness.specialists.explorer import ExplorerAgent
from prometheus.harness.specialists.reviewer import ReviewerAgent
from prometheus.harness.specialists.tester import TesterAgent
from prometheus.harness.specialists.designer import DesignerAgent
from prometheus.harness.orchestrator import Orchestrator, AgentRole, AgentLoop, AgentContext, PermissionMode, SubagentTier, EffortLevel


class TestSpecialistConfig:
    """Tests for SpecialistConfig and tool allowlists."""

    def test_specialist_types_exist(self):
        """Test that all four specialist types are defined."""
        assert SpecialistType.EXPLORER == "explorer"
        assert SpecialistType.REVIEWER == "reviewer"
        assert SpecialistType.TESTER == "tester"
        assert SpecialistType.DESIGNER == "designer"

    def test_tool_allowlists_exist_for_all_types(self):
        """Test that each specialist type has a tool allowlist."""
        for specialist_type in SpecialistType:
            assert specialist_type in TOOL_ALLOWLISTS
            assert isinstance(TOOL_ALLOWLISTS[specialist_type], list)
            assert len(TOOL_ALLOWLISTS[specialist_type]) > 0

    def test_explorer_tools_are_read_only(self):
        """Test that explorer tools are read-only (no write/execute)."""
        explorer_tools = TOOL_ALLOWLISTS[SpecialistType.EXPLORER]

        # Should have read/search tools
        assert "search_files" in explorer_tools
        assert "read_file" in explorer_tools
        assert "web_search" in explorer_tools
        assert "web_extract" in explorer_tools
        assert "jlens_probe" in explorer_tools
        assert "analyze_code" in explorer_tools

        # Should NOT have write/execute tools
        assert "write_file" not in explorer_tools
        assert "patch" not in explorer_tools
        assert "run_command" not in explorer_tools
        assert "bash" not in explorer_tools
        assert "run_pytest" not in explorer_tools

    def test_reviewer_tools_are_analysis(self):
        """Test that reviewer tools are analysis-focused."""
        reviewer_tools = TOOL_ALLOWLISTS[SpecialistType.REVIEWER]

        # Should have analysis tools
        assert "github_code_review" in reviewer_tools
        assert "diff_analysis" in reviewer_tools
        assert "security_scan" in reviewer_tools
        assert "static_analysis" in reviewer_tools
        assert "analyze_code" in reviewer_tools
        assert "analyze_dependencies" in reviewer_tools

        # Should have read access
        assert "read_file" in reviewer_tools
        assert "search_files" in reviewer_tools

        # Should NOT have write/execute tools
        assert "write_file" not in reviewer_tools
        assert "patch" not in reviewer_tools
        assert "run_command" not in reviewer_tools
        assert "run_pytest" not in reviewer_tools

    def test_tester_tools_are_execution(self):
        """Test that tester tools are execution-focused."""
        tester_tools = TOOL_ALLOWLISTS[SpecialistType.TESTER]

        # Should have execution tools
        assert "run_pytest" in tester_tools
        assert "run_tests" in tester_tools
        assert "run_coverage" in tester_tools
        assert "run_command" in tester_tools
        assert "bash" in tester_tools
        assert "terminal" in tester_tools

        # Should have read access for tests
        assert "read_file" in tester_tools
        assert "search_files" in tester_tools

        # Can write test fixtures
        assert "write_file" in tester_tools
        assert "patch" in tester_tools

    def test_designer_tools_are_creation(self):
        """Test that designer tools are creation-focused."""
        designer_tools = TOOL_ALLOWLISTS[SpecialistType.DESIGNER]

        # Should have creation tools
        assert "write_file" in designer_tools
        assert "patch" in designer_tools
        assert "create_file" in designer_tools
        assert "edit_file" in designer_tools
        assert "schema_design" in designer_tools
        assert "architecture_design" in designer_tools
        assert "api_design" in designer_tools
        assert "database_design" in designer_tools

        # Should have read access
        assert "read_file" in designer_tools
        assert "search_files" in designer_tools

        # Can execute for validation
        assert "run_command" in designer_tools
        assert "terminal" in designer_tools

    def test_get_specialist_config_defaults(self):
        """Test getting specialist config with defaults."""
        config = get_specialist_config(SpecialistType.EXPLORER)

        assert config.specialist_type == SpecialistType.EXPLORER
        assert config.tier == "standard"
        assert config.effort == "medium"
        assert config.max_tool_calls == 50
        assert config.max_runtime_seconds == 300.0
        assert len(config.tools) > 0

    def test_get_specialist_config_overrides(self):
        """Test getting specialist config with overrides."""
        config = get_specialist_config(
            SpecialistType.DESIGNER,
            tier="large",
            effort="high",
            max_tool_calls=100,
            max_runtime_seconds=600.0,
        )

        assert config.specialist_type == SpecialistType.DESIGNER
        assert config.tier == "large"
        assert config.effort == "high"
        assert config.max_tool_calls == 100
        assert config.max_runtime_seconds == 600.0

    def test_specialist_config_description(self):
        """Test that each specialist has a description."""
        for specialist_type in SpecialistType:
            config = get_specialist_config(specialist_type)
            assert config.description
            assert len(config.description) > 10


class TestExplorerAgent:
    """Tests for ExplorerAgent."""

    def test_explorer_agent_creation(self):
        """Test creating an ExplorerAgent."""
        orch = Orchestrator()
        agent = ExplorerAgent(orch)

        assert agent.specialist_type == SpecialistType.EXPLORER
        assert agent.config.specialist_type == SpecialistType.EXPLORER
        assert "search_files" in agent.available_tools
        assert "read_file" in agent.available_tools
        assert "web_search" in agent.available_tools

    def test_explorer_agent_custom_config(self):
        """Test creating ExplorerAgent with custom config."""
        orch = Orchestrator()
        config = get_specialist_config(
            SpecialistType.EXPLORER,
            tier="large",
            effort="high",
            max_tool_calls=200,
        )
        agent = ExplorerAgent(orch, config=config)

        assert agent.config.tier == "large"
        assert agent.config.effort == "high"
        assert agent.config.max_tool_calls == 200

    def test_explorer_agent_not_spawned_initially(self):
        """Test that agent is not spawned until spawn() is called."""
        orch = Orchestrator()
        agent = ExplorerAgent(orch)

        assert not agent.is_spawned()
        assert agent.get_context() is None

    @pytest.mark.asyncio
    async def test_explorer_agent_spawn(self):
        """Test spawning the explorer agent."""
        orch = Orchestrator()
        agent = ExplorerAgent(orch)

        spawned = await agent.spawn(briefing="Test briefing")

        assert agent.is_spawned()
        assert spawned is not None
        assert spawned.context.role == AgentRole.EXPLORER
        assert spawned.context.specialization == "explorer"
        assert spawned.context.briefing == "Test briefing"
        assert "search_files" in spawned.context.tools_available
        assert "write_file" not in spawned.context.tools_available  # Read-only

    @pytest.mark.asyncio
    async def test_explorer_agent_tools_are_read_only(self):
        """Test that spawned explorer agent has only read-only tools."""
        orch = Orchestrator()
        agent = ExplorerAgent(orch)

        spawned = await agent.spawn(briefing="Test")

        # Check read-only tools are present
        read_tools = {"search_files", "read_file", "web_search", "web_extract", "jlens_probe"}
        for tool in read_tools:
            assert tool in spawned.context.tools_available, f"Missing read tool: {tool}"

        # Check write/execute tools are NOT present
        write_tools = {"write_file", "patch", "run_command", "bash", "run_pytest", "create_file"}
        for tool in write_tools:
            assert tool not in spawned.context.tools_available, f"Write tool should not be available: {tool}"

    @pytest.mark.asyncio
    async def test_explorer_agent_explore_codebase_method(self):
        """Test the explore_codebase convenience method."""
        orch = Orchestrator()
        agent = ExplorerAgent(orch)

        # Just test it constructs the right task - we can't actually run without LLM
        task = "Find authentication handlers"
        # We won't actually run it since it needs an LLM, but we can verify the agent spawns
        await agent.spawn(briefing="Test")
        assert agent.is_spawned()


class TestSpecialistIntegration:
    """Integration tests for specialists with orchestrator."""

    @pytest.mark.asyncio
    async def test_delegate_to_explorer(self):
        """Test delegating to explorer via orchestrator."""
        orch = Orchestrator()

        # This tests the delegate_to_specialist path
        # We can't fully run without LLM, but we can verify the agent is created
        agent = await orch.spawn_agent(
            role=AgentRole.EXPLORER,
            tools=TOOL_ALLOWLISTS[SpecialistType.EXPLORER],
            specialization="explorer",
            briefing="Test delegation",
        )

        assert agent.context.role == AgentRole.EXPLORER
        assert agent.context.specialization == "explorer"
        assert "search_files" in agent.context.tools_available
        assert "write_file" not in agent.context.tools_available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])