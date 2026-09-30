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


class TestSpecialistConfigs:
    """Tests for specialist configurations and tool allowlists."""

    def test_explorer_config_defaults(self):
        """Test ExplorerAgent default configuration."""
        config = get_specialist_config(SpecialistType.EXPLORER)
        assert config.specialist_type == SpecialistType.EXPLORER
        assert config.tier == "standard"
        assert config.effort == "medium"
        assert config.max_tool_calls == 50
        assert config.max_runtime_seconds == 300.0
        assert "web_search" in config.tools
        assert "web_extract" in config.tools
        assert "search_files" in config.tools
        assert "read_file" in config.tools
        assert "jlens_probe" in config.tools

    def test_reviewer_config_defaults(self):
        """Test ReviewerAgent default configuration."""
        config = get_specialist_config(SpecialistType.REVIEWER)
        assert config.specialist_type == SpecialistType.REVIEWER
        assert config.tier == "standard"
        assert config.effort == "medium"
        assert "read_file" in config.tools
        assert "search_files" in config.tools
        assert "github_code_review" in config.tools
        assert "diff_analysis" in config.tools
        assert "security_scan" in config.tools
        assert "analyze_code" in config.tools
        assert "jlens_probe" in config.tools

    def test_tester_config_defaults(self):
        """Test TesterAgent default configuration."""
        config = get_specialist_config(SpecialistType.TESTER)
        assert config.specialist_type == SpecialistType.TESTER
        assert config.tier == "standard"
        assert config.effort == "medium"
        assert "run_pytest" in config.tools
        assert "run_tests" in config.tools
        assert "run_coverage" in config.tools
        assert "run_command" in config.tools
        assert "bash" in config.tools
        assert "terminal" in config.tools
        assert "read_file" in config.tools
        assert "write_file" in config.tools
        assert "patch" in config.tools

    def test_designer_config_defaults(self):
        """Test DesignerAgent default configuration."""
        config = get_specialist_config(SpecialistType.DESIGNER)
        assert config.specialist_type == SpecialistType.DESIGNER
        assert config.tier == "standard"
        assert config.effort == "medium"
        assert "write_file" in config.tools
        assert "patch" in config.tools
        assert "create_file" in config.tools
        assert "edit_file" in config.tools
        assert "schema_design" in config.tools
        assert "architecture_design" in config.tools
        assert "api_design" in config.tools
        assert "database_design" in config.tools
        assert "read_file" in config.tools
        assert "run_command" in config.tools

    def test_tool_allowlists_exist(self):
        """Test that all specialist types have tool allowlists."""
        assert SpecialistType.EXPLORER in TOOL_ALLOWLISTS
        assert SpecialistType.REVIEWER in TOOL_ALLOWLISTS
        assert SpecialistType.TESTER in TOOL_ALLOWLISTS
        assert SpecialistType.DESIGNER in TOOL_ALLOWLISTS

    def test_config_overrides(self):
        """Test that config overrides work correctly."""
        config = get_specialist_config(
            SpecialistType.REVIEWER,
            tier="large",
            effort="high",
            max_tool_calls=100,
            max_runtime_seconds=600.0,
        )
        assert config.tier == "large"
        assert config.effort == "high"
        assert config.max_tool_calls == 100
        assert config.max_runtime_seconds == 600.0


class TestReviewerAgent:
    """Tests for ReviewerAgent."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orch = MagicMock(spec=Orchestrator)
        orch.spawn_agent = AsyncMock()
        return orch

    @pytest.fixture
    def reviewer_agent(self, mock_orchestrator):
        """Create a ReviewerAgent with mock orchestrator."""
        return ReviewerAgent(orchestrator=mock_orchestrator)

    @pytest.mark.asyncio
    async def test_reviewer_initialization(self, reviewer_agent):
        """Test ReviewerAgent initializes correctly."""
        assert reviewer_agent.specialist_type == SpecialistType.REVIEWER
        assert "read_file" in reviewer_agent.available_tools
        assert "security_scan" in reviewer_agent.available_tools
        assert "diff_analysis" in reviewer_agent.available_tools

    @pytest.mark.asyncio
    async def test_reviewer_spawn(self, reviewer_agent, mock_orchestrator):
        """Test spawning a reviewer agent."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-reviewer-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        agent = await reviewer_agent.spawn(briefing="Test review", parent_id="parent-123")

        assert agent == mock_agent
        mock_orchestrator.spawn_agent.assert_called_once()
        call_kwargs = mock_orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["role"] == AgentRole.REVIEWER
        assert call_kwargs["briefing"] == "Test review"
        assert call_kwargs["parent_id"] == "parent-123"
        assert "security_scan" in call_kwargs["tools"]

    @pytest.mark.asyncio
    async def test_reviewer_run(self, reviewer_agent, mock_orchestrator):
        """Test running a review task."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Review complete"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-reviewer-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await reviewer_agent.run("Review this code", briefing="Security review")

        assert result["status"] == "complete"
        mock_agent.run.assert_called_once_with(initial_message="Review this code")

    @pytest.mark.asyncio
    async def test_review_code(self, reviewer_agent, mock_orchestrator):
        """Test review_code method."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Code reviewed"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-reviewer-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await reviewer_agent.review_code("src/auth.py")

        assert result["status"] == "complete"
        # Check the task was passed correctly
        call_args = mock_agent.run.call_args
        # run is called with initial_message=task
        assert "initial_message" in call_args.kwargs
        assert "src/auth.py" in call_args.kwargs["initial_message"]

    @pytest.mark.asyncio
    async def test_analyze_diff(self, reviewer_agent, mock_orchestrator):
        """Test analyze_diff method."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Diff analyzed"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-reviewer-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        diff = "@@ -1,3 +1,4 @@\n+new line\n existing\n lines\n"
        result = await reviewer_agent.analyze_diff(diff)

        assert result["status"] == "complete"
        call_args = mock_agent.run.call_args
        assert "initial_message" in call_args.kwargs
        assert diff in call_args.kwargs["initial_message"]

    @pytest.mark.asyncio
    async def test_security_audit(self, reviewer_agent, mock_orchestrator):
        """Test security_audit method."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Audit complete"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-reviewer-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await reviewer_agent.security_audit("src/")

        assert result["status"] == "complete"
        call_args = mock_agent.run.call_args
        assert "initial_message" in call_args.kwargs
        assert "security audit" in call_args.kwargs["initial_message"].lower()


class TestTesterAgent:
    """Tests for TesterAgent."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orch = MagicMock(spec=Orchestrator)
        orch.spawn_agent = AsyncMock()
        return orch

    @pytest.fixture
    def tester_agent(self, mock_orchestrator):
        """Create a TesterAgent with mock orchestrator."""
        return TesterAgent(orchestrator=mock_orchestrator)

    @pytest.mark.asyncio
    async def test_tester_initialization(self, tester_agent):
        """Test TesterAgent initializes correctly."""
        assert tester_agent.specialist_type == SpecialistType.TESTER
        assert "run_pytest" in tester_agent.available_tools
        assert "run_coverage" in tester_agent.available_tools
        assert "bash" in tester_agent.available_tools
        assert "write_file" in tester_agent.available_tools
        assert "patch" in tester_agent.available_tools

    @pytest.mark.asyncio
    async def test_tester_spawn(self, tester_agent, mock_orchestrator):
        """Test spawning a tester agent."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-tester-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        agent = await tester_agent.spawn(briefing="Run tests", parent_id="parent-123")

        assert agent == mock_agent
        call_kwargs = mock_orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["role"] == AgentRole.TESTER
        assert "run_pytest" in call_kwargs["tools"]
        assert "run_coverage" in call_kwargs["tools"]

    @pytest.mark.asyncio
    async def test_tester_run(self, tester_agent, mock_orchestrator):
        """Test running a test task."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Tests passed"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-tester-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await tester_agent.run("Run unit tests")

        assert result["status"] == "complete"
        mock_agent.run.assert_called_once_with(initial_message="Run unit tests")

    @pytest.mark.asyncio
    async def test_run_tests(self, tester_agent, mock_orchestrator):
        """Test run_tests method."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Tests completed"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-tester-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await tester_agent.run_tests("tests/", test_args="-v -k unit")

        assert result["status"] == "complete"
        call_args = mock_agent.run.call_args
        assert "initial_message" in call_args.kwargs
        assert "tests/" in call_args.kwargs["initial_message"]
        assert "-v -k unit" in call_args.kwargs["initial_message"]

    @pytest.mark.asyncio
    async def test_run_with_coverage(self, tester_agent, mock_orchestrator):
        """Test run_with_coverage method."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Coverage: 85%"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-tester-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await tester_agent.run_with_coverage("src/", coverage_args="--cov=src")

        assert result["status"] == "complete"
        call_args = mock_agent.run.call_args
        assert "initial_message" in call_args.kwargs
        assert "coverage" in call_args.kwargs["initial_message"].lower()
        assert "--cov=src" in call_args.kwargs["initial_message"]

    @pytest.mark.asyncio
    async def test_validate_fix(self, tester_agent, mock_orchestrator):
        """Test validate_fix method."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Fix validated"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-tester-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await tester_agent.validate_fix("tests/test_auth.py", "Auth should reject invalid tokens")

        assert result["status"] == "complete"
        call_args = mock_agent.run.call_args
        assert "initial_message" in call_args.kwargs
        assert "test_auth.py" in call_args.kwargs["initial_message"]
        assert "invalid tokens" in call_args.kwargs["initial_message"]

    @pytest.mark.asyncio
    async def test_create_test_fixtures(self, tester_agent, mock_orchestrator):
        """Test create_test_fixtures method."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Fixtures created"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-tester-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await tester_agent.create_test_fixtures("User authentication fixtures")

        assert result["status"] == "complete"
        call_args = mock_agent.run.call_args
        assert "initial_message" in call_args.kwargs
        assert "fixtures" in call_args.kwargs["initial_message"].lower()


class TestDesignerAgent:
    """Tests for DesignerAgent."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orch = MagicMock(spec=Orchestrator)
        orch.spawn_agent = AsyncMock()
        return orch

    @pytest.fixture
    def designer_agent(self, mock_orchestrator):
        """Create a DesignerAgent with mock orchestrator."""
        return DesignerAgent(orchestrator=mock_orchestrator)

    @pytest.mark.asyncio
    async def test_designer_initialization(self, designer_agent):
        """Test DesignerAgent initializes correctly."""
        assert designer_agent.specialist_type == SpecialistType.DESIGNER
        assert "write_file" in designer_agent.available_tools
        assert "patch" in designer_agent.available_tools
        assert "schema_design" in designer_agent.available_tools
        assert "architecture_design" in designer_agent.available_tools
        assert "api_design" in designer_agent.available_tools

    @pytest.mark.asyncio
    async def test_designer_spawn(self, designer_agent, mock_orchestrator):
        """Test spawning a designer agent."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-designer-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        agent = await designer_agent.spawn(briefing="Design API", parent_id="parent-123")

        assert agent == mock_agent
        call_kwargs = mock_orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["role"] == AgentRole.DESIGNER
        assert "write_file" in call_kwargs["tools"]
        assert "schema_design" in call_kwargs["tools"]
        assert "architecture_design" in call_kwargs["tools"]

    @pytest.mark.asyncio
    async def test_designer_run(self, designer_agent, mock_orchestrator):
        """Test running a design task."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Design complete"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-designer-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await designer_agent.run("Design a schema")

        assert result["status"] == "complete"
        mock_agent.run.assert_called_once_with(initial_message="Design a schema")

    @pytest.mark.asyncio
    async def test_design_schema(self, designer_agent, mock_orchestrator):
        """Test design_schema method."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Schema designed"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-designer-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await designer_agent.design_schema("User table with email, name, created_at", "database")

        assert result["status"] == "complete"
        call_args = mock_agent.run.call_args
        assert "initial_message" in call_args.kwargs
        assert "database" in call_args.kwargs["initial_message"]
        assert "User table" in call_args.kwargs["initial_message"]

    @pytest.mark.asyncio
    async def test_design_architecture(self, designer_agent, mock_orchestrator):
        """Test design_architecture method."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Architecture designed"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-designer-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await designer_agent.design_architecture("Microservices with event-driven communication")

        assert result["status"] == "complete"
        call_args = mock_agent.run.call_args
        assert "initial_message" in call_args.kwargs
        assert "architecture" in call_args.kwargs["initial_message"].lower()
        assert "microservices" in call_args.kwargs["initial_message"].lower()

    @pytest.mark.asyncio
    async def test_design_api(self, designer_agent, mock_orchestrator):
        """Test design_api method."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "API designed"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-designer-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await designer_agent.design_api("REST API for user management with CRUD operations")

        assert result["status"] == "complete"
        call_args = mock_agent.run.call_args
        assert "initial_message" in call_args.kwargs
        assert "api" in call_args.kwargs["initial_message"].lower()
        assert "CRUD" in call_args.kwargs["initial_message"]

    @pytest.mark.asyncio
    async def test_implement_feature(self, designer_agent, mock_orchestrator):
        """Test implement_feature method."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Feature implemented"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-designer-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await designer_agent.implement_feature(
            "Add user authentication middleware",
            target_files=["src/middleware/auth.py", "tests/test_auth_middleware.py"]
        )

        assert result["status"] == "complete"
        call_args = mock_agent.run.call_args
        assert "initial_message" in call_args.kwargs
        assert "authentication middleware" in call_args.kwargs["initial_message"]
        assert "src/middleware/auth.py" in call_args.kwargs["initial_message"]

    @pytest.mark.asyncio
    async def test_create_project_structure(self, designer_agent, mock_orchestrator):
        """Test create_project_structure method."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Project created"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "test-designer-123"
        mock_orchestrator.spawn_agent.return_value = mock_agent

        result = await designer_agent.create_project_structure("python", "FastAPI microservice with auth")

        assert result["status"] == "complete"
        call_args = mock_agent.run.call_args
        assert "initial_message" in call_args.kwargs
        assert "python" in call_args.kwargs["initial_message"].lower()
        assert "fastapi" in call_args.kwargs["initial_message"].lower()


class TestOrchestratorDelegateToSpecialist:
    """Tests for Orchestrator.delegate_to_specialist with specialist tool allowlists."""

    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator with mocks."""
        orch = Orchestrator()
        # Mock the subagent pool get_or_create to return a new agent
        orch.subagent_pool.get_or_create = AsyncMock()
        orch.provider_factory = MagicMock()
        orch.provider_factory.build_from_tier_effort = MagicMock()
        return orch

    @pytest.mark.asyncio
    async def test_delegate_to_reviewer_uses_allowlist(self, orchestrator):
        """Test that delegating to REVIEWER uses the reviewer tool allowlist."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Reviewed"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "reviewer-123"
        orchestrator.spawn_agent = AsyncMock(return_value=mock_agent)

        result = await orchestrator.delegate_to_specialist(
            task="Review this PR",
            specialist=AgentRole.REVIEWER,
            briefing="Security review of auth changes",
            tools=None,  # Should use default allowlist
        )

        # Verify spawn_agent was called with reviewer tools
        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["role"] == AgentRole.REVIEWER
        assert "github_code_review" in call_kwargs["tools"]
        assert "security_scan" in call_kwargs["tools"]
        assert "diff_analysis" in call_kwargs["tools"]

    @pytest.mark.asyncio
    async def test_delegate_to_tester_uses_allowlist(self, orchestrator):
        """Test that delegating to TESTER uses the tester tool allowlist."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Tested"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "tester-123"
        orchestrator.spawn_agent = AsyncMock(return_value=mock_agent)

        result = await orchestrator.delegate_to_specialist(
            task="Run tests",
            specialist=AgentRole.TESTER,
            briefing="Run full test suite",
            tools=None,
        )

        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["role"] == AgentRole.TESTER
        assert "run_pytest" in call_kwargs["tools"]
        assert "run_coverage" in call_kwargs["tools"]
        assert "bash" in call_kwargs["tools"]

    @pytest.mark.asyncio
    async def test_delegate_to_designer_uses_allowlist(self, orchestrator):
        """Test that delegating to DESIGNER uses the designer tool allowlist."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Designed"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "designer-123"
        orchestrator.spawn_agent = AsyncMock(return_value=mock_agent)

        result = await orchestrator.delegate_to_specialist(
            task="Design a schema",
            specialist=AgentRole.DESIGNER,
            briefing="Database schema for users",
            tools=None,
        )

        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["role"] == AgentRole.DESIGNER
        assert "schema_design" in call_kwargs["tools"]
        assert "architecture_design" in call_kwargs["tools"]
        assert "api_design" in call_kwargs["tools"]
        assert "write_file" in call_kwargs["tools"]

    @pytest.mark.asyncio
    async def test_delegate_to_explorer_uses_allowlist(self, orchestrator):
        """Test that delegating to EXPLORER uses the explorer tool allowlist."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Explored"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "explorer-123"
        orchestrator.spawn_agent = AsyncMock(return_value=mock_agent)

        result = await orchestrator.delegate_to_specialist(
            task="Explore codebase",
            specialist=AgentRole.EXPLORER,
            briefing="Find auth handlers",
            tools=None,
        )

        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["role"] == AgentRole.EXPLORER
        assert "web_search" in call_kwargs["tools"]
        assert "jlens_probe" in call_kwargs["tools"]
        assert "sovereign_search" in call_kwargs["tools"]

    @pytest.mark.asyncio
    async def test_delegate_with_custom_tools_overrides_allowlist(self, orchestrator):
        """Test that providing custom tools overrides the default allowlist."""
        mock_agent = MagicMock(spec=AgentLoop)
        mock_agent.run = AsyncMock(return_value={"status": "complete", "output": "Done"})
        mock_agent.context = MagicMock(spec=AgentContext)
        mock_agent.context.agent_id = "custom-123"
        orchestrator.spawn_agent = AsyncMock(return_value=mock_agent)

        custom_tools = ["custom_tool_1", "custom_tool_2"]
        result = await orchestrator.delegate_to_specialist(
            task="Custom task",
            specialist=AgentRole.REVIEWER,
            briefing="Custom tools test",
            tools=custom_tools,
        )

        call_kwargs = orchestrator.spawn_agent.call_args.kwargs
        assert call_kwargs["tools"] == custom_tools


class TestSpecialistIntegration:
    """Integration tests for specialist agents."""

    @pytest.mark.asyncio
    async def test_all_specialists_can_be_imported(self):
        """Test that all specialist classes can be imported."""
        from prometheus.harness.specialists import (
            ExplorerAgent,
            ReviewerAgent,
            TesterAgent,
            DesignerAgent,
        )
        assert ExplorerAgent is not None
        assert ReviewerAgent is not None
        assert TesterAgent is not None
        assert DesignerAgent is not None

    @pytest.mark.asyncio
    async def test_specialist_types_enum_complete(self):
        """Test that SpecialistType enum has all four types."""
        assert SpecialistType.EXPLORER == "explorer"
        assert SpecialistType.REVIEWER == "reviewer"
        assert SpecialistType.TESTER == "tester"
        assert SpecialistType.DESIGNER == "designer"
        assert len(SpecialistType) == 4

    @pytest.mark.asyncio
    async def test_agent_role_enum_has_specialists(self):
        """Test that AgentRole enum has all specialist roles."""
        assert AgentRole.EXPLORER == "explorer"
        assert AgentRole.REVIEWER == "reviewer"
        assert AgentRole.TESTER == "tester"
        assert AgentRole.DESIGNER == "designer"