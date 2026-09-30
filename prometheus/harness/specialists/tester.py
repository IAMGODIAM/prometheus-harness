"""TesterAgent — Execution specialist for test running, coverage, and validation."""

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


class TesterAgent:
    """Execution specialist agent for running tests, coverage, and validation.

    Capabilities:
    - Pytest and test runner execution
    - Coverage analysis and reporting
    - Command and terminal execution
    - Test fixture creation and management
    - Test modification and patching
    - Validation and verification
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        config: SpecialistConfig | None = None,
    ):
        """Initialize the TesterAgent.

        Args:
            orchestrator: The parent Orchestrator instance for spawning agents.
            config: Optional SpecialistConfig override. If None, uses default TESTER config.
        """
        self.orchestrator = orchestrator
        self.config = config or get_specialist_config(SpecialistType.TESTER)
        self._agent: AgentLoop | None = None

    @property
    def specialist_type(self) -> SpecialistType:
        return SpecialistType.TESTER

    @property
    def available_tools(self) -> list[str]:
        return self.config.tools

    async def spawn(self, briefing: str = "", parent_id: str | None = None) -> AgentLoop:
        """Spawn the tester agent with isolated context.

        Args:
            briefing: Detailed briefing/context for the testing task.
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
            role=AgentRole.TESTER,
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
            f"Spawned TesterAgent: {self._agent.context.agent_id} "
            f"(tier={self.config.tier}, effort={self.config.effort}, "
            f"tools={len(self.config.tools)})"
        )

        return self._agent

    async def run(self, task: str, briefing: str = "") -> dict[str, Any]:
        """Run a testing/validation task.

        Args:
            task: The testing task description.
            briefing: Additional context/briefing for the agent.

        Returns:
            Task result with status, output, and metadata.
        """
        if self._agent is None:
            await self.spawn(briefing=briefing)

        assert self._agent is not None
        result = await self._agent.run(initial_message=task)

        logger.info(
            f"TesterAgent {self._agent.context.agent_id} completed: "
            f"status={result.get('status')}, tool_calls={len(result.get('tool_calls', []))}"
        )

        return result

    async def run_tests(
        self,
        target: str = ".",
        test_args: str = "",
        briefing: str = "Run the test suite and report results.",
    ) -> dict[str, Any]:
        """Run tests for the target.

        Args:
            target: Path to test directory or module (default: current directory).
            test_args: Additional pytest/test runner arguments.
            briefing: Optional additional briefing context.

        Returns:
            Test execution result with pass/fail status and details.
        """
        task = f"""
Run tests for: {target}

Additional test arguments: {test_args}

Use the available tools:
- run_pytest: Run pytest with specified arguments
- run_tests: Run tests using the configured test runner
- run_coverage: Run tests with coverage reporting
- run_command: Execute arbitrary test commands
- bash/terminal: Run shell commands for test setup/teardown
- read_file: Read test files for understanding test structure
- search_files: Find test files and patterns
- write_file: Create test fixtures if needed
- patch: Modify test files if needed

Provide a test report with:
1. Test execution summary (passed, failed, skipped, errors)
2. Failed test details with tracebacks
3. Coverage report (if run_coverage used)
4. Performance/timing information
5. Flaky test identification
6. Recommendations for test improvements
"""
        return await self.run(task, briefing)

    async def run_with_coverage(
        self,
        target: str = ".",
        coverage_args: str = "--cov=src --cov-report=term-missing",
        briefing: str = "Run tests with coverage analysis.",
    ) -> dict[str, Any]:
        """Run tests with coverage reporting.

        Args:
            target: Path to test directory or module.
            coverage_args: Coverage arguments for pytest-cov.
            briefing: Optional additional briefing context.

        Returns:
            Coverage report with line/branch coverage details.
        """
        task = f"""
Run tests with coverage for: {target}

Coverage arguments: {coverage_args}

Use the available tools:
- run_coverage: Run tests with coverage reporting
- run_pytest: Run pytest with coverage flags
- run_command: Execute coverage commands directly
- bash/terminal: Run shell commands for coverage
- read_file: Read source files to understand coverage gaps
- search_files: Find untested code patterns

Provide a coverage report with:
1. Overall coverage percentage (lines, branches)
2. Per-file coverage breakdown
3. Missing coverage highlights (uncovered lines)
4. Critical untested paths
5. Coverage trend analysis (if historical data available)
6. Recommendations for improving coverage
"""
        return await self.run(task, briefing)

    async def validate_fix(
        self,
        test_target: str,
        expected_behavior: str,
        briefing: str = "Validate that a fix works correctly.",
    ) -> dict[str, Any]:
        """Validate that a fix works by running relevant tests.

        Args:
            test_target: Specific test file or pattern to run.
            expected_behavior: Description of expected behavior after fix.
            briefing: Optional additional briefing context.

        Returns:
            Validation result confirming fix or identifying regressions.
        """
        task = f"""
Validate the fix by running tests for: {test_target}

Expected behavior: {expected_behavior}

Use the available tools:
- run_pytest: Run specific test files/patterns
- run_tests: Run targeted test suite
- run_command: Execute validation commands
- bash/terminal: Run shell validation scripts
- read_file: Read test and source files to understand the fix
- search_files: Find related tests

Provide a validation report with:
1. Test results for the targeted area
2. Whether expected behavior is confirmed
3. Any regressions detected
4. Edge cases tested
5. Confidence level in the fix
"""
        return await self.run(task, briefing)

    async def create_test_fixtures(
        self,
        specification: str,
        briefing: str = "Create test fixtures and test cases.",
    ) -> dict[str, Any]:
        """Create test fixtures and test cases based on specification.

        Args:
            specification: Description of what fixtures/tests to create.
            briefing: Optional additional briefing context.

        Returns:
            Result with created fixtures and test cases.
        """
        task = f"""
Create test fixtures and test cases for: {specification}

Use the available tools:
- write_file: Create test fixture files
- patch: Modify existing test files
- read_file: Read existing tests for patterns
- search_files: Find test patterns and conventions
- run_command: Run commands to validate fixtures

Provide:
1. Created fixture files with descriptions
2. Created test cases with descriptions
3. Validation that fixtures work correctly
4. Integration with existing test suite
"""
        return await self.run(task, briefing)

    def get_context(self) -> AgentContext | None:
        """Get the agent's context if spawned."""
        return self._agent.context if self._agent else None

    def is_spawned(self) -> bool:
        """Check if the agent has been spawned."""
        return self._agent is not None