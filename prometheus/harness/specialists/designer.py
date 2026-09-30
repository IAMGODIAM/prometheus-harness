"""DesignerAgent — Creation specialist for schema design, architecture, and code writing."""

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


class DesignerAgent:
    """Creation specialist agent for writing code, designing schemas, and architecture.

    Capabilities:
    - File creation, editing, and patching
    - Schema design (database, API, configuration)
    - Architecture design and documentation
    - API design and specification
    - Database schema design
    - Command execution for scaffolding
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        config: SpecialistConfig | None = None,
    ):
        """Initialize the DesignerAgent.

        Args:
            orchestrator: The parent Orchestrator instance for spawning agents.
            config: Optional SpecialistConfig override. If None, uses default DESIGNER config.
        """
        self.orchestrator = orchestrator
        self.config = config or get_specialist_config(SpecialistType.DESIGNER)
        self._agent: AgentLoop | None = None

    @property
    def specialist_type(self) -> SpecialistType:
        return SpecialistType.DESIGNER

    @property
    def available_tools(self) -> list[str]:
        return self.config.tools

    async def spawn(self, briefing: str = "", parent_id: str | None = None) -> AgentLoop:
        """Spawn the designer agent with isolated context.

        Args:
            briefing: Detailed briefing/context for the design task.
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
            role=AgentRole.DESIGNER,
            tools=self.config.tools,
            permission_mode=PermissionMode.DEFAULT,
            parent_id=parent_id,
            max_tool_calls=self.config.max_tool_calls,
            max_runtime_seconds=self.config.max_runtime_seconds,
            subagent_tier=tier_map.get(self.config.tier, SubagentTier.LARGE),
            subagent_effort=effort_map.get(self.config.effort, EffortLevel.HIGH),
            specialization=self.config.specialist_type.value,
            briefing=briefing,
            use_pool=True,
        )

        logger.info(
            f"Spawned DesignerAgent: {self._agent.context.agent_id} "
            f"(tier={self.config.tier}, effort={self.config.effort}, "
            f"tools={len(self.config.tools)})"
        )

        return self._agent

    async def run(self, task: str, briefing: str = "") -> dict[str, Any]:
        """Run a design/creation task.

        Args:
            task: The design task description.
            briefing: Additional context/briefing for the agent.

        Returns:
            Task result with status, output, and metadata.
        """
        if self._agent is None:
            await self.spawn(briefing=briefing)

        assert self._agent is not None
        result = await self._agent.run(initial_message=task)

        logger.info(
            f"DesignerAgent {self._agent.context.agent_id} completed: "
            f"status={result.get('status')}, tool_calls={len(result.get('tool_calls', []))}"
        )

        return result

    async def design_schema(
        self,
        specification: str,
        schema_type: str = "database",
        briefing: str = "Design a schema based on the specification.",
    ) -> dict[str, Any]:
        """Design a schema (database, API, configuration, etc.).

        Args:
            specification: Description of the schema requirements.
            schema_type: Type of schema (database, api, config, graphql, etc.).
            briefing: Optional additional briefing context.

        Returns:
            Schema design with definitions, relationships, and documentation.
        """
        task = f"""
Design a {schema_type} schema for: {specification}

Use the available tools:
- schema_design: Design the schema structure
- write_file: Write schema definition files (SQL, JSON Schema, OpenAPI, etc.)
- patch: Modify existing schema files
- read_file: Read existing schemas for reference
- search_files: Find schema patterns and conventions
- run_command: Run schema validation/generation commands
- bash/terminal: Execute schema tools

Provide a schema design with:
1. Schema definition (SQL DDL, JSON Schema, OpenAPI spec, etc.)
2. Entity/relationship diagram description
3. Field definitions with types, constraints, and descriptions
4. Indexes, foreign keys, and constraints
5. Migration strategy (if database)
6. Versioning and evolution considerations
7. Example queries/usage
"""
        return await self.run(task, briefing)

    async def design_architecture(
        self,
        requirements: str,
        briefing: str = "Design a system architecture for the requirements.",
    ) -> dict[str, Any]:
        """Design a system architecture.

        Args:
            requirements: System requirements and constraints.
            briefing: Optional additional briefing context.

        Returns:
            Architecture design with components, interfaces, and diagrams.
        """
        task = f"""
Design a system architecture for: {requirements}

Use the available tools:
- architecture_design: Design the system architecture
- write_file: Write architecture documentation (ADR, diagrams as code, etc.)
- patch: Modify existing architecture docs
- read_file: Read existing architecture for context
- search_files: Find architectural patterns and conventions
- api_design: Design API interfaces
- database_design: Design data layer
- run_command: Run architecture validation tools
- bash/terminal: Execute architecture tools

Provide an architecture design with:
1. High-level architecture diagram (Mermaid, PlantUML, or ASCII)
2. Component breakdown with responsibilities
3. Service/API interfaces and contracts
4. Data flow diagrams
5. Technology stack recommendations
6. Scalability and reliability patterns
7. Security boundaries and trust zones
8. Deployment architecture
9. ADR (Architecture Decision Records) for key decisions
"""
        return await self.run(task, briefing)

    async def design_api(
        self,
        specification: str,
        briefing: str = "Design an API based on the specification.",
    ) -> dict[str, Any]:
        """Design an API (REST, GraphQL, gRPC, etc.).

        Args:
            specification: API requirements and use cases.
            briefing: Optional additional briefing context.

        Returns:
            API design with endpoints, schemas, and documentation.
        """
        task = f"""
Design an API for: {specification}

Use the available tools:
- api_design: Design the API structure
- write_file: Write OpenAPI/Swagger spec, GraphQL schema, Protobuf definitions
- patch: Modify existing API definitions
- read_file: Read existing APIs for consistency
- search_files: Find API patterns and conventions
- schema_design: Design request/response schemas
- run_command: Run API validation/linting tools
- bash/terminal: Execute API design tools

Provide an API design with:
1. OpenAPI 3.0 / GraphQL schema / Protobuf definitions
2. Endpoint definitions with methods, paths, parameters
3. Request/response schemas with examples
4. Authentication and authorization model
5. Error handling patterns
6. Rate limiting and pagination strategies
7. Versioning strategy
8. SDK/client library considerations
"""
        return await self.run(task, briefing)

    async def implement_feature(
        self,
        specification: str,
        target_files: list[str] | None = None,
        briefing: str = "Implement the feature based on the specification.",
    ) -> dict[str, Any]:
        """Implement a feature by writing code.

        Args:
            specification: Feature requirements and behavior.
            target_files: Optional list of files to create/modify.
            briefing: Optional additional briefing context.

        Returns:
            Implementation result with created/modified files.
        """
        target_info = f"\nTarget files: {', '.join(target_files)}" if target_files else ""
        task = f"""
Implement the following feature:{target_info}

Specification: {specification}

Use the available tools:
- write_file: Create new implementation files
- patch: Modify existing files
- create_file: Create new files
- edit_file: Edit existing files
- read_file: Read existing code for context and patterns
- search_files: Find implementation patterns and conventions
- run_command: Run build/lint/test commands to validate
- bash/terminal: Execute development commands
- terminal: Interactive terminal for development

Provide an implementation with:
1. Created/modified files with descriptions
2. Code following project conventions and patterns
3. Basic validation (lint, type-check, tests if applicable)
4. Integration points with existing code
5. Any configuration or dependency updates needed
"""
        return await self.run(task, briefing)

    async def create_project_structure(
        self,
        project_type: str,
        specification: str,
        briefing: str = "Create a new project structure.",
    ) -> dict[str, Any]:
        """Create a new project structure with scaffolding.

        Args:
            project_type: Type of project (python, node, go, rust, etc.).
            specification: Project requirements and structure.
            briefing: Optional additional briefing context.

        Returns:
            Project structure with scaffolded files.
        """
        task = f"""
Create a {project_type} project structure for: {specification}

Use the available tools:
- write_file: Create project files (config, source, tests, docs)
- create_file: Create new files and directories
- patch: Modify template files
- read_file: Read template/reference files
- search_files: Find project templates and conventions
- run_command: Run project initialization commands (poetry, npm, cargo, etc.)
- bash/terminal: Execute scaffolding commands
- schema_design: Design project configuration schemas
- architecture_design: Design initial project architecture

Provide:
1. Complete project directory structure
2. Configuration files (pyproject.toml, package.json, Cargo.toml, etc.)
3. Source code scaffolding with module structure
4. Test scaffolding
5. Documentation scaffolding (README, CONTRIBUTING, etc.)
6. CI/CD configuration
7. Development tooling configuration
"""
        return await self.run(task, briefing)

    def get_context(self) -> AgentContext | None:
        """Get the agent's context if spawned."""
        return self._agent.context if self._agent else None

    def is_spawned(self) -> bool:
        """Check if the agent has been spawned."""
        return self._agent is not None