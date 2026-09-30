"""Specialist configuration and tool allowlists."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SpecialistType(str, Enum):
    """Specialist agent types."""

    EXPLORER = "explorer"      # Read-only: search, read, analyze
    REVIEWER = "reviewer"      # Analysis: code review, diff analysis, security scan
    TESTER = "tester"          # Execution: pytest, test runners, coverage
    DESIGNER = "designer"      # Creation: write, patch, schema design, architecture


@dataclass(frozen=True)
class SpecialistConfig:
    """Configuration for a specialist agent.

    Defines the tool allowlist, tier, effort, and capabilities per specialist type.
    """

    specialist_type: SpecialistType
    tools: list[str]
    description: str
    tier: str = "standard"
    effort: str = "medium"
    max_tool_calls: int = 50
    max_runtime_seconds: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)


# Tool allowlists per specialist type
# These map to the tools available in the SafeToolRouter and other tool registries
TOOL_ALLOWLISTS: dict[SpecialistType, list[str]] = {
    SpecialistType.EXPLORER: [
        # Read-only search and discovery tools
        "web_search",
        "web_extract",
        "search_files",
        "read_file",
        "grep",
        "glob",
        "list_files",
        "analyze_code",
        "analyze_architecture",
        "jlens_probe",
        "jlens_decompose",
        "jlens_watchlist_scores",
        "sovereign_search",
        "sovereign_knowledge_search",
        "skill_view",
        "skills_list",
    ],
    SpecialistType.REVIEWER: [
        # Analysis and review tools
        "read_file",
        "search_files",
        "github_code_review",
        "diff_analysis",
        "security_scan",
        "analyze_code",
        "analyze_dependencies",
        "static_analysis",
        "jlens_probe",
        "jlens_decompose",
        "jlens_watchlist_scores",
    ],
    SpecialistType.TESTER: [
        # Execution and testing tools
        "run_pytest",
        "run_tests",
        "run_coverage",
        "run_command",
        "bash",
        "terminal",
        "read_file",
        "search_files",
        "write_file",  # For test fixtures
        "patch",       # For test modifications
    ],
    SpecialistType.DESIGNER: [
        # Creation and design tools
        "write_file",
        "patch",
        "create_file",
        "edit_file",
        "schema_design",
        "architecture_design",
        "api_design",
        "database_design",
        "read_file",
        "search_files",
        "run_command",
        "terminal",
    ],
}


def get_specialist_config(
    specialist_type: SpecialistType,
    tier: str | None = None,
    effort: str | None = None,
    max_tool_calls: int | None = None,
    max_runtime_seconds: float | None = None,
) -> SpecialistConfig:
    """Get a SpecialistConfig for the given type with optional overrides."""
    tools = TOOL_ALLOWLISTS.get(specialist_type, [])

    descriptions = {
        SpecialistType.EXPLORER: "Read-only exploration agent for search, discovery, and analysis",
        SpecialistType.REVIEWER: "Analysis agent for code review, security scanning, and diff analysis",
        SpecialistType.TESTER: "Execution agent for running tests, coverage, and validation",
        SpecialistType.DESIGNER: "Creation agent for writing code, designing schemas, and architecture",
    }

    return SpecialistConfig(
        specialist_type=specialist_type,
        tools=tools,
        description=descriptions[specialist_type],
        tier=tier or "standard",
        effort=effort or "medium",
        max_tool_calls=max_tool_calls or 50,
        max_runtime_seconds=max_runtime_seconds or 300.0,
    )