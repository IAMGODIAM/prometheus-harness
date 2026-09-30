"""Specialist agents for the Prometheus harness.

Specialist types:
- EXPLORER: Read-only exploration (search, read, analyze)
- REVIEWER: Analysis and review (code review, diff analysis, security scan)
- TESTER: Execution and testing (pytest, test runners, coverage)
- DESIGNER: Creation and design (write, patch, schema design, architecture)
"""

from prometheus.harness.specialists.config import (
    SpecialistConfig,
    SpecialistType,
    TOOL_ALLOWLISTS,
)
from prometheus.harness.specialists.explorer import ExplorerAgent
from prometheus.harness.specialists.reviewer import ReviewerAgent
from prometheus.harness.specialists.tester import TesterAgent
from prometheus.harness.specialists.designer import DesignerAgent

__all__ = [
    "SpecialistConfig",
    "SpecialistType",
    "TOOL_ALLOWLISTS",
    "ExplorerAgent",
    "ReviewerAgent",
    "TesterAgent",
    "DesignerAgent",
]