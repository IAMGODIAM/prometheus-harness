"""Prometheus — MCP-First, J-Lens-Anchored Agentic Harness

A model-agnostic interpretability and safety runtime that exposes every capability
as an MCP server, uses the Jacobian Lens for alignment monitoring, and implements
architectural prompt-injection defense via dual-LLM (CaMeL pattern).
"""

__version__ = "0.1.0"
__all__ = [
    "jlens",
    "mcp_server",
    "harness",
    "memory",
    "observability",
    "authz",
    "config",
]
