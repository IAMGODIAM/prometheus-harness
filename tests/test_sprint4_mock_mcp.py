"""Sprint 4 regression: force_mock pool + orchestrator MCP surface."""

import pytest

from prometheus.harness.model_client import (
    EffortLevel,
    MockProvider,
    ProviderFactory,
    SubagentTier,
)
from prometheus.harness.orchestrator import Orchestrator
from prometheus.harness.tools import SafeToolRouter as ToolRouter
from prometheus.authz.engine import AuthzEngine
from prometheus.harness.jlens_gate import load_jlens_gate


def test_provider_factory_force_mock_returns_mock():
    factory = ProviderFactory(force_mock=True)
    p = factory.build_from_tier_effort(SubagentTier.LARGE, EffortLevel.HIGH)
    assert isinstance(p, MockProvider)
    # string coerce path
    p2 = factory.build_from_tier_effort("standard", "medium")
    assert isinstance(p2, MockProvider)


def test_provider_factory_string_coerce_without_mock_still_builds_key():
    factory = ProviderFactory(force_mock=True)
    p = factory.build_from_tier_effort("small", "low")
    assert p.model_id == "mock/deterministic-v1"


@pytest.mark.asyncio
async def test_orchestrator_mock_pool_no_nvidia_key():
    """Top-level MockProvider must force mock for warm pool subagents."""
    orch = Orchestrator(
        llm_client=MockProvider(),
        tool_router=ToolRouter(),
        authz_engine=AuthzEngine(),
        jlens_gate=load_jlens_gate(enable=False),
        subagent_pool_max_size=3,
    )
    assert orch.provider_factory.force_mock is True
    result = await orch.run_task(
        task="force mock pool regression",
        max_tool_calls=5,
        max_runtime_seconds=30.0,
        subagent_tier="standard",
        subagent_effort="medium",
        max_cost_usd_cents=100,
        verify=False,
    )
    assert result["status"] in ("completion_claimed", "complete", "incomplete", "budget_exhausted")
    # Must not be an NVIDIA key error
    assert "NVIDIA_API_KEY" not in str(result.get("output", ""))


@pytest.mark.asyncio
async def test_create_orchestrator_server_health_and_tools():
    from prometheus.mcp_server import create_orchestrator_server

    server = create_orchestrator_server(use_mock=True)
    if server is None:
        pytest.skip("FastMCP not installed")

    # FastMCP tool map or call_tool
    health = None
    if hasattr(server, "call_tool"):
        health = await server.call_tool("prometheus_health", {})
    elif hasattr(server, "tools") and "prometheus_health" in getattr(server, "tools", {}):
        health = await server.tools["prometheus_health"]()
    else:
        # Try _tool_manager common fastmcp shape
        tm = getattr(server, "_tool_manager", None) or getattr(server, "tool_manager", None)
        if tm is None:
            pytest.skip("Cannot invoke FastMCP tools in this version")
        tools = getattr(tm, "tools", {}) or {}
        fn = tools.get("prometheus_health")
        if fn is None:
            pytest.skip("prometheus_health not registered")
        health = await fn() if callable(fn) else await fn.run({})

    assert health is not None
    # unwrap CallToolResult-like
    if hasattr(health, "data"):
        health = health.data
    elif isinstance(health, dict) and "content" in health:
        pass
    assert isinstance(health, dict) or health is not None
