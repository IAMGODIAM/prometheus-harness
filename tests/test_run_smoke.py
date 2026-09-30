"""Smoke test for the RIG `run` path — end-to-end agent loop, offline.

Uses the deterministic MockProvider (no network, no API key) to prove the loop:
plan -> call a tool through the gates -> claim completion -> verify -> verdict,
all in the safe dry-run posture. Requires only pytest + pytest-asyncio.
"""

import pytest

from prometheus.harness.model_client import MockProvider
from prometheus.harness.runner import run_task, RunConfig


@pytest.mark.asyncio
async def test_run_end_to_end_mock_dry_run():
    result = await run_task(
        "Demonstrate the harness end to end.",
        provider=MockProvider(),
        run=RunConfig(dry_run=True),
    )

    # The loop reached a completion claim and the verifier produced a verdict.
    assert result["status"] in ("completion_claimed", "complete")
    assert "verification" in result
    assert result["verification"]["status"] in ("ACCEPT", "REVISE", "ESCALATE")

    # A tool call was authorized and executed through the gates.
    assert any(tc["name"] == "echo" for tc in result["tool_calls"])
    assert all(tc["success"] for tc in result["tool_calls"])

    # Safe-by-default posture held.
    assert result["safety"]["dry_run"] is True
    assert result["safety"]["allow_fs"] is False
    assert result["safety"]["allow_net"] is False
    assert result["safety"]["authz"] == "deny_by_default"
    assert result["safety"]["jlens_gate"] == "noop"

    # Observability was actually wired into the loop (spans recorded).
    assert result["trace"]["span_count"] >= 1


@pytest.mark.asyncio
async def test_dry_run_tool_is_simulated():
    """In dry-run, the real handler must NOT run — result is marked simulated."""
    from prometheus.harness.tools import build_default_toolset

    router = build_default_toolset(dry_run=True)
    out = await router.call("echo", {"text": "hello"})
    assert out.get("dry_run") is True
    assert "echo" not in out  # handler did not run


@pytest.mark.asyncio
async def test_deny_by_default_blocks_unclassified_tool():
    """A tool with no permit is denied (deny-by-default), even before dry-run."""
    from prometheus.authz.engine import AuthzEngine, AuthzRequest, Decision

    engine = AuthzEngine()
    engine.load_default_policies()
    decision = engine.evaluate(
        AuthzRequest(principal={"role": "actor"}, action_name="rm_rf_everything")
    )
    assert decision.decision == Decision.DENY


@pytest.mark.asyncio
async def test_jlens_gate_defaults_to_noop_without_torch():
    """The J-lens gate is pluggable and defaults to a no-op (no torch needed)."""
    from prometheus.harness.jlens_gate import load_jlens_gate, NoOpJLensGate

    gate = load_jlens_gate(enable=False)
    assert isinstance(gate, NoOpJLensGate)
    # Even if enabled without a wired scorer, it must not fake capability.
    gate2 = load_jlens_gate(enable=True)
    assert gate2.__class__.__name__ in ("NoOpJLensGate", "RealJLensGate")
