"""Tests for the Agentic Harness modules."""

import asyncio
import pytest

from prometheus.harness.orchestrator import (
    Orchestrator,
    AgentLoop,
    AgentContext,
    AgentRole,
    HookType,
    HookResult,
    PermissionMode,
)
from prometheus.harness.dual_llm import (
    DualLLMGate,
    TrustLevel,
    TaintLabel,
    ContentBlock,
    SessionTaintState,
)
from prometheus.harness.verifier import (
    AsyncVerifier,
    CompletionClaim,
    Verdict,
    VerdictStatus,
    VerificationFailure,
)
from prometheus.harness.tool_parser import (
    ToolCallParser,
    ToolCall,
    ToolCallFormat,
)


class TestOrchestrator:
    """Tests for the Orchestrator."""

    @pytest.mark.asyncio
    async def test_spawn_agent(self):
        """Test spawning an agent with isolated context."""
        orch = Orchestrator()
        agent = await orch.spawn_agent(role=AgentRole.ACTOR, tools=["jlens_probe"])
        assert agent.context.role == AgentRole.ACTOR
        assert "jlens_probe" in agent.context.tools_available
        assert agent.context.agent_id in orch.agents

    def test_kill_switch(self):
        """Test global kill switch stops all agents."""
        orch = Orchestrator()
        orch.kill()
        assert orch._kill_switch is True

    def test_budget_exhaustion(self):
        """Test agent budget tracking."""
        context = AgentContext(
            max_tool_calls=3,
            max_runtime_seconds=300.0,
        )
        context.tool_calls_made = 3
        assert context.budget_exhausted is True

    def test_hook_registration(self):
        """Test hook registration and execution."""
        orch = Orchestrator()

        async def test_hook(data):
            return HookResult(allow=True)

        orch.register_global_hook(HookType.PRE_TOOL_USE, test_hook)
        assert len(orch._global_hooks[HookType.PRE_TOOL_USE]) == 1


class TestDualLLMGate:
    """Tests for the Dual-LLM security gate."""

    def test_taint_label_trifecta(self):
        """Test Lethal Trifecta detection."""
        # Not a trifecta
        label = TaintLabel(reads_private_data=True, sees_untrusted_content=True, can_exfiltrate=False)
        assert not label.is_lethal_trifecta

        # Is a trifecta
        label = TaintLabel(reads_private_data=True, sees_untrusted_content=True, can_exfiltrate=True)
        assert label.is_lethal_trifecta

    def test_taint_label_count(self):
        """Test trifecta leg counting."""
        label = TaintLabel(reads_private_data=True, sees_untrusted_content=False, can_exfiltrate=True)
        assert label.trifecta_count == 2

    def test_session_taint_accumulation(self):
        """Test that session taint accumulates correctly."""
        state = SessionTaintState()

        state.has_private_data = True
        assert not state.is_compromised

        state.has_untrusted_content = True
        assert not state.is_compromised

        state.has_exfiltration_capability = True
        assert state.is_compromised

    @pytest.mark.asyncio
    async def test_trifecta_blocking(self):
        """Test that trifecta violations are blocked in strict mode."""
        gate = DualLLMGate(strict_mode=True)
        gate.declare_tool_taint("send_message", TaintLabel(can_exfiltrate=True))

        # Simulate compromised session
        gate.session_state.has_private_data = True
        gate.session_state.has_untrusted_content = True
        gate.session_state.has_exfiltration_capability = True

        result = await gate.check({
            "tool_name": "send_message",
            "arguments": {},
        })
        assert not result.allow
        assert "Lethal Trifecta" in result.reason

    def test_symbolic_ref_allocation(self):
        """Test symbolic reference allocation."""
        state = SessionTaintState()
        ref1 = state.allocate_symbolic_ref()
        ref2 = state.allocate_symbolic_ref()
        assert ref1 == "$VAR1"
        assert ref2 == "$VAR2"

    def test_session_reset(self):
        """Test session reset clears taint."""
        gate = DualLLMGate()
        gate.session_state.has_private_data = True
        gate.session_state.has_untrusted_content = True
        gate.reset_session()
        assert not gate.session_state.has_private_data
        assert not gate.session_state.has_untrusted_content

    def test_tool_taint_declaration(self):
        """Test declaring tool taints."""
        gate = DualLLMGate()
        gate.declare_tool_taint("read_db", TaintLabel(reads_private_data=True))
        taint = gate.get_tool_taint("read_db")
        assert taint.reads_private_data
        assert not taint.can_exfiltrate

    def test_unknown_tool_maximally_tainted(self):
        """Test that unknown tools are treated as maximally tainted."""
        gate = DualLLMGate()
        taint = gate.get_tool_taint("unknown_tool")
        assert taint.reads_private_data
        assert taint.sees_untrusted_content
        assert taint.can_exfiltrate


class TestAsyncVerifier:
    """Tests for the async verifier."""

    @pytest.mark.asyncio
    async def test_valid_claim_passes(self):
        """Test that a valid claim passes deterministic checks."""
        verifier = AsyncVerifier()
        result = await verifier.verify({
            "goal": "Build a website",
            "deliverables": ["index.html", "styles.css"],
            "evidence": ["All tests pass", "Site renders correctly"],
        })
        # Should pass deterministic stage
        assert result["stage_results"]["deterministic"]["passed"]

    @pytest.mark.asyncio
    async def test_bare_done_rejection(self):
        """Test that bare 'done' without evidence is rejected."""
        verifier = AsyncVerifier()
        result = await verifier.verify({
            "goal": "Do something",
            "deliverables": ["thing"],
            "evidence": [],
            "self_report": "done",
        })
        # Should have a failure about missing evidence
        assert any(
            "evidence" in f.get("criterion_id", "").lower() or "evidence" in f.get("evidence", "").lower()
            for f in result.get("failures", [])
        )

    @pytest.mark.asyncio
    async def test_missing_goal_fails(self):
        """Test that missing goal fails validation."""
        verifier = AsyncVerifier()
        result = await verifier.verify({
            "goal": "",
            "deliverables": ["output.txt"],
            "evidence": ["exists"],
        })
        assert any(
            "structure" in f.get("criterion_id", "")
            for f in result.get("failures", [])
        )

    @pytest.mark.asyncio
    async def test_retry_budget_escalation(self):
        """Test that retry budget leads to escalation."""
        verifier = AsyncVerifier(max_retries=2)

        # Simulate repeated failures
        for _ in range(3):
            result = await verifier.verify({
                "goal": "",
                "deliverables": [],
                "evidence": [],
            })

        assert result["status"] == "ESCALATE"

    def test_completion_claim_validation(self):
        """Test CompletionClaim structural validation."""
        # Valid
        claim = CompletionClaim(
            goal="Build it",
            deliverables=["file.py"],
            evidence=["Tests pass"],
        )
        assert len(claim.validate_structure()) == 0

        # Invalid
        claim = CompletionClaim(goal="", deliverables=[], evidence=[])
        errors = claim.validate_structure()
        assert len(errors) == 3


class TestToolCallParser:
    """Tests for the model-agnostic tool call parser."""

    def test_parse_openai_native(self):
        """Test parsing native OpenAI tool_calls format."""
        parser = ToolCallParser()
        response = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "jlens_probe",
                        "arguments": '{"prompt": "hello world", "top_k": 5}',
                    },
                }
            ]
        }
        calls = parser.parse(response)
        assert len(calls) == 1
        assert calls[0].name == "jlens_probe"
        assert calls[0].arguments["prompt"] == "hello world"
        assert calls[0].arguments["top_k"] == 5

    def test_parse_hermes_xml(self):
        """Test parsing Hermes-XML format."""
        parser = ToolCallParser()
        content = '<tool_call>{"name": "jlens_probe", "arguments": {"prompt": "test"}}</tool_call>'
        calls = parser.parse(content, format=ToolCallFormat.HERMES_XML)
        assert len(calls) == 1
        assert calls[0].name == "jlens_probe"
        assert calls[0].arguments["prompt"] == "test"

    def test_parse_kimi_special(self):
        """Test parsing Kimi special token format."""
        parser = ToolCallParser()
        content = (
            '<|tool_calls_section_begin|>'
            '<|tool_call_begin|>functions.jlens_probe:0'
            '<|tool_sep|>{"prompt": "hello"}'
            '<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        calls = parser.parse(content, format=ToolCallFormat.KIMI_SPECIAL)
        assert len(calls) == 1
        assert calls[0].name == "jlens_probe"
        assert calls[0].id == "functions.jlens_probe:0"

    def test_auto_detect_hermes(self):
        """Test auto-detection of Hermes format."""
        parser = ToolCallParser()
        content = 'Let me call the tool: <tool_call>{"name": "test", "arguments": {}}</tool_call>'
        calls = parser.parse(content)
        assert len(calls) == 1
        assert calls[0].name == "test"

    def test_auto_detect_kimi(self):
        """Test auto-detection of Kimi format."""
        parser = ToolCallParser()
        content = '<|tool_calls_section_begin|><|tool_call_begin|>functions.test:0<|tool_sep|>{}<|tool_call_end|><|tool_calls_section_end|>'
        calls = parser.parse(content)
        assert len(calls) == 1

    def test_empty_response(self):
        """Test parsing response with no tool calls."""
        parser = ToolCallParser()
        calls = parser.parse("Just a regular text response with no tool calls.")
        assert len(calls) == 0

    def test_multiple_tool_calls(self):
        """Test parsing multiple tool calls."""
        parser = ToolCallParser()
        content = (
            '<tool_call>{"name": "tool1", "arguments": {"a": 1}}</tool_call>'
            'Some text in between'
            '<tool_call>{"name": "tool2", "arguments": {"b": 2}}</tool_call>'
        )
        calls = parser.parse(content, format=ToolCallFormat.HERMES_XML)
        assert len(calls) == 2
        assert calls[0].name == "tool1"
        assert calls[1].name == "tool2"
