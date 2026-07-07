"""Tests for Authorization and Memory modules."""

import pytest
import time
from pathlib import Path

from prometheus.authz.engine import (
    AuthzEngine,
    Policy,
    PolicyEffect,
    Decision,
    AuthzRequest,
    AuthzDecision,
    ToolCategory,
)
from prometheus.authz.taint import TaintTracker, TaintLabel
from prometheus.memory.blocks import MemoryBlock, CoreMemory, MemoryManager
from prometheus.memory.vault import ObsidianVault, VaultNote


class TestAuthzEngine:
    """Tests for the Cedar-style authorization engine."""

    def test_deny_by_default(self):
        """Test that requests are denied when no policies match."""
        engine = AuthzEngine()
        request = AuthzRequest(
            principal={"agent_id": "agent1", "role": "actor"},
            action_name="some_tool",
        )
        decision = engine.evaluate(request)
        assert decision.decision == Decision.DENY

    def test_permit_policy(self):
        """Test that a matching permit policy allows the request."""
        engine = AuthzEngine()
        engine.add_policy(Policy(
            id="allow_read",
            effect=PolicyEffect.PERMIT,
            action={"category": "read_only"},
        ))
        engine.classify_tool("jlens_probe", ToolCategory.READ_ONLY)

        request = AuthzRequest(
            principal={"agent_id": "agent1"},
            action_name="jlens_probe",
        )
        decision = engine.evaluate(request)
        assert decision.decision == Decision.ALLOW

    def test_deny_overrides_permit(self):
        """Test deny-first invariant: forbid overrides permit."""
        engine = AuthzEngine()
        engine.add_policy(Policy(
            id="allow_all",
            effect=PolicyEffect.PERMIT,
            action={"name": "*"},
            priority=10,
        ))
        engine.add_policy(Policy(
            id="deny_dangerous",
            effect=PolicyEffect.FORBID,
            action={"name": "delete_everything"},
            priority=100,
        ))

        request = AuthzRequest(
            principal={"agent_id": "agent1"},
            action_name="delete_everything",
        )
        decision = engine.evaluate(request)
        assert decision.decision == Decision.DENY

    def test_jlens_condition(self):
        """Test J-lens score conditions in policies."""
        engine = AuthzEngine()
        engine.add_policy(Policy(
            id="block_deception",
            effect=PolicyEffect.FORBID,
            action={"category": "external_visible"},
            jlens_conditions={"deception": 0.15},
            priority=100,
        ))
        engine.add_policy(Policy(
            id="allow_external",
            effect=PolicyEffect.PERMIT,
            action={"category": "external_visible"},
            priority=10,
        ))
        engine.classify_tool("send_email", ToolCategory.EXTERNAL_VISIBLE)

        # High deception score → blocked
        request = AuthzRequest(
            principal={"agent_id": "agent1"},
            action_name="send_email",
            jlens_scores={"deception": 0.25},
        )
        decision = engine.evaluate(request)
        assert decision.decision == Decision.DENY

        # Low deception score → allowed
        request = AuthzRequest(
            principal={"agent_id": "agent1"},
            action_name="send_email",
            jlens_scores={"deception": 0.05},
        )
        decision = engine.evaluate(request)
        assert decision.decision == Decision.ALLOW

    def test_budget_enforcement(self):
        """Test budget limits are enforced."""
        engine = AuthzEngine()
        engine.set_budget_limit("max_tool_calls", 3)
        engine.add_policy(Policy(
            id="allow_all",
            effect=PolicyEffect.PERMIT,
            action={"name": "*"},
        ))

        request = AuthzRequest(
            principal={"agent_id": "agent1"},
            action_name="tool1",
        )

        # First 3 calls should be allowed
        for _ in range(3):
            decision = engine.evaluate(request)
            assert decision.decision == Decision.ALLOW
            engine._increment_budget(request)

        # 4th call should be denied
        decision = engine.evaluate(request)
        assert decision.decision == Decision.DENY

    def test_default_policies(self):
        """Test loading default policies."""
        engine = AuthzEngine()
        engine.load_default_policies()
        assert len(engine.policies) > 0

    def test_classify_tool(self):
        """Test tool classification."""
        engine = AuthzEngine()
        engine.classify_tool("read_file", ToolCategory.READ_ONLY)
        engine.classify_tool("send_email", ToolCategory.EXTERNAL_VISIBLE)
        assert engine._tool_categories["read_file"] == ToolCategory.READ_ONLY
        assert engine._tool_categories["send_email"] == ToolCategory.EXTERNAL_VISIBLE


class TestTaintTracker:
    """Tests for taint tracking."""

    def test_taint_label_basics(self):
        """Test TaintLabel properties."""
        label = TaintLabel(reads_private_data=True, sees_untrusted_content=False, can_exfiltrate=False)
        assert not label.is_lethal_trifecta
        assert label.trifecta_count == 1

    def test_trifecta_detection(self):
        """Test Lethal Trifecta detection."""
        tracker = TaintTracker()
        tracker.declare_tool("read_db", TaintLabel(reads_private_data=True))
        tracker.declare_tool("fetch_url", TaintLabel(sees_untrusted_content=True))
        tracker.declare_tool("send_email", TaintLabel(can_exfiltrate=True))

        # First two are fine
        allowed, _ = tracker.check_tool_call("read_db")
        assert allowed
        tracker.record_tool_call("read_db")

        allowed, _ = tracker.check_tool_call("fetch_url")
        assert allowed
        tracker.record_tool_call("fetch_url")

        # Third would complete trifecta
        allowed, reason = tracker.check_tool_call("send_email")
        assert not allowed
        assert "Lethal Trifecta" in reason

    def test_session_risk_levels(self):
        """Test session risk level calculation."""
        tracker = TaintTracker()
        tracker.declare_tool("read_db", TaintLabel(reads_private_data=True))

        assert tracker.session_risk == "low"
        tracker.record_tool_call("read_db")
        assert tracker.session_risk == "medium"

    def test_reset(self):
        """Test session reset."""
        tracker = TaintTracker()
        tracker.declare_tool("read_db", TaintLabel(reads_private_data=True))
        tracker.record_tool_call("read_db")
        assert tracker.session_risk == "medium"
        tracker.reset()
        assert tracker.session_risk == "low"


class TestMemoryBlocks:
    """Tests for Letta-style memory blocks."""

    def test_block_append(self):
        """Test appending to a memory block."""
        block = MemoryBlock(name="test", max_tokens=100)
        assert block.append("Hello, world!")
        assert "Hello, world!" in block.content

    def test_block_budget_enforcement(self):
        """Test that block budget is enforced."""
        block = MemoryBlock(name="test", max_tokens=5)
        # 5 tokens ≈ 20 chars
        result = block.append("x" * 100)  # Way over budget
        assert not result

    def test_block_replace(self):
        """Test replacing text in a block."""
        block = MemoryBlock(name="test", max_tokens=100)
        block.append("Hello, world!")
        assert block.replace("world", "Prometheus")
        assert "Prometheus" in block.content

    def test_core_memory_context(self):
        """Test core memory context rendering."""
        core = CoreMemory()
        core.set_content("persona", "I am Prometheus, an interpretability agent.")
        core.set_content("user", "The user is a researcher.")

        context = core.to_context()
        assert "<core_memory>" in context
        assert "Prometheus" in context
        assert "researcher" in context

    def test_memory_manager_operations(self):
        """Test MemoryManager CRUD operations."""
        manager = MemoryManager()

        # Append to core
        result = manager.core_memory_append("persona", "I am helpful.")
        assert result["success"]

        # Search recall (empty)
        results = manager.recall_memory_search("test")
        assert len(results) == 0

        # Insert to archival
        result = manager.archival_memory_insert("Important fact about J-lens")
        assert result["success"]

        # Search archival
        results = manager.archival_memory_search("J-lens")
        assert len(results) == 1


class TestObsidianVault:
    """Tests for Obsidian vault integration."""

    def test_vault_creation(self, tmp_path):
        """Test vault directory structure creation."""
        vault = ObsidianVault(tmp_path / "test_vault")
        assert (tmp_path / "test_vault" / "core").exists()
        assert (tmp_path / "test_vault" / "sessions").exists()
        assert (tmp_path / "test_vault" / "concepts").exists()

    def test_note_write_read(self, tmp_path):
        """Test writing and reading a note."""
        vault = ObsidianVault(tmp_path / "test_vault")

        note = VaultNote(
            title="Test Note",
            content="This is a test note about [[J-lens]].",
            tags=["test", "jlens"],
        )
        vault.write_note("concepts", note)

        read_note = vault.read_note("concepts", "Test Note")
        assert read_note is not None
        assert "J-lens" in read_note.content
        assert "J-lens" in read_note.links

    def test_vault_search(self, tmp_path):
        """Test searching the vault."""
        vault = ObsidianVault(tmp_path / "test_vault")

        vault.write_note("concepts", VaultNote(
            title="Deception",
            content="Deception detection using J-lens watchlists.",
            tags=["safety"],
        ))
        vault.write_note("concepts", VaultNote(
            title="Steering",
            content="Steering interventions modify model behavior.",
            tags=["intervention"],
        ))

        results = vault.search("J-lens")
        assert len(results) == 1
        assert results[0].title == "Deception"

    def test_note_frontmatter(self, tmp_path):
        """Test that notes have proper YAML frontmatter."""
        vault = ObsidianVault(tmp_path / "test_vault")

        note = VaultNote(
            title="Frontmatter Test",
            content="Content here.",
            tags=["test"],
            metadata={"category": "testing"},
        )
        path = vault.write_note("core", note)

        content = path.read_text()
        assert content.startswith("---")
        assert "title:" in content
        assert "created:" in content
        assert "tags:" in content

    def test_vault_list_notes(self, tmp_path):
        """Test listing notes in a subdirectory."""
        vault = ObsidianVault(tmp_path / "test_vault")

        vault.write_note("concepts", VaultNote(title="Note1", content="A"))
        vault.write_note("concepts", VaultNote(title="Note2", content="B"))

        notes = vault.list_notes("concepts")
        assert len(notes) == 2
