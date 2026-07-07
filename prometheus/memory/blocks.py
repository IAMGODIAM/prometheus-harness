"""Memory Blocks — Letta-style editable memory with token budgets.

Each memory block is:
- Named and typed (persona, user, system, project, scratch)
- Token-budgeted (enforced max size)
- Editable by the agent via core_memory_append/replace/delete tools
- Always included in context (core memory) or retrieved on demand (recall/archival)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MemoryBlock:
    """A single memory block with content and metadata."""
    name: str
    content: str = ""
    max_tokens: int = 2000
    block_type: str = "core"  # core, recall, archival
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(self.content) // 4

    @property
    def usage_pct(self) -> float:
        """Percentage of token budget used."""
        return min(100.0, (self.token_count / self.max_tokens) * 100)

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of content for change detection."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]

    def append(self, text: str) -> bool:
        """Append text to the block if within budget."""
        projected = (len(self.content) + len(text)) // 4
        if projected > self.max_tokens:
            logger.warning(f"Block '{self.name}' would exceed budget: {projected}/{self.max_tokens}")
            return False
        self.content += text
        self.updated_at = time.time()
        return True

    def replace(self, old: str, new: str) -> bool:
        """Replace text in the block."""
        if old not in self.content:
            return False
        projected = (len(self.content) - len(old) + len(new)) // 4
        if projected > self.max_tokens:
            return False
        self.content = self.content.replace(old, new, 1)
        self.updated_at = time.time()
        return True

    def delete(self, text: str) -> bool:
        """Delete text from the block."""
        if text not in self.content:
            return False
        self.content = self.content.replace(text, "", 1)
        self.updated_at = time.time()
        return True

    def to_context_string(self) -> str:
        """Format block for inclusion in LLM context."""
        return f"<memory_block name=\"{self.name}\" tokens=\"{self.token_count}/{self.max_tokens}\">\n{self.content}\n</memory_block>"


class CoreMemory:
    """Core memory — always in-context, editable by agent.

    Standard blocks:
    - persona: Agent's self-description and capabilities
    - user: Information about the user
    - system: System configuration and constraints
    - project: Current project context
    - scratch: Temporary working memory
    """

    DEFAULT_BLOCKS = {
        "persona": 1500,
        "user": 1500,
        "system": 1000,
        "project": 2000,
        "scratch": 1000,
    }

    def __init__(self, blocks: dict[str, int] | None = None):
        self.blocks: dict[str, MemoryBlock] = {}
        block_config = blocks or self.DEFAULT_BLOCKS
        for name, max_tokens in block_config.items():
            self.blocks[name] = MemoryBlock(
                name=name,
                max_tokens=max_tokens,
                block_type="core",
            )

    def get(self, name: str) -> MemoryBlock | None:
        return self.blocks.get(name)

    def set_content(self, name: str, content: str) -> bool:
        """Set the full content of a block."""
        block = self.blocks.get(name)
        if not block:
            return False
        if len(content) // 4 > block.max_tokens:
            return False
        block.content = content
        block.updated_at = time.time()
        return True

    def to_context(self) -> str:
        """Render all core memory blocks for LLM context."""
        parts = ["<core_memory>"]
        for block in self.blocks.values():
            if block.content:
                parts.append(block.to_context_string())
        parts.append("</core_memory>")
        return "\n".join(parts)

    @property
    def total_tokens(self) -> int:
        return sum(b.token_count for b in self.blocks.values())


class MemoryManager:
    """Manages all three memory tiers with search and retrieval.

    Provides MCP-compatible tools for memory operations:
    - core_memory_append
    - core_memory_replace
    - core_memory_delete
    - recall_memory_search
    - archival_memory_insert
    - archival_memory_search
    """

    def __init__(
        self,
        core: CoreMemory | None = None,
        vault_path: str | None = None,
    ):
        self.core = core or CoreMemory()
        self.recall: list[dict[str, Any]] = []
        self.archival: list[dict[str, Any]] = []
        self._vault_path = vault_path

    def core_memory_append(self, block_name: str, content: str) -> dict[str, Any]:
        """Append to a core memory block."""
        block = self.core.get(block_name)
        if not block:
            return {"success": False, "error": f"Block '{block_name}' not found"}
        success = block.append(content)
        return {
            "success": success,
            "block": block_name,
            "tokens_used": block.token_count,
            "tokens_max": block.max_tokens,
        }

    def core_memory_replace(self, block_name: str, old: str, new: str) -> dict[str, Any]:
        """Replace text in a core memory block."""
        block = self.core.get(block_name)
        if not block:
            return {"success": False, "error": f"Block '{block_name}' not found"}
        success = block.replace(old, new)
        return {"success": success, "block": block_name}

    def recall_memory_search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search recall memory (conversation history)."""
        results = []
        query_lower = query.lower()
        for entry in reversed(self.recall):
            content = entry.get("content", "")
            if query_lower in content.lower():
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def archival_memory_insert(self, content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Insert into archival memory."""
        entry = {
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "id": hashlib.sha256(content.encode()).hexdigest()[:12],
        }
        self.archival.append(entry)
        return {"success": True, "id": entry["id"], "total_entries": len(self.archival)}

    def archival_memory_search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search archival memory."""
        results = []
        query_lower = query.lower()
        for entry in self.archival:
            if query_lower in entry.get("content", "").lower():
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def add_to_recall(self, message: dict[str, Any]) -> None:
        """Add a message to recall memory."""
        self.recall.append({
            **message,
            "timestamp": time.time(),
        })

    def get_context_string(self) -> str:
        """Get the full memory context string for LLM inclusion."""
        return self.core.to_context()

    def export_state(self) -> dict[str, Any]:
        """Export full memory state for persistence."""
        return {
            "core": {
                name: {"content": block.content, "updated_at": block.updated_at}
                for name, block in self.core.blocks.items()
            },
            "recall_count": len(self.recall),
            "archival_count": len(self.archival),
        }
