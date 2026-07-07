"""Obsidian Vault — Persistent memory storage using Markdown + wikilinks.

Maps memory blocks and archival entries to an Obsidian-compatible vault:
- Each memory atom is a Markdown file with YAML frontmatter
- Wikilinks ([[concept]]) create concept connections
- Frontmatter includes timestamps, tags, and metadata
- Compatible with Dataview queries for retrieval
- Supports daily notes for session logs
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VaultNote:
    """A single note in the Obsidian vault."""
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    modified: float = field(default_factory=time.time)

    def to_markdown(self) -> str:
        """Render note as Obsidian-compatible Markdown with frontmatter."""
        import datetime

        frontmatter_lines = [
            "---",
            f"title: \"{self.title}\"",
            f"created: {datetime.datetime.fromtimestamp(self.created).isoformat()}",
            f"modified: {datetime.datetime.fromtimestamp(self.modified).isoformat()}",
        ]

        if self.tags:
            frontmatter_lines.append(f"tags: [{', '.join(self.tags)}]")

        for key, value in self.metadata.items():
            if isinstance(value, str):
                frontmatter_lines.append(f"{key}: \"{value}\"")
            else:
                frontmatter_lines.append(f"{key}: {value}")

        frontmatter_lines.append("---")
        frontmatter = "\n".join(frontmatter_lines)

        return f"{frontmatter}\n\n{self.content}\n"

    @classmethod
    def from_markdown(cls, filepath: Path) -> VaultNote:
        """Parse a Markdown file into a VaultNote."""
        text = filepath.read_text(encoding="utf-8")
        title = filepath.stem

        # Parse frontmatter
        metadata = {}
        content = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                fm_text = parts[1].strip()
                content = parts[2].strip()
                for line in fm_text.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip('"')

        # Extract tags
        tags = []
        if "tags" in metadata:
            tag_str = metadata.pop("tags")
            tags = [t.strip() for t in tag_str.strip("[]").split(",") if t.strip()]

        # Extract wikilinks
        links = re.findall(r'\[\[(.*?)\]\]', content)

        return cls(
            title=title,
            content=content,
            tags=tags,
            links=links,
            metadata=metadata,
            created=filepath.stat().st_ctime,
            modified=filepath.stat().st_mtime,
        )


class ObsidianVault:
    """Obsidian vault interface for persistent memory storage.

    Directory structure:
    vault/
    ├── core/           # Core memory blocks
    ├── sessions/       # Session logs (daily notes)
    ├── concepts/       # Concept notes (from J-space decomposition)
    ├── agents/         # Agent persona and configuration
    ├── projects/       # Project-specific memory
    └── archive/        # Archived entries
    """

    SUBDIRS = ["core", "sessions", "concepts", "agents", "projects", "archive"]

    def __init__(self, vault_path: str | Path):
        self.vault_path = Path(vault_path)
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        """Create vault directory structure if it doesn't exist."""
        self.vault_path.mkdir(parents=True, exist_ok=True)
        for subdir in self.SUBDIRS:
            (self.vault_path / subdir).mkdir(exist_ok=True)

    def write_note(self, subdir: str, note: VaultNote) -> Path:
        """Write a note to the vault."""
        safe_title = re.sub(r'[^\w\s-]', '', note.title).strip().replace(' ', '_')
        filepath = self.vault_path / subdir / f"{safe_title}.md"
        filepath.write_text(note.to_markdown(), encoding="utf-8")
        return filepath

    def read_note(self, subdir: str, title: str) -> VaultNote | None:
        """Read a note from the vault."""
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        filepath = self.vault_path / subdir / f"{safe_title}.md"
        if filepath.exists():
            return VaultNote.from_markdown(filepath)
        return None

    def search(self, query: str, subdir: str | None = None) -> list[VaultNote]:
        """Search vault notes by content."""
        results = []
        search_dirs = [subdir] if subdir else self.SUBDIRS
        query_lower = query.lower()

        for sd in search_dirs:
            dir_path = self.vault_path / sd
            if not dir_path.exists():
                continue
            for filepath in dir_path.glob("*.md"):
                try:
                    note = VaultNote.from_markdown(filepath)
                    if query_lower in note.content.lower() or query_lower in note.title.lower():
                        results.append(note)
                except Exception as e:
                    logger.debug(f"Error reading {filepath}: {e}")

        return results

    def list_notes(self, subdir: str) -> list[str]:
        """List all note titles in a subdirectory."""
        dir_path = self.vault_path / subdir
        if not dir_path.exists():
            return []
        return [f.stem for f in dir_path.glob("*.md")]

    def save_core_memory(self, core_memory: Any) -> None:
        """Persist core memory blocks to the vault."""
        for name, block in core_memory.blocks.items():
            note = VaultNote(
                title=f"core_{name}",
                content=block.content,
                tags=["core_memory", name],
                metadata={
                    "block_type": "core",
                    "max_tokens": block.max_tokens,
                    "token_count": block.token_count,
                },
            )
            self.write_note("core", note)

    def save_session_log(self, session_id: str, messages: list[dict[str, Any]]) -> Path:
        """Save a session log as a daily note."""
        import datetime

        content_parts = [f"# Session: {session_id}\n"]
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                content_parts.append(f"**{role}**: {content[:200]}\n")

        note = VaultNote(
            title=f"session_{session_id}",
            content="\n".join(content_parts),
            tags=["session", datetime.date.today().isoformat()],
            metadata={"session_id": session_id, "message_count": len(messages)},
        )
        return self.write_note("sessions", note)

    def save_concept(self, concept: str, description: str, related: list[str] | None = None) -> Path:
        """Save a concept note (from J-space decomposition)."""
        links_text = ""
        if related:
            links_text = "\n\nRelated: " + ", ".join(f"[[{r}]]" for r in related)

        note = VaultNote(
            title=concept,
            content=f"{description}{links_text}",
            tags=["concept", "jspace"],
            links=related or [],
        )
        return self.write_note("concepts", note)

    def get_graph_data(self) -> dict[str, Any]:
        """Get vault graph data (nodes and edges from wikilinks)."""
        nodes = []
        edges = []

        for subdir in self.SUBDIRS:
            for title in self.list_notes(subdir):
                note = self.read_note(subdir, title)
                if note:
                    nodes.append({"id": title, "group": subdir, "tags": note.tags})
                    for link in note.links:
                        edges.append({"source": title, "target": link})

        return {"nodes": nodes, "edges": edges}
