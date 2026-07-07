"""Taint Tracking — Information flow control for the Lethal Trifecta rule.

Tracks how data flows through the system and prevents combinations that
would enable prompt-injection exploitation:
- private_data + untrusted_content + exfiltration = BLOCKED

Every tool declares its taint properties. The tracker maintains a session-level
view of accumulated taint and blocks dangerous combinations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TaintLabel:
    """Taint properties for a tool or content block."""
    reads_private_data: bool = False
    sees_untrusted_content: bool = False
    can_exfiltrate: bool = False

    @property
    def is_lethal_trifecta(self) -> bool:
        return all([self.reads_private_data, self.sees_untrusted_content, self.can_exfiltrate])

    @property
    def trifecta_count(self) -> int:
        """Count how many trifecta legs are active."""
        return sum([self.reads_private_data, self.sees_untrusted_content, self.can_exfiltrate])

    @property
    def risk_level(self) -> str:
        count = sum([self.reads_private_data, self.sees_untrusted_content, self.can_exfiltrate])
        if count >= 3:
            return "critical"
        elif count == 2:
            return "high"
        elif count == 1:
            return "medium"
        return "low"


class TaintTracker:
    """Session-level taint tracker enforcing the Lethal Trifecta rule."""

    def __init__(self):
        self._session_taint = TaintLabel()
        self._tool_taints: dict[str, TaintLabel] = {}
        self._history: list[dict[str, Any]] = []

    def declare_tool(self, tool_name: str, taint: TaintLabel) -> None:
        """Declare taint properties for a tool."""
        self._tool_taints[tool_name] = taint

    def check_tool_call(self, tool_name: str) -> tuple[bool, str]:
        """Check if a tool call would create a trifecta violation.

        Returns:
            Tuple of (allowed, reason).
        """
        tool_taint = self._tool_taints.get(tool_name, TaintLabel(
            reads_private_data=True, sees_untrusted_content=True, can_exfiltrate=True
        ))

        # Simulate what session taint would be after this call
        projected = TaintLabel(
            reads_private_data=self._session_taint.reads_private_data or tool_taint.reads_private_data,
            sees_untrusted_content=self._session_taint.sees_untrusted_content or tool_taint.sees_untrusted_content,
            can_exfiltrate=self._session_taint.can_exfiltrate or tool_taint.can_exfiltrate,
        )

        if projected.is_lethal_trifecta:
            return False, (
                f"Tool '{tool_name}' would complete the Lethal Trifecta: "
                f"private_data={projected.reads_private_data}, "
                f"untrusted_content={projected.sees_untrusted_content}, "
                f"exfiltration={projected.can_exfiltrate}"
            )

        return True, "OK"

    def record_tool_call(self, tool_name: str) -> None:
        """Record that a tool was called, updating session taint."""
        tool_taint = self._tool_taints.get(tool_name, TaintLabel())
        self._session_taint.reads_private_data |= tool_taint.reads_private_data
        self._session_taint.sees_untrusted_content |= tool_taint.sees_untrusted_content
        self._session_taint.can_exfiltrate |= tool_taint.can_exfiltrate
        self._history.append({"tool": tool_name, "taint": tool_taint})

    def reset(self) -> None:
        """Reset session taint (new conversation)."""
        self._session_taint = TaintLabel()
        self._history.clear()

    @property
    def session_risk(self) -> str:
        return self._session_taint.risk_level
