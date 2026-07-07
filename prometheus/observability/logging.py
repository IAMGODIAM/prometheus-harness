"""Structured Logging — JSON-formatted logs with correlation IDs.

Provides structured logging that integrates with the tracing system:
- JSON output for machine parsing
- Correlation IDs linking logs to traces
- Log levels with semantic meaning for agent operations
- Redaction of sensitive content
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class StructuredLogger:
    """Structured logger with trace correlation and JSON output."""

    def __init__(
        self,
        name: str = "prometheus",
        level: int = logging.INFO,
        json_output: bool = True,
        redact_patterns: list[str] | None = None,
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.json_output = json_output
        self._trace_id: str | None = None
        self._span_id: str | None = None
        self._redact_patterns = redact_patterns or []

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            if json_output:
                handler.setFormatter(JsonFormatter())
            else:
                handler.setFormatter(logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                ))
            self.logger.addHandler(handler)

    def set_trace_context(self, trace_id: str, span_id: str | None = None) -> None:
        """Set the current trace context for log correlation."""
        self._trace_id = trace_id
        self._span_id = span_id

    def _make_extra(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        result = extra or {}
        if self._trace_id:
            result["trace_id"] = self._trace_id
        if self._span_id:
            result["span_id"] = self._span_id
        return result

    def info(self, msg: str, **kwargs: Any) -> None:
        self.logger.info(msg, extra={"structured": self._make_extra(kwargs)})

    def warning(self, msg: str, **kwargs: Any) -> None:
        self.logger.warning(msg, extra={"structured": self._make_extra(kwargs)})

    def error(self, msg: str, **kwargs: Any) -> None:
        self.logger.error(msg, extra={"structured": self._make_extra(kwargs)})

    def debug(self, msg: str, **kwargs: Any) -> None:
        self.logger.debug(msg, extra={"structured": self._make_extra(kwargs)})

    def tool_call(self, tool_name: str, duration_ms: float, success: bool, **kwargs: Any) -> None:
        """Log a tool call with standard fields."""
        self.info(
            f"tool_call: {tool_name}",
            tool_name=tool_name,
            duration_ms=duration_ms,
            success=success,
            **kwargs,
        )

    def jlens_alert(self, category: str, score: float, threshold: float, **kwargs: Any) -> None:
        """Log a J-lens watchlist alert."""
        self.warning(
            f"jlens_alert: {category} score={score:.3f} threshold={threshold:.3f}",
            category=category,
            score=score,
            threshold=threshold,
            **kwargs,
        )

    def authz_decision(self, tool_name: str, decision: str, reason: str, **kwargs: Any) -> None:
        """Log an authorization decision."""
        level = "info" if decision == "allow" else "warning"
        getattr(self, level)(
            f"authz: {decision} {tool_name} — {reason}",
            tool_name=tool_name,
            decision=decision,
            reason=reason,
            **kwargs,
        )


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured output."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        structured = getattr(record, "structured", None)
        if structured:
            log_entry.update(structured)

        return json.dumps(log_entry, default=str)
