"""Tracing — OpenTelemetry spans with Langfuse-compatible export.

Every MCP tool call, LLM generation, and agent loop iteration is traced.
Spans carry:
- W3C trace context (traceparent, tracestate)
- J-lens scores as span attributes
- Token counts and cost estimates
- Tool call metadata
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator

logger = logging.getLogger(__name__)


class SpanKind(str, Enum):
    AGENT_LOOP = "agent_loop"
    LLM_GENERATION = "llm_generation"
    TOOL_CALL = "tool_call"
    VERIFICATION = "verification"
    JLENS_PROBE = "jlens_probe"
    AUTHZ_CHECK = "authz_check"
    MEMORY_OP = "memory_op"


@dataclass
class Span:
    """A single trace span."""
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = ""
    parent_span_id: str | None = None
    name: str = ""
    kind: SpanKind = SpanKind.TOOL_CALL
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok, error
    error_message: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    @property
    def traceparent(self) -> str:
        """W3C traceparent header value."""
        return f"00-{self.trace_id}-{self.span_id}-01"

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def end(self, status: str = "ok", error: str | None = None) -> None:
        self.end_time = time.time()
        self.status = status
        if error:
            self.error_message = error

    def to_langfuse_dict(self) -> dict[str, Any]:
        """Convert to Langfuse-compatible format."""
        base = {
            "id": self.span_id,
            "traceId": self.trace_id,
            "parentObservationId": self.parent_span_id,
            "name": self.name,
            "startTime": self.start_time,
            "endTime": self.end_time,
            "metadata": self.attributes,
            "statusMessage": self.error_message,
        }

        if self.kind == SpanKind.LLM_GENERATION:
            base["type"] = "generation"
            base["model"] = self.attributes.get("model", "unknown")
            base["usage"] = {
                "input": self.attributes.get("input_tokens", 0),
                "output": self.attributes.get("output_tokens", 0),
                "total": self.attributes.get("total_tokens", 0),
            }
        else:
            base["type"] = "span"

        return base

    def to_otel_dict(self) -> dict[str, Any]:
        """Convert to OpenTelemetry-compatible format."""
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id or "",
            "name": self.name,
            "kind": self.kind.value,
            "startTimeUnixNano": int(self.start_time * 1e9),
            "endTimeUnixNano": int((self.end_time or time.time()) * 1e9),
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in self.attributes.items()
            ],
            "events": [
                {
                    "name": e["name"],
                    "timeUnixNano": int(e["timestamp"] * 1e9),
                    "attributes": [
                        {"key": k, "value": {"stringValue": str(v)}}
                        for k, v in e.get("attributes", {}).items()
                    ],
                }
                for e in self.events
            ],
            "status": {"code": 1 if self.status == "ok" else 2, "message": self.error_message or ""},
        }


class Tracer:
    """Trace manager for the Prometheus harness.

    Creates and manages spans across the agent lifecycle.
    Supports export to Langfuse, OTel collector, or local file.
    """

    def __init__(
        self,
        service_name: str = "prometheus",
        export_endpoint: str | None = None,
        langfuse_config: dict[str, str] | None = None,
    ):
        self.service_name = service_name
        self.export_endpoint = export_endpoint
        self.langfuse_config = langfuse_config
        self._spans: list[Span] = []
        self._active_spans: dict[str, Span] = {}
        self._current_trace_id: str = uuid.uuid4().hex

    def new_trace(self) -> str:
        """Start a new trace and return its ID."""
        self._current_trace_id = uuid.uuid4().hex
        return self._current_trace_id

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.TOOL_CALL,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Context manager for creating and managing spans."""
        s = Span(
            trace_id=self._current_trace_id,
            parent_span_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            attributes=attributes or {},
        )
        self._active_spans[s.span_id] = s

        try:
            yield s
        except Exception as e:
            s.end(status="error", error=str(e))
            raise
        finally:
            if s.end_time is None:
                s.end()
            del self._active_spans[s.span_id]
            self._spans.append(s)

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        parent: Span | None = None,
        cost: float | None = None,
    ) -> Span:
        """Record an LLM generation span."""
        s = Span(
            trace_id=self._current_trace_id,
            parent_span_id=parent.span_id if parent else None,
            name=f"llm.{model}",
            kind=SpanKind.LLM_GENERATION,
            attributes={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "duration_ms": duration_ms,
                "cost_usd": cost,
            },
        )
        s.end_time = s.start_time + (duration_ms / 1000)
        self._spans.append(s)
        return s

    def record_jlens_scores(
        self,
        span: Span,
        scores: dict[str, float],
    ) -> None:
        """Record J-lens watchlist scores on a span."""
        for category, score in scores.items():
            span.set_attribute(f"jlens.{category}", score)

    def export_traces(self) -> list[dict[str, Any]]:
        """Export all recorded spans."""
        return [s.to_langfuse_dict() for s in self._spans]

    def export_otel(self) -> list[dict[str, Any]]:
        """Export spans in OTel format."""
        return [s.to_otel_dict() for s in self._spans]

    async def flush(self) -> None:
        """Flush spans to the configured export endpoint."""
        if not self._spans:
            return

        if self.export_endpoint:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                try:
                    payload = {"spans": self.export_otel()}
                    async with session.post(
                        self.export_endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        if resp.status != 200:
                            logger.warning(f"Failed to export traces: {resp.status}")
                except Exception as e:
                    logger.error(f"Trace export failed: {e}")

        # Clear exported spans
        self._spans.clear()

    @property
    def span_count(self) -> int:
        return len(self._spans)

    @property
    def active_span_count(self) -> int:
        return len(self._active_spans)
