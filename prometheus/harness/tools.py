"""Safe tool router + default toolset (RIG addition).

The review flagged that tool execution has no real sandbox (verifier.sandbox is a
stub). To run SAFELY despite that, this module:

  * ships only side-effect-free, in-process tools by default (echo, get_time);
  * defaults to dry-run, where even those are simulated (no execution at all);
  * declares each tool's authz category and dual-LLM taint so the orchestrator's
    gates apply to every call;
  * keeps filesystem/network tools OFF unless the operator opts in
    (allow_fs / allow_net), and even then they remain unsandboxed host-process
    calls — see RUN.md / RIG_NOTES.md for the loud caveat.

Every tool call still passes through the orchestrator's PreToolUse hooks
(dual-LLM taint gate + authz engine + optional J-lens gate) before reaching here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from prometheus.authz.engine import ToolCategory
from prometheus.harness.dual_llm import TaintLabel

logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Any]


@dataclass
class ToolSpec:
    """A registered tool: metadata for the model + gates + the handler."""
    name: str
    description: str
    parameters: dict[str, Any]          # JSON schema (OpenAI tool params)
    category: ToolCategory              # for the authz engine
    taint: TaintLabel                   # for the dual-LLM gate
    handler: Handler                    # the actual (in-process) implementation

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class SafeToolRouter:
    """Minimal, safe tool router with a dry-run mode.

    ``call()`` is what the orchestrator invokes (``tool_router.call(name, args)``).
    In dry-run, no handler is executed — a simulated result is returned — so a
    fresh ``prometheus run`` cannot touch the real filesystem or network.
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._specs.keys())

    def specs(self) -> dict[str, dict[str, Any]]:
        return {name: spec.to_openai() for name, spec in self._specs.items()}

    def categories(self) -> dict[str, ToolCategory]:
        return {name: spec.category for name, spec in self._specs.items()}

    def taints(self) -> dict[str, TaintLabel]:
        return {name: spec.taint for name, spec in self._specs.items()}

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown or unregistered tool: {name!r}")

        if self.dry_run:
            return {
                "dry_run": True,
                "tool": name,
                "arguments": arguments,
                "note": "simulated — no side effects (dry-run is the safe default)",
            }

        result = spec.handler(arguments)
        if hasattr(result, "__await__"):
            result = await result  # support async handlers
        return result


# --------------------------------------------------------------------------
# Default SAFE tool handlers (pure, in-process, no I/O)
# --------------------------------------------------------------------------

def _echo(args: dict[str, Any]) -> dict[str, Any]:
    return {"echo": args.get("text", "")}


def _get_time(args: dict[str, Any]) -> dict[str, Any]:
    return {"unix_time": time.time()}


# --------------------------------------------------------------------------
# Opt-in tools (registered only when the operator passes allow_fs / allow_net).
# Still unsandboxed host-process calls — off by default for that reason.
# --------------------------------------------------------------------------

def _make_read_file(fs_root: Path) -> Handler:
    fs_root = fs_root.resolve()

    def _read_file(args: dict[str, Any]) -> dict[str, Any]:
        raw = str(args.get("path", ""))
        target = (fs_root / raw).resolve()
        # Path-restriction guard (NOT a sandbox): must stay within fs_root.
        if fs_root not in target.parents and target != fs_root:
            return {"error": f"path escapes allowed root {fs_root}"}
        if not target.is_file():
            return {"error": f"not a file: {target}"}
        data = target.read_text(encoding="utf-8", errors="replace")
        return {"path": str(target), "content": data[:10000]}

    return _read_file


def _http_get(args: dict[str, Any]) -> dict[str, Any]:
    import httpx  # lazy
    url = str(args.get("url", ""))
    resp = httpx.get(url, timeout=15.0, follow_redirects=True)
    return {"url": url, "status": resp.status_code, "body": resp.text[:10000]}


def build_default_toolset(
    dry_run: bool = True,
    allow_fs: bool = False,
    allow_net: bool = False,
    fs_root: str | Path | None = None,
) -> SafeToolRouter:
    """Build the default safe toolset.

    Default = two pure in-process tools (echo, get_time). Filesystem/network tools
    are added only on explicit opt-in and are classified so the authz engine +
    dual-LLM gate govern them (net tools are additionally deny-by-default until the
    runner adds an explicit permit — see runner.py).
    """
    router = SafeToolRouter(dry_run=dry_run)

    router.register(ToolSpec(
        name="echo",
        description="Echo back the provided text. Pure, no side effects.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        category=ToolCategory.READ_ONLY,
        taint=TaintLabel(),  # no private data, no untrusted content, no exfiltration
        handler=_echo,
    ))
    router.register(ToolSpec(
        name="get_time",
        description="Return the current unix timestamp. Pure, no side effects.",
        parameters={"type": "object", "properties": {}},
        category=ToolCategory.READ_ONLY,
        taint=TaintLabel(),
        handler=_get_time,
    ))

    if allow_fs:
        root = Path(fs_root) if fs_root else Path.cwd()
        router.register(ToolSpec(
            name="read_file",
            description="Read a UTF-8 text file within the allowed root (read-only).",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            category=ToolCategory.READ_ONLY,     # read-only, but touches the FS
            taint=TaintLabel(reads_private_data=True, sees_untrusted_content=True),
            handler=_make_read_file(root),
        ))

    if allow_net:
        router.register(ToolSpec(
            name="http_get",
            description="HTTP GET a URL and return the body (truncated).",
            parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            category=ToolCategory.EXTERNAL_VISIBLE,  # deny-by-default until runner adds a permit
            taint=TaintLabel(sees_untrusted_content=True, can_exfiltrate=True),
            handler=_http_get,
        ))

    return router
