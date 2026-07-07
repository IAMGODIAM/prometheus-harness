"""MCP Transport Layer — Stdio, Streamable HTTP, and Cloudflare Workers support.

Supports three transport modes:
1. stdio: Local communication (default for co-located tools)
2. Streamable HTTP: Stateless remote communication (for shared services)
3. Cloudflare Workers: Edge deployment via Workers AI binding

The transport layer handles:
- W3C trace context propagation in _meta
- MCP-Protocol-Version header emission
- Mcp-Method / Mcp-Name routing headers
- RFC 8707 Resource Indicators for audience-bound tokens
"""

from __future__ import annotations

import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """Parsed MCP request."""
    jsonrpc: str = "2.0"
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    id: str | int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResponse:
    """MCP response to send."""
    jsonrpc: str = "2.0"
    result: Any = None
    error: dict[str, Any] | None = None
    id: str | int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        resp: dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error:
            resp["error"] = self.error
        else:
            resp["result"] = self.result
        if self.meta:
            resp["_meta"] = self.meta
        return resp


class MCPTransport(ABC):
    """Abstract base class for MCP transport implementations."""

    @abstractmethod
    async def start(self) -> None:
        """Start the transport."""
        ...

    @abstractmethod
    async def receive(self) -> AsyncIterator[MCPRequest]:
        """Receive incoming MCP requests."""
        ...

    @abstractmethod
    async def send(self, response: MCPResponse) -> None:
        """Send an MCP response."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport."""
        ...


class StdioTransport(MCPTransport):
    """stdio transport for local MCP communication.

    Reads JSON-RPC messages from stdin, writes responses to stdout.
    Used for co-located tools running in the same process or via subprocess.
    """

    def __init__(self):
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("Stdio transport started")

    async def receive(self) -> AsyncIterator[MCPRequest]:
        import asyncio

        while self._running:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                yield MCPRequest(
                    jsonrpc=data.get("jsonrpc", "2.0"),
                    method=data.get("method", ""),
                    params=data.get("params", {}),
                    id=data.get("id"),
                    meta=data.get("_meta", {}),
                )
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON on stdin: {e}")
            except Exception as e:
                logger.error(f"Error reading stdin: {e}")
                break

    async def send(self, response: MCPResponse) -> None:
        line = json.dumps(response.to_dict()) + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()

    async def stop(self) -> None:
        self._running = False


class StreamableHTTPTransport(MCPTransport):
    """Streamable HTTP transport for remote MCP communication.

    Stateless HTTP endpoint that handles MCP requests as POST bodies.
    Compatible with the 2025-11-25 spec's Streamable HTTP transport.

    Emits required headers:
    - MCP-Protocol-Version
    - Mcp-Method / Mcp-Name (for routing)
    - W3C traceparent/tracestate (via _meta)
    """

    MCP_PROTOCOL_VERSION = "2025-11-25"

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self._app = None
        self._server = None

    async def start(self) -> None:
        """Start the HTTP server using uvicorn."""
        import uvicorn
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def handle_mcp(request: Request) -> JSONResponse:
            body = await request.json()

            mcp_request = MCPRequest(
                jsonrpc=body.get("jsonrpc", "2.0"),
                method=body.get("method", ""),
                params=body.get("params", {}),
                id=body.get("id"),
                meta=body.get("_meta", {}),
            )

            # Process request (to be connected to server handler)
            response = await self._handle_request(mcp_request)

            return JSONResponse(
                content=response.to_dict(),
                headers={
                    "MCP-Protocol-Version": self.MCP_PROTOCOL_VERSION,
                    "Mcp-Method": mcp_request.method,
                    "Mcp-Name": mcp_request.params.get("name", ""),
                    "Content-Type": "application/json",
                },
            )

        app = Starlette(routes=[
            Route("/mcp", handle_mcp, methods=["POST"]),
            Route("/health", lambda r: JSONResponse({"status": "ok"})),
        ])

        self._app = app
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def _handle_request(self, request: MCPRequest) -> MCPResponse:
        """Override this to connect to the actual server handler."""
        return MCPResponse(
            id=request.id,
            error={"code": -32601, "message": "Method not found"},
        )

    async def receive(self) -> AsyncIterator[MCPRequest]:
        # HTTP transport is request-response, not streaming
        raise NotImplementedError("HTTP transport uses request-response pattern")
        yield  # Make it a generator

    async def send(self, response: MCPResponse) -> None:
        # Handled in the HTTP response
        pass

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True


class CloudflareWorkersTransport(MCPTransport):
    """Cloudflare Workers transport for edge deployment.

    Generates a Workers-compatible handler that can be deployed to Cloudflare's
    edge network. The handler processes MCP requests via the Workers AI binding
    or proxies to an upstream Prometheus instance.

    Deployment modes:
    1. Edge proxy: Workers routes requests to a Prometheus backend
    2. Edge compute: Workers runs lightweight J-lens operations via Workers AI
    3. Hybrid: Edge handles probes/watchlists, backend handles fitting/steering
    """

    def __init__(
        self,
        worker_name: str = "prometheus-mcp",
        upstream_url: str | None = None,
        ai_binding: bool = False,
    ):
        self.worker_name = worker_name
        self.upstream_url = upstream_url
        self.ai_binding = ai_binding

    def generate_worker_script(self) -> str:
        """Generate the Cloudflare Worker script for MCP handling.

        Returns:
            JavaScript/TypeScript source for the Worker.
        """
        return f'''// Prometheus MCP Worker — Auto-generated
// Handles MCP requests at the edge with optional Workers AI binding

export default {{
  async fetch(request, env, ctx) {{
    if (request.method === "OPTIONS") {{
      return new Response(null, {{
        headers: {{
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, MCP-Protocol-Version, Authorization",
        }},
      }});
    }}

    if (request.method !== "POST") {{
      return new Response(JSON.stringify({{ error: "Method not allowed" }}), {{
        status: 405,
        headers: {{ "Content-Type": "application/json" }},
      }});
    }}

    const url = new URL(request.url);
    if (url.pathname !== "/mcp") {{
      return new Response(JSON.stringify({{ error: "Not found" }}), {{
        status: 404,
        headers: {{ "Content-Type": "application/json" }},
      }});
    }}

    try {{
      const body = await request.json();
      const method = body.method || "";
      const params = body.params || {{}};

      // Route based on method
      let result;
      if (method === "tools/list") {{
        result = await handleToolsList();
      }} else if (method === "tools/call") {{
        result = await handleToolCall(params, env);
      }} else if (method === "initialize") {{
        result = await handleInitialize();
      }} else {{
        return new Response(JSON.stringify({{
          jsonrpc: "2.0",
          id: body.id,
          error: {{ code: -32601, message: "Method not found" }},
        }}), {{
          status: 200,
          headers: {{
            "Content-Type": "application/json",
            "MCP-Protocol-Version": "2025-11-25",
          }},
        }});
      }}

      return new Response(JSON.stringify({{
        jsonrpc: "2.0",
        id: body.id,
        result: result,
      }}), {{
        headers: {{
          "Content-Type": "application/json",
          "MCP-Protocol-Version": "2025-11-25",
          "Mcp-Method": method,
          "Mcp-Name": params.name || "",
        }},
      }});
    }} catch (err) {{
      return new Response(JSON.stringify({{
        jsonrpc: "2.0",
        id: null,
        error: {{ code: -32603, message: err.message }},
      }}), {{
        status: 500,
        headers: {{ "Content-Type": "application/json" }},
      }});
    }}
  }},
}};

async function handleInitialize() {{
  return {{
    protocolVersion: "2025-11-25",
    capabilities: {{
      tools: {{ listChanged: true }},
    }},
    serverInfo: {{
      name: "{self.worker_name}",
      version: "0.1.0",
    }},
  }};
}}

async function handleToolsList() {{
  return {{
    tools: [
      {{
        name: "jlens_probe",
        description: "Run J-lens readout on a prompt to see what tokens/concepts the model is disposed to say",
        inputSchema: {{
          type: "object",
          properties: {{
            prompt: {{ type: "string", description: "Text prompt to analyze" }},
            layers: {{ type: "array", items: {{ type: "integer" }}, description: "Layers to probe" }},
            top_k: {{ type: "integer", default: 10, description: "Top-k tokens to return" }},
          }},
          required: ["prompt"],
        }},
        annotations: {{
          readOnlyHint: true,
          destructiveHint: false,
          idempotentHint: true,
          openWorldHint: false,
        }},
      }},
      {{
        name: "jlens_watchlist_scores",
        description: "Score a prompt against safety watchlist categories",
        inputSchema: {{
          type: "object",
          properties: {{
            prompt: {{ type: "string", description: "Text prompt to score" }},
            categories: {{ type: "array", items: {{ type: "string" }}, description: "Categories to check" }},
          }},
          required: ["prompt"],
        }},
        annotations: {{
          readOnlyHint: true,
          destructiveHint: false,
          idempotentHint: true,
          openWorldHint: false,
        }},
      }},
      {{
        name: "jlens_decompose",
        description: "Decompose activation into J-space sparse representation",
        inputSchema: {{
          type: "object",
          properties: {{
            prompt: {{ type: "string", description: "Text prompt" }},
            layer: {{ type: "integer", description: "Layer to decompose at" }},
            position: {{ type: "integer", description: "Token position" }},
            k: {{ type: "integer", default: 25, description: "Sparsity level" }},
          }},
          required: ["prompt", "layer", "position"],
        }},
        annotations: {{
          readOnlyHint: true,
          destructiveHint: false,
          idempotentHint: true,
          openWorldHint: false,
        }},
      }},
    ],
  }};
}}

async function handleToolCall(params, env) {{
  const toolName = params.name;
  const args = params.arguments || {{}};

  {"// Proxy to upstream Prometheus instance" if self.upstream_url else "// Direct Workers AI execution"}
  {f'const upstream = "{self.upstream_url}";' if self.upstream_url else ""}

  {self._generate_proxy_logic() if self.upstream_url else self._generate_edge_logic()}
}}
'''

    def _generate_proxy_logic(self) -> str:
        """Generate proxy logic for upstream Prometheus instance."""
        return '''
  const response = await fetch(upstream + "/mcp", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "MCP-Protocol-Version": "2025-11-25",
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      method: "tools/call",
      params: params,
    }),
  });

  const data = await response.json();
  return data.result || data;
'''

    def _generate_edge_logic(self) -> str:
        """Generate edge compute logic using Workers AI."""
        return '''
  // Edge-native tool execution
  if (toolName === "jlens_watchlist_scores") {
    // Lightweight watchlist scoring at the edge
    const watchlists = {
      deception: ["fake", "secretly", "fraud", "trick", "hidden", "deliberately"],
      prompt_injection: ["injection", "poison", "override", "ignore", "bypass"],
      eval_awareness: ["fake", "fictional", "scenario", "benchmark", "simulation"],
    };

    const categories = args.categories || Object.keys(watchlists);
    const prompt = args.prompt || "";
    const promptLower = prompt.toLowerCase();

    const scores = {};
    let anyTriggered = false;

    for (const cat of categories) {
      if (!watchlists[cat]) continue;
      scores[cat] = {};
      for (const token of watchlists[cat]) {
        // Simple heuristic scoring at edge (full J-lens requires backend)
        const count = (promptLower.match(new RegExp(token, "gi")) || []).length;
        scores[cat][token] = count > 0 ? Math.min(count * 0.3, 1.0) : 0;
      }
      const maxScore = Math.max(...Object.values(scores[cat]));
      if (maxScore > 0.15) anyTriggered = true;
    }

    return {
      content: [{
        type: "text",
        text: JSON.stringify({ scores, any_triggered: anyTriggered, mode: "edge_heuristic" }),
      }],
    };
  }

  // For other tools, return an error indicating backend is needed
  return {
    content: [{
      type: "text",
      text: JSON.stringify({
        error: "This tool requires the full Prometheus backend. Deploy with upstream_url configured.",
        tool: toolName,
      }),
    }],
    isError: true,
  };
'''

    def generate_wrangler_config(self) -> str:
        """Generate wrangler.toml for Cloudflare Workers deployment."""
        config = f'''name = "{self.worker_name}"
main = "src/index.js"
compatibility_date = "2024-01-01"

[vars]
PROMETHEUS_VERSION = "0.1.0"
'''
        if self.upstream_url:
            config += f'\nUPSTREAM_URL = "{self.upstream_url}"\n'

        if self.ai_binding:
            config += '''
[ai]
binding = "AI"
'''
        return config

    async def start(self) -> None:
        logger.info(f"Cloudflare Workers transport configured: {self.worker_name}")

    async def receive(self) -> AsyncIterator[MCPRequest]:
        raise NotImplementedError("Workers transport is deployed, not run locally")
        yield

    async def send(self, response: MCPResponse) -> None:
        raise NotImplementedError("Workers transport is deployed, not run locally")

    async def stop(self) -> None:
        pass
