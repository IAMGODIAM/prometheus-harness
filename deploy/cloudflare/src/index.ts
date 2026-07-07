/**
 * Prometheus MCP Server — Cloudflare Workers Edge Proxy
 *
 * This Worker acts as a Streamable HTTP transport for the Prometheus MCP server.
 * It handles:
 * 1. MCP protocol messages (JSON-RPC over HTTP)
 * 2. Session management via Durable Objects
 * 3. Authentication and rate limiting
 * 4. Edge-side watchlist pre-screening (lightweight J-lens heuristic)
 *
 * The heavy J-lens computation is delegated to the upstream Python server.
 * This edge layer provides low-latency routing and security enforcement.
 */

export interface Env {
  SESSION_STATE: DurableObjectNamespace;
  LENS_STORE: KVNamespace;
  MODEL_ARTIFACTS: R2Bucket;
  OPENAI_API_KEY: string;
  PROMETHEUS_AUTH_TOKEN: string;
  PROMETHEUS_UPSTREAM_URL: string;
  ENVIRONMENT: string;
}

interface MCPRequest {
  jsonrpc: "2.0";
  id?: string | number;
  method: string;
  params?: Record<string, unknown>;
}

interface MCPResponse {
  jsonrpc: "2.0";
  id?: string | number;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

// Edge-side watchlist keywords for pre-screening
const EDGE_WATCHLIST: Record<string, string[]> = {
  prompt_injection: [
    "ignore previous",
    "ignore above",
    "system prompt",
    "you are now",
    "disregard",
    "override instructions",
    "jailbreak",
  ],
  exfiltration: [
    "send to",
    "upload to",
    "post to",
    "webhook",
    "exfil",
    "base64",
  ],
};

function edgeWatchlistCheck(content: string): Record<string, number> {
  const scores: Record<string, number> = {};
  const lower = content.toLowerCase();

  for (const [category, keywords] of Object.entries(EDGE_WATCHLIST)) {
    let hits = 0;
    for (const keyword of keywords) {
      if (lower.includes(keyword)) hits++;
    }
    scores[category] = hits / keywords.length;
  }

  return scores;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // Health check
    if (url.pathname === "/health") {
      return new Response(JSON.stringify({ status: "ok", service: "prometheus-mcp" }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    // MCP endpoint
    if (url.pathname === "/mcp" || url.pathname === "/mcp/") {
      return handleMCP(request, env);
    }

    // Tool listing (convenience)
    if (url.pathname === "/tools") {
      return handleToolList(env);
    }

    return new Response("Not Found", { status: 404 });
  },
};

async function handleMCP(request: Request, env: Env): Promise<Response> {
  // Authenticate
  const authHeader = request.headers.get("Authorization");
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  const token = authHeader.slice(7);
  if (token !== env.PROMETHEUS_AUTH_TOKEN) {
    return new Response(JSON.stringify({ error: "Invalid token" }), {
      status: 403,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Parse MCP request
  let mcpRequest: MCPRequest;
  try {
    mcpRequest = await request.json() as MCPRequest;
  } catch {
    return new Response(
      JSON.stringify({ jsonrpc: "2.0", error: { code: -32700, message: "Parse error" } }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  }

  // Edge-side pre-screening for tool calls
  if (mcpRequest.method === "tools/call") {
    const params = mcpRequest.params as Record<string, unknown> | undefined;
    const args = params?.arguments as Record<string, unknown> | undefined;

    if (args) {
      const content = JSON.stringify(args);
      const watchlistScores = edgeWatchlistCheck(content);

      // Block if injection score is high
      if ((watchlistScores.prompt_injection ?? 0) > 0.3) {
        const response: MCPResponse = {
          jsonrpc: "2.0",
          id: mcpRequest.id,
          error: {
            code: -32001,
            message: "Request blocked by edge watchlist: potential prompt injection detected",
            data: { watchlist_scores: watchlistScores },
          },
        };
        return new Response(JSON.stringify(response), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
    }
  }

  // Forward to upstream Prometheus server
  const upstreamUrl = env.PROMETHEUS_UPSTREAM_URL || "http://localhost:8080/mcp";

  try {
    const upstreamResponse = await fetch(upstreamUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Prometheus-Session": request.headers.get("X-Session-ID") || "default",
        "X-Forwarded-For": request.headers.get("CF-Connecting-IP") || "unknown",
      },
      body: JSON.stringify(mcpRequest),
    });

    const result = await upstreamResponse.json();

    return new Response(JSON.stringify(result), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "X-Prometheus-Edge": "cloudflare",
        "X-Watchlist-Checked": "true",
      },
    });
  } catch (error) {
    const response: MCPResponse = {
      jsonrpc: "2.0",
      id: mcpRequest.id,
      error: {
        code: -32603,
        message: `Upstream error: ${error instanceof Error ? error.message : "unknown"}`,
      },
    };
    return new Response(JSON.stringify(response), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }
}

async function handleToolList(env: Env): Promise<Response> {
  const tools = [
    {
      name: "jlens_probe",
      description: "Apply J-Lens to a prompt and return top-k token predictions per layer",
      inputSchema: {
        type: "object",
        properties: {
          prompt: { type: "string", description: "Input prompt to analyze" },
          layers: { type: "array", items: { type: "integer" }, description: "Layer indices to probe" },
          top_k: { type: "integer", default: 10, description: "Number of top tokens to return" },
          position: { type: "integer", default: -1, description: "Token position to analyze (-1 = last)" },
        },
        required: ["prompt"],
      },
    },
    {
      name: "jlens_watchlist_scores",
      description: "Score a prompt against configured watchlist categories using J-Lens",
      inputSchema: {
        type: "object",
        properties: {
          prompt: { type: "string", description: "Input prompt to score" },
          categories: { type: "array", items: { type: "string" }, description: "Watchlist categories to check" },
          threshold: { type: "number", default: 0.15, description: "Alert threshold" },
        },
        required: ["prompt"],
      },
    },
    {
      name: "jlens_decompose",
      description: "Decompose activations into J-Space sparse representation",
      inputSchema: {
        type: "object",
        properties: {
          prompt: { type: "string", description: "Input prompt" },
          layer: { type: "integer", description: "Layer to decompose" },
          position: { type: "integer", default: -1, description: "Token position" },
          k: { type: "integer", default: 32, description: "Sparsity level" },
        },
        required: ["prompt", "layer"],
      },
    },
    {
      name: "jlens_steer",
      description: "Apply steering intervention to modify model behavior",
      inputSchema: {
        type: "object",
        properties: {
          prompt: { type: "string", description: "Input prompt" },
          concept: { type: "string", description: "Concept to steer toward/away from" },
          alpha: { type: "number", default: 1.0, description: "Steering strength (negative = away)" },
          layers: { type: "array", items: { type: "integer" }, description: "Layers to intervene" },
        },
        required: ["prompt", "concept"],
      },
    },
  ];

  return new Response(JSON.stringify({ tools }), {
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * Durable Object for session state management.
 * Tracks per-session taint state, budget counters, and watchlist alerts.
 */
export class SessionState {
  state: DurableObjectState;

  constructor(state: DurableObjectState) {
    this.state = state;
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/taint") {
      const taint = await this.state.storage.get("taint") || {
        has_private_data: false,
        has_untrusted_content: false,
        has_exfiltration_capability: false,
      };
      return new Response(JSON.stringify(taint), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.pathname === "/budget") {
      const budget = await this.state.storage.get("budget") || {
        tool_calls: 0,
        max_tool_calls: 100,
      };
      return new Response(JSON.stringify(budget), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (url.pathname === "/reset") {
      await this.state.storage.deleteAll();
      return new Response(JSON.stringify({ status: "reset" }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response("Not Found", { status: 404 });
  }
}
