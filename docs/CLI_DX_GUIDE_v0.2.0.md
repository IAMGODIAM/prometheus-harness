# CLI / DX Guide — Prometheus-OS v0.2.0
**Sprint 4 CLI/DX Track**  
**Verified:** 2026-09-30 (mock provider)

---

## Quick Start
```bash
cd prometheus-harness
python -m venv .venv && source .venv/bin/activate
pip install -e .
export PROMETHEUS_LLM_PROVIDER=mock   # offline smoke
python -m prometheus.cli run --tier standard --effort medium --max-cost 500 --json "hello"
```

---

## Flags (verified)

| Flag | Values | Effect |
|------|--------|--------|
| `--tier` | `standard`, `large` | Primary agent compute tier |
| `--effort` | `low`, `medium`, `high` | Reasoning effort |
| `--return-to` | string | Return target label (orchestration metadata) |
| `--max-cost` | int (USD cents) | Delegation budget ceiling |
| `--dry-run` / `--no-dry-run` | default dry-run ON | Side-effect safety |
| `--allow-fs` / `--allow-net` | opt-in | Expand tool surface |
| `--enable-jlens-gate` | flag | Real gate only if torch+lens present; else no-op |
| `--json` | flag | Machine-readable result |
| `--provider` | `openai`, `mock` | Top-level provider (pool mock needs env — see below) |

### Mock mode caveat (fixed Sprint 4)
- Top-level `--provider mock` alone used to leave **subagent pool** on real NVIDIA providers → `NVIDIA_API_KEY` required.
- **Fix:** `ProviderFactory(force_mock=True)` when top-level client is `MockProvider`, or set `PROMETHEUS_LLM_PROVIDER=mock`.
- Prefer: `PROMETHEUS_LLM_PROVIDER=mock python -m prometheus.cli run ...`

---

## Smoke Result (2026-09-30)
```
PROMETHEUS_LLM_PROVIDER=mock python -m prometheus.cli run \
  --tier standard --effort medium --max-cost 500 --json "Sprint4 CLI DX smoke"
```
- status: `completion_claimed`
- verification: `ACCEPT` score 1.0
- budget_status.limits.max_cost_usd_cents: 500
- budget_status.spent.max_tool_calls: 1
- provider: `mock/deterministic-v1`

---

## Edge Cases
1. **Missing NVIDIA_API_KEY** with live provider → hard error (by design; keys never hardcoded)
2. **String tier/effort** into spawn_agent → now coerced to enums (Sprint 4 fix)
3. **max-cost** is accounting estimate, not live vendor billing
4. **TesterAgent** pytest warning: class name matches test collector — cosmetic only

---

## DX Recommendations (Sprint 5)
- Add explicit `--mock` flag on `run` that sets force_mock end-to-end
- Print `jlens_gate=off|noop|real` always
- Document tier→model map from `TIER_REGISTRY` in `prometheus status`

---

## MCP Client Configuration

### Local Upstream Server (Python MCP)
```bash
# Start upstream MCP server (J-Lens)
PROMETHEUS_LLM_PROVIDER=mock python -m prometheus.cli serve --transport http --host 0.0.0.0 --port 8084
```

### Cloudflare Worker (Edge Proxy)
Worker deployed at:
- `https://prometheus.e5enclave.com/mcp*`
- `https://prometheus-mcp.yisraelleemccartney.workers.dev/mcp*`

### MCP Client Config (Forge/Claude Code/any MCP client)
```json
{
  "mcpServers": {
    "prometheus": {
      "command": "npx",
      "args": ["mcp-remote", "https://prometheus.e5enclave.com/mcp"],
      "env": {
        "PROMETHEUS_AUTH_TOKEN": "prometheus-mcp-auth-2026"
      }
    }
  }
}
```

### Direct HTTP Client (curl)
```bash
# Initialize session
curl -H "Authorization: Bearer prometheus-mcp-auth-2026" \
     -H "Content-Type: application/json" \
     -H "Accept: application/json, text/event-stream" \
     -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' \
     https://prometheus.e5enclave.com/mcp

# List tools (use session ID from initialize response)
curl -H "Authorization: Bearer prometheus-mcp-auth-2026" \
     -H "Content-Type: application/json" \
     -H "Accept: application/json, text/event-stream" \
     -H "Mcp-Session-Id: <session-id-from-initialize>" \
     -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
     https://prometheus.e5enclave.com/mcp

# Call jlens_watchlist_scores
curl -H "Authorization: Bearer prometheus-mcp-auth-2026" \
     -H "Content-Type: application/json" \
     -H "Accept: application/json, text/event-stream" \
     -H "Mcp-Session-Id: <session-id>" \
     -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"jlens_watchlist_scores","arguments":{"prompt":"test prompt","categories":["prompt_injection"],"threshold":0.15}}}' \
     https://prometheus.e5enclave.com/mcp
```

### Available Tools on Worker
| Tool | Description |
|------|-------------|
| `jlens_probe` | Apply J-Lens to prompt, return top-k tokens per layer |
| `jlens_watchlist_scores` | Score prompt against watchlist categories |
| `jlens_decompose` | Decompose activation into J-space sparse representation |
| `jlens_steer` | Apply steering intervention (Phase 3, feature-flagged) |

### Health Check
```bash
curl https://prometheus.e5enclave.com/mcp/health
# Returns: {"status":"ok","service":"prometheus-mcp"}
```

### Tool List (Convenience Endpoint)
```bash
curl https://prometheus.e5enclave.com/mcp/tools
# Returns: {"tools":[...]}
```

---

## Environment Variables
| Variable | Purpose | Required |
|----------|---------|----------|
| `PROMETHEUS_LLM_PROVIDER` | `mock` or `nvidia` | For local server |
| `PROMETHEUS_AUTH_TOKEN` | Worker auth (Bearer) | For Worker calls |
| `NVIDIA_API_KEY` | Live provider key | For live mode |
| `OPENAI_API_KEY` | OpenAI for Worker steering | For steering tool |

---

## Sprint 4 Tracks Status
| Track | Status | Notes |
|-------|--------|-------|
| **Validation** | ✅ Benchmarks run | TB 0/5, DeepSWE 0/5 (mock - expected) |
| **MCP Integration** | ⚠️ Forge test pending | Worker live, upstream tunnel blocked |
| **Security (Real Gate)** | ⏳ Blocked on torch | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| **CLI/DX** | ✅ Guide updated | This document |