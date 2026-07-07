# AGENT_README — Prometheus

This document is optimized for consumption by AI agents, providing structured information about the Prometheus project for automated reasoning, tool discovery, and integration.

---

## Identity

```yaml
name: prometheus
type: mcp-server + agentic-harness
version: 0.1.0
language: python
min_python: "3.11"
paradigm: mcp-first, model-agnostic
```

## Capabilities

```yaml
provides:
  - jlens_probe: "Read model internal dispositions at any layer/position"
  - jlens_watchlist_scores: "Score prompts against safety concept categories"
  - jlens_decompose: "Sparse decomposition of activations into interpretable tokens"
  - jlens_steer: "Apply steering interventions during generation (Phase 3, feature-flagged)"
  - verifier_check: "5-stage async verification of completion claims"
  - harness_status: "System status and configuration"

requires:
  - pytorch: ">=2.0"
  - transformers: ">=4.40"
  - any_causal_lm: "HuggingFace-compatible decoder model"

optional:
  - fastmcp: "For full MCP server functionality"
  - langfuse: "For trace export"
  - pyyaml: "For YAML configuration"
  - cloudflare_wrangler: "For edge deployment"
```

## Tool Schemas

### jlens_probe

```json
{
  "name": "jlens_probe",
  "input": {
    "prompt": "string (required)",
    "layers": "int[] | null (default: workspace band)",
    "positions": "int[] | null (default: all)",
    "top_k": "int (default: 10)",
    "model_id": "string | null",
    "lens_id": "string | null"
  },
  "output": {
    "model_id": "string",
    "lens_id": "string",
    "layers_probed": "int[]",
    "results": "[{layer, positions: [{position, token_at_position, top_tokens: [{token, token_id, score}]}]}]"
  },
  "annotations": {"readOnlyHint": true, "idempotentHint": true}
}
```

### jlens_watchlist_scores

```json
{
  "name": "jlens_watchlist_scores",
  "input": {
    "prompt": "string (required)",
    "categories": "string[] | null (default: all)",
    "layers": "int[] | null",
    "model_id": "string | null",
    "lens_id": "string | null"
  },
  "output": {
    "alerts": "[{category, layer, max_score, threshold, top_tokens}]",
    "scores": "{category: {token: score}}",
    "any_triggered": "boolean"
  },
  "annotations": {"readOnlyHint": true, "idempotentHint": true}
}
```

### jlens_decompose

```json
{
  "name": "jlens_decompose",
  "input": {
    "prompt": "string (required)",
    "layer": "int (required)",
    "position": "int (required)",
    "k": "int (default: 25)",
    "model_id": "string | null",
    "lens_id": "string | null"
  },
  "output": {
    "top_tokens": "[{token, token_id, coefficient, rank}]",
    "variance_explained": "float (0-1)",
    "occupancy": "int"
  },
  "annotations": {"readOnlyHint": true, "idempotentHint": true}
}
```

## Security Invariants

```yaml
invariants:
  - "Deny-first: any FORBID policy overrides all PERMIT policies"
  - "Lethal Trifecta: block exfiltration when session has private_data AND untrusted_content"
  - "Dual-LLM: P-LLM never sees raw untrusted content; Q-LLM never gets tool access"
  - "Bare 'done' rejection: CompletionClaims must include goal + deliverables + evidence"
  - "Budget enforcement: max_tool_calls, max_irreversible, max_external_calls"
  - "J-lens gating: elevated concept scores block downstream actions"
```

## Integration Patterns

### As MCP Server (recommended)

```bash
prometheus serve --transport stdio
# Or HTTP for remote:
prometheus serve --transport http --port 8080
```

### As Python Library

```python
from prometheus.jlens import JacobianLens
from prometheus.jlens.model_adapter import ModelAdapter
from prometheus.harness import Orchestrator, DualLLMGate, AsyncVerifier
from prometheus.authz import AuthzEngine
from prometheus.memory import MemoryManager
from prometheus.observability import Tracer
```

### As Cloudflare Worker (edge)

```bash
prometheus deploy --name my-prometheus --upstream https://backend:8080
```

## File Map

```
prometheus/jlens/lens.py          — Core JacobianLens class (fit, apply, save/load)
prometheus/jlens/fitting.py       — VJP-based lens fitting
prometheus/jlens/jspace.py        — Gradient pursuit decomposition
prometheus/jlens/interventions.py — Steer/ablate/swap hooks
prometheus/jlens/watchlist.py     — Category scoring engine
prometheus/jlens/model_adapter.py — Model-agnostic HF adapter
prometheus/mcp_server/server.py   — FastMCP tool definitions
prometheus/mcp_server/transport.py— stdio/HTTP/Workers transports
prometheus/harness/orchestrator.py— Agent loop + hook system
prometheus/harness/dual_llm.py    — CaMeL dual-LLM gate
prometheus/harness/verifier.py    — 5-stage verification pipeline
prometheus/harness/tool_parser.py — 4-format tool call parser
prometheus/authz/engine.py        — Cedar-style policy engine
prometheus/authz/taint.py         — Information flow taint tracking
prometheus/memory/blocks.py       — Letta-style memory blocks
prometheus/memory/vault.py        — Obsidian vault persistence
prometheus/observability/tracing.py — OTel/Langfuse tracing
prometheus/config.py              — Hierarchical configuration
prometheus/cli.py                 — CLI entry point
```

## Extension Points

```yaml
hooks:
  - PreToolUse: "Add custom checks before any tool execution"
  - PostToolUse: "Add custom processing after tool execution"
  - PreCompact: "Custom context management before compaction"
  - UserPromptSubmit: "Intercept user messages"
  - SessionStart/End: "Session lifecycle events"

policies:
  - "Add custom Cedar-style policies via AuthzEngine.add_policy()"
  - "J-lens conditions can reference any concept category"

watchlists:
  - "Add custom categories via Watchlist.add_category()"
  - "Token lists and thresholds are configurable"

memory:
  - "Add custom core memory blocks via CoreMemory"
  - "Vault supports custom subdirectories"

transports:
  - "Implement MCPTransport interface for custom transports"
```

## Agent Collaboration

This project supports multi-agent collaboration patterns:

1. **Linked Agents**: Multiple agents can connect to the same Prometheus MCP server for shared interpretability monitoring.
2. **Verifier Independence**: The verifier agent uses a different model family than the actor for uncorrelated verification.
3. **Context Isolation**: Subagents operate in isolated contexts to prevent cross-contamination.
4. **Shared Memory**: The Obsidian vault provides persistent shared memory across sessions and agents.

---

*This document follows the agent-friendly README convention for automated consumption.*
