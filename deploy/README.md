# Prometheus Deployment

## Architecture Options

Prometheus supports three deployment topologies:

### 1. Local (Development)

Run the MCP server directly:

```bash
# Install dependencies
pip install -e ".[all]"

# Run MCP server (stdio transport for Claude/Cursor)
prometheus serve --transport stdio

# Run MCP server (HTTP for remote access)
prometheus serve --transport streamable-http --port 8080
```

### 2. Docker (Self-hosted)

```bash
cd deploy
docker compose up -d

# With observability stack (Langfuse)
docker compose --profile observability up -d
```

### 3. Cloudflare Workers (Edge)

The Cloudflare Worker acts as an edge proxy that:
- Authenticates requests
- Performs lightweight edge-side watchlist pre-screening
- Manages session state via Durable Objects
- Forwards heavy computation to the upstream Python server

```bash
cd deploy/cloudflare
npm install

# Configure secrets
wrangler secret put PROMETHEUS_AUTH_TOKEN
wrangler secret put PROMETHEUS_UPSTREAM_URL
wrangler secret put OPENAI_API_KEY

# Deploy
npm run deploy
```

## Topology: Edge + Origin

```
Client (Claude/Cursor/Agent)
    │
    ▼
┌─────────────────────────────────┐
│  Cloudflare Worker (Edge)       │
│  • Auth + rate limiting         │
│  • Edge watchlist pre-screen    │
│  • Session state (DO)           │
│  • Lens cache (KV)             │
└─────────────┬───────────────────┘
              │ HTTPS
              ▼
┌─────────────────────────────────┐
│  Prometheus Python Server       │
│  • Full J-Lens computation      │
│  • Model inference              │
│  • Authz engine                 │
│  • Memory subsystem             │
└─────────────────────────────────┘
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PROMETHEUS_MODEL_ID` | HuggingFace model ID for J-Lens | No (default: gpt2) |
| `PROMETHEUS_LOG_LEVEL` | Logging level | No (default: info) |
| `OPENAI_API_KEY` | OpenAI API key for verifier LLM | No |
| `PROMETHEUS_AUTH_TOKEN` | Bearer token for API auth | Yes (production) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry collector URL | No |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | No |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | No |

## MCP Client Configuration

### Claude Desktop

```json
{
  "mcpServers": {
    "prometheus": {
      "command": "prometheus",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
```

### Remote (via Cloudflare Worker)

```json
{
  "mcpServers": {
    "prometheus": {
      "url": "https://prometheus-mcp.your-worker.workers.dev/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```
