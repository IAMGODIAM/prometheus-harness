# Hermie MCP Server Design v0.1
**Sprint 4 — MCP Integration Track**  
**Date:** 2026-09-30  
**Status:** Draft for Forge + Cerebro implementation

---

## Goal
Expose Prometheus-OS orchestration to the Hallways mesh via MCP so board agents can:
- spawn specialists (explorer/reviewer/tester/designer)
- run tasks with tier/effort/budget controls
- read warm-pool + budget metrics
- without public unauthenticated exposure

**Constraint (mission law):** no public MCP without auth.

---

## Transport & Auth
| Surface | Choice | Auth |
|---------|--------|------|
| Local mesh | stdio MCP (Hermie process) | process-local trust |
| Boardroom node | streamable HTTP on `127.0.0.1` only | bearer token = `HALLWAYS_AUTH_TOKEN` |
| Public edge | **FORBIDDEN** until Cloudflare Access + scoped token | — |

Default bind: `127.0.0.1:8787` (matches Frankenstein mission MCP note).

---

## Tools (MCP `tools/list`)

| Tool | Maps to | Args | Returns |
|------|---------|------|---------|
| `prometheus_run_task` | `Orchestrator.run_task` | `task`, `tier?`, `effort?`, `max_tool_calls?`, `max_cost_usd_cents?`, `tools?`, `verify?` | status, output, budget_status, verification |
| `prometheus_delegate_specialist` | `Orchestrator.delegate_to_specialist` | `task`, `specialist`∈{explorer,reviewer,tester,designer}, `briefing`, `tier?`, `effort?`, `max_cost_usd_cents?` | specialist result + budget |
| `prometheus_pool_metrics` | `get_pool_metrics()` | none | pool hit rate, size, tier/effort/spec dist |
| `prometheus_budget_status` | `AuthzEngine.get_delegation_budget_status` | `subagent_id` | limits/spent/remaining |
| `prometheus_list_specialists` | static | none | allowlists + default tier/effort |
| `prometheus_health` | local | none | version, mock/live mode, jlens on/off |

### Denied by default (not exposed v0.1)
- raw shell / network without orchestrator gates
- authz policy mutation
- kill-switch flip (Chair-only via Boardroom)

---

## Resources (MCP `resources/list`)
| URI | Content |
|-----|---------|
| `prometheus://status` | version tag, git sha, test summary |
| `prometheus://specialists` | SpecialistConfig dump |
| `prometheus://pool/metrics` | live JSON metrics |
| `prometheus://docs/run` | RUN.md |
| `prometheus://docs/release` | RELEASE_NOTES_v0.2.0.md |

---

## Prompts (MCP `prompts/list`)
| Prompt | Purpose |
|--------|---------|
| `delegate_explorer` | research-only briefing template |
| `delegate_reviewer` | code review + security audit template |
| `delegate_tester` | pytest/coverage template |
| `delegate_designer` | schema/architecture template |
| `war_room_status` | condensed sprint gate report |

---

## Implementation Plan (Forge)
1. Package `prometheus/mcp_server.py` using existing MCP patterns from Frankenstein revival
2. Wire `ProviderFactory(force_mock=...)` from env `PROMETHEUS_LLM_PROVIDER`
3. Register tools above; refuse unknown tools
4. Smoke test via `mcporter` / native MCP client from Hermie
5. Board agent client test: Sue → `prometheus_run_task` dry-run

## Acceptance Gate
- [ ] `prometheus_health` returns v0.2.0+
- [ ] `prometheus_run_task` works with `PROMETHEUS_LLM_PROVIDER=mock`
- [ ] Specialist delegate respects tool allowlists
- [ ] Unauthenticated remote bind refused
- [ ] Metrics resource matches pool endpoint JSON shape

---

## Out of Scope (Sprint 5+)
- Dual-LLM taint gate via MCP
- Real J-Lens fitted gate (needs torch)
- Cloudflare Container public MCP
