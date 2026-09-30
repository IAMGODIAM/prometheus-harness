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
