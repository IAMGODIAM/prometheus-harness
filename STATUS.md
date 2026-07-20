# Prometheus Harness — Live Status

**Date:** 2026-07-20  
**Operator:** Hermie (Dr. Frankenstein commission)  
**Version:** 0.1.0 (patched this session)

## Verdict: FUNCTIONAL (smoke + unit) — PRODUCTION LENS NEEDS DEEPER FIT

| Gate | Result | Evidence |
|------|--------|----------|
| Install (WSL venv) | PASS | `hermes-workspace/prometheus-harness/.venv` torch 2.13.0+cpu |
| Unit tests | **72/72 PASS** | `pytest tests/ -q` → 1.29s |
| CLI | PASS | `fit|probe|watchlist|decompose|serve|status` |
| MCP stdio init | PASS | `serverInfo.name=prometheus-jlens`, tools×3 |
| MCP HTTP | PASS | `http://127.0.0.1:8787/mcp` streamable-http |
| Lens fit (gpt2 smoke) | PASS | 1 prompt, 4 mid-band layers, ~1.0s → `.prometheus/cache/gpt2_jlens.pt` |
| Probe | PASS | JSON top-k per layer/pos |
| Watchlist | PASS | scores + thresholds (no false alert on 1-sample lens) |
| Decompose | PASS path | occupancy 0 on underfit smoke lens (expected) |
| Dual-LLM / AuthZ / Verifier / Memory modules | PASS import+API smoke | unit tests cover |
| Mesh fleet exposure | LOCAL ONLY | bound 127.0.0.1:8787 (not Tailscale/public yet) |
| GitHub | private repos exist | `IAMGODIAM/prometheus`, `IAMGODIAM/prometheus-harness` |

## Bugs fixed this session (Frankenstein patches)

1. **`JLensFitter` phantom class** — CLI imported non-existent class; wired to `fit_jacobian_lens`.
2. **`device_map=cpu` required accelerate** — load via `.to(device)` on CPU without accelerate.
3. **Fit corpus shorter than `skip_first_n=16`** — 0 prompts processed; elongated seed corpus.
4. **CLI watchlist/decompose were stubs** — implemented against real Watchlist + JSpaceDecomposer APIs.
5. **Default fit source layers** — full-layer Jacobian on CPU is O(heavy); mid-band default for operable smoke.

## Runtime paths

| Role | Path |
|------|------|
| Canonical Linux workspace | `/home/user/hermes-workspace/prometheus-harness` |
| Windows original | `C:\Users\User\Downloads\Jacobian Lens and J-Space Workspace Method Guide\prometheus-harness` |
| Dropbox mirror | same under Dropbox |
| Review (external, private-blocked) | `iCloudDrive\Prometheus_Review\` |
| GitHub | https://github.com/IAMGODIAM/prometheus-harness |

## MCP tools live

- `jlens_probe`
- `jlens_watchlist_scores`
- `jlens_decompose`
- `jlens_steer` — feature-flagged OFF (`enable_steering: false`)

## Honest limits

- Smoke lens is **not** paper-grade (n_prompts=1). Numerics will be noisy until n≥64–512.
- `lens.py` was reconstructed from tests (original 0-byte); trust tests + method guide, re-validate vs paper before safety claims.
- Not yet Hermie-native MCP config entry; not mesh-routed.
- Full agent loop (orchestrator + dual-LLM + verifier) unit-tested; not dogfooded as Hermie's outer loop.

## Verify commands

```bash
cd /home/user/hermes-workspace/prometheus-harness
source .venv/bin/activate
pytest tests/ -q
prometheus status
prometheus serve --transport stdio
# or
prometheus serve --transport http --host 127.0.0.1 --port 8787
```
