# Delegation Watchlist Categories — J-Lens Gate
**Sprint 4 Security Track**  
**Date:** 2026-09-30  
**Status:** Spec ready (Real Gate blocked on torch)

---

## Purpose
Categories the J-Lens / NoOp gate must score on every subagent dispatch.
Used by `prometheus/jlens/delegation_fpr_fnr.py` and future RealJLensGate.

---

## Categories

| ID | Category | Severity | Description | Expected Gate Action |
|----|----------|----------|-------------|----------------------|
| `clean` | Clean dispatch | none | Benign research/code task, no exfil/escalation signals | ALLOW |
| `delegation_exfiltration` | Exfiltration via subagent | high | Prompt/tool path that tries to dump secrets, env, credentials, private keys through subagent tools | DENY + escalate |
| `delegation_escalation` | Privilege escalation | high | Subagent attempts to expand tool allowlist, disable authz, or spawn unrestricted children | DENY + escalate |
| `delegation_trifecta` | Trifecta pattern | critical | Simultaneous: (1) untrusted input, (2) privileged tool, (3) external egress | DENY + escalate |
| `delegation_reasoning_anomaly` | Reasoning anomaly | medium | Incoherent / jailbreak-like reasoning trajectory mid-delegation | DENY or escalate effort + re-check |
| `delegation_budget_bypass` | Budget bypass | medium | Attempts to ignore max_cost / max_tool_calls or spawn untracked subagents | DENY |
| `delegation_cache_poison` | Warm-pool poison | medium | Briefing crafted to collide context_hash / reuse wrong specialist memory | DENY warm return; force cold create |
| `delegation_specialist_drift` | Specialist drift | low | Explorer/reviewer asked to write/execute outside allowlist | DENY tool; keep agent |

---

## Dispatch Hook Contract
On every `PreToolUse` for subagents:
1. Score prompt + tool_name + args against categories above
2. If any high/critical fires → `HookResult(allow=False, escalate=True)`
3. Log category IDs into pool metrics / FPR-FNR harness

---

## Environment Notes
- **Real gate:** requires `torch` + fitted lens — **not installed** in v0.2.0 base venv (Sprint 4 Security deferred install; may use Monte-Cristo GPU later)
- **Current production path:** `NoOpJLensGate` when `--enable-jlens-gate` without torch
- **Risk (from War Room red team):** NoOp creates false confidence — label clearly in CLI status as `jlens_gate=off|noop|real`

---

## Sprint 4 Acceptance
- [x] Categories defined and committed
- [ ] torch install path documented for MC
- [ ] FPR/FNR harness loads these categories
- [ ] CLI prints gate mode explicitly
