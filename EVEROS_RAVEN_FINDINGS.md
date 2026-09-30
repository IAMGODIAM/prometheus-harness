# EverOS + Raven Findings for Prometheus-OS
**Compiled:** 2026-09-29  
**Source Reviews:** EverOS repo review + Raven deep dive (Chinese tech coverage translation)  
**Target Repo:** `/home/user/src/prom/repo/` (Prometheus-OS)

---

## Executive Summary

| Component | What It Is | Strategic Value for Prometheus |
|-----------|------------|--------------------------------|
| **EverOS** | Local-first, Markdown-native memory runtime with dual-track memory, hybrid retrieval, offline Reflection | **Missing memory layer** — solves Prometheus' lack of cross-session persistence, skill distillation, evolution |
| **Raven** | Self-improving agent harness built on EverOS; HarnessBank evolution engine + Harness of Harnesses orchestration | **Missing evolution engine** — provides verified RSI at Harness layer (not model weights) |
| **EverMe** | Consumer product supplying longitudinal user trajectory data flywheel | **Missing data flywheel** — real user data for sustainable self-improvement |

**Convergence Thesis:** EverOS = memory, Raven = evolution runtime, EverMe = data. Prometheus = security + interpretability + edge deployment. **All four needed for complete stack.**

---

## 1. EverOS Findings (from `https://github.com/EverMind-AI/EverOS`)

### Core Architecture
- **Storage:** Markdown (source of truth) + SQLite (state/audit/queue) + LanceDB (vector + BM25 + scalar)
- **Dual-Track Memory:**
  - **User Track:** Profile + Episodes (identity, preferences, skills, goals)
  - **Agent Track:** Cases + Skills (distilled trajectories, reusable capabilities)
- **Retrieval:** Hybrid (keyword + vector + rerank), orthogonal by `user_id`/`agent_id`/`app_id`/`project_id`/`session_id`
- **Evolution:** Offline "Reflection" — merges episode clusters, refines profiles/skills between sessions
- **Knowledge Wiki:** Editable Markdown pages with taxonomy, CRUD APIs, topic search
- **Multimodal:** Images, PDF, audio, Office docs via `everalgo-parser` (LibreOffice for Office)
- **Benchmarks:** EverMemBench (factual recall/reasoning/personalization), EvoAgentBench (longitudinal self-evolution)

### Hermes Integration (Production-Ready)
- **Plugin:** `EverMind-AI/plugins/hermes` — Native `MemoryProvider` (single-select via `memory.provider: "everos"`)
- **Transport:** HTTP to local EverOS server (`127.0.0.1:8000`) — **not in-process import**
- **Pipeline:**
  - `prefetch(query)` → concurrent `/search` on user + agent tracks → fenced "untrusted historical data" block
  - `sync_turn` → `/add` (full turn incl. tool calls) in daemon thread (non-blocking)
  - `on_session_end` / `on_pre_compress` / `on_session_switch` / `shutdown` → `/flush`
  - `on_memory_write` → mirrors built-in `MEMORY.md`/`USER.md` edits to `/add`
  - `on_delegation` → captures subagent task+result as distilled trajectories
- **Provisioning:** Detect-then-start EverOS in `initialize()` (daemon thread, fail-open)
- **Config:** `hermes memory setup` → `$HERMES_HOME/everos.json`
- **No Tools:** Context-mode only — zero model-callable memory tools

### Installation
```bash
uv pip install everos
everos init          # creates ~/.everos/everos.toml (add OpenRouter key)
everos server start  # runs on 127.0.0.1:8000
```

---

## 2. Raven Findings (Self-Improving Agent Harness v0.2.0)

### Core Thesis
> **"Same model, different Harness → vastly different performance."**  
> Model weights frozen; Harness (prompt + knowledge + tools + runtime + config) evolves.  
> Stack: **EverOS (memory) → HarnessBank (verified method) → Raven (runtime) → EverMe (data)**

### HarnessBank: Verified Evolution Engine
| Component | Role |
|-----------|------|
| **Task Agent** | Executes under current Harness (frozen backbone, default Qwen3.6-27B) |
| **Evolver Agent** | Stronger model (e.g., Claude Opus 4.8) — reads trajectories, diagnoses failures, generates Harness candidates |
| **HarnessBank Archive** | MAP-Elites: x-axis = what to change (prompt/knowledge/runtime/config), y-axis = why (failure pathology) |
| **4-Gate Screening** | 1) Significance test 2) Ablation 3) Infrastructure fault detection 4) Test-set holdout validation |

**Results:** Qwen3.6-27B on AppWorld: **41.3% → 56.7% Pass@1** (Harness evolution only). 7 benchmarks: **+5.1 to +15.4 pp**.

### Raven Architecture: "Harness of Harnesses"

| Layer | Responsibility |
|-------|----------------|
| **Outer (Raven)** | Member selection, task DAG decomposition, dependency scheduling, result handoff, cross-model/framework orchestration |
| **Inner (Members)** | Each agent runs own Harness — Claude Code, Codex, Raven-Research, Raven-Code, Raven-Design, Raven-Oncall |

**4 Native Specialized Sub-Agents:**
- **Raven-Research** — Deep research (MiroThinker multi-source investigation)
- **Raven-Code** — Coding/data analysis
- **Raven-Design** — Design delivery
- **Raven-Oncall** — Long-running execution/monitoring

**Integration:** ACP, CLI, OpenAI-compatible API for third-party members.

### RSI at Harness Layer (4 Mutable Policy Surfaces)
1. **Modules** — Playbook + sub-Harness composition
2. **Code** — Execution-check strategies, runtime logic
3. **Prompt** — System prompts, onboarding manuals
4. **Policy** — Tool gates, flash configurations

**Curator (Experimental v0.2.0):** Reads current Harness + task requirements + execution traces + user feedback → generates diffs → declaration check → real assembly + preflight run → install (with rollback). All changes land on extension points; core source untouched.

### Digital Life Ladder (EverMind Staging)
| Stage | Description | Status |
|-------|-------------|--------|
| L1 | Role-playing, no memory | Most wrapper AIs |
| L2 | Memory-augmented, basic planning | RAG + long-context |
| **L3** | **Self-evolving — summarizes experience, rewrites code, improves skills** | **Raven (target)** |
| L4 | Fully autonomous digital life | Future |

### Key Metrics
| Metric | Value |
|--------|-------|
| Built-in skills | 100,000+ (evaluated, continuous eval/add/remove/recombine) |
| Token efficiency | 1/10 traditional for >Full Context accuracy |
| Memory type | Bidirectional — internalization, not just retrieval |
| Skill evolution | Continuous: evaluate → retire failing → generate new combos |
| Code rewriting | Skills, runtime logic, policies, **dynamic weight tuning via EverBrain** |

---

## 3. EverMe: Consumer Data Flywheel

| Module | Function |
|--------|----------|
| **Memory Hub** | 11 agents connected (Claude Code, Codex, OpenClaw, Hermes, Kimi Code, Raven...); cold-start import + real-time write; cross-agent Timeline |
| **Knowledge Base** | Notion, Obsidian, local files → personal KB; Deep Research with private data |
| **Digital Twin** | Primary avatar, onboarding, chat flow, memory provenance (Reference) |
| **Agent Hub** | Agent sharing, discovery, rating, collaboration |

**Strategic Role:** Supplies *continuous, real, user-owned, longitudinal trajectory data* — the missing piece for sustainable self-evolution.

---

## 4. Three-Layer Product Loop

```
Academic Research (HarnessBank) → "HOW to evolve safely"
       ↓
Runtime Harness (Raven)       → "WHERE evolution runs continuously"
       ↓
Consumer Product (EverMe)     → "WHAT data evolves on" (real user trajectories)
```

---

## 5. Strategic Comparison: EverOS/Raven vs. Prometheus-OS

| Dimension | **EverOS + Raven** | **Prometheus-OS (Local)** |
|-----------|-------------------|---------------------------|
| **Primary Goal** | Long-term memory + self-evolution across agents | J-Lens interpretability + secure agentic harness |
| **Memory Model** | Dual-track (user episodes/profile + agent cases/skills) | Letta blocks + Obsidian vault (single namespace) |
| **Storage** | Markdown + SQLite + LanceDB | Obsidian vault (Markdown) + Letta blocks |
| **Retrieval** | Hybrid BM25+vector+rerank, orthogonal scopes | Vault search (basic) |
| **Evolution** | Offline Reflection (merges clusters, refines skills) | Not implemented |
| **Agent Track** | First-class cases + skills distillation | Not present |
| **Harness RSI** | HarnessBank (gated, verified) + Curator (AI-rewritable) | Not implemented |
| **Orchestration** | Harness of Harnesses (multi-agent coordination) | Single-agent orchestrator (Phase 1) |
| **Integrations** | Hermes, OpenClaw, DSH, Dify, Claude Code, Codex | Hermes MCP only (Phase 1) |
| **Multimodal** | Images, PDF, audio, Office (via everalgo-parser) | Not implemented |
| **Benchmarks** | EverMemBench, EvoAgentBench (longitudinal) | None |
| **Security** | Local-first, extraction calls your LLM/embedding | **Dual-LLM CaMeL, Cedar AuthZ, Lethal Trifecta, J-Lens gating** |
| **Observability** | OTel/Langfuse hooks in EverOS | **OTel/Langfuse in Prometheus** |
| **Deployment** | Local server + optional cloud LLM | **Local MCP + Cloudflare Workers edge** |
| **Architecture** | Python library + HTTP API server | **Python package + MCP stdio/HTTP** |
| **Verification** | Not implemented | **5-stage async verification pipeline** |

---

## 6. Convergence Path: Prometheus + EverOS + Raven

### Phase 1: Memory Upgrade (Prometheus ← EverOS)
- [ ] Replace Letta blocks + Obsidian vault with **EverOS as memory backend**
- [ ] Install `EverMind-AI/plugins/hermes` via `hermes plugins install`
- [ ] Run `hermes memory setup` → point to local EverOS server
- [ ] Configure EverOS `mode = "agent"` for dual-track (cases + skills)
- [ ] Add embedding/rerank providers to EverOS for hybrid search
- [ ] Gain: cross-session persistence, dual-track memory, offline Reflection, skill distillation

### Phase 2: Interpretability Overlay (EverOS → Prometheus)
- [ ] Expose Prometheus J-Lens tools via MCP to EverOS-backed agents
- [ ] Add J-Lens watchlist gating to EverOS extraction pipeline
- [ ] Security: J-Lens concept scores → EverOS reflection filter
- [ ] Gain: real-time model internals probing for evolving agents

### Phase 3: Unified Agent Harness (Prometheus + Raven)
- [ ] Single agent runtime: **EverOS memory + Prometheus security + J-Lens observability**
- [ ] MCP server exposes both memory tools (`search`, `add`, `flush`) and interpretability tools (`probe`, `watchlist`, `decompose`, `steer`)
- [ ] Add HarnessBank-style gated evolution to Prometheus orchestrator
- [ ] Build "Harness of Harnesses" layer — Prometheus coordinates specialized sub-agents (Research, Code, Design, Oncall equivalents)
- [ ] Edge deployment via Cloudflare Workers for both
- [ ] Instrument for EverMemBench / EvoAgentBench — measure if harness actually improves over time

---

## 7. Immediate Action Items

| Priority | Action | Owner |
|----------|--------|-------|
| **High** | Install `EverMind-AI/plugins/hermes` via `hermes plugins install` | Hermie |
| **High** | Run `hermes memory setup` → configure local EverOS server | Hermie |
| **High** | Configure EverOS `mode = "agent"` for dual-track | Hermie |
| **Medium** | Add embedding/rerank providers to EverOS `everos.toml` | Hermie |
| **Medium** | Expose Prometheus MCP tools to EverOS agents | Prometheus team |
| **Medium** | Evaluate Raven harness for self-improving agent loop | Research |
| **Low** | Benchmark with EverMemBench / EvoAgentBench | Research |
| **Low** | Build Harness of Harnesses orchestration layer | Prometheus team |

---

## 8. Original Prometheus-OS Context (for Reference)

### Prometheus-OS Repo: `/home/user/src/prom/repo/`
- **Phase 1 Status:** J-Lens fitting/probing, MCP server, basic harness ✓
- **72 tests passing**, `JacobianLens` reimplemented from method guide
- **Capabilities:** `jlens_probe`, `jlens_watchlist_scores`, `jlens_decompose`, `verifier_check`, `harness_status`
- **Security:** Dual-LLM CaMeL gate + Cedar AuthZ + Lethal Trifecta defense
- **Deployment:** Cloudflare Workers edge ready
- **Config:** `prometheus.yaml` (model: gpt2, device: cpu, watchlist categories defined)

### Sprint 1 Status (Completed 2026-09-29)
- ✅ Prometheus MCP server (stdio) running
- ✅ Hermes MCP discovery: 3 tools discovered
- ✅ Gateway restart → tools injected into Telegram session
- ✅ End-to-end test: `POST /v1/runs` with `toolsets: ["mcp"]` → J-lens probe executed successfully
- ⏳ Telegram slash commands (`/probe`, `/watchlist`, `/verify`) — pending bot token from vault

### Key Files
- `/home/user/src/prom/repo/README.md` — Full architecture docs
- `/home/user/src/prom/repo/AGENT_README.md` — Agent-friendly documentation
- `/home/user/src/prom/repo/prometheus.yaml` — Configuration
- `/home/user/hermes-workspace/prometheus-harness/` — Full harness with tests

---

## 9. References & Sources

| Source | Type | Accessed |
|--------|------|----------|
| `https://github.com/EverMind-AI/EverOS` | GitHub repo | 2026-09-29 |
| `https://github.com/EverMind-AI/plugins/tree/main/hermes` | Hermes plugin | 2026-09-29 |
| `https://github.com/EverMind-AI/raven` | Raven repo | 2026-09-29 |
| `https://raven.evermind.ai/` | Raven website | 2026-09-29 |
| `https://finance.sina.com.cn/tech/roll/2026-08-11/doc-inimxewt4365552.shtml` | Sina Tech (Chinese) | 2026-09-29 |
| `http://bbs.scitoday.cn/htmlnews/2026/7/63293.shtm` | Science Today (Chinese) | 2026-09-29 |
| `https://www.gushiio.com/ai/28471.html` | Gushiio (Chinese) | 2026-09-29 |
| `https://www.donews.com/news/detail/4/6626000.html` | DoNews (Chinese) | 2026-09-29 |
| `http://damoai.com.cn/archives/16833` | Damai AI (Chinese) | 2026-09-29 |
| `http://www.cctime.com/html/2026-8-12/1740663.htm` | CCTIME (Chinese) | 2026-09-29 |

---

## 9. Next Steps for Other Agents

This document is now committed to the Prometheus-OS repo at `/home/user/src/prom/repo/EVEROS_RAVEN_FINDINGS.md`. All agents with repo access can reference it.

**For Sue:** Use this as briefing for EverOS plugin installation and memory backend migration tasks.

**For Forge:** Reference for training/evaluation pipeline integration with EverMemBench/EvoAgentBench.

**For Scout:** Monitor EverMind ecosystem updates (EverOS, Raven, EverMe, HyperMem, MSA).

**For Draco:** Infrastructure planning for EverOS server deployment + Prometheus MCP edge integration.

---

*End of findings document.*