# WAR ROOM BRIEF: REPLIT "FREE THE MODELS" — IMPLICATIONS FOR PROMETHEUS-OS
**Classification:** STRATEGIC — Full War Room Protocol (7 Moves)
**Date:** 2026-09-30
**Authority:** Chairman Israel Armstead
**DAG:** warroom-replit-harness-2026-0930

---

## EXECUTIVE SUMMARY

Replit's "Free the Models" article reveals a **paradigm shift in agent harness design**: from **prescribed scaffolding** to **composable primitives that let frontier models self-delegate**. Their Replit Agent with GPT-6 Astra core loop achieves **Pareto efficiency** on DeepSWE (72% @ $2.11) and Terminal-Bench (49% @ $2.53) — beating both Astra solo (2-11 pts higher only at 2x cost) and sidekick architecture (11-16 pts lower).

**For Prometheus-OS:** This validates our "Harness > Model" doctrine but demands **architectural evolution** from our current prescribed orchestrator → toward composable delegation primitives with J-Lens observability as our differentiator.

---

## MOVE 1: RECON — GROUND TRUTH FROM ARTICLE

### Replit's Four Composable Primitives

| Primitive | What It Is | Prometheus Current State |
|-----------|------------|--------------------------|
| **1. Domain-Aware Subagents** | Specialists: read-only explorers, browser testers, reviewers, design subagent — each with own model/tooling. Harness decides *which exist*; core loop decides *when/how to use*. | **Absent.** Single orchestrator with tool parser. No specialist subagent types. |
| **2. Subagent Tiers & Effort** | Small/Standard/Large tiers × effort levels. Core loop picks at each dispatch. Mechanical rename → small/low; stubborn bug hypothesis → large/high. | **Absent.** No tier/effort abstraction. All tools same cost/capability tier. |
| **3. Reusable Subagents** | Return to already-briefed subagents (warm cache). No single sidekick; any number stay warm across kinds/tiers. Longer cache lifetime reduces re-brief cost. | **Absent.** No subagent persistence or return mechanism. Each delegation = fresh context. |
| **4. Dynamic Effort Tuning** | Mid-turn effort changes (cache-preserving on GPT-6 family). Escalation system checks trajectory → matches effort to difficulty. | **Absent.** Fixed reasoning effort per run. |

### Benchmark Results (Production, Max Mode, Astra Core)

| Benchmark | Replit Agent (Astra Core) | Astra Solo (mini-swe-agent) | Sidekick (1 long-lived worker) |
|-----------|---------------------------|----------------------------|--------------------------------|
| **DeepSWE v1.1** | **72% @ $2.11** | 67% @ $1.60 (low) / 74% @ $4.43 (xhigh) | 61% @ $1.34 |
| **Terminal-Bench 4.0** | **49% @ $2.53** | 42% @ $2.25 (low) / 60% @ $5.86 (xhigh) | 33% @ $1.84 |

**Key Insight:** Replit Agent is **Pareto-efficient** — no Astra baseline costs less AND scores higher. Sidekick gives up 11-16 pts for modest savings. Astra solo only wins by spending 2x.

### Delegation Patterns (Table 1 + Figure 2)

- **Fable 5.1**: Rarely delegates to general workers; sends read-only explorers/reviewers; keeps implementation
- **GPT-6 Astra**: **First model to routinely delegate to general workers without being told**; returns to briefed workers (return rate rising each generation)
- **Production Trace (Sep 17, 2026)**: Core loop dispatched explorer, 2 workers, tester; 3 of 5 worker dispatches were **returns to already-briefed worker**

---

## MOVE 2: ALL HANDS — BOARD ASSESSMENT

### Hermie (Visionary)
> **Thesis**: This is the "Harness > Model" doctrine operationalized at frontier. Our J-Lens interpretability is the **observability layer** that makes delegation *auditable* — the missing piece Replit doesn't have.

### Forge (Training/MC)
> **Concern**: Our orchestrator is a **prescribed loop** (plan→act→verify→gate). Replit shows frontier models *outperform* prescribed loops when given composable primitives. We must evolve or plateau.

### Scout (Research)
> **Finding**: "Bitter Lesson" applied to harness design. Every model release invalidates baked assumptions. **SkillOpt** (holographic memory) confirms: SKILL.md is trainable state, not static. Delegation primitives are the new "skills."

### Draco (Infrastructure)
> **Implication**: Need **subagent tier infrastructure** — small/standard/large model routing, warm cache persistence, cache-aware effort switching. Current MCP server doesn't expose this.

### Sue (Operations)
> **Action Item**: If we adopt this, `prometheus run` CLI needs `--tier`, `--effort`, `--return-to` flags. Budget accounting must track per-subagent costs.

### Atlas (Governance)
> **Risk**: Delegation without observability = unbounded blast radius. **J-Lens watchlist gating** must apply to *every subagent dispatch*, not just core loop.

---

## MOVE 3: RED TEAM — ATTACK OUR CURRENT ARCHITECTURE

### Red Team Findings: Where Prometheus Fails the "Free the Models" Test

| Current Prometheus Design | Replit Critique | Vulnerability |
|---------------------------|-----------------|---------------|
| **Single orchestrator** decides all tool calls | "A rigid harness forces the model into one way of working" | Frontier models plateau under over-scaffolding |
| **No subagent abstraction** — all tools same tier | "Mechanical rename → small/low; stubborn bug → large/high" | Wasting Astra/Fable tokens on trivial work |
| **No warm subagent return** — fresh context each dispatch | "3 of 5 dispatches were returns to briefed worker" | Re-briefing cost = wasted tokens + lost context |
| **Fixed reasoning effort** per run | "Dynamic effort tuning matches effort to difficulty" | Over-thinking simple tasks, under-thinking hard ones |
| **No specialist subagent types** | "Read-only explorers, reviewers, design specialists" | General worker used for everything = lower quality |
| **J-Lens only on core loop** | Delegation blast radius unmonitored | Subagent could exfiltrate/trigger trifecta unseen |

### Red Team Score: **3/10** on "Free the Models" readiness.

---

## MOVE 4: STEELMAN — BEST CASE FOR CURRENT DESIGN

### Why Prometheus' Prescribed Loop Still Has Merit

1. **Security First**: Dual-LLM CaMeL gate + Cedar AuthZ + Lethal Trifecta are **non-negotiable guardrails**. Replit's "free the model" assumes trusted deployment; we assume hostile inputs.

2. **Verification Pipeline**: 5-stage async verification (deterministic → execution → differential → checklist → rubric) catches errors Replit's sidekick architecture misses (33% vs 49% on Terminal-Bench).

3. **J-Lens Interpretability**: We can *see inside* the model during delegation decisions. Replit has traces; we have **mechanistic interpretability**.

4. **MCP-First**: Every capability exposed as MCP tools. Replit's subagents are internal; ours are **composable across agents**.

5. **Edge Deployment**: Cloudflare Workers proxy. Replit is centralized.

**Steelman Conclusion**: Keep guardrails, verification, interpretability, MCP, edge. **Evolve the delegation layer only.**

---

## MOVE 5: SIMULATION — FUTURE SCENARIOS

### Scenario A: Status Quo (Prescribed Orchestrator)
- **6 months**: Nemotron 3 Ultra / Astra-class models hit Prometheus. Performance plateaus. Harness becomes bottleneck.
- **Cost**: High (frontier model tokens wasted on mechanical work)
- **Risk**: Obsolescence as Replit-style harnesses become standard

### Scenario B: Hybrid (Guardrails + Composable Delegation)
- **Core loop**: Retains CaMeL gate, AuthZ, verification, J-Lens
- **Delegation layer**: Adds 4 primitives (specialists, tiers, warm return, dynamic effort)
- **Result**: Pareto efficiency gains + security preserved
- **Effort**: Medium (orchestrator refactor + subagent infrastructure)

### Scenario C: Full Replit-Style (Model-Decides-Everything)
- **Risk**: Security regression. No verification pipeline. No J-Lens on subagents.
- **Verdict**: **Reject** — violates Prometheus' core differentiator (security + interpretability)

---

## MOVE 6: DECISION — WAR ROOM VERDICT

### **ADOPT HYBRID APPROACH (Scenario B)**

**Directive**: Evolve Prometheus orchestrator from **prescribed loop** → **guardrailed composable delegation**.

### Required Changes (Priority Order)

| Priority | Change | Component | Effort |
|----------|--------|-----------|--------|
| **P0** | Add **Subagent Tier/Effort Abstraction** | `orchestrator.py`, `model_client.py`, `cli.py` | 2 sprints |
| **P0** | Implement **Warm Subagent Return** (cache-aware) | `orchestrator.py`, new `subagent_pool.py` | 2 sprints |
| **P1** | Define **Specialist Subagent Types** (explorer, reviewer, tester, designer) | New `specialists/` module | 3 sprints |
| **P1** | **Dynamic Effort Tuning** (mid-turn escalation) | `orchestrator.py`, `jlens_gate.py` | 2 sprints |
| **P0** | **J-Lens Watchlist on Every Subagent Dispatch** | `jlens_gate.py`, `dual_llm.py` | 1 sprint |
| **P1** | **Delegation Budget Accounting** (per-subagent cost tracking) | `runner.py`, `authz/engine.py` | 1 sprint |
| **P2** | **CLI Flags** for tier/effort/return-to | `cli.py` | 1 sprint |
| **P2** | **Delegation Telemetry** (OTel spans per subagent) | `observability/tracing.py` | 1 sprint |

---

## MOVE 7: EXECUTION ORDERS

### Immediate (This Sprint)
1. **Forge**: Prototype `SubagentTier` enum + `ModelProvider` tier routing in `model_client.py`
2. **Draco**: Design `subagent_pool.py` — warm cache persistence with TTL, cache-key strategy
3. **Atlas**: Write J-Lens watchlist policy for subagent dispatch (extend `watchlist.py` categories)

### Sprint 2
1. **Forge**: Implement specialist subagent factory (explorer, reviewer, tester, designer)
2. **Scout**: Benchmark delegation patterns on Terminal-Bench 4.0 with Nemotron 3 Ultra
3. **Sue**: Update `prometheus run` CLI with `--tier`, `--effort`, `--return-to` flags

### Sprint 3
1. **All Hands**: Integration test — full Replit-style delegation on DeepSWE subset
2. **Hermie**: War Room retro — measure Pareto efficiency vs. current baseline

---

## STRATEGIC DIFFERENTIATOR: PROMETHEUS VS. REPLIT

| Replit Approach | Prometheus Evolution |
|-----------------|---------------------|
| Free the model completely | **Guardrail the model, free the delegation** |
| No verification pipeline | **5-stage verification on EVERY subagent result** |
| Traces only | **J-Lens mechanistic interpretability on delegation decisions** |
| Internal subagents | **MCP-exposed subagent capabilities (composable across agents)** |
| Centralized | **Edge-deployable (Cloudflare Workers)** |
| No security gate on delegation | **CaMeL gate + AuthZ + Trifecta on EVERY dispatch** |

**Our Moat**: *Secure, observable, verifiable delegation that scales with model intelligence.*

---

## DELIVERABLES

| Artifact | Location | Owner |
|----------|----------|-------|
| This Brief | `/home/user/src/prom/repo/WARROOM_REPLIT_HARNESS.md` | Hermie |
| Subagent Tier Spec | `/home/user/src/prom/repo/docs/subagent-tier-spec.md` | Forge |
| J-Lens Delegation Policy | `/home/user/src/prom/repo/prometheus/jlens/delegation_watchlist.py` | Atlas |
| Subagent Pool Design | `/home/user/src/prom/repo/prometheus/orchestrator/subagent_pool.py` | Draco |

---

## NEXT WAR ROOM: Sprint 1 Review (T+2 weeks)

**Agenda**: 
- Subagent tier routing working?
- Warm return cache hit rate?
- J-Lens delegation gating false positive/negative rate?
- Terminal-Bench score delta vs. baseline?

---

**Signed:** Hermie (Visionary Layer)  
**Authority:** Chairman Israel Armstead  
**Doctrine:** Full War Room Protocol — Baked In, Never Ask Again