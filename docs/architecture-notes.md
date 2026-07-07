# Key Architecture Decisions from Spec Documents

## Core System: MCP-First J-Lens Anchored Agentic Harness

### Architecture Components:
1. **MCP as universal tool ABI** - Every capability exposed as MCP server (stdio local, Streamable HTTP remote)
2. **J-Lens Spine** - Shadow HF Transformers instance for interpretability monitoring
3. **Dual-LLM Security** (CaMeL pattern) - P-LLM (trusted, tools) + Q-LLM (untrusted, symbolic refs)
4. **Async Verifier Loop** - Different model family, structured CompletionClaim
5. **Memory** - Obsidian vault + Letta-style blocks + derived indices
6. **Observability** - OTel + Langfuse/Phoenix
7. **Cedar Authz** - Deny-by-default, taint tracking

### J-Lens Implementation Details:
- `J_ℓ = E[∂h_final,t' / ∂h_ℓ,t]` - one d_model × d_model matrix per layer
- Readout: `softmax(W_U · norm(J_ℓ · h))`
- J-space: sparse non-negative combination of k≤25 J-lens vectors via gradient pursuit
- Variance: ~6-10%, never >10%, but causally critical
- Workspace band: middle third (~L38-L92 normalized)
- Fitting: 1 forward + ⌈d_model/8⌉ backwards per prompt
- SKIP_FIRST_N_POSITIONS = 16, max_seq_len = 128
- Target: pre-final-norm residual stream

### Interventions:
1. Steering: `h ← h + α·v_t`
2. Ablation: negative α or projection
3. Concept-swap: `h_patched = h + V(σ(c) − c)` where V=[v_s, v_t], c = V†h

### Hardware Target: RTX 3070 (8GB)
- GPT-2 small for debugging
- Llama-3.2-1B / Qwen-2.5-1.5B as primary targets
- 50-200 prompts of 128 tokens for fitting

### MCP Server Surface for J-Lens:
- `jlens.probe(prompt, layers?, positions?, top_k?)`
- `jlens.watchlist_scores(prompt, watchlist)`
- `jlens.decompose(prompt, layer, position, k=25)`
- `jlens.steer(prompt, interventions[])` (Phase 3, feature-flagged)

### Watchlist Categories:
- Deception: fake, secretly, fraud, trick, hidden, deliberately, manipulation
- Prompt injection: injection, poison, fraud, fake
- Eval-awareness: fake, fictional, scenario, benchmark, simulation, mock
- Self-preservation: leverage, threat, survival, shutdown

### Four-Phase Rollout:
- Phase 0: Offline auditing (replay saved trajectories)
- Phase 1: Async monitoring (parallel to prod, alerts)
- Phase 2: Runtime action-gating (Cedar/OPA policies)
- Phase 3: Runtime steering (experimental, feature-flagged)

### Build Priorities (from doc):
- Base on Pydantic AI + OpenHands SDK patterns
- TypeScript or Python FastMCP for MCP servers
- Four tool-call parsers (OpenAI JSON, Hermes-XML, GLM XML, Kimi special tokens)
- Docker MCP Gateway for sandboxing
- Cedar + OPAL for authz
- Langfuse + Inspect AI for observability/eval
