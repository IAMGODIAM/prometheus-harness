# RIG_NOTES — what was wired, what's still stubbed, and how to go live

Branch: `rig/nvidia-runnable` · Base: `main` @ `b0f5d84` (Prometheus v0.1.0)
Scope: the P0/P1 items from the code-verified review — make the harness actually
run, wired to NVIDIA inference, safe by default. Integrity rule kept: **only wired
what exists; where a stage is a stub, the run is made safe despite it and flagged —
no faked capability.**

## What I added / changed

New files:
- `prometheus/harness/model_client.py` — `ModelProvider` abstraction + `OpenAICompatibleProvider` (NVIDIA hosted default, local-NIM swap via one env var, key from env — never hardcoded) + deterministic `MockProvider` + `build_provider_from_env()`.
- `prometheus/harness/tools.py` — `SafeToolRouter` + `build_default_toolset()`: two pure in-process tools (`echo`, `get_time`) by default; opt-in `read_file` (path-restricted) / `http_get`; dry-run simulation; per-tool authz category + dual-LLM taint declarations.
- `prometheus/harness/jlens_gate.py` — `NoOpJLensGate` (default, no torch) + `RealJLensGate` (optional) + `load_jlens_gate()` that falls back to no-op rather than faking.
- `prometheus/harness/runner.py` — assembles provider + tools + deny-by-default authz + dual-LLM gate + verifier + tracer + logger + J-lens gate and drives `Orchestrator.run_task`.
- `tests/test_run_smoke.py` — offline end-to-end smoke + safety assertions.
- `.env.example`, `RUN.md`, this file.

Edited files:
- `prometheus/harness/orchestrator.py` — `Orchestrator`/`AgentLoop` now accept optional `dual_llm_gate`, `tracer`, `logger`; the loop attaches the dual-LLM gate as a PreToolUse hook, records LLM + tool spans, and emits structured tool/authz logs. `run_task` now forwards `max_tool_calls`/`max_runtime_seconds`. **All new params default to None/off, so the existing 45 tests pass unchanged.**
- `prometheus/cli.py` — new `prometheus run "<task>"` subcommand with safety flags.
- `pyproject.toml` — moved the heavy ML stack (torch/transformers/numpy/scipy/safetensors) out of core deps into an optional `[jlens]` extra, so `pip install -e .` is torch-free and the agent runs on a plain machine; also relaxed `requires-python` from `>=3.11` to `>=3.10` (see the flagged Compatibility note below).

**Compatibility note (flagged change):** the base repo set `requires-python = ">=3.11"`, which makes `pip install -e .` fail on stock WSL2 Ubuntu 22.04 (Python 3.10). The full test suite (49 tests) and the end-to-end CLI smoke run **pass on Python 3.10.12**, so the code is 3.10-compatible in practice. I relaxed the gate to `>=3.10` so the founder's machine installs cleanly. If you standardize on Python 3.11+, revert that one line. This is an author-set constraint I changed deliberately — flagging it rather than smoothing it over.

## Verified in-sandbox (offline, no NVIDIA key used)

- `pytest tests/test_harness.py tests/test_authz_memory.py tests/test_run_smoke.py` → **49 passed** (the original 45 + 4 new). No regressions.
- `prometheus run "..." --provider mock` → `status: completion_claimed`, `verification: ACCEPT (1.0)`, gated `echo` executed, 3 trace spans, safe posture reported.
- Clean-room: fresh unzip → `pip install -e .` (torch-free) → smoke green (see the "Clean-room" section at the bottom / delivery log).

I did **not** use any real NVIDIA key (we don't have one and shouldn't). The hosted
path is exercised structurally via the mock; the live NVIDIA call is a standard
OpenAI-compatible POST and is covered by RUN.md for you to run with your own key.

## What is now WIRED (was unwired before)

| Review gap | Status now | Evidence |
|---|---|---|
| No LLM client shipped | Wired | `model_client.py`; orchestrator's `_call_llm` now has a real client |
| No CLI command runs an agent | Wired | `cli.py` `run` subcommand → `runner.run_task` |
| Tracer not invoked by loop | Wired | `orchestrator.AgentLoop.run` records LLM + tool spans when a tracer is attached |
| StructuredLogger not invoked | Wired | loop emits `tool_call` / `authz_decision` structured logs |
| DualLLMGate not in the loop | Wired | attached as a PreToolUse hook in `spawn_agent` |
| `jlens_gate` hook had no class | Resolved | `NoOpJLensGate` default (runs without torch); pluggable real gate |
| torch mandatory | Fixed | `[jlens]` optional extra |

## What is still STUBBED / not done (and the safety implication)

1. **Per-tool execution sandbox — still a stub** (`verifier.sandbox` / no process isolation). *Implication:* with `--no-dry-run --allow-fs/--allow-net`, tools run unsandboxed in your WSL process. *Mitigation shipped:* dry-run is the default; every call is gated by deny-by-default authz + dual-LLM trifecta gate; `read_file` is path-restricted; network tools stay denied until `--allow-net` adds an explicit permit; budgets cap volume. **Do not run untrusted inputs with `--no-dry-run`.** A real sandbox (Firecracker/E2B/bubblewrap) is the next step.
2. **Real J-lens gate — wired but UNVERIFIED.** `RealJLensGate` exists and `load_jlens_gate(enable=True)` is the entry point, but it will **not fabricate scoring**: without torch + a fitted lens + a scorer it logs a warning and returns the no-op gate. Live concept-gating of the running agent is therefore not yet active. *Implication:* the safety watchlist does not block tool calls yet; authz + dual-LLM do.
3. **Verifier stages** `execution` and `differential` remain stubs, and `position-swap` bias mitigation is still not implemented (carried over from the review — not in scope here). The verifier still runs deterministic + checklist + rubric checks.
4. **Verifier shares the actor's model in the runner.** For the mock/smoke that's fine, but in production the verifier SHOULD use a *different* model family (model routing). Set that up before trusting ACCEPT verdicts. See "Model routing" below.
5. **"Facund ModelProvider pattern"** — I did not have that reference available, so I implemented a conventional provider abstraction (`ModelProvider` + `OpenAICompatibleProvider`/`MockProvider`). If Facund is an internal interface, renaming/aligning `model_client.py` to it is mechanical.
6. **Tool schemas are minimal.** The harness tracks tool *names*; the runner injects real JSON schemas into the provider, but hosted-model tool-calling quality still depends on the model. This is the one-call-at-a-time limitation from the review (no code-mode); unchanged.

## Go-live commands on Monte Cristo (with YOUR key)

```bash
cd ~/src/prometheus-harness && source .venv/bin/activate
export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx          # your key, shell only
prometheus run "Draft a 3-bullet status update on the rig." --provider openai
# ^ safe: dry-run ON, tools simulated, NVIDIA used for plan/verify.
```

Model routing (recommended before real use): run a second, different model as the
verifier — e.g. set the actor model in `.env` and construct the verifier with a
different `OpenAICompatibleProvider(model_id=...)`. A one-line hook for this lives in
`runner.py` where `AsyncVerifier(verifier_llm_client=provider)` is built.

Local NIM swap (if the WSL `nvidia-smi` GPU check passes — see RUN.md §7):

```bash
export PROMETHEUS_LLM_BASE_URL=http://localhost:8000/v1
prometheus run "Say hello via a safe tool." --provider openai
```

## Enabling the real J-lens gate (when you want it)

1. `pip install -e '.[jlens]'` (installs torch/transformers).
2. `prometheus fit -m <model> -o lens.pt` to fit a Jacobian lens.
3. Construct a scorer + `RealJLensGate(scorer, Watchlist(), tokenizer)` and pass it to
   `Orchestrator(jlens_gate=...)` (or extend `load_jlens_gate` to build it from a lens
   path). Until then `--enable-jlens-gate` safely no-ops with a warning.
