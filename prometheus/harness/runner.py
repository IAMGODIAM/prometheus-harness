"""Runner — assemble and drive the agent loop end-to-end (RIG addition).

Wires the previously-unwired pieces into one runnable path:
provider + safe tools + deny-by-default authz + dual-LLM taint gate + verifier +
tracer + structured logger + (optional) J-lens gate, then runs a task through the
Orchestrator.

Safe-by-default posture: dry-run ON, authz deny-by-default, budget caps ON, a
conservative in-process tool allowlist, network tools deny-by-default unless the
operator explicitly opts in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from prometheus.config import HarnessConfig
from prometheus.harness.orchestrator import Orchestrator
from prometheus.harness.verifier import AsyncVerifier
from prometheus.harness.dual_llm import DualLLMGate
from prometheus.harness.model_client import ModelProvider, build_provider_from_env
from prometheus.harness.tools import build_default_toolset
from prometheus.harness.jlens_gate import load_jlens_gate
from prometheus.authz.engine import AuthzEngine, Policy, PolicyEffect
from prometheus.observability.tracing import Tracer
from prometheus.observability.logging import StructuredLogger

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Operator-facing run/safety knobs (all default to the safe posture)."""
    dry_run: bool = True
    allow_fs: bool = False
    allow_net: bool = False
    enable_jlens_gate: bool = False
    max_tool_calls: int | None = None
    max_runtime_seconds: float = 300.0
    fs_root: str | None = None
    verify: bool = True


async def run_task(
    task: str,
    *,
    config: HarnessConfig | None = None,
    provider: ModelProvider | None = None,
    run: RunConfig | None = None,
) -> dict[str, Any]:
    """Run a single task end-to-end and return the orchestrator result enriched
    with the safety posture and a trace summary."""
    config = config or HarnessConfig()
    run = run or RunConfig()
    provider = provider or build_provider_from_env()

    # --- Tools (safe default; dry-run governs side effects) ---
    router = build_default_toolset(
        dry_run=run.dry_run,
        allow_fs=run.allow_fs,
        allow_net=run.allow_net,
        fs_root=run.fs_root,
    )
    provider.set_tool_specs(router.specs())

    # --- Authz: deny-by-default + defaults + budgets; classify every tool ---
    authz = AuthzEngine()
    authz.load_default_policies()
    for name, category in router.categories().items():
        authz.classify_tool(name, category)
    authz.set_budget_limit("max_tool_calls", config.authz.max_tool_calls)
    authz.set_budget_limit("max_external_calls", config.authz.max_external_calls)
    authz.set_budget_limit("max_irreversible", config.authz.max_irreversible)
    if run.allow_net:
        # Make --allow-net meaningful, but still bounded by budgets + the dual-LLM
        # trifecta gate. Without this explicit permit, external tools stay denied.
        authz.add_policy(Policy(
            id="rig_permit_external_optin",
            effect=PolicyEffect.PERMIT,
            principal={"role": "actor"},
            action={"category": "external_visible"},
            description="Operator opted into network tools (--allow-net)",
            priority=20,
        ))

    # --- Dual-LLM taint gate + per-tool taint declarations ---
    dual = DualLLMGate(strict_mode=True)
    for name, taint in router.taints().items():
        dual.declare_tool_taint(name, taint)

    # --- Verifier. NOTE: uses the same provider here; in production it SHOULD be a
    # different model family (model routing — see RIG_NOTES). Fine for the mock. ---
    verifier = AsyncVerifier(verifier_llm_client=provider)

    # --- Observability, now actually wired into the loop ---
    tracer = Tracer(service_name="prometheus")
    slog = StructuredLogger(name="prometheus.run", json_output=config.observability.json_logs)

    # --- Optional J-lens gate (no-op default; no torch required) ---
    jlens_gate = load_jlens_gate(config, enable=run.enable_jlens_gate)

    orch = Orchestrator(
        llm_client=provider,
        tool_router=router,
        verifier=verifier,
        authz_engine=authz,
        jlens_gate=jlens_gate,
        dual_llm_gate=dual,
        tracer=tracer,
        logger=slog,
    )

    max_calls = run.max_tool_calls or config.authz.max_tool_calls
    result = await orch.run_task(
        task,
        tools=router.names(),
        verify=run.verify,
        max_tool_calls=max_calls,
        max_runtime_seconds=run.max_runtime_seconds,
        # Pass tier/effort for the primary agent
        subagent_tier=None,  # Let orchestrator decide based on role
        subagent_effort=None,
    )

    result["safety"] = {
        "dry_run": run.dry_run,
        "allow_fs": run.allow_fs,
        "allow_net": run.allow_net,
        "authz": config.authz.default_mode,
        "max_tool_calls": max_calls,
        "jlens_gate": getattr(jlens_gate, "name", "noop"),
        "dual_llm_gate": "strict",
        "provider": getattr(provider, "model_id", "unknown"),
        "tools_available": router.names(),
    }
    result["trace"] = {"span_count": tracer.span_count, "spans": tracer.export_traces()}
    return result
