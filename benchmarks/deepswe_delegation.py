#!/usr/bin/env python3
"""
DeepSWE Delegation Runner for Prometheus Orchestrator.

Runs a subset of DeepSWE tasks using the Prometheus delegation primitives:
- Specialist delegation (explorer, reviewer, tester, designer)
- Tier/effort routing (SMALL/STANDARD/LARGE × LOW/MEDIUM/HIGH/XHIGH)
- Warm pool returns (SubagentPool)
- J-Lens gating on every dispatch
- Dynamic effort escalation

Outputs structured JSON for war room review.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Add prometheus to path
sys.path.insert(0, "/home/user/src/prom/repo")

from prometheus.harness.model_client import (
    SubagentTier,
    EffortLevel,
    ProviderFactory,
    MockProvider,
)
from prometheus.harness.orchestrator import (
    Orchestrator,
    AgentRole,
    EffortEscalationHook,
)
from prometheus.harness.subagent_pool import SubagentPool, init_global_pool, get_global_pool
from prometheus.harness.jlens_gate import load_jlens_gate
from prometheus.harness.dual_llm import DualLLMGate
from prometheus.authz.engine import AuthzEngine, ToolCategory
from prometheus.harness.types import PermissionMode, HookType
from prometheus.harness.tools import SafeToolRouter as ToolRouter, build_default_toolset

logger = logging.getLogger(__name__)


@dataclass
class DeepSWETask:
    """A DeepSWE task specification."""
    task_id: str
    repo: str
    language: str
    prompt: str
    verifier_script: str  # Path or command to run verifier
    setup_commands: list[str] = field(default_factory=list)
    timeout_seconds: float = 1800.0  # 30 min default for long-horizon
    max_tool_calls: int = 200
    # Delegation config
    tier: SubagentTier = SubagentTier.STANDARD
    effort: EffortLevel = EffortLevel.MEDIUM
    specialist: AgentRole = AgentRole.ACTOR


@dataclass
class DeepSWEResult:
    """Result of running a DeepSWE task."""
    task_id: str
    repo: str
    language: str
    status: str  # "passed", "failed", "error", "timeout", "budget_exhausted"
    duration_seconds: float
    tool_calls: int
    tokens_used: dict[str, int]
    cost_usd: float
    output: str
    error: str | None = None
    patch_generated: str = ""
    verifier_output: str = ""
    # Delegation metrics
    pool_hits: int = 0
    pool_misses: int = 0
    pool_hit_rate: float = 0.0
    escalations: int = 0
    jlens_triggers: int = 0
    dual_llm_triggers: int = 0
    specialist_used: str = ""
    tier_used: str = ""
    effort_used: str = ""
    # Multi-agent trajectory
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    # Raw orchestrator result
    raw_result: dict[str, Any] = field(default_factory=dict)


class DeepSWERunner:
    """DeepSWE runner using Prometheus orchestrator with delegation primitives."""

    def __init__(
        self,
        use_mock: bool = False,
        enable_jlens: bool = False,
        enable_dual_llm: bool = False,
        pool_max_size: int = 20,
        work_dir: str = "/tmp/deepswe_runs",
    ):
        self.use_mock = use_mock
        self.enable_jlens = enable_jlens
        self.enable_dual_llm = enable_dual_llm
        self.pool_max_size = pool_max_size
        self.work_dir = Path(work_dir)
        self.orchestrator: Optional[Orchestrator] = None
        self.results: list[DeepSWEResult] = []

    async def setup(self) -> None:
        """Initialize orchestrator and dependencies."""
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Build provider
        if self.use_mock:
            provider = MockProvider()
        else:
            provider = ProviderFactory().build_from_env()

        # Tool router with filesystem access for repo operations
        tool_router = build_default_toolset(
            dry_run=False,  # Need real FS for repo ops
            allow_fs=True,
            allow_net=False,
            fs_root=self.work_dir,
        )

        # J-Lens gate
        jlens_gate = load_jlens_gate(enable=self.enable_jlens)

        # Dual-LLM gate
        dual_llm_gate = None
        if self.enable_dual_llm:
            dual_llm_gate = DualLLMGate(
                p_llm_client=provider,
                q_llm_client=provider,
                strict_mode=True,
            )

        # Authz engine
        authz_engine = AuthzEngine()
        authz_engine.load_default_policies()
        # Add permit for read_file and other repo tools
        authz_engine.classify_tool("read_file", ToolCategory.READ_ONLY)
        authz_engine.classify_tool("echo", ToolCategory.READ_ONLY)
        authz_engine.classify_tool("get_time", ToolCategory.READ_ONLY)

        # Initialize orchestrator
        self.orchestrator = Orchestrator(
            llm_client=provider,
            tool_router=tool_router,
            authz_engine=authz_engine,
            jlens_gate=jlens_gate,
            dual_llm_gate=dual_llm_gate,
            subagent_pool_max_size=self.pool_max_size,
        )

        # Register effort escalation hook
        escalation_hook = EffortEscalationHook(
            failure_threshold=0.3,
            complexity_threshold=3000,
            max_escalations=3,
        )
        self.orchestrator.register_global_hook(HookType.PRE_COMPACT, escalation_hook)

        logger.info("DeepSWE delegation runner initialized")

    async def run_task(self, task: DeepSWETask) -> DeepSWEResult:
        """Run a single DeepSWE task with delegation."""
        start_time = time.time()
        task_work_dir = self.work_dir / task.task_id
        task_work_dir.mkdir(parents=True, exist_ok=True)

        # Track pool stats before
        pool_before = get_global_pool().get_stats() if get_global_pool() else {"hits": 0, "misses": 0}

        # Build briefing with task details
        briefing = f"""DeepSWE Task: {task.task_id}
Repository: {task.repo} ({task.language})
Task Description:
{task.prompt}

Working Directory: {task_work_dir}
Verifier: {task.verifier_script}

You are a {task.specialist.value} specialist. Your goal is to implement the requested change in the repository.
The task requires long-horizon engineering work - explore the codebase, understand the structure,
make the necessary changes, and verify they work by running the verifier.

Available tools: read_file, echo, get_time, and shell commands via the orchestrator.
"""

        if task.setup_commands:
            briefing += "\n\nSetup Commands:\n" + "\n".join(task.setup_commands)

        try:
            # Run task with delegation parameters - use specialist delegation
            if task.specialist != AgentRole.ACTOR:
                # Use specialist delegation
                result = await asyncio.wait_for(
                    self.orchestrator.delegate_to_specialist(
                        task=briefing,
                        specialist=task.specialist,
                        briefing=briefing,
                        tier=task.tier,
                        effort=task.effort,
                        verify=False,  # We run our own verifier
                    ),
                    timeout=task.timeout_seconds + 60,
                )
            else:
                # Run as general actor
                result = await asyncio.wait_for(
                    self.orchestrator.run_task(
                        task=briefing,
                        max_tool_calls=task.max_tool_calls,
                        max_runtime_seconds=task.timeout_seconds,
                        subagent_tier=task.tier,
                        subagent_effort=task.effort,
                        verify=False,
                    ),
                    timeout=task.timeout_seconds + 60,
                )

            duration = time.time() - start_time

            # Track pool stats after
            pool_after = get_global_pool().get_stats() if get_global_pool() else {"hits": 0, "misses": 0}
            pool_hits = pool_after.get("hits", 0) - pool_before.get("hits", 0)
            pool_misses = pool_after.get("misses", 0) - pool_before.get("misses", 0)
            pool_hit_rate = pool_hits / (pool_hits + pool_misses) if (pool_hits + pool_misses) > 0 else 0.0

            # Extract metrics
            tokens = result.get("usage", {}) if isinstance(result, dict) else {}
            if isinstance(result, dict) and "tool_calls" in result:
                tool_calls_count = len(result["tool_calls"])
            else:
                tool_calls_count = result.get("tool_calls_made", 0) if hasattr(result, "tool_calls_made") else 0

            # Calculate cost
            cost = 0.0
            if tokens:
                input_cost = tokens.get("input_tokens", 0) / 1000 * 0.0005
                output_cost = tokens.get("output_tokens", 0) / 1000 * 0.001
                cost = input_cost + output_cost

            # Extract patch/output
            output = str(result.get("output", "")) if isinstance(result, dict) else str(result)
            patch = self._extract_patch(output)

            # Run verifier if available
            verifier_output = ""
            if task.verifier_script:
                verifier_output = await self._run_verifier(task_work_dir, task.verifier_script)

            # Determine status
            status = result.get("status", "unknown") if isinstance(result, dict) else str(result)
            if status == "completion_claimed":
                # Check verifier output
                if "passed" in verifier_output.lower() or "success" in verifier_output.lower():
                    status = "passed"
                else:
                    status = "failed"
            elif status == "budget_exhausted":
                status = "budget_exhausted"
            elif status == "error":
                status = "error"
            elif status == "escalated":
                status = "failed"

            ds_result = DeepSWEResult(
                task_id=task.task_id,
                repo=task.repo,
                language=task.language,
                status=status,
                duration_seconds=duration,
                tool_calls=tool_calls_count,
                tokens_used=tokens,
                cost_usd=cost,
                output=output[:5000],
                error=result.get("output") if status == "error" and isinstance(result, dict) else None,
                patch_generated=patch,
                verifier_output=verifier_output[:2000],
                pool_hits=pool_hits,
                pool_misses=pool_misses,
                pool_hit_rate=pool_hit_rate,
                escalations=pool_after.get("escalations", 0) - pool_before.get("escalations", 0),
                jlens_triggers=pool_after.get("jlens_triggers", 0) - pool_before.get("jlens_triggers", 0),
                dual_llm_triggers=pool_after.get("dual_llm_triggers", 0) - pool_before.get("dual_llm_triggers", 0),
                specialist_used=task.specialist.value,
                tier_used=task.tier.value,
                effort_used=task.effort.value,
                raw_result=result if isinstance(result, dict) else {"output": str(result)},
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            ds_result = DeepSWEResult(
                task_id=task.task_id,
                repo=task.repo,
                language=task.language,
                status="timeout",
                duration_seconds=duration,
                tool_calls=0,
                tokens_used={},
                cost_usd=0.0,
                output="",
                error=f"Task timed out after {task.timeout_seconds}s",
            )
        except Exception as e:
            duration = time.time() - start_time
            ds_result = DeepSWEResult(
                task_id=task.task_id,
                repo=task.repo,
                language=task.language,
                status="error",
                duration_seconds=duration,
                tool_calls=0,
                tokens_used={},
                cost_usd=0.0,
                output="",
                error=str(e),
            )

        self.results.append(ds_result)
        return ds_result

    def _extract_patch(self, output: str) -> str:
        """Extract git diff/patch from output."""
        import re
        # Look for git diff or patch markers
        patterns = [
            r"```diff\n(.*?)\n```",
            r"```patch\n(.*?)\n```",
            r"(diff --git.*?)(?:\n\n|\n```|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.DOTALL)
            if match:
                return match.group(1)[:5000]
        return ""

    async def _run_verifier(self, work_dir: Path, verifier_script: str) -> str:
        """Run the verifier script."""
        try:
            # Could be a command or a script path
            if verifier_script.startswith("./") or verifier_script.startswith("/"):
                proc = await asyncio.create_subprocess_shell(
                    verifier_script,
                    cwd=work_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                # Assume it's a test command like "pytest" or "go test"
                proc = await asyncio.create_subprocess_shell(
                    verifier_script,
                    cwd=work_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            stdout, stderr = await proc.communicate()
            return f"stdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"
        except Exception as e:
            return f"Verifier error: {e}"

    async def run_suite(
        self,
        tasks: list[DeepSWETask],
        parallel: bool = False,
        max_concurrent: int = 2,  # Lower concurrency for long-horizon tasks
    ) -> list[DeepSWEResult]:
        """Run a suite of DeepSWE tasks."""
        if parallel:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def run_with_sem(task: DeepSWETask):
                async with semaphore:
                    return await self.run_task(task)

            await asyncio.gather(*[run_with_sem(t) for t in tasks])
        else:
            for task in tasks:
                await self.run_task(task)

        return self.results

    def to_json(self, output_path: str | None = None) -> str:
        """Export results as JSON."""
        data = {
            "benchmark": "deepswe_delegation",
            "timestamp": time.time(),
            "total_tasks": len(self.results),
            "passed": sum(1 for r in self.results if r.status == "passed"),
            "failed": sum(1 for r in self.results if r.status == "failed"),
            "error": sum(1 for r in self.results if r.status == "error"),
            "timeout": sum(1 for r in self.results if r.status == "timeout"),
            "budget_exhausted": sum(1 for r in self.results if r.status == "budget_exhausted"),
            "total_duration_seconds": sum(r.duration_seconds for r in self.results),
            "total_cost_usd": sum(r.cost_usd for r in self.results),
            "total_tool_calls": sum(r.tool_calls for r in self.results),
            "avg_pool_hit_rate": (
                sum(r.pool_hit_rate for r in self.results) / len(self.results)
                if self.results else 0.0
            ),
            "total_escalations": sum(r.escalations for r in self.results),
            "total_jlens_triggers": sum(r.jlens_triggers for r in self.results),
            "total_dual_llm_triggers": sum(r.dual_llm_triggers for r in self.results),
            "results": [
                {
                    "task_id": r.task_id,
                    "repo": r.repo,
                    "language": r.language,
                    "status": r.status,
                    "duration_seconds": r.duration_seconds,
                    "tool_calls": r.tool_calls,
                    "tokens_used": r.tokens_used,
                    "cost_usd": r.cost_usd,
                    "output": r.output[:2000],
                    "error": r.error,
                    "patch_generated": r.patch_generated[:2000],
                    "verifier_output": r.verifier_output[:2000],
                    "pool_hits": r.pool_hits,
                    "pool_misses": r.pool_misses,
                    "pool_hit_rate": r.pool_hit_rate,
                    "escalations": r.escalations,
                    "jlens_triggers": r.jlens_triggers,
                    "dual_llm_triggers": r.dual_llm_triggers,
                    "specialist_used": r.specialist_used,
                    "tier_used": r.tier_used,
                    "effort_used": r.effort_used,
                }
                for r in self.results
            ],
        }

        json_str = json.dumps(data, indent=2)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_str)
        return json_str

    async def shutdown(self) -> None:
        """Shutdown orchestrator and pool."""
        if self.orchestrator:
            await self.orchestrator.shutdown()


# Sample DeepSWE tasks for testing (subset - real tasks would be loaded from DeepSWE dataset)
DEEPSWE_TASKS = [
    DeepSWETask(
        task_id="deepswe-001",
        repo="test-repo-python",
        language="python",
        prompt="Add a new function `calculate_fibonacci(n: int) -> int` to the math module that computes the nth Fibonacci number efficiently using memoization. The function should handle n up to 1000.",
        verifier_script="python -c \"import sys; sys.path.insert(0, '.'); from math_utils import calculate_fibonacci; assert calculate_fibonacci(10) == 55; assert calculate_fibonacci(20) == 6765; print('PASS')\"",
        setup_commands=[
            "mkdir -p /tmp/deepswe_runs/deepswe-001",
            "echo -e 'def add(a, b):\\n    return a + b' > /tmp/deepswe_runs/deepswe-001/math_utils.py",
        ],
        timeout_seconds=600.0,
        max_tool_calls=100,
        tier=SubagentTier.STANDARD,
        effort=EffortLevel.MEDIUM,
        specialist=AgentRole.DESIGNER,
    ),
    DeepSWETask(
        task_id="deepswe-002",
        repo="test-repo-typescript",
        language="typescript",
        prompt="Create a TypeScript utility class `DataProcessor` with methods: `parse(jsonString: string): any`, `validate(data: any, schema: object): boolean`, and `transform(data: any, mapper: Function): any`. Include proper type definitions.",
        verifier_script="npx tsc --noEmit && node -e \"const {DataProcessor} = require('./dist/data-processor'); const dp = new DataProcessor(); console.log('PASS');\"",
        setup_commands=[
            "mkdir -p /tmp/deepswe_runs/deepswe-002",
            "echo '{}' > /tmp/deepswe_runs/deepswe-002/package.json",
        ],
        timeout_seconds=600.0,
        max_tool_calls=100,
        tier=SubagentTier.STANDARD,
        effort=EffortLevel.HIGH,
        specialist=AgentRole.DESIGNER,
    ),
    DeepSWETask(
        task_id="deepswe-003",
        repo="test-repo-go",
        language="go",
        prompt="Implement a concurrent-safe cache with TTL expiration in Go. The cache should support Get(key), Set(key, value, ttl), and Delete(key) operations. Use sync.Map for concurrency.",
        verifier_script="go test -v -run TestCache",
        setup_commands=[
            "mkdir -p /tmp/deepswe_runs/deepswe-003",
            "echo -e 'package main\\n\\nfunc main() {}' > /tmp/deepswe_runs/deepswe-003/main.go",
        ],
        timeout_seconds=600.0,
        max_tool_calls=100,
        tier=SubagentTier.LARGE,
        effort=EffortLevel.HIGH,
        specialist=AgentRole.DESIGNER,
    ),
    DeepSWETask(
        task_id="deepswe-004",
        repo="test-repo-rust",
        language="rust",
        prompt="Create a Rust struct `TaskQueue` with methods `enqueue(task: Task)`, `dequeue() -> Option<Task>`, and `len() -> usize`. Implement it using a `VecDeque` with proper thread safety using `Arc<Mutex<>>`.",
        verifier_script="cargo test",
        setup_commands=[
            "mkdir -p /tmp/deepswe_runs/deepswe-004",
            "cd /tmp/deepswe_runs/deepswe-004 && cargo init --name task_queue",
        ],
        timeout_seconds=900.0,
        max_tool_calls=150,
        tier=SubagentTier.LARGE,
        effort=EffortLevel.XHIGH,
        specialist=AgentRole.DESIGNER,
    ),
    DeepSWETask(
        task_id="deepswe-005",
        repo="test-repo-javascript",
        language="javascript",
        prompt="Build a simple event emitter class `EventEmitter` with `on(event, listener)`, `emit(event, ...args)`, and `off(event, listener)` methods. Support multiple listeners per event.",
        verifier_script="node -e \"const {EventEmitter} = require('./event-emitter'); const ee = new EventEmitter(); let count = 0; ee.on('test', () => count++); ee.emit('test'); ee.emit('test'); console.assert(count === 2); console.log('PASS');\"",
        setup_commands=[
            "mkdir -p /tmp/deepswe_runs/deepswe-005",
        ],
        timeout_seconds=300.0,
        max_tool_calls=50,
        tier=SubagentTier.SMALL,
        effort=EffortLevel.LOW,
        specialist=AgentRole.ACTOR,
    ),
]


async def main():
    """Main entry point for DeepSWE delegation runner."""
    import argparse

    parser = argparse.ArgumentParser(description="DeepSWE Delegation Runner")
    parser.add_argument("--mock", action="store_true", help="Use mock provider")
    parser.add_argument("--jlens", action="store_true", help="Enable J-Lens gate")
    parser.add_argument("--dual-llm", action="store_true", help="Enable Dual-LLM gate")
    parser.add_argument("--parallel", action="store_true", help="Run tasks in parallel")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent tasks")
    parser.add_argument("--output", type=str, default="benchmarks/deepswe_results.json", help="Output JSON path")
    parser.add_argument("--task-ids", type=str, help="Comma-separated task IDs to run")
    parser.add_argument("--pool-size", type=int, default=20, help="Warm pool max size")
    parser.add_argument("--work-dir", type=str, default="/tmp/deepswe_runs", help="Working directory")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Filter tasks if specified
    tasks = DEEPSWE_TASKS
    if args.task_ids:
        selected = set(args.task_ids.split(","))
        tasks = [t for t in tasks if t.task_id in selected]

    # Initialize runner
    runner = DeepSWERunner(
        use_mock=args.mock,
        enable_jlens=args.jlens,
        enable_dual_llm=args.dual_llm,
        pool_max_size=args.pool_size,
        work_dir=args.work_dir,
    )

    try:
        await runner.setup()
        logger.info(f"Running {len(tasks)} DeepSWE tasks...")
        await runner.run_suite(tasks, parallel=args.parallel, max_concurrent=args.max_concurrent)

        # Export results
        output_path = args.output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        json_output = runner.to_json(output_path)
        logger.info(f"Results written to {output_path}")

        # Print summary
        print(json_output)

    finally:
        await runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())