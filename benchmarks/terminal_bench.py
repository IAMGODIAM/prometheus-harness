#!/usr/bin/env python3
"""
Terminal-Bench 4.0 Harness for Prometheus Orchestrator.

Runs Terminal-Bench tasks using the Prometheus delegation primitives:
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
from prometheus.authz.engine import AuthzEngine
from prometheus.harness.types import PermissionMode, HookType
from prometheus.harness.tools import SafeToolRouter as ToolRouter

logger = logging.getLogger(__name__)


@dataclass
class TerminalBenchTask:
    """A Terminal-Bench task specification."""
    task_id: str
    name: str
    description: str
    setup_commands: list[str] = field(default_factory=list)
    task_command: str = ""
    expected_outcome: str = ""
    timeout_seconds: float = 300.0
    max_tool_calls: int = 50
    # Delegation config
    tier: SubagentTier = SubagentTier.STANDARD
    effort: EffortLevel = EffortLevel.MEDIUM
    specialist: AgentRole = AgentRole.ACTOR


@dataclass
class TerminalBenchResult:
    """Result of running a Terminal-Bench task."""
    task_id: str
    name: str
    status: str  # "passed", "failed", "error", "timeout", "budget_exhausted"
    duration_seconds: float
    tool_calls: int
    tokens_used: dict[str, int]
    cost_usd: float
    output: str
    error: str | None = None
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
    # Raw orchestrator result
    raw_result: dict[str, Any] = field(default_factory=dict)


class TerminalBenchHarness:
    """Terminal-Bench 4.0 harness using Prometheus orchestrator."""

    def __init__(
        self,
        use_mock: bool = False,
        enable_jlens: bool = False,
        enable_dual_llm: bool = False,
        pool_max_size: int = 20,
    ):
        self.use_mock = use_mock
        self.enable_jlens = enable_jlens
        self.enable_dual_llm = enable_dual_llm
        self.pool_max_size = pool_max_size
        self.orchestrator: Optional[Orchestrator] = None
        self.results: list[TerminalBenchResult] = []

    async def setup(self) -> None:
        """Initialize orchestrator and dependencies."""
        # Build provider
        if self.use_mock:
            provider = MockProvider()
        else:
            provider = ProviderFactory().build_from_env()

        # Tool router
        tool_router = ToolRouter()

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
            complexity_threshold=2000,
            max_escalations=2,
        )
        self.orchestrator.register_global_hook(HookType.PRE_COMPACT, escalation_hook)

        logger.info("Terminal-Bench harness initialized")

    async def run_task(self, task: TerminalBenchTask) -> TerminalBenchResult:
        """Run a single Terminal-Bench task."""
        start_time = time.time()
        task_id = task.task_id or str(uuid.uuid4())[:8]

        # Track pool stats before
        pool_before = get_global_pool().get_stats() if get_global_pool() else {"hits": 0, "misses": 0}

        # Build briefing with setup commands
        briefing = task.description
        if task.setup_commands:
            briefing += "\n\nSetup:\n" + "\n".join(task.setup_commands)
        briefing += f"\n\nTask: {task.task_command}"

        try:
            # Run task with delegation parameters
            result = await asyncio.wait_for(
                self.orchestrator.run_task(
                    task=briefing,
                    max_tool_calls=task.max_tool_calls,
                    max_runtime_seconds=task.timeout_seconds,
                    subagent_tier=task.tier,
                    subagent_effort=task.effort,
                ),
                timeout=task.timeout_seconds + 30,  # Buffer
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

            # Calculate cost (rough estimate)
            cost = 0.0
            if tokens:
                input_cost = tokens.get("input_tokens", 0) / 1000 * 0.0005
                output_cost = tokens.get("output_tokens", 0) / 1000 * 0.001
                cost = input_cost + output_cost

            # Determine status
            status = result.get("status", "unknown") if isinstance(result, dict) else str(result)
            if status == "completion_claimed":
                verification = result.get("verification", {})
                if verification.get("status") == "PASS":
                    status = "passed"
                else:
                    status = "failed"
            elif status == "budget_exhausted":
                status = "budget_exhausted"
            elif status == "error":
                status = "error"
            elif status == "escalated":
                status = "failed"  # Escalation = failed for benchmark

            tb_result = TerminalBenchResult(
                task_id=task_id,
                name=task.name,
                status=status,
                duration_seconds=duration,
                tool_calls=tool_calls_count,
                tokens_used=tokens,
                cost_usd=cost,
                output=str(result.get("output", "")) if isinstance(result, dict) else str(result),
                error=result.get("output") if status == "error" and isinstance(result, dict) else None,
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
            tb_result = TerminalBenchResult(
                task_id=task_id,
                name=task.name,
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
            tb_result = TerminalBenchResult(
                task_id=task_id,
                name=task.name,
                status="error",
                duration_seconds=duration,
                tool_calls=0,
                tokens_used={},
                cost_usd=0.0,
                output="",
                error=str(e),
            )

        self.results.append(tb_result)
        return tb_result

    async def run_suite(
        self,
        tasks: list[TerminalBenchTask],
        parallel: bool = False,
        max_concurrent: int = 3,
    ) -> list[TerminalBenchResult]:
        """Run a suite of Terminal-Bench tasks."""
        if parallel:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def run_with_sem(task: TerminalBenchTask):
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
            "harness": "terminal_bench_4.0",
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
                    "name": r.name,
                    "status": r.status,
                    "duration_seconds": r.duration_seconds,
                    "tool_calls": r.tool_calls,
                    "tokens_used": r.tokens_used,
                    "cost_usd": r.cost_usd,
                    "output": r.output[:2000] if r.output else "",
                    "error": r.error,
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
            with open(output_path, "w") as f:
                f.write(json_str)
        return json_str

    async def shutdown(self) -> None:
        """Shutdown orchestrator and pool."""
        if self.orchestrator:
            await self.orchestrator.shutdown()


# Built-in Terminal-Bench task suite
TERMINAL_BENCH_TASKS = [
    TerminalBenchTask(
        task_id="tb-001",
        name="file_navigation",
        description="Navigate directory structure and find files matching pattern",
        setup_commands=["mkdir -p /tmp/tb_test/{src,tests,docs}", "echo 'test' > /tmp/tb_test/src/main.py", "echo 'test' > /tmp/tb_test/tests/test_main.py"],
        task_command="Find all .py files in /tmp/tb_test and list them",
        expected_outcome="Lists main.py and test_main.py",
        tier=SubagentTier.SMALL,
        effort=EffortLevel.LOW,
        specialist=AgentRole.EXPLORER,
    ),
    TerminalBenchTask(
        task_id="tb-002",
        name="grep_search",
        description="Search for patterns in files using grep",
        setup_commands=["echo -e 'function foo()\nfunction bar()\nfunction baz()' > /tmp/tb_test/code.js"],
        task_command="Find all lines containing 'function' in /tmp/tb_test/code.js",
        expected_outcome="Returns 3 lines with function declarations",
        tier=SubagentTier.SMALL,
        effort=EffortLevel.LOW,
        specialist=AgentRole.EXPLORER,
    ),
    TerminalBenchTask(
        task_id="tb-003",
        name="file_creation",
        description="Create files with specific content",
        task_command="Create a file /tmp/tb_test/output.txt with content 'Hello Terminal-Bench'",
        expected_outcome="File created with correct content",
        tier=SubagentTier.SMALL,
        effort=EffortLevel.LOW,
        specialist=AgentRole.ACTOR,
    ),
    TerminalBenchTask(
        task_id="tb-004",
        name="multi_step_refactor",
        description="Multi-step code refactoring task",
        setup_commands=["mkdir -p /tmp/tb_refactor", "echo -e 'def add(a,b):\n  return a+b\n\ndef sub(a,b):\n  return a-b' > /tmp/tb_refactor/calc.py"],
        task_command="Rename 'add' to 'sum' and 'sub' to 'difference' in /tmp/tb_refactor/calc.py, then verify changes",
        expected_outcome="Functions renamed correctly",
        tier=SubagentTier.STANDARD,
        effort=EffortLevel.MEDIUM,
        specialist=AgentRole.DESIGNER,
    ),
    TerminalBenchTask(
        task_id="tb-005",
        name="test_generation",
        description="Generate unit tests for existing code",
        setup_commands=["mkdir -p /tmp/tb_testgen", "echo -e 'def factorial(n):\n  if n <= 1:\n    return 1\n  return n * factorial(n-1)' > /tmp/tb_testgen/math.py"],
        task_command="Write pytest unit tests for the factorial function in /tmp/tb_testgen/math.py",
        expected_outcome="Valid pytest tests created",
        tier=SubagentTier.STANDARD,
        effort=EffortLevel.MEDIUM,
        specialist=AgentRole.TESTER,
    ),
    TerminalBenchTask(
        task_id="tb-006",
        name="bug_reproduction",
        description="Reproduce and diagnose a bug",
        setup_commands=["mkdir -p /tmp/tb_bug", "echo -e 'def divide(a,b):\n  return a/b' > /tmp/tb_bug/divide.py"],
        task_command="Find the bug in /tmp/tb_bug/divide.py (division by zero) and write a fix",
        expected_outcome="Bug identified and fix proposed",
        tier=SubagentTier.STANDARD,
        effort=EffortLevel.HIGH,
        specialist=AgentRole.REVIEWER,
    ),
    TerminalBenchTask(
        task_id="tb-007",
        name="complex_analysis",
        description="Analyze a codebase and produce a report",
        setup_commands=["mkdir -p /tmp/tb_analysis/src", "for i in {1..5}; do echo \"def func$i():\n  pass\" > /tmp/tb_analysis/src/module$i.py; done"],
        task_command="Analyze /tmp/tb_analysis and produce a summary report of all functions found",
        expected_outcome="Report listing all 5 functions",
        tier=SubagentTier.LARGE,
        effort=EffortLevel.HIGH,
        specialist=AgentRole.DESIGNER,
    ),
    TerminalBenchTask(
        task_id="tb-008",
        name="git_operations",
        description="Perform git operations",
        setup_commands=["mkdir -p /tmp/tb_git && cd /tmp/tb_git && git init && echo 'initial' > README.md && git add . && git commit -m 'init'"],
        task_command="Create a new branch 'feature', add a file, commit, and merge back to main",
        expected_outcome="Branch created, file added, merged successfully",
        tier=SubagentTier.STANDARD,
        effort=EffortLevel.MEDIUM,
        specialist=AgentRole.ACTOR,
    ),
]


async def main():
    """Main entry point for Terminal-Bench harness."""
    import argparse

    parser = argparse.ArgumentParser(description="Terminal-Bench 4.0 Harness")
    parser.add_argument("--mock", action="store_true", help="Use mock provider")
    parser.add_argument("--jlens", action="store_true", help="Enable J-Lens gate")
    parser.add_argument("--dual-llm", action="store_true", help="Enable Dual-LLM gate")
    parser.add_argument("--parallel", action="store_true", help="Run tasks in parallel")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent tasks")
    parser.add_argument("--output", type=str, default="benchmarks/terminal_bench_results.json", help="Output JSON path")
    parser.add_argument("--task-ids", type=str, help="Comma-separated task IDs to run")
    parser.add_argument("--pool-size", type=int, default=20, help="Warm pool max size")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Filter tasks if specified
    tasks = TERMINAL_BENCH_TASKS
    if args.task_ids:
        selected = set(args.task_ids.split(","))
        tasks = [t for t in tasks if t.task_id in selected]

    # Initialize harness
    harness = TerminalBenchHarness(
        use_mock=args.mock,
        enable_jlens=args.jlens,
        enable_dual_llm=args.dual_llm,
        pool_max_size=args.pool_size,
    )

    try:
        await harness.setup()
        logger.info(f"Running {len(tasks)} Terminal-Bench tasks...")
        await harness.run_suite(tasks, parallel=args.parallel, max_concurrent=args.max_concurrent)

        # Export results
        output_path = args.output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        json_output = harness.to_json(output_path)
        logger.info(f"Results written to {output_path}")

        # Print summary
        print(json_output)

    finally:
        await harness.shutdown()


if __name__ == "__main__":
    asyncio.run(main())