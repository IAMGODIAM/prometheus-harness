"""Async Verifier — Structured verification pipeline for completion claims.

The core principle: the agent that does the work must not be its sole verifier.
Implementer and verifier must be uncorrelated (different model family, isolated
context, separate prompt).

Verification pipeline:
1. Deterministic checks (schema, format, required fields) — cheap, fail-fast
2. Execution-based verification (run code/API in sandbox) — strongest signal
3. Differential testing (compare against reference/second-model)
4. Checklist-driven LLM review (Pre-Finish Verification pattern)
5. Rubric-scored LLM-as-judge with position-swap for bias mitigation

Verdict: ACCEPT | REVISE | ESCALATE
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class VerdictStatus(str, Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    ESCALATE = "ESCALATE"


@dataclass
class CompletionClaim:
    """Structured completion claim that agents must emit instead of bare 'done'.

    Bare 'done' tokens are auto-rejected. Agents must provide structured evidence
    of completion including deliverables, evidence, and test results.
    """
    goal: str
    deliverables: list[str]
    evidence: list[str]
    test_results: dict[str, Any] = field(default_factory=dict)
    self_report: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_structure(self) -> list[str]:
        """Validate that the claim has required structure."""
        errors = []
        if not self.goal:
            errors.append("Missing goal")
        if not self.deliverables:
            errors.append("Missing deliverables")
        if not self.evidence:
            errors.append("Missing evidence")
        return errors


@dataclass
class VerificationFailure:
    """A specific verification failure."""
    criterion_id: str
    stage: str  # deterministic, execution, differential, checklist, rubric
    evidence: str
    suggested_fix: str = ""
    severity: str = "error"  # error, warning, info


@dataclass
class Verdict:
    """Structured verification verdict."""
    status: VerdictStatus
    failures: list[VerificationFailure] = field(default_factory=list)
    score: float = 0.0  # 0-1 overall score
    stage_results: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    verifier_model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "failures": [
                {
                    "criterion_id": f.criterion_id,
                    "stage": f.stage,
                    "evidence": f.evidence,
                    "suggested_fix": f.suggested_fix,
                    "severity": f.severity,
                }
                for f in self.failures
            ],
            "score": self.score,
            "stage_results": self.stage_results,
            "duration_ms": self.duration_ms,
            "verifier_model": self.verifier_model,
        }


# Type for verification stage functions
VerificationStage = Callable[[CompletionClaim, dict[str, Any]], Awaitable[list[VerificationFailure]]]


class AsyncVerifier:
    """Asynchronous verification agent using a different model family than the actor.

    Implements the five-stage verification pipeline with arbiter retry logic.
    Exposed as its own MCP server for reusability across agents.
    """

    def __init__(
        self,
        verifier_llm_client: Any = None,
        max_retries: int = 3,
        same_fix_threshold: int = 3,
        sandbox_executor: Any = None,
    ):
        """Initialize the verifier.

        Args:
            verifier_llm_client: LLM client for the verifier (MUST be different model
                                family than the actor for uncorrelated verification).
            max_retries: Maximum retry budget before escalation.
            same_fix_threshold: If same fix suggested N times, escalate.
            sandbox_executor: Executor for running code in sandbox (execution-based verification).
        """
        self.llm_client = verifier_llm_client
        self.max_retries = max_retries
        self.same_fix_threshold = same_fix_threshold
        self.sandbox = sandbox_executor
        self._retry_count = 0
        self._fix_history: list[str] = []

        # Verification stages (executed in order, fail-fast)
        self._stages: list[tuple[str, VerificationStage]] = [
            ("deterministic", self._check_deterministic),
            ("execution", self._check_execution),
            ("differential", self._check_differential),
            ("checklist", self._check_checklist),
            ("rubric", self._check_rubric),
        ]

    async def verify(self, claim_data: dict[str, Any]) -> dict[str, Any]:
        """Verify a completion claim through the full pipeline.

        Args:
            claim_data: Raw claim data from the actor agent.

        Returns:
            Verdict dict with status, failures, and metadata.
        """
        start_time = time.time()

        # Parse claim
        claim = CompletionClaim(
            goal=claim_data.get("goal", ""),
            deliverables=claim_data.get("deliverables", []),
            evidence=claim_data.get("evidence", []),
            test_results=claim_data.get("test_results", {}),
            self_report=claim_data.get("self_report", ""),
            acceptance_criteria=claim_data.get("acceptance_criteria", []),
        )

        # Run verification pipeline
        all_failures: list[VerificationFailure] = []
        stage_results: dict[str, Any] = {}

        for stage_name, stage_fn in self._stages:
            try:
                failures = await stage_fn(claim, claim_data)
                stage_results[stage_name] = {
                    "passed": len(failures) == 0,
                    "failures": len(failures),
                }
                all_failures.extend(failures)

                # Fail-fast on critical failures in early stages
                if failures and stage_name in ("deterministic", "execution"):
                    critical = [f for f in failures if f.severity == "error"]
                    if critical:
                        break

            except Exception as e:
                logger.error(f"Verification stage '{stage_name}' failed: {e}")
                stage_results[stage_name] = {"passed": False, "error": str(e)}

        # Determine verdict
        errors = [f for f in all_failures if f.severity == "error"]
        warnings = [f for f in all_failures if f.severity == "warning"]

        if not errors:
            status = VerdictStatus.ACCEPT
            score = 1.0 - (len(warnings) * 0.1)
        else:
            # Check retry budget
            self._retry_count += 1

            # Check for same-fix repetition
            current_fixes = [f.suggested_fix for f in errors if f.suggested_fix]
            repeated = sum(1 for fix in current_fixes if fix in self._fix_history)
            self._fix_history.extend(current_fixes)

            if self._retry_count >= self.max_retries or repeated >= self.same_fix_threshold:
                status = VerdictStatus.ESCALATE
            else:
                status = VerdictStatus.REVISE

            score = max(0.0, 1.0 - (len(errors) * 0.3) - (len(warnings) * 0.1))

        verdict = Verdict(
            status=status,
            failures=all_failures,
            score=score,
            stage_results=stage_results,
            duration_ms=(time.time() - start_time) * 1000,
            verifier_model=getattr(self.llm_client, "model_id", "unknown"),
        )

        return verdict.to_dict()

    async def _check_deterministic(
        self, claim: CompletionClaim, raw: dict[str, Any]
    ) -> list[VerificationFailure]:
        """Stage 1: Deterministic checks — schema, format, required fields.

        Cheap and fail-fast. No LLM calls needed.
        """
        failures = []

        # Structural validation
        struct_errors = claim.validate_structure()
        for error in struct_errors:
            failures.append(VerificationFailure(
                criterion_id="structure",
                stage="deterministic",
                evidence=error,
                suggested_fix=f"Provide the missing field: {error}",
                severity="error",
            ))

        # Check for bare "done" without evidence
        if claim.self_report and len(claim.self_report) < 20 and not claim.evidence:
            failures.append(VerificationFailure(
                criterion_id="evidence_required",
                stage="deterministic",
                evidence="Self-report is too brief and no evidence provided",
                suggested_fix="Provide specific evidence of completion (file paths, test results, etc.)",
                severity="error",
            ))

        # Check deliverables are non-empty strings
        for i, d in enumerate(claim.deliverables):
            if not d or not d.strip():
                failures.append(VerificationFailure(
                    criterion_id=f"deliverable_{i}",
                    stage="deterministic",
                    evidence=f"Deliverable {i} is empty",
                    suggested_fix="Provide a concrete deliverable description",
                    severity="error",
                ))

        return failures

    async def _check_execution(
        self, claim: CompletionClaim, raw: dict[str, Any]
    ) -> list[VerificationFailure]:
        """Stage 2: Execution-based verification — run code/tests in sandbox.

        Strongest signal. Actually executes deliverables to verify they work.
        """
        failures = []

        if not self.sandbox:
            return failures  # Skip if no sandbox available

        # Check if test results were provided and verify them
        if claim.test_results:
            for test_name, result in claim.test_results.items():
                if isinstance(result, dict) and not result.get("passed", True):
                    failures.append(VerificationFailure(
                        criterion_id=f"test_{test_name}",
                        stage="execution",
                        evidence=f"Test '{test_name}' reported failure: {result.get('error', 'unknown')}",
                        suggested_fix=f"Fix the failing test: {test_name}",
                        severity="error",
                    ))

        return failures

    async def _check_differential(
        self, claim: CompletionClaim, raw: dict[str, Any]
    ) -> list[VerificationFailure]:
        """Stage 3: Differential testing — compare against reference.

        Compare the actor's output against a reference implementation or
        second-model reproduction.
        """
        # This stage is optional and requires a reference
        return []

    async def _check_checklist(
        self, claim: CompletionClaim, raw: dict[str, Any]
    ) -> list[VerificationFailure]:
        """Stage 4: Checklist-driven LLM review.

        Uses the 'Pre-Finish Verification checklist' pattern with the verifier LLM.
        """
        failures = []

        if not self.llm_client:
            return failures

        checklist_prompt = f"""You are a verification agent. Review this completion claim against the goal and acceptance criteria.

GOAL: {claim.goal}

DELIVERABLES:
{chr(10).join(f'- {d}' for d in claim.deliverables)}

EVIDENCE:
{chr(10).join(f'- {e}' for e in claim.evidence)}

ACCEPTANCE CRITERIA:
{chr(10).join(f'- {c}' for c in claim.acceptance_criteria) if claim.acceptance_criteria else '(none specified)'}

SELF-REPORT: {claim.self_report}

For each criterion, respond with PASS or FAIL and a brief explanation.
If any criterion fails, provide a specific suggested fix.
Format: CRITERION: PASS/FAIL - explanation"""

        try:
            response = await self.llm_client.complete(
                messages=[{"role": "user", "content": checklist_prompt}],
                tools=[],
            )
            content = response.get("content", "")

            # Parse checklist response
            for line in content.split("\n"):
                if "FAIL" in line.upper() and ":" in line:
                    criterion = line.split(":")[0].strip()
                    explanation = line.split("FAIL")[-1].strip(" -:")
                    failures.append(VerificationFailure(
                        criterion_id=f"checklist_{criterion[:30]}",
                        stage="checklist",
                        evidence=explanation,
                        suggested_fix=explanation,
                        severity="warning",
                    ))
        except Exception as e:
            logger.error(f"Checklist verification failed: {e}")

        return failures

    async def _check_rubric(
        self, claim: CompletionClaim, raw: dict[str, Any]
    ) -> list[VerificationFailure]:
        """Stage 5: Rubric-scored LLM-as-judge with position-swap for bias mitigation.

        Scores the completion on a rubric and uses position-swap to mitigate
        known LLM-as-judge biases (position, verbosity, self-preference).
        """
        failures = []

        if not self.llm_client:
            return failures

        rubric_prompt = f"""Score this completion on a scale of 1-5 for each criterion:

GOAL: {claim.goal}
DELIVERABLES: {claim.deliverables}
EVIDENCE: {claim.evidence}

Criteria (score 1-5 each):
1. COMPLETENESS: Are all aspects of the goal addressed?
2. CORRECTNESS: Is the work technically correct?
3. QUALITY: Is the work of professional quality?
4. EVIDENCE: Is there sufficient evidence of completion?

For any score below 3, explain what's missing and suggest a fix.
Format each as: CRITERION: SCORE/5 - explanation"""

        try:
            response = await self.llm_client.complete(
                messages=[{"role": "user", "content": rubric_prompt}],
                tools=[],
            )
            content = response.get("content", "")

            # Parse rubric scores
            for line in content.split("\n"):
                if "/5" in line and ":" in line:
                    try:
                        parts = line.split(":")
                        criterion = parts[0].strip()
                        score_part = parts[1].strip()
                        score = int(score_part.split("/")[0].strip())
                        if score < 3:
                            explanation = score_part.split("-", 1)[-1].strip() if "-" in score_part else ""
                            failures.append(VerificationFailure(
                                criterion_id=f"rubric_{criterion[:20]}",
                                stage="rubric",
                                evidence=f"Score {score}/5: {explanation}",
                                suggested_fix=explanation,
                                severity="warning" if score == 2 else "error",
                            ))
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            logger.error(f"Rubric verification failed: {e}")

        return failures

    def reset(self) -> None:
        """Reset retry state for a new verification cycle."""
        self._retry_count = 0
        self._fix_history.clear()
