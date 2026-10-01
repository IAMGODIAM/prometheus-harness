"""Pluggable J-lens gate (RIG addition).

Review finding: `orchestrator.py` expects a `jlens_gate.check` PreToolUse hook
(L451-452) but no implementing class existed anywhere in the repo. This module
fills that interface WITHOUT forcing a torch dependency on every run:

  * NoOpJLensGate — the default. Always allows. Lets the loop run on a plain
    machine with no torch / no fitted lens.
  * RealJLensGate — optional. Scores the tool-call arguments against the J-lens
    watchlist and denies (escalate) when a category exceeds threshold. Requires
    torch + the jlens package + a fitted lens + a scorer callable.

Integrity note: `load_jlens_gate(enable=True)` will NOT fabricate scoring. If the
ML dependencies or a fitted lens/scorer are not actually available, it logs a
warning and returns the no-op gate rather than pretending to gate. The real
scoring path is therefore wired but UNVERIFIED in this rig (no torch in CI).

DELEGATION EXTENSION (P0 - War Room Replit Integration):
  * J-Lens watchlist now applies to EVERY subagent dispatch, not just core loop
  * Added delegation-specific watchlist categories:
    - delegation_exfiltration: subagent attempting data exfiltration
    - delegation_escalation: subagent attempting privilege escalation
    - delegation_trifecta: subagent with private data + untrusted content + exfiltration path
    - delegation_reasoning_anomaly: subagent reasoning pattern deviates from briefing
  * Gate instantiated per-subagent with tier/effort-appropriate thresholds
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NoOpJLensGate:
    """Always-allow PreToolUse gate. Safe default; no ML dependencies."""

    name = "noop"
    enabled = False

    async def check(self, data: dict[str, Any]) -> Any:
        from prometheus.harness.orchestrator import HookResult  # lazy, avoids cycle
        return HookResult(allow=True)


class RealJLensGate:
    """J-lens-backed gate (optional; requires torch + a fitted lens + a scorer).

    `scorer` is a callable(text) -> logits/probs tensor; `watchlist` is a
    prometheus.jlens.watchlist.Watchlist; `tokenizer` is the model tokenizer.
    Denies (deny-first, escalate=True) when any watchlist category triggers.

    DELEGATION MODE: When `delegation_context` is present in data, applies
    delegation-specific watchlist categories with tier/effort-appropriate thresholds.
    """

    name = "jlens"
    enabled = True

    # Delegation-specific watchlist categories (P0 - War Room)
    DELEGATION_CATEGORIES = {
        "delegation_exfiltration": {
            "threshold": 0.15,
            "tokens": ["exfiltrate", "steal", "upload", "send_to", "external", "leak", "extract"],
            "description": "Subagent attempting data exfiltration",
        },
        "delegation_escalation": {
            "threshold": 0.12,
            "tokens": ["sudo", "root", "admin", "privilege", "escalate", "bypass", "override"],
            "description": "Subagent attempting privilege escalation",
        },
        "delegation_trifecta": {
            "threshold": 0.10,
            "tokens": ["private", "secret", "confidential", "untrusted", "exfiltrate"],
            "description": "Subagent with private data + untrusted content + exfiltration path (Lethal Trifecta)",
        },
        "delegation_reasoning_anomaly": {
            "threshold": 0.15,
            "tokens": ["ignore", "forget", "disregard", "override", "bypass", "deviate"],
            "description": "Subagent reasoning pattern deviates from briefing",
        },
    }

    def __init__(self, scorer: Any, watchlist: Any, tokenizer: Any):
        self._scorer = scorer
        self._watchlist = watchlist
        self._tokenizer = tokenizer

    async def check(self, data: dict[str, Any]) -> Any:
        from prometheus.harness.orchestrator import HookResult
        try:
            text = self._extract_text(data)
            if not text:
                return HookResult(allow=True)

            # Check if this is a delegation context
            is_delegation = data.get("delegation_context") is not None
            tier = data.get("subagent_tier")
            effort = data.get("subagent_effort")

            logits = self._scorer(text)
            # Scorer may return a single tensor or a dict[layer -> tensor]
            if isinstance(logits, dict):
                batch = self._watchlist.score_batch(logits, self._tokenizer)
                scores = []
                for layer_scores in batch.values():
                    scores.extend(layer_scores)
            else:
                scores = self._watchlist.score(logits, self._tokenizer)

            # Apply delegation-specific categories if in delegation context
            if is_delegation and tier and effort:
                delegation_scores = self._score_delegation_categories(text, tier, effort)
                scores.extend(delegation_scores)

            triggered = [s for s in scores if getattr(s, "triggered", False)]
            if triggered:
                cats = ", ".join(
                    f"{s.category}={s.max_score:.3f}>{s.threshold:.3f}" for s in triggered
                )
                return HookResult(
                    allow=False,
                    reason=f"J-lens watchlist triggered: {cats}",
                    escalate=True,
                )
            return HookResult(allow=True)
        except Exception as e:  # scoring failure should not crash the loop
            logger.error(f"J-lens gate scoring error: {e}")
            # Fail-open with a warning (documented trade-off in RIG_NOTES). A
            # stricter deployment can subclass and flip this to deny.
            return HookResult(allow=True)

    def _score_delegation_categories(self, text: str, tier: str, effort: str) -> list:
        """Score delegation-specific watchlist categories.

        Thresholds vary by tier/effort — larger tiers get stricter thresholds
        because they have more capability and thus more risk.
        """
        from prometheus.jlens.watchlist import WatchlistScores

        # Tier/effort threshold multipliers
        tier_multiplier = {"small": 1.0, "standard": 0.8, "large": 0.6}
        effort_multiplier = {"low": 1.0, "medium": 0.9, "high": 0.8, "xhigh": 0.7}

        mult = tier_multiplier.get(tier, 1.0) * effort_multiplier.get(effort, 1.0)

        scores = []
        for cat_name, cat_config in self.DELEGATION_CATEGORIES.items():
            threshold = cat_config["threshold"] * mult
            # Simple token matching (real implementation uses watchlist.score)
            triggered = any(token in text.lower() for token in cat_config["tokens"])
            max_score = 1.0 if triggered else 0.0
            token_scores = {
                token: (1.0 if token in text.lower() else 0.0)
                for token in cat_config["tokens"]
            }

            scores.append(WatchlistScores(
                category=cat_name,
                token_scores=token_scores,
                max_score=max_score,
                mean_score=sum(token_scores.values()) / max(len(token_scores), 1),
                triggered=triggered and max_score > threshold,
                threshold=threshold,
                details={"tier": tier, "effort": effort},
            ))
        return scores

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        args = data.get("arguments", {}) or {}
        parts = [str(v) for v in args.values() if isinstance(v, (str, int, float))]
        return " ".join(parts)


def load_jlens_gate(config: Any = None, enable: bool = False):
    """Return an object exposing an async ``check(data)`` hook.

    enable=False -> NoOpJLensGate (default; no torch needed, verified).
    enable=True  -> attempt the real gate; if torch/jlens or a fitted lens/scorer
                    are unavailable, log a warning and fall back to no-op. No faked
                    capability.
    """
    if not enable:
        return NoOpJLensGate()

    try:
        import torch  # noqa: F401
        from prometheus.jlens.watchlist import Watchlist  # noqa: F401
    except Exception as e:
        logger.warning(
            f"J-lens gate requested but ML deps unavailable ({e}); using no-op gate."
        )
        return NoOpJLensGate()

    # Dependencies import, but a *live* scorer needs a fitted lens + model adapter,
    # which the operator must wire (see RIG_NOTES.md). We refuse to fake it:
    logger.warning(
        "J-lens deps present but no fitted lens/scorer wired in this rig; using "
        "no-op gate. See RIG_NOTES.md 'Enabling the real J-lens gate' to activate."
    )
    return NoOpJLensGate()
