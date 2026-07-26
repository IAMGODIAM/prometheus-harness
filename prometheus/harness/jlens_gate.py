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
    """

    name = "jlens"
    enabled = True

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
            logits = self._scorer(text)
            scores = self._watchlist.score(logits, self._tokenizer)
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
