"""
Resonance scoring — measures alignment between intent and output in vector space.

    R = cos_sim(embed(intent), embed(output))

This replaces "does this look right?" with a computable signal:
    R > 0.7  → high resonance — output aligns with intent
    R ∈ [0.4, 0.7) → partial resonance — review or retry
    R < 0.4  → low resonance — output drifted from intent, escalate or retry

The OLW system already uses "777" as a completion/resonance signal. This makes
that signal computable rather than symbolic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class ResonanceLevel(str, Enum):
    HIGH = "high"       # R ≥ 0.7
    PARTIAL = "partial" # 0.4 ≤ R < 0.7
    LOW = "low"         # R < 0.4


@dataclass
class ResonanceResult:
    score: float
    level: ResonanceLevel
    intent: str
    output: str
    action: str

    @property
    def is_acceptable(self) -> bool:
        return self.level != ResonanceLevel.LOW


class ResonanceEvaluator:
    """
    Evaluates how well an output satisfies the original intent.

    Requires an embed function (str → list[float]).
    Use EmbeddingRouter.embed as the embed_fn in practice.

    Example:
        evaluator = ResonanceEvaluator(embed_fn=router.embed)
        result = evaluator.evaluate(
            intent="score this lead as tier A or B",
            output="Based on the data, this lead is Tier A with high urgency."
        )
        result.score    # 0.82
        result.level    # ResonanceLevel.HIGH
        result.action   # "accept"
    """

    HIGH_THRESHOLD = 0.70
    PARTIAL_THRESHOLD = 0.40

    ACTION_MAP = {
        ResonanceLevel.HIGH:    "accept",
        ResonanceLevel.PARTIAL: "review",
        ResonanceLevel.LOW:     "retry",
    }

    def __init__(self, embed_fn: Callable[[str], list[float]]):
        self._embed = embed_fn

    def evaluate(self, intent: str, output: str) -> ResonanceResult:
        import math

        iv = self._embed(intent)
        ov = self._embed(output)

        dot = sum(a * b for a, b in zip(iv, ov))
        norm_i = math.sqrt(sum(a * a for a in iv))
        norm_o = math.sqrt(sum(a * a for a in ov))
        score = dot / (norm_i * norm_o) if (norm_i and norm_o) else 0.0

        if score >= self.HIGH_THRESHOLD:
            level = ResonanceLevel.HIGH
        elif score >= self.PARTIAL_THRESHOLD:
            level = ResonanceLevel.PARTIAL
        else:
            level = ResonanceLevel.LOW

        return ResonanceResult(
            score=round(score, 4),
            level=level,
            intent=intent,
            output=output,
            action=self.ACTION_MAP[level],
        )

    def batch_evaluate(
        self, intent: str, outputs: list[str]
    ) -> list[ResonanceResult]:
        results = [self.evaluate(intent, o) for o in outputs]
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def best_output(self, intent: str, outputs: list[str]) -> ResonanceResult | None:
        if not outputs:
            return None
        return self.batch_evaluate(intent, outputs)[0]
