"""
Vector context manager — session state as an evolving embedding, not a chat log.

Instead of appending messages to a text buffer, we maintain a running vector:

    C_t = α · e_t + (1 − α) · C_{t-1}

where:
    C_t   = context vector at turn t
    e_t   = embedding of the current message
    α     = decay factor (default 0.3 — recent messages matter more)

Drift is detected when cos_sim(e_t, C_{t-1}) falls below a threshold.
Drift means the conversation has shifted topic — the model's implicit context
is stale, and you should either inject a context reset or escalate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _ewma(current: list[float], new: list[float], alpha: float) -> list[float]:
    return [alpha * n + (1.0 - alpha) * c for c, n in zip(current, new)]


@dataclass
class ContextSnapshot:
    turn: int
    vector: list[float]
    drift_score: float
    drifted: bool


class VectorContextManager:
    """
    Maintains conversation context as a running embedding vector.

    Example:
        vcm = VectorContextManager(alpha=0.3, drift_threshold=0.4)
        vcm.update(embed("user wants to score leads"))
        vcm.update(embed("now they're asking about pricing"))
        snap = vcm.current_snapshot()
        snap.drifted  # True if topic shifted
    """

    def __init__(self, alpha: float = 0.3, drift_threshold: float = 0.4):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.drift_threshold = drift_threshold
        self._vector: list[float] | None = None
        self._turn = 0
        self._history: list[ContextSnapshot] = []

    def update(self, embedding: list[float]) -> ContextSnapshot:
        if self._vector is None:
            self._vector = embedding
            drift_score = 1.0
            drifted = False
        else:
            drift_score = _cosine(embedding, self._vector)
            drifted = drift_score < self.drift_threshold
            self._vector = _ewma(self._vector, embedding, self.alpha)

        self._turn += 1
        snap = ContextSnapshot(
            turn=self._turn,
            vector=list(self._vector),
            drift_score=drift_score,
            drifted=drifted,
        )
        self._history.append(snap)
        return snap

    def current_snapshot(self) -> ContextSnapshot | None:
        return self._history[-1] if self._history else None

    def reset(self) -> None:
        self._vector = None
        self._turn = 0
        self._history = []

    def similarity_to_context(self, embedding: list[float]) -> float:
        if self._vector is None:
            return 0.0
        return _cosine(embedding, self._vector)

    @property
    def turn_count(self) -> int:
        return self._turn

    @property
    def drift_events(self) -> list[ContextSnapshot]:
        return [s for s in self._history if s.drifted]
