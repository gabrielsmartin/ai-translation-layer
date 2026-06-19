"""
AITranslationLayer — the main orchestrator.

This is the interface between GTLL agents and AI models. It replaces:

    "Build me a prompt string, pass it to Claude, hope for the best"

with:

    1. Encode intent as a typed schema (PromptEncoder)
    2. Route to the best agent/model by semantic similarity (EmbeddingRouter)
    3. Track session context as an evolving vector (VectorContextManager)
    4. Score output resonance with the original intent (ResonanceEvaluator)
    5. Return structured result — accept / review / retry — not raw text

The mathematical layer that sits between intent and model call.
"""

from __future__ import annotations

from typing import Any

from .context import VectorContextManager
from .embeddings import EmbeddingRouter
from .resonance import ResonanceEvaluator
from .schemas import IntentSchema, PromptEncoder, TaskType


class TranslationResult:
    def __init__(
        self,
        encoded_prompt: str,
        temperature: float,
        routing: list[tuple[str, float]],
        resonance_score: float | None,
        resonance_action: str,
        context_drift: bool,
        context_turn: int,
        schema: IntentSchema,
    ):
        self.encoded_prompt = encoded_prompt
        self.temperature = temperature
        self.routing = routing
        self.resonance_score = resonance_score
        self.resonance_action = resonance_action
        self.context_drift = context_drift
        self.context_turn = context_turn
        self.schema = schema

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoded_prompt": self.encoded_prompt,
            "temperature": self.temperature,
            "routing": self.routing,
            "resonance_score": self.resonance_score,
            "resonance_action": self.resonance_action,
            "context_drift": self.context_drift,
            "context_turn": self.context_turn,
            "task_type": self.schema.task_type.value,
            "token_budget": self.schema.token_budget(),
        }


class AITranslationLayer:
    """
    Main entry point. Holds all sub-components and orchestrates the translation.

    Usage:
        tl = AITranslationLayer()
        tl.register_route("lead-scoring", "Score inbound leads by tier and urgency")
        tl.register_route("outreach", "Draft cold outreach sequences")

        result = tl.translate({
            "intent": "score this new lead from the contact form",
            "task_type": "analytical",
            "context": {"company": "Acme", "size": "50 employees"}
        })

        result.encoded_prompt   # canonical, typed prompt string
        result.routing          # [("lead-scoring", 0.87), ("outreach", 0.23)]
        result.temperature      # 0.3 (analytical task)

        # After getting AI output, score it:
        resonance = tl.score_output(
            intent="score this lead",
            output="This lead is Tier A with high urgency based on company size."
        )
        resonance.action  # "accept"
    """

    def __init__(self, litellm_key: str = "sk-no-key", context_alpha: float = 0.3):
        self._encoder = PromptEncoder()
        self._router = EmbeddingRouter(litellm_key=litellm_key)
        self._context = VectorContextManager(alpha=context_alpha)
        self._evaluator = ResonanceEvaluator(embed_fn=self._router.embed)

    def register_route(self, agent_id: str, description: str) -> None:
        self._router.register(agent_id, description)

    def translate(self, intent_data: dict[str, Any]) -> TranslationResult:
        schema = IntentSchema(**intent_data)

        encoded_prompt = self._encoder.encode(schema)
        temperature = self._encoder.temperature_for(schema.task_type)

        intent_vec = self._router.embed(schema.intent)
        ctx_snap = self._context.update(intent_vec)

        routing = self._router.route(schema.intent, top_k=3)

        return TranslationResult(
            encoded_prompt=encoded_prompt,
            temperature=temperature,
            routing=routing,
            resonance_score=None,
            resonance_action="pending",
            context_drift=ctx_snap.drifted,
            context_turn=ctx_snap.turn,
            schema=schema,
        )

    def score_output(self, intent: str, output: str):
        return self._evaluator.evaluate(intent, output)

    def best_output(self, intent: str, outputs: list[str]):
        return self._evaluator.best_output(intent, outputs)

    def context_similarity(self, text: str) -> float:
        vec = self._router.embed(text)
        return self._context.similarity_to_context(vec)

    def reset_context(self) -> None:
        self._context.reset()

    @property
    def context_drift_events(self):
        return self._context.drift_events

    @property
    def registered_routes(self) -> list[str]:
        return list(self._router._descriptions.keys())
