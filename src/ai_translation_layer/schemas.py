"""
Typed prompt schemas — deterministic canonical encoding over ad-hoc strings.

The core insight: a typed schema collapses prompt variance the same way a typed API
collapses RPC variance. The model sees the same token sequence for the same intent,
every time. No synonym drift, no instruction-order effects, no forgotten context.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    ANALYTICAL = "analytical"     # low variance needed → T ∈ [0.1, 0.4]
    GENERATIVE = "generative"     # high variance acceptable → T ∈ [0.7, 1.0]
    ROUTING = "routing"           # zero variance needed → T = 0.0
    EXTRACTION = "extraction"     # low variance → T ∈ [0.0, 0.2]
    SYNTHESIS = "synthesis"       # moderate variance → T ∈ [0.3, 0.6]


class ConstraintSet(BaseModel):
    max_tokens: int = 1024
    format: str = "prose"
    language: str = "en"
    persona: str | None = None
    forbidden_topics: list[str] = Field(default_factory=list)
    required_inclusions: list[str] = Field(default_factory=list)


class IntentSchema(BaseModel):
    """
    Typed representation of what the caller wants from the AI.

    Instead of free-text prompts, callers declare intent structurally.
    The PromptEncoder converts this to a canonical token sequence.
    """
    intent: str
    task_type: TaskType = TaskType.ANALYTICAL
    context: dict[str, Any] = Field(default_factory=dict)
    constraints: ConstraintSet = Field(default_factory=ConstraintSet)
    examples: list[dict[str, str]] = Field(default_factory=list)
    agent_id: str | None = None
    session_id: str | None = None

    def token_budget(self) -> int:
        """Estimate token budget from schema complexity."""
        base = 200
        context_tokens = len(str(self.context)) // 4
        example_tokens = sum(len(str(e)) // 4 for e in self.examples)
        constraint_tokens = 50 if self.constraints.forbidden_topics else 0
        return min(base + context_tokens + example_tokens + constraint_tokens,
                   self.constraints.max_tokens)


class PromptEncoder:
    """
    Converts IntentSchema → canonical prompt string.

    Deterministic: same schema instance always produces the same token sequence.
    This is the anti-prompt-engineering primitive — structure over prose.
    """

    TASK_PREAMBLES = {
        TaskType.ANALYTICAL: "Analyze the following with precision. Prefer facts over interpretation.",
        TaskType.GENERATIVE: "Generate creative output grounded in the provided context.",
        TaskType.ROUTING:    "Classify and route. Output only the routing target, nothing else.",
        TaskType.EXTRACTION: "Extract structured data from the input. Output only what was requested.",
        TaskType.SYNTHESIS:  "Synthesize the provided information into a coherent unified view.",
    }

    def encode(self, schema: IntentSchema) -> str:
        parts: list[str] = []

        parts.append(f"[TASK: {schema.task_type.value.upper()}]")
        parts.append(self.TASK_PREAMBLES[schema.task_type])

        if schema.constraints.persona:
            parts.append(f"[PERSONA: {schema.constraints.persona}]")

        if schema.context:
            parts.append("[CONTEXT]")
            for k, v in schema.context.items():
                parts.append(f"  {k}: {v}")

        if schema.constraints.required_inclusions:
            parts.append("[MUST INCLUDE: " + ", ".join(schema.constraints.required_inclusions) + "]")

        if schema.constraints.forbidden_topics:
            parts.append("[MUST NOT MENTION: " + ", ".join(schema.constraints.forbidden_topics) + "]")

        if schema.examples:
            parts.append("[EXAMPLES]")
            for ex in schema.examples:
                inp = ex.get("input", "")
                out = ex.get("output", "")
                parts.append(f"  input: {inp}")
                parts.append(f"  output: {out}")

        parts.append(f"[INTENT] {schema.intent}")
        parts.append(f"[FORMAT: {schema.constraints.format} | MAX_TOKENS: {schema.constraints.max_tokens}]")

        return "\n".join(parts)

    def temperature_for(self, task_type: TaskType) -> float:
        mapping = {
            TaskType.ROUTING:    0.0,
            TaskType.EXTRACTION: 0.1,
            TaskType.ANALYTICAL: 0.3,
            TaskType.SYNTHESIS:  0.5,
            TaskType.GENERATIVE: 0.8,
        }
        return mapping[task_type]
