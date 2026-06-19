"""
Semantic embedding + cosine similarity routing.

The gap this closes: OLW routes agents by string fingerprint (exact or keyword match).
This routes by *meaning* — cosine similarity in embedding space.

sim(a, b) = (a · b) / (‖a‖ · ‖b‖)

Provider priority (auto-detected at init, no config required):
  1. Ollama :11434 — local, no restart risk, pull nomic-embed-text once
  2. LiteLLM :4000  — if embedding model configured (DO NOT restart to add one)
  3. TF-IDF         — token overlap fallback, always works

To activate full semantic routing on the server:
  ollama pull nomic-embed-text
That's it. No restarts, no config changes, no API keys.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
import urllib.error
from collections import Counter
from enum import Enum


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tfidf_vector(text: str, vocabulary: dict[str, int]) -> list[float]:
    tokens = re.findall(r"\w+", text.lower())
    counts = Counter(tokens)
    vec = [0.0] * len(vocabulary)
    for token, count in counts.items():
        if token in vocabulary:
            vec[vocabulary[token]] = count
    return vec


class EmbedBackend(str, Enum):
    OLLAMA = "ollama"
    LITELLM = "litellm"
    TFIDF = "tfidf"


class EmbeddingRouter:
    """
    Routes intents to registered targets using cosine similarity on embeddings.

    Auto-detects the best available embedding backend at init:
      - Ollama (preferred, no restart risk): ollama pull nomic-embed-text
      - LiteLLM (if already has embedding model)
      - TF-IDF (always available, token overlap only)

    Usage:
        router = EmbeddingRouter()
        print(router.backend)  # tells you which backend is active
        router.register("lead-scoring", "Score and classify inbound leads by tier and urgency")
        router.register("outreach", "Draft and send cold outreach sequences to prospects")
        target = router.route("I need to email 50 new prospects")
        # → "outreach"
    """

    OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"
    OLLAMA_MODEL = "nomic-embed-text"
    LITELLM_URL = "http://127.0.0.1:4000/v1/embeddings"
    LITELLM_MODEL = "text-embedding-3-small"

    def __init__(self, litellm_key: str = "sk-no-key"):
        self._key = litellm_key
        self._targets: dict[str, list[float]] = {}
        self._descriptions: dict[str, str] = {}
        self._vocabulary: dict[str, int] = {}
        self.backend = self._detect_backend()

    def _detect_backend(self) -> EmbedBackend:
        # Try Ollama first — local, no restart risk, no API key needed
        if self._probe_ollama():
            return EmbedBackend.OLLAMA
        # LiteLLM second — only if it already has an embedding model
        if self._probe_litellm():
            return EmbedBackend.LITELLM
        return EmbedBackend.TFIDF

    def _probe_ollama(self) -> bool:
        """Check if Ollama has nomic-embed-text available."""
        try:
            body = json.dumps({"model": self.OLLAMA_MODEL, "prompt": "test"}).encode()
            req = urllib.request.Request(
                self.OLLAMA_URL,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=2)
            data = json.loads(resp.read())
            return "embedding" in data
        except Exception:
            return False

    def _probe_litellm(self) -> bool:
        try:
            body = json.dumps({"model": self.LITELLM_MODEL, "input": "test"}).encode()
            req = urllib.request.Request(
                self.LITELLM_URL,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._key}",
                },
            )
            resp = urllib.request.urlopen(req, timeout=2)
            data = json.loads(resp.read())
            return bool(data.get("data"))
        except Exception:
            return False

    def _embed_ollama(self, text: str) -> list[float] | None:
        try:
            body = json.dumps({"model": self.OLLAMA_MODEL, "prompt": text}).encode()
            req = urllib.request.Request(
                self.OLLAMA_URL,
                data=body,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            return data.get("embedding")
        except Exception:
            return None

    def _embed_litellm(self, text: str) -> list[float] | None:
        try:
            body = json.dumps({"model": self.LITELLM_MODEL, "input": text}).encode()
            req = urllib.request.Request(
                self.LITELLM_URL,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._key}",
                },
            )
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read())
            return data["data"][0]["embedding"]
        except Exception:
            return None

    def _rebuild_vocab(self) -> None:
        all_tokens: set[str] = set()
        for desc in self._descriptions.values():
            all_tokens.update(re.findall(r"\w+", desc.lower()))
        self._vocabulary = {t: i for i, t in enumerate(sorted(all_tokens))}

    def _embed_local(self, text: str) -> list[float]:
        if not self._vocabulary:
            self._rebuild_vocab()
        return _tfidf_vector(text, self._vocabulary)

    def embed(self, text: str) -> list[float]:
        if self.backend == EmbedBackend.OLLAMA:
            vec = self._embed_ollama(text)
            if vec is not None:
                return vec
        elif self.backend == EmbedBackend.LITELLM:
            vec = self._embed_litellm(text)
            if vec is not None:
                return vec
        return self._embed_local(text)

    def register(self, target_id: str, description: str) -> None:
        self._descriptions[target_id] = description
        self._vocabulary = {}
        # Rebuild ALL target embeddings after vocabulary expands — vectors must
        # share the same coordinate space for cosine similarity to be valid.
        for tid, desc in self._descriptions.items():
            self._targets[tid] = self.embed(desc)

    def route(self, intent: str, top_k: int = 1) -> list[tuple[str, float]]:
        """
        Returns ranked list of (target_id, similarity_score).
        Similarity ∈ [−1, 1]; higher = better semantic match.
        """
        if not self._targets:
            return []
        query_vec = self.embed(intent)
        scores = [
            (tid, _cosine(query_vec, tvec))
            for tid, tvec in self._targets.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def best_route(self, intent: str) -> str | None:
        results = self.route(intent, top_k=1)
        if results:
            return results[0][0]
        return None

    def similarity(self, a: str, b: str) -> float:
        """Direct semantic similarity between two texts."""
        return _cosine(self.embed(a), self.embed(b))
