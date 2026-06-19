"""
OpenAI-compatible inference adapter.

Takes a TranslationResult and calls a live model via any OpenAI-compatible
endpoint (Groq, LiteLLM, Ollama, OpenAI, etc.).
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any

DEFAULT_ENDPOINT = "http://localhost:4000/v1/chat/completions"
DEFAULT_MODEL = "gpt-3.5-turbo"


def call(
    encoded_prompt: str,
    temperature: float,
    max_tokens: int = 1024,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    api_key: str = "sk-no-key",
) -> dict[str, Any]:
    """
    Send an encoded prompt to an OpenAI-compatible endpoint.

    Use AITranslationLayer.translate() first, then pass
    result.encoded_prompt and result.temperature here.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": encoded_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        choices = data.get("choices", [])
        if choices:
            return {
                "ok": True,
                "text": choices[0].get("message", {}).get("content", ""),
                "model": model,
                "usage": data.get("usage", {}),
            }
        return {"ok": False, "error": "no choices in response", "raw": data}
    except urllib.error.URLError as e:
        return {"ok": False, "error": str(e)}
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"json decode: {e}"}
