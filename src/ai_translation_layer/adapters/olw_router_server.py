"""
OLW Semantic Routing Sidecar — HTTP server wrapping EmbeddingRouter.

Runs alongside OLW at :3779. OLW (or any caller) POSTs an intent,
gets back agents ranked by cosine similarity instead of string fingerprint.

Endpoints:
  POST /route     { "intent": "...", "top_k": 3 }
                  → [{"agent_id": "lily", "score": 0.91}, ...]

  POST /refresh   → re-seeds agent list from OLW /agents (no body required)

  GET  /health    → {"status": "ok", "agents": N, "backend": "ollama|tfidf"}

Deploy on the box:
  python3 src/ai_translation_layer/adapters/olw_router_server.py

Systemd unit: see deploy/olw-semantic-router.service
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# Make importable when run directly from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from src.ai_translation_layer.embeddings import EmbeddingRouter

OLW_URL = os.environ.get("OLW_URL", "http://localhost:3778")
LISTEN_PORT = int(os.environ.get("SEMANTIC_ROUTER_PORT", "3779"))
REFRESH_INTERVAL = int(os.environ.get("REFRESH_INTERVAL_SEC", "300"))  # 5 min


class SemanticRouterState:
    def __init__(self) -> None:
        self.router = EmbeddingRouter()
        self.agent_count = 0
        self.last_refresh: float = 0.0
        self._lock = threading.Lock()

    def seed_from_olw(self) -> int:
        """Pull agents from OLW and register descriptions. Returns agent count."""
        try:
            resp = urllib.request.urlopen(f"{OLW_URL}/agents", timeout=5)
            data = json.loads(resp.read())
        except Exception as exc:
            print(f"[seed] OLW unreachable: {exc}", flush=True)
            return self.agent_count

        agents = data if isinstance(data, list) else data.get("agents", [])

        new_router = EmbeddingRouter()
        count = 0
        for agent in agents:
            agent_id = agent.get("id") or agent.get("name") or agent.get("address", "")
            description = (
                agent.get("description")
                or agent.get("fingerprint")
                or agent.get("capabilities")
                or agent_id
            )
            if agent_id and description:
                new_router.register(str(agent_id), str(description))
                count += 1

        with self._lock:
            self.router = new_router
            self.agent_count = count
            self.last_refresh = time.time()

        print(f"[seed] {count} agents registered — backend: {new_router.backend}", flush=True)
        return count

    def route(self, intent: str, top_k: int = 3) -> list[dict[str, Any]]:
        with self._lock:
            results = self.router.route(intent, top_k=top_k)
        return [{"agent_id": aid, "score": round(score, 4)} for aid, score in results]

    @property
    def backend(self) -> str:
        return self.router.backend.value


_state = SemanticRouterState()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access log noise

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "agents": _state.agent_count,
                "backend": _state.backend,
                "last_refresh": int(time.time() - _state.last_refresh),
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/route":
            body = self._read_body()
            intent = body.get("intent", "")
            top_k = int(body.get("top_k", 3))
            if not intent:
                self._send_json(400, {"error": "intent required"})
                return
            results = _state.route(intent, top_k=top_k)
            self._send_json(200, {"results": results, "backend": _state.backend})

        elif self.path == "/refresh":
            count = _state.seed_from_olw()
            self._send_json(200, {"agents": count, "backend": _state.backend})

        else:
            self._send_json(404, {"error": "not found"})


def _refresh_loop() -> None:
    while True:
        time.sleep(REFRESH_INTERVAL)
        _state.seed_from_olw()


if __name__ == "__main__":
    print(f"[boot] Seeding from OLW at {OLW_URL} ...", flush=True)
    _state.seed_from_olw()

    t = threading.Thread(target=_refresh_loop, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", LISTEN_PORT), Handler)
    print(f"[boot] Semantic router listening on :{LISTEN_PORT}", flush=True)
    print(f"[boot] Backend: {_state.backend} | Agents: {_state.agent_count}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[boot] Shutting down.", flush=True)
