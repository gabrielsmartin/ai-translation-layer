# AI Translation Layer

A mathematical interface between agent intents and AI model calls.

Replaces ad-hoc prompt strings with typed schemas, cosine similarity routing,
EWMA context tracking, and resonance scoring.

---

## The Problem

Every AI implementation does some version of this:

```python
prompt = "You are a lead scorer. Here is the data: " + input
response = model(prompt)
```

That is not AI engineering. That is negotiating with a mathematical function using prose.

A language model is:

- **Token embedding** — every word maps to a vector in R^d (768-4096 dimensions)
- **Attention** — softmax(QK^T / sqrt(d_k)) . V — geometric relationships across all token pairs simultaneously
- **Output** — a probability distribution over vocabulary, not a "response"

The translation layer speaks to it in its native language: structure, vectors, types.

## Quickstart

```bash
pip install pydantic
```

```python
from src.ai_translation_layer import AITranslationLayer

tl = AITranslationLayer()

tl.register_route("lead-scoring", "Score and classify inbound leads by tier and urgency")
tl.register_route("outreach", "Draft cold outreach email sequences for prospects")

result = tl.translate({
    "intent": "Score this new contact form submission",
    "task_type": "analytical",
    "context": {"company": "Acme Corp", "employees": 200}
})

result.encoded_prompt   # canonical typed prompt
result.routing          # [("lead-scoring", 0.87), ("outreach", 0.12)]
result.temperature      # 0.3 (analytical - low variance)
result.context_drift    # False (first turn)

# After the model responds, score it
r = tl.score_output(
    intent="Score this lead by tier and urgency",
    output="This is a Tier A lead with high urgency based on company size."
)
r.score    # 0.82
r.action   # "accept"
```

## What's Built

| Module | What it does | The math |
|---|---|---|
| schemas.py | Typed IntentSchema -> deterministic canonical prompt | Structure collapses prompt variance |
| embeddings.py | EmbeddingRouter - semantic routing | sim(a,b) = (a.b)/(\|a\|.\|b\|) |
| context.py | VectorContextManager - session state as evolving vector | C_t = alpha.e_t + (1-alpha).C_{t-1} |
| resonance.py | ResonanceEvaluator - intent-output alignment score | R = cos_sim(embed(intent), embed(output)) |
| core.py | AITranslationLayer - full pipeline orchestrator | |
| adapters/inference.py | OpenAI-compatible inference adapter | |
| adapters/olw_router_server.py | HTTP sidecar for semantic agent routing at scale | |

## Semantic Routing

Activate full cross-vocabulary routing - one command, no restart required:

```bash
ollama pull nomic-embed-text
```

Embedding backend auto-detected at init:

- **Ollama** :11434 - local, no API key
- **LiteLLM** :4000 - if embedding model configured
- **TF-IDF** - token overlap fallback, zero dependencies

```python
tl = AITranslationLayer()
print(tl._router.backend)  # "ollama" | "litellm" | "tfidf"
```

## Resonance Scoring

R = cos_sim(embed(intent), embed(output))

| Score | Level | Action |
|---|---|---|
| >= 0.70 | HIGH | accept |
| 0.40-0.69 | PARTIAL | review |
| < 0.40 | LOW | retry |

Build the retry loop:

```python
for attempt in range(3):
    output = model(result.encoded_prompt, temperature=result.temperature)
    r = tl.score_output(intent=intent, output=output)
    if r.action == "accept":
        break
```

## Context Drift Detection

```python
# Drift = cos_sim(e_t, C_{t-1}) < 0.4
# You know before the outputs go bad.

events = tl.context_drift_events       # turns where topic shifted
sim = tl.context_similarity("still on topic?")  # < 0.4 = drifted
```

## Routing Sidecar

Seeds from your agent registry on boot, refreshes every 5 min,
routes by cosine similarity over all registered agents.

```bash
python3 src/ai_translation_layer/adapters/olw_router_server.py
# Listening on :3779
```

```bash
curl -X POST http://localhost:3779/route \
  -H 'Content-Type: application/json' \
  -d '{"intent": "score this inbound lead", "top_k": 3}'
# -> {"results": [{"agent_id": "lead-scorer", "score": 0.91}], "backend": "ollama"}

curl http://localhost:3779/health
# -> {"status": "ok", "agents": 85, "backend": "ollama", "last_refresh": 42}
```

Configure via environment:

```bash
REGISTRY_URL=http://your-agent-registry:3778
SEMANTIC_ROUTER_PORT=3779
REFRESH_INTERVAL_SEC=300
```

## AI Interface Audit

If you're running AI agents in production and outputs are inconsistent,
the gap is measurable. We score your current implementation:

- Baseline average resonance score across your current outputs
- Routing accuracy (where the wrong agent is receiving tasks)
- Context drift rate across multi-turn sessions
- Concrete remediation path

Request an audit -> services.gtll.app
gabriel@gtll.app

## License

MIT
