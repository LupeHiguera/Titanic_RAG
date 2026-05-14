# Contradiction Detection — Implementation Plan

## Goal
Ship the killer feature: surface and visually highlight contradictions between
Titanic witness testimonies for any user query. This is the project's main
differentiator from generic RAG systems.

## Approach
LLM-based pairwise comparison using Claude Haiku 4.5, with a pluggable cache
interface so local dev uses SQLite and production swaps to DynamoDB via an env
flag — no code rewrite for AWS migration.

## End-to-End Flow
1. User submits query via existing `/search` endpoint (or new `/search/contradictions`).
2. `SemanticSearchEngine.search()` returns top-K chunks (existing code path).
3. New `ContradictionDetector` groups results by witness, builds cross-witness pairs.
4. For each pair: check cache → fall back to Claude Haiku → store result.
5. Return contradictions list: `{claim_a, claim_b, witness_a, witness_b, confidence, explanation, contradicts: bool}`.
6. UI renders side-by-side cards with the explanation under each pair.

---

## File-by-File Plan

### New files
- **`Services/contradiction_detector.py`**
  - `class ContradictionDetector`
  - `detect(chunks: List[EmbeddedChunk], query: str) -> List[Contradiction]`
  - Builds pairs, calls `ContradictionCache.get_or_compute(...)`.
  - LLM call uses structured JSON output (`response_format` / tool use).
  - Model: `claude-haiku-4-5-20251001`.

- **`Services/contradiction_cache.py`**
  - `class ContradictionCache(ABC)` — abstract interface: `get(key)`, `put(key, value)`.
  - `class SQLiteCache(ContradictionCache)` — local dev, file at `./cache/contradictions.db`.
  - `class DynamoDBCache(ContradictionCache)` — production, table name from env.
  - `def make_cache() -> ContradictionCache` — factory reads `CACHE_BACKEND` env var (`sqlite` | `dynamodb`).
  - Key format: `sha256(sorted([chunk_id_a, chunk_id_b]) + query_topic)`.

- **`Testing/Contradictions/test_contradiction_detector.py`**
  - Mock the Anthropic client; assert pair generation, cache hits, JSON parsing.
  - One real-API smoke test gated by `RUN_LIVE_LLM_TESTS=1`.

### Modified files
- **`Services/semantic_search.py`**
  - Replace stub `get_related_contradictions()` (line ~186) with a real call
    that delegates to `ContradictionDetector`.

- **`app.py`**
  - Add `POST /search/contradictions` endpoint (same request shape as `/search`
    plus `min_confidence: float = 0.6`).
  - Wire `ContradictionDetector` into app startup via dependency.

- **`static/index.html`**
  - Add "Show contradictions" toggle in the filters panel.
  - When toggled on, hit `/search/contradictions` instead of `/search`.
  - New result card style: two-column layout, witness A vs witness B, confidence
    badge, explanation footer.

- **`requirements.txt`**
  - Add `anthropic`, `boto3` (DynamoDB), `aiosqlite` (or use stdlib `sqlite3`).

---

## Implementation Phases

### Phase 1 — Detector + cache (local only)
- Build `ContradictionDetector` and `SQLiteCache`.
- Pluggable interface in place but DynamoDB impl can be a `NotImplementedError` stub.
- CLI smoke test: `python -m Services.contradiction_detector "how fast was the ship"`.
- **Done when:** running the CLI on 3-5 witness chunks returns a JSON contradiction list with explanations.

### Phase 2 — Wire into search + API
- `get_related_contradictions()` calls the detector.
- New `POST /search/contradictions` endpoint.
- Integration test against the running app.

### Phase 3 — UI
- "Show contradictions" toggle.
- Side-by-side card component.
- Confidence badge + explanation.

### Phase 4 — Production cache + hosting
- Implement `DynamoDBCache`.
- Add `slowapi` rate limiter (per IP, e.g. 30 req/min).
- Anthropic spend cap set in console.
- Dockerfile + App Runner config.
- Route 53 + ACM cert.

---

## LLM Prompt Shape (for reference)
```
System: You analyze pairs of Titanic witness statements and detect factual
contradictions. Reply only with valid JSON matching the schema. A contradiction
requires both statements to make claims on the same specific fact (numbers,
times, sequence, identity) that cannot both be true.

User: Query: "{query}"
Witness A ({name_a}): "{chunk_a}"
Witness B ({name_b}): "{chunk_b}"

Schema:
{
  "contradicts": bool,
  "claim_a": "short paraphrase of A's specific claim",
  "claim_b": "short paraphrase of B's specific claim",
  "confidence": 0.0-1.0,
  "explanation": "one sentence on why these conflict (or why they don't)"
}
```

---

## Constraints / Don't Build Yet
- **No auth** — portfolio site is public read-only.
- **No DynamoDB code in Phase 1** — interface only, SQLite impl is enough.
- **No streaming** — pairwise calls are fast enough, batch JSON response is fine.
- **No fancy reranking** — use the existing semantic search output as-is for now.
- **No British Inquiry data** — keep scope to US Senate corpus until killer feature lands.

---

## Hosting Target (for context)
- **Compute:** AWS App Runner running FastAPI container, ~$10/mo.
- **Static:** served by FastAPI's `StaticFiles` (no separate CDN at portfolio scale).
- **Vector DB:** Pinecone (already cloud).
- **Cache:** DynamoDB, on-demand pricing (~free at portfolio traffic).
- **Domain:** Route 53 → App Runner, ACM cert.
- **Secrets:** Anthropic key in AWS Secrets Manager, read at container start.

---

## Success Criteria
- `POST /search/contradictions` returns ≥1 real contradiction for: "how fast was the ship", "how many people in Ismay's lifeboat", "when did the band stop playing".
- UI displays them side-by-side with readable explanations.
- Repeat queries hit cache (p50 < 200ms).
- Live demo at the portfolio domain.