# Titanic Historical RAG

A retrieval-augmented search engine over the 1912 Titanic inquiry transcripts that **surfaces contradictions between witnesses** instead of collapsing them into a single answer.

Standard RAG tries to give you "the truth." The Titanic witnesses gave conflicting accounts on almost everything — speed at the time of collision, lifeboat occupancy, who heard which order, whether the Californian saw rockets. Asking *"How fast was the ship?"* should not return one number. It should show you that Lightoller said 21.5 knots, Hitchins said 45 knots, and that those statements are inconsistent with each other.

This system does that, with confidence-scored explanations.

## What it does

- **Semantic search** across both 1912 inquiries — 1,173 pages of US Senate testimony and 2,253 pages of British Wreck Commissioner's testimony, ingested as **~40,000 chunks across 158 witnesses**
- **Side-by-side contradiction view** — an LLM (Claude Haiku 4.5) pairwise-compares witness statements and surfaces only the factual conflicts
- **Cross-inquiry comparison** — 13 witnesses testified in *both* inquiries. When their accounts disagree across the two proceedings, the system flags it as a self-contradiction
- **Confidence scoring + explanation** for every flagged conflict
- **Filterable** by witness, inquiry, and minimum confidence

## Evaluated, not just shipped

I ran a 30-query retrieval evaluation against a hand-curated gold set with labeled relevant witnesses. The headline numbers (full report: [`Evals/EVAL_RESULTS.md`](Evals/EVAL_RESULTS.md)):

| Metric | Value |
|---|---|
| Hit Rate @ 5 | **70%** (21/30 queries surface ≥1 relevant witness in top 5) |
| Mean Reciprocal Rank | **0.53** (first relevant hit is around rank 2 on average) |
| Recall @ 5 / Recall @ 10 | 0.21 / 0.30 |
| p50 latency (search) | 408 ms |

The threshold sweep in the eval report shows why `similarity_threshold = 0.5` is the production default: at 0.7 (the FastAPI/RAG-tutorial default), 25/30 queries silently return zero results, because cosine similarities on short witness Q&A chunks cap around 0.77.

The 9 queries that miss are interesting — abstract phrasings ("Were ice warnings received by wireless?"), topics with sparse coverage in the corpus ("Did the ship break in two?"), and questions where the relevant witness's name doesn't share vocabulary with their testimony. Those are written up honestly in the eval report rather than hidden.

## Architecture

```
PDF  ─►  pymupdf extraction (per-page text)
     ─►  Witness attribution  (US: WitnessIndex / British: BritishWitnessIndex + PDF→transcript page map)
     ─►  IntelligentChunker(splitter=...)
          ├── SenateBoundarySplitter   (Senator SMITH. / Mr. LOWE.)
          └── BritishBoundarySplitter  (numbered Q&A: "190. Question? - Answer")
     ─►  OpenAI text-embedding-3-large @ 1024d
     ─►  Pinecone (cosine, AWS serverless, 40K vectors)
              │
              ▼
       SemanticSearchEngine
              │
              ├──► /search                  (top-K chunks)
              └──► /search/contradictions   (Claude Haiku 4.5 pairwise verdicts, cached)
                       │
                       └── SQLiteCache (dev) / DynamoDBCache (prod, 90-day TTL)
```

Two inquiries with **different transcript formats** required two separate parsers (a strategy-pattern refactor of the chunker) and two witness indices. The British transcript uses transcript-page numbers that don't align 1:1 with PDF pages, so the British index includes a `pdf_to_transcript` map built by scanning embedded `Page N` markers across all 2,253 PDF pages.

Witness names are stored *per-inquiry, not canonicalized*. The contradiction detector excludes same-witness pairs, so if "Charles Herbert Lightoller" (US) were aliased to "Charles Lightoller" (UK), his US and British testimonies would collapse into one group and the cross-inquiry self-contradiction demo would silently break. This is a non-obvious gotcha documented in the engineering notes.

## Stack

| Layer | Choice |
|---|---|
| PDF extraction | pymupdf |
| Embeddings | OpenAI `text-embedding-3-large` @ 1024d |
| Vector store | Pinecone serverless (AWS, cosine) |
| Contradiction LLM | Claude Haiku 4.5 with structured JSON output via `messages.parse` |
| API | FastAPI + uvicorn + slowapi rate limiter |
| Verdict cache | SQLite (dev) / DynamoDB with 90-day TTL (prod) |
| Frontend | Single-page HTML/JS, no framework |
| Deployment | Dockerfile (multi-stage, non-root, ~120MB) + AWS App Runner |

## Quick start

```bash
pip install -r requirements.txt

cat > .env <<EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
PINECONE_API_KEY=pcsk_...
EOF

# Verify Pinecone (creates the index on first run)
python Services/pinecone_upload.py --test

# Ingest the US Senate Inquiry — ~10-15 min, ~$0.10 in embeddings
python Services/pinecone_upload.py --full-ingest

# Ingest the British Inquiry — ~20 min, ~$0.20
python Services/pinecone_upload.py --ingest-british

# Run the app
python app.py     # → http://localhost:8000

# Run the test suite (89 tests, no network required for most)
python -m pytest Testing/ -q

# Run the retrieval evaluation (requires live Pinecone + OpenAI)
python Evals/run_retrieval_eval.py
```

## What I'd do next

- **Async pairwise calls** — the contradiction detector currently issues sequential Haiku calls. `asyncio.gather` on N witness pairs would cut latency 5–10× on multi-witness queries.
- **Reranking ablation** — try a cross-encoder rerank step (BGE-reranker) and measure the delta on the gold set. Even a negative result is a useful interview story.
- **Contradiction-detection eval** — manually label ~20 pairs as contradicts/doesn't/partial and report the detector's precision and recall, not just retrieval metrics.
- **Cross-inquiry UI badge** — when a contradiction pair is the same person across inquiries, surface "same witness, different inquiry" using `BRITISH_TO_US_CANONICAL`.
- **Expand the gold set** — 30 queries is indicative, not statistically tight. 200 would be a real benchmark.

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — engineering-side architecture, conventions, and gotchas
- [`CONTRADICTION_PLAN.md`](CONTRADICTION_PLAN.md) — original plan for the killer feature
- [`Evals/EVAL_RESULTS.md`](Evals/EVAL_RESULTS.md) — full retrieval evaluation report

## License

Not yet licensed.
