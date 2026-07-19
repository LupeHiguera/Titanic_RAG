# Titanic Historical RAG

A retrieval-augmented search engine over the 1912 Titanic inquiry transcripts that **surfaces contradictions between witnesses** instead of collapsing them into a single answer.

Standard RAG tries to give you "the truth." The Titanic witnesses gave conflicting accounts on almost everything — speed at the time of collision, lifeboat occupancy, who heard which order, whether the Californian saw rockets. Asking *"How fast was the ship?"* should not return one number. It should show you that Lightoller said 21.5 knots, Hitchins said 45 knots, and that those statements are inconsistent with each other.

This system does that, with confidence-scored explanations.

## What it does

- **Semantic search** across both 1912 inquiries — 1,173 pages of US Senate testimony and 2,253 pages of British Wreck Commissioner's testimony, with **complete witness coverage** (70 US + 97 British indexed witnesses, every one attributed)
- **Real citations** — every chunk carries the printed inquiry page it actually came from (the PDFs' own pagination drifts from the originals by up to 10 pages in the US and 2.5× in the British; a noise-filtered page-marker map corrects both)
- **Side-by-side contradiction view** — an LLM (Claude Haiku 4.5) pairwise-compares witness statements in parallel and surfaces only the factual conflicts, each cited to inquiry + page + witness role
- **Cross-inquiry self-contradiction** — 20+ witnesses testified in *both* inquiries. When their accounts disagree across the two proceedings, the UI badges the pair "same person — both inquiries"
- **Confidence scoring + explanation** for every flagged conflict
- **Filterable** by witness (including "contradictions involving this witness"), inquiry, and minimum confidence

## Evaluated, not just shipped

I ran a 30-query retrieval evaluation against a hand-curated gold set with labeled relevant witnesses, and re-ran it after the 2026-07 ingestion overhaul (packed ~700-char chunks, corrected attribution). The headline numbers (full report: [`Evals/EVAL_RESULTS.md`](Evals/EVAL_RESULTS.md)):

| Metric | Value |
|---|---|
| Hit Rate @ 5 | **70%** (21/30 queries surface ≥1 relevant witness in top 5) |
| Mean Reciprocal Rank | **0.46** (first relevant hit is around rank 2 on average) |
| Recall @ 5 / Recall @ 10 | 0.22 / 0.27 |
| p50 latency (search) | 276 ms |

Honest caveat: the pre-overhaul run scored a slightly higher MRR (0.53) — but against an index whose witness attribution was systematically off by 4–10 pages, so some of those "hits" credited the wrong witness's words. The current numbers are measured against verified attribution.

The threshold sweep shows why the production default sits at 0.4–0.5: metrics are flat from 0.3–0.5, degrade at 0.6, and at 0.7 (the FastAPI/RAG-tutorial default) 29/30 queries silently return zero results, because cosine similarities on witness Q&A chunks cap well below that.

The 9 queries that miss are interesting — abstract phrasings ("Were ice warnings received by wireless?"), topics with sparse coverage in the corpus ("Did the ship break in two?"), and questions where the relevant witness's name doesn't share vocabulary with their testimony. Those are written up honestly in the eval report rather than hidden.

## Architecture

```
PDF  ─►  pymupdf extraction (per-page text)
     ─►  PDF→printed page map  (noise-filtered `Page N` markers; both inquiries drift)
     ─►  Witness attribution   (US: WitnessIndex / British: BritishWitnessIndex;
          sessions split at caps-surname headings so shared pages don't lose witnesses)
     ─►  IntelligentChunker(splitter=...)
          ├── SenateBoundarySplitter   (Senator SMITH. / Mr. LOWE.)
          ├── BritishBoundarySplitter  (numbered Q&A: "190. Question? - Answer")
          └── packs whole Q&As to ~800 chars; per-chunk page citations
     ─►  OpenAI text-embedding-3-large @ 1024d
     ─►  Pinecone (cosine, AWS serverless, ~11.6K vectors, deterministic IDs)
              │
              ▼
       SemanticSearchEngine
              │
              ├──► /search                  (top-K chunks)
              └──► /search/contradictions   (Claude Haiku 4.5 pairwise verdicts,
                       │                     thread-pooled, cached)
                       └── SQLiteCache (dev) / DynamoDBCache (prod, 90-day TTL)
```

Two inquiries with **different transcript formats** required two separate parsers (a strategy-pattern refactor of the chunker) and two witness indices. Both PDFs' page numbering drifts from the printed originals (the US by an irregular 4–10 pages, the British by ~2.5×), so attribution and citations run on a printed-page map recovered from embedded `Page N` markers — with filtering, because the tables of contents emit fake markers.

Witness names are stored *per-inquiry, not canonicalized*, and the contradiction detector groups statements by **(witness, inquiry)**. That's what makes cross-inquiry self-contradiction work uniformly: "Charles Herbert Lightoller" (US) vs "Charles Lightoller" (British) pairs naturally, and witnesses whose names match exactly across inquiries (Ismay, Captain Lord, Fleet) pair too instead of being silently skipped as "the same witness." A 15-entry alias map marks both kinds as `same_person` for the UI badge.

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

# Ingest both inquiries — ~30-45 min, ~$0.30 in embeddings total
python Services/pinecone_upload.py --full-ingest --ingest-british

# Verify attribution coverage offline (no API keys needed)
python Evals/attribution_check.py

# Run the app
python app.py     # → http://localhost:8000

# Run the test suite (112 tests, no network required for most)
python -m pytest Testing/ -q

# Run the retrieval evaluation (requires live Pinecone + OpenAI)
python Evals/run_retrieval_eval.py
```

## What I'd do next

- **Reranking ablation** — try a cross-encoder rerank step (BGE-reranker) and measure the delta on the gold set. Even a negative result is a useful interview story.
- **Contradiction-detection eval** — manually label ~20 pairs as contradicts/doesn't/partial and report the detector's precision and recall, not just retrieval metrics.
- **Ship/role filters in the UI** — chunks now carry `ship` and `witness_type` metadata; "only Californian crew" is a one-line Pinecone filter away.
- **Expand the gold set** — 30 queries is indicative, not statistically tight. 200 would be a real benchmark.

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — engineering-side architecture, conventions, and gotchas
- [`CONTRADICTION_PLAN.md`](CONTRADICTION_PLAN.md) — original plan for the killer feature
- [`Evals/EVAL_RESULTS.md`](Evals/EVAL_RESULTS.md) — full retrieval evaluation report

## License

Not yet licensed.
