# Titanic Historical RAG

Engineering documentation for the codebase. For the public-facing intro, see [README.md](README.md). For the original implementation plan that drove the killer feature, see [CONTRADICTION_PLAN.md](CONTRADICTION_PLAN.md).

## What this is

A RAG system over Titanic inquiry transcripts that **surfaces contradictions between witnesses** rather than collapsing them into a single answer. Standard RAG would hand you "the truth" — this one shows you that Ismay said his lifeboat had 45 people while Officer Lowe said it had 12, side-by-side, with an explanation of why those statements conflict.

## Current state

- US Senate Inquiry (1,173 pages, `Text/USInq.pdf`) fully ingested into Pinecone — ~16K chunks across 68 real witnesses
- British Inquiry: 7-page format sample only; full ~2,200-page transcript not yet sourced
- Contradiction detection wired end-to-end via Claude Haiku 4.5
- FastAPI app on port 8000 with `/search` and `/search/contradictions` endpoints
- Single-page UI with a "Show contradictions only" toggle and side-by-side card layout

## Architecture

```
PDF  ─►  DocumentIngestion (pymupdf, per-page text)
     ─►  WitnessIndex.get_witness_by_page_range()        ← canonical attribution
     ─►  IntelligentChunker (Q&A boundary splits, ~200-800 chars)
     ─►  EmbeddingService (OpenAI text-embedding-3-large @ 1024d)
     ─►  PineconeVectorStore (cosine, AWS serverless)
                │
                ▼
       SemanticSearchEngine.search()
                │
                ├──► /search                  (top-K results as-is)
                │
                └──► ContradictionDetector    (Claude Haiku 4.5)
                          │
                          ├── pair witnesses across results
                          ├── ContradictionCache.get_or_compute()
                          │     └── SQLiteCache (local) / DynamoDBCache (stub)
                          ▼
                     /search/contradictions   (filtered by min_confidence)
```

## Services

| File | Role |
|---|---|
| `Services/document_ingestion.py` | PDF text extraction via pymupdf, per-page or full-doc |
| `Services/witness_index.py` | Canonical source of truth for witness attribution by page number (98 witnesses) |
| `Services/chunking.py` | Splits testimony on Q&A boundaries (`Senator SMITH.` / `Mr. LOWE.`), respects sentence breaks |
| `Services/embeddings.py` | OpenAI embeddings, 1024-dim, with in-memory cache |
| `Services/vector_storage.py` | `PineconeVectorStore` (prod, cosine similarity, AWS serverless) |
| `Services/semantic_search.py` | Search orchestration; `get_related_contradictions()` delegates to the detector |
| `Services/contradiction_detector.py` | Claude Haiku 4.5 pairwise comparison, structured JSON output, cache-first |
| `Services/contradiction_cache.py` | Pluggable cache: `SQLiteCache` (local) + `DynamoDBCache` stub for production |
| `Services/pinecone_upload.py` | Ingestion CLI: `--test`, `--full-ingest`, `--stats`. Migration paths are legacy |

## Running things

```bash
# Setup
pip install -r requirements.txt

# .env required:  OPENAI_API_KEY, ANTHROPIC_API_KEY, PINECONE_API_KEY
# .env optional:  PINECONE_ENVIRONMENT=us-east-1      (default us-east-1; AWS region — do NOT pass a GCP-style region)
#                 PINECONE_INDEX_NAME=titanic-rag     (default titanic-rag)
#                 ALLOWED_ORIGINS=http://localhost:8000,...  (default localhost; "*" for wildcard)
#                 CACHE_BACKEND=sqlite                (default sqlite; "dynamodb" is a stub)

# Verify Pinecone connection
python Services/pinecone_upload.py --test

# Full re-ingestion (if index is empty/stale) — ~10-15 min, ~$0.10 in OpenAI
python Services/pinecone_upload.py --full-ingest

# Run the app
python app.py        # → http://localhost:8000

# Run tests
python -m pytest Testing/Search Testing/Chunking Testing/Witnesses -q
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/search` | POST | Standard semantic search, returns top-K chunks |
| `/search/contradictions` | POST | Same shape + `min_confidence` (default 0.6); returns LLM-verified contradictions |
| `/witnesses` | GET | Witness filter list derived from `WitnessIndex` (70 names, optional `?search=` substring) |
| `/health` | GET | Vector store + chunk count |
| `/documents` | GET | Index metadata |

## Conventions and gotchas

- **WitnessIndex is the only correct attribution path.** Any code doing regex-based witness extraction from body text is wrong and predates the refactor.
- **Q&A format is Senate-specific** (`Senator SMITH.` / `Mr. LOWE.`). British Inquiry uses `Speaker:` style; needs its own parser.
- **Witness names in storage are lowercase real names** (e.g. `Charles Herbert Lightoller`). If you see ALL-CAPS or sentence-fragment names in Pinecone, that's stale data from a pre-refactor ingestion.
- **Chunking strategy is Q&A-bound, not size-bound.** ~16K chunks for the US corpus, avg ~200 chars/chunk. Don't assume 800.
- **Pinecone region defaults to `us-east-1`** (AWS), matching the AWS serverless spec the code creates. Override via `PINECONE_ENVIRONMENT`. Do NOT pass a GCP-style region — index create will 400.
- **`messages.parse()` requires `anthropic>=0.95`** (we pin `>=0.100`). Older versions silently lack the method and return AttributeError at runtime.

## Test suite state

Run `python -m pytest Testing/ --tb=no` for a snapshot. The legacy ChromaDB tests
and `Testing/Ingestion/ingest_full_usinq.py` have been deleted, so the suite
runs cleanly in a fresh env (modulo any tests that require a live
OpenAI/Pinecone connection, which will skip or error on missing credentials).

## Known issues / what's next

| Priority | Item |
|---|---|
| Medium | British Inquiry pipeline: speaker-tag parser (`Name:` at line start), British witness index, source corpus (`Text/BritishInquiry.pdf` now in repo) |
| Medium | Phase 4 from the contradiction plan: `DynamoDBCache` impl, `slowapi` rate limit, Dockerfile + App Runner deploy |
| Medium | Async/parallel pair calls in `ContradictionDetector` — O(N²) sequential Haiku calls can be `asyncio.gather`'d for 5–10× latency drop on multi-witness queries |
| Low | 2 `PytestReturnNotNoneWarning`s in `test_bold_artifacts.py` |

## File map (rough)

```
.
├── app.py                              # FastAPI entry
├── requirements.txt
├── CLAUDE.md                           # this file
├── README.md                           # public intro
├── CONTRADICTION_PLAN.md               # original killer-feature plan
├── Services/
│   ├── document_ingestion.py
│   ├── witness_index.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_storage.py
│   ├── semantic_search.py
│   ├── contradiction_detector.py
│   ├── contradiction_cache.py
│   └── pinecone_upload.py              # ingestion CLI
├── Testing/                            # pytest layout
│   ├── Chunking/  DocumentIngestion/  Embeddings/
│   ├── Ingestion/  Search/  Storage/  Witnesses/
│   └── *.md                            # test specs
├── Text/                               # source PDFs
│   ├── USInq.pdf                       # 1,173 pages, US Senate (ingested)
│   ├── British_Data.pdf                # 7-page format sample
│   └── *.pdf                           # other reference PDFs
└── static/
    └── index.html                      # single-page UI
```
