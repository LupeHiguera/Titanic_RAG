# Titanic Historical RAG

Engineering documentation for the codebase. For the public-facing intro, see [README.md](README.md). For the original implementation plan that drove the killer feature, see [CONTRADICTION_PLAN.md](CONTRADICTION_PLAN.md).

## What this is

A RAG system over Titanic inquiry transcripts that **surfaces contradictions between witnesses** rather than collapsing them into a single answer. Standard RAG would hand you "the truth" — this one shows you that Ismay said his lifeboat had 45 people while Officer Lowe said it had 12, side-by-side, with an explanation of why those statements conflict.

## Current state

- US Senate Inquiry (1,173 pages, `Text/USInq.pdf`) fully ingested into Pinecone — ~16K chunks across 68 real witnesses
- British Wreck Commissioner's Inquiry (2,253 pages, `Text/BritishInquiry.pdf`) ingested — **~23.6K chunks across 90 witnesses** (incl. Lightoller @ 1,732 chunks, Ismay @ 976, Stanley Lord @ 729)
- Pinecone index `titanic-rag` now holds ~40K chunks total — both inquiries co-located, distinguished by `metadata.source_type` ("us_inquiry" / "british_inquiry")
- Cross-inquiry witnesses (13 confirmed: Lightoller, Pitman, Boxhall, Lowe, Bride, Fleet, Ismay, Rostron, Lord, Marconi, Symons, Hogg, Cottam) are stored under their per-inquiry name strings — the contradiction detector pairs them naturally as "Charles Herbert Lightoller VS Charles Lightoller", which surfaces self-contradiction across inquiries as the killer demo
- Contradiction detection wired end-to-end via Claude Haiku 4.5
- FastAPI app on port 8000 with `/search` and `/search/contradictions` endpoints
- Single-page UI with a "Show contradictions only" toggle and side-by-side card layout

## Architecture

```
PDF  ─►  DocumentIngestion (pymupdf, per-page text)
     ─►  Witness attribution (US: WitnessIndex / British: BritishWitnessIndex
          + PDF→transcript-page map)                     ← canonical attribution
     ─►  IntelligentChunker(splitter=...)
          ├── SenateBoundarySplitter   (Senator SMITH. / Mr. LOWE.)
          └── BritishBoundarySplitter  (numbered Q&A: "190. Question? - Answer")
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
| `Services/witness_index.py` | US Senate witness attribution by page number (98 entries, 68 unique witnesses) |
| `Services/british_witness_index.py` | British Inquiry witness attribution (121 entries, 97 unique) keyed on **transcript pages**, plus `build_pdf_to_transcript_map()` helper and `BRITISH_TO_US_CANONICAL` alias map (informational; not applied at ingest) |
| `Services/chunking.py` | `IntelligentChunker` + `BoundarySplitter` strategy. Default `SenateBoundarySplitter` matches `Senator SMITH.` / `Mr. LOWE.`; `BritishBoundarySplitter` splits on numbered Q&A (`190.`), strips embedded `Page N` markers |
| `Services/embeddings.py` | OpenAI embeddings, 1024-dim, with in-memory cache |
| `Services/vector_storage.py` | `PineconeVectorStore` (prod, cosine similarity, AWS serverless) |
| `Services/semantic_search.py` | Search orchestration; `get_related_contradictions()` delegates to the detector |
| `Services/contradiction_detector.py` | Claude Haiku 4.5 pairwise comparison, structured JSON output, cache-first |
| `Services/contradiction_cache.py` | Pluggable cache: `SQLiteCache` (local) + `DynamoDBCache` stub for production |
| `Services/pinecone_upload.py` | Ingestion CLI: `--test`, `--full-ingest` (US), `--ingest-british`, `--stats` |

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

# US re-ingestion (if index is empty/stale) — ~10-15 min, ~$0.10 in OpenAI
python Services/pinecone_upload.py --full-ingest

# British re-ingestion — ~20 min, ~$0.20 in OpenAI, ~23.6K chunks
python Services/pinecone_upload.py --ingest-british

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

- **WitnessIndex / BritishWitnessIndex are the only correct attribution paths.** Any code doing regex-based witness extraction from body text is wrong and predates the refactor. Pick the right index for the inquiry.
- **Inquiry formats differ.** US Senate: `Senator SMITH.` / `Mr. LOWE.` speaker turns → `SenateBoundarySplitter`. British: numbered Q&A like `190. Question? - Answer` → `BritishBoundarySplitter`. The dispatcher is `IntelligentChunker(splitter=...)`.
- **British PDF page ≠ transcript page.** The British TOC and witness index are keyed on **transcript pages**; PDF pages run ~2.5× faster. Use `build_pdf_to_transcript_map()` to scan the PDF for `Page N` markers, then `BritishWitnessIndex(pdf_to_transcript=map).get_witness_by_pdf_page(pdf_page)` for attribution. Lookups outside `[FIRST_WITNESS_PAGE, LAST_WITNESS_PAGE]` (17, 748) return None — opening statements and closing arguments are intentionally unattributed.
- **Witness names are stored per-inquiry, not canonicalized.** US has `Charles Herbert Lightoller`, British has `Charles Lightoller`. They are deliberately NOT aliased at ingest time — the contradiction detector excludes same-witness pairs, so if we collapsed both into one name the cross-inquiry self-contradiction demo would silently break. `BRITISH_TO_US_CANONICAL` exists in `british_witness_index.py` for future display-layer hints ("note: same person") but is not applied at ingest.
- **Stale-data marker**: witness names in Pinecone should be proper case (e.g. `Charles Herbert Lightoller`). ALL-CAPS or sentence-fragment names = pre-refactor ingestion artifacts.
- **Chunking strategy is Q&A-bound, not size-bound.** US: ~16K chunks avg ~200 chars. British: ~23.6K chunks avg ~140 chars (Q&As are shorter and more numerous in the British format).
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
| Medium | Phase 4 from the contradiction plan: `DynamoDBCache` impl, `slowapi` rate limit, Dockerfile + App Runner deploy |
| Medium | Async/parallel pair calls in `ContradictionDetector` — O(N²) sequential Haiku calls can be `asyncio.gather`'d for 5–10× latency drop on multi-witness queries |
| Medium | UI hint for cross-inquiry pairs: when contradiction detector returns (e.g.) "Charles Herbert Lightoller" vs "Charles Lightoller", surface a "same person, different inquiry" badge using `BRITISH_TO_US_CANONICAL` |
| Low | Roles in `BritishWitnessIndex` are hand-curated for major witnesses; ~30 minor Board-of-Trade officials default to "Master Mariner" / "Engineer Surveyor" and may be imprecise |
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
│   ├── witness_index.py                # US Senate
│   ├── british_witness_index.py        # British Inquiry + PDF→transcript map + alias map
│   ├── chunking.py                     # IntelligentChunker + Senate/British BoundarySplitters
│   ├── embeddings.py
│   ├── vector_storage.py
│   ├── semantic_search.py
│   ├── contradiction_detector.py
│   ├── contradiction_cache.py
│   └── pinecone_upload.py              # ingestion CLI: --full-ingest, --ingest-british
├── Testing/                            # pytest layout
│   ├── Chunking/  DocumentIngestion/  Embeddings/
│   ├── Ingestion/  Search/  Storage/  Witnesses/
│   └── *.md                            # test specs
├── Text/                               # source PDFs
│   ├── USInq.pdf                       # 1,173 pages, US Senate (ingested)
│   ├── BritishInquiry.pdf              # 2,253 pages, British Wreck Commissioner's (ingested)
│   └── *.pdf                           # other reference PDFs
└── static/
    └── index.html                      # single-page UI
```
