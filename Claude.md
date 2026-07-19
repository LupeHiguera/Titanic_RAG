# Titanic Historical RAG

Engineering documentation for the codebase. For the public-facing intro, see [README.md](README.md). For the original implementation plan that drove the killer feature, see [CONTRADICTION_PLAN.md](CONTRADICTION_PLAN.md).

## What this is

A RAG system over Titanic inquiry transcripts that **surfaces contradictions between witnesses** rather than collapsing them into a single answer. Standard RAG would hand you "the truth" — this one shows you that Ismay said his lifeboat had 45 people while Officer Lowe said it had 12, side-by-side, with an explanation of why those statements conflict.

## Current state

- US Senate Inquiry (1,173 pages, `Text/USInq.pdf`) fully ingested into Pinecone — ~4.4K packed chunks (mean ~710 chars) across **all 70** indexed witnesses
- British Wreck Commissioner's Inquiry (2,253 pages, `Text/BritishInquiry.pdf`) ingested — ~7.2K packed chunks across **all 97** indexed witnesses
- Pinecone index `titanic-rag` holds ~11.6K chunks total — both inquiries co-located, distinguished by `metadata.source_type` ("us_inquiry" / "british_inquiry"); IDs are deterministic (`us:`/`br:` prefix + content hash) so re-ingest upserts instead of duplicating
- Every chunk cites the **printed inquiry page it actually came from** (page tags carried through chunking), plus `role`, `ship`, `witness_type` metadata
- Cross-inquiry witnesses are stored under their per-inquiry name strings; the contradiction detector groups by **(witness_name, source_type)**, so both differently-named (Lightoller) and same-named (Ismay, Stanley Lord, Fleet) witnesses get compared against their own other-inquiry testimony — `same_person: true` in the response drives the UI badge
- Contradiction detection wired end-to-end via Claude Haiku 4.5, pair checks parallelized on a thread pool (max 45 pairs/query)
- FastAPI app on port 8000 with `/search` and `/search/contradictions` endpoints (sync handlers → threadpool; the event loop is never blocked)
- Single-page UI: side-by-side contradiction cards with per-side role + inquiry + page citations and a gold "same person — both inquiries" badge

## Architecture

```
PDF  ─►  DocumentIngestion (pymupdf, per-page text)
     ─►  page_map.build_page_map (PDF page → printed page, noise-filtered)
     ─►  build_witness_contexts (session splitting at caps-surname headings;
          US: WitnessIndex / British: BritishWitnessIndex ← canonical attribution;
          ⟦p:N⟧ page tags embedded where the printed page changes)
     ─►  IntelligentChunker(splitter=...)
          ├── SenateBoundarySplitter   (Senator SMITH. / Mr. LOWE.)
          ├── BritishBoundarySplitter  (numbered Q&A: "190. Question? - Answer")
          └── packs adjacent Q&As up to chunk_size; assigns per-chunk pages
     ─►  EmbeddingService (OpenAI text-embedding-3-large @ 1024d)
     ─►  PineconeVectorStore (cosine, AWS serverless, deterministic us:/br: IDs)
                │
                ▼
       SemanticSearchEngine.search()
                │
                ├──► /search                  (top-K results as-is)
                │
                └──► ContradictionDetector    (Claude Haiku 4.5)
                          │
                          ├── group by (witness, inquiry); over-fetched results
                          ├── pair checks on a thread pool (≤45 pairs)
                          ├── ContradictionCache.get_or_compute()
                          │     └── SQLiteCache (local) / DynamoDBCache (prod)
                          ▼
                     /search/contradictions   (filtered by min_confidence)
```

## Services

| File | Role |
|---|---|
| `Services/document_ingestion.py` | PDF text extraction via pymupdf, per-page or full-doc; `clean_extracted_text()` |
| `Services/page_map.py` | Shared PDF-page → printed-page mapper for both inquiries; filters TOC/reference `Page N` marker noise (monotonicity, plausibility, marker-count rules) |
| `Services/witness_index.py` | US Senate witness attribution by **printed** page (98 entries, 70 unique witnesses); bounds `[2, 1142]` exclude front matter/appendices |
| `Services/british_witness_index.py` | British Inquiry witness attribution (121 entries, 97 unique) keyed on **transcript pages**, plus `BRITISH_TO_US_CANONICAL` alias map (15 entries; display-layer `same_person` hint, not applied at ingest) |
| `Services/chunking.py` | `IntelligentChunker` + `BoundarySplitter` strategy. Splits on Q&A boundaries, packs adjacent Q&As up to `chunk_size` (800), resolves per-chunk pages from `⟦p:N⟧` tags. `SenateBoundarySplitter` matches `Senator SMITH.` / `Mr. LOWE.`; `BritishBoundarySplitter` splits on numbered Q&A (`190.`) |
| `Services/embeddings.py` | OpenAI embeddings, 1024-dim, with in-memory cache |
| `Services/vector_storage.py` | `PineconeVectorStore` (cosine, AWS serverless); deterministic `us:`/`br:` IDs, `delete_by_prefix()`, `delete_all()` |
| `Services/semantic_search.py` | Search orchestration; `get_related_contradictions()` over-fetches for witness diversity, supports "contradictions involving witness X" filter mode |
| `Services/contradiction_detector.py` | Claude Haiku 4.5 pairwise comparison on a thread pool, structured JSON output, cache-first, groups by (witness, inquiry) |
| `Services/contradiction_cache.py` | Pluggable cache: `SQLiteCache` (local, lock-guarded) + `DynamoDBCache` (prod, boto3-backed, 90-day TTL, errors degrade silently to cache miss) |
| `Services/pinecone_upload.py` | Ingestion CLI: `--test`, `--full-ingest` (US), `--ingest-british`, `--stats`, `--clear-all`, `--clear-source {us,british}`; `build_witness_contexts()` is the session builder (also used by `Evals/attribution_check.py`) |

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

# Full rebuild (deterministic IDs make ingest idempotent, but --clear-all
# also removes legacy UUID-id vectors) — ~30-45 min, ~$0.30 in OpenAI total
python Services/pinecone_upload.py --clear-all --full-ingest --ingest-british

# Individual inquiries / partial clears
python Services/pinecone_upload.py --full-ingest          # US only
python Services/pinecone_upload.py --ingest-british       # British only
python Services/pinecone_upload.py --clear-source british # delete br:* vectors

# Attribution coverage check (offline, no API keys) — run after touching
# witness indexes, page mapping, or session splitting
python Evals/attribution_check.py

# Run the app
python app.py        # → http://localhost:8000

# Run tests
python -m pytest Testing/ -q
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/search` | POST | Standard semantic search, returns top-K chunks (incl. `role`, `ship`, citable `page_number`) |
| `/search/contradictions` | POST | Same shape + `min_confidence` (default 0.6); returns LLM-verified contradictions with per-side `source`/`page`/`role` citations and `same_person`. Over-fetches internally for witness diversity; a `witness_name` filter means "contradictions *involving* this witness" |
| `/witnesses` | GET | Witness filter list from both indexes (optional `?search=` substring) |
| `/health` | GET | Vector store + chunk count |
| `/documents` | GET | Index metadata (witness count deduped across inquiries via canonical map) |

Request validation: `query` ≤500 chars, `top_k` 1–25, thresholds 0–1,
`source_type` must be `us_inquiry`/`british_inquiry`. 429s return a
`detail` message the UI shows verbatim.

## Conventions and gotchas

- **WitnessIndex / BritishWitnessIndex are the only correct attribution paths.** Any code doing regex-based witness extraction from body text is wrong and predates the refactor. Pick the right index for the inquiry.
- **BOTH inquiries need the PDF→printed page map.** The witness indexes are keyed on **printed** inquiry pages, and PDF pages drift from them in *both* documents (US: +4 to +10 pages; British: ~2.5 PDF pages per transcript page). `Services/page_map.build_page_map()` builds the mapping from embedded `Page N` markers with noise filtering — never feed raw PDF page numbers to `get_witness_by_page_range`. Lookups outside each index's `[FIRST_WITNESS_PAGE, LAST_WITNESS_PAGE]` (US: 2–1142, British: 17–748) return None — opening statements, affidavits, and the digest are intentionally unattributed.
- **Same-page witness handoffs are split at caps-surname headings.** `build_witness_contexts()` starts each session at the witness's heading (`TESTIMONY OF JAMES WIDGERY` / `FREDERICK SHEATH, Sworn.`) so two witnesses sharing a printed page both keep their testimony. If a heading isn't found, the session falls back to the top of the following page. Same-page tie-breaks in `get_witness_by_page_range` give the page to the *later* TOC entry.
- **Inquiry formats differ.** US Senate: `Senator SMITH.` / `Mr. LOWE.` speaker turns → `SenateBoundarySplitter`. British: numbered Q&A like `190. Question? - Answer` → `BritishBoundarySplitter`. The dispatcher is `IntelligentChunker(splitter=...)`.
- **`page_number` in chunk metadata is the citable printed/transcript page** of the text the chunk starts on — not a PDF page, and not the session's first page. Page identity travels through cleaning/chunking as `⟦p:N⟧` tags (see `chunking.PAGE_TAG`), which are stripped before storage.
- **Witness names are stored per-inquiry, not canonicalized.** US has `Charles Herbert Lightoller`, British has `Charles Lightoller`. The contradiction detector groups by **(witness_name, source_type)**, so cross-inquiry self-comparison works for BOTH differently-named and identically-named (Ismay, Lord, Fleet, Barrett, Gill, Crawford, Archer) witnesses. `BRITISH_TO_US_CANONICAL` (15 entries, incl. spelling drift like Hitchins→Hichens) powers the `same_person` flag at display time and is not applied at ingest.
- **Chunks are Q&A-bound AND size-packed.** Splitters cut on Q&A boundaries; the chunker then packs adjacent whole Q&As up to `chunk_size=800`. Result: ~4.4K US + ~7.2K British chunks, mean ~710 chars. A one-Q&A-per-chunk index (~40K tiny chunks) is pre-overhaul data.
- **Chunk IDs are deterministic**: `us:`/`br:` prefix + sha256 of witness|page|index|content. Re-running ingest upserts in place; `--clear-source us|british` deletes by prefix; `--clear-all` also removes legacy UUID-id vectors.
- **Pinecone region defaults to `us-east-1`** (AWS), matching the AWS serverless spec the code creates. Override via `PINECONE_ENVIRONMENT`. Do NOT pass a GCP-style region — index create will 400.
- **`messages.parse()` requires `anthropic>=0.95`** (we pin `>=0.100`). Older versions silently lack the method and return AttributeError at runtime.
- **Keep FastAPI search endpoints sync (`def`).** Everything they call (OpenAI, Pinecone, Anthropic) is blocking; `async def` would freeze the event loop — including `/health`, which App Runner uses for liveness.

## Test suite state

112 tests, all passing (`python -m pytest Testing/ -q`). Notable coverage:
`Testing/Witnesses/test_attribution_fixes.py` (tie-breaks, bounds, page-map
plausibility), `Testing/Chunking/test_page_tracking_and_packing.py` (⟦p:N⟧
tags, packing, decimal-safe sentence splitting),
`Testing/Contradictions/test_detector_grouping.py` ((witness, inquiry)
grouping, same_person, failure tolerance). Tests requiring live
OpenAI/Pinecone skip or error on missing credentials.
`Evals/attribution_check.py` is the offline end-to-end attribution gate —
it must report 70/70 US and 97/97 British witnesses covered.

## Deployment (AWS App Runner)

The container, cache backend, and rate limiter are all production-ready. The
remaining work is one-shot infra setup outside the codebase.

### Build and test the image locally

```bash
docker build -t titanic-rag:latest .

# Run with .env values mapped in
docker run --rm -p 8000:8000 --env-file .env \
    -e CACHE_BACKEND=sqlite \
    titanic-rag:latest
# → http://localhost:8000
```

### Required infra (one-time)

1. **DynamoDB table** for the contradiction cache:
   ```bash
   aws dynamodb create-table \
       --table-name titanic-rag-contradictions \
       --attribute-definitions AttributeName=cache_key,AttributeType=S \
       --key-schema AttributeName=cache_key,KeyType=HASH \
       --billing-mode PAY_PER_REQUEST
   aws dynamodb update-time-to-live \
       --table-name titanic-rag-contradictions \
       --time-to-live-specification "Enabled=true,AttributeName=expires_at"
   ```
2. **Secrets Manager** entries for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
   `PINECONE_API_KEY` (see `apprunner.yaml` for the ARN slots to fill in).
3. **ECR repo** + push the built image, OR set App Runner to source-from-GitHub.
4. **App Runner service** pointed at the image. Uses `apprunner.yaml` for env
   wiring; port 8000; healthcheck `/health`.
5. **Instance role IAM** (App Runner attaches this to the running container):
   - `dynamodb:GetItem`, `dynamodb:PutItem` on the cache table ARN
   - `secretsmanager:GetSecretValue` on the three API-key secrets
6. **Anthropic spend cap** in console (`https://console.anthropic.com/settings/billing`)
   — set a monthly limit so a runaway loop can't drain the account.
7. **Route 53 + ACM cert** for the custom domain → App Runner custom-domain
   association.

### Production env vars (set in apprunner.yaml or App Runner console)

| Var | Required | Notes |
|---|---|---|
| `CACHE_BACKEND` | yes | `dynamodb` in prod, `sqlite` in dev |
| `CONTRADICTION_CACHE_TABLE` | yes | DynamoDB table name |
| `OPENAI_API_KEY` | yes | Secrets Manager reference |
| `ANTHROPIC_API_KEY` | yes | Secrets Manager reference |
| `PINECONE_API_KEY` | yes | Secrets Manager reference |
| `PINECONE_INDEX_NAME` | no | Defaults to `titanic-rag` |
| `ALLOWED_ORIGINS` | yes | Comma-separated list, e.g. `https://titanic.higuera.io` |
| `RATE_LIMIT_SEARCH` | no | Defaults to `30/minute` |
| `RATE_LIMIT_CONTRADICTIONS` | no | Defaults to `10/minute`; LLM-bound, more expensive |

### Rate limiting

`slowapi` rate-limits `/search` and `/search/contradictions` per-IP. App Runner
sits in front of the container, so the real client IP arrives via
`X-Forwarded-For` — the Dockerfile runs uvicorn with `--proxy-headers
--forwarded-allow-ips '*'` so `get_remote_address` picks it up correctly.
Limits are env-tunable (see table above).

## Known issues / what's next

| Priority | Item |
|---|---|
| Medium | Re-run `Evals/run_retrieval_eval.py` whenever chunking changes — the gold set's Hit Rate/MRR are sensitive to chunk size and the packed chunks (2026-07) changed the distribution |
| Low | Session-heading fallback: if a witness's caps-surname heading isn't found on their start page, their session starts at the top of the next page (a few lines of the previous witness may bleed in) |
| Low | Roles in `BritishWitnessIndex` are hand-curated for major witnesses; ~30 minor Board-of-Trade officials default to "Master Mariner" / "Engineer Surveyor" and may be imprecise |
| Low | 2 `PytestReturnNotNoneWarning`s (`test_bold_artifacts.py`, `test_real_embedding_with_pdf.py`) |

## File map (rough)

```
.
├── app.py                              # FastAPI entry
├── requirements.txt
├── CLAUDE.md                           # this file
├── README.md                           # public intro
├── CONTRADICTION_PLAN.md               # original killer-feature plan
├── Dockerfile                          # multi-stage; non-root; uvicorn with --proxy-headers
├── .dockerignore
├── apprunner.yaml                      # App Runner deploy config (image-based)
├── Services/
│   ├── document_ingestion.py
│   ├── page_map.py                     # PDF→printed page mapping (both inquiries)
│   ├── witness_index.py                # US Senate (printed-page keyed)
│   ├── british_witness_index.py        # British Inquiry + alias map
│   ├── chunking.py                     # IntelligentChunker + splitters + page tags + packing
│   ├── embeddings.py
│   ├── vector_storage.py               # deterministic IDs, delete_by_prefix
│   ├── semantic_search.py
│   ├── contradiction_detector.py       # (witness, inquiry) groups, thread pool
│   ├── contradiction_cache.py          # SQLiteCache (dev) / DynamoDBCache (prod)
│   └── pinecone_upload.py              # ingestion CLI + build_witness_contexts
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
