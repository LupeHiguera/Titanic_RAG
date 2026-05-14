# Titanic Historical RAG

A retrieval-augmented search engine over the 1912 Titanic inquiry transcripts that **surfaces contradictions between witnesses** instead of hiding them.

Most RAG systems try to give you one answer. The witnesses in the Titanic inquiries gave conflicting accounts on almost everything — speed, lifeboat counts, ice warnings, who gave which order. This tool embraces that. Ask "How many people were in Ismay's lifeboat?" and you'll see Ismay's "about 45" set next to Officer Lowe's "only twelve", with a confidence-scored explanation of why the statements conflict.

## What it does

- **Semantic search** across all witness testimony — 1,173 pages of US Senate Inquiry transcript, ~16,000 indexed chunks across 68 witnesses
- **Side-by-side contradiction view** — toggle "Show contradictions only" and the system uses an LLM to pairwise-compare witness statements and surface only the factual conflicts
- **Confidence scoring** — every flagged contradiction comes with a 0–1 confidence and a one-sentence explanation
- **Filterable** — narrow by witness, source type (US Senate / British Inquiry), or minimum confidence

## Stack

| Layer | Choice |
|---|---|
| PDF extraction | pymupdf |
| Embeddings | OpenAI `text-embedding-3-large` @ 1024d |
| Vector store | Pinecone (serverless, AWS) |
| Contradiction LLM | Claude Haiku 4.5 with structured JSON output |
| API | FastAPI + uvicorn |
| Frontend | Single-page HTML/JS, no framework |
| Verdict cache | SQLite (local), DynamoDB (planned for production) |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up .env in the project root (required keys)
echo "OPENAI_API_KEY=sk-..."           >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..."    >> .env
echo "PINECONE_API_KEY=pcsk_..."       >> .env

# Optional (defaults shown):
# echo "PINECONE_INDEX_NAME=titanic-rag"               >> .env
# echo "PINECONE_ENVIRONMENT=us-east-1"                >> .env
# echo "ALLOWED_ORIGINS=http://localhost:8000"         >> .env  # or "*" for wildcard
# echo "CACHE_BACKEND=sqlite"                          >> .env  # "dynamodb" is a stub

# 3. Verify Pinecone connectivity (creates the index on first run)
python Services/pinecone_upload.py --test

# 4. Ingest the corpus (~10-15 minutes, ~$0.10 in OpenAI embeddings)
python Services/pinecone_upload.py --full-ingest

# 5. Start the app
python app.py
# Then open http://localhost:8000
```

## Using it

The single-page UI has a search box and a filters panel. Try queries like:

- *"how fast was the ship going"*
- *"how many people were in Ismay's lifeboat"*
- *"when did the band stop playing"*
- *"ice warnings"*

Flip the **"Show contradictions only"** toggle and adjust the minimum confidence slider to surface only LLM-verified factual disagreements between witnesses. Each card shows witness A vs witness B, their specific claims, a confidence percentage, and a one-line explanation of why the statements conflict.

## API endpoints

| Method | Path | Body |
|---|---|---|
| `POST` | `/search` | `{ query, top_k, similarity_threshold, witness_name?, source_type? }` |
| `POST` | `/search/contradictions` | Same shape + `min_confidence` (default 0.6) |
| `GET` | `/witnesses?search=...` | Filter witness list |
| `GET` | `/documents` | Index metadata |
| `GET` | `/health` | Pinecone state + chunk count |

`POST /search/contradictions` returns:
```json
{
  "query": "how many people were in Ismay's lifeboat",
  "total_contradictions": 1,
  "contradictions": [
    {
      "witness_a": "Joseph Bruce Ismay",
      "witness_b": "Harold Godfrey Lowe",
      "claim_a": "Ismay's lifeboat had approximately forty-five people",
      "claim_b": "Ismay's lifeboat (Boat C) had only twelve people",
      "confidence": 0.95,
      "explanation": "The two witnesses provide directly conflicting specific counts: forty-five versus twelve, which cannot both be true.",
      "chunk_a": "...full source quote...",
      "chunk_b": "...full source quote..."
    }
  ]
}
```

## Source documents

| File | Inquiry | Pages | Status |
|---|---|---|---|
| `Text/USInq.pdf` | US Senate, 1912 | 1,173 | Fully ingested |
| `Text/British_Data.pdf` | British Board of Trade | 7 | Format sample only; full transcript pending |

Long-term goal is to ingest the full British Inquiry (~2,200 pages, ~25,000 numbered Q&As) so the system can surface cross-inquiry contradictions — where the same witness testified differently in the two proceedings.

## Roadmap

- **Now** — Killer feature shipped end-to-end (search → LLM verdicts → side-by-side UI)
- **Next** — Clean up legacy ChromaDB code, hook `/witnesses` to the canonical `WitnessIndex`
- **Soon** — British Inquiry pipeline (speaker-tag parser, separate witness index, ingest the full corpus)
- **Production** — DynamoDB-backed verdict cache, rate limiter on `/search/contradictions`, Dockerfile + AWS App Runner deploy, custom domain

See [CONTRADICTION_PLAN.md](CONTRADICTION_PLAN.md) for the implementation plan that drove the killer feature. See [CLAUDE.md](CLAUDE.md) for engineering-side architecture and conventions.

## License

Not yet licensed.
