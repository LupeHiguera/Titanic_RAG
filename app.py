import logging
import os
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from Services.semantic_search import SemanticSearchEngine, SearchQuery
from Services.embeddings import EmbeddingService
from Services.vector_storage import PineconeVectorStore
from Services.witness_index import WitnessIndex

# Load environment variables from .env file
load_dotenv()

# Log exceptions server-side; never include raw error strings in HTTP responses.
logger = logging.getLogger(__name__)

# Per-IP rate limiter. App Runner sits in front of the app, so the real
# client IP arrives via X-Forwarded-For — uvicorn must run with
# --proxy-headers --forwarded-allow-ips='*' for get_remote_address to see it.
# Limits are tunable via env (defaults set for portfolio traffic).
_SEARCH_RATE = os.getenv("RATE_LIMIT_SEARCH", "30/minute")
_CONTRADICTIONS_RATE = os.getenv("RATE_LIMIT_CONTRADICTIONS", "10/minute")
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Titanic Historical RAG", description="Search Titanic witness testimonies")
app.state.limiter = limiter


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    # `detail` key so the frontend surfaces a real message instead of the
    # generic "Search failed" (slowapi's default handler uses an `error` key).
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit reached ({exc.detail}) — "
                           "give it a minute and try again."},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)

# CORS: comma-separated list in ALLOWED_ORIGINS, defaults to localhost only.
# Set ALLOWED_ORIGINS=* in dev if you really want wildcard.
_allowed = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
allow_origins = ["*"] if _allowed.strip() == "*" else [o.strip() for o in _allowed.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Search system is initialized lazily: constructing PineconeVectorStore makes
# network calls, and doing that at import time means the container can't even
# boot (or serve /health) when Pinecone hiccups.
embedding_service = None
vector_store = None
search_engine = None
witness_index = WitnessIndex()
_init_lock = threading.Lock()  # endpoints run in a threadpool


def get_vector_store() -> PineconeVectorStore:
    global vector_store
    with _init_lock:
        if vector_store is None:
            vector_store = PineconeVectorStore()
        return vector_store


def get_search_engine():
    """Initialize search engine lazily with proper error handling."""
    global embedding_service, search_engine
    try:
        store = get_vector_store()
        with _init_lock:
            if search_engine is None:
                # text-embedding-3-large @ 1024 dimensions to match Pinecone data
                embedding_service = EmbeddingService(model="text-embedding-3-large", dimensions=1024)
                search_engine = SemanticSearchEngine(embedding_service, store)
            return search_engine
    except ValueError as e:
        if "API key" in str(e).lower():
            if "pinecone" in str(e).lower():
                raise HTTPException(status_code=500,
                                    detail="Pinecone API key not configured. Please set PINECONE_API_KEY environment variable.")
            else:
                raise HTTPException(status_code=500,
                                    detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
        logger.exception("Failed to initialize search engine")
        raise HTTPException(status_code=500, detail="Failed to initialize search engine")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=25)
    similarity_threshold: float = Field(0.4, ge=0.0, le=1.0)
    witness_name: Optional[str] = Field(None, max_length=120)
    source_type: Optional[str] = Field(None, pattern="^(us_inquiry|british_inquiry)$")


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int


class ContradictionSearchRequest(SearchRequest):
    min_confidence: float = Field(0.6, ge=0.0, le=1.0)


class ContradictionSearchResponse(BaseModel):
    query: str
    contradictions: List[Dict[str, Any]]
    total_contradictions: int


@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        stats = get_vector_store().get_collection_stats()
        return {
            "status": "healthy",
            "vector_store": "operational",
            "total_documents": stats.get("total_chunks", 0)
        }
    except Exception:
        logger.exception("/health failed")
        raise HTTPException(status_code=500, detail="System unhealthy")


# NB: search endpoints are deliberately sync (`def`) — FastAPI runs them in a
# threadpool. The embedding, Pinecone, and Anthropic calls they make are all
# blocking; as `async def` they'd freeze the event loop (incl. /health) for
# the duration of every request.
@app.post("/search", response_model=SearchResponse)
@limiter.limit(_SEARCH_RATE)
def search_documents(request: Request, payload: SearchRequest):
    """Search historical documents."""
    try:
        filters = {}
        if payload.witness_name:
            filters["witness_name"] = payload.witness_name
        if payload.source_type:
            filters["source_type"] = payload.source_type

        search_query = SearchQuery(
            text=payload.query,
            top_k=payload.top_k,
            filters=filters,
            similarity_threshold=payload.similarity_threshold
        )

        engine = get_search_engine()
        results = engine.search(search_query)

        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.highlighted_content or result.chunk.chunk.content,
                "witness_name": result.chunk.chunk.witness_name,
                "source_type": result.chunk.chunk.metadata.source_type,
                "page_number": result.chunk.chunk.metadata.page_number,
                "role": result.chunk.chunk.metadata.role,
                "ship": result.chunk.chunk.metadata.ship,
                "similarity_score": round(result.similarity_score, 3),
                "relevance_score": round(result.relevance_score, 3),
                "explanation": result.relevance_explanation
            })

        return SearchResponse(
            query=payload.query,
            results=formatted_results,
            total_results=len(formatted_results)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("/search failed")
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/search/contradictions", response_model=ContradictionSearchResponse)
@limiter.limit(_CONTRADICTIONS_RATE)
def search_contradictions(request: Request, payload: ContradictionSearchRequest):
    """Find contradictory statements across witnesses for a query."""
    try:
        filters = {}
        if payload.witness_name:
            filters["witness_name"] = payload.witness_name
        if payload.source_type:
            filters["source_type"] = payload.source_type

        search_query = SearchQuery(
            text=payload.query,
            top_k=payload.top_k,
            filters=filters,
            similarity_threshold=payload.similarity_threshold,
        )

        engine = get_search_engine()
        contradictions = engine.get_related_contradictions(
            search_query, min_confidence=payload.min_confidence
        )

        return ContradictionSearchResponse(
            query=payload.query,
            contradictions=contradictions,
            total_contradictions=len(contradictions),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("/search/contradictions failed")
        raise HTTPException(status_code=500, detail="Contradiction search failed")


@app.get("/documents")
def get_documents():
    """Get available document metadata."""
    try:
        from Services.british_witness_index import british_witness_index, canonical_witness_name
        stats = get_vector_store().get_collection_stats()
        # Union on canonical names so the 15 cross-inquiry witnesses aren't
        # counted twice.
        unique = len(
            {w.name for w in witness_index.get_unique_witnesses()}
            | {canonical_witness_name(w.name)
               for w in british_witness_index.get_unique_witnesses()}
        )
        return {
            "total_chunks": stats.get("total_chunks", 0),
            "unique_witnesses": unique,
            "document_types": ["us_inquiry", "british_inquiry"],
            "collection_info": {
                "name": stats.get("index_name", "titanic-rag"),
                "storage_type": "Pinecone",
                "dimensions": stats.get("dimension", 1024),
                "model": "text-embedding-3-large"
            }
        }
    except Exception:
        logger.exception("/documents failed")
        raise HTTPException(status_code=500, detail="Failed to get documents")


@app.get("/witnesses")
def get_witnesses(search: Optional[str] = None):
    """Get the canonical list of witnesses, optionally filtered by search term.

    Sourced from WitnessIndex (the source of truth for witness attribution),
    not from the vector store — which doesn't enumerate metadata efficiently.
    """
    try:
        from Services.british_witness_index import british_witness_index
        names = sorted(
            {w.name for w in witness_index.get_unique_witnesses()}
            | {w.name for w in british_witness_index.get_unique_witnesses()}
        )

        if search and search.strip():
            q = search.strip().lower()
            names = [n for n in names if q in n.lower()]

        return {
            "witnesses": names,
            "total_count": len(names),
        }

    except Exception:
        logger.exception("/witnesses failed")
        raise HTTPException(status_code=500, detail="Failed to get witnesses")


# Mount static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
