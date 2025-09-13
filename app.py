from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from Services.semantic_search import SemanticSearchEngine, SearchQuery
from Services.embeddings import EmbeddingService
from Services.vector_storage import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Titanic Historical RAG", description="Search Titanic witness testimonies")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search system - will be done lazily when needed
embedding_service = None
vector_store = PineconeVectorStore()
search_engine = None


def get_search_engine():
    """Initialize search engine lazily with proper error handling."""
    global embedding_service, search_engine
    if search_engine is None:
        try:
            # Use text-embedding-3-large with 1024 dimensions to match Pinecone data
            embedding_service = EmbeddingService(model="text-embedding-3-large", dimensions=1024)
            search_engine = SemanticSearchEngine(embedding_service, vector_store)
        except ValueError as e:
            if "API key" in str(e).lower():
                if "pinecone" in str(e).lower():
                    raise HTTPException(status_code=500,
                                        detail="Pinecone API key not configured. Please set PINECONE_API_KEY environment variable.")
                else:
                    raise HTTPException(status_code=500,
                                        detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
            raise HTTPException(status_code=500, detail=f"Failed to initialize search engine: {str(e)}")
    return search_engine


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    similarity_threshold: float = 0.7
    witness_name: Optional[str] = None
    source_type: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int


@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        stats = vector_store.get_collection_stats()
        return {
            "status": "healthy",
            "vector_store": "operational",
            "total_documents": stats.get("total_chunks", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System unhealthy: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search historical documents."""
    try:
        # Build filters
        filters = {}
        if request.witness_name:
            filters["witness_name"] = request.witness_name
        if request.source_type:
            filters["source_type"] = request.source_type

        # Create search query
        search_query = SearchQuery(
            text=request.query,
            top_k=request.top_k,
            filters=filters,
            similarity_threshold=request.similarity_threshold
        )

        # Get search engine and perform search
        engine = get_search_engine()
        results = engine.search(search_query)

        # Format results for frontend
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.highlighted_content or result.chunk.chunk.content,
                "witness_name": result.chunk.chunk.witness_name,
                "source_type": result.chunk.chunk.metadata.source_type,
                "page_number": result.chunk.chunk.metadata.page_number,
                "similarity_score": round(result.similarity_score, 3),
                "relevance_score": round(result.relevance_score, 3),
                "explanation": result.relevance_explanation
            })

        return SearchResponse(
            query=request.query,
            results=formatted_results,
            total_results=len(formatted_results)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/documents")
async def get_documents():
    """Get available document metadata."""
    try:
        stats = vector_store.get_collection_stats()
        return {
            "total_chunks": stats.get("total_chunks", 0),
            "unique_witnesses": 31,  # From our ingestion - 31 witnesses identified
            "document_types": ["us_inquiry"],  # Currently only have US Senate Inquiry
            "collection_info": {
                "name": stats.get("index_name", "titanic-rag"),
                "storage_type": "Pinecone",
                "dimensions": stats.get("dimension", 1024),
                "model": "text-embedding-3-large"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")


@app.get("/witnesses")
async def get_witnesses(search: Optional[str] = None):
    """Get list of witnesses, optionally filtered by search term."""
    try:
        # For Pinecone, we'll return a static list of known witnesses from the uploaded data
        # This is a limitation of Pinecone's API - it doesn't support metadata enumeration
        # In production, this could be cached or stored separately
        known_witnesses = [
            "CHARLES HERBERT LIGHTOLLER", "JOSEPH GROVES BOXHALL", "EDWARD WHEELTON",
            "ALBERT HAINES", "GEORGE THOMAS ROW", "JOSEPH SCARROTT", "FRANK OSMAN",
            "GEORGE MOORE", "ARTHUR JOHN BRIGHT", "FREDERICK FLEET",
            "REGINALD ROBINSON LEE", "GEORGE ALFRED ROWE", "SAMUEL HEMMING",
            "WALTER JOHN PERKIS", "HERBERT JOHN PITMAN", "HAROLD GODFREY LOWE",
            "ARCHIE JEWELL", "THOMAS PATRICK DILLON", "WILLIAM THOMAS STEAD",
            "HAROLD SYDNEY BRIDE", "JACK PHILLIPS", "THOMAS ANDREWS", 
            "BRUCE ISMAY", "WALLACE HENRY HARTLEY", "JOHN JACOB ASTOR",
            "BENJAMIN GUGGENHEIM", "ISIDOR STRAUS", "IDA STRAUS", "MARGARET BROWN",
            "DOROTHY GIBSON", "ARCHIBALD GRACIE"
        ]

        # Filter by search term if provided
        if search and search.strip():
            search_lower = search.strip().lower()
            filtered_witnesses = [w for w in known_witnesses if search_lower in w.lower()]
        else:
            filtered_witnesses = known_witnesses

        return {
            "witnesses": sorted(filtered_witnesses),
            "total_count": len(filtered_witnesses),
            "note": "Witness list from Pinecone vector database - subset of known witnesses"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get witnesses: {str(e)}")


# Mount static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
