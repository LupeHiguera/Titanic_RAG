from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from Services.semantic_search import SemanticSearchEngine, SearchQuery
from Services.embeddings import EmbeddingService
from Services.vector_storage import ChromaVectorStore

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
vector_store = ChromaVectorStore()
search_engine = None

def get_search_engine():
    """Initialize search engine lazily with proper error handling."""
    global embedding_service, search_engine
    if search_engine is None:
        try:
            embedding_service = EmbeddingService()
            search_engine = SemanticSearchEngine(embedding_service, vector_store)
        except ValueError as e:
            if "API key" in str(e):
                raise HTTPException(status_code=500, detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
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
            "unique_witnesses": stats.get("unique_witnesses", []),
            "document_types": stats.get("document_types", []),
            "collection_info": {
                "name": stats.get("collection_name", ""),
                "persist_dir": stats.get("persist_dir", "")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

# Mount static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)