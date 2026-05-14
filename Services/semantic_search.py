import numpy as np
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from Services.embeddings import EmbeddingService, EmbeddedChunk
from Services.vector_storage import VectorStore
from Services.chunking import WitnessChunk


@dataclass
class SearchQuery:
    """Query structure for semantic search with filters and parameters."""
    text: str
    top_k: int = 5
    filters: Dict[str, Any] = field(default_factory=dict)
    similarity_threshold: float = 0.5


@dataclass
class SearchResult:
    """Enhanced search result with explanations and metadata."""
    chunk: EmbeddedChunk
    similarity_score: float
    relevance_score: float
    relevance_explanation: str
    highlighted_content: Optional[str] = None


class SemanticSearchEngine:
    """Main search orchestrator for Titanic witness testimony semantic search."""
    
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore,
                 default_top_k: int = 5, similarity_threshold: float = 0.5,
                 contradiction_detector: Optional[Any] = None):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold
        self._contradiction_detector = contradiction_detector
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Main search method that returns ranked results with explanations."""
        if not query.text or query.text.strip() == "":
            raise ValueError("Search query cannot be empty")
        
        # Get query embedding
        query_embedding = self._get_query_embedding(query.text)
        
        # Query vector store
        vector_results = self.vector_store.query(
            query_vector=query_embedding,
            top_k=query.top_k or self.default_top_k,
            filters=query.filters
        )
        
        # Filter by similarity threshold
        filtered_results = [
            (embedded_chunk, similarity) for embedded_chunk, similarity in vector_results
            if similarity >= query.similarity_threshold
        ]
        
        # Re-rank and enhance results
        search_results = self._rerank_results(filtered_results, query)
        
        return search_results
    
    def _get_query_embedding(self, query_text: str) -> np.ndarray:
        """Get embedding for the search query."""
        # Check if embedding service has embed_text method, fallback to embed_chunk
        if hasattr(self.embedding_service, 'embed_text'):
            return self.embedding_service.embed_text(query_text)
        else:
            # Create a temporary chunk for embedding
            from Services.chunking import ChunkMetadata
            temp_metadata = ChunkMetadata(
                document_name="query",
                source_type="query",
                page_number=0,
                credibility_score=0.0,
                chunk_index=0,
                total_chunks_for_witness=1
            )
            temp_chunk = WitnessChunk(
                content=query_text,
                witness_name="query",
                metadata=temp_metadata
            )
            embedded_chunk = self.embedding_service.embed_chunk(temp_chunk)
            return embedded_chunk.embedding
    
    def _rerank_results(self, vector_results: List[Tuple[EmbeddedChunk, float]], 
                       query: SearchQuery) -> List[SearchResult]:
        """Re-rank results by relevance and add explanations."""
        search_results = []
        
        for embedded_chunk, similarity_score in vector_results:
            # Calculate relevance score (combining similarity with other factors)
            relevance_score = self._calculate_relevance_score(
                embedded_chunk, similarity_score, query
            )
            
            # Generate relevance explanation
            explanation = self._explain_relevance(query, embedded_chunk.chunk, similarity_score)
            
            # Highlight key terms
            highlighted_content = self._highlight_terms(
                embedded_chunk.chunk.content, query.text
            )
            
            search_result = SearchResult(
                chunk=embedded_chunk,
                similarity_score=similarity_score,
                relevance_score=relevance_score,
                relevance_explanation=explanation,
                highlighted_content=highlighted_content
            )
            
            search_results.append(search_result)
        
        # Sort by relevance score (descending)
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return search_results
    
    def _calculate_relevance_score(self, embedded_chunk: EmbeddedChunk,
                                  similarity_score: float, query: SearchQuery) -> float:
        """Calculate relevance score combining similarity and other factors."""
        relevance = similarity_score

        query_words = set(query.text.lower().split())
        content_words = set(embedded_chunk.chunk.content.lower().split())
        keyword_overlap = len(query_words.intersection(content_words))
        if keyword_overlap > 0:
            relevance += 0.1 * keyword_overlap

        if query.filters.get('witness_name'):
            if embedded_chunk.chunk.witness_name == query.filters['witness_name']:
                relevance += 0.05

        return min(relevance, 1.0)
    
    def _explain_relevance(self, query: SearchQuery, chunk: WitnessChunk,
                          similarity_score: float) -> str:
        """Generate explanation for why this result is relevant."""
        explanation = f"Similarity score: {similarity_score:.2f}. "

        query_words = set(query.text.lower().split())
        content_words = set(chunk.content.lower().split())
        matching_words = query_words.intersection(content_words)

        if matching_words:
            explanation += f"Key matching terms: {', '.join(sorted(matching_words))}. "

        explanation += f"Witness: {chunk.witness_name} from {chunk.metadata.source_type}."
        return explanation
    
    def _highlight_terms(self, content: str, query_text: str) -> str:
        """Highlight query terms in the content."""
        query_words = query_text.lower().split()
        highlighted = content

        for word in query_words:
            if len(word) > 2:  # only highlight meaningful words
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                highlighted = pattern.sub(lambda m: f"**{m.group(0)}**", highlighted)

        return highlighted
    
    def get_related_contradictions(self, query: SearchQuery,
                                   min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """Find contradictory statements across witnesses for the query.

        Delegates to ContradictionDetector for LLM-based pairwise comparison.
        Verdicts are cached, so repeat queries hit cache instead of the LLM.
        """
        results = self.search(query)
        if not results:
            return []

        if self._contradiction_detector is None:
            from Services.contradiction_detector import ContradictionDetector
            self._contradiction_detector = ContradictionDetector()

        chunks = [r.chunk for r in results]
        contradictions = self._contradiction_detector.detect(chunks, query.text)

        return [
            {
                "witness_a": c.witness_a,
                "witness_b": c.witness_b,
                "chunk_a": c.chunk_a,
                "chunk_b": c.chunk_b,
                "claim_a": c.claim_a,
                "claim_b": c.claim_b,
                "confidence": round(c.confidence, 3),
                "explanation": c.explanation,
            }
            for c in contradictions
            if c.confidence >= min_confidence
        ]
    
    def get_witness_perspective_summary(self, query: SearchQuery) -> Dict[str, Any]:
        """Get summary of different witness perspectives on a topic."""
        results = self.search(query)
        
        # Group results by witness type/role
        perspectives = defaultdict(list)
        
        for result in results:
            witness_name = result.chunk.chunk.witness_name
            source_type = result.chunk.chunk.metadata.source_type
            
            # Categorize witnesses (simple categorization for now)
            if "officer" in witness_name.lower():
                category = "officer_perspective"
            elif "crew" in witness_name.lower():
                category = "crew_perspective"
            elif "ismay" in witness_name.lower():
                category = "management_perspective"
            else:
                category = "passenger_perspective"
            
            perspectives[category].append({
                "witness": witness_name,
                "content": result.chunk.chunk.content,
                "source": source_type,
                "relevance": result.relevance_score
            })
        
        # Add conflicting accounts detection
        summary = dict(perspectives)
        summary["conflicting_accounts"] = len(perspectives) > 1
        
        return summary