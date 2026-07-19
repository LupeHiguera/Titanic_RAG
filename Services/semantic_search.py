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
    
    def _meaningful_query_words(self, text: str) -> set:
        """Query words that carry signal — stopwords like 'the'/'was' would
        otherwise inflate every overlap count and saturate scores at 1.0."""
        return {w for w in text.lower().split()
                if len(w) > 2 and w not in self._HIGHLIGHT_STOPWORDS}

    def _calculate_relevance_score(self, embedded_chunk: EmbeddedChunk,
                                  similarity_score: float, query: SearchQuery) -> float:
        """Calculate relevance score combining similarity and other factors."""
        relevance = similarity_score

        query_words = self._meaningful_query_words(query.text)
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

        query_words = self._meaningful_query_words(query.text)
        content_words = set(chunk.content.lower().split())
        matching_words = query_words.intersection(content_words)

        if matching_words:
            explanation += f"Key matching terms: {', '.join(sorted(matching_words))}. "

        explanation += f"Witness: {chunk.witness_name} from {chunk.metadata.source_type}."
        return explanation
    
    # Don't highlight stopwords — they appear so often they create wall-to-wall
    # marks that visually merge into one big highlighted region, hiding the
    # actually interesting matches.
    _HIGHLIGHT_STOPWORDS = frozenset({
        "the", "was", "and", "for", "you", "with", "that", "this", "are",
        "his", "her", "had", "have", "has", "did", "any", "all", "but",
        "not", "from", "they", "them", "their", "there", "what", "when",
        "where", "who", "how", "why", "did", "your", "our", "one",
    })

    def _highlight_terms(self, content: str, query_text: str) -> str:
        """Highlight query terms in the content as **word** for the UI to
        convert to <mark> tags. Uses word-boundary matching to avoid wrapping
        substrings (no more `Bel**fast**` or `ra**the**r`)."""
        query_words = [w for w in query_text.lower().split()
                       if len(w) > 2 and w not in self._HIGHLIGHT_STOPWORDS]
        highlighted = content
        seen = set()  # dedupe so duplicate query words don't double-wrap

        for word in query_words:
            if word in seen:
                continue
            seen.add(word)
            # \b...\b anchors prevent substring matches like "fast" in "Belfast".
            pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
            highlighted = pattern.sub(lambda m: f"**{m.group(0)}**", highlighted)

        return highlighted
    
    # Contradictions need witness *diversity*: the user's top_k (default 5)
    # often yields chunks from only 1-2 witnesses — zero pairs to compare.
    # Over-fetch, then let the detector take the best chunk per witness.
    _CONTRADICTION_FETCH_MIN = 15

    def get_related_contradictions(self, query: SearchQuery,
                                   min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """Find contradictory statements across witnesses for the query.

        Delegates to ContradictionDetector for LLM-based pairwise comparison.
        Verdicts are cached, so repeat queries hit cache instead of the LLM.

        A witness_name filter means "contradictions involving this witness":
        their chunks are searched alongside an unfiltered pass (a filter that
        pinned every chunk to one witness could never produce a pair).
        """
        witness_filter = query.filters.get("witness_name")
        fetch_k = max(query.top_k * 3, self._CONTRADICTION_FETCH_MIN)

        results = self.search(SearchQuery(
            text=query.text, top_k=fetch_k, filters=dict(query.filters),
            similarity_threshold=query.similarity_threshold,
        ))
        if witness_filter:
            other_filters = {k: v for k, v in query.filters.items() if k != "witness_name"}
            field_results = self.search(SearchQuery(
                text=query.text, top_k=fetch_k, filters=other_filters,
                similarity_threshold=query.similarity_threshold,
            ))
            seen = set()
            merged = []
            for r in results + field_results:
                key = (r.chunk.chunk.witness_name, r.chunk.chunk.content)
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
            results = merged

        if not results:
            return []

        if self._contradiction_detector is None:
            from Services.contradiction_detector import ContradictionDetector
            self._contradiction_detector = ContradictionDetector()

        chunks = [r.chunk for r in results]
        contradictions = self._contradiction_detector.detect(chunks, query.text)

        if witness_filter:
            contradictions = [
                c for c in contradictions
                if witness_filter in (c.witness_a, c.witness_b)
            ]

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
                "source_a": c.source_a,
                "source_b": c.source_b,
                "page_a": c.page_a,
                "page_b": c.page_b,
                "role_a": c.role_a,
                "role_b": c.role_b,
                "same_person": c.same_person,
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