# Semantic Search & Vector Storage Implementation Plan

## Overview
Implement the core RAG search functionality to complete the Titanic Historical Document system. Focus on surfacing contradictions between witness testimonies rather than providing single "truth" answers.

## Missing Components to Implement

### 1. Vector Storage (`vector_storage.py`)

**Classes Needed:**
- `ChromaVectorStore` - Local development storage
- `PineconeVectorStore` - Production cloud storage  
- Base `VectorStore` interface

**Key Methods:**
```python
# ChromaVectorStore
- store_chunks(embedded_chunks: List[EmbeddedChunk]) -> List[str]
- query(query_vector: np.ndarray, top_k: int, filters: Dict) -> List[Tuple[EmbeddedChunk, float]]
- delete_chunks(chunk_ids: List[str]) -> bool
- get_collection_stats() -> Dict
- backup_collection(backup_path: str) -> bool

# PineconeVectorStore  
- store_chunks(embedded_chunks: List[EmbeddedChunk]) -> List[str]
- query(query_vector: np.ndarray, top_k: int, filters: Dict) -> List[Tuple[EmbeddedChunk, float]]
- delete_chunks(chunk_ids: List[str]) -> bool
```

**Features Required:**
- Metadata filtering (witness credibility, document source, page ranges)
- Persistence for ChromaDB
- Pinecone cloud integration for production
- Backup/restore for local development

### 2. Semantic Search (`semantic_search.py`)

**Classes Needed:**
- `SemanticSearchEngine` - Main search orchestrator
- `SearchQuery` - Query structure with filters
- `SearchResult` - Enhanced result with explanations

**Core Functionality:**
```python
# SemanticSearchEngine
- search(query: SearchQuery) -> List[SearchResult]
- get_related_contradictions(query: SearchQuery) -> List[Dict]
- get_witness_perspective_summary(query: SearchQuery) -> Dict
- _rerank_results(results, query) -> List[SearchResult] 
- _explain_relevance(query, chunk, similarity) -> str
```

**Key Features:**
- **Contradiction Detection**: Surface conflicting testimonies on same topics
- **Witness Perspective Grouping**: Officer vs passenger vs crew viewpoints
- **Credibility-Based Ranking**: Weight results by witness reliability
- **Relevance Explanations**: Show why each result was returned
- **Multi-perspective Results**: Don't hide disagreements

### 3. Integration Points

**Document Processing Pipeline:**
```
PDF → DocumentIngestion → Chunking → Embeddings → VectorStorage
                ↓
User Query → SemanticSearch → VectorStorage → Ranked Results + Contradictions
```

**Filter Types:**
- `witness_name`: Specific witness testimony
- `source_type`: "us_inquiry" | "british_inquiry" | "other"  
- `credibility_score`: Minimum reliability threshold
- `page_range`: Specific document sections
- `min_credibility`: Filter by witness authority level

## Implementation Priority

### Phase 1: Vector Storage Foundation
1. **ChromaVectorStore** for local development
   - Collection creation and persistence
   - Metadata filtering capabilities
   - Basic CRUD operations

2. **PineconeVectorStore** for production
   - API integration setup
   - Batch upload optimization
   - Index management

### Phase 2: Search Engine Core
1. **Basic Semantic Search**
   - Query embedding and similarity matching
   - Result ranking and filtering
   - Top-k retrieval with thresholds

2. **Contradiction Detection System**
   - Topic-based grouping of chunks
   - Cross-witness comparison logic
   - Confidence scoring for conflicts

### Phase 3: Enhanced User Experience  
1. **Relevance Explanations**
   - Keyword highlighting in results
   - Reasoning for result selection
   - Witness credibility context

2. **Perspective Aggregation**
   - Officer vs passenger viewpoints
   - Timeline-based contradictions
   - Source inquiry comparisons

## Technical Requirements

### Dependencies
```python
# Add to requirements.txt
chromadb>=0.4.0      # Local vector storage
pinecone-client>=2.2.0   # Cloud vector storage  
sentence-transformers>=2.2.0  # Fallback embeddings
```

### Data Structures
```python
@dataclass
class SearchQuery:
    text: str
    top_k: int = 5
    filters: Dict[str, Any] = field(default_factory=dict)
    similarity_threshold: float = 0.7

@dataclass  
class SearchResult:
    chunk: WitnessChunk
    similarity_score: float
    relevance_score: float
    relevance_explanation: str
    highlighted_content: Optional[str] = None
```

### Configuration
```python
# Local Development
VECTOR_STORE = "chroma"
CHROMA_PERSIST_DIR = "./chroma_db"

# Production
VECTOR_STORE = "pinecone"  
PINECONE_INDEX = "titanic-prod"
PINECONE_ENVIRONMENT = "us-east1-gcp"
```

## Success Criteria

### Week 3 Goals (Current)
- [ ] ChromaDB local storage working
- [ ] Basic semantic search returning relevant passages  
- [ ] Contradiction detection for simple cases
- [ ] All existing tests passing with real implementations

### Week 4 Goals (Next)
- [ ] Pinecone production storage
- [ ] Advanced contradiction scoring
- [ ] Side-by-side comparison UI data
- [ ] Multi-witness perspective aggregation

## Testing Strategy

### Unit Tests
- Vector storage CRUD operations
- Search query parsing and filtering
- Contradiction detection algorithms
- Result ranking and explanations

### Integration Tests  
- Full pipeline: PDF → chunks → embeddings → storage → search
- Cross-witness contradiction detection with real data
- Performance testing with 1000+ document chunks

### Real Data Validation
- Query: "Did the band play?" → Multiple witness perspectives
- Query: "Lifeboat loading procedure" → Officer vs passenger conflicts
- Query: "Women and children first" → Implementation variations

## Notes for Implementation

### Historical Context Requirements
- **Preserve Bias**: Don't normalize conflicting accounts
- **Credibility Weighting**: Officers > Crew > Passengers (generally)
- **Source Context**: British inquiry (formal) vs US Senate (aggressive)
- **Citation Accuracy**: Every claim must link to specific page/line

### Technical Priorities
1. **Chunking Quality**: Q&A pairs must stay together
2. **Search Relevance**: Multiple perspectives > single answer
3. **Contradiction Visibility**: Make conflicts obvious, not hidden
4. **Performance**: Sub-second search for 2000+ pages of documents

This implementation will complete the core RAG functionality needed for the Titanic Historical Document exploration system.