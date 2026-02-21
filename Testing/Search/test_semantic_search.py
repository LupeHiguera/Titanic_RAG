import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock

# Add the root directory to path so we can import Services
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from Services.semantic_search import SemanticSearchEngine, SearchResult, SearchQuery
from Services.embeddings import EmbeddingService, EmbeddedChunk
from Services.vector_storage import VectorStore
from Services.chunking import WitnessChunk, ChunkMetadata


class TestSemanticSearchEngine:
    
    @pytest.fixture
    def mock_embedding_service(self):
        service = Mock(spec=EmbeddingService)
        service.embed_text.return_value = np.random.rand(1536)  # OpenAI ada-002 dimension
        return service
    
    @pytest.fixture
    def mock_vector_store(self):
        store = Mock(spec=VectorStore)
        return store
    
    @pytest.fixture
    def search_engine(self, mock_embedding_service, mock_vector_store):
        return SemanticSearchEngine(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            default_top_k=5,
            similarity_threshold=0.7
        )
    
    @pytest.fixture
    def real_ismay_search_results(self):
        """Real Ismay testimony chunks from one page.pdf for testing semantic search"""
        metadata1 = ChunkMetadata(
            document_name="one page.pdf",
            source_type="us_inquiry",
            page_number=1,
            credibility_score=0.0,
            chunk_index=0,
            total_chunks_for_witness=4
        )
        
        chunk1 = WitnessChunk(
            content="Q: First state your full name, please? A: Joseph Bruce Ismay. Q: And your place of residence? A: Liverpool. Q: And your occupation? A: Ship owner.",
            witness_name="Joseph Bruce Ismay",
            metadata=metadata1
        )
        
        embedded1 = EmbeddedChunk(
            chunk=chunk1,
            embedding=np.random.rand(1536)  # OpenAI ada-002 dimension
        )
        
        metadata2 = ChunkMetadata(
            document_name="one page.pdf", 
            source_type="us_inquiry",
            page_number=1,
            credibility_score=0.0,
            chunk_index=1,
            total_chunks_for_witness=4
        )
        
        chunk2 = WitnessChunk(
            content="Q: As such officer, were you officially designated to make the trial trip of the Titanic? A: No. Q: Were you a voluntary passenger? A: A voluntary passenger, yes.",
            witness_name="Joseph Bruce Ismay",
            metadata=metadata2
        )
        
        embedded2 = EmbeddedChunk(
            chunk=chunk2,
            embedding=np.random.rand(1536)
        )
        
        metadata3 = ChunkMetadata(
            document_name="one page.pdf",
            source_type="us_inquiry", 
            page_number=1,
            credibility_score=0.0,
            chunk_index=2,
            total_chunks_for_witness=4
        )
        
        chunk3 = WitnessChunk(
            content="The ship was built in Belfast. She was the latest thing in the art of shipbuilding; absolutely no money was spared in her construction. She was not built by contract.",
            witness_name="Joseph Bruce Ismay",
            metadata=metadata3
        )
        
        embedded3 = EmbeddedChunk(
            chunk=chunk3,
            embedding=np.random.rand(1536)
        )
        
        metadata4 = ChunkMetadata(
            document_name="one page.pdf",
            source_type="us_inquiry",
            page_number=1, 
            credibility_score=0.0,
            chunk_index=3,
            total_chunks_for_witness=4
        )
        
        chunk4 = WitnessChunk(
            content="In the first place, I would like to express my sincere grief at this deplorable catastrophe. We welcome the fullest inquiry. We have nothing to conceal; nothing to hide.",
            witness_name="Joseph Bruce Ismay",
            metadata=metadata4
        )
        
        embedded4 = EmbeddedChunk(
            chunk=chunk4,
            embedding=np.random.rand(1536)
        )
        
        return [
            (embedded1, 0.95),  # Personal information chunk - highest similarity
            (embedded2, 0.87),  # Titanic trip designation - high similarity  
            (embedded3, 0.82),  # Ship construction details - medium similarity
            (embedded4, 0.79)   # Grief/inquiry statement - lower similarity
        ]
    
    def test_search_basic_query_returns_results(self, search_engine, mock_vector_store, real_ismay_search_results):
        mock_vector_store.query.return_value = real_ismay_search_results
        
        query = SearchQuery(
            text="Who is Joseph Bruce Ismay and what was his role?",
            top_k=5,
            filters={}
        )
        
        results = search_engine.search(query)
        
        assert isinstance(results, list)
        assert len(results) == len(real_ismay_search_results)
        
        for result in results:
            assert isinstance(result, SearchResult)
            assert hasattr(result, 'chunk')
            assert hasattr(result, 'similarity_score')
            assert hasattr(result, 'relevance_explanation')
    
    def test_search_with_witness_filters(self, search_engine, mock_vector_store, real_ismay_search_results):
        mock_vector_store.query.return_value = real_ismay_search_results
        
        query = SearchQuery(
            text="Titanic construction and ship owner testimony",
            filters={"witness_name": "Joseph Bruce Ismay"}
        )
        
        results = search_engine.search(query)
        
        mock_vector_store.query.assert_called_once()
        call_args = mock_vector_store.query.call_args
        assert call_args[1]['filters']['witness_name'] == "Joseph Bruce Ismay"
    
    
    def test_search_with_document_source_filter(self, search_engine, mock_vector_store, real_ismay_search_results):
        mock_vector_store.query.return_value = real_ismay_search_results
        
        query = SearchQuery(
            text="US Senate inquiry testimony",
            filters={"source_type": "us_inquiry"}
        )
        
        results = search_engine.search(query)
        
        mock_vector_store.query.assert_called_once()
        call_args = mock_vector_store.query.call_args
        assert call_args[1]['filters']['source_type'] == "us_inquiry"
    
    def test_search_ismay_titanic_construction(self, search_engine, mock_vector_store, real_ismay_search_results):
        """Test searching for Ismay's testimony about Titanic construction"""
        mock_vector_store.query.return_value = real_ismay_search_results
        
        query = SearchQuery(
            text="ship construction Belfast shipbuilding money spared",
            top_k=3,
            filters={}
        )
        
        results = search_engine.search(query)
        
        assert len(results) >= 1
        # Should find the construction-related chunk
        construction_found = any("Belfast" in result.chunk.chunk.content for result in results)
        assert construction_found

    def test_rerank_results_by_relevance(self, search_engine, real_ismay_search_results):
        query = SearchQuery(text="ship owner managing director")
        
        reranked = search_engine._rerank_results(real_ismay_search_results, query)
        
        assert len(reranked) == len(real_ismay_search_results)
        assert all(isinstance(result, SearchResult) for result in reranked)
        
        # Check that results are ranked by relevance
        relevance_scores = [r.relevance_score for r in reranked]
        assert relevance_scores == sorted(relevance_scores, reverse=True)
    
    def test_explain_relevance_provides_reasoning(self, search_engine):
        query = SearchQuery(text="ship owner managing director White Star Line")
        
        chunk = WitnessChunk(
            content="Q: And your occupation? A: Ship owner. Q: Are you an officer of the White Star Line? A: I am. Q: In what capacity? A: Managing Director.",
            witness_name="Joseph Bruce Ismay",
            metadata=Mock()
        )
        
        explanation = search_engine._explain_relevance(query, chunk, 0.9)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert any(keyword in explanation.lower() for keyword in ["ship", "owner", "managing", "director"])
    
    def test_search_handles_empty_query(self, search_engine):
        query = SearchQuery(text="")
        
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            search_engine.search(query)
    
    def test_search_handles_no_results(self, search_engine, mock_vector_store):
        mock_vector_store.query.return_value = []
        
        query = SearchQuery(text="nonexistent topic")
        results = search_engine.search(query)
        
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_search_applies_similarity_threshold(self, search_engine, mock_vector_store):
        low_similarity_results = [
            (Mock(), 0.3),  # Below threshold
            (Mock(), 0.8),  # Above threshold
        ]
        mock_vector_store.query.return_value = low_similarity_results
        
        query = SearchQuery(text="test query")
        results = search_engine.search(query)
        
        assert len(results) == 1  # Only one above threshold
    
    def test_get_related_contradictions(self, search_engine, mock_vector_store, real_ismay_search_results):
        mock_vector_store.query.return_value = real_ismay_search_results
        
        query = SearchQuery(text="men in lifeboats")
        contradictions = search_engine.get_related_contradictions(query)
        
        assert isinstance(contradictions, list)
        if len(contradictions) > 0:
            for contradiction in contradictions:
                assert "conflicting_statements" in contradiction
                assert "topic" in contradiction
                assert "confidence_score" in contradiction
    
    def test_search_with_date_range_filter(self, search_engine, mock_vector_store, real_ismay_search_results):
        mock_vector_store.query.return_value = real_ismay_search_results
        
        query = SearchQuery(
            text="testimony about the disaster",
            filters={"page_range": {"min": 200, "max": 300}}
        )
        
        results = search_engine.search(query)
        
        mock_vector_store.query.assert_called_once()
    
    def test_highlight_key_terms_in_results(self, search_engine, real_ismay_search_results):
        query = SearchQuery(text="women children lifeboats")
        
        results = search_engine._rerank_results(real_ismay_search_results, query)
        
        for result in results:
            if hasattr(result, 'highlighted_content'):
                assert isinstance(result.highlighted_content, str)
    
    def test_search_performance_with_large_result_set(self, search_engine, mock_vector_store):
        # Mock should return only top_k results (10) when called with top_k=10
        mock_result_set = [(Mock(), 0.8) for _ in range(10)]
        mock_vector_store.query.return_value = mock_result_set
        
        query = SearchQuery(text="test query", top_k=10)
        results = search_engine.search(query)
        
        # Should call vector store with top_k=10 and return at most 10 results  
        mock_vector_store.query.assert_called_once()
        call_args = mock_vector_store.query.call_args
        assert call_args[1]['top_k'] == 10
        assert len(results) <= 10
    
    def test_search_aggregates_witness_perspectives(self, search_engine, mock_vector_store, real_ismay_search_results):
        mock_vector_store.query.return_value = real_ismay_search_results
        
        query = SearchQuery(text="lifeboat loading")
        summary = search_engine.get_witness_perspective_summary(query)
        
        assert isinstance(summary, dict)
        assert "officer_perspective" in summary or "passenger_perspective" in summary or "management_perspective" in summary
        assert "conflicting_accounts" in summary