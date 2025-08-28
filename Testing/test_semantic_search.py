import pytest
import numpy as np
from unittest.mock import Mock

from semantic_search import SemanticSearchEngine, SearchResult, SearchQuery
from Services.embeddings import EmbeddingService, EmbeddedChunk
from vector_storage import VectorStore
from Services.chunking import WitnessChunk, ChunkMetadata


class TestSemanticSearchEngine:
    
    @pytest.fixture
    def mock_embedding_service(self):
        service = Mock(spec=EmbeddingService)
        service.embed_text.return_value = np.array([0.1, 0.2, 0.3] * 512)
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
    def sample_search_results(self):
        metadata1 = ChunkMetadata(
            document_name="British Inquiry Day 5",
            source_type="british_inquiry",
            page_number=247,
            credibility_score=0.9,
            chunk_index=0,
            total_chunks_for_witness=1
        )
        
        chunk1 = WitnessChunk(
            content="Q: What was your position? A: I was Second Officer. We followed the women and children first protocol.",
            witness_name="Charles Herbert Lightoller",
            metadata=metadata1
        )
        
        embedded1 = EmbeddedChunk(
            chunk=chunk1,
            embedding=np.array([0.1, 0.2, 0.3] * 512)
        )
        
        metadata2 = ChunkMetadata(
            document_name="US Senate Inquiry Day 12", 
            source_type="us_inquiry",
            page_number=891,
            credibility_score=0.7,
            chunk_index=0,
            total_chunks_for_witness=1
        )
        
        chunk2 = WitnessChunk(
            content="Q: Did you see men in lifeboats? A: Yes, when no women were present nearby.",
            witness_name="Hugh Woolner",
            metadata=metadata2
        )
        
        embedded2 = EmbeddedChunk(
            chunk=chunk2,
            embedding=np.array([0.4, 0.5, 0.6] * 512)
        )
        
        return [
            (embedded1, 0.95),
            (embedded2, 0.85)
        ]
    
    def test_search_basic_query_returns_results(self, search_engine, mock_vector_store, sample_search_results):
        mock_vector_store.query.return_value = sample_search_results
        
        query = SearchQuery(
            text="What happened with the lifeboats?",
            top_k=5,
            filters={}
        )
        
        results = search_engine.search(query)
        
        assert isinstance(results, list)
        assert len(results) == len(sample_search_results)
        
        for result in results:
            assert isinstance(result, SearchResult)
            assert hasattr(result, 'chunk')
            assert hasattr(result, 'similarity_score')
            assert hasattr(result, 'relevance_explanation')
    
    def test_search_with_witness_filters(self, search_engine, mock_vector_store, sample_search_results):
        mock_vector_store.query.return_value = sample_search_results
        
        query = SearchQuery(
            text="lifeboat procedures",
            filters={"witness_name": "Charles Herbert Lightoller"}
        )
        
        results = search_engine.search(query)
        
        mock_vector_store.query.assert_called_once()
        call_args = mock_vector_store.query.call_args
        assert call_args[1]['filters']['witness_name'] == "Charles Herbert Lightoller"
    
    def test_search_with_credibility_filtering(self, search_engine, mock_vector_store, sample_search_results):
        mock_vector_store.query.return_value = sample_search_results
        
        query = SearchQuery(
            text="officer testimony",
            filters={"min_credibility": 0.8}
        )
        
        results = search_engine.search(query)
        
        mock_vector_store.query.assert_called_once()
        call_args = mock_vector_store.query.call_args
        assert 'credibility_score' in call_args[1]['filters']
    
    def test_search_with_document_source_filter(self, search_engine, mock_vector_store, sample_search_results):
        mock_vector_store.query.return_value = sample_search_results
        
        query = SearchQuery(
            text="inquiry testimony",
            filters={"source_type": "british_inquiry"}
        )
        
        results = search_engine.search(query)
        
        mock_vector_store.query.assert_called_once()
        call_args = mock_vector_store.query.call_args
        assert call_args[1]['filters']['source_type'] == "british_inquiry"
    
    def test_rerank_results_by_relevance(self, search_engine, sample_search_results):
        query = SearchQuery(text="lifeboat women children")
        
        reranked = search_engine._rerank_results(sample_search_results, query)
        
        assert len(reranked) == len(sample_search_results)
        assert all(isinstance(result, SearchResult) for result in reranked)
        
        lifeboat_scores = [r.relevance_score for r in reranked if "lifeboat" in r.chunk.content.lower()]
        if len(lifeboat_scores) > 1:
            assert lifeboat_scores[0] >= lifeboat_scores[1]
    
    def test_explain_relevance_provides_reasoning(self, search_engine):
        query = SearchQuery(text="women and children first")
        
        chunk = WitnessChunk(
            content="We followed the women and children first protocol strictly",
            witness_name="Test Officer",
            metadata=Mock()
        )
        
        explanation = search_engine._explain_relevance(query, chunk, 0.9)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert any(keyword in explanation.lower() for keyword in ["women", "children", "protocol"])
    
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
    
    def test_get_related_contradictions(self, search_engine, mock_vector_store, sample_search_results):
        mock_vector_store.query.return_value = sample_search_results
        
        query = SearchQuery(text="men in lifeboats")
        contradictions = search_engine.get_related_contradictions(query)
        
        assert isinstance(contradictions, list)
        if len(contradictions) > 0:
            for contradiction in contradictions:
                assert "conflicting_statements" in contradiction
                assert "topic" in contradiction
                assert "confidence_score" in contradiction
    
    def test_search_with_date_range_filter(self, search_engine, mock_vector_store, sample_search_results):
        mock_vector_store.query.return_value = sample_search_results
        
        query = SearchQuery(
            text="testimony about the disaster",
            filters={"page_range": {"min": 200, "max": 300}}
        )
        
        results = search_engine.search(query)
        
        mock_vector_store.query.assert_called_once()
    
    def test_highlight_key_terms_in_results(self, search_engine, sample_search_results):
        query = SearchQuery(text="women children lifeboats")
        
        results = search_engine._rerank_results(sample_search_results, query)
        
        for result in results:
            if hasattr(result, 'highlighted_content'):
                assert isinstance(result.highlighted_content, str)
    
    def test_search_performance_with_large_result_set(self, search_engine, mock_vector_store):
        large_result_set = [(Mock(), 0.8) for _ in range(1000)]
        mock_vector_store.query.return_value = large_result_set
        
        query = SearchQuery(text="test query", top_k=10)
        results = search_engine.search(query)
        
        assert len(results) <= 10
    
    def test_search_aggregates_witness_perspectives(self, search_engine, mock_vector_store, sample_search_results):
        mock_vector_store.query.return_value = sample_search_results
        
        query = SearchQuery(text="lifeboat loading")
        summary = search_engine.get_witness_perspective_summary(query)
        
        assert isinstance(summary, dict)
        assert "officer_perspective" in summary or "passenger_perspective" in summary
        assert "conflicting_accounts" in summary