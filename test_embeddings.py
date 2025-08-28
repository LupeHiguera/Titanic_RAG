import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from embeddings import EmbeddingService, EmbeddedChunk
from chunking import WitnessChunk, ChunkMetadata


class TestEmbeddingService:
    
    @pytest.fixture
    def embedding_service(self):
        return EmbeddingService(
            provider="openai",
            model="text-embedding-ada-002",
            api_key="test-key"
        )
    
    @pytest.fixture
    def sample_chunks(self):
        metadata1 = ChunkMetadata(
            document_name="British Inquiry Day 5",
            source_type="british_inquiry",
            page_number=247,
            credibility_score=0.9,
            chunk_index=0,
            total_chunks_for_witness=2
        )
        
        metadata2 = ChunkMetadata(
            document_name="US Senate Inquiry Day 12",
            source_type="us_inquiry", 
            page_number=891,
            credibility_score=0.7,
            chunk_index=0,
            total_chunks_for_witness=1
        )
        
        return [
            WitnessChunk(
                content="Q: What was your position? A: I was Second Officer. We lowered lifeboats in order, women and children first.",
                witness_name="Charles Herbert Lightoller",
                metadata=metadata1
            ),
            WitnessChunk(
                content="Q: Did you see men in lifeboats? A: Yes, I saw men getting into lifeboats when no women were nearby.",
                witness_name="Hugh Woolner", 
                metadata=metadata2
            )
        ]
    
    @patch('embeddings.OpenAI')
    def test_embed_single_chunk_returns_vector(self, mock_openai_class, embedding_service, sample_chunks):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3] * 512)]
        mock_client.embeddings.create.return_value = mock_response
        
        # Recreate service to use mocked client
        embedding_service = EmbeddingService(api_key="test-key")
        
        result = embedding_service.embed_chunk(sample_chunks[0])
        
        assert isinstance(result, EmbeddedChunk)
        assert isinstance(result.embedding, np.ndarray)
        assert len(result.embedding) == 1536  # OpenAI ada-002 dimension
        assert result.chunk == sample_chunks[0]
    
    @patch('embeddings.OpenAI')
    def test_embed_batch_chunks_maintains_order(self, mock_openai_class, embedding_service, sample_chunks):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536)
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        # Recreate service to use mocked client
        embedding_service = EmbeddingService(api_key="test-key")
        
        results = embedding_service.embed_batch(sample_chunks)
        
        assert len(results) == len(sample_chunks)
        assert results[0].chunk.witness_name == "Charles Herbert Lightoller"
        assert results[1].chunk.witness_name == "Hugh Woolner"
    
    def test_embed_handles_empty_content(self, embedding_service):
        empty_chunk = WitnessChunk(
            content="",
            witness_name="Test Witness",
            metadata=Mock()
        )
        
        with pytest.raises(ValueError, match="Cannot embed empty content"):
            embedding_service.embed_chunk(empty_chunk)
    
    @patch('embeddings.OpenAI')
    def test_embed_handles_api_rate_limiting(self, mock_openai_class, embedding_service, sample_chunks):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_client.embeddings.create.side_effect = [
            Exception("Rate limit exceeded"),
            Mock(data=[Mock(embedding=[0.1] * 1536)])
        ]
        
        # Recreate service to use mocked client
        embedding_service = EmbeddingService(api_key="test-key")
        
        with patch('time.sleep') as mock_sleep:
            result = embedding_service.embed_chunk(sample_chunks[0])
            
            assert mock_sleep.called
            assert isinstance(result, EmbeddedChunk)
    
    @patch('embeddings.OpenAI')
    def test_embed_chunk_preserves_metadata(self, mock_openai_class, embedding_service, sample_chunks):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        
        # Recreate service to use mocked client
        embedding_service = EmbeddingService(api_key="test-key")
        
        result = embedding_service.embed_chunk(sample_chunks[0])
        
        assert result.chunk.metadata.document_name == "British Inquiry Day 5"
        assert result.chunk.metadata.page_number == 247
        assert result.chunk.witness_name == "Charles Herbert Lightoller"
    
    def test_cosine_similarity_calculation(self, embedding_service):
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([1, 0, 0])
        
        similarity_different = embedding_service.cosine_similarity(vec1, vec2)
        similarity_same = embedding_service.cosine_similarity(vec1, vec3)
        
        assert similarity_different == 0.0
        assert similarity_same == 1.0
    
    @patch('embeddings.OpenAI')
    def test_find_similar_chunks(self, mock_openai_class, embedding_service, sample_chunks):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[1.0] + [0.0] * 1535),
            Mock(embedding=[0.9] + [0.1] * 1535)
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        # Recreate service to use mocked client
        embedding_service = EmbeddingService(api_key="test-key")
        
        embedded_chunks = embedding_service.embed_batch(sample_chunks)
        query_embedding = np.array([1.0] + [0.0] * 1535)
        
        similar_chunks = embedding_service.find_similar_chunks(
            query_embedding, 
            embedded_chunks, 
            top_k=1,
            similarity_threshold=0.5
        )
        
        assert len(similar_chunks) == 1
        assert isinstance(similar_chunks[0], tuple)
        assert isinstance(similar_chunks[0][0], EmbeddedChunk)
        assert isinstance(similar_chunks[0][1], float)
    
    def test_batch_size_optimization(self, embedding_service):
        large_chunk_list = [Mock() for _ in range(1000)]
        
        batches = embedding_service._split_into_batches(large_chunk_list, batch_size=100)
        
        assert len(batches) == 10
        assert all(len(batch) <= 100 for batch in batches)
    
    @patch('embeddings.OpenAI')
    def test_embed_with_different_providers(self, mock_openai_class):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        
        openai_service = EmbeddingService(provider="openai", model="text-embedding-ada-002", api_key="test-key")
        # Note: cohere_service would need separate implementation
        cohere_service = EmbeddingService(provider="cohere", model="embed-english-v2.0", api_key="test-key")
        
        chunk = Mock()
        chunk.content = "test content"
        
        openai_result = openai_service.embed_chunk(chunk)
        
        assert len(openai_result.embedding) == 1536
    
    @patch('embeddings.OpenAI')
    def test_embedding_cache_functionality(self, mock_openai_class, embedding_service, sample_chunks):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        
        # Recreate service to use mocked client
        embedding_service = EmbeddingService(api_key="test-key")
        
        with patch.object(embedding_service, '_get_from_cache', return_value=None) as mock_get, \
             patch.object(embedding_service, '_store_in_cache') as mock_store:
            
            embedding_service.embed_chunk(sample_chunks[0])
            
            mock_get.assert_called_once()
            mock_store.assert_called_once()