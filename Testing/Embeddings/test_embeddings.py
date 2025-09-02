import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add the root directory to path so we can import Services
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from Services.embeddings import EmbeddingService, EmbeddedChunk
from Services.chunking import WitnessChunk, ChunkMetadata, IntelligentChunker
from Services.document_ingestion import DocumentIngestion


class TestEmbeddingService:
    
    @pytest.fixture
    def embedding_service(self):
        return EmbeddingService(
            provider="openai",
            model="text-embedding-ada-002",
            api_key="test-key"
        )
    
    @pytest.fixture
    def real_chunks(self):
        """Create real chunks from one page.pdf"""
        # Extract real data
        ingestion = DocumentIngestion()
        chunker = IntelligentChunker(chunk_size=300, overlap_size=30)
        
        pdf_path = root_dir / "Text" / "one page.pdf"
        if not pdf_path.exists():
            pytest.skip("PDF file not found")
        
        result = ingestion.extract_text_from_pdf(pdf_path)
        witnesses = ingestion.identify_witness_names(result["text"])
        
        witness_context = {
            'witness': witnesses[0] if witnesses else 'Ismay',
            'testimony': result["text"],
            'page_number': 1,
            'document_name': result["metadata"].document_name
        }
        
        chunks = chunker.chunk_witness_contexts([witness_context])
        return chunks[:2]  # Return first 2 chunks for testing
    
    @pytest.fixture
    def mock_small_chunks(self):
        """Smaller mock chunks for tests that need specific content"""
        metadata1 = ChunkMetadata(
            document_name="US Senate Inquiry - Titanic Disaster",
            source_type="us_inquiry",
            page_number=1,
            credibility_score=0.6,  # Basic score
            chunk_index=0,
            total_chunks_for_witness=2
        )
        
        metadata2 = ChunkMetadata(
            document_name="US Senate Inquiry - Titanic Disaster",
            source_type="us_inquiry",
            page_number=1,
            credibility_score=0.6,  # Basic score
            chunk_index=1,
            total_chunks_for_witness=2
        )
        
        return [
            WitnessChunk(
                content="Senator SMITH. Mr. Ismay, what was your position on the Titanic? Mr. ISMAY. I was Managing Director of White Star Line.",
                witness_name="Ismay",
                metadata=metadata1
            ),
            WitnessChunk(
                content="Senator SMITH. Were you a voluntary passenger? Mr. ISMAY. A voluntary passenger, yes.",
                witness_name="Ismay", 
                metadata=metadata2
            )
        ]
    
    @patch('Services.embeddings.OpenAI')
    def test_embed_single_chunk_returns_vector(self, mock_openai_class, embedding_service, real_chunks):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3] * 512)]
        mock_client.embeddings.create.return_value = mock_response
        
        # Recreate service to use mocked client
        embedding_service = EmbeddingService(api_key="test-key")
        
        result = embedding_service.embed_chunk(real_chunks[0])
        
        assert isinstance(result, EmbeddedChunk)
        assert isinstance(result.embedding, np.ndarray)
        assert len(result.embedding) == 1536  # OpenAI ada-002 dimension
        assert result.chunk == real_chunks[0]
    
    @patch('Services.embeddings.OpenAI')
    def test_embed_batch_chunks_maintains_order(self, mock_openai_class, embedding_service, real_chunks):
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
        
        results = embedding_service.embed_batch(real_chunks)
        
        assert len(results) == len(real_chunks)
        assert results[0].chunk.witness_name == "Ismay"
        assert results[1].chunk.witness_name == "Ismay"
    
    def test_embed_handles_empty_content(self, embedding_service):
        empty_chunk = WitnessChunk(
            content="",
            witness_name="Test Witness",
            metadata=Mock()
        )
        
        with pytest.raises(ValueError, match="Cannot embed empty content"):
            embedding_service.embed_chunk(empty_chunk)
    
    @patch('Services.embeddings.OpenAI')
    def test_embed_handles_api_rate_limiting(self, mock_openai_class, embedding_service, mock_small_chunks):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_client.embeddings.create.side_effect = [
            Exception("Rate limit exceeded"),
            Mock(data=[Mock(embedding=[0.1] * 1536)])
        ]
        
        # Recreate service to use mocked client
        embedding_service = EmbeddingService(api_key="test-key")
        
        with patch('time.sleep') as mock_sleep:
            result = embedding_service.embed_chunk(mock_small_chunks[0])
            
            assert mock_sleep.called
            assert isinstance(result, EmbeddedChunk)
    
    @patch('Services.embeddings.OpenAI')
    def test_embed_chunk_preserves_metadata(self, mock_openai_class, embedding_service, real_chunks):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        
        # Recreate service to use mocked client
        embedding_service = EmbeddingService(api_key="test-key")
        
        result = embedding_service.embed_chunk(real_chunks[0])
        
        assert result.chunk.metadata.document_name == "US Senate Inquiry - Titanic Disaster"
        assert result.chunk.metadata.page_number == 1
        assert result.chunk.witness_name == "Ismay"
    
    def test_cosine_similarity_calculation(self, embedding_service):
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([1, 0, 0])
        
        similarity_different = embedding_service.cosine_similarity(vec1, vec2)
        similarity_same = embedding_service.cosine_similarity(vec1, vec3)
        
        assert similarity_different == 0.0
        assert similarity_same == 1.0
    
    @patch('Services.embeddings.OpenAI')
    def test_find_similar_chunks(self, mock_openai_class, embedding_service, mock_small_chunks):
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
        
        embedded_chunks = embedding_service.embed_batch(mock_small_chunks)
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
    
    @patch('Services.embeddings.OpenAI')
    def test_embed_with_different_providers(self, mock_openai_class):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        
        openai_service = EmbeddingService(provider="openai", model="text-embedding-ada-002", api_key="test-key")
        
        chunk = Mock()
        chunk.content = "test content"
        
        openai_result = openai_service.embed_chunk(chunk)
        
        assert len(openai_result.embedding) == 1536
    
    @patch('Services.embeddings.OpenAI')
    def test_embedding_cache_functionality(self, mock_openai_class, embedding_service, mock_small_chunks):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        
        # Recreate service to use mocked client
        embedding_service = EmbeddingService(api_key="test-key")
        
        with patch.object(embedding_service, '_get_from_cache', return_value=None) as mock_get, \
             patch.object(embedding_service, '_store_in_cache') as mock_store:
            
            embedding_service.embed_chunk(mock_small_chunks[0])
            
            mock_get.assert_called_once()
            mock_store.assert_called_once()
    
    def test_real_pdf_embedding_integration(self, embedding_service):
        """Test that we can process real PDF content through the full pipeline"""
        # Extract real data
        ingestion = DocumentIngestion()
        chunker = IntelligentChunker(chunk_size=200, overlap_size=20)
        
        pdf_path = root_dir / "Text" / "one page.pdf"
        if not pdf_path.exists():
            pytest.skip("PDF file not found")
        
        # Full pipeline test
        result = ingestion.extract_text_from_pdf(pdf_path)
        witnesses = ingestion.identify_witness_names(result["text"])
        
        witness_context = {
            'witness': witnesses[0] if witnesses else 'Ismay',
            'testimony': result["text"][:500],  # Use first 500 chars
            'page_number': 1,
            'document_name': result["metadata"].document_name
        }
        
        chunks = chunker.chunk_witness_contexts([witness_context])
        
        # Verify chunks are created properly for embedding
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, WitnessChunk)
            assert len(chunk.content) > 0
            assert chunk.witness_name == "Ismay"
            assert "ismay" in chunk.content.lower() or "senator" in chunk.content.lower()
            
            # Test that chunks have the right structure for embedding
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'witness_name')
            assert hasattr(chunk, 'metadata')
            assert hasattr(chunk.metadata, 'document_name')
            assert hasattr(chunk.metadata, 'source_type')
            
        print(f"\n✅ Successfully created {len(chunks)} chunks from real PDF for embedding")
        print(f"   Sample chunk: {chunks[0].content[:100]}...")