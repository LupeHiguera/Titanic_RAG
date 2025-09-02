import pytest
import numpy as np
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add the root directory to path so we can import Services
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from Services.vector_storage import ChromaVectorStore, PineconeVectorStore
from Services.embeddings import EmbeddedChunk, EmbeddingService
from Services.chunking import WitnessChunk, ChunkMetadata, IntelligentChunker
from Services.document_ingestion import DocumentIngestion


class TestVectorStore:
    
    @pytest.fixture
    def temp_dir(self):
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def chroma_store(self, temp_dir):
        return ChromaVectorStore(
            collection_name="titanic_test",
            persist_directory=str(temp_dir)
        )
    
    @pytest.fixture
    def pinecone_store(self):
        return PineconeVectorStore(
            index_name="titanic-test",
            api_key="test-key",
            environment="test-env"
        )
    
    @pytest.fixture
    def real_ismay_embedded_chunks(self):
        """Create real embedded chunks from Ismay testimony in one page.pdf"""
        # Real Q&A pairs from the Ismay testimony
        metadata1 = ChunkMetadata(
            document_name="one page.pdf",
            source_type="us_inquiry", 
            page_number=1,
            credibility_score=0.0,  # Not using credibility scoring
            chunk_index=0,
            total_chunks_for_witness=3
        )
        
        chunk1 = WitnessChunk(
            content="Q: First state your full name, please? A: Joseph Bruce Ismay. Q: And your place of residence? A: Liverpool. Q: And your occupation? A: Ship owner.",
            witness_name="Joseph Bruce Ismay",
            metadata=metadata1
        )
        
        metadata2 = ChunkMetadata(
            document_name="one page.pdf",
            source_type="us_inquiry",
            page_number=1, 
            credibility_score=0.0,
            chunk_index=1,
            total_chunks_for_witness=3
        )
        
        chunk2 = WitnessChunk(
            content="Q: As such officer, were you officially designated to make the trial trip of the Titanic? A: No. Q: Were you a voluntary passenger? A: A voluntary passenger, yes.",
            witness_name="Joseph Bruce Ismay", 
            metadata=metadata2
        )
        
        metadata3 = ChunkMetadata(
            document_name="one page.pdf",
            source_type="us_inquiry",
            page_number=1,
            credibility_score=0.0,
            chunk_index=2,
            total_chunks_for_witness=3
        )
        
        chunk3 = WitnessChunk(
            content="The ship was built in Belfast. She was the latest thing in the art of shipbuilding; absolutely no money was spared in her construction. She was not built by contract. She was simply built on a commission.",
            witness_name="Joseph Bruce Ismay",
            metadata=metadata3
        )
        
        # Create realistic embeddings (different but similar for same witness)
        return [
            EmbeddedChunk(
                chunk=chunk1,
                embedding=np.random.rand(1536)  # OpenAI ada-002 dimension
            ),
            EmbeddedChunk(
                chunk=chunk2, 
                embedding=np.random.rand(1536)
            ),
            EmbeddedChunk(
                chunk=chunk3,
                embedding=np.random.rand(1536)
            )
        ]
        
    @pytest.fixture
    def sample_embedded_chunks(self):
        """Keep original mock data for basic tests"""
        metadata1 = ChunkMetadata(
            document_name="British Inquiry Day 5",
            source_type="british_inquiry",
            page_number=247,
            credibility_score=0.0,
            chunk_index=0,
            total_chunks_for_witness=1
        )

        
        chunk1 = WitnessChunk(
            content="Q: What was your position? A: I was Second Officer.",
            witness_name="Charles Herbert Lightoller",
            metadata=metadata1
        )
        
        metadata2 = ChunkMetadata(
            document_name="US Senate Inquiry Day 12",
            source_type="us_inquiry",
            page_number=891,
            credibility_score=0.0,
            chunk_index=0,
            total_chunks_for_witness=1
        )
        
        chunk2 = WitnessChunk(
            content="Q: Did you see men in lifeboats? A: Yes, when no women were nearby.",
            witness_name="Hugh Woolner",
            metadata=metadata2
        )
        
        return [
            EmbeddedChunk(
                chunk=chunk1,
                embedding=np.array([0.1, 0.2, 0.3] * 512)
            ),
            EmbeddedChunk(
                chunk=chunk2,
                embedding=np.array([0.4, 0.5, 0.6] * 512)
            )
        ]


class TestChromaVectorStore:
    
    def test_store_chunks_persists_data(self, chroma_store, sample_embedded_chunks):
        chunk_ids = chroma_store.store_chunks(sample_embedded_chunks)
        
        assert len(chunk_ids) == len(sample_embedded_chunks)
        assert all(isinstance(chunk_id, str) for chunk_id in chunk_ids)
        assert len(set(chunk_ids)) == len(chunk_ids)  # All unique IDs
    
    def test_query_returns_relevant_chunks(self, chroma_store, sample_embedded_chunks):
        chroma_store.store_chunks(sample_embedded_chunks)
        
        query_vector = np.array([0.1, 0.2, 0.3] * 512)
        results = chroma_store.query(query_vector, top_k=1)
        
        assert len(results) == 1
        assert isinstance(results[0], tuple)
        chunk, similarity = results[0]
        assert isinstance(chunk, EmbeddedChunk)
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_query_with_metadata_filters(self, chroma_store, sample_embedded_chunks):
        chroma_store.store_chunks(sample_embedded_chunks)
        
        query_vector = np.array([0.1, 0.2, 0.3] * 512)
        filters = {"source_type": "british_inquiry"}
        
        results = chroma_store.query(query_vector, top_k=5, filters=filters)
        
        british_results = [r for r in results if r[0].chunk.metadata.source_type == "british_inquiry"]
        assert len(british_results) > 0
    
    
    def test_delete_chunks_by_id(self, chroma_store, sample_embedded_chunks):
        chunk_ids = chroma_store.store_chunks(sample_embedded_chunks)
        
        success = chroma_store.delete_chunks([chunk_ids[0]])
        assert success
        
        query_vector = np.array([0.1, 0.2, 0.3] * 512)
        results = chroma_store.query(query_vector, top_k=5)
        
        assert len(results) == len(sample_embedded_chunks) - 1
    
    def test_update_chunk_metadata(self, chroma_store, sample_embedded_chunks):
        chunk_ids = chroma_store.store_chunks(sample_embedded_chunks)
        
        new_metadata = {"updated": True, "version": 2}
        success = chroma_store.update_chunk_metadata(chunk_ids[0], new_metadata)
        
        assert success
    
    def test_get_collection_stats(self, chroma_store, sample_embedded_chunks):
        chroma_store.store_chunks(sample_embedded_chunks)
        
        stats = chroma_store.get_collection_stats()
        
        assert "total_chunks" in stats
        assert "unique_witnesses" in stats
        assert "document_types" in stats
        assert stats["total_chunks"] >= len(sample_embedded_chunks)
    
    def test_backup_and_restore_collection(self, chroma_store, sample_embedded_chunks, temp_dir):
        chroma_store.store_chunks(sample_embedded_chunks)
        
        backup_path = temp_dir / "backup"
        success = chroma_store.backup_collection(str(backup_path))
        assert success
        assert backup_path.exists()
        
        new_store = ChromaVectorStore(
            collection_name="titanic_restored",
            persist_directory=str(temp_dir)
        )
        restore_success = new_store.restore_collection(str(backup_path))
        assert restore_success


class TestPineconeVectorStore:
    
    @patch('pinecone.init')
    @patch('pinecone.Index')
    def test_store_chunks_uploads_to_pinecone(self, mock_index_class, mock_init, pinecone_store, sample_embedded_chunks):
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.upsert.return_value = {"upserted_count": 2}
        
        chunk_ids = pinecone_store.store_chunks(sample_embedded_chunks)
        
        assert len(chunk_ids) == len(sample_embedded_chunks)
        mock_index.upsert.assert_called_once()
    
    @patch('pinecone.init')
    @patch('pinecone.Index')
    def test_query_pinecone_with_filters(self, mock_index_class, mock_init, pinecone_store, sample_embedded_chunks):
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "test-id-1",
                    "score": 0.95,
                    "metadata": {
                        "witness_name": "Charles Herbert Lightoller",
                        "source_type": "british_inquiry"
                    }
                }
            ]
        }
        
        query_vector = np.array([0.1, 0.2, 0.3] * 512)
        results = pinecone_store.query(query_vector, top_k=1)
        
        assert len(results) == 1
        mock_index.query.assert_called_once()
    
    @patch('pinecone.init')
    @patch('pinecone.Index')  
    def test_delete_chunks_from_pinecone(self, mock_index_class, mock_init, pinecone_store):
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.delete.return_value = {"deleted_count": 1}
        
        success = pinecone_store.delete_chunks(["test-id-1"])
        
        assert success
        mock_index.delete.assert_called_once_with(ids=["test-id-1"])


class TestVectorStoreIntegration:
    
    def test_store_and_retrieve_workflow(self, chroma_store, sample_embedded_chunks):
        stored_ids = chroma_store.store_chunks(sample_embedded_chunks)
        
        query_embedding = sample_embedded_chunks[0].embedding
        results = chroma_store.query(query_embedding, top_k=1)
        
        assert len(results) == 1
        retrieved_chunk, similarity = results[0]
        assert similarity > 0.9  # Should be very similar to itself
        assert retrieved_chunk.chunk.witness_name == sample_embedded_chunks[0].chunk.witness_name
    
    def test_similarity_search_ranking(self, chroma_store, sample_embedded_chunks):
        chroma_store.store_chunks(sample_embedded_chunks)
        
        query_vector = sample_embedded_chunks[0].embedding
        results = chroma_store.query(query_vector, top_k=2)
        
        assert len(results) == 2
        similarities = [result[1] for result in results]
        assert similarities[0] >= similarities[1]  # Results should be ranked by similarity