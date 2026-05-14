import numpy as np
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import pinecone as pc_client
except ImportError:
    pc_client = None

from Services.embeddings import EmbeddedChunk
from Services.chunking import WitnessChunk, ChunkMetadata


class VectorStore(ABC):
    """Abstract base class for vector storage implementations."""
    
    @abstractmethod
    def store_chunks(self, embedded_chunks: List[EmbeddedChunk]) -> List[str]:
        """Store embedded chunks and return their IDs."""
        pass
    
    @abstractmethod
    def query(self, query_vector: np.ndarray, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[EmbeddedChunk, float]]:
        """Query for similar chunks and return (chunk, similarity_score) tuples."""
        pass
    
    @abstractmethod
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete chunks by their IDs."""
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        pass


class PineconeVectorStore(VectorStore):
    """Pinecone implementation for production cloud storage."""
    
    def __init__(self, index_name: Optional[str] = None, api_key: Optional[str] = None, environment: Optional[str] = None):
        if pc_client is None:
            raise ImportError("pinecone package is required but not installed")
        
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "titanic-rag")
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        # NB: default is an AWS region because we create AWS ServerlessSpec below.
        # Setting a GCP-style region here (e.g. us-east1-gcp) will 400 at index create.
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required but not provided")
        
        # Initialize Pinecone client
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            # Create Pinecone instance
            pc = Pinecone(api_key=self.api_key)
            
            # List existing indexes
            existing_indexes = [index.name for index in pc.list_indexes()]
            
            # Create index if it doesn't exist
            if self.index_name not in existing_indexes:
                print(f"Creating new Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=1024,  # OpenAI text-embedding-3-large with reduced dimensions
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.environment  # Use environment from config
                    )
                )
                print(f"Index {self.index_name} created successfully")
            else:
                print(f"Using existing Pinecone index: {self.index_name}")
            
            self.index = pc.Index(self.index_name)
        except Exception as e:
            # Print the actual error for debugging
            print(f"Pinecone initialization failed: {e}")
            raise  # preserve original traceback
    
    def store_chunks(self, embedded_chunks: List[EmbeddedChunk]) -> List[str]:
        """Store embedded chunks in Pinecone and return their IDs."""
        if not embedded_chunks:
            return []
        
        vectors = []
        chunk_ids = []
        
        for embedded_chunk in embedded_chunks:
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            # Prepare metadata
            metadata = {
                "witness_name": embedded_chunk.chunk.witness_name,
                "document_name": embedded_chunk.chunk.metadata.document_name,
                "source_type": embedded_chunk.chunk.metadata.source_type,
                "page_number": embedded_chunk.chunk.metadata.page_number,
                "chunk_index": embedded_chunk.chunk.metadata.chunk_index,
                "total_chunks_for_witness": embedded_chunk.chunk.metadata.total_chunks_for_witness,
                "content": embedded_chunk.chunk.content  # Store content in metadata
            }
            
            vectors.append({
                "id": chunk_id,
                "values": embedded_chunk.embedding.tolist(),
                "metadata": metadata
            })
        
        # Upsert vectors in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        return chunk_ids
    
    def query(self, query_vector: np.ndarray, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[EmbeddedChunk, float]]:
        """Query for similar chunks using vector similarity."""
        # Prepare filter for Pinecone
        pinecone_filter = {}
        if filters:
            for key, value in filters.items():
                if key in ["witness_name", "source_type"]:
                    pinecone_filter[key] = value
                elif key == "page_range" and isinstance(value, tuple) and len(value) == 2:
                    pinecone_filter["page_number"] = {"$gte": value[0], "$lte": value[1]}
        
        # Query Pinecone
        response = self.index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            filter=pinecone_filter if pinecone_filter else None,
            include_metadata=True
        )
        
        # Convert results back to EmbeddedChunk objects
        embedded_chunks = []
        for match in response.matches:
            metadata_dict = match.metadata
            
            # Reconstruct metadata
            metadata = ChunkMetadata(
                document_name=metadata_dict["document_name"],
                source_type=metadata_dict["source_type"],
                page_number=metadata_dict["page_number"],
                credibility_score=0.0,  # Not using credibility scoring
                chunk_index=metadata_dict["chunk_index"],
                total_chunks_for_witness=metadata_dict["total_chunks_for_witness"]
            )
            
            # Reconstruct WitnessChunk
            chunk = WitnessChunk(
                content=metadata_dict["content"],
                witness_name=metadata_dict["witness_name"],
                metadata=metadata
            )
            
            # Create EmbeddedChunk (without embedding for efficiency)
            embedded_chunk = EmbeddedChunk(chunk=chunk, embedding=np.array([]))
            
            embedded_chunks.append((embedded_chunk, match.score))
        
        return embedded_chunks
    
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete chunks by their IDs. Raises on Pinecone errors."""
        self.index.delete(ids=chunk_ids)
        return True

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get index statistics. Raises on Pinecone errors so /health surfaces real outages."""
        stats = self.index.describe_index_stats()
        return {
            "total_chunks": stats.get('total_vector_count', 0),
            "index_name": self.index_name,
            "dimension": stats.get('dimension', 1024),
        }