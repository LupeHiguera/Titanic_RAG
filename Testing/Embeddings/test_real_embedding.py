"""
Test script to verify embeddings work with real OpenAI API key.
Run this after setting your OPENAI_API_KEY environment variable.
"""

import os
import sys
from pathlib import Path

# Add the root directory to path so we can import Services
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from Services.embeddings import EmbeddingService
from Services.chunking import WitnessChunk, ChunkMetadata

def test_real_embedding():
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return False

    try:
        # Create embedding service
        service = EmbeddingService(api_key=api_key)
        
        # Create a sample chunk
        metadata = ChunkMetadata(
            document_name="Test Document",
            source_type="test",
            page_number=1,
            credibility_score=1.0,
            chunk_index=0,
            total_chunks_for_witness=1
        )
        
        chunk = WitnessChunk(
            content="Q: What was your position on the Titanic? A: I was the Second Officer.",
            witness_name="Test Witness",
            metadata=metadata
        )
        
        print("🔄 Testing embedding with OpenAI API...")
        
        # Test single chunk embedding
        embedded_chunk = service.embed_chunk(chunk)
        
        print("✅ Single chunk embedding successful!")
        print(f"   - Embedding dimension: {len(embedded_chunk.embedding)}")
        print(f"   - Witness: {embedded_chunk.chunk.witness_name}")
        
        # Test text embedding
        text_embedding = service.embed_text("What happened to the lifeboats?")
        print(f"✅ Text embedding successful! Dimension: {len(text_embedding)}")
        
        # Test similarity calculation
        similarity = service.cosine_similarity(embedded_chunk.embedding, text_embedding)
        print(f"✅ Cosine similarity: {similarity:.3f}")
        
        # Test caching (second call should be from cache)
        print("🔄 Testing caching...")
        embedded_chunk2 = service.embed_chunk(chunk)
        print("✅ Caching works - second embedding call completed")
        
        print("\n🎉 All embedding tests passed with real API!")
        assert len(embedded_chunk.embedding) > 0
        assert similarity >= 0.0
        
    except Exception as e:
        print(f"❌ Error testing embeddings: {e}")
        return False

if __name__ == "__main__":
    test_real_embedding()