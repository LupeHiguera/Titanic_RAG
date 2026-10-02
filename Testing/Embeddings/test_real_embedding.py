"""
Test script to verify embeddings work with real OpenAI API key.
Opt in with python -m pytest --run-live (requires OPENAI_API_KEY; incurs API cost).
"""

import os
import pytest
import sys
from pathlib import Path
from dotenv import load_dotenv

pytestmark = pytest.mark.live

# Add the root directory to path so we can import Services
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from Services.embeddings import EmbeddingService
from Services.chunking import WitnessChunk, ChunkMetadata

def test_real_embedding():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for live tests")

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
    assert len(embedded_chunk.embedding) == service.dimensions
    assert (embedded_chunk2.embedding == embedded_chunk.embedding).all()
    assert similarity >= 0.0



if __name__ == "__main__":
    raise SystemExit("Run with python -m pytest --run-live Testing/Embeddings/test_real_embedding.py")
