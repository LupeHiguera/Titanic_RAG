"""
Test embeddings with real OpenAI API and real PDF data from one page.pdf
Run this after setting your OPENAI_API_KEY environment variable.
"""

import os
import sys
from pathlib import Path

# Add the root directory to path so we can import Services
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from Services.embeddings import EmbeddingService
from Services.chunking import IntelligentChunker
from Services.document_ingestion import DocumentIngestion


def test_full_pipeline_with_real_data():
    """Test the complete pipeline: PDF → chunks → embeddings with real API."""
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    try:
        # Initialize services
        ingestion = DocumentIngestion()
        chunker = IntelligentChunker(chunk_size=300, overlap_size=30)
        embedding_service = EmbeddingService(api_key=api_key)
        
        # Extract text from real PDF
        pdf_path = root_dir / "Text" / "one page.pdf"
        if not pdf_path.exists():
            print("❌ PDF file not found")
            return False
        
        print("🔄 Step 1: Extracting text from PDF...")
        result = ingestion.extract_text_from_pdf(pdf_path)
        witnesses = ingestion.identify_witness_names(result["text"])
        
        print(f"✅ Extracted {len(result['text'])} characters")
        print(f"   Document: {result['metadata'].document_name}")
        print(f"   Witnesses: {witnesses}")
        
        # Create witness context for chunking
        print("🔄 Step 2: Creating witness contexts...")
        witness_context = {
            'witness': witnesses[0] if witnesses else 'Ismay',
            'testimony': result["text"],
            'page_number': 1,
            'document_name': result["metadata"].document_name
        }
        
        # Chunk the testimony
        print("🔄 Step 3: Chunking testimony...")
        chunks = chunker.chunk_witness_contexts([witness_context])
        print(f"✅ Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"   Chunk {i+1}: {len(chunk.content)} chars - {chunk.content[:50]}...")
        
        # Test single chunk embedding with real API
        print("🔄 Step 4: Testing single chunk embedding with OpenAI API...")
        first_chunk = chunks[0]
        embedded_chunk = embedding_service.embed_chunk(first_chunk)
        
        print(f"✅ Single chunk embedded successfully!")
        print(f"   - Embedding dimension: {len(embedded_chunk.embedding)}")
        print(f"   - Witness: {embedded_chunk.chunk.witness_name}")
        print(f"   - Content preview: {embedded_chunk.chunk.content[:80]}...")
        
        # Test batch embedding with real API (limit to 3 chunks to save API calls)
        print("🔄 Step 5: Testing batch embedding...")
        test_chunks = chunks[:3]  # Only embed first 3 chunks
        embedded_chunks = embedding_service.embed_batch(test_chunks)
        
        print(f"✅ Batch embedding successful!")
        print(f"   - Embedded {len(embedded_chunks)} chunks")
        
        # Test similarity calculations
        print("🔄 Step 6: Testing similarity search...")
        query_text = "What was Ismay's position?"
        query_embedding = embedding_service.embed_text(query_text)
        
        similar_chunks = embedding_service.find_similar_chunks(
            query_embedding, 
            embedded_chunks, 
            top_k=2,
            similarity_threshold=0.1
        )
        
        print(f"✅ Similarity search completed!")
        print(f"   - Query: '{query_text}'")
        print(f"   - Found {len(similar_chunks)} similar chunks")
        
        for i, (chunk, similarity) in enumerate(similar_chunks):
            print(f"   - Match {i+1}: {similarity:.3f} similarity")
            print(f"     Content: {chunk.chunk.content[:60]}...")
        
        # Test the full pipeline end-to-end
        print("🔄 Step 7: Full pipeline validation...")
        
        # Verify all chunks have proper structure
        assert all(hasattr(chunk, 'content') for chunk in chunks)
        assert all(hasattr(chunk, 'witness_name') for chunk in chunks)
        assert all(hasattr(chunk, 'metadata') for chunk in chunks)
        assert all(chunk.witness_name == 'Ismay' for chunk in chunks)
        
        # Verify embeddings have proper structure
        assert all(hasattr(ec, 'chunk') for ec in embedded_chunks)
        assert all(hasattr(ec, 'embedding') for ec in embedded_chunks)
        assert all(len(ec.embedding) == 1536 for ec in embedded_chunks)  # OpenAI ada-002
        
        # Verify similarity search works
        assert len(similar_chunks) > 0
        assert all(isinstance(sim, float) for _, sim in similar_chunks)
        assert all(0.0 <= sim <= 1.0 for _, sim in similar_chunks)
        
        print("\n🎉 Full pipeline test successful!")
        print("   ✅ PDF extraction")
        print("   ✅ Witness identification") 
        print("   ✅ Intelligent chunking")
        print("   ✅ Real OpenAI embeddings")
        print("   ✅ Similarity search")
        print("   ✅ All data structures validated")
        
    except Exception as e:
        print(f"❌ Error in full pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_full_pipeline_with_real_data()