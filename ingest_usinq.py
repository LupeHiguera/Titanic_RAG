#!/usr/bin/env python3
"""
Ingest the full USInq.pdf (2,000+ pages) using the improved chunking strategy (800/80)
and update the vector database for comprehensive testing.
"""

import sys
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the root directory to path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from Services.document_ingestion import DocumentIngestion
from Services.chunking import IntelligentChunker
from Services.embeddings import EmbeddingService
from Services.vector_storage import ChromaVectorStore

def main():
    print("🚀 Starting USInq.pdf Ingestion with Improved Chunking Strategy (800/80)")
    print("=" * 70)
    
    # Initialize services with improved chunking
    print("📋 Initializing services...")
    ingestion = DocumentIngestion()
    chunker = IntelligentChunker()  # Now uses 800/80 by default
    embeddings = EmbeddingService()
    vector_store = ChromaVectorStore()
    
    # Path to the large PDF
    usinq_path = root_dir / "Text" / "USInq.pdf"
    
    if not usinq_path.exists():
        print(f"❌ Error: {usinq_path} not found!")
        return
    
    print(f"📄 Processing: {usinq_path}")
    print(f"📏 Chunking Strategy: {chunker.chunk_size} chars, {chunker.overlap_size} overlap")
    
    # Step 1: Extract text from PDF
    print("\n🔍 Step 1: Extracting text from PDF...")
    start_time = time.time()
    
    try:
        result = ingestion.extract_text_from_pdf(usinq_path)
        extract_time = time.time() - start_time
        
        print(f"✅ Extraction complete!")
        print(f"   📊 Text length: {len(result['text']):,} characters")
        print(f"   ⏱️  Time taken: {extract_time:.2f} seconds")
        print(f"   📄 Document: {result['metadata'].document_name}")
        
    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        return
    
    # Step 2: Identify witnesses
    print("\n👤 Step 2: Identifying witnesses...")
    witnesses = ingestion.identify_witness_names(result["text"])
    print(f"✅ Found {len(witnesses)} witnesses: {witnesses[:10]}{'...' if len(witnesses) > 10 else ''}")
    
    # Step 3: Create witness contexts (sample for testing)
    print("\n📝 Step 3: Creating witness contexts...")
    # For the full document, we'll create contexts by splitting the text into manageable sections
    # This is a simplified approach - in production you'd want more sophisticated witness segmentation
    
    text_length = len(result["text"])
    section_size = 50000  # 50k characters per section for processing
    sections = []
    
    for i in range(0, text_length, section_size):
        section_text = result["text"][i:i + section_size]
        section_witness = witnesses[0] if witnesses else "Multiple Witnesses"
        
        section = {
            'witness': section_witness,
            'testimony': section_text,
            'page_number': (i // section_size) + 1,
            'document_name': result["metadata"].document_name
        }
        sections.append(section)
    
    print(f"✅ Created {len(sections)} sections for processing")
    
    # Step 4: Chunk the testimony with improved strategy
    print("\n✂️  Step 4: Chunking testimony with improved strategy...")
    start_time = time.time()
    
    all_chunks = []
    for i, section in enumerate(sections):  # Process ALL sections
        print(f"   Processing section {i+1}/{len(sections)}...")
        chunks = chunker.chunk_witness_contexts([section])
        all_chunks.extend(chunks)
        
        if i == 0:  # Show details for first section
            print(f"   📊 First section generated {len(chunks)} chunks")
            if chunks:
                print(f"   📏 Average chunk size: {sum(len(c.content) for c in chunks) / len(chunks):.0f} chars")
        
        # Progress indicator every 10 sections
        if (i + 1) % 10 == 0:
            print(f"   🔄 Progress: {i+1}/{len(sections)} sections processed")
    
    chunk_time = time.time() - start_time
    print(f"✅ Chunking complete!")
    print(f"   📊 Total chunks: {len(all_chunks)}")
    print(f"   📏 Average chunk size: {sum(len(c.content) for c in all_chunks) / len(all_chunks):.0f} chars")
    print(f"   ⏱️  Time taken: {chunk_time:.2f} seconds")
    
    # Step 5: Generate embeddings for ALL chunks
    print(f"\n🧠 Step 5: Generating embeddings for all {len(all_chunks)} chunks...")
    print("⚠️  This may take several minutes due to OpenAI API rate limits...")
    start_time = time.time()
    
    # Process all chunks with progress tracking
    sample_chunks = all_chunks
    
    try:
        embedded_chunks = embeddings.embed_batch(sample_chunks)
        embedding_time = time.time() - start_time
        
        print(f"✅ Embeddings complete!")
        print(f"   📊 Embedded {len(embedded_chunks)} chunks")
        print(f"   📏 Embedding dimension: {len(embedded_chunks[0].embedding)}")
        print(f"   ⏱️  Time taken: {embedding_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return
    
    # Step 6: Store in vector database
    print("\n💾 Step 6: Storing in vector database...")
    start_time = time.time()
    
    try:
        # Store the embedded chunks directly
        stored_ids = vector_store.store_chunks(embedded_chunks)
        storage_time = time.time() - start_time
        
        print(f"✅ Vector storage complete!")
        print(f"   📊 Stored {len(embedded_chunks)} document chunks")
        print(f"   🆔 Generated IDs: {len(stored_ids)}")
        print(f"   ⏱️  Time taken: {storage_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error storing vectors: {e}")
        return
    
    # Step 7: Test search functionality
    print("\n🔍 Step 7: Testing search functionality...")
    
    # Initialize search engine for testing
    from Services.semantic_search import SemanticSearchEngine, SearchQuery
    search_engine = SemanticSearchEngine(embeddings, vector_store)
    
    test_queries = [
        "What was Ismay's position",
        "ship speed revolutions", 
        "lifeboat loading procedure",
        "collision response actions",
        "ice warnings knowledge"
    ]
    
    for query in test_queries:
        print(f"\n   🔎 Query: '{query}'")
        try:
            search_query = SearchQuery(text=query, top_k=3, similarity_threshold=0.4)
            results = search_engine.search(search_query)
            print(f"   📊 Found {len(results)} results")
            if results:
                best_result = results[0]
                preview = best_result.chunk.chunk.content[:150] + "..."
                print(f"   📝 Best match (score: {best_result.similarity_score:.3f}): {preview}")
        except Exception as e:
            print(f"   ❌ Search error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 USInq.pdf Ingestion Complete!")
    print(f"📊 Total processing time: {(time.time() - start_time + extract_time + chunk_time + embedding_time + storage_time):.2f} seconds")
    print(f"📄 Document: {result['metadata'].document_name}")
    print(f"👤 Witnesses: {len(witnesses)}")
    print(f"✂️  Chunks generated: {len(all_chunks)} (with {chunker.chunk_size}/{chunker.overlap_size} strategy)")
    print(f"🧠 Embeddings: {len(embedded_chunks)} chunks embedded")
    print(f"💾 Vector DB: {len(stored_ids)} chunks stored successfully")
    print("\n🚀 Your RAG system is now ready with the full USInq.pdf data!")
    print("🌐 Test it at: http://localhost:8000")

if __name__ == "__main__":
    main()