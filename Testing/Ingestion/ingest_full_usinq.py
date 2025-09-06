#!/usr/bin/env python3
"""
Complete ingestion of USInq.pdf with progress tracking and batch processing.
Estimated time: ~20-25 minutes for full 4000+ chunks.
"""

import sys
from pathlib import Path
import time
from dotenv import load_dotenv
import math

# Load environment variables from .env file
load_dotenv()

# Add the root directory to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from Services.document_ingestion import DocumentIngestion
from Services.chunking import IntelligentChunker
from Services.embeddings import EmbeddingService
from Services.vector_storage import ChromaVectorStore

def main():
    print("🚀 FULL USInq.pdf INGESTION - Complete Dataset")
    print("=" * 70)
    print("⏱️  Estimated time: 20-25 minutes")
    print("📊 Expected output: ~4,000 chunks from 1,173 pages")
    print("=" * 70)
    
    # Initialize services
    print("📋 Initializing services...")
    ingestion = DocumentIngestion()
    chunker = IntelligentChunker()
    embeddings = EmbeddingService()
    vector_store = ChromaVectorStore()
    
    # Path to the large PDF
    usinq_path = root_dir / "Text" / "USInq.pdf"
    
    if not usinq_path.exists():
        print(f"❌ Error: {usinq_path} not found!")
        return
    
    print(f"📄 Processing: {usinq_path}")
    
    # Step 1: Extract text from PDF
    print("\n🔍 Step 1: Extracting text from PDF...")
    start_time = time.time()
    
    try:
        result = ingestion.extract_text_from_pdf(usinq_path)
        extract_time = time.time() - start_time
        
        print(f"✅ Extraction complete!")
        print(f"   📊 Text length: {len(result['text']):,} characters")
        print(f"   📄 Pages: {result['metadata'].total_pages}")
        print(f"   ⏱️  Time: {extract_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error extracting PDF: {e}")
        return
    
    # Step 2: Identify witnesses
    print("\n👤 Step 2: Identifying witnesses...")
    witnesses = ingestion.identify_witness_names(result["text"])
    print(f"✅ Found {len(witnesses)} witnesses")
    
    # Step 3: Create sections
    print("\n📝 Step 3: Creating witness contexts...")
    text_length = len(result["text"])
    section_size = 50000  # 50k characters per section
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
    
    # Step 4: Chunk ALL sections
    print(f"\n✂️  Step 4: Chunking all {len(sections)} sections...")
    chunk_start = time.time()
    
    all_chunks = []
    for i, section in enumerate(sections):
        if (i + 1) % 10 == 0:
            print(f"   🔄 Chunking progress: {i+1}/{len(sections)} sections")
        
        chunks = chunker.chunk_witness_contexts([section])
        all_chunks.extend(chunks)
    
    chunk_time = time.time() - chunk_start
    print(f"✅ Chunking complete!")
    print(f"   📊 Total chunks: {len(all_chunks):,}")
    print(f"   📏 Average chunk size: {sum(len(c.content) for c in all_chunks) / len(all_chunks):.0f} chars")
    print(f"   ⏱️  Time taken: {chunk_time:.2f} seconds")
    
    # Step 5: Embed in batches with progress
    print(f"\n🧠 Step 5: Embedding {len(all_chunks):,} chunks...")
    print("⚠️  This will take 20-25 minutes due to API rate limits...")
    
    batch_size = 50  # Process 50 chunks at a time
    total_batches = math.ceil(len(all_chunks) / batch_size)
    embedded_chunks = []
    
    embed_start = time.time()
    
    for batch_num in range(total_batches):
        batch_start_idx = batch_num * batch_size
        batch_end_idx = min((batch_num + 1) * batch_size, len(all_chunks))
        batch = all_chunks[batch_start_idx:batch_end_idx]
        
        print(f"   🧠 Batch {batch_num + 1}/{total_batches}: Embedding chunks {batch_start_idx + 1}-{batch_end_idx}...")
        
        try:
            batch_embedded = embeddings.embed_batch(batch)
            embedded_chunks.extend(batch_embedded)
            
            # Progress update
            elapsed = time.time() - embed_start
            chunks_done = len(embedded_chunks)
            chunks_remaining = len(all_chunks) - chunks_done
            
            if chunks_done > 0:
                rate = chunks_done / elapsed  # chunks per second
                eta_seconds = chunks_remaining / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                
                print(f"       📈 Progress: {chunks_done:,}/{len(all_chunks):,} chunks ({chunks_done/len(all_chunks)*100:.1f}%)")
                print(f"       ⏱️  ETA: {eta_minutes:.1f} minutes remaining")
            
        except Exception as e:
            print(f"   ❌ Batch {batch_num + 1} failed: {e}")
            print("   🔄 Continuing with next batch...")
            continue
    
    embed_time = time.time() - embed_start
    print(f"\n✅ Embedding complete!")
    print(f"   📊 Successfully embedded: {len(embedded_chunks):,} chunks")
    print(f"   ⏱️  Total time: {embed_time/60:.1f} minutes")
    
    # Step 6: Store all chunks
    print(f"\n💾 Step 6: Storing {len(embedded_chunks):,} chunks in vector database...")
    storage_start = time.time()
    
    try:
        stored_ids = vector_store.store_chunks(embedded_chunks)
        storage_time = time.time() - storage_start
        
        print(f"✅ Vector storage complete!")
        print(f"   📊 Stored: {len(stored_ids):,} chunks")
        print(f"   ⏱️  Time: {storage_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error storing vectors: {e}")
        return
    
    # Step 7: Final verification
    print("\n🔍 Step 7: Verifying vector database...")
    stats = vector_store.get_collection_stats()
    print(f"   📊 Total chunks in DB: {stats.get('total_chunks', 0):,}")
    print(f"   👤 Unique witnesses: {len(stats.get('unique_witnesses', []))}")
    print(f"   📄 Document types: {stats.get('document_types', [])}")
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("🎉 FULL USInq.pdf INGESTION COMPLETE!")
    print("=" * 70)
    print(f"📊 Total processing time: {total_time/60:.1f} minutes")
    print(f"📄 Document: {result['metadata'].document_name}")
    print(f"📝 Pages processed: {result['metadata'].total_pages:,}")
    print(f"👤 Witnesses identified: {len(witnesses)}")
    print(f"✂️  Chunks generated: {len(all_chunks):,}")
    print(f"🧠 Chunks embedded: {len(embedded_chunks):,}")
    print(f"💾 Chunks stored: {len(stored_ids):,}")
    print(f"🗃️  Vector DB size: {stats.get('total_chunks', 0):,} chunks")
    print("\n🚀 Your complete Titanic RAG system is now ready!")
    print("🌐 Test it at: http://localhost:8000")
    print("🔍 Expected search results: Rich historical testimony data")

if __name__ == "__main__":
    main()