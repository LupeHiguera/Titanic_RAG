#!/usr/bin/env python3
"""
Pinecone Upload Script for Titanic RAG

This script handles:
1. Full document ingestion with text-embedding-3-large model
2. Upload to Pinecone vector database

Usage:
  python pinecone_upload.py --full-ingest     # Process full USInq.pdf and upload to Pinecone
  python pinecone_upload.py --test            # Test Pinecone connection and basic operations
  python pinecone_upload.py --stats           # Show current Pinecone statistics
"""

import argparse
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Add the root directory to path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from Services.document_ingestion import DocumentIngestion
from Services.chunking import IntelligentChunker, BritishBoundarySplitter, SenateBoundarySplitter
from Services.embeddings import EmbeddingService
from Services.vector_storage import PineconeVectorStore
from Services.witness_index import WitnessIndex
from Services.british_witness_index import (
    BritishWitnessIndex,
    build_pdf_to_transcript_map,
)


class PineconeManager:
    """Handles Pinecone operations for the Titanic RAG system."""
    
    def __init__(self):
        """Initialize services with new embedding model."""
        self.doc_ingestion = DocumentIngestion()
        self.chunker = IntelligentChunker()
        self.witness_index = WitnessIndex()
        # Use text-embedding-3-large with 1024 dimensions (free tier compatible)
        self.embedding_service = EmbeddingService(model="text-embedding-3-large", dimensions=1024)

        self.pinecone_store = PineconeVectorStore()

        print("🚀 Pinecone Manager initialized with text-embedding-3-large (1024d) model")
    
    def test_pinecone_connection(self):
        """Test Pinecone connection and basic operations."""
        print("=== TESTING PINECONE CONNECTION ===")
        
        try:
            # Test connection by getting stats
            stats = self.pinecone_store.get_collection_stats()
            print(f"✅ Connected to Pinecone index: {self.pinecone_store.index_name}")
            print(f"📊 Current vectors in index: {stats.get('total_chunks', 0)}")
            print(f"📏 Vector dimensions: {stats.get('dimension', 'unknown')}")
            
            # Test embedding with new model
            test_embedding = self.embedding_service.embed_text("This is a test query")
            print(f"✅ Embedding service working - dimension: {len(test_embedding)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Pinecone connection failed: {e}")
            return False
    
    def _build_witness_contexts(self, page_texts, attribute_fn, document_name, name_remap=None):
        """Walk PDF pages, attribute each to a witness, accumulate testimony sections.

        attribute_fn: callable(pdf_page: int) -> Witness | None
        name_remap: optional callable(name: str) -> str applied to each witness
                    name before it's stored — used to canonicalize British names
                    against US ones for cross-inquiry contradiction matching.
        Returns: list of context dicts ready for IntelligentChunker.
        """
        contexts = []
        current_witness = None
        current_pages_text = []
        current_start_page = None

        def remap(name):
            return name_remap(name) if name_remap else name

        def flush():
            if not (current_witness and current_pages_text):
                return
            combined = "\n".join(current_pages_text)
            cleaned = self.doc_ingestion._clean_extracted_text(combined)
            if len(cleaned) > 100:
                contexts.append({
                    'witness': remap(current_witness.name),
                    'testimony': cleaned,
                    'page_number': current_start_page,
                    'document_name': document_name,
                })

        for page_num in sorted(page_texts.keys()):
            witness = attribute_fn(page_num)
            if witness is None:
                continue

            is_new_testimony_section = witness.page == page_num
            is_different_witness = current_witness is None or witness.name != current_witness.name

            if is_new_testimony_section and current_witness is not None:
                flush()
                current_witness = witness
                current_pages_text = [page_texts[page_num]]
                current_start_page = page_num
            elif is_different_witness:
                flush()
                current_witness = witness
                current_pages_text = [page_texts[page_num]]
                current_start_page = page_num
            else:
                current_pages_text.append(page_texts[page_num])

        flush()
        return contexts

    def _embed_and_upload(self, chunks):
        """Embed chunks in batches, push to Pinecone, print progress. Returns True on success."""
        print("🔄 Generating embeddings with text-embedding-3-large...")
        embedded_chunks = []
        batch_size = 50

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embedded = self.embedding_service.embed_batch(batch)
            embedded_chunks.extend(batch_embedded)

            progress = min(i + batch_size, len(chunks))
            print(f"   Progress: {progress}/{len(chunks)} chunks embedded ({progress/len(chunks)*100:.1f}%)")
            time.sleep(0.1)

        print(f"✅ Generated {len(embedded_chunks)} embeddings")
        print(f"📏 Embedding dimension: {len(embedded_chunks[0].embedding)} (text-embedding-3-large @ 1024d)")

        print("🔄 Uploading to Pinecone...")
        chunk_ids = self.pinecone_store.store_chunks(embedded_chunks)
        print(f"✅ Uploaded {len(chunk_ids)} chunks to Pinecone")

        stats = self.pinecone_store.get_collection_stats()
        print(f"✅ Verification: {stats.get('total_chunks', 0)} total vectors in '{stats.get('index_name', 'unknown')}'")
        return True

    def _run_ingestion(self, pdf_path, attribute_fn, document_name, chunker, name_remap=None):
        """Shared ingestion path: extract → attribute → chunk → embed → upload."""
        print(f"🔄 Step 1: Extracting text from {pdf_path}...")
        page_texts = self.doc_ingestion.extract_pages_from_pdf(Path(pdf_path))
        print(f"   Found {len(page_texts)} pages")

        print("🔄 Step 2: Attributing pages to witnesses via index...")
        contexts = self._build_witness_contexts(page_texts, attribute_fn, document_name, name_remap=name_remap)
        print(f"✅ Built {len(contexts)} witness testimony sections")

        print("🔄 Step 3: Intelligent chunking...")
        chunks = chunker.chunk_witness_contexts(contexts)
        print(f"✅ Created {len(chunks)} chunks")

        witnesses = {}
        for chunk in chunks:
            witnesses[chunk.witness_name] = witnesses.get(chunk.witness_name, 0) + 1
        print(f"🎭 Found {len(witnesses)} witnesses across {len(chunks)} chunks")
        for witness, count in sorted(witnesses.items(), key=lambda x: -x[1])[:10]:
            print(f"   {witness}: {count} chunks")
        if len(witnesses) > 10:
            print(f"   ... +{len(witnesses) - 10} more")

        return self._embed_and_upload(chunks)

    def full_ingestion_and_upload(self, pdf_path: str = "Text/USInq.pdf"):
        """Ingest the US Senate Inquiry — backwards-compatible default."""
        print("=== US SENATE INQUIRY INGESTION ===")
        try:
            return self._run_ingestion(
                pdf_path=pdf_path,
                attribute_fn=self.witness_index.get_witness_by_page_range,
                document_name="US Senate Inquiry - Titanic Disaster",
                chunker=self.chunker,
            )
        except Exception as e:
            print(f"❌ US ingestion failed: {e}")
            return False

    def ingest_british(self, pdf_path: str = "Text/BritishInquiry.pdf"):
        """Ingest the British Wreck Commissioner's Inquiry."""
        print("=== BRITISH WRECK COMMISSIONERS' INQUIRY INGESTION ===")
        try:
            print("🔄 Step 0: Building PDF→transcript page map...")
            pdf_to_transcript = build_pdf_to_transcript_map(pdf_path)
            print(f"   Mapped {len(pdf_to_transcript)} PDF pages")

            british_index = BritishWitnessIndex(pdf_to_transcript=pdf_to_transcript)
            british_chunker = IntelligentChunker(splitter=BritishBoundarySplitter())

            # Keep British names AS-IS (no remap to US-canonical). The
            # contradiction detector groups by witness_name and excludes
            # same-witness pairs — if we aliased "Charles Lightoller" to
            # "Charles Herbert Lightoller", his US and British testimony
            # would collapse into one group and never get compared.
            # Distinct names let the detector surface them as a cross-inquiry
            # pair, which is exactly the killer demo.
            return self._run_ingestion(
                pdf_path=pdf_path,
                attribute_fn=british_index.get_witness_by_pdf_page,
                document_name="British Wreck Commissioners' Inquiry - Titanic Disaster",
                chunker=british_chunker,
            )
        except Exception as e:
            print(f"❌ British ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def show_database_stats(self):
        """Show current Pinecone index statistics."""
        print("=== DATABASE STATISTICS ===")
        try:
            stats = self.pinecone_store.get_collection_stats()
            print("📊 Pinecone (Cloud):")
            print(f"   Total vectors: {stats.get('total_chunks', 0)}")
            print(f"   Dimensions:    {stats.get('dimension', 'unknown')}")
            print(f"   Index name:    {stats.get('index_name', 'unknown')}")
        except Exception as e:
            print(f"❌ Pinecone stats failed: {e}")
    
def main():
    parser = argparse.ArgumentParser(description="Pinecone Upload and Migration for Titanic RAG")
    parser.add_argument("--full-ingest", action="store_true",
                       help="Process full USInq.pdf and upload to Pinecone")
    parser.add_argument("--ingest-british", action="store_true",
                       help="Process Text/BritishInquiry.pdf and upload to Pinecone")
    parser.add_argument("--test", action="store_true",
                       help="Test Pinecone connection and operations")
    parser.add_argument("--stats", action="store_true",
                       help="Show database statistics")
    
    args = parser.parse_args()
    
    # Initialize manager
    try:
        manager = PineconeManager()
    except Exception as e:
        print(f"❌ Failed to initialize Pinecone Manager: {e}")
        print("💡 Make sure your .env file has valid PINECONE_API_KEY and PINECONE_INDEX_NAME")
        return 1
    
    success = True
    
    if args.test:
        success &= manager.test_pinecone_connection()
    
    if args.stats:
        manager.show_database_stats()
    
    if args.full_ingest:
        success &= manager.full_ingestion_and_upload()

    if args.ingest_british:
        success &= manager.ingest_british()

    if not any([args.full_ingest, args.ingest_british, args.test, args.stats]):
        print("📋 No action specified. Use --help for available options.")
        print("💡 Quick start: python pinecone_upload.py --test")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())