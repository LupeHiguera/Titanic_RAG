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
from Services.chunking import IntelligentChunker
from Services.embeddings import EmbeddingService
from Services.vector_storage import PineconeVectorStore
from Services.witness_index import WitnessIndex


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
    
    def full_ingestion_and_upload(self, pdf_path: str = "Text/USInq.pdf"):
        """Process full document and upload to Pinecone with progress tracking.

        Uses the witness index for page-based witness attribution instead of
        regex-based extraction.
        """
        print("=== FULL DOCUMENT INGESTION & PINECONE UPLOAD ===")

        try:
            # Step 1: Extract per-page text from PDF
            print(f"🔄 Step 1: Extracting text from {pdf_path}...")
            pdf_path_obj = Path(pdf_path)
            page_texts = self.doc_ingestion.extract_pages_from_pdf(pdf_path_obj)
            total_pages = len(page_texts)
            print(f"   Found {total_pages} pages")

            # Step 2: Attribute pages to witnesses using the index
            print("🔄 Step 2: Attributing pages to witnesses via index...")
            witness_contexts = []
            current_witness = None
            current_pages_text = []
            current_start_page = None

            for page_num in sorted(page_texts.keys()):
                witness = self.witness_index.get_witness_by_page_range(page_num)
                if witness is None:
                    continue

                if current_witness is None or witness.name != current_witness.name or witness.page == page_num:
                    # If witness changed (or this is a new testimony section for same witness),
                    # flush the accumulated text
                    if witness.page == page_num and current_witness is not None:
                        # New testimony section — flush previous
                        if current_pages_text:
                            combined = "\n".join(current_pages_text)
                            cleaned = self.doc_ingestion._clean_extracted_text(combined)
                            if len(cleaned) > 100:
                                witness_contexts.append({
                                    'witness': current_witness.name,
                                    'testimony': cleaned,
                                    'page_number': current_start_page,
                                    'document_name': 'US Senate Inquiry - Titanic Disaster',
                                })
                        current_witness = witness
                        current_pages_text = [page_texts[page_num]]
                        current_start_page = page_num
                    elif current_witness is None or witness.name != current_witness.name:
                        # Different witness — flush previous
                        if current_witness and current_pages_text:
                            combined = "\n".join(current_pages_text)
                            cleaned = self.doc_ingestion._clean_extracted_text(combined)
                            if len(cleaned) > 100:
                                witness_contexts.append({
                                    'witness': current_witness.name,
                                    'testimony': cleaned,
                                    'page_number': current_start_page,
                                    'document_name': 'US Senate Inquiry - Titanic Disaster',
                                })
                        current_witness = witness
                        current_pages_text = [page_texts[page_num]]
                        current_start_page = page_num
                    else:
                        current_pages_text.append(page_texts[page_num])
                else:
                    # Same witness, accumulate
                    current_pages_text.append(page_texts[page_num])

            # Flush last witness
            if current_witness and current_pages_text:
                combined = "\n".join(current_pages_text)
                cleaned = self.doc_ingestion._clean_extracted_text(combined)
                if len(cleaned) > 100:
                    witness_contexts.append({
                        'witness': current_witness.name,
                        'testimony': cleaned,
                        'page_number': current_start_page,
                        'document_name': 'US Senate Inquiry - Titanic Disaster',
                    })

            print(f"✅ Built {len(witness_contexts)} witness testimony sections")

            # Step 3: Chunk the witness contexts
            print("🔄 Step 3: Intelligent chunking...")
            chunks = self.chunker.chunk_witness_contexts(witness_contexts)
            print(f"✅ Created {len(chunks)} chunks")

            # Show witness breakdown
            witnesses = {}
            for chunk in chunks:
                witness = chunk.witness_name
                witnesses[witness] = witnesses.get(witness, 0) + 1
            print(f"🎭 Found {len(witnesses)} witnesses:")
            for witness, count in sorted(witnesses.items()):
                print(f"   {witness}: {count} chunks")

            # Step 4: Generate embeddings with progress tracking
            print("🔄 Step 4: Generating embeddings with text-embedding-3-large...")
            embedded_chunks = []
            batch_size = 50

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_embedded = self.embedding_service.embed_batch(batch)
                embedded_chunks.extend(batch_embedded)

                progress = min(i + batch_size, len(chunks))
                print(f"   Progress: {progress}/{len(chunks)} chunks embedded ({progress/len(chunks)*100:.1f}%)")

                # Small delay to avoid rate limits
                time.sleep(0.1)

            print(f"✅ Generated {len(embedded_chunks)} embeddings")
            print(f"📏 Embedding dimension: {len(embedded_chunks[0].embedding)} (text-embedding-3-large @ 1024d)")

            # Step 5: Upload to Pinecone
            print("🔄 Step 5: Uploading to Pinecone...")
            chunk_ids = self.pinecone_store.store_chunks(embedded_chunks)
            print(f"✅ Uploaded {len(chunk_ids)} chunks to Pinecone")

            # Step 6: Verify upload
            print("🔄 Step 6: Verifying upload...")
            stats = self.pinecone_store.get_collection_stats()
            print(f"✅ Verification complete:")
            print(f"   Total vectors in Pinecone: {stats.get('total_chunks', 0)}")
            print(f"   Index name: {stats.get('index_name', 'unknown')}")

            return True

        except Exception as e:
            print(f"❌ Full ingestion failed: {e}")
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

    if not any([args.full_ingest, args.test, args.stats]):
        print("📋 No action specified. Use --help for available options.")
        print("💡 Quick start: python pinecone_upload.py --test")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())