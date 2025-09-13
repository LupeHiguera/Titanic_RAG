#!/usr/bin/env python3
"""
Pinecone Upload and Migration Script for Titanic RAG

This script handles:
1. Full document ingestion with text-embedding-3-large model
2. Upload to Pinecone vector database  
3. Migration from ChromaDB to Pinecone
4. Backup and restore operations

Usage:
  python pinecone_upload.py --full-ingest     # Process full USInq.pdf and upload to Pinecone
  python pinecone_upload.py --migrate         # Migrate existing ChromaDB data to Pinecone
  python pinecone_upload.py --test            # Test Pinecone connection and basic operations
  python pinecone_upload.py --stats           # Show current database statistics
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
from Services.vector_storage import ChromaVectorStore, PineconeVectorStore
from Services.semantic_search import SemanticSearchEngine


class PineconeManager:
    """Handles Pinecone operations for the Titanic RAG system."""
    
    def __init__(self):
        """Initialize services with new embedding model."""
        self.doc_ingestion = DocumentIngestion()
        self.chunker = IntelligentChunker()
        # Use text-embedding-3-large with 1024 dimensions (free tier compatible)
        self.embedding_service = EmbeddingService(model="text-embedding-3-large", dimensions=1024)
        
        # Initialize vector stores
        self.chroma_store = ChromaVectorStore()
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
        """Process full document and upload to Pinecone with progress tracking."""
        print("=== FULL DOCUMENT INGESTION & PINECONE UPLOAD ===")
        
        try:
            # Step 1: Extract text from PDF
            print(f"🔄 Step 1: Extracting text from {pdf_path}...")
            from pathlib import Path
            pdf_path_obj = Path(pdf_path)
            doc_result = self.doc_ingestion.extract_text_from_pdf(pdf_path_obj)
            extracted_text = doc_result["text"]
            metadata = doc_result["metadata"]
            
            print(f"✅ Extracted {len(extracted_text):,} characters")
            print(f"   Document: {metadata.document_name}")
            
            # Step 2: Chunk the document
            print("🔄 Step 2: Intelligent chunking...")
            chunks = self.chunker.chunk_document_by_witness(
                text=extracted_text,
                document_metadata=metadata
            )
            print(f"✅ Created {len(chunks)} chunks")
            
            # Show witness breakdown
            witnesses = {}
            for chunk in chunks:
                witness = chunk.witness_name
                witnesses[witness] = witnesses.get(witness, 0) + 1
            print(f"🎭 Found {len(witnesses)} witnesses:")
            for witness, count in sorted(witnesses.items()):
                print(f"   {witness}: {count} chunks")
            
            # Step 3: Generate embeddings with progress tracking
            print("🔄 Step 3: Generating embeddings with text-embedding-3-large...")
            embedded_chunks = []
            batch_size = 50  # Process in smaller batches for better progress tracking
            
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
            
            # Step 4: Upload to Pinecone
            print("🔄 Step 4: Uploading to Pinecone...")
            chunk_ids = self.pinecone_store.store_chunks(embedded_chunks)
            print(f"✅ Uploaded {len(chunk_ids)} chunks to Pinecone")
            
            # Step 5: Verify upload
            print("🔄 Step 5: Verifying upload...")
            stats = self.pinecone_store.get_collection_stats()
            print(f"✅ Verification complete:")
            print(f"   Total vectors in Pinecone: {stats.get('total_chunks', 0)}")
            print(f"   Index name: {stats.get('index_name', 'unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Full ingestion failed: {e}")
            return False
    
    def migrate_from_chroma(self):
        """Migrate existing ChromaDB data to Pinecone."""
        print("=== MIGRATING FROM CHROMADB TO PINECONE ===")
        
        try:
            # Check ChromaDB status
            chroma_stats = self.chroma_store.get_collection_stats()
            print(f"📊 ChromaDB status: {chroma_stats['total_chunks']} chunks")
            
            if chroma_stats['total_chunks'] == 0:
                print("⚠️  No data in ChromaDB to migrate")
                return False
            
            # Get all data from ChromaDB
            print("🔄 Retrieving data from ChromaDB...")
            all_data = self.chroma_store.collection.get(
                include=['documents', 'metadatas', 'embeddings']
            )
            
            print(f"✅ Retrieved {len(all_data['documents'])} chunks from ChromaDB")
            
            # Check if embeddings are compatible (should be ada-002 = 1536 dimensions)
            if all_data['embeddings'] and len(all_data['embeddings'][0]) == 1536:
                print("⚠️  ChromaDB contains ada-002 embeddings (1536d)")
                print("   Need to re-embed with text-embedding-3-large (3072d)")
                
                # Re-embed all chunks with new model
                print("🔄 Re-embedding chunks with text-embedding-3-large...")
                
                # Convert back to WitnessChunk objects and re-embed
                embedded_chunks = []
                for i, (doc, meta) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
                    # Create WitnessChunk from stored data
                    from Services.chunking import WitnessChunk, ChunkMetadata
                    
                    chunk_meta = ChunkMetadata(
                        document_name=meta['document_name'],
                        source_type=meta['source_type'],
                        page_number=meta['page_number'],
                        credibility_score=0.0,
                        chunk_index=meta['chunk_index'],
                        total_chunks_for_witness=meta['total_chunks_for_witness']
                    )
                    
                    chunk = WitnessChunk(
                        content=doc,
                        witness_name=meta['witness_name'],
                        metadata=chunk_meta
                    )
                    
                    # Re-embed with new model
                    embedded_chunk = self.embedding_service.embed_chunk(chunk)
                    embedded_chunks.append(embedded_chunk)
                    
                    if (i + 1) % 50 == 0:
                        print(f"   Re-embedded {i + 1}/{len(all_data['documents'])} chunks")
                
            else:
                print("⚠️  Unexpected embedding dimensions or missing embeddings")
                return False
            
            # Upload to Pinecone
            print("🔄 Uploading to Pinecone...")
            chunk_ids = self.pinecone_store.store_chunks(embedded_chunks)
            print(f"✅ Migration complete: {len(chunk_ids)} chunks uploaded to Pinecone")
            
            # Verify migration
            pinecone_stats = self.pinecone_store.get_collection_stats()
            print(f"📊 Pinecone now contains: {pinecone_stats.get('total_chunks', 0)} vectors")
            
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
    
    def show_database_stats(self):
        """Show current database statistics for both ChromaDB and Pinecone."""
        print("=== DATABASE STATISTICS ===")
        
        try:
            # ChromaDB stats
            chroma_stats = self.chroma_store.get_collection_stats()
            print("📊 ChromaDB (Local):")
            print(f"   Total chunks: {chroma_stats['total_chunks']}")
            print(f"   Witnesses: {len(chroma_stats['unique_witnesses'])}")
            print(f"   Collection: {chroma_stats['collection_name']}")
            
        except Exception as e:
            print(f"❌ ChromaDB stats failed: {e}")
        
        try:
            # Pinecone stats
            pinecone_stats = self.pinecone_store.get_collection_stats()
            print("📊 Pinecone (Cloud):")
            print(f"   Total vectors: {pinecone_stats.get('total_chunks', 0)}")
            print(f"   Dimensions: {pinecone_stats.get('dimension', 'unknown')}")
            print(f"   Index name: {pinecone_stats.get('index_name', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Pinecone stats failed: {e}")
    
    def test_search_functionality(self):
        """Test search functionality with both databases."""
        print("=== TESTING SEARCH FUNCTIONALITY ===")
        
        test_queries = [
            "How many people were in the lifeboats?",
            "What was Ismay's role on the ship?",
            "When did the ship sink?",
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            try:
                # Test with Pinecone
                search_engine = SemanticSearchEngine(vector_store=self.pinecone_store)
                results = search_engine.search(query, top_k=3)
                
                print("📊 Pinecone results:")
                for i, (chunk, score) in enumerate(results, 1):
                    print(f"   {i}. [{score:.3f}] {chunk.chunk.witness_name}: {chunk.chunk.content[:100]}...")
                    
            except Exception as e:
                print(f"❌ Pinecone search failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Pinecone Upload and Migration for Titanic RAG")
    parser.add_argument("--full-ingest", action="store_true", 
                       help="Process full USInq.pdf and upload to Pinecone")
    parser.add_argument("--migrate", action="store_true",
                       help="Migrate existing ChromaDB data to Pinecone")
    parser.add_argument("--test", action="store_true",
                       help="Test Pinecone connection and operations")
    parser.add_argument("--stats", action="store_true",
                       help="Show database statistics")
    parser.add_argument("--test-search", action="store_true",
                       help="Test search functionality")
    
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
    
    if args.migrate:
        success &= manager.migrate_from_chroma()
    
    if args.full_ingest:
        success &= manager.full_ingestion_and_upload()
    
    if args.test_search:
        manager.test_search_functionality()
    
    if not any([args.full_ingest, args.migrate, args.test, args.stats, args.test_search]):
        print("📋 No action specified. Use --help for available options.")
        print("💡 Quick start: python pinecone_upload.py --test")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())