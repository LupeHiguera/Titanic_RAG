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
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Add the project root to path so `python Services/pinecone_upload.py` works
# from anywhere (Python puts the script's own dir on sys.path, not the cwd).
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from Services.document_ingestion import DocumentIngestion
from Services.chunking import IntelligentChunker, BritishBoundarySplitter, SenateBoundarySplitter, page_tag
from Services.embeddings import EmbeddingService
from Services.page_map import build_page_map
from Services.vector_storage import PineconeVectorStore
from Services.witness_index import WitnessIndex
from Services.british_witness_index import BritishWitnessIndex


def _find_session_start(text, witness, cursor):
    """Position where a witness's session heading appears in page text: the
    first whole-word ALL-CAPS occurrence of their surname at or after cursor.
    Session headings — `TESTIMONY OF JAMES WIDGERY` (US), `FREDERICK SHEATH,
    Sworn.` (British) — print names in caps; prose mentions don't."""
    surname = witness.name.replace('.', ' ').replace(',', ' ').split()[-1].upper()
    match = re.search(r'\b' + re.escape(surname) + r'\b', text[cursor:])
    return cursor + match.start() if match else None


def build_witness_contexts(page_texts, witness_index, printed_page_map,
                           document_name, doc_ingestion=None):
    """Walk PDF pages, attribute text to witnesses, accumulate testimony
    sessions ready for IntelligentChunker.

    printed_page_map: pdf_page → printed/transcript page (see
    Services.page_map). Attribution and citations both run on printed pages —
    witness indexes are keyed on them, and they're what a reader of the
    original inquiry would cite. ⟦p:N⟧ tags are embedded wherever the printed
    page changes so the chunker can give every chunk its real page instead of
    the session's first page.

    Sessions are driven by the index's TOC entries ("starters"). Each
    starter's session begins at their caps-surname heading within the page
    text, so two witnesses sharing a printed page are split at the heading
    instead of one swallowing the other. If a heading can't be found, the
    session falls back to starting at the top of the following page.
    """
    doc_ingestion = doc_ingestion or DocumentIngestion()

    starters_by_page = {}
    for w in witness_index.witnesses:
        starters_by_page.setdefault(w.page, []).append(w)

    contexts = []
    dropped = 0
    current_witness = None
    current_start_page = None
    parts = []              # text parts incl. page tags
    last_cite = None

    def flush():
        nonlocal dropped
        if not (current_witness and parts):
            return
        combined = "\n".join(parts)
        cleaned = doc_ingestion.clean_extracted_text(combined)
        if len(cleaned) > 100:
            contexts.append({
                'witness': current_witness.name,
                'testimony': cleaned,
                'page_number': current_start_page,
                'document_name': document_name,
                'role': current_witness.role,
                'ship': current_witness.ship_affiliation,
                'witness_type': current_witness.witness_type,
            })
        else:
            dropped += 1
            print(f"   ⚠ Dropped short session for {current_witness.name} "
                  f"@ p.{current_start_page} ({len(cleaned)} chars)")

    def start_session(witness, printed):
        nonlocal current_witness, current_start_page, parts, last_cite
        flush()
        current_witness = witness
        current_start_page = printed
        parts = []
        last_cite = None

    def append_text(printed, text):
        nonlocal last_cite
        if current_witness is None or not text:
            return
        if printed != last_cite:
            parts.append(page_tag(printed))
            last_cite = printed
        parts.append(text)

    pending = []  # starters whose session hasn't begun yet, TOC order
    last_printed = witness_index.FIRST_WITNESS_PAGE - 1

    for pdf_page in sorted(page_texts.keys()):
        printed = printed_page_map.get(pdf_page)
        if printed is None:
            continue
        if not (witness_index.FIRST_WITNESS_PAGE <= printed <= witness_index.LAST_WITNESS_PAGE):
            continue

        if printed != last_printed:
            for p in range(last_printed + 1, printed + 1):
                pending.extend(starters_by_page.get(p, []))
            # Heading never found on the starter's own page(s): fall back
            # to page-level attribution from the top of this page.
            while pending and pending[0].page < printed:
                start_session(pending.pop(0), printed)
            last_printed = printed

        # Bootstrap: the first witness owns their start page from the top
        # (the very first US session has no TESTIMONY OF heading).
        if current_witness is None and pending and pending[0].page <= printed:
            start_session(pending.pop(0), printed)

        text = page_texts[pdf_page]
        cursor = 0
        while pending and pending[0].page <= printed:
            pos = _find_session_start(text, pending[0], cursor)
            if pos is None:
                break
            append_text(printed, text[cursor:pos])
            start_session(pending.pop(0), printed)
            cursor = pos
        append_text(printed, text[cursor:])

    flush()
    if dropped:
        print(f"   ⚠ {dropped} session(s) dropped as too short")
    return contexts


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
    
    def _build_witness_contexts(self, page_texts, witness_index, printed_page_map, document_name):
        return build_witness_contexts(
            page_texts, witness_index, printed_page_map, document_name,
            doc_ingestion=self.doc_ingestion,
        )

    def _embed_and_upload(self, chunks):
        """Embed chunks in batches, push to Pinecone, print progress. Returns True on success."""
        print("🔄 Generating embeddings with text-embedding-3-large...")
        embedded_chunks = []
        batch_size = 100  # matches EmbeddingService's per-API-call batch size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embedded_chunks.extend(self.embedding_service.embed_batch(batch))
            progress = min(i + batch_size, len(chunks))
            print(f"   Progress: {progress}/{len(chunks)} chunks embedded ({progress/len(chunks)*100:.1f}%)")

        print(f"✅ Generated {len(embedded_chunks)} embeddings")
        print(f"📏 Embedding dimension: {len(embedded_chunks[0].embedding)} (text-embedding-3-large @ 1024d)")

        print("🔄 Uploading to Pinecone...")
        chunk_ids = self.pinecone_store.store_chunks(embedded_chunks)
        print(f"✅ Uploaded {len(chunk_ids)} chunks to Pinecone")

        stats = self.pinecone_store.get_collection_stats()
        print(f"✅ Verification: {stats.get('total_chunks', 0)} total vectors in '{stats.get('index_name', 'unknown')}'")
        return True

    def _run_ingestion(self, pdf_path, witness_index, document_name, chunker):
        """Shared ingestion path: extract → map pages → attribute → chunk → embed → upload."""
        print(f"🔄 Step 1: Extracting text from {pdf_path}...")
        page_texts = self.doc_ingestion.extract_pages_from_pdf(Path(pdf_path))
        print(f"   Found {len(page_texts)} pages")

        print("🔄 Step 2: Building PDF→printed page map...")
        printed_page_map = build_page_map(pdf_path)
        print(f"   Mapped {len(printed_page_map)} PDF pages "
              f"(printed {min(printed_page_map.values())}–{max(printed_page_map.values())})")

        print("🔄 Step 3: Attributing pages to witnesses via index...")
        contexts = self._build_witness_contexts(page_texts, witness_index, printed_page_map, document_name)
        print(f"✅ Built {len(contexts)} witness testimony sections")

        print("🔄 Step 4: Intelligent chunking...")
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
                witness_index=self.witness_index,
                document_name="US Senate Inquiry - Titanic Disaster",
                chunker=self.chunker,
            )
        except Exception as e:
            print(f"❌ US ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def ingest_british(self, pdf_path: str = "Text/BritishInquiry.pdf"):
        """Ingest the British Wreck Commissioner's Inquiry.

        British names are kept AS-IS (no remap to US-canonical): per-inquiry
        names are what let the contradiction detector pair a witness against
        their own other-inquiry testimony — the killer demo. The display
        layer uses BRITISH_TO_US_CANONICAL for the "same person" hint.
        """
        print("=== BRITISH WRECK COMMISSIONERS' INQUIRY INGESTION ===")
        try:
            return self._run_ingestion(
                pdf_path=pdf_path,
                witness_index=BritishWitnessIndex(),
                document_name="British Wreck Commissioners' Inquiry - Titanic Disaster",
                chunker=IntelligentChunker(splitter=BritishBoundarySplitter()),
            )
        except Exception as e:
            print(f"❌ British ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def clear_source(self, source: str):
        """Delete all vectors for one inquiry ('us' or 'british')."""
        prefix = {"us": "us:", "british": "br:"}[source]
        print(f"🔄 Deleting vectors with prefix '{prefix}'...")
        deleted = self.pinecone_store.delete_by_prefix(prefix)
        print(f"✅ Deleted {deleted} vectors")
        if deleted == 0:
            print("   (Pre-deterministic-ID vectors have UUID ids — use --clear-all to remove those.)")
        return True

    def clear_all(self):
        """Delete every vector in the index (incl. legacy UUID-id vectors)."""
        print("🔄 Deleting ALL vectors in the index...")
        self.pinecone_store.delete_all()
        print("✅ Index cleared")
        return True

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
    parser.add_argument("--clear-source", choices=["us", "british"],
                       help="Delete all vectors for one inquiry (by ID prefix)")
    parser.add_argument("--clear-all", action="store_true",
                       help="Delete ALL vectors in the index (incl. legacy UUID-id vectors)")

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

    if args.clear_all:
        success &= manager.clear_all()
    elif args.clear_source:
        success &= manager.clear_source(args.clear_source)

    if args.full_ingest:
        success &= manager.full_ingestion_and_upload()

    if args.ingest_british:
        success &= manager.ingest_british()

    if not any([args.full_ingest, args.ingest_british, args.test, args.stats,
                args.clear_all, args.clear_source]):
        print("📋 No action specified. Use --help for available options.")
        print("💡 Quick start: python pinecone_upload.py --test")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())