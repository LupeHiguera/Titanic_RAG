#!/usr/bin/env python3
"""Complete reprocessing script to extract all witnesses from US Inquiry."""

import re
from pathlib import Path
import sys
from pathlib import Path

# Add the root directory to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from Services.document_ingestion import DocumentIngestion
from Services.chunking import IntelligentChunker
from Services.embeddings import EmbeddingService, EmbeddedChunk
from Services.vector_storage import ChromaVectorStore

def extract_all_witnesses_from_document(text):
    """Extract all witness testimonies with proper name identification."""
    print("Extracting witnesses from document...")
    
    # Split by clear witness section markers
    sections = []
    
    # Pattern 1: "TESTIMONY OF [NAME]" - most reliable
    testimony_sections = re.split(r'\n\s*TESTIMONY OF ([A-Z][A-Za-z\s]+[A-Za-z])\s*\.?\s*\n', text, flags=re.IGNORECASE)
    
    if len(testimony_sections) > 1:
        for i in range(1, len(testimony_sections), 2):
            if i + 1 < len(testimony_sections):
                witness_name = testimony_sections[i].strip()
                testimony_text = testimony_sections[i + 1].strip()
                
                # Clean witness name
                witness_name = re.sub(r'\s+', ' ', witness_name).title()
                
                # Skip if testimony too short or name too generic
                if len(testimony_text) > 500 and 5 < len(witness_name) < 50:
                    if not re.search(r'\d', witness_name) and witness_name not in ['Committee', 'Subcommittee']:
                        sections.append({
                            'witness': witness_name,
                            'testimony': testimony_text[:10000],  # Limit size
                            'page_number': 1,
                            'document_name': 'US Senate Inquiry'
                        })
    
    # Pattern 2: Look for "[NAME], sworn and examined" patterns
    sworn_pattern = r'([A-Z][A-Za-z\s]{5,35}),\s+sworn and examined\.?\s*\n(.*?)(?=\n[A-Z][A-Za-z\s]+,\s+sworn and examined|\n\s*TESTIMONY OF|\Z)'
    sworn_matches = re.finditer(sworn_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for match in sworn_matches:
        witness_name = match.group(1).strip()
        testimony_text = match.group(2).strip()
        
        # Clean and validate
        witness_name = re.sub(r'\s+', ' ', witness_name).title()
        
        if len(testimony_text) > 500 and 5 < len(witness_name) < 50:
            if not re.search(r'\d', witness_name):
                # Check if we already have this witness
                existing = next((s for s in sections if s['witness'].lower() == witness_name.lower()), None)
                if not existing:
                    sections.append({
                        'witness': witness_name,
                        'testimony': testimony_text[:10000],
                        'page_number': 1,
                        'document_name': 'US Senate Inquiry'
                    })
    
    # Pattern 3: Manual extraction for known witnesses that might be missed
    known_witnesses = [
        'Bruce Ismay', 'Charles Herbert Lightoller', 'Harold Thomas Cottam',
        'Alfred Crawford', 'Joseph Groves Boxhall', 'Herbert John Pitman',
        'Frederick Fleet', 'Harold Godfrey Lowe', 'Arthur Henry Rostron'
    ]
    
    for known in known_witnesses:
        # Look for sections containing this witness
        pattern = rf'\b{re.escape(known)}\b.*?(\n.*?){{500,5000}}'
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        
        testimony_parts = []
        for match in matches:
            testimony_parts.append(match.group(0))
        
        if testimony_parts:
            combined = '\n'.join(testimony_parts)[:8000]
            # Check if we already have this witness
            existing = next((s for s in sections if known.lower() in s['witness'].lower()), None)
            if not existing and len(combined) > 800:
                sections.append({
                    'witness': known,
                    'testimony': combined,
                    'page_number': 1,
                    'document_name': 'US Senate Inquiry'
                })
    
    print(f"Extracted {len(sections)} witness sections")
    for i, section in enumerate(sections):
        print(f"  {i+1}. {section['witness']} - {len(section['testimony'])} chars")
    
    return sections

def main():
    """Main reprocessing function."""
    try:
        print("=== COMPLETE WITNESS REPROCESSING ===")
        
        # Initialize services
        print("Initializing services...")
        doc_processor = DocumentIngestion()
        chunker = IntelligentChunker()
        embedding_service = EmbeddingService()
        vector_store = ChromaVectorStore()
        
        # Process document
        pdf_path = Path('Text/USInq.pdf')
        if not pdf_path.exists():
            print("ERROR: USInq.pdf not found")
            return
        
        print(f"Processing {pdf_path}...")
        result = doc_processor.extract_text_from_pdf(pdf_path)
        text = result['text']
        print(f"Extracted {len(text)} characters from {result['metadata'].total_pages} pages")
        
        # Extract all witness sections
        witness_sections = extract_all_witnesses_from_document(text)
        
        if not witness_sections:
            print("ERROR: No witness sections found")
            return
        
        # Process through chunking pipeline
        print("\nChunking witness testimonies...")
        all_chunks = chunker.chunk_witness_contexts(witness_sections)
        print(f"Created {len(all_chunks)} total chunks")
        
        # Verify witness diversity in chunks
        witness_counts = {}
        for chunk in all_chunks:
            name = chunk.witness_name
            witness_counts[name] = witness_counts.get(name, 0) + 1
        
        print(f"Witness distribution in chunks:")
        for name, count in sorted(witness_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {count} chunks")
        
        # Create embeddings and store in batches
        print(f"\nCreating embeddings and storing {len(all_chunks)} chunks...")
        batch_size = 25
        total_stored = 0
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(all_chunks)-1)//batch_size + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            # Create embedded chunks
            embedded_chunks = []
            for chunk in batch:
                embedding = embedding_service.embed_text(chunk.content)
                embedded_chunk = EmbeddedChunk(chunk=chunk, embedding=embedding)
                embedded_chunks.append(embedded_chunk)
            
            # Store batch
            chunk_ids = vector_store.store_chunks(embedded_chunks)
            total_stored += len(chunk_ids)
            
            if batch_num % 5 == 0:
                print(f"  ✓ Progress: {total_stored}/{len(all_chunks)} chunks stored")
        
        print(f"\n✅ SUCCESS: Stored {total_stored} chunks total")
        
        # Final verification
        print("\n=== FINAL VERIFICATION ===")
        stats = vector_store.get_collection_stats()
        print(f"Total chunks in database: {stats.get('total_chunks', 0)}")
        print(f"Unique witnesses: {len(stats.get('unique_witnesses', []))}")
        print(f"Witness list: {sorted(stats.get('unique_witnesses', []))}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()