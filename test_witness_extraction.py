#!/usr/bin/env python3
"""Test witness name extraction from actual data."""

from Services.vector_storage import ChromaVectorStore
from Services.document_ingestion import DocumentIngestion
from pathlib import Path
import re

def test_current_database():
    """Test what witness names are currently in the database."""
    print("=== CURRENT DATABASE TEST ===")
    
    try:
        vector_store = ChromaVectorStore()
        
        # Get larger sample
        sample_data = vector_store.collection.get(
            limit=100,
            include=['documents', 'metadatas']
        )
        
        # Analyze witness names
        witness_counts = {}
        content_witnesses = set()
        
        for doc, metadata in zip(sample_data['documents'], sample_data['metadatas']):
            # Count stored witness names
            stored_name = metadata.get('witness_name', 'UNKNOWN')
            witness_counts[stored_name] = witness_counts.get(stored_name, 0) + 1
            
            # Look for other names in content
            potential_names = re.findall(r'(LIGHTOLLER|COTTAM|CRAWFORD|FLEET|BOXHALL|PITMAN|LOWE)', doc)
            content_witnesses.update(potential_names)
        
        print(f"Stored witness name counts: {witness_counts}")
        print(f"Other witness names found in content: {sorted(content_witnesses)}")
        print(f"Total chunks analyzed: {len(sample_data['documents'])}")
        
        # Show sample content with context
        print("\n--- Sample chunks with potential other witnesses ---")
        for i, (doc, metadata) in enumerate(zip(sample_data['documents'][:5], sample_data['metadatas'][:5])):
            if any(name in doc for name in ['LIGHTOLLER', 'COTTAM', 'CRAWFORD']):
                print(f"\nChunk {i+1}:")
                print(f"  Stored witness: {metadata.get('witness_name')}")
                print(f"  Content snippet: {doc[:150]}...")
                break
        
    except Exception as e:
        print(f"Database test error: {e}")

def test_raw_document():
    """Test witness extraction from raw document."""
    print("\n=== RAW DOCUMENT TEST ===")
    
    try:
        doc_processor = DocumentIngestion()
        pdf_path = Path('Text/USInq.pdf')
        
        if pdf_path.exists():
            result = doc_processor.extract_text_from_pdf(pdf_path)
            text = result['text']
            print(f"Extracted {len(text)} characters from document")
            
            # Look for witness examination patterns in raw text
            witness_patterns = [
                r'([A-Z][A-Za-z\s]{5,30}), sworn and examined',
                r'([A-Z][A-Za-z\s]{5,30}), being duly sworn',
                r'TESTIMONY OF ([A-Z][A-Za-z\s]{5,30})',
                r'([A-Z]{3,})\s+sworn',
            ]
            
            found_witnesses = set()
            for pattern in witness_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    clean_name = re.sub(r'\s+', ' ', match.strip())
                    if 3 < len(clean_name) < 40:
                        found_witnesses.add(clean_name)
            
            print(f"Raw document witness patterns found: {sorted(found_witnesses)[:10]}")
            
            # Look for specific known witnesses
            known_witnesses = ['LIGHTOLLER', 'COTTAM', 'CRAWFORD', 'FLEET', 'BOXHALL', 'PITMAN']
            found_known = []
            for witness in known_witnesses:
                if witness in text:
                    # Find context around the witness name
                    idx = text.find(witness)
                    if idx != -1:
                        context = text[max(0, idx-50):idx+100]
                        found_known.append((witness, context.replace('\n', ' ')))
            
            print(f"\nKnown witnesses found in raw text:")
            for name, context in found_known[:3]:
                print(f"  {name}: ...{context}...")
                
        else:
            print("USInq.pdf not found")
            
    except Exception as e:
        print(f"Raw document test error: {e}")

def create_test_witness_data():
    """Create test data with known witness names."""
    print("\n=== CREATE TEST DATA ===")
    
    # Sample witness contexts that should have different names
    test_contexts = [
        {
            'witness': 'Charles Herbert Lightoller', 
            'testimony': 'Q: What was your position on the Titanic? A: I was Second Officer. Q: Can you tell us about the collision? A: I was off duty when the ship struck the iceberg.',
            'page_number': 1,
            'document_name': 'Test US Senate Inquiry'
        },
        {
            'witness': 'Harold Thomas Cottam',
            'testimony': 'Q: Were you the wireless operator on the Carpathia? A: Yes, I was. Q: When did you receive the distress call? A: I received the CQD call at approximately 12:35 AM.',
            'page_number': 2, 
            'document_name': 'Test US Senate Inquiry'
        },
        {
            'witness': 'Frederick Fleet',
            'testimony': 'Q: You were on lookout duty? A: Yes, sir. Q: What did you see? A: I saw an iceberg right ahead. I immediately rang the bell three times.',
            'page_number': 3,
            'document_name': 'Test US Senate Inquiry'
        }
    ]
    
    try:
        from Services.chunking import IntelligentChunker
        from Services.embeddings import EmbeddingService
        from Services.vector_storage import ChromaVectorStore
        
        print("Processing test witness contexts...")
        chunker = IntelligentChunker()
        embedding_service = EmbeddingService()  
        vector_store = ChromaVectorStore()
        
        # Process test contexts
        chunks = chunker.chunk_witness_contexts(test_contexts)
        print(f"Created {len(chunks)} test chunks")
        
        # Check witness names in chunks
        for i, chunk in enumerate(chunks):
            print(f"Test chunk {i+1}: witness='{chunk.witness_name}', content_preview='{chunk.content[:50]}...'")
        
        # Add to database with test prefix to identify them
        embeddings = []
        for chunk in chunks:
            embedding = embedding_service.embed_text(chunk.content)
            embeddings.append(embedding)
            
        vector_store.add_chunks(chunks, embeddings)
        print(f"Added {len(chunks)} test chunks to database")
        
        return True
        
    except Exception as e:
        print(f"Test data creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_current_database()
    test_raw_document()
    
    if create_test_witness_data():
        print("\n=== VERIFICATION ===")
        test_current_database()  # Re-run to see if test data was added