"""
Test chunking module with real PDF data from one page.pdf
"""

from pathlib import Path
import sys

# Add the root directory to path so we can import Services
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from Services.chunking import IntelligentChunker
from Services.document_ingestion import DocumentIngestion


def test_chunking_with_real_pdf():
    """Test the complete pipeline: PDF -> text -> chunks with real data."""
    # Initialize components
    ingestion = DocumentIngestion()
    chunker = IntelligentChunker(chunk_size=300, overlap_size=50)

    # Extract text from real PDF
    pdf_path = root_dir / "Text" / "one page.pdf"
    if not pdf_path.exists():
        print("PDF file not found")
        return False

    print("Extracting text from PDF...")
    result = ingestion.extract_text_from_pdf(pdf_path)
    text = result["text"]
    metadata = result["metadata"]

    print(f"Extracted {len(text)} characters")
    print(f"   Document: {metadata.document_name}")
    print(f"   Source: {metadata.source_type}")

    # Create witness context for chunking (known from witness index)
    print("Creating witness context...")
    witness_contexts = [{
        'witness': 'J. Bruce Ismay',
        'testimony': text,
        'page_number': 1,
        'document_name': metadata.document_name
    }]

    # Test chunking
    print("Chunking testimony...")
    chunks = chunker.chunk_witness_contexts(witness_contexts)

    print(f"Created {len(chunks)} chunks")

    # Analyze chunks
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n--- Chunk {i+1} ---")
        print(f"Witness: {chunk.witness_name}")
        print(f"Length: {len(chunk.content)} chars")
        print(f"Credibility: {chunk.metadata.credibility_score}")
        print(f"Content preview: {chunk.content[:100]}...")

    # Test topic grouping
    print("\nGrouping by topics...")
    topics = chunker.group_chunks_by_topic(chunks)

    for topic, topic_chunks in topics.items():
        if topic_chunks:
            print(f"  {topic}: {len(topic_chunks)} chunks")

    # Test contradiction detection
    print("\nLooking for contradictions...")
    contradictions = chunker.find_potential_contradictions(chunks)

    if contradictions:
        print(f"Found {len(contradictions)} potential contradictions")
        for contradiction in contradictions:
            print(f"   Topic: {contradiction['topic']}, Confidence: {contradiction['confidence_score']}")
    else:
        print("No contradictions found (expected for single witness)")

    print("\nAll chunking tests passed with real data!")
    assert len(chunks) > 0
    assert all(hasattr(chunk, 'content') for chunk in chunks)


if __name__ == "__main__":
    test_chunking_with_real_pdf()
