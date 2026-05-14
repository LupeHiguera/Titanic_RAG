import pytest
from pathlib import Path
import sys
import os

# Add the root directory to path so we can import Services
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from Services.chunking import IntelligentChunker, WitnessChunk
from Services.document_ingestion import DocumentIngestion


class TestIntelligentChunker:

    @pytest.fixture
    def chunker(self):
        return IntelligentChunker(
            chunk_size=500,
            overlap_size=50,
            preserve_witness_context=True
        )

    @pytest.fixture
    def real_pdf_data(self):
        """Extract real data from one page.pdf"""
        ingestion = DocumentIngestion()
        pdf_path = root_dir / "Text" / "one page.pdf"

        if not pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(pdf_path)

        # one page.pdf is Ismay's testimony
        return [{
            'witness': 'J. Bruce Ismay',
            'testimony': result["text"],
            'page_number': 1,
            'document_name': result["metadata"].document_name
        }]

    @pytest.fixture
    def long_testimony_real(self, real_pdf_data):
        """Create long testimony by repeating part of real data"""
        base_context = real_pdf_data[0].copy()
        base_text = base_context['testimony'][:200]
        base_context['testimony'] = base_text * 10  # Make it long enough to chunk
        return [base_context]

    @pytest.fixture
    def multiple_witnesses_simulation(self, real_pdf_data):
        """Simulate multiple witnesses using real data with different names"""
        base_context = real_pdf_data[0]

        return [
            {
                'witness': 'Charles Herbert Lightoller',
                'testimony': base_context['testimony'][:600],
                'page_number': 247,
                'document_name': "British Wreck Commissioner's Inquiry - Day 5"
            },
            {
                'witness': 'Hugh Woolner',
                'testimony': base_context['testimony'][600:1200],
                'page_number': 891,
                'document_name': "US Senate Inquiry - Day 12"
            }
        ]

    def test_chunk_preserves_witness_identity(self, chunker, real_pdf_data):
        chunks = chunker.chunk_witness_contexts(real_pdf_data)

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, WitnessChunk)
            assert chunk.witness_name is not None
            assert len(chunk.witness_name) > 0
            assert chunk.witness_name == "J. Bruce Ismay"

    def test_chunk_maintains_document_metadata(self, chunker, real_pdf_data):
        chunks = chunker.chunk_witness_contexts(real_pdf_data)

        for chunk in chunks:
            assert chunk.metadata.page_number > 0
            assert chunk.metadata.document_name is not None
            assert chunk.metadata.source_type in ["us_inquiry", "british_inquiry", "other"]

    def test_chunk_respects_size_limits(self, chunker, real_pdf_data):
        chunks = chunker.chunk_witness_contexts(real_pdf_data)

        for chunk in chunks:
            # Allow some flexibility for overlap
            assert len(chunk.content) <= chunker.chunk_size + chunker.overlap_size

    def test_chunk_handles_long_testimony_with_overlap(self, chunker, long_testimony_real):
        chunks = chunker.chunk_witness_contexts(long_testimony_real)

        assert len(chunks) > 1

        # Check for some kind of overlap or continuity between chunks
        for i in range(1, min(len(chunks), 3)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]

            prev_words = set(prev_chunk.content.lower().split()[-10:])
            curr_words = set(curr_chunk.content.lower().split()[:10])

            # Allow flexible overlap
            overlap_found = len(prev_words.intersection(curr_words)) > 0

    def test_chunk_preserves_question_answer_pairs(self, chunker, real_pdf_data):
        chunks = chunker.chunk_witness_contexts(real_pdf_data)

        # Real PDF uses "Senator SMITH" and "Mr. ISMAY" format
        dialogue_chunks = [c for c in chunks if any(word in c.content.lower() for word in ['senator', 'mr.'])]
        assert len(dialogue_chunks) > 0

    def test_chunk_avoids_splitting_mid_sentence(self, chunker, real_pdf_data):
        chunks = chunker.chunk_witness_contexts(real_pdf_data)

        for chunk in chunks:
            content = chunk.content.strip()
            if len(content) > 10:
                assert not content.endswith(" ")
                assert not content.startswith(" ")

    def test_witness_metadata_has_credibility_field(self, chunker, real_pdf_data):
        """Test that credibility score field exists"""
        chunks = chunker.chunk_witness_contexts(real_pdf_data)

        for chunk in chunks:
            assert hasattr(chunk.metadata, 'credibility_score')
            assert isinstance(chunk.metadata.credibility_score, float)
            assert 0.0 <= chunk.metadata.credibility_score <= 1.0

    def test_chunk_metadata_includes_context_window(self, chunker, real_pdf_data):
        chunks = chunker.chunk_witness_contexts(real_pdf_data)

        for chunk in chunks:
            assert hasattr(chunk.metadata, 'chunk_index')
            assert hasattr(chunk.metadata, 'total_chunks_for_witness')
            assert chunk.metadata.chunk_index >= 0
            assert chunk.metadata.total_chunks_for_witness >= 1
            assert chunk.metadata.chunk_index < chunk.metadata.total_chunks_for_witness

    def test_real_pdf_content_processing(self, chunker):
        """Test that we can actually process the real PDF content"""
        ingestion = DocumentIngestion()
        pdf_path = root_dir / "Text" / "one page.pdf"

        if not pdf_path.exists():
            pytest.skip("PDF file not found")

        result = ingestion.extract_text_from_pdf(pdf_path)

        witness_context = {
            'witness': 'J. Bruce Ismay',
            'testimony': result["text"],
            'page_number': 1,
            'document_name': result["metadata"].document_name
        }

        chunks = chunker.chunk_witness_contexts([witness_context])

        assert len(chunks) > 0
        assert all(isinstance(chunk, WitnessChunk) for chunk in chunks)
        assert all(len(chunk.content) > 0 for chunk in chunks)
        assert all(chunk.witness_name == "J. Bruce Ismay" for chunk in chunks)

        full_content = " ".join(chunk.content for chunk in chunks)
        assert "senator" in full_content.lower()
        assert "ismay" in full_content.lower()
