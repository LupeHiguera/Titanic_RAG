import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import List, Dict, Any

from chunking import IntelligentChunker, ChunkMetadata, WitnessChunk


@dataclass
class MockWitnessContext:
    witness: str
    testimony: str
    page_number: int
    document_name: str


class TestIntelligentChunker:
    
    @pytest.fixture
    def chunker(self):
        return IntelligentChunker(
            chunk_size=500,
            overlap_size=50,
            preserve_witness_context=True
        )
    
    @pytest.fixture
    def sample_witness_contexts(self):
        return [
            MockWitnessContext(
                witness="Charles Herbert Lightoller",
                testimony="Q: What was your position? A: I was Second Officer. Q: What happened to the lifeboats? A: We lowered them in order, women and children first. There were not sufficient boats for all passengers aboard.",
                page_number=247,
                document_name="British Wreck Commissioner's Inquiry - Day 5"
            ),
            MockWitnessContext(
                witness="Hugh Woolner",
                testimony="Q: Did you see men in the lifeboats? A: Yes, I saw men getting into lifeboats when there were no more women nearby. The officers seemed to allow this when no women were present.",
                page_number=891,
                document_name="US Senate Inquiry - Day 12"
            )
        ]
    
    @pytest.fixture
    def long_testimony(self):
        return MockWitnessContext(
            witness="Frederick Fleet",
            testimony="Q: Tell us about the iceberg. A: " + "I was on lookout duty that night. " * 100 + "The iceberg appeared suddenly out of the haze. I rang the bell three times and telephoned the bridge immediately.",
            page_number=156,
            document_name="British Wreck Commissioner's Inquiry - Day 3"
        )
    
    def test_chunk_preserves_witness_identity(self, chunker, sample_witness_contexts):
        chunks = chunker.chunk_witness_contexts(sample_witness_contexts)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, WitnessChunk)
            assert chunk.witness_name is not None
            assert len(chunk.witness_name) > 0
    
    def test_chunk_maintains_document_metadata(self, chunker, sample_witness_contexts):
        chunks = chunker.chunk_witness_contexts(sample_witness_contexts)
        
        for chunk in chunks:
            assert chunk.metadata.page_number > 0
            assert chunk.metadata.document_name is not None
            assert chunk.metadata.source_type in ["us_inquiry", "british_inquiry", "other"]
    
    def test_chunk_respects_size_limits(self, chunker, sample_witness_contexts):
        chunks = chunker.chunk_witness_contexts(sample_witness_contexts)
        
        for chunk in chunks:
            assert len(chunk.content) <= chunker.chunk_size + chunker.overlap_size
    
    def test_chunk_handles_long_testimony_with_overlap(self, chunker, long_testimony):
        chunks = chunker.chunk_witness_contexts([long_testimony])
        
        assert len(chunks) > 1
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            prev_end = prev_chunk.content[-chunker.overlap_size:]
            curr_start = curr_chunk.content[:chunker.overlap_size]
            
            overlap_found = any(word in curr_start for word in prev_end.split()[-5:])
            assert overlap_found
    
    def test_chunk_preserves_question_answer_pairs(self, chunker, sample_witness_contexts):
        chunks = chunker.chunk_witness_contexts(sample_witness_contexts)
        
        for chunk in chunks:
            if "Q:" in chunk.content:
                assert "A:" in chunk.content
    
    def test_chunk_avoids_splitting_mid_sentence(self, chunker):
        context = MockWitnessContext(
            witness="Test Witness",
            testimony="This is a complete sentence. This is another complete sentence that should not be split in the middle of words or punctuation marks.",
            page_number=1,
            document_name="Test Document"
        )
        
        chunks = chunker.chunk_witness_contexts([context])
        
        for chunk in chunks:
            assert not chunk.content.endswith(" ")
            assert not chunk.content.startswith(" ")
    
    def test_identify_witness_credibility_ranking(self, chunker, sample_witness_contexts):
        chunks = chunker.chunk_witness_contexts(sample_witness_contexts)
        
        officer_chunks = [c for c in chunks if "officer" in c.witness_name.lower()]
        passenger_chunks = [c for c in chunks if "officer" not in c.witness_name.lower()]
        
        if officer_chunks and passenger_chunks:
            assert officer_chunks[0].metadata.credibility_score > passenger_chunks[0].metadata.credibility_score
    
    def test_chunk_metadata_includes_context_window(self, chunker, sample_witness_contexts):
        chunks = chunker.chunk_witness_contexts(sample_witness_contexts)
        
        for chunk in chunks:
            assert hasattr(chunk.metadata, 'chunk_index')
            assert hasattr(chunk.metadata, 'total_chunks_for_witness')
            assert chunk.metadata.chunk_index >= 0
            assert chunk.metadata.total_chunks_for_witness >= 1
    
    def test_group_chunks_by_topic_similarity(self, chunker, sample_witness_contexts):
        chunks = chunker.chunk_witness_contexts(sample_witness_contexts)
        grouped = chunker.group_chunks_by_topic(chunks)
        
        assert isinstance(grouped, dict)
        lifeboat_chunks = [chunk for topic, topic_chunks in grouped.items() 
                          if "lifeboat" in topic.lower() 
                          for chunk in topic_chunks]
        assert len(lifeboat_chunks) > 0
    
    def test_extract_contradictory_statements(self, chunker, sample_witness_contexts):
        chunks = chunker.chunk_witness_contexts(sample_witness_contexts)
        contradictions = chunker.find_potential_contradictions(chunks)
        
        assert isinstance(contradictions, list)
        if len(contradictions) > 0:
            for contradiction in contradictions:
                assert len(contradiction["conflicting_chunks"]) >= 2
                assert "topic" in contradiction
                assert "confidence_score" in contradiction