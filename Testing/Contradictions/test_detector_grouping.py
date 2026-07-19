"""Detector grouping: (witness, inquiry) keys let same-named cross-inquiry
witnesses (Ismay, Lord, Fleet...) be compared against themselves, and the
same_person flag identifies cross-inquiry self-contradictions."""

from unittest.mock import MagicMock

import numpy as np

from Services.chunking import WitnessChunk, ChunkMetadata
from Services.contradiction_detector import ContradictionDetector, ContradictionVerdict
from Services.embeddings import EmbeddedChunk


def chunk(witness, source, content, page=1, role=""):
    meta = ChunkMetadata(
        document_name="doc", source_type=source, page_number=page,
        credibility_score=0.0, chunk_index=0, total_chunks_for_witness=1,
        role=role,
    )
    return EmbeddedChunk(
        chunk=WitnessChunk(content=content, witness_name=witness, metadata=meta),
        embedding=np.array([]),
    )


class DictCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def put(self, key, value):
        self.store[key] = value


def make_detector(verdict: ContradictionVerdict):
    client = MagicMock()
    client.messages.parse.return_value = MagicMock(parsed_output=verdict)
    return ContradictionDetector(cache=DictCache(), client=client), client


CONTRADICTS = ContradictionVerdict(
    contradicts=True, claim_a="45 aboard", claim_b="12 aboard",
    confidence=0.9, explanation="Counts conflict.",
)


class TestGrouping:
    def test_same_name_cross_inquiry_is_compared(self):
        # Ismay has the SAME name string in both inquiries — under name-only
        # grouping this pair was structurally unreachable.
        detector, client = make_detector(CONTRADICTS)
        chunks = [
            chunk("J. Bruce Ismay", "us_inquiry", "There were 45 people in my boat.", page=8),
            chunk("J. Bruce Ismay", "british_inquiry", "About 12 people were aboard.", page=440),
        ]
        result = detector.detect(chunks, "lifeboat count")
        assert len(result) == 1
        assert result[0].same_person is True
        assert result[0].source_a != result[0].source_b
        assert client.messages.parse.called

    def test_same_name_same_inquiry_not_compared(self):
        detector, client = make_detector(CONTRADICTS)
        chunks = [
            chunk("J. Bruce Ismay", "us_inquiry", "Statement one.", page=8),
            chunk("J. Bruce Ismay", "us_inquiry", "Statement two.", page=9),
        ]
        assert detector.detect(chunks, "anything") == []
        assert not client.messages.parse.called

    def test_canonical_alias_marks_same_person(self):
        detector, _ = make_detector(CONTRADICTS)
        chunks = [
            chunk("Charles Herbert Lightoller", "us_inquiry", "I left in boat D.", page=46),
            chunk("Charles Lightoller", "british_inquiry", "I never left in a boat.", page=310),
        ]
        result = detector.detect(chunks, "how did you leave")
        assert len(result) == 1
        assert result[0].same_person is True

    def test_different_people_not_flagged_same_person(self):
        detector, _ = make_detector(CONTRADICTS)
        chunks = [
            chunk("Stanley Lord", "us_inquiry", "We saw no rockets.", page=714, role="Captain, Californian"),
            chunk("Ernest Gill", "us_inquiry", "I saw rockets fired.", page=710, role="Donkeyman, Californian"),
        ]
        result = detector.detect(chunks, "rockets")
        assert len(result) == 1
        assert result[0].same_person is False
        assert result[0].role_a == "Captain, Californian"
        assert result[0].page_a == 714

    def test_one_failed_pair_does_not_sink_request(self):
        client = MagicMock()
        client.messages.parse.side_effect = [RuntimeError("boom"),
                                             MagicMock(parsed_output=CONTRADICTS)]
        # single worker so the side_effect sequence is consumed deterministically
        detector = ContradictionDetector(cache=DictCache(), client=client, max_workers=1)
        chunks = [
            chunk("A", "us_inquiry", "one"),
            chunk("B", "us_inquiry", "two"),
            chunk("C", "us_inquiry", "three"),
        ]
        # 3 pairs, one errors → the others still return
        result = detector.detect(chunks, "q")
        assert len(result) >= 1
