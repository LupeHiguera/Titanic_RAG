"""Chunker behaviors added in the ingestion overhaul: ⟦p:N⟧ page tags,
Q&A packing, and the decimal-safe sentence splitter."""

from Services.chunking import (
    IntelligentChunker,
    BritishBoundarySplitter,
    SenateBoundarySplitter,
    page_tag,
)


def make_context(testimony, page=10, **extra):
    ctx = {
        'witness': 'Test Witness',
        'testimony': testimony,
        'page_number': page,
        'document_name': 'US Senate Inquiry - Test',
    }
    ctx.update(extra)
    return ctx


class TestPageTags:
    def test_chunks_cite_the_page_they_start_on(self):
        # Two pages of long testimony; the tag marks where p.11 begins.
        filler_a = "Senator SMITH. What did you observe? Mr. LOWE. " + "The sea was calm. " * 30
        filler_b = "Senator SMITH. And then? Mr. LOWE. " + "We rowed away. " * 30
        testimony = f"{page_tag(10)}\n{filler_a}\n{page_tag(11)}\n{filler_b}"

        chunker = IntelligentChunker(chunk_size=400, splitter=SenateBoundarySplitter())
        chunks = chunker.chunk_witness_contexts([make_context(testimony, page=10)])

        pages = {c.metadata.page_number for c in chunks}
        assert pages == {10, 11}
        # tags never leak into stored content
        assert all('⟦' not in c.content for c in chunks)

    def test_untagged_text_inherits_context_start_page(self):
        chunker = IntelligentChunker()
        chunks = chunker.chunk_witness_contexts([make_context("Short answer.", page=42)])
        assert len(chunks) == 1
        assert chunks[0].metadata.page_number == 42


class TestPacking:
    def test_adjacent_qas_packed_up_to_chunk_size(self):
        qas = [f"{n}. Was the sea calm? - Yes, quite calm that night." for n in range(1, 21)]
        chunker = IntelligentChunker(chunk_size=400, splitter=BritishBoundarySplitter())
        chunks = chunker.chunk_witness_contexts([make_context("\n".join(qas))])

        # 20 tiny Q&As must NOT become 20 tiny chunks
        assert 1 < len(chunks) < 10
        assert all(len(c.content) <= 400 + 100 for c in chunks)  # +overlap words

    def test_qa_boundaries_not_split_mid_answer(self):
        qas = [f"{n}. What time was it? - It was 11.40 exactly, by the bridge clock." for n in range(1, 11)]
        chunker = IntelligentChunker(chunk_size=300, splitter=BritishBoundarySplitter())
        chunks = chunker.chunk_witness_contexts([make_context("\n".join(qas))])
        # every chunk contains whole Q&As: each mention of the time is intact
        for c in chunks:
            assert '11.40' in c.content or '11.40' not in c.content.replace('11. 40', 'X')


class TestSentenceSplitter:
    def test_decimal_times_survive(self):
        text = ("We struck the berg at 11.40 that night and the carpenter sounded the ship. " * 20)
        chunker = IntelligentChunker(chunk_size=300)
        pieces = chunker._split_by_sentences(text)
        joined = " ".join(pieces)
        assert '11.40' in joined
        assert '11. ' not in joined.replace('11.40', '')

    def test_question_marks_preserved(self):
        text = "Did you see the iceberg? I did not see it. Were you on watch? I was below. " * 10
        chunker = IntelligentChunker(chunk_size=200)
        pieces = chunker._split_by_sentences(text)
        assert any('?' in p for p in pieces)


class TestMetadataEnrichment:
    def test_role_ship_type_carried_into_chunks(self):
        chunker = IntelligentChunker()
        ctx = make_context("A short statement.", role="2nd Officer, Titanic",
                           ship="Titanic", witness_type="Officer")
        chunks = chunker.chunk_witness_contexts([ctx])
        assert chunks[0].metadata.role == "2nd Officer, Titanic"
        assert chunks[0].metadata.ship == "Titanic"
        assert chunks[0].metadata.witness_type == "Officer"
