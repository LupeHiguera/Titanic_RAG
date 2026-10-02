import pytest

from Services.chunking import BritishBoundarySplitter, IntelligentChunker


@pytest.mark.parametrize('overlap_size', [0, 1, 9])
def test_zero_effective_overlap_does_not_copy_previous_chunk(overlap_size):
    chunker = IntelligentChunker(overlap_size=overlap_size)
    assert chunker._add_overlap(['One two three four.', 'Five six.']) == [
        'One two three four.', 'Five six.'
    ]


def test_single_word_previous_chunk_does_not_become_overlap():
    assert IntelligentChunker()._add_overlap(['First', 'Second']) == ['First', 'Second']


def test_positive_overlap_copies_only_bounded_tail():
    assert IntelligentChunker(overlap_size=20)._add_overlap(
        ['One two three four.', 'Five six seven eight.', 'Nine ten.']
    ) == ['One two three four.', 'three four. Five six seven eight.', 'seven eight. Nine ten.']


def test_zero_overlap_preserves_whole_qa_and_page_citations():
    exchanges = ['1. Who? - First witness.', '2. When? - After midnight.', '3. Where? - On deck.']
    chunks = IntelligentChunker(
        chunk_size=40, overlap_size=0, splitter=BritishBoundarySplitter()
    ).chunk_witness_contexts([{
        'witness': 'Witness', 'document_name': 'British inquiry', 'page_number': 10,
        'testimony': f'⟦p:10⟧{exchanges[0]}\n⟦p:11⟧{exchanges[1]}\n{exchanges[2]}',
    }])
    assert [c.content for c in chunks] == exchanges
    assert [c.metadata.page_number for c in chunks] == [10, 11, 11]
