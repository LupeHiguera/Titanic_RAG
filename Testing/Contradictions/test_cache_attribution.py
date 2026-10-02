from unittest.mock import MagicMock

from Services.contradiction_detector import ContradictionDetector, ContradictionVerdict
from Testing.Contradictions.test_detector_grouping import DictCache, chunk


def test_reversed_pair_keeps_claims_and_directional_explanation_with_witnesses():
    client = MagicMock()
    client.messages.parse.side_effect = [
        MagicMock(parsed_output=ContradictionVerdict(
            contradicts=True, claim_a='45 aboard', claim_b='12 aboard',
            confidence=0.9, explanation='A reports more than B.',
        )),
        MagicMock(parsed_output=ContradictionVerdict(
            contradicts=True, claim_a='12 aboard', claim_b='45 aboard',
            confidence=0.9, explanation='A reports fewer than B.',
        )),
    ]
    detector = ContradictionDetector(cache=DictCache(), client=client)
    a = chunk('Ismay', 'us_inquiry', '45 aboard', page=8)
    b = chunk('Lowe', 'british_inquiry', '12 aboard', page=44)
    first = detector.detect([a, b], 'count')[0]
    reverse = detector.detect([b, a], 'count')[0]
    assert (first.witness_a, first.claim_a, first.page_a) == ('Ismay', '45 aboard', 8)
    assert (reverse.witness_a, reverse.claim_a, reverse.page_a) == ('Lowe', '12 aboard', 44)
    assert (reverse.witness_b, reverse.claim_b, reverse.page_b) == ('Ismay', '45 aboard', 8)
    assert reverse.explanation == 'A reports fewer than B.'
    assert detector.detect([a, b], 'count')[0] == first
    assert detector.detect([b, a], 'count')[0] == reverse
    assert client.messages.parse.call_count == 2


def test_cache_identity_includes_model_prompt_schema_and_exact_query(monkeypatch):
    import Services.contradiction_detector as module

    detector = ContradictionDetector(cache=DictCache(), client=MagicMock())
    a, b = chunk('A', 'us_inquiry', 'one'), chunk('B', 'us_inquiry', 'two')
    original = detector._cache_key(a, b, 'Count')
    assert original != detector._cache_key(a, b, 'count')
    assert original != detector._cache_key(b, a, 'Count')
    detector.model = 'another-model'
    assert original != detector._cache_key(a, b, 'Count')
    detector.model = module.MODEL_ID
    monkeypatch.setattr(module, 'SYSTEM_PROMPT', 'Changed instructions')
    assert original != detector._cache_key(a, b, 'Count')
    monkeypatch.undo()
    monkeypatch.setattr(module.ContradictionVerdict, 'model_json_schema', lambda: {'changed': True})
    assert original != detector._cache_key(a, b, 'Count')
