from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Evals.run_retrieval_eval import evaluate


def run(names, relevant):
    engine = Mock()
    engine.search.return_value = [
        SimpleNamespace(chunk=SimpleNamespace(chunk=SimpleNamespace(witness_name=name)))
        for name in names
    ]
    return evaluate(engine, [{'query': 'q', 'topic': 'test', 'relevant_witnesses': relevant}], 0.5)


def test_repeated_witnesses_do_not_promote_sixth_chunk_into_top_five():
    result = run(['A'] * 5 + ['B'], ['B'])
    assert result['hit_rate_at_5'] == 0
    assert result['recall_at_5_mean'] == 0
    assert result['recall_at_10_mean'] == 1
    assert result['mrr'] == pytest.approx(1 / 6)
    assert result['per_query'][0]['retrieved_top5'] == ['A'] * 5


def test_recall_counts_each_relevant_witness_once_within_chunk_cutoff():
    result = run(['A'] * 5 + ['B'] * 5 + ['C'], ['A', 'B', 'C'])
    assert result['recall_at_5_mean'] == pytest.approx(1 / 3)
    assert result['recall_at_10_mean'] == pytest.approx(2 / 3)
    assert result['mrr'] == 1


@pytest.mark.parametrize('names', [[], ['irrelevant'] * 10])
def test_no_relevant_results(names):
    result = run(names, ['A'])
    assert result['hit_rate_at_5'] == result['recall_at_10_mean'] == result['mrr'] == 0
