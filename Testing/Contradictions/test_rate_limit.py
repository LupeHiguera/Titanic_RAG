"""Rate-limit smoke test — verify slowapi 429s on excess requests."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Build the app with a tight rate limit and patched dependencies so the
    test never reaches Pinecone or OpenAI."""
    # Tighter limits so the test runs fast.
    monkeypatch.setenv("RATE_LIMIT_SEARCH", "3/minute")
    monkeypatch.setenv("RATE_LIMIT_CONTRADICTIONS", "1/minute")

    # Stub out vector store at import time so PineconeVectorStore() in app
    # module-init doesn't try to talk to Pinecone.
    with patch("Services.vector_storage.PineconeVectorStore") as mock_store_cls:
        mock_store_cls.return_value.get_collection_stats.return_value = {
            "total_chunks": 0, "index_name": "test", "dimension": 1024
        }
        # Fresh import so the slowapi decorators see the patched env values.
        import importlib
        import app as app_module
        importlib.reload(app_module)

        # Stub the search engine so /search returns quickly.
        with patch.object(app_module, "get_search_engine") as mock_get_engine:
            mock_get_engine.return_value.search.return_value = []
            yield TestClient(app_module.app)


def test_search_429s_after_burst(client):
    payload = {"query": "ice", "top_k": 3, "similarity_threshold": 0.5}

    # Limit is 3/minute. The 4th request should hit 429.
    for _ in range(3):
        r = client.post("/search", json=payload)
        assert r.status_code == 200, r.text

    r = client.post("/search", json=payload)
    assert r.status_code == 429
    assert "rate limit" in r.text.lower() or "too many" in r.text.lower()


def test_contradictions_429s_immediately_on_second_call(client):
    payload = {"query": "speed", "top_k": 3, "similarity_threshold": 0.5}

    r = client.post("/search/contradictions", json=payload)
    assert r.status_code in (200, 500)  # 500 ok if engine stub doesn't have the method

    r2 = client.post("/search/contradictions", json=payload)
    assert r2.status_code == 429
