"""DynamoDBCache unit tests — mocks boto3, no real AWS calls."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_boto3():
    """Patches `boto3` inside contradiction_cache so .Table() returns a mock."""
    fake_table = MagicMock()
    fake_dynamodb = MagicMock()
    fake_dynamodb.Table.return_value = fake_table

    with patch("boto3.resource", return_value=fake_dynamodb) as mock_resource:
        yield {"resource": mock_resource, "dynamodb": fake_dynamodb, "table": fake_table}


def test_get_returns_value_when_item_present(fake_boto3):
    from Services.contradiction_cache import DynamoDBCache

    fake_boto3["table"].get_item.return_value = {
        "Item": {"cache_key": "k1", "value": json.dumps({"contradicts": True, "confidence": 0.9})}
    }
    cache = DynamoDBCache(table_name="test-table")
    result = cache.get("k1")
    assert result == {"contradicts": True, "confidence": 0.9}
    fake_boto3["table"].get_item.assert_called_once_with(Key={"cache_key": "k1"})


def test_get_returns_none_on_cache_miss(fake_boto3):
    from Services.contradiction_cache import DynamoDBCache

    fake_boto3["table"].get_item.return_value = {}
    cache = DynamoDBCache(table_name="test-table")
    assert cache.get("missing") is None


def test_get_returns_none_on_boto_error(fake_boto3):
    """Cache errors must not propagate — degraded cache still serves traffic."""
    from Services.contradiction_cache import DynamoDBCache

    fake_boto3["table"].get_item.side_effect = RuntimeError("simulated AWS outage")
    cache = DynamoDBCache(table_name="test-table")
    assert cache.get("k") is None


def test_get_returns_none_on_invalid_json(fake_boto3):
    from Services.contradiction_cache import DynamoDBCache

    fake_boto3["table"].get_item.return_value = {
        "Item": {"cache_key": "k", "value": "not-valid-json{"}
    }
    cache = DynamoDBCache(table_name="test-table")
    assert cache.get("k") is None


def test_put_serializes_value_and_sets_ttl(fake_boto3):
    from Services.contradiction_cache import DynamoDBCache

    cache = DynamoDBCache(table_name="test-table", ttl_seconds=3600)
    cache.put("k1", {"contradicts": False, "confidence": 0.1})

    fake_boto3["table"].put_item.assert_called_once()
    item = fake_boto3["table"].put_item.call_args.kwargs["Item"]
    assert item["cache_key"] == "k1"
    assert json.loads(item["value"]) == {"contradicts": False, "confidence": 0.1}
    assert "expires_at" in item and isinstance(item["expires_at"], int)


def test_put_swallows_boto_error(fake_boto3):
    """Put errors must not crash the request path."""
    from Services.contradiction_cache import DynamoDBCache

    fake_boto3["table"].put_item.side_effect = RuntimeError("simulated throttling")
    cache = DynamoDBCache(table_name="test-table")
    cache.put("k", {"contradicts": True, "confidence": 0.5})  # no raise


def test_table_name_from_env_when_not_passed(fake_boto3, monkeypatch):
    from Services.contradiction_cache import DynamoDBCache

    monkeypatch.setenv("CONTRADICTION_CACHE_TABLE", "env-supplied-table")
    cache = DynamoDBCache()
    assert cache.table_name == "env-supplied-table"
    fake_boto3["dynamodb"].Table.assert_called_with("env-supplied-table")


def test_factory_picks_dynamodb_via_env(fake_boto3, monkeypatch):
    from Services.contradiction_cache import DynamoDBCache, make_cache

    monkeypatch.setenv("CACHE_BACKEND", "dynamodb")
    monkeypatch.setenv("CONTRADICTION_CACHE_TABLE", "from-env")
    cache = make_cache()
    assert isinstance(cache, DynamoDBCache)
