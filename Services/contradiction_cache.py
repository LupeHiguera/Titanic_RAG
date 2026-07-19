"""Pluggable cache for contradiction-detection verdicts.

Local dev uses SQLite at ./cache/contradictions.db. Production swaps to
DynamoDB via the CACHE_BACKEND env var without code changes elsewhere.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ContradictionCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[dict]:
        ...

    @abstractmethod
    def put(self, key: str, value: dict) -> None:
        ...


class SQLiteCache(ContradictionCache):
    def __init__(self, path: str = "./cache/contradictions.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        # One connection shared across the detector's worker threads — sqlite3
        # connections are not safe for concurrent use, so serialize access.
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS contradictions ("
                "  key TEXT PRIMARY KEY,"
                "  value TEXT NOT NULL,"
                "  created_at REAL NOT NULL DEFAULT (strftime('%s','now'))"
                ")"
            )
            self._conn.commit()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM contradictions WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO contradictions (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
            self._conn.commit()


class DynamoDBCache(ContradictionCache):
    """Production cache backed by DynamoDB.

    Schema:
      Table: <CONTRADICTION_CACHE_TABLE> (default: titanic-rag-contradictions)
      Partition key: cache_key (S)
      Attributes:
        value      (S) — JSON-serialized verdict
        expires_at (N) — epoch seconds, used by DynamoDB TTL (set on the
                         table with `aws dynamodb update-time-to-live`)

    Cache errors never block the request path — get() and put() log and
    return None / no-op on transient failures so a degraded cache backend
    still lets the LLM path run.
    """

    # Verdicts about static historical testimony don't really "expire" but
    # we set a TTL so dev/preview tables don't accumulate forever.
    DEFAULT_TTL_SECONDS = 90 * 24 * 60 * 60  # 90 days

    def __init__(self, table_name: Optional[str] = None, ttl_seconds: Optional[int] = None):
        # Lazy import — boto3 is a heavy dep we only need in prod.
        import boto3

        self.table_name = table_name or os.getenv(
            "CONTRADICTION_CACHE_TABLE", "titanic-rag-contradictions"
        )
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else self.DEFAULT_TTL_SECONDS
        # Region from boto3 default chain: AWS_REGION env, profile, then IMDS
        # on App Runner. boto3.resource("dynamodb") will raise NoRegionError
        # at first call if none of these are set — caught at request time
        # by our error handlers.
        self._table = boto3.resource("dynamodb").Table(self.table_name)

    def get(self, key: str) -> Optional[dict]:
        try:
            resp = self._table.get_item(Key={"cache_key": key})
        except Exception:
            logger.exception("DynamoDB get_item failed for key=%s", key)
            return None
        item = resp.get("Item")
        if not item:
            return None
        raw = item.get("value")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.exception("DynamoDB cache value not valid JSON for key=%s", key)
            return None

    def put(self, key: str, value: dict) -> None:
        item = {
            "cache_key": key,
            "value": json.dumps(value),
        }
        if self.ttl_seconds:
            item["expires_at"] = int(time.time()) + self.ttl_seconds
        try:
            self._table.put_item(Item=item)
        except Exception:
            logger.exception("DynamoDB put_item failed for key=%s", key)
            # Swallow — cache write failures must not break the request.


def make_cache() -> ContradictionCache:
    """Factory — reads CACHE_BACKEND env var. Defaults to SQLite."""
    backend = os.getenv("CACHE_BACKEND", "sqlite").lower()
    if backend == "sqlite":
        return SQLiteCache(
            path=os.getenv("CONTRADICTION_CACHE_PATH", "./cache/contradictions.db")
        )
    if backend == "dynamodb":
        return DynamoDBCache()
    raise ValueError(f"Unknown CACHE_BACKEND: {backend!r}")