"""Pluggable cache for contradiction-detection verdicts.

Local dev uses SQLite at ./cache/contradictions.db. Production swaps to
DynamoDB via the CACHE_BACKEND env var without code changes elsewhere.
"""
from __future__ import annotations

import json
import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


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
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS contradictions ("
            "  key TEXT PRIMARY KEY,"
            "  value TEXT NOT NULL,"
            "  created_at REAL NOT NULL DEFAULT (strftime('%s','now'))"
            ")"
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT value FROM contradictions WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO contradictions (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self._conn.commit()


class DynamoDBCache(ContradictionCache):
    def __init__(self, table_name: Optional[str] = None):
        self.table_name = table_name or os.getenv("CONTRADICTION_CACHE_TABLE")
        raise NotImplementedError(
            "DynamoDBCache is a stub for Phase 4 (production hosting). "
            "Implement with boto3 when wiring up App Runner deployment."
        )

    def get(self, key: str) -> Optional[dict]:
        raise NotImplementedError

    def put(self, key: str, value: dict) -> None:
        raise NotImplementedError


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