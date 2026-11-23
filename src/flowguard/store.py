"""Event storage. One row per accepted event, keyed by the caller's idempotency key."""
import hashlib
import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    idempotency_key TEXT PRIMARY KEY,
    event_id        TEXT NOT NULL,
    payload_hash    TEXT NOT NULL,
    payload         TEXT NOT NULL,
    received_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"""


class KeyReusedWithDifferentPayload(Exception):
    """The same idempotency key arrived with a different body; accepting it would hide a client bug."""


@dataclass(frozen=True)
class Accepted:
    event_id: str
    replayed: bool


def _hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class EventStore:
    def __init__(self, path: str = ":memory:"):
        # One connection shared across request threads, so every access takes the lock.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(SCHEMA)
            self._db.commit()

    def accept(self, key: str, payload: dict) -> Accepted:
        """Store the event once. A retry with the same key and body returns the first event_id."""
        digest = _hash(payload)
        with self._lock:
            row = self._db.execute(
                "SELECT event_id, payload_hash FROM events WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if row is not None:
                if row[1] != digest:
                    raise KeyReusedWithDifferentPayload(key)
                return Accepted(event_id=row[0], replayed=True)
            event_id = uuid.uuid4().hex
            self._db.execute(
                "INSERT INTO events (idempotency_key, event_id, payload_hash, payload) VALUES (?, ?, ?, ?)",
                (key, event_id, digest, json.dumps(payload, sort_keys=True)),
            )
            self._db.commit()
            return Accepted(event_id=event_id, replayed=False)

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
