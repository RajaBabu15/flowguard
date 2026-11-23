"""Event storage and the work queue on top of it.

One SQLite file holds the events, their state, and an append-only audit log. Every method takes the
lock, so one process can share a connection across threads. Several processes need a server database.
"""
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    idempotency_key TEXT PRIMARY KEY,
    event_id        TEXT NOT NULL UNIQUE,
    payload_hash    TEXT NOT NULL,
    payload         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    attempts        INTEGER NOT NULL DEFAULT 0,
    not_before      REAL NOT NULL DEFAULT 0,
    leased_until    REAL NOT NULL DEFAULT 0,
    result          TEXT,
    last_error      TEXT
);
CREATE TABLE IF NOT EXISTS audit (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    at       REAL NOT NULL,
    kind     TEXT NOT NULL,
    detail   TEXT NOT NULL DEFAULT ''
);
"""

# pending -> processing -> done | needs_human | dead.  A retry sends processing back to pending.
STATUSES = ("pending", "processing", "done", "needs_human", "dead")


class KeyReusedWithDifferentPayload(Exception):
    """The same idempotency key arrived with a different body; accepting it would hide a client bug."""


class LeaseLost(Exception):
    """The caller no longer owns this event: another worker claimed it after the lease expired."""


@dataclass(frozen=True)
class Accepted:
    event_id: str
    replayed: bool


@dataclass(frozen=True)
class Claim:
    event_id: str
    payload: dict
    attempt: int


def _hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class EventStore:
    def __init__(self, path: str = ":memory:"):
        # One connection shared across request threads, so every access takes the lock.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def _audit(self, event_id: str, at: float, kind: str, detail: str = "") -> None:
        self._db.execute("INSERT INTO audit (event_id, at, kind, detail) VALUES (?, ?, ?, ?)", (event_id, at, kind, detail))

    def accept(self, key: str, payload: dict, now: float | None = None) -> Accepted:
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
            self._audit(event_id, time.time() if now is None else now, "accepted")
            self._db.commit()
            return Accepted(event_id=event_id, replayed=False)

    def claim(self, now: float, lease_seconds: float, max_attempts: int) -> Claim | None:
        """Hand the oldest ready event to one caller, or None. A claim whose lease ran out is claimable again."""
        with self._lock:
            stuck = self._db.execute(
                "SELECT event_id FROM events WHERE status = 'processing' AND leased_until < ? AND attempts >= ?",
                (now, max_attempts),
            ).fetchall()
            for (event_id,) in stuck:
                self._db.execute(
                    "UPDATE events SET status = 'dead', last_error = 'lease expired on the final attempt' WHERE event_id = ?",
                    (event_id,),
                )
                self._audit(event_id, now, "dead", "lease expired on the final attempt")
            row = self._db.execute(
                "SELECT event_id, payload, attempts FROM events "
                "WHERE (status = 'pending' AND not_before <= ?) OR (status = 'processing' AND leased_until < ?) "
                "ORDER BY rowid LIMIT 1",
                (now, now),
            ).fetchone()
            if row is None:
                self._db.commit()
                return None
            event_id, payload, attempts = row
            attempt = attempts + 1
            self._db.execute(
                "UPDATE events SET status = 'processing', attempts = ?, leased_until = ? WHERE event_id = ?",
                (attempt, now + lease_seconds, event_id),
            )
            self._audit(event_id, now, "claimed", f"attempt {attempt}")
            self._db.commit()
            return Claim(event_id=event_id, payload=json.loads(payload), attempt=attempt)

    def _owned(self, event_id: str, attempt: int) -> None:
        row = self._db.execute("SELECT status, attempts FROM events WHERE event_id = ?", (event_id,)).fetchone()
        if row is None or row[0] != "processing" or row[1] != attempt:
            raise LeaseLost(event_id)

    def complete(self, event_id: str, attempt: int, status: str, result: dict, now: float) -> None:
        if status not in ("done", "needs_human"):
            raise ValueError(f"cannot complete with status {status}")
        with self._lock:
            self._owned(event_id, attempt)
            self._db.execute(
                "UPDATE events SET status = ?, result = ?, last_error = NULL WHERE event_id = ?",
                (status, json.dumps(result, sort_keys=True), event_id),
            )
            self._audit(event_id, now, status, json.dumps(result, sort_keys=True))
            self._db.commit()

    def fail(self, event_id: str, attempt: int, error: str, now: float, retry_in: float | None, max_attempts: int) -> str:
        """Record a failed attempt. Returns 'pending' if it will be retried, 'dead' if it will not."""
        with self._lock:
            self._owned(event_id, attempt)
            if retry_in is None or attempt >= max_attempts:
                status, not_before = "dead", 0.0
            else:
                status, not_before = "pending", now + retry_in
            self._db.execute(
                "UPDATE events SET status = ?, not_before = ?, last_error = ? WHERE event_id = ?",
                (status, not_before, error, event_id),
            )
            self._audit(event_id, now, "retry" if status == "pending" else "dead", error)
            self._db.commit()
            return status

    def get(self, event_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT status, attempts, result, last_error FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if row is None:
                return None
            trail = self._db.execute(
                "SELECT at, kind, detail FROM audit WHERE event_id = ? ORDER BY id", (event_id,)
            ).fetchall()
        return {
            "event_id": event_id,
            "status": row[0],
            "attempts": row[1],
            "result": json.loads(row[2]) if row[2] else None,
            "last_error": row[3],
            "audit": [{"at": at, "kind": kind, "detail": detail} for at, kind, detail in trail],
        }

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def count_by_status(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) FROM events GROUP BY status").fetchall()
        return {status: n for status, n in rows}
