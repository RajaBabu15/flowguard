"""Takes events off the queue, asks a system to decide, and records the outcome.

Only TransientError is retried. Any other exception is a bug in the system and is allowed to
propagate: the event stays claimed, its lease runs out, and it is retried and finally marked dead,
so a crashing system can never lose an event or run it forever.
"""
import time
from typing import Callable

from flowguard.evals import System
from flowguard.store import EventStore


class TransientError(Exception):
    """Raised by a system for a failure worth retrying, such as a timeout or a rate limit."""


class Worker:
    def __init__(
        self,
        store: EventStore,
        system: System,
        *,
        lease_seconds: float = 30.0,
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
        clock: Callable[[], float] = time.time,
    ):
        if max_attempts < 1 or lease_seconds <= 0 or backoff_seconds < 0:
            raise ValueError("max_attempts must be >= 1, lease_seconds > 0, backoff_seconds >= 0")
        self.store = store
        self.system = system
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.clock = clock

    def run_once(self) -> str | None:
        """Process one ready event. Returns done, needs_human, retry or dead, or None if nothing was ready."""
        now = self.clock()
        claim = self.store.claim(now, self.lease_seconds, self.max_attempts)
        if claim is None:
            return None

        text = claim.payload.get("text")
        if not isinstance(text, str) or not text.strip():
            self.store.fail(claim.event_id, claim.attempt, "payload has no text", self.clock(), None, self.max_attempts)
            return "dead"

        try:
            decision = self.system.decide(text)
        except TransientError as exc:
            retry_in = self.backoff_seconds * 2 ** (claim.attempt - 1)
            status = self.store.fail(claim.event_id, claim.attempt, str(exc) or "transient error", self.clock(), retry_in, self.max_attempts)
            return "retry" if status == "pending" else "dead"

        status = "needs_human" if decision.needs_human else "done"
        self.store.complete(
            claim.event_id,
            claim.attempt,
            status,
            {"category": decision.category, "needs_human": decision.needs_human},
            self.clock(),
        )
        return status

    def drain(self, limit: int = 1000) -> int:
        """Process ready events until none are left. Returns how many were handled."""
        handled = 0
        while handled < limit and self.run_once() is not None:
            handled += 1
        return handled
