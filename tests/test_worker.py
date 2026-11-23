import threading

import pytest
from fastapi.testclient import TestClient

from flowguard.api import create_app
from flowguard.evals import Decision
from flowguard.store import EventStore, LeaseLost
from flowguard.worker import TransientError, Worker


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class Scripted:
    """Answers from a list, one per call; an Exception in the list is raised instead."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = 0

    def decide(self, text):
        self.calls += 1
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def kinds(store, event_id):
    return [entry["kind"] for entry in store.get(event_id)["audit"]]


@pytest.fixture
def store():
    return EventStore()


@pytest.fixture
def clock():
    return Clock()


def submit(store, key="k", text="where is my order"):
    return store.accept(key, {"text": text}).event_id


def test_empty_queue_returns_none(store, clock):
    assert Worker(store, Scripted(), clock=clock).run_once() is None


def test_event_is_processed_once_and_the_audit_trail_shows_it(store, clock):
    event_id = submit(store)
    worker = Worker(store, Scripted(Decision("shipping", False)), clock=clock)
    assert worker.run_once() == "done"
    assert worker.run_once() is None
    event = store.get(event_id)
    assert event["status"] == "done" and event["attempts"] == 1
    assert event["result"] == {"category": "shipping", "needs_human": False}
    assert kinds(store, event_id) == ["accepted", "claimed", "done"]


def test_decision_that_needs_a_person_is_parked_not_done(store, clock):
    event_id = submit(store, text="my lawyer will call you")
    Worker(store, Scripted(Decision("other", True)), clock=clock).run_once()
    assert store.get(event_id)["status"] == "needs_human"


def test_transient_error_is_retried_after_backoff_then_succeeds(store, clock):
    event_id = submit(store)
    system = Scripted(TransientError("rate limited"), Decision("shipping", False))
    worker = Worker(store, system, backoff_seconds=10, clock=clock)

    assert worker.run_once() == "retry"
    assert worker.run_once() is None  # still inside the backoff window
    clock.now += 9
    assert worker.run_once() is None
    clock.now += 1
    assert worker.run_once() == "done"
    event = store.get(event_id)
    assert event["attempts"] == 2 and event["last_error"] is None
    assert kinds(store, event_id) == ["accepted", "claimed", "retry", "claimed", "done"]


def test_backoff_doubles_each_attempt(store, clock):
    submit(store)
    system = Scripted(TransientError("a"), TransientError("b"), Decision("other", False))
    worker = Worker(store, system, backoff_seconds=5, max_attempts=3, clock=clock)
    worker.run_once()
    clock.now += 5
    worker.run_once()
    clock.now += 9
    assert worker.run_once() is None  # second wait is 10 seconds
    clock.now += 1
    assert worker.run_once() == "done"


def test_event_goes_dead_after_max_attempts_and_keeps_the_error(store, clock):
    event_id = submit(store)
    system = Scripted(*[TransientError("timeout")] * 3)
    worker = Worker(store, system, max_attempts=3, backoff_seconds=1, clock=clock)
    outcomes = []
    for _ in range(3):
        outcomes.append(worker.run_once())
        clock.now += 100
    assert outcomes == ["retry", "retry", "dead"]
    event = store.get(event_id)
    assert event["status"] == "dead" and event["attempts"] == 3 and event["last_error"] == "timeout"
    assert worker.run_once() is None


def test_payload_without_text_is_dead_at_once_and_not_retried(store, clock):
    event_id = store.accept("k", {"kind": "ticket"}).event_id
    system = Scripted()
    assert Worker(store, system, clock=clock).run_once() == "dead"
    assert system.calls == 0
    assert store.get(event_id)["attempts"] == 1


def test_a_crashing_system_does_not_lose_the_event(store, clock):
    event_id = submit(store)
    worker = Worker(store, Scripted(RuntimeError("bug"), RuntimeError("bug"), RuntimeError("bug")),
                    lease_seconds=30, max_attempts=3, clock=clock)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            worker.run_once()
        assert store.get(event_id)["status"] == "processing"
        assert worker.run_once() is None  # lease has not expired, nobody else may take it
        clock.now += 31
    assert worker.run_once() is None  # reaped on the final attempt
    assert store.get(event_id)["status"] == "dead"
    assert "lease expired" in store.get(event_id)["last_error"]


def test_a_slow_worker_cannot_overwrite_the_worker_that_took_over(store, clock):
    event_id = submit(store)
    first = store.claim(clock(), 30, 3)
    clock.now += 31
    second = store.claim(clock(), 30, 3)
    assert second.event_id == first.event_id and second.attempt == 2
    with pytest.raises(LeaseLost):
        store.complete(event_id, first.attempt, "done", {"category": "other"}, clock())
    store.complete(event_id, second.attempt, "done", {"category": "shipping"}, clock())
    assert store.get(event_id)["result"] == {"category": "shipping"}


def test_complete_and_fail_reject_an_event_not_being_processed(store, clock):
    event_id = submit(store)
    with pytest.raises(LeaseLost):
        store.complete(event_id, 1, "done", {}, clock())
    with pytest.raises(LeaseLost):
        store.fail(event_id, 1, "x", clock(), 1.0, 3)


def test_parallel_workers_decide_each_event_exactly_once(store):
    ids = [submit(store, key=f"k{i}", text=f"ticket {i}") for i in range(40)]
    seen, lock = [], threading.Lock()

    class Counting:
        def decide(self, text):
            with lock:
                seen.append(text)
            return Decision("other", False)

    def run():
        Worker(store, Counting()).drain()

    threads = [threading.Thread(target=run) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(seen) == sorted(f"ticket {i}" for i in range(40))
    assert store.count_by_status() == {"done": 40}
    assert all(store.get(i)["attempts"] == 1 for i in ids)


def test_a_retried_post_is_processed_once_and_visible_over_http(store):
    client = TestClient(create_app(store))
    body = {"text": "I was charged twice"}
    first = client.post("/events", json=body, headers={"Idempotency-Key": "k"}).json()
    client.post("/events", json=body, headers={"Idempotency-Key": "k"})
    system = Scripted(Decision("billing", False))
    worker = Worker(store, system)
    assert worker.drain() == 1 and system.calls == 1
    shown = client.get(f"/events/{first['event_id']}").json()
    assert shown["status"] == "done" and shown["result"]["category"] == "billing"
    assert [e["kind"] for e in shown["audit"]] == ["accepted", "claimed", "done"]
    assert client.get("/events/nope").status_code == 404


@pytest.mark.parametrize("kwargs", [{"max_attempts": 0}, {"lease_seconds": 0}, {"backoff_seconds": -1}])
def test_worker_rejects_bad_settings(store, kwargs):
    with pytest.raises(ValueError):
        Worker(store, Scripted(), **kwargs)
