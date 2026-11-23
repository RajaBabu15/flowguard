import threading

import pytest
from fastapi.testclient import TestClient

from flowguard.api import create_app
from flowguard.store import EventStore, KeyReusedWithDifferentPayload


@pytest.fixture
def store():
    return EventStore()


@pytest.fixture
def client(store):
    return TestClient(create_app(store))


def test_retry_returns_first_event_and_stores_once(client, store):
    body = {"kind": "ticket", "text": "order arrived broken"}
    first = client.post("/events", json=body, headers={"Idempotency-Key": "k1"})
    retry = client.post("/events", json=body, headers={"Idempotency-Key": "k1"})
    assert first.status_code == retry.status_code == 202
    assert first.json()["replayed"] is False
    assert retry.json() == {"event_id": first.json()["event_id"], "replayed": True}
    assert store.count() == 1


def test_key_order_in_body_does_not_change_identity(client, store):
    client.post("/events", json={"a": 1, "b": 2}, headers={"Idempotency-Key": "k"})
    retry = client.post("/events", json={"b": 2, "a": 1}, headers={"Idempotency-Key": "k"})
    assert retry.json()["replayed"] is True
    assert store.count() == 1


def test_same_key_different_body_is_a_conflict(client, store):
    client.post("/events", json={"text": "one"}, headers={"Idempotency-Key": "k"})
    clash = client.post("/events", json={"text": "two"}, headers={"Idempotency-Key": "k"})
    assert clash.status_code == 409
    assert store.count() == 1


def test_missing_or_empty_key_is_rejected(client, store):
    assert client.post("/events", json={"x": 1}).status_code == 422
    assert client.post("/events", json={"x": 1}, headers={"Idempotency-Key": ""}).status_code == 422
    assert store.count() == 0


def test_concurrent_retries_create_one_event(store):
    results = []

    def send():
        results.append(store.accept("same", {"text": "hello"}))

    threads = [threading.Thread(target=send) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({r.event_id for r in results}) == 1
    assert sum(1 for r in results if not r.replayed) == 1
    assert store.count() == 1


def test_store_raises_on_reused_key(store):
    store.accept("k", {"a": 1})
    with pytest.raises(KeyReusedWithDifferentPayload):
        store.accept("k", {"a": 2})
