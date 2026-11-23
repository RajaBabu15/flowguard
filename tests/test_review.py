import threading

import pytest
from fastapi.testclient import TestClient

from flowguard.api import create_app
from flowguard.evals import Decision
from flowguard.store import EventStore, InvalidReview, NotReviewable, UnknownEvent
from flowguard.worker import Worker


class Fixed:
    def __init__(self, decision):
        self.decision = decision

    def decide(self, text):
        return self.decision


def processed(store, decision, key="k", text="I was charged twice, I want a supervisor"):
    event_id = store.accept(key, {"text": text}).event_id
    Worker(store, Fixed(decision)).run_once()
    return event_id


BILLING = Decision("billing", False, ())
ESCALATED = Decision("billing", True, ("asks_for_person",))


@pytest.fixture
def store():
    return EventStore()


def test_correction_changes_the_category_and_keeps_the_original(store):
    event_id = processed(store, BILLING)
    event = store.review(event_id, "asha", "correct", "this is a refund request", 5.0, category="refund")
    assert event["result"]["category"] == "billing"
    assert event["decision"]["category"] == "refund" and event["decision"]["version"] == 2
    review = event["reviews"][0]
    assert (review["category_before"], review["category_after"], review["reviewer"]) == ("billing", "refund", "asha")
    assert review["reason"] == "this is a refund request"


def test_a_correction_cannot_hide_a_mandatory_escalation(store):
    event_id = processed(store, ESCALATED)
    event = store.review(event_id, "asha", "correct", "wrong bucket", 5.0, category="account")
    assert event["status"] == "needs_human"
    assert event["decision"]["needs_human"] is True
    assert event["decision"]["reasons"] == ["asks_for_person"]
    assert event["decision"]["resolved"] is False


def test_resolving_closes_the_escalation_and_keeps_the_reasons(store):
    event_id = processed(store, ESCALATED)
    event = store.review(event_id, "ravi", "resolve", "called the customer, refunded the duplicate", 6.0)
    assert event["status"] == "resolved"
    assert event["decision"]["resolved"] is True
    assert event["decision"]["needs_human"] is True and event["decision"]["reasons"] == ["asks_for_person"]
    assert event["result"]["needs_human"] is True


def test_only_an_escalated_event_can_be_resolved(store):
    plain = processed(store, BILLING, key="a")
    with pytest.raises(NotReviewable):
        store.review(plain, "ravi", "resolve", "nothing to resolve", 1.0)
    escalated = processed(store, ESCALATED, key="b")
    store.review(escalated, "ravi", "resolve", "done", 1.0)
    with pytest.raises(NotReviewable):
        store.review(escalated, "ravi", "resolve", "again", 2.0)


def test_accept_records_agreement_without_changing_anything(store):
    event_id = processed(store, BILLING)
    event = store.review(event_id, "asha", "accept", "looks right", 2.0)
    assert event["decision"]["category"] == "billing" and event["decision"]["version"] == 2
    assert event["reviews"][0]["category_before"] == event["reviews"][0]["category_after"] == "billing"


def test_versions_count_up_and_later_corrections_start_from_the_latest(store):
    event_id = processed(store, BILLING)
    store.review(event_id, "asha", "correct", "refund", 1.0, category="refund")
    event = store.review(event_id, "ravi", "correct", "actually account closure", 2.0, category="account")
    assert event["decision"]["version"] == 3 and event["decision"]["category"] == "account"
    assert [(r["category_before"], r["category_after"]) for r in event["reviews"]] == [
        ("billing", "refund"),
        ("refund", "account"),
    ]


@pytest.mark.parametrize("status_setup", ["pending", "processing", "dead"])
def test_events_that_are_not_finished_cannot_be_reviewed(store, status_setup):
    event_id = store.accept("k", {"text": "x" if status_setup != "dead" else ""}).event_id
    if status_setup == "processing":
        store.claim(0.0, 30, 3)
    if status_setup == "dead":
        Worker(store, Fixed(BILLING)).run_once()
    with pytest.raises(NotReviewable):
        store.review(event_id, "asha", "accept", "ok", 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reviewer": " ", "action": "accept", "reason": "r"},
        {"reviewer": "a", "action": "accept", "reason": "  "},
        {"reviewer": "a", "action": "delete", "reason": "r"},
        {"reviewer": "a", "action": "correct", "reason": "r", "category": "banana"},
        {"reviewer": "a", "action": "correct", "reason": "r"},
        {"reviewer": "a", "action": "accept", "reason": "r", "category": "refund"},
        {"reviewer": "a", "action": "correct", "reason": "r", "category": "billing"},
    ],
)
def test_malformed_reviews_are_rejected_and_leave_no_trace(store, kwargs):
    event_id = processed(store, BILLING)
    with pytest.raises(InvalidReview):
        store.review(event_id, now=1.0, **kwargs)
    assert store.get(event_id)["reviews"] == []


def test_unknown_event_is_reported(store):
    with pytest.raises(UnknownEvent):
        store.review("nope", "a", "accept", "r", 1.0)


def test_audit_trail_names_the_reviewer_and_reason(store):
    event_id = processed(store, BILLING)
    store.review(event_id, "asha", "correct", "refund request", 3.0, category="refund")
    entry = store.get(event_id)["audit"][-1]
    assert entry["kind"] == "reviewed"
    assert '"reviewer": "asha"' in entry["detail"] and '"reason": "refund request"' in entry["detail"]


def test_two_reviewers_at_once_are_both_recorded_in_order(store):
    event_id = processed(store, BILLING)
    categories = ["refund", "account", "shipping", "technical"]
    errors = []

    def go(name, category):
        try:
            store.review(event_id, name, "correct", f"{name} says {category}", 1.0, category=category)
        except InvalidReview as exc:  # a reviewer whose pick matched the current category loses the race
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(f"r{i}", c)) for i, c in enumerate(categories)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    event = store.get(event_id)
    reviews = event["reviews"]
    assert [r["version"] for r in reviews] == list(range(2, 2 + len(reviews)))
    assert event["decision"]["version"] == 1 + len(reviews)
    for earlier, later in zip(reviews, reviews[1:]):
        assert earlier["category_after"] == later["category_before"]


def test_review_over_http_end_to_end(store):
    client = TestClient(create_app(store))
    posted = client.post("/events", json={"text": "charged twice, get me a supervisor"}, headers={"Idempotency-Key": "k"}).json()
    Worker(store, Fixed(ESCALATED)).run_once()
    body = {"reviewer": "asha", "action": "correct", "reason": "refund request", "category": "refund"}
    shown = client.post(f"/events/{posted['event_id']}/review", json=body).json()
    assert shown["result"]["category"] == "billing" and shown["decision"]["category"] == "refund"
    assert shown["decision"]["needs_human"] is True
    assert client.get(f"/events/{posted['event_id']}").json()["reviews"][0]["reviewer"] == "asha"
    assert client.post("/events/nope/review", json=body).status_code == 404
    assert client.post(f"/events/{posted['event_id']}/review", json={**body, "category": "refund"}).status_code == 422
    resolve = {"reviewer": "ravi", "action": "resolve", "reason": "sorted by phone"}
    assert client.post(f"/events/{posted['event_id']}/review", json=resolve).status_code == 200
    assert client.post(f"/events/{posted['event_id']}/review", json=resolve).status_code == 409
