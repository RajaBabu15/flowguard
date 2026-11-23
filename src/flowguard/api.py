import time

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from flowguard.store import EventStore, InvalidReview, KeyReusedWithDifferentPayload, NotReviewable, UnknownEvent


class ReviewBody(BaseModel):
    reviewer: str
    action: str
    reason: str
    category: str | None = None


def create_app(store: EventStore | None = None) -> FastAPI:
    store = store or EventStore()
    app = FastAPI(title="flowguard")

    @app.post("/events", status_code=202)
    def post_event(payload: dict, idempotency_key: str = Header(min_length=1, max_length=200)):
        try:
            accepted = store.accept(idempotency_key, payload)
        except KeyReusedWithDifferentPayload:
            raise HTTPException(status_code=409, detail="idempotency key was already used with a different body")
        return {"event_id": accepted.event_id, "replayed": accepted.replayed}

    @app.get("/events/{event_id}")
    def get_event(event_id: str):
        event = store.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="no such event")
        return event

    @app.post("/events/{event_id}/review")
    def review_event(event_id: str, body: ReviewBody):
        try:
            return store.review(event_id, body.reviewer, body.action, body.reason, time.time(), body.category)
        except UnknownEvent:
            raise HTTPException(status_code=404, detail="no such event")
        except NotReviewable as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except InvalidReview as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    return app
