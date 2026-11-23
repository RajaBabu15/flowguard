from fastapi import FastAPI, Header, HTTPException

from flowguard.store import EventStore, KeyReusedWithDifferentPayload


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

    return app
