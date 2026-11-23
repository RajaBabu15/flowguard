# flowguard

A backend for AI workflows that must be safe to retry and easy to audit. Today it contains one piece: an event endpoint that accepts each event once, however many times the client retries.

## What works now

`POST /events` needs an `Idempotency-Key` header.

- Same key and same body: returns the first `event_id` with `replayed: true`, stores nothing new.
- Same key and a different body: `409`, because that is a client bug and silently accepting it would hide it.
- Missing or empty key: `422`.
- Key order inside the JSON body does not change identity.

Events are stored in SQLite with the key as primary key, behind a lock, so concurrent retries create one row (tested with 16 threads).

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/uvicorn --factory flowguard.api:create_app
```

`uvicorn` is not a declared dependency yet; install it to serve the app. Tests use the in-process client and need no server.

## Limits

- Storage is in memory by default; pass a file path to `EventStore` to keep events.
- Nothing processes the stored events yet. There is no model call, retrieval, tool use, approval step or eval in this repository so far.
- One shared SQLite connection behind a lock is fine for one process, not for several.
- Tests run on Python 3.15 only; the declared minimum, 3.12, has not been tested.
