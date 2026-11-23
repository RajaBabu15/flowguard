# Design

## Event endpoint

`POST /events` needs an `Idempotency-Key` header.

- Same key and same body: returns the first `event_id` with `replayed: true`, stores nothing new.
- Same key and a different body: `409`, because that is a client bug and silently accepting it would hide it.
- Missing or empty key: `422`.
- Key order inside the JSON body does not change identity.

Events are stored in SQLite with the key as primary key, behind a lock, so concurrent retries create one row (tested with 16 threads).

## Worker

`Worker(store, system).run_once()` claims the oldest ready event, asks the system for a `Decision`, and records it. `GET /events/{id}` shows the status, attempts, result and the audit trail.

States: `pending`, `processing`, then `done`, `needs_human` (the system asked for a person) or `dead`.

- A claim is a lease. If the worker dies, the lease runs out and the event is claimed again; after the last allowed attempt it becomes `dead` instead of looping.
- `TransientError` from a system means retry later, with doubling backoff. Any other exception is a bug: it is not caught, the event is not lost, and it ends up `dead` after the attempts run out.
- A worker that finishes after another worker took over its lease gets `LeaseLost` and cannot overwrite the new owner's result.
- A payload with no `text` goes straight to `dead`; retrying cannot fix it.
- Every transition is appended to an audit log.

Tests cover each rule, including six workers draining 40 events with each decided exactly once.

## Model

`LLMSystem` asks a model for a category, a refund amount and four yes/no risk flags (security or privacy, legal or safety, text aimed at the system, asks for a person). The code, not the model, decides `needs_human`: any flag, or a refund amount above 5,000. The reasons are kept on the decision and in the audit trail. The ticket is wrapped in tags and the prompt says to treat it as data, but a prompt is not a boundary; the injection cases measure how often the model obeys text inside a ticket. Answers are checked strictly. An unusable answer raises a retryable error, so the worker retries it and finally dead-letters it.

`OllamaProvider` talks to a local Ollama server with temperature 0 and a fixed seed, and asks for output constrained to a JSON schema. Timeouts, connection errors and 5xx replies are retryable; a 4xx is not. The provider is an interface, so a hosted model can replace it without touching the worker. The prompt contains no eval ticket (a test checks that).
