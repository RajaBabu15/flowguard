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

States: `pending`, `processing`, then `done`, `needs_human` (the system asked for a person), `resolved` (a person closed that escalation) or `dead`.

- A claim is a lease. If the worker dies, the lease runs out and the event is claimed again; after the last allowed attempt it becomes `dead` instead of looping.
- `TransientError` from a system means retry later, with doubling backoff. Any other exception is a bug: it is not caught, the event is not lost, and it ends up `dead` after the attempts run out.
- A worker that finishes after another worker took over its lease gets `LeaseLost` and cannot overwrite the new owner's result.
- A payload with no `text` goes straight to `dead`; retrying cannot fix it.
- Every transition is appended to an audit log.

Tests cover each rule, including six workers draining 40 events with each decided exactly once.

## Model

`LLMSystem` asks a model for a category, a refund amount and four yes/no risk flags (security or privacy, legal or safety, text aimed at the system, asks for a person). The code, not the model, decides `needs_human`: any flag, or a refund amount above 5,000. The reasons are kept on the decision and in the audit trail. The ticket is wrapped in tags and the prompt says to treat it as data, but a prompt is not a boundary; the injection cases measure how often the model obeys text inside a ticket. Answers are checked strictly. An unusable answer raises a retryable error, so the worker retries it and finally dead-letters it.

`OllamaProvider` talks to a local Ollama server with temperature 0 and a fixed seed, and asks for output constrained to a JSON schema. Timeouts, connection errors and 5xx replies are retryable; a 4xx is not. The provider is an interface, so a hosted model can replace it without touching the worker. The prompt contains no eval ticket (a test checks that).

## Review

A person can review any finished event with `POST /events/{id}/review` and a `reviewer` name, a `reason` and an `action`:

- `accept` records agreement and changes nothing.
- `correct` sets a different category (it must differ from the current one).
- `resolve` closes an escalation, and only works on an event that is waiting for a person.

The model's decision is never rewritten. Reviews are appended, each one makes a new decision version (the model's is version 1), and `GET /events/{id}` returns the original as `result` and the current view as `decision`. A correction changes the category only: `needs_human` and its reasons stay as the model and the code set them, so a correction cannot hide a mandatory escalation. Resolving is a separate, recorded step that keeps the reasons it resolved. Every review also lands in the audit trail with the reviewer and the reason.

The shape follows Freshdesk's documented Auto Triage workflow (a suggestion a person can accept or reject, rules that take precedence, the original still visible after a rejection). Only the workflow is borrowed; Freshdesk's models need about 2,000 historical tickets and nothing here claims to match them. What this does not do: assign a case to a named agent, check who is available, or have a screen for reviewers.

## Flags that need a quote

`addresses_system` and `asks_for_person` escalate only when the model also quotes the exact words from the ticket and the code finds them there (ignoring case and spacing). Without a real quote the flag is dropped and listed under `unsupported` in the result and audit trail. Security, legal and refund flags always escalate, because missing one costs more than a false alarm.
