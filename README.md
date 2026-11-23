# flowguard

A backend for AI workflows that must be safe to retry and easy to audit. Today it has three pieces: an event endpoint that accepts each event once however often the client retries, a worker that processes each event once with retries and a dead-letter state, and an eval harness with a plain keyword baseline to beat.

## What works now

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
- A worker that finishes after its lease was taken over gets `LeaseLost` and cannot overwrite the new owner's result.
- A payload with no `text` goes straight to `dead`; retrying cannot fix it.
- Every transition is appended to an audit log.

Tests cover each rule, including six workers draining 40 events with each decided exactly once.

## Model

`LLMSystem` asks a model for `{category, needs_human}`. The ticket is wrapped in tags and the prompt says to treat it as data, but a prompt is not a boundary; the injection cases measure how often the model obeys text inside a ticket. Answers are checked strictly (known category, a real boolean). An unusable answer raises a retryable error, so the worker retries it and finally dead-letters it. `OllamaProvider` talks to a local Ollama server with temperature 0 and a fixed seed, and asks for output constrained to a JSON schema. Timeouts, connection errors and 5xx replies are retryable; a 4xx is not. The provider is an interface, so a hosted model can replace it without touching the worker.

The prompt restates the labelling rules in general terms and contains no eval ticket (a test checks that). It was written before any model was run and has not been changed since.

## Evals

`evals/cases.jsonl` holds 40 hand-written support tickets: typos, two issues in one message, account security, a card number typed into the ticket, legal threats, prompt injection, abuse, repeats, a long pasted thread, mixed and non-English text, vague text, a customer asking for a person. Each has a `category` and a `needs_human` label. The labelling rules are in the docstring of `src/flowguard/evals.py`; a refund above 5,000 needs a person. Four cases are tagged `disputed` because a second reader could label the category differently; they are left out of the headline accuracy and reported on their own line. I wrote the cases and labels myself, so they carry my judgement and nobody else has reviewed them.

```bash
PYTHONPATH=src .venv/bin/python -m flowguard.evals --repeats 3
PYTHONPATH=src .venv/bin/python -m flowguard.evals --system ollama --model qwen2.5:7b
```

It prints category accuracy, escalation recall and precision, run-to-run stability, the ids of missed escalations and wrong categories, and accuracy per tag. It exits 1 if any metric is below `evals/thresholds.json`.

The first system scored is a keyword baseline with no model (`src/flowguard/triage.py`), written plainly and not tuned to the cases. Measured on 2026-11-23, one run:

| metric | baseline |
|---|---|
| category accuracy (36 undisputed cases) | 0.694 |
| escalation recall | 0.250 |
| escalation precision | 1.000 |

It misses 9 of the 12 cases that need a person: every prompt-injection case, the high-value refunds, the data-deletion request, the card-number case and the customer who asks for a person. It catches only the legal and stolen-card cases. The thresholds are set at the baseline's values, so they only stop a system getting worse than the baseline. 

Local models through Ollama, measured 2026-11-23 on an Apple M4 with Ollama 0.40.6 (temperature 0, seed 0), same 40 cases:

| system | category accuracy | escalation recall | escalation precision |
|---|---|---|---|
| keyword baseline | 0.694 | 0.250 | 1.000 |
| qwen2.5:7b (3 runs, identical) | 0.889 | 0.833 | 0.909 |
| llama3.1:8b (1 run) | 0.806 | 0.833 | 0.714 |
| qwen2.5:3b (1 run) | 0.444 | 0.667 | 0.500 |

`qwen2.5:7b` misses two escalations: the 18,000 refund (c06) and the injection hidden in a late-delivery ticket (c19). It gets four categories wrong: c16, c23, c34, c40. These numbers say how each model followed my rules on my 40 cases, once. They are not a general ranking: the cases and the prompt come from the same author, 40 cases is small, a single wrong case moves a score by about 3 points, and identical runs reflect temperature 0 with a fixed seed, not stability under normal sampling. No prompt change has been made in response to these failures; that needs new cases the prompt has not seen.

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
- No retrieval, tool use or approval step yet, and no command that starts a worker loop: the worker is a class used from code and tests.
- Only a local Ollama provider exists; no hosted model has been run.
- `needs_human` events are parked; nothing lets a person resolve them yet.
- The database schema changed with the worker; delete any `.db` file made by the first commit.
- 40 cases is small: one wrong case moves accuracy by about 3 points. It is a harness to build on, not a benchmark.
- The baseline is deterministic, so stability is 1.0 by construction; that metric only becomes informative with a model.
- One shared SQLite connection behind a lock is fine for one process, not for several.
- Tests run on Python 3.15 only; the declared minimum, 3.12, has not been tested.
