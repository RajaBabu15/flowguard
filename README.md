# flowguard

A backend for AI workflows that must be safe to retry and easy to audit. Today it has two pieces: an event endpoint that accepts each event once, however many times the client retries, and an eval harness with a plain keyword baseline to beat.

## What works now

`POST /events` needs an `Idempotency-Key` header.

- Same key and same body: returns the first `event_id` with `replayed: true`, stores nothing new.
- Same key and a different body: `409`, because that is a client bug and silently accepting it would hide it.
- Missing or empty key: `422`.
- Key order inside the JSON body does not change identity.

Events are stored in SQLite with the key as primary key, behind a lock, so concurrent retries create one row (tested with 16 threads).

## Evals

`evals/cases.jsonl` holds 32 hand-written support tickets: typos, two issues in one message, account security, legal threats, prompt injection, non-English text, vague text. Each has a `category` and a `needs_human` label. The labelling rules are in the docstring of `src/flowguard/evals.py`. I wrote the cases and labels myself, so they carry my judgement and nobody else has reviewed them.

```bash
PYTHONPATH=src .venv/bin/python -m flowguard.evals --repeats 3
```

It prints category accuracy, escalation recall and precision, run-to-run stability, the ids of missed escalations and wrong categories, and accuracy per tag. It exits 1 if any metric is below `evals/thresholds.json`.

The first system scored is a keyword baseline with no model (`src/flowguard/triage.py`), written plainly and not tuned to the cases. Measured on 2026-11-23, one run:

| metric | baseline |
|---|---|
| category accuracy | 0.688 |
| escalation recall | 0.250 |
| escalation precision | 1.000 |

It misses 9 of 12 cases that need a person, including every prompt-injection case but none of the legal ones. The thresholds are set at these values, so they only stop it getting worse. Any model-backed system added later is expected to beat them and to raise the floors.

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
- Nothing processes the stored events yet. There is no model call, retrieval, tool use or approval step in this repository so far.
- 32 cases is small: one wrong case moves accuracy by 3 points. It is a harness to build on, not a benchmark.
- The baseline is deterministic, so stability is 1.0 by construction; that metric only becomes informative with a model.
- One shared SQLite connection behind a lock is fine for one process, not for several.
- Tests run on Python 3.15 only; the declared minimum, 3.12, has not been tested.
