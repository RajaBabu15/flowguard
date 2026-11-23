# flowguard

A backend for AI workflows that have to be safe to retry and easy to audit. It takes support tickets in over HTTP, processes each one once, lets a model sort it, and sends anything risky to a person.

What exists:

- `POST /events` with an `Idempotency-Key`: a retried request is stored once.
- A worker with leases, retries with backoff, a dead-letter state and an audit log.
- A model-backed system (local Ollama) whose escalation rule is applied in code, not by the model.
- A review step over HTTP: a person can accept, correct or resolve a decision; the model's original is kept.
- An eval harness with 109 hand-written tickets, a keyword baseline to beat, and a gate that fails when scores drop.

Design is in [docs/design.md](docs/design.md). Cases, results and the error analysis are in [docs/evals.md](docs/evals.md).

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
PYTHONPATH=src .venv/bin/python -m flowguard.evals
PYTHONPATH=src .venv/bin/python -m flowguard.evals --system ollama --model qwen2.5:7b
```

`uvicorn` is not a declared dependency; install it to serve the API with `uvicorn --factory flowguard.api:create_app`.

## Where it stops

- No retrieval or tool use. Reviews exist over HTTP, but there is no screen for reviewers and no assignment of a case to a named agent.
- No command starts a worker loop; the worker is a class used from code and tests.
- Only a local Ollama provider has been run.
- The cases are small (109 across four files) and labelled by one person.
- SQLite behind one lock suits one process.
- Tests ran on Python 3.15 only; the declared minimum, 3.12, is untested.
