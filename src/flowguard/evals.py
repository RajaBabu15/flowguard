"""Eval harness: load labelled cases, run a system over them, score, and gate on thresholds.

Labels follow two rules written by hand. `category` is the issue named first in the ticket.
`needs_human` is true when acting without a person would be risky:
  - account security or privacy requests, or personal data such as a card number typed into the ticket;
  - legal threats or physical safety;
  - a refund above 5,000;
  - text that tries to instruct the system;
  - the customer asks for a person.
Having two issues is not enough on its own; one of them has to be risky by the list above.
Escalation recall is the number that matters most, because a missed escalation is the costly error.

Cases tagged `disputed` have a category a second reader could reasonably label differently. They are
left out of the headline category accuracy and reported on their own, so one person's taste cannot
move the score. They still count for escalation.
"""
import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from flowguard.errors import TransientError

CATEGORIES = ("billing", "refund", "shipping", "technical", "account", "other")
GATED = ("category_accuracy", "escalation_recall", "escalation_precision", "stability")


class CaseError(ValueError):
    pass


@dataclass(frozen=True)
class Case:
    id: str
    text: str
    category: str
    needs_human: bool
    tags: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    category: str
    needs_human: bool
    reasons: tuple[str, ...] = ()


class System(Protocol):
    def decide(self, text: str) -> Decision: ...


@dataclass
class Report:
    cases: int
    repeats: int
    category_accuracy: float
    escalation_recall: float
    escalation_precision: float
    stability: float
    disputed_accuracy: float | None = None
    missed_escalations: list[str] = field(default_factory=list)
    false_escalations: list[str] = field(default_factory=list)
    wrong_categories: list[str] = field(default_factory=list)
    wrong_disputed: list[str] = field(default_factory=list)
    failed_calls: list[str] = field(default_factory=list)
    by_tag: dict[str, tuple[int, float]] = field(default_factory=dict)


def load_cases(path: Path) -> list[Case]:
    cases, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CaseError(f"{path}:{n}: not valid JSON ({exc.msg})") from exc
        for key in ("id", "text", "category", "needs_human", "tags"):
            if key not in raw:
                raise CaseError(f"{path}:{n}: missing '{key}'")
        if raw["id"] in seen:
            raise CaseError(f"{path}:{n}: duplicate id {raw['id']}")
        seen.add(raw["id"])
        if raw["category"] not in CATEGORIES:
            raise CaseError(f"{path}:{n}: category must be one of {CATEGORIES}")
        if not isinstance(raw["needs_human"], bool):
            raise CaseError(f"{path}:{n}: needs_human must be true or false")
        if not raw["text"].strip():
            raise CaseError(f"{path}:{n}: empty text")
        cases.append(Case(raw["id"], raw["text"], raw["category"], raw["needs_human"], tuple(raw["tags"])))
    if not cases:
        raise CaseError(f"{path}: no cases")
    return cases


def evaluate(system: System, cases: list[Case], repeats: int = 1) -> Report:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    failed: set[str] = set()

    def decide(case: Case) -> Decision | None:
        # A call that fails with a retryable error counts as a wrong answer here: in production it would be
        # retried, but a system that keeps failing on a ticket has not handled it.
        try:
            return system.decide(case.text)
        except TransientError:
            failed.add(case.id)
            return None

    runs = [[decide(c) for c in cases] for _ in range(repeats)]

    correct = counted = d_correct = d_counted = tp = fp = fn = 0
    wrong_categories, wrong_disputed, missed, false_alarms = set(), set(), set(), set()
    tag_hits: dict[str, list[int]] = defaultdict(list)
    for run in runs:
        for case, got in zip(cases, run):
            ok = got is not None and got.category == case.category
            if "disputed" in case.tags:
                d_counted += 1
                d_correct += ok
                if not ok:
                    wrong_disputed.add(case.id)
            else:
                counted += 1
                correct += ok
                if not ok:
                    wrong_categories.add(case.id)
            for tag in case.tags:
                tag_hits[tag].append(int(ok))
            flagged = got is not None and got.needs_human
            if case.needs_human and flagged:
                tp += 1
            elif case.needs_human:
                fn += 1
                missed.add(case.id)
            elif flagged:
                fp += 1
                false_alarms.add(case.id)

    if counted == 0:
        raise ValueError("every case is disputed; the headline accuracy would be empty")
    # With no predicted escalations precision is undefined; 0.0 keeps the gate conservative.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    stable = sum(1 for i in range(len(cases)) if len({runs[r][i] for r in range(repeats)}) == 1)
    return Report(
        cases=len(cases),
        repeats=repeats,
        category_accuracy=correct / counted,
        disputed_accuracy=d_correct / d_counted if d_counted else None,
        escalation_recall=recall,
        escalation_precision=precision,
        stability=stable / len(cases),
        missed_escalations=sorted(missed),
        false_escalations=sorted(false_alarms),
        wrong_categories=sorted(wrong_categories),
        wrong_disputed=sorted(wrong_disputed),
        failed_calls=sorted(failed),
        by_tag={t: (len(h), sum(h) / len(h)) for t, h in sorted(tag_hits.items())},
    )


def check(report: Report, thresholds: dict[str, float]) -> list[str]:
    """Return one message per metric below its minimum. Unknown threshold names are errors, not ignored."""
    unknown = set(thresholds) - set(GATED)
    if unknown:
        raise CaseError(f"unknown threshold names: {sorted(unknown)}")
    return [
        f"{name} {getattr(report, name):.3f} is below the minimum {minimum:.3f}"
        for name, minimum in thresholds.items()
        if getattr(report, name) < minimum
    ]


def format_report(report: Report) -> str:
    lines = [
        f"cases {report.cases}, repeats {report.repeats}",
        f"category accuracy    {report.category_accuracy:.3f}",
        f"escalation recall    {report.escalation_recall:.3f}",
        f"escalation precision {report.escalation_precision:.3f}",
        f"stability            {report.stability:.3f}",
        f"missed escalations   {', '.join(report.missed_escalations) or 'none'}",
        f"false escalations   {', '.join(report.false_escalations) or 'none'}",
        f"failed calls         {', '.join(report.failed_calls) or 'none'}",
        f"wrong categories     {', '.join(report.wrong_categories) or 'none'}",
        "disputed accuracy    "
        + ("n/a" if report.disputed_accuracy is None else f"{report.disputed_accuracy:.3f} (wrong: {', '.join(report.wrong_disputed) or 'none'})"),
        "category accuracy by tag:",
    ]
    lines += [f"  {tag:<16} n={n:<3} {acc:.3f}" for tag, (n, acc) in report.by_tag.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from flowguard.llm import LLMSystem, OllamaProvider
    from flowguard.triage import KeywordBaseline

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the eval cases against a system and gate on the thresholds.")
    parser.add_argument("--system", choices=("keyword", "ollama"), default="keyword")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name, used with --system ollama")
    parser.add_argument("--cases", type=Path, default=root / "evals" / "cases.jsonl")
    parser.add_argument("--thresholds", type=Path, default=root / "evals" / "thresholds.json")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)

    system = KeywordBaseline() if args.system == "keyword" else LLMSystem(OllamaProvider(args.model))
    report = evaluate(system, load_cases(args.cases), args.repeats)
    print(f"system {args.system}" + (f" {args.model}" if args.system == "ollama" else ""))
    print(format_report(report))
    failures = check(report, json.loads(args.thresholds.read_text()))
    for message in failures:
        print(f"FAIL {message}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
