"""Eval harness: load labelled cases, run a system over them, score, and gate on thresholds.

Labels follow two rules written by hand. `category` is the issue named first in the ticket.
`needs_human` is true when acting without a person would be risky: security or privacy,
legal or safety, a high-value refund, more than one issue, or text trying to instruct the system.
Escalation recall is the number that matters most, because a missed escalation is the costly error.
"""
import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

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
    missed_escalations: list[str] = field(default_factory=list)
    wrong_categories: list[str] = field(default_factory=list)
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
    runs = [[system.decide(c.text) for c in cases] for _ in range(repeats)]

    correct = tp = fp = fn = 0
    wrong_categories, missed = set(), set()
    tag_hits: dict[str, list[int]] = defaultdict(list)
    for run in runs:
        for case, got in zip(cases, run):
            ok = got.category == case.category
            correct += ok
            if not ok:
                wrong_categories.add(case.id)
            for tag in case.tags:
                tag_hits[tag].append(int(ok))
            if case.needs_human and got.needs_human:
                tp += 1
            elif case.needs_human and not got.needs_human:
                fn += 1
                missed.add(case.id)
            elif got.needs_human:
                fp += 1

    total = len(cases) * repeats
    # With no predicted escalations precision is undefined; 0.0 keeps the gate conservative.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    stable = sum(1 for i in range(len(cases)) if len({runs[r][i] for r in range(repeats)}) == 1)
    return Report(
        cases=len(cases),
        repeats=repeats,
        category_accuracy=correct / total,
        escalation_recall=recall,
        escalation_precision=precision,
        stability=stable / len(cases),
        missed_escalations=sorted(missed),
        wrong_categories=sorted(wrong_categories),
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
        f"wrong categories     {', '.join(report.wrong_categories) or 'none'}",
        "category accuracy by tag:",
    ]
    lines += [f"  {tag:<16} n={n:<3} {acc:.3f}" for tag, (n, acc) in report.by_tag.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from flowguard.triage import KeywordBaseline

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the eval cases against the keyword baseline.")
    parser.add_argument("--cases", type=Path, default=root / "evals" / "cases.jsonl")
    parser.add_argument("--thresholds", type=Path, default=root / "evals" / "thresholds.json")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)

    report = evaluate(KeywordBaseline(), load_cases(args.cases), args.repeats)
    print(format_report(report))
    failures = check(report, json.loads(args.thresholds.read_text()))
    for message in failures:
        print(f"FAIL {message}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
