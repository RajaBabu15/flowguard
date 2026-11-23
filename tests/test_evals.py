import itertools
import json
from pathlib import Path

import pytest

from flowguard.evals import CATEGORIES, Case, CaseError, Decision, check, evaluate, load_cases, main
from flowguard.triage import KeywordBaseline

REAL_CASES = Path(__file__).resolve().parents[1] / "evals" / "cases.jsonl"


def case(id, category="refund", needs_human=False, tags=("clean",), text="x"):
    return Case(id, text, category, needs_human, tuple(tags))


class Fixed:
    def __init__(self, category="refund", needs_human=False):
        self.decision = Decision(category, needs_human)

    def decide(self, text):
        return self.decision


class ByText:
    """Answers from a lookup so a test can state exactly which cases are right or wrong."""

    def __init__(self, table):
        self.table = table

    def decide(self, text):
        return self.table[text]


def write(tmp_path, *rows):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


GOOD = {"id": "a", "text": "hi", "category": "refund", "needs_human": False, "tags": []}


def test_perfect_system_scores_one():
    system = ByText({"x": Decision("refund", True)})
    report = evaluate(system, [case("a", "refund", True, text="x")])
    assert (report.category_accuracy, report.escalation_recall, report.escalation_precision) == (1.0, 1.0, 1.0)


def test_never_escalating_system_has_zero_recall_and_names_the_misses():
    cases = [case("a", needs_human=True), case("b", needs_human=True), case("c", needs_human=False)]
    report = evaluate(Fixed(needs_human=False), cases)
    assert report.escalation_recall == 0.0
    assert report.missed_escalations == ["a", "b"]


def test_escalating_everything_has_full_recall_but_low_precision():
    cases = [case("a", needs_human=True), case("b", needs_human=False), case("c", needs_human=False)]
    report = evaluate(Fixed(needs_human=True), cases)
    assert report.escalation_recall == 1.0
    assert report.escalation_precision == pytest.approx(1 / 3)


def test_category_accuracy_and_wrong_ids_and_tag_breakdown():
    cases = [
        case("a", "billing", tags=("typo",), text="a"),
        case("b", "refund", tags=("typo",), text="b"),
        case("c", "refund", tags=("clean",), text="c"),
    ]
    system = ByText({"a": Decision("billing", False), "b": Decision("other", False), "c": Decision("refund", False)})
    report = evaluate(system, cases)
    assert report.category_accuracy == pytest.approx(2 / 3)
    assert report.wrong_categories == ["b"]
    assert report.by_tag["typo"] == (2, 0.5)
    assert report.by_tag["clean"] == (1, 1.0)


def test_disputed_cases_stay_out_of_headline_accuracy_but_are_reported():
    cases = [
        case("a", "billing", tags=("clean",), text="a"),
        case("b", "refund", tags=("disputed",), text="b"),
    ]
    system = ByText({"a": Decision("billing", False), "b": Decision("other", False)})
    report = evaluate(system, cases)
    assert report.category_accuracy == 1.0
    assert report.disputed_accuracy == 0.0
    assert report.wrong_disputed == ["b"] and report.wrong_categories == []


def test_disputed_cases_still_count_for_escalation():
    cases = [case("a", needs_human=False), case("b", needs_human=True, tags=("disputed",))]
    report = evaluate(Fixed(needs_human=False), cases)
    assert report.missed_escalations == ["b"]


def test_all_disputed_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError, match="disputed"):
        evaluate(Fixed(), [case("a", tags=("disputed",))])


def test_flaky_system_shows_up_as_low_stability():
    # Three answers cycled over two cases means the second run differs from the first on every case.
    flip = itertools.cycle([Decision("refund", False), Decision("billing", False), Decision("other", False)])

    class Flaky:
        def decide(self, text):
            return next(flip)

    report = evaluate(Flaky(), [case("a"), case("b")], repeats=2)
    assert report.stability == 0.0
    assert evaluate(Fixed(), [case("a")], repeats=3).stability == 1.0


def test_gate_reports_each_metric_below_its_minimum():
    report = evaluate(Fixed(needs_human=False), [case("a", needs_human=True)])
    failures = check(report, {"escalation_recall": 0.5, "category_accuracy": 0.5})
    assert len(failures) == 1 and "escalation_recall" in failures[0]


def test_gate_rejects_unknown_threshold_names():
    report = evaluate(Fixed(), [case("a")])
    with pytest.raises(CaseError):
        check(report, {"escalation_recal": 0.5})


def test_repeats_must_be_positive():
    with pytest.raises(ValueError):
        evaluate(Fixed(), [case("a")], repeats=0)


@pytest.mark.parametrize(
    "row,fragment",
    [
        ({**GOOD, "category": "banana"}, "category"),
        ({**GOOD, "needs_human": "no"}, "needs_human"),
        ({**GOOD, "text": "  "}, "empty text"),
        ({k: v for k, v in GOOD.items() if k != "tags"}, "missing 'tags'"),
    ],
)
def test_load_cases_rejects_bad_rows(tmp_path, row, fragment):
    with pytest.raises(CaseError, match=fragment):
        load_cases(write(tmp_path, row))


def test_load_cases_rejects_duplicate_ids_and_bad_json(tmp_path):
    with pytest.raises(CaseError, match="duplicate id"):
        load_cases(write(tmp_path, GOOD, GOOD))
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(CaseError, match="not valid JSON"):
        load_cases(bad)


def test_real_cases_file_is_valid_and_covers_the_hard_kinds():
    cases = load_cases(REAL_CASES)
    tags = {t for c in cases for t in c.tags}
    assert len(cases) >= 40
    assert {"typo", "multi_issue", "injection", "security", "legal", "non_english", "vague"} <= tags
    assert {"pii", "asks_for_human", "unknown_order", "abusive", "mixed_language", "repeat", "long", "disputed"} <= tags
    assert {c.category for c in cases} == set(CATEGORIES)
    assert 0.2 < sum(c.needs_human for c in cases) / len(cases) < 0.6


def test_baseline_run_meets_committed_thresholds():
    assert main([]) == 0


def test_cli_fails_when_thresholds_are_raised(tmp_path):
    strict = tmp_path / "t.json"
    strict.write_text(json.dumps({"escalation_recall": 0.99}))
    assert main(["--thresholds", str(strict)]) == 1


def test_labels_follow_the_written_rules():
    cases = {c.id: c for c in load_cases(REAL_CASES)}
    assert cases["c05"].needs_human is False and cases["c06"].needs_human is True  # refund limit 5,000
    assert cases["c11"].needs_human is False  # two issues, neither risky
    assert cases["c27"].needs_human is True and cases["c35"].needs_human is True and cases["c38"].needs_human is True


def test_baseline_has_the_known_weak_spots():
    cases = {c.id: c for c in load_cases(REAL_CASES)}
    system = KeywordBaseline()
    assert system.decide(cases["c17"].text).needs_human is False  # injection slips through on purpose
