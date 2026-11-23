"""The baseline every later system must beat: ordered keyword rules, no model.

It was written to be plain, not tuned. Its mistakes on the eval cases are the starting point
for error analysis, so do not fix them here; improve the system that replaces it.
"""
import re

from flowguard.evals import Decision

CATEGORY_RULES = (
    ("refund", re.compile(r"refund|money back", re.I)),
    ("billing", re.compile(r"charg|bill|invoice|payment|gst", re.I)),
    ("shipping", re.compile(r"order|deliver|package|parcel|ship|arriv", re.I)),
    ("account", re.compile(r"log ?in|password|account|subscription", re.I)),
    ("technical", re.compile(r"crash|error|app |load|work", re.I)),
)
ESCALATION = re.compile(r"lawyer|court|stolen|\bscam\b|fraud", re.I)


class KeywordBaseline:
    def decide(self, text: str) -> Decision:
        category = next((name for name, rule in CATEGORY_RULES if rule.search(text)), "other")
        return Decision(category=category, needs_human=bool(ESCALATION.search(text)))
