"""The published rules. Verdict labels stay as they are until the owner changes them."""

from __future__ import annotations

from factcheck.aggregate import RULES_VERSION

_SCALE = [
    "math and logic, recomputed in code",
    "a law of nature or an established theory",
    "controlled studies and the historical record",
    "expert or official testimony",
    "an eyewitness account",
    "an anecdote",
]

_VERDICTS = [
    {
        "verdict": "Supported",
        "meaning": "A tool result, a structured fact of record, two independent upper-scale sources, or two explicit debunks of a denial quote the claim.",
    },
    {
        "verdict": "likely_true",
        "meaning": "One grade A or B source, or one explicit debunk, backs the claim. The site label is Likely true. This is not Supported.",
    },
    {
        "verdict": "Contradicted",
        "meaning": "A retrieved quote goes against the claim as written. Absence of results is never Contradicted.",
    },
    {
        "verdict": "Unverified",
        "meaning": "No usable passage backs the claim. The claim was not confirmed. It was not shown to be false.",
    },
    {
        "verdict": "Conflicting",
        "meaning": "Quoted sources disagree. A review disagreement does not create this verdict on its own.",
    },
    {
        "verdict": "Hypothesis",
        "meaning": "A prediction or conjecture. Status is supporting evidence, contradicting evidence, or untested. Never true or false.",
    },
]

_RULES = [
    "Language models rank and phrase evidence. They do not decide the verdict.",
    "A Supported verdict quotes a retrieved passage or a tool trace. A model's own knowledge is not evidence.",
    "No retrieved passage means Unverified, including when a search returns nothing.",
    "Conflicting and Contradicted require a quote that goes against the claim.",
    "When two review passes disagree, the original verdict stays if its quotes still check out.",
    "likely_true needs a grade A or B source, or one explicit fact-checker debunk of a denial. Grade C blogs stay Unverified.",
    "One encyclopedia can carry a fact of record, such as an office or a date. Every Wikipedia page counts as one origin.",
    "A fact-checker or encyclopedia quote found for one claim is shared with related claims in the same check.",
    "A page about a different person, or a page that does not discuss the claim, is not attached.",
    "Opinions are reported as what mainstream fact-checkers say. They are not labeled true or false.",
    "A check stops at 60 Desearch calls, counting retries, web search, dated search, extracts, and the mainstream lookup.",
]


def constitution_markdown() -> str:
    verdicts = "\n".join(f"- **{item['verdict']}.** {item['meaning']}" for item in _VERDICTS)
    scale = "\n".join(f"{index}. {item}" for index, item in enumerate(_SCALE, start=1))
    rules = "\n".join(f"- {item}" for item in _RULES)
    return f"""# Factcheck constitution

Version {RULES_VERSION}. These are the rules the checker applies. The verdict labels are Supported, likely_true, Contradicted, Unverified, Conflicting, and Hypothesis.

## Evidence scale

Strongest first:

{scale}

## Verdicts

{verdicts}

## Rules

{rules}

A verdict is an evidence summary. It cannot prove that no better evidence exists, and it is not proof.
"""


def constitution_document() -> dict:
    return {
        "version": RULES_VERSION,
        "title": "Factcheck constitution",
        "verdicts": _VERDICTS,
        "scale": _SCALE,
        "rules": _RULES,
        "markdown": constitution_markdown(),
    }
