"""The published rules. The public API has five verdicts."""

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
        "verdict": "supported",
        "meaning": "A tool result, a structured fact of record, two independent upper-scale sources, or two explicit debunks of a denial quote the claim.",
    },
    {
        "verdict": "hypothesis",
        "meaning": "Either one good source makes the claim probably true (verdict_detail likely_true) or the claim is speculative (verdict_detail speculative). It is not stated as confirmed. It still counts toward the rating.",
    },
    {
        "verdict": "opinion",
        "meaning": "Not a fact check. The report says what mainstream fact-checkers wrote, which is not a true or false label.",
    },
    {
        "verdict": "conflicting",
        "meaning": "Sources disagree, or a quoted source goes against the claim (verdict_detail contradicted). Absence of results is never conflicting.",
    },
    {
        "verdict": "unverified",
        "meaning": "No usable passage backs the claim. The claim was not confirmed. It was not shown to be false.",
    },
]

_RULES = [
    "The public verdict is only supported, hypothesis, opinion, conflicting, or unverified, in that order, and lowercase.",
    "verdict_detail is likely_true, speculative, or contradicted when the client needs the finer state. verdict_label is the display name, such as Hypothesis.",
    "Older names (Supported, likely_true, Contradicted, Unverified, Conflicting, Unsupported, Hypothesis) are still accepted on input.",
    "Language models rank and phrase evidence. They do not decide the verdict.",
    "A supported verdict quotes a retrieved passage or a tool trace. A model's own knowledge is not evidence.",
    "No retrieved passage means unverified, including when a search returns nothing.",
    "A claim becomes conflicting only when a quote goes against it or sources disagree.",
    "When two review passes disagree, the original verdict stays if its quotes still check out.",
    "One grade A or B source, or one explicit debunk of a denial, is hypothesis with verdict_detail likely_true. Grade C blogs stay unverified.",
    "likely_true still counts toward Partly verified and Mostly accurate. The remap does not turn that rating into not enough evidence.",
    "A contradicted claim is public conflicting with verdict_detail contradicted, and the summary describes it as disputed by sources.",
    "One encyclopedia can carry a fact of record, such as an office or a date. Every Wikipedia page counts as one origin.",
    "A fact-checker or encyclopedia quote found for one claim is shared with related claims in the same check.",
    "A page about a different person, or a page that does not discuss the claim, is not attached.",
    "A check stops at 60 Desearch calls, counting retries, web search, dated search, extracts, and the mainstream lookup.",
]


def constitution_markdown() -> str:
    verdicts = "\n".join(f"- **{item['verdict']}.** {item['meaning']}" for item in _VERDICTS)
    scale = "\n".join(f"{index}. {item}" for index, item in enumerate(_SCALE, start=1))
    rules = "\n".join(f"- {item}" for item in _RULES)
    return f"""# Factcheck constitution

Version {RULES_VERSION}. The public verdicts, in order, are supported, hypothesis, opinion, conflicting, and unverified.

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
