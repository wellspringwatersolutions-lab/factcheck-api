"""Verdicts rest on passages and tool results, not on a model's vote.

Judges may rank or filter evidence. They cannot create a Supported verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from factcheck.credibility import source_weight
from factcheck.textutil import hostname
from factcheck.types import Evidence

RULES_VERSION = "4"

STRENGTH_NOTE = (
    "Strength of support is based on the evidence: source grade, how many independent "
    "sources agree, and whether the passages line up. It is not a model's confidence and it is not proof."
)


@dataclass
class Decision:
    verdict: str
    strength_of_support: float
    reasoning: str
    needs_human_review: bool = False
    hypothesis_status: str | None = None
    supporting_quotes: list[str] = field(default_factory=list)
    contradicting_quotes: list[str] = field(default_factory=list)
    would_settle: str = ""


_UPPER = {
    "systematic_review",
    "rct",
    "observational",
    "textbook",
    "primary",
    "primary_source",
    "data",
    "peer_reviewed",
    "structured",
}
_WEAK = {"eyewitness", "anecdote", "syndicated", "fact_checker"}


def fact_of_record(passage: Evidence) -> bool:
    """A structured record such as a Wikidata date can support a simple claim on its own."""
    return passage.provider == "wikidata" or (passage.source_type == "structured" and passage.grade in {"A", "B"})


def on_upper_scale(passage: Evidence) -> bool:
    """Math through expert/official testimony. Eyewitness and anecdote are below the line."""
    if passage.source_type in _WEAK:
        return False
    if passage.source_type in _UPPER:
        return True
    host = hostname(passage.url)
    return host.endswith(".gov") or host.endswith(".gov.uk") or host.endswith(".edu") or host.endswith(".ac.uk")


def is_authoritative(passage: Evidence) -> bool:
    """Kept for callers. A Supported label no longer follows from this alone."""
    host = hostname(passage.url)
    if passage.source_type in {"peer_reviewed", "structured"} and passage.grade in {"A", "B"}:
        return True
    if passage.source_type in {"primary", "textbook", "systematic_review", "data", "rct"} and passage.grade in {"A", "B"}:
        return True
    if host.endswith(".gov") or host.endswith(".gov.uk") or host.endswith(".edu") or host.endswith(".ac.uk"):
        return True
    if host in {"britannica.com", "nasa.gov", "who.int", "nih.gov", "cdc.gov", "nature.com", "science.org", "nejm.org"}:
        return True
    if passage.url.startswith("scifact://"):
        return True
    return False


def _independent(passages: list[Evidence], relation: str) -> list[Evidence]:
    chosen: dict[str, Evidence] = {}
    for passage in passages:
        if passage.entailment != relation or not passage.used or not passage.quote:
            continue
        origin = passage.origin_id or passage.url
        current = chosen.get(origin)
        if current is None or len(passage.quote) > len(current.quote):
            chosen[origin] = passage
    return list(chosen.values())


def _strength(passages: list[Evidence], *, category: str, verdict: str, origins: int) -> float:
    if verdict == "Unverified":
        return 0.15
    if verdict == "Hypothesis":
        return round(min(0.55, 0.2 + 0.1 * origins), 3)
    weights = [source_weight(passage, category) for passage in passages]
    quality = min(sum(weights), 2.0) / 2.0
    base = 0.2 + 0.45 * quality + 0.08 * min(origins, 4)
    if verdict == "Conflicting":
        base = 0.35
    elif verdict == "Unsupported":
        base = min(base, 0.42)
    elif verdict == "Contradicted" and origins < 2:
        base = min(base, 0.6)
    if category == "T7":
        base = min(base, 0.4)
    if category == "T6":
        base = min(base, 0.72)
    return round(min(0.95, max(0.0, base)), 3)


def decide(
    passages: list[Evidence],
    *,
    category: str,
    hypothesis: bool = False,
    high_stakes: bool = False,
    tool: dict | None = None,
    judge_votes_ignored: bool = True,
) -> Decision:
    """Decide from evidence. `judge_votes_ignored` documents that votes are not an input."""
    del judge_votes_ignored
    supporting = _independent(passages, "entails")
    contradicting = _independent(passages, "contradicts")
    for passage in passages:
        if passage.entailment == "conflicts" and passage.used and passage.quote:
            supporting.append(passage)
            contradicting.append(passage)
    support_quotes = [item.quote for item in supporting if item.quote]
    contradict_quotes = [item.quote for item in contradicting if item.quote]

    if tool and tool.get("tool") == "calculator" and tool.get("actual") is not None:
        trace = str(tool.get("trace") or "")
        if tool.get("matches"):
            return Decision(
                verdict="Supported",
                strength_of_support=0.95,
                reasoning=f"Recomputed with code: {trace}. The claimed value matches. {STRENGTH_NOTE}",
                needs_human_review=False,
                supporting_quotes=[trace],
            )
        return Decision(
            verdict="Contradicted",
            strength_of_support=0.95,
            reasoning=(
                f"Recomputed with code: {trace}. The claim says {tool.get('claimed')}, "
                f"which does not match. {STRENGTH_NOTE}"
            ),
            needs_human_review=False,
            contradicting_quotes=[trace],
        )
    if tool and tool.get("tool") and tool.get("trace"):
        matched = bool(tool.get("matches"))
        trace = str(tool["trace"])
        verdict = "Supported" if matched else "Contradicted"
        return Decision(
            verdict=verdict,
            strength_of_support=0.9,
            reasoning=f"Checked against {tool['tool']}: {trace}. {STRENGTH_NOTE}",
            needs_human_review=high_stakes and not matched,
            supporting_quotes=[trace] if matched else [],
            contradicting_quotes=[] if matched else [trace],
        )

    if hypothesis:
        if supporting and contradicting:
            status = "contradicting_evidence" if len(contradicting) >= len(supporting) else "supporting_evidence"
            review = True
        elif supporting:
            status = "supporting_evidence"
            review = high_stakes
        elif contradicting:
            status = "contradicting_evidence"
            review = True
        else:
            status = "untested"
            review = False
        settle = (
            "A quoted passage from an independent source, or a tool result, that directly "
            "confirms or refutes the claim as written would settle it. Until then it is not a fact."
        )
        return Decision(
            verdict="Hypothesis",
            strength_of_support=_strength(supporting + contradicting, category=category, verdict="Hypothesis", origins=len(supporting) + len(contradicting)),
            reasoning=(
                f"This is a hypothesis ({status.replace('_', ' ')}). "
                "It is not a true or false verdict. "
                + settle
                + " "
                + STRENGTH_NOTE
            ),
            needs_human_review=review,
            hypothesis_status=status,
            supporting_quotes=support_quotes,
            contradicting_quotes=contradict_quotes,
            would_settle=settle,
        )

    if not supporting and not contradicting:
        return Decision(
            verdict="Unverified",
            strength_of_support=0.15,
            reasoning="No retrieved passage or tool result backs this claim. " + STRENGTH_NOTE,
            needs_human_review=high_stakes,
        )

    if supporting and contradicting:
        return Decision(
            verdict="Conflicting",
            strength_of_support=_strength(supporting + contradicting, category=category, verdict="Conflicting", origins=len(supporting) + len(contradicting)),
            reasoning=(
                "Independent sources disagree. The report shows both passages. "
                "This is not a guess at which side is right. " + STRENGTH_NOTE
            ),
            needs_human_review=True,
            supporting_quotes=support_quotes,
            contradicting_quotes=contradict_quotes,
        )

    if supporting and not contradicting:
        upper = [item for item in supporting if on_upper_scale(item)]
        records = [item for item in supporting if fact_of_record(item)]
        if records:
            return Decision(
                verdict="Supported",
                strength_of_support=max(0.7, _strength(records, category=category, verdict="Supported", origins=len(records))),
                reasoning=(
                    "A structured fact of record quotes a passage that matches the claim. "
                    + STRENGTH_NOTE
                ),
                needs_human_review=high_stakes,
                supporting_quotes=[item.quote for item in records if item.quote],
            )
        if len(upper) >= 2:
            return Decision(
                verdict="Supported",
                strength_of_support=max(0.72, _strength(upper, category=category, verdict="Supported", origins=len(upper))),
                reasoning=(
                    f"{len(upper)} independent sources from the upper part of the evidence scale "
                    "contain passages that line up with the claim. The quotes are in the report. "
                    + STRENGTH_NOTE
                ),
                needs_human_review=high_stakes,
                supporting_quotes=[item.quote for item in upper if item.quote],
            )
        return Decision(
            verdict="Unverified",
            strength_of_support=min(0.35, _strength(supporting, category=category, verdict="Unverified", origins=len(supporting))),
            reasoning=(
                "A passage was found, but it is not enough to confirm the claim. "
                "Anecdote, a single source, or a source below the historical and official record "
                "is not rounded up to supported. " + STRENGTH_NOTE
            ),
            needs_human_review=True,
            supporting_quotes=support_quotes,
        )

    credible_against = [item for item in contradicting if item.source_type not in {"anecdote", "eyewitness", "syndicated"}]
    if not credible_against:
        return Decision(
            verdict="Unverified",
            strength_of_support=0.15,
            reasoning="The only passage against the claim is too weak to treat the claim as false. " + STRENGTH_NOTE,
            needs_human_review=high_stakes,
            contradicting_quotes=contradict_quotes,
        )
    return Decision(
        verdict="Contradicted",
        strength_of_support=_strength(contradicting, category=category, verdict="Contradicted", origins=len(contradicting)),
        reasoning=(
            "A retrieved passage conflicts with the claim. The quote is in the report. "
            + ("Only one independent source was found, so a person should look at the trail. " if len(contradicting) < 2 else "")
            + STRENGTH_NOTE
        ),
        needs_human_review=high_stakes or len(contradicting) < 2,
        contradicting_quotes=contradict_quotes,
    )
