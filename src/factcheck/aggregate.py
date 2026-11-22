"""Verdicts rest on passages and tool results, not on a model's vote.

Judges may rank or filter evidence. They cannot create a Supported verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from factcheck.checks import is_denial
from factcheck.credibility import is_encyclopedia, source_weight
from factcheck.textutil import hostname, split_sentences
from factcheck.types import Evidence

RULES_VERSION = "5"

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
    "encyclopedia",
    "primary",
    "primary_source",
    "data",
    "peer_reviewed",
    "structured",
}
_RECORD_CLAIM = re.compile(
    r"\b(president|office|born|birth|died|death|general|commander|king|queen|founded|ordinal)\b",
    re.I,
)
_NEGATIVE_CRIME = re.compile(
    r"\b(not a criminal|never charged|never convicted|no criminal|was not charged|was not a criminal|wasn't a criminal)\b",
    re.I,
)
_GUILT = re.compile(r"\b(convicted|was charged|found guilty|pleaded guilty|criminal charge)\b", re.I)
_CRIME_TALK = re.compile(
    r"\b(criminal|crimes?|felon|felony|convict(?:ed|ion)?|arrest(?:ed)?|indict(?:ed|ment)?|charged|guilty|"
    r"prison|jail(?:ed)?|sentenced|fraud|murder(?:ed)?|theft|treason)\b",
    re.I,
)
# A loaded accusation: "X was a criminal", "X is a convicted felon", "X committed crimes".
_POSITIVE_CRIME = re.compile(
    r"\b(?:an?\s+)?(?:convicted\s+)?(?:criminal|felon|crook|convict)s?\b"
    r"(?!\s+(?:charges?|cases?|investigations?|records?|courts?|law|trials?|justice|complaints?|probes?|"
    r"defen[cs]e|lawyers?|organi[sz]ations?|enterprises?|networks?|history|activit(?:y|ies)|code|procedure))"
    r"|\bcommitted\s+(?:a\s+)?crimes?\b",
    re.I,
)
_ROLE_WORDS = re.compile(r"\b(president|father|general|born|office|statesman|minister|commander)\b", re.I)
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


def _biographical_sentence(text: str) -> str:
    parts = split_sentences(text or "")
    for part in parts:
        if _ROLE_WORDS.search(part) and not _GUILT.search(part):
            return part
    return parts[0] if parts else ""


_CURRENT = re.compile(
    r"\b(20(?:2[3-9]|3\d)|today|this week|hack(?:ed|ing)?|cyber|stockpile|preparedness|attack)\b",
    re.I,
)
_DATELINE = re.compile(r"^(published|updated|posted)\b", re.I)


def _current_event(text: str) -> bool:
    return bool(_CURRENT.search(text or ""))


def _body_quote(passage: Evidence) -> bool:
    quote = (passage.quote or "").strip()
    if len(quote) < 40 or _DATELINE.match(quote):
        return False
    title = " ".join((passage.title or "").lower().split())
    return " ".join(quote.lower().split()) != title


def _reputable(passage: Evidence) -> bool:
    """Tier B or better: major news, government, or an encyclopedia. Trade press stays out."""
    if passage.grade not in {"A", "B"}:
        return False
    if passage.source_type in {"anecdote", "eyewitness", "syndicated"}:
        return False
    return True


def _primary_source(passage: Evidence) -> bool:
    """An official page can carry likely_true on its own."""
    host = hostname(passage.url)
    if host.endswith(".gov") or host.endswith(".gov.uk") or host.endswith(".gouv.fr") or host.endswith(".gc.ca"):
        return True
    if host == "europa.eu" or host.endswith(".europa.eu"):
        return True
    if host == "ready.gov" or host.endswith(".ready.gov"):
        return True
    return passage.source_type in {"primary", "primary_source"} and passage.grade in {"A", "B"}


def _agrees(claim_text: str, passage: Evidence) -> bool:
    text = (passage.quote or passage.snippet or passage.page_text or "").strip()
    if len(text) < 40 or _DATELINE.match(text) or passage.entailment != "entails":
        return False
    from factcheck.checks import _clearly_backs

    return _clearly_backs(claim_text, text)


def _independent_agreements(claim_text: str, passages: list[Evidence]) -> list[Evidence]:
    """Three agreeing sources count only when each one is grade A or B."""
    chosen: dict[str, Evidence] = {}
    for passage in passages:
        if passage.grade not in {"A", "B"} and not _reputable(passage):
            continue
        if passage.source_type in {"anecdote", "eyewitness", "syndicated"}:
            continue
        if not _agrees(claim_text, passage):
            continue
        origin = passage.origin_id or hostname(passage.url)
        if not origin or origin in chosen:
            continue
        chosen[origin] = passage
    return list(chosen.values())


def _backs_substance(claim_text: str, passage: Evidence) -> bool:
    if not _reputable(passage):
        return False
    text = (passage.quote or passage.snippet or passage.page_text or "").strip()
    if len(text) < 40 or _DATELINE.match(text):
        return False
    if passage.entailment != "entails":
        return False
    from factcheck.checks import _clearly_backs

    return _clearly_backs(claim_text, text)


def _mark_likely(passages: list[Evidence]) -> None:
    for passage in passages:
        if not passage.quote:
            passage.quote = (passage.snippet or passage.page_text or "")[:400]
        passage.used = True
        if passage.entailment not in {"entails", "contradicts"}:
            passage.entailment = "entails"


def _two_reliable_news(passages: list[Evidence]) -> bool:
    origins = {
        passage.origin_id or hostname(passage.url)
        for passage in passages
        if passage.grade in {"A", "B"} and _body_quote(passage)
    }
    origins.discard("")
    return len(origins) >= 2


def criminality_absence(claim_text: str, passages: list[Evidence]) -> Evidence | None:
    """A reliable biography with no charge or conviction supports 'was not a criminal'."""
    if not _NEGATIVE_CRIME.search(claim_text or ""):
        return None
    record = " ".join((item.page_text or item.snippet or "") for item in passages)
    if _GUILT.search(record):
        return None
    for item in passages:
        reliable = item.source_type == "encyclopedia" or item.provider in {"wikipedia", "wikidata"} or item.grade in {"A", "B"}
        text = item.page_text or item.snippet or ""
        if reliable and _ROLE_WORDS.search(text) and not _GUILT.search(text):
            return item
    return None


def criminality_accusation_absence(claim_text: str, passages: list[Evidence]) -> Evidence | None:
    """Mirror of criminality_absence for the loaded form, e.g. 'X was a criminal'.

    Absence of evidence alone stays Unverified. This only fires when a reliable biography of
    the person (encyclopedia, Wikipedia/Wikidata, or an A/B source) covers their record and
    nothing retrieved mentions a charge or conviction. That is an authoritative source
    covering the topic, so the accusation is contradicted.
    """
    text = claim_text or ""
    if not _POSITIVE_CRIME.search(text) or _NEGATIVE_CRIME.search(text):
        return None
    return criminality_absence("was not a criminal", passages)


def decide(
    passages: list[Evidence],
    *,
    category: str,
    hypothesis: bool = False,
    high_stakes: bool = False,
    tool: dict | None = None,
    claim_text: str = "",
    judge_votes_ignored: bool = True,
) -> Decision:
    """Decide from evidence. `judge_votes_ignored` documents that votes are not an input."""
    del judge_votes_ignored
    from factcheck.checks import entailment as _entail

    for passage in passages:
        if passage.entailment:
            continue
        text = passage.quote or passage.snippet or passage.page_text or ""
        relation = _entail(claim_text, text) if claim_text else "neutral"
        if relation in {"entails", "contradicts"}:
            passage.entailment = relation
            passage.used = True
            if not passage.quote:
                passage.quote = text[:400]
    if _POSITIVE_CRIME.search(claim_text or "") and not _NEGATIVE_CRIME.search(claim_text or ""):
        # A passage that never mentions a crime cannot back "X was a criminal", however many
        # of the claim's other words (the name, the office) it shares.
        for passage in passages:
            body = passage.quote or passage.snippet or passage.page_text or ""
            if passage.entailment == "entails" and not _CRIME_TALK.search(body):
                passage.entailment = "neutral"
                passage.used = False
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

    absence = None if contradicting else criminality_absence(claim_text, passages)
    if absence is not None:
        quote = _biographical_sentence(absence.page_text or absence.snippet or absence.quote)
        if quote:
            absence.quote = quote
            absence.used = True
            absence.entailment = "entails"
            return Decision(
                verdict="Supported",
                strength_of_support=0.62,
                reasoning=(
                    "Supported by absence of evidence for criminality. "
                    "A reliable source describes the person in non-criminal terms and the record "
                    f"shows no conviction or criminal charge. {quote} {STRENGTH_NOTE}"
                ),
                needs_human_review=False,
                supporting_quotes=[quote],
            )

    accusation = None if supporting else criminality_accusation_absence(claim_text, passages)
    if accusation is not None:
        quote = _biographical_sentence(accusation.page_text or accusation.snippet or accusation.quote)
        if quote:
            accusation.quote = quote
            accusation.used = True
            accusation.entailment = "contradicts"
            return Decision(
                verdict="Contradicted",
                strength_of_support=0.62,
                reasoning=(
                    "Contradicted. A reliable biography covers this person's record and describes them "
                    "in non-criminal terms, and no retrieved source shows any criminal charge or conviction. "
                    f"{quote} {STRENGTH_NOTE}"
                ),
                needs_human_review=False,
                contradicting_quotes=[quote],
            )

    if not supporting and not contradicting:
        primaries = [item for item in passages if _primary_source(item) and _agrees(claim_text, item)]
        if primaries:
            _mark_likely(primaries[:1])
            return Decision(
                verdict="likely_true",
                strength_of_support=0.6,
                reasoning=(
                    "Likely true. One official source backs the substance of the claim. " + STRENGTH_NOTE
                ),
                needs_human_review=high_stakes,
                supporting_quotes=[item.quote for item in primaries if item.quote][:1],
            )
        group = _independent_agreements(claim_text, passages)
        if len(group) >= 3:
            _mark_likely(group[:3])
            return Decision(
                verdict="likely_true",
                strength_of_support=0.56,
                reasoning=(
                    f"Likely true. {len(group)} independent sources agree on the substance of the claim. "
                    + STRENGTH_NOTE
                ),
                needs_human_review=high_stakes,
                supporting_quotes=[item.quote for item in group if item.quote][:3],
            )
        backers = [item for item in passages if _backs_substance(claim_text, item)]
        if backers:
            _mark_likely(backers[:2])
            return Decision(
                verdict="likely_true",
                strength_of_support=0.58,
                reasoning=(
                    "Likely true. Reputable sources back the substance of the claim, "
                    "but there is no exact entailing quote to mark it supported. " + STRENGTH_NOTE
                ),
                needs_human_review=high_stakes,
                supporting_quotes=[item.quote for item in backers if item.quote][:2],
            )
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
        records = [
            item
            for item in supporting
            if fact_of_record(item) or (item.source_type == "encyclopedia" and _RECORD_CLAIM.search(claim_text or ""))
        ]
        if records:
            kind = "An encyclopedia" if records[0].source_type == "encyclopedia" else "A structured fact of record"
            return Decision(
                verdict="Supported",
                strength_of_support=max(0.7, _strength(records, category=category, verdict="Supported", origins=len(records))),
                reasoning=(
                    f"{kind} quotes a passage that matches the claim. "
                    + STRENGTH_NOTE
                ),
                needs_human_review=high_stakes,
                supporting_quotes=[item.quote for item in records if item.quote],
            )
        if _current_event(claim_text) and _two_reliable_news(supporting):
            return Decision(
                verdict="Supported",
                strength_of_support=max(0.7, _strength(supporting, category=category, verdict="Supported", origins=2)),
                reasoning=(
                    "Two independent major outlets quote passages that line up with this current-events claim. "
                    + STRENGTH_NOTE
                ),
                needs_human_review=high_stakes,
                supporting_quotes=[item.quote for item in supporting if item.quote][:3],
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
        reputable = [item for item in supporting if _reputable(item)]
        if reputable:
            _mark_likely(reputable[:2])
            return Decision(
                verdict="likely_true",
                strength_of_support=0.62,
                reasoning=(
                    "Likely true. "
                    + (
                        "One reputable source quotes a passage that lines up with the claim. "
                        if len(reputable) == 1
                        else "Reputable sources back the claim, but not as two independent exact quotes. "
                    )
                    + STRENGTH_NOTE
                ),
                needs_human_review=high_stakes,
                supporting_quotes=[item.quote for item in reputable if item.quote][:2],
            )
        primaries = [item for item in supporting if _primary_source(item)]
        if primaries:
            return Decision(
                verdict="likely_true",
                strength_of_support=0.6,
                reasoning="Likely true. One official source quotes a passage that lines up with the claim. " + STRENGTH_NOTE,
                needs_human_review=high_stakes,
                supporting_quotes=[item.quote for item in primaries if item.quote][:1],
            )
        from factcheck.checks import is_debunk

        debunks = [
            item
            for item in supporting
            if is_denial(claim_text) and is_debunk(item.quote or item.snippet or item.page_text or "")
        ]
        debunk_origins = {item.origin_id or hostname(item.url) for item in debunks}
        debunk_origins.discard("")
        if len(debunk_origins) >= 2:
            return Decision(
                verdict="Supported",
                strength_of_support=0.7,
                reasoning=(
                    "Supported. Independent sources explicitly say the rumor is not what the government told people. "
                    + STRENGTH_NOTE
                ),
                needs_human_review=high_stakes,
                supporting_quotes=[item.quote for item in debunks if item.quote][:3],
            )
        if debunks:
            _mark_likely(debunks[:1])
            return Decision(
                verdict="likely_true",
                strength_of_support=0.6,
                reasoning=(
                    "Likely true. A source explicitly denies the rumor this claim rejects. " + STRENGTH_NOTE
                ),
                needs_human_review=high_stakes,
                supporting_quotes=[item.quote for item in debunks if item.quote][:1],
            )
        good = [item for item in supporting if item.grade in {"A", "B"} or _reputable(item)]
        origins = {item.origin_id or hostname(item.url) for item in good}
        origins.discard("")
        if len(origins) >= 3:
            return Decision(
                verdict="likely_true",
                strength_of_support=0.56,
                reasoning=(
                    f"Likely true. {len(origins)} independent reputable sources quote passages that line up with the claim. "
                    + STRENGTH_NOTE
                ),
                needs_human_review=high_stakes,
                supporting_quotes=[item.quote for item in good if item.quote][:3],
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
