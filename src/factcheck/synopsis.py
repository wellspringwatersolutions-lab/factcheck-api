"""A short bottom line built only from claim verdicts and the quotes already cited.

The rating is deterministic and follows evidence quality. One optional language-model
call may phrase that rating. If the call fails, runs long, or adds a fact that is
not already in the brief, the deterministic sentence is what the report shows.
"""

from __future__ import annotations

import re
import time

from factcheck.llm import LLMError, estimate_cost, loads_object
from factcheck.request_ctx import charge_tokens
from factcheck.schemas import ClaimOut, SynopsisCounts, SynopsisOut

_RATINGS = (
    "Accurate",
    "Mostly accurate",
    "Mixed",
    "Mostly inaccurate",
    "Inaccurate",
    "Not enough evidence",
)
_NUANCE = re.compile(r"\b(but|however|although|though|technically|except|nonetheless|nevertheless)\b", re.I)
_NEGATION = re.compile(r"\b(never|not|no|wasn't|weren't|isn't|aren't|didn't|without|neither)\b", re.I)
_YESNO = re.compile(r"^(was|is|did|were|has|have|do|does|had|are)\b", re.I)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z][A-Za-z']{3,}")
_STOP = {
    "about", "after", "again", "also", "although", "among", "because", "before", "being",
    "between", "both", "claim", "claims", "checked", "could", "does", "doing", "during",
    "each", "enough", "evidence", "every", "from", "have", "having", "however", "important",
    "into", "just", "many", "more", "most", "mostly", "much", "not", "only", "other",
    "others", "over", "should", "since", "some", "still", "such", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "those", "through", "under", "until",
    "using", "very", "were", "what", "when", "where", "which", "while", "with", "without",
    "would", "accurate", "inaccurate", "mixed", "supported", "contradicted", "conflicting",
    "unverified", "hypothesis", "answer", "question", "hold", "holds", "been", "were",
    "technically", "never", "does",
}
_ABSTAIN = "There wasn't enough evidence to settle this."
_INSTRUCTION = """You phrase a veracity synopsis. You do not decide it.
The rating is already fixed and must be the first words of summary: {rating}
Write summary as one to three plain sentences that answer the question from the claim verdicts and cited quotes below.
Also write tldr as one complete sentence of at most 30 words. It must start with a capital letter, end with a period, and must not start with a comma or the word but.
Do not add any person, place, date, number, or event that is not written in the brief.
If the rating is "Not enough evidence", both summary and tldr must say there was not enough evidence, and must not sound confident.
Return JSON {{"summary": "...", "tldr": "..."}}"""


def _unchecked(claim: ClaimOut) -> bool:
    if claim.effort == "skipped":
        return True
    return (claim.reasoning or "").startswith("Not checked")


def _weight(claim: ClaimOut) -> float:
    try:
        value = float(claim.strength_of_support or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _checked_facts(claims: list[ClaimOut]) -> list[ClaimOut]:
    return [
        claim
        for claim in claims
        if claim.claim_type == "fact" and claim.verdict != "Hypothesis" and not _unchecked(claim)
    ]


def _caveats(claims: list[ClaimOut]) -> list[str]:
    notes: list[str] = []
    for claim in claims:
        if _unchecked(claim) or claim.claim_type != "fact":
            continue
        if claim.verdict in {"Contradicted", "Conflicting"}:
            notes.append(claim.text)
    supported = [claim for claim in _checked_facts(claims) if claim.verdict == "Supported"]
    negations = [claim for claim in supported if _NEGATION.search(claim.text)]
    for claim in supported:
        qualifies = bool(_NUANCE.search(claim.text)) or (bool(negations) and claim not in negations)
        if qualifies:
            notes.append(claim.text)
    seen: list[str] = []
    for note in notes:
        if note and note not in seen:
            seen.append(note)
    return seen[:4]


def _has_nuance(supported: list[ClaimOut]) -> bool:
    if any(_NUANCE.search(claim.text) for claim in supported):
        return True
    negations = [claim for claim in supported if _NEGATION.search(claim.text)]
    return bool(negations) and len(negations) < len(supported)


def rate_claims(claims: list[ClaimOut]) -> tuple[str, list[str], int]:
    """Evidence quality only. A weak or mostly unverified set stays 'Not enough evidence'."""
    caveats = _caveats(claims)
    unchecked_count = sum(1 for claim in claims if _unchecked(claim))
    facts = _checked_facts(claims)
    if not facts:
        return "Not enough evidence", caveats, unchecked_count
    decided = [claim for claim in facts if claim.verdict in {"Supported", "likely_true", "Contradicted", "Conflicting"}]
    unverified = [claim for claim in facts if claim.verdict in {"Unverified", "Unsupported", None}]
    if not decided or len(unverified) >= len(decided):
        return "Not enough evidence", caveats, unchecked_count
    supported = [claim for claim in decided if claim.verdict == "Supported"]
    likely = [claim for claim in decided if claim.verdict == "likely_true"]
    contradicted = [claim for claim in decided if claim.verdict == "Contradicted"]
    conflicting = [claim for claim in decided if claim.verdict == "Conflicting"]
    if not any(_weight(claim) >= 0.45 for claim in decided):
        return "Not enough evidence", caveats, unchecked_count
    support_w = sum(_weight(claim) for claim in supported) + sum(_weight(claim) for claim in likely)
    contra_w = sum(_weight(claim) for claim in contradicted) + sum(_weight(claim) for claim in conflicting)
    total = support_w + contra_w
    if total <= 0:
        return "Not enough evidence", caveats, unchecked_count
    support_share = support_w / total
    contra_share = contra_w / total
    if contra_w == 0 and support_share >= 0.999:
        rating = "Mostly accurate" if unverified or likely or _has_nuance(supported) else "Accurate"
    elif support_share >= 0.6 and contra_share < 0.4:
        rating = "Mostly accurate"
    elif contra_share >= 0.85 and support_w == 0 and not unverified:
        rating = "Inaccurate"
    elif contra_share >= 0.6:
        rating = "Mostly inaccurate"
    else:
        rating = "Mixed"
    if rating not in _RATINGS:
        rating = "Not enough evidence"
    return rating, caveats, unchecked_count


def _soft(text: str) -> str:
    cleaned = (text or "").strip().rstrip(".")
    if not cleaned:
        return ""
    return cleaned[0].lower() + cleaned[1:]


def _direct(question: str | None, main: list[str]) -> str:
    if not question or not _YESNO.match(question.strip()):
        return ""
    if main and all(_NEGATION.search(text) for text in main):
        return "No"
    if main and not any(_NEGATION.search(text) for text in main):
        return "Yes"
    return ""


def fallback_summary(rating: str, question: str | None, claims: list[ClaimOut], caveats: list[str], unchecked_count: int) -> str:
    facts = _checked_facts(claims)
    supported = [claim.text for claim in facts if claim.verdict == "Supported"]
    contradicted = [claim.text for claim in facts if claim.verdict == "Contradicted"]
    unverified = [claim for claim in facts if claim.verdict in {"Unverified", "Unsupported", None}]
    if rating == "Not enough evidence":
        parts = ["Not enough evidence."]
        if question:
            parts.append(f"This does not settle “{question.strip().rstrip('?')}”.")
        elif facts or unchecked_count:
            parts.append("Most of the checkable claims are unverified or were not checked.")
        else:
            parts.append("No checkable claims were found.")
        if unverified:
            shown = "; ".join(claim.text.rstrip(".") for claim in unverified[:3])
            parts.append(f"Unverified: {shown}.")
        if unchecked_count:
            word = "claim" if unchecked_count == 1 else "claims"
            verb = "was" if unchecked_count == 1 else "were"
            parts.append(f"{unchecked_count} important {word} {verb} not checked.")
        corrections = _cited_corrections(claims)
        if corrections:
            parts.append("The checked claim does not hold.")
            parts.append("The cited evidence says: " + "; ".join(item.rstrip(".") for item in corrections) + ".")
        return " ".join(parts)

    main = [text for text in supported if text not in caveats][:2]
    caveat_bits = [_soft(text) for text in caveats if text not in main][:2]
    corrections = _cited_corrections(claims)
    if rating in {"Inaccurate", "Mostly inaccurate"} and contradicted:
        sentence = "The checked claims do not hold: " + "; ".join(_soft(text) for text in contradicted[:3])
        if corrections:
            sentence += ". The cited evidence says: " + "; ".join(item.rstrip(".") for item in corrections)
    elif rating == "Mixed" and supported and contradicted:
        sentence = (
            "Some checked claims hold ("
            + "; ".join(_soft(text) for text in supported[:2])
            + "), but others do not ("
            + "; ".join(_soft(text) for text in contradicted[:2])
            + ")"
        )
        if corrections:
            sentence += ". The cited evidence says: " + "; ".join(item.rstrip(".") for item in corrections)
    else:
        direct = _direct(question, main or supported[:1])
        body = main or supported[:1]
        if direct and body:
            sentence = f"{direct}, " + ", and ".join(_soft(text) for text in body)
        else:
            sentence = "; ".join(text.strip().rstrip(".") for text in body)
        if caveat_bits and sentence.strip():
            sentence = sentence.rstrip(".") + ", but " + " and ".join(caveat_bits)
        elif caveat_bits:
            sentence = caveat_bits[0][:1].upper() + caveat_bits[0][1:]
    sentence = sentence.rstrip(".") + "."
    tail = ""
    notes = []
    if unverified and rating != "Not enough evidence":
        notes.append(f"{len(unverified)} stayed unverified")
    if unchecked_count:
        notes.append(f"{unchecked_count} {'was' if unchecked_count == 1 else 'were'} not checked")
    if notes:
        tail = "Also, " + " and ".join(notes) + "."
    return " ".join(part for part in (f"{rating}.", sentence, tail) if part)


def _cited_corrections(claims: list[ClaimOut]) -> list[str]:
    found: list[str] = []
    for claim in claims:
        if claim.verdict != "Contradicted":
            continue
        for source in claim.sources:
            if not source.quote:
                continue
            if source.entailment == "contradicts" or source.stance == "contradicts":
                text = source.quote.strip()
                if text and text not in found:
                    found.append(text)
    return found[:2]


def _brief(rating: str, question: str | None, claims: list[ClaimOut]) -> str:
    lines = [f"Rating: {rating}"]
    if question:
        lines.append(f"Question: {question}")
    for claim in claims:
        lines.append(f"Claim ({claim.verdict}, strength {claim.strength_of_support:.2f}): {claim.text}")
        for source in claim.sources:
            if source.quote:
                lines.append(f"Cited quote: {source.quote}")
    return "\n".join(lines)


def claim_counts(claims: list[ClaimOut]) -> SynopsisCounts:
    counts = SynopsisCounts()
    for claim in claims:
        if _unchecked(claim):
            counts.not_checked += 1
        elif claim.verdict == "Hypothesis":
            counts.hypothesis += 1
        elif claim.claim_type == "opinion":
            counts.opinion += 1
        elif claim.verdict == "Supported":
            counts.supported += 1
        elif claim.verdict == "likely_true":
            counts.likely_true += 1
        elif claim.verdict == "Contradicted":
            counts.contradicted += 1
        elif claim.verdict == "Conflicting":
            counts.conflicting += 1
        else:
            counts.unverified += 1
    return counts


_DANGLING = {"and", "or", "but", "the", "a", "an", "of", "to", "for", "with", "his", "her", "their"}


def complete_sentence(text: str) -> bool:
    """A usable bottom-line sentence, not a cut-off clause."""
    cleaned = " ".join((text or "").split())
    if len(cleaned) < 8 or cleaned[0] in ",;:" or not cleaned[0].isupper():
        return False
    if cleaned[-1] not in ".!?":
        return False
    words = cleaned[:-1].split()
    if len(words) < 4 or words[-1].lower().strip(",;:") in _DANGLING:
        return False
    if words[0].lower() in {"but", "and", "or"}:
        return False
    return True


def _template_sentence(rating: str, claims: list[ClaimOut]) -> str:
    if rating == "Not enough evidence":
        return _ABSTAIN
    for claim in claims:
        if claim.verdict not in {"Supported", "likely_true"}:
            continue
        sentence = claim.text.strip()
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        if complete_sentence(sentence):
            return sentence
    fixed = {
        "Accurate": "The checked claims are accurate.",
        "Mostly accurate": "The checked claims are mostly accurate.",
        "Mixed": "The checked claims are mixed.",
        "Mostly inaccurate": "The checked claims are mostly inaccurate.",
        "Inaccurate": "The checked claims are inaccurate.",
    }
    return fixed.get(rating, _ABSTAIN)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text or ""))


def fallback_tldr(rating: str, summary: str, claims: list[ClaimOut] | None = None) -> str:
    """One whole sentence. A fragment is replaced with a template."""
    if rating == "Not enough evidence":
        return _ABSTAIN
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", summary or "") if part.strip()]
    for part in parts:
        if part.rstrip(".").lower() == rating.lower():
            continue
        if _word_count(part) > 30:
            continue
        if complete_sentence(part):
            return part
    return _template_sentence(rating, claims or [])


def grounded(summary: str, corpus: str) -> bool:
    """Reject a phrasing that names a number or content word absent from the brief."""
    if not summary or len(summary) > 700:
        return False
    haystack = corpus.lower()
    for number in _NUMBER.findall(summary):
        if number not in corpus:
            return False
    for word in _WORD.findall(summary):
        token = word.lower()
        if token in _STOP:
            continue
        if token not in haystack:
            return False
    return True


def _says_not_enough(text: str) -> bool:
    lowered = (text or "").lower()
    return "not enough evidence" in lowered or "enough evidence to settle" in lowered or "wasn't enough evidence" in lowered


def _accept_tldr(text: str, corpus: str, rating: str) -> bool:
    cleaned = (text or "").strip()
    if not complete_sentence(cleaned) or _word_count(cleaned) > 30:
        return False
    allowed = corpus
    if rating == "Not enough evidence":
        if not _says_not_enough(cleaned):
            return False
        allowed = f"{corpus}\n{_ABSTAIN}"
    return grounded(cleaned, allowed)


def build_synopsis(
    claims: list[ClaimOut],
    *,
    question: str | None = None,
    phrasing=None,
    deadline: float | None = None,
) -> SynopsisOut:
    rating, caveats, unchecked_count = rate_claims(claims)
    summary = fallback_summary(rating, question, claims, caveats, unchecked_count)
    tldr = fallback_tldr(rating, summary, claims)
    counts = claim_counts(claims)
    # A contradiction is stated from the cited quote. Do not let a phrasing call rewrite it.
    contradicted = any(claim.verdict == "Contradicted" for claim in claims)
    remaining = None if deadline is None else deadline - time.monotonic()
    if phrasing is not None and not contradicted and (remaining is None or remaining >= 3.5):
        brief = _brief(rating, question, claims)
        try:
            phrased = phrasing(brief, rating)
        except Exception:  # noqa: BLE001 - a failed phrasing call must not fail the report
            phrased = None
        summary_text, tldr_text = _phrased_parts(phrased)
        if summary_text and grounded(summary_text, brief):
            cleaned = summary_text.strip()
            body = cleaned[len(rating):].lstrip(" .") if cleaned.lower().startswith(rating.lower()) else cleaned
            if complete_sentence(body):
                summary = cleaned if cleaned.lower().startswith(rating.lower()) else f"{rating}. {body}"
        if tldr_text and _accept_tldr(tldr_text, brief, rating):
            tldr = tldr_text.strip()
    from factcheck.reconcile import answer_for_rating

    supported_texts = [claim.text for claim in claims if claim.verdict == "Supported"]
    direct = _direct(question, supported_texts)
    negatives = [text for text in supported_texts if _NEGATION.search(text)]
    if rating in {"Accurate", "Mostly accurate"} and question and _YESNO.match(question.strip()):
        if negatives and (direct == "No" or len(negatives) < len(supported_texts)):
            answer = "no"
        elif direct == "Yes" or (supported_texts and not negatives):
            answer = "yes"
        else:
            answer = answer_for_rating(rating)
    else:
        answer = answer_for_rating(rating)
    return SynopsisOut(
        rating=rating,
        summary=summary,
        tldr=tldr,
        answer=answer,
        caveats=caveats,
        unchecked_count=unchecked_count,
        counts=counts,
    )


def _phrased_parts(phrased) -> tuple[str | None, str | None]:
    if isinstance(phrased, str):
        return phrased, None
    if isinstance(phrased, dict):
        summary = phrased.get("summary")
        tldr = phrased.get("tldr")
        return (
            summary.strip() if isinstance(summary, str) and summary.strip() else None,
            tldr.strip() if isinstance(tldr, str) and tldr.strip() else None,
        )
    return None, None


class LLMSynopsis:
    """One short phrasing call. Returns None on any failure so the caller keeps the fallback."""

    def __init__(self, client, model: str, meter, settings):
        self.client = client
        self.model = model
        self.meter = meter
        self.settings = settings

    def __call__(self, brief: str, rating: str) -> dict | None:
        try:
            result = self.client.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": _INSTRUCTION.format(rating=rating)},
                    {"role": "user", "content": brief},
                ],
                temperature=0.0,
                max_tokens=220,
                timeout=3.0,
            )
        except (LLMError, Exception):  # noqa: BLE001
            return None
        charge_tokens(
            self.meter,
                model=result.model,
                provider=result.provider,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost=estimate_cost(self.settings, result.model, result.input_tokens, result.output_tokens),
            )
        try:
            payload = loads_object(result)
        except (LLMError, ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        summary = payload.get("summary")
        tldr = payload.get("tldr")
        if not isinstance(summary, str) or not summary.strip():
            return None
        return {
            "summary": summary.strip(),
            "tldr": tldr.strip() if isinstance(tldr, str) else "",
        }
