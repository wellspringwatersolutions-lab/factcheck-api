"""A short bottom line built only from claim verdicts and the quotes already cited.

The rating is deterministic and follows evidence quality. One optional language-model
call may phrase that rating. If the call fails, runs long, or adds a fact that is
not already in the brief, the deterministic sentence is what the report shows.
"""

from __future__ import annotations

import logging
import re
import time

from factcheck.llm import LLMError, estimate_cost, loads_object
from factcheck.request_ctx import charge_tokens
from factcheck.schemas import ClaimOut, SynopsisCounts, SynopsisOut
from factcheck.textutil import split_sentences

logger = logging.getLogger("factcheck")

_RATINGS = (
    "Accurate",
    "Mostly accurate",
    "Mixed",
    "Mostly inaccurate",
    "Inaccurate",
    "Not enough evidence",
    "Partly verified",
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
_INSTRUCTION = """You write the bottom line from the verdict list only. You do not decide verdicts and you do not add facts.
The rating is already fixed. summary must start with "{rating}" and be 2 to 4 complete sentences.
tldr is one complete sentence of at most 35 words. It leads with a claim marked Supported or likely_true, preferably a denial.
The sentences after the tldr state other claims one per sentence. Never join two claims with ", and".
A Supported claim is stated plainly. A likely_true claim is "Reputable sources report that ...".
State a Supported claim as true. A likely_true or Hypothesis claim is only probably true: phrase it as "Reputable sources report ...", not as a confirmed fact.
A Contradicted claim is disputed: say the sources contradict or dispute it. Do not state it as true.
Mention an Unverified claim only to say it could not be confirmed, and never lead with it.
Address every claim in the list. Never skip a Contradicted or Conflicting claim.
Do not cut a sentence off in the middle. Use a shorter confirmed claim instead of trimming a long one.
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
    good = [claim for claim in facts if claim.verdict in {"Supported", "likely_true"}]
    if good and not any(claim.verdict in {"Contradicted", "Conflicting"} for claim in facts) and len(good) * 2 >= len(facts):
        if unverified:
            return "Partly verified", caveats, unchecked_count
    if not decided or len(unverified) > len(decided):
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
        rating = "Partly verified" if (contradicted or conflicting) else "Mostly accurate"
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


def fallback_summary(
    rating: str,
    question: str | None,
    claims: list[ClaimOut],
    caveats: list[str],
    unchecked_count: int,
    topics: list[str] | None = None,
) -> str:
    facts = _checked_facts(claims)
    supported = [claim.text for claim in facts if claim.verdict in {"Supported", "likely_true"}]
    contradicted = [claim.text for claim in facts if claim.verdict == "Contradicted"]
    unverified = [claim for claim in facts if claim.verdict in {"Unverified", "Unsupported", None}]
    if unverified and rating in {"Partly verified", "Mostly accurate", "Accurate"}:
        sentence = _headline_text(facts) or _template_sentence(rating, claims)
        count = len(unverified)
        word = "claim" if count == 1 else "claims"
        parts = [f"{rating}.", sentence, *_extra_confirmed(claims, sentence), f"{count} {word} could not be confirmed."]
        return polish_prose(" ".join(parts))
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
        headline = _headline_text(facts)
        direct = _direct(question, main or supported[:1])
        body = [headline] if headline else (main or supported[:1])
        if direct and body:
            sentence = f"{direct}, " + ", and ".join(_soft(text) for text in body)
        else:
            sentence = "; ".join(text.strip().rstrip(".") for text in body)
        if caveat_bits and sentence.strip():
            sentence = sentence.rstrip(".") + ", but " + " and ".join(caveat_bits)
        elif caveat_bits:
            sentence = caveat_bits[0][:1].upper() + caveat_bits[0][1:]
    if not sentence.strip():
        sentence = _template_sentence(rating, claims)
    sentence = sentence.strip().rstrip(".") + "."
    notes = []
    if unverified and rating != "Not enough evidence":
        count = len(unverified)
        word = "claim" if count == 1 else "claims"
        notes.append(f"{count} {word} could not be confirmed.")
    if unchecked_count:
        word = "claim" if unchecked_count == 1 else "claims"
        verb = "was" if unchecked_count == 1 else "were"
        notes.append(f"{unchecked_count} {word} {verb} not checked.")
    return polish_prose(" ".join(part for part in (f"{rating}.", sentence, *notes) if part))


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
    """Verdict list only. Quotes are not given to the phrasing model."""
    del question
    lines = [f"Rating: {rating}"]
    for claim in _checked_facts(claims):
        lines.append(f"{claim.verdict}: {claim.text}")
    return "\n".join(lines)


def claim_counts(claims: list[ClaimOut]) -> SynopsisCounts:
    counts = SynopsisCounts(details={"likely_true": 0, "speculative": 0, "contradicted": 0})
    for claim in claims:
        if _unchecked(claim):
            counts.not_checked += 1
        elif claim.claim_type == "opinion":
            counts.opinion += 1
        elif claim.verdict == "Supported":
            counts.supported += 1
        elif claim.verdict == "likely_true":
            counts.hypothesis += 1
            counts.details["likely_true"] += 1
        elif claim.verdict == "Hypothesis":
            counts.hypothesis += 1
            counts.details["speculative"] += 1
        elif claim.verdict == "Contradicted":
            counts.conflicting += 1
            counts.details["contradicted"] += 1
        elif claim.verdict in {"Conflicting", "Unsupported"}:
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
    if words[0].lower().strip(",;:") in {"but", "and", "or", "also"}:
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


def polish_prose(text: str) -> str:
    """Drop a leading period or 'Also,' and collapse '. .'."""
    cleaned = " ".join((text or "").split())
    cleaned = re.sub(r"\.\s*\.", ".", cleaned)
    cleaned = re.sub(r"^(?:\.\s*)+", "", cleaned).strip()
    cleaned = re.sub(r"(?i)^also,\s*", "", cleaned).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


_DOUBLE_QUOTE = re.compile(r'["“]([^"”]{1,80})["”]')
_DENIAL = re.compile(r"\b(never|not|no|did not|n't)\b", re.I)
_VERDICT_RANK = {"Supported": 0, "likely_true": 1}


def _claim_words(text: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text or "")}


def _trim_words(text: str, limit: int) -> str:
    words = [word for word in (text or "").split() if word]
    return " ".join(words[:limit])


def _valid_topic(phrase: str, claim: str) -> bool:
    """A noun phrase of at most 8 words, each one a whole word from the claim."""
    words = (phrase or "").split()
    if not words or len(words) > 8:
        return False
    if re.search(r"\band\b", phrase or "", re.I) and len(words) > 6:
        return False
    known = _claim_words(claim)
    cleaned = [re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", word) for word in words]
    return all(word and word.lower() in known for word in cleaned)


def _topic(text: str) -> str:
    """A short topic. Quotation marks are \" and curly quotes, never apostrophes."""
    quoted = _DOUBLE_QUOTE.findall(text or "")
    if quoted:
        return _trim_words(quoted[0].strip(), 8)
    return _first_clause(text)


def _first_clause(text: str) -> str:
    cleaned = " ".join((text or "").split()).strip().rstrip(".!?")
    clause = re.split(r"[,;:]", cleaned, maxsplit=1)[0].strip()
    return _trim_words(clause, 8)


def _unconfirmed_sentence(texts: list[str], topics: list[str] | None = None) -> str:
    """One sentence per claim. Two fragments are never glued together with and."""
    sentences = []
    for index, text in enumerate(texts[:2]):
        if not text or not text.strip():
            continue
        supplied = topics[index] if topics and index < len(topics) else ""
        topic = supplied.strip() if supplied and _valid_topic(supplied, text) else _topic(text)
        if not topic:
            continue
        sentences.append(f"Claims about {topic} could not be confirmed.")
    return " ".join(sentences)


def _voiced(text: str, verdict: str | None) -> str:
    """Supported claims are plain. likely_true is only what reputable sources report."""
    sentence = _as_sentence(text)
    if verdict not in {"likely_true", "Hypothesis"}:
        return sentence
    body = sentence[:-1] if sentence.endswith(".") else sentence
    if body.lower().startswith("reputable sources report"):
        return sentence
    lowered = body[0].lower() + body[1:] if body else body
    return f"Reputable sources report that {lowered}."


def _extra_confirmed(claims: list[ClaimOut], headline: str, limit: int = 2) -> list[str]:
    """Up to two other confirmed claims, each its own sentence. Never joined with ', and'."""
    head = headline.rstrip(".!?").strip().lower()
    extras: list[str] = []
    for claim in _checked_facts(claims):
        if claim.verdict not in {"Supported", "likely_true"}:
            continue
        sentence = _voiced(claim.text, claim.verdict)
        if sentence.rstrip(".!?").strip().lower() == head:
            continue
        if "reputable sources report that " + _as_sentence(claim.text).rstrip(".").lower() == head:
            continue
        extras.append(sentence)
        if len(extras) >= limit:
            break
    return extras


def _as_sentence(text: str) -> str:
    sentence = " ".join((text or "").split()).strip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _headline_text(claims: list[ClaimOut] | None, limit: int = 35) -> str:
    """Full text of a denial, otherwise the strongest confirmed claim. Never trimmed.

    A claim longer than `limit` words is skipped in favor of another confirmed claim.
    If every confirmed claim is longer, the preferred one is kept whole.
    """
    facts = [claim for claim in _checked_facts(claims or []) if claim.verdict in {"Supported", "likely_true"}]
    denials = [claim for claim in facts if _DENIAL.search(claim.text or "")]
    others = [claim for claim in facts if claim not in denials]
    denials.sort(key=_weight, reverse=True)
    others.sort(key=_weight, reverse=True)
    ordered = denials + others
    fallback = ""
    for claim in ordered:
        sentence = _voiced(claim.text, claim.verdict)
        if not complete_sentence(sentence):
            continue
        if not fallback:
            fallback = sentence
        if len(sentence.split()) <= limit:
            return sentence
    return fallback


def _trim_sentence(text: str, limit: int = 35) -> str:
    cleaned = " ".join((text or "").split()).strip().rstrip(".!?")
    words = cleaned.split()
    if len(words) > limit:
        words = words[:limit]
    while words and words[-1].lower().strip(",;:") in _DANGLING:
        words.pop()
    if len(words) < 4:
        return ""
    sentence = " ".join(words).rstrip(",;:")
    if sentence[0].islower():
        sentence = sentence[0].upper() + sentence[1:]
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence if complete_sentence(sentence) else ""


def _shares_conclusion(sentence: str, headline: str) -> bool:
    left = _claim_words(sentence)
    right = [word for word in _claim_words(headline) if len(word) >= 4]
    return any(word in left for word in right[:8])


def fallback_tldr(rating: str, summary: str, claims: list[ClaimOut] | None = None) -> str:
    """One whole sentence. The confirmed claim leads, even when that sentence is long."""
    if rating == "Not enough evidence":
        return _ABSTAIN
    headline = _headline_text(claims)
    if headline:
        return headline
    parts = split_sentences(summary or "")
    for part in parts:
        if part.rstrip(".").lower() == rating.lower():
            continue
        if len(part.split()) > 35:
            continue
        if complete_sentence(part) and not part.lower().startswith("claims about"):
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


def _summary_sentence_ok(sentence: str, rating: str) -> bool:
    if sentence.rstrip(".").lower() == (rating or "").lower():
        return True
    if re.fullmatch(r"\d+ claims? could not be confirmed\.", sentence or ""):
        return True
    return complete_sentence(sentence)


def _phrasing_ok(summary: str, tldr: str, rating: str, claims: list[ClaimOut] | None) -> bool:
    """Accept a model bottom line only when it is whole sentences drawn from the verdicts."""
    tldr = (tldr or "").strip()
    summary = (summary or "").strip()
    claims = claims or []
    if not complete_sentence(tldr) or len(tldr.split()) > 35:
        return False
    if tldr.lower().startswith("claims about"):
        return False
    if not summary.lower().startswith((rating or "").lower()):
        return False
    sentences = split_sentences(summary)
    if not 2 <= len(sentences) <= 5:
        return False
    if not all(_summary_sentence_ok(sentence, rating) for sentence in sentences):
        return False
    settled = set()
    probable = set()
    disputed = set()
    unverified_only = set()
    allowed_text = []
    for claim in _checked_facts(claims):
        words = _marks(claim.text)
        if claim.verdict == "Supported":
            settled |= words
            allowed_text.append(claim.text or "")
        elif claim.verdict in {"likely_true", "Hypothesis"}:
            probable |= words
            allowed_text.append(claim.text or "")
        elif claim.verdict == "Contradicted":
            disputed |= words
            allowed_text.append(claim.text or "")
        elif claim.verdict in {"Unverified", "Unsupported", None}:
            unverified_only |= words
    confirmed = settled | probable
    unverified_only -= confirmed
    probable -= settled
    disputed -= confirmed
    tldr_words = _claim_words(tldr)
    if confirmed and not (tldr_words & confirmed):
        return False
    if tldr_words & unverified_only:
        return False
    if not _voice_ok(tldr, probable, disputed):
        return False
    for number in _NUMBER.findall(tldr):
        if number not in " ".join(allowed_text):
            return False
    content = sentences[1:] if sentences[0].rstrip(".").lower() == (rating or "").lower() else sentences
    if confirmed and content and not (_claim_words(content[0]) & confirmed):
        return False
    if content and re.search(r"could not be confirmed|unconfirmed", content[0], re.I):
        return False
    for sentence in sentences:
        leaked = _claim_words(sentence) & unverified_only
        if leaked and not re.search(r"could not be confirmed|unconfirmed|not confirmed", sentence, re.I):
            return False
        if not _voice_ok(sentence, probable, disputed):
            return False
        if ", and " in sentence.lower() and _uses_two_claims(sentence, claims):
            return False
    summary_words = _claim_words(summary)
    for claim in _checked_facts(claims):
        # A disputed claim must be addressed, not silently dropped from the bottom line.
        if claim.verdict in {"Contradicted", "Conflicting"} and _marks(claim.text) and not (_marks(claim.text) & summary_words):
            return False
    return True


def _uses_two_claims(sentence: str, claims: list[ClaimOut]) -> bool:
    marks = _marks(sentence)
    hits = 0
    for claim in _checked_facts(claims):
        if claim.verdict not in {"Supported", "likely_true", "Hypothesis", "Contradicted"}:
            continue
        if _marks(claim.text) & marks:
            hits += 1
    return hits >= 2


def _marks(text: str) -> set[str]:
    """Words and numbers distinctive enough to show which claim a sentence is using."""
    return {word for word in _claim_words(text) if len(word) >= 5} | set(_NUMBER.findall(text or ""))


def _voice_ok(sentence: str, probable: set[str], disputed: set[str]) -> bool:
    """likely_true is probably true. A contradicted claim stays disputed after it is renamed conflicting."""
    words = _marks(sentence)
    if words & probable and "reputable sources report" not in (sentence or "").lower():
        return False
    if words & disputed and not re.search(r"contradict|disput", sentence or "", re.I):
        return False
    return True


def _accept_tldr(text: str, corpus: str, rating: str, claims: list[ClaimOut] | None = None) -> bool:
    cleaned = (text or "").strip()
    if not complete_sentence(cleaned) or len(cleaned.split()) > 35:
        return False
    headline = _headline_text(claims)
    if headline:
        if cleaned.lower().startswith("claims about"):
            return False
        if not _shares_conclusion(cleaned, headline):
            return False
    allowed = corpus
    if rating == "Not enough evidence":
        if not _says_not_enough(cleaned):
            return False
        allowed = f"{corpus}\n{_ABSTAIN}"
    return grounded(cleaned, allowed)


def _usable_topics(phrased, claims: list[ClaimOut]) -> list[str] | None:
    if not isinstance(phrased, dict) or not isinstance(phrased.get("topics"), list):
        return None
    texts = [claim.text for claim in _checked_facts(claims) if claim.verdict in {"Unverified", "Unsupported", None}]
    if not texts:
        return None
    chosen = []
    used_model = False
    for index, text in enumerate(texts[:2]):
        raw = phrased["topics"][index] if index < len(phrased["topics"]) else ""
        phrase = raw.strip() if isinstance(raw, str) else ""
        if phrase and _valid_topic(phrase, text):
            chosen.append(" ".join(phrase.split()))
            used_model = True
        else:
            chosen.append(_topic(text))
    return chosen if used_model else None


_GLUED_TOPICS = re.compile(r"Claims about .{1,120} and .{1,120} could not be confirmed", re.I)


def _summary_leads(summary: str, rating: str, claims: list[ClaimOut]) -> bool:
    cleaned = (summary or "").strip()
    body = cleaned[len(rating):].lstrip(" .") if cleaned.lower().startswith(rating.lower()) else cleaned
    if not complete_sentence(body) or _GLUED_TOPICS.search(body):
        return False
    headline = _headline_text(claims)
    if not headline:
        return True
    lead = split_sentences(body)[:1]
    sentence = lead[0] if lead else ""
    if sentence.lower().startswith("claims about"):
        return False
    return _shares_conclusion(sentence, headline)


def build_synopsis(
    claims: list[ClaimOut],
    *,
    question: str | None = None,
    phrasing=None,
    deadline: float | None = None,
) -> SynopsisOut:
    rating, caveats, unchecked_count = rate_claims(claims)
    counts = claim_counts(claims)
    # A contradiction is stated from the cited quote. Do not let a phrasing call rewrite it.
    contradicted = any(claim.verdict == "Contradicted" for claim in claims)
    remaining = None if deadline is None else deadline - time.monotonic()
    phrased = None
    if phrasing is not None and not contradicted and (remaining is None or remaining >= 3.5):
        brief = _brief(rating, question, claims)
        try:
            phrased = phrasing(brief, rating, claims)
        except TypeError:
            try:
                phrased = phrasing(brief, rating)
            except Exception:  # noqa: BLE001 - a failed phrasing call must not fail the report
                phrased = None
        except Exception:  # noqa: BLE001 - a failed phrasing call must not fail the report
            phrased = None
    summary_text, tldr_text = _phrased_parts(phrased)
    if not _phrasing_ok(summary_text or "", tldr_text or "", rating, claims):
        summary_text, tldr_text = None, None
    summary = fallback_summary(rating, question, claims, caveats, unchecked_count)
    tldr = fallback_tldr(rating, summary, claims)
    if summary_text and tldr_text:
        summary = summary_text.strip()
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
    summary = polish_prose(summary)
    tldr = polish_prose(tldr)
    if not complete_sentence(tldr):
        tldr = _template_sentence(rating, claims)
    if re.match(r"(?i)^also\b", summary) or summary.startswith("."):
        summary = f"{rating}. {_template_sentence(rating, claims)}"
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
    """Chutes writes the bottom line. A failed check retries once on the other model."""

    def __init__(self, client, model: str, meter, settings):
        self.client = client
        self.model = model
        self.meter = meter
        self.settings = settings
        primary = getattr(settings, "judge_b_model", None) or model
        secondary = getattr(settings, "judge_a_model", None) or model
        self.models = []
        for name in (primary, secondary):
            if name and name not in self.models:
                self.models.append(name)

    def __call__(self, brief: str, rating: str, claims: list[ClaimOut] | None = None) -> dict | None:
        for model in self.models:
            payload = self._once(model, brief, rating)
            if payload is None:
                logger.warning("synopsis model=%s failed", model)
                continue
            if _phrasing_ok(payload["summary"], payload["tldr"], rating, claims):
                logger.warning("synopsis model=%s", model)
                return payload
            logger.warning("synopsis model=%s rejected", model)
        return None

    def _once(self, model: str, brief: str, rating: str) -> dict | None:
        try:
            result = self.client.complete(
                model=model,
                messages=[
                    {"role": "system", "content": _INSTRUCTION.format(rating=rating)},
                    {"role": "user", "content": brief},
                ],
                temperature=0.0,
                max_tokens=280,
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
        if not isinstance(summary, str) or not isinstance(tldr, str):
            return None
        return {"summary": summary.strip(), "tldr": tldr.strip()}
