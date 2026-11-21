"""Make the draft and the bottom line say the same thing in question mode."""

from __future__ import annotations

import re

from factcheck.questions import parse_stance, stance_of_text
from factcheck.schemas import VerificationOut

_YESNO = re.compile(r"^(was|is|did|were|has|have|do|does|had|are)\b", re.I)
_GUILT = re.compile(r"\b(convicted|was charged|found guilty|pleaded guilty|pled guilty)\b", re.I)
_CAVEAT = re.compile(r"\b(but|however|although|technically)\b", re.I)


def _checked(claim) -> bool:
    return claim.claim_type == "fact" and claim.verdict != "Hypothesis" and claim.effort != "skipped"


def _rating_for(answer: str, caveats: list[str], supported: list) -> str:
    if answer == "not_enough_evidence":
        return "Not enough evidence"
    if answer == "partly":
        return "Mixed"
    if caveats or not supported:
        return "Mostly accurate"
    return "Accurate"


def answer_for_rating(rating: str) -> str:
    return {
        "Not enough evidence": "not_enough_evidence",
        "Mixed": "partly",
        "Inaccurate": "no",
        "Mostly inaccurate": "no",
        "Accurate": "yes",
        "Mostly accurate": "yes",
    }.get(rating, "not_enough_evidence")


def _lead(answer: str) -> str:
    return {"yes": "Yes", "no": "No", "partly": "Partly"}.get(answer, "")


def _starts(text: str, answer: str) -> bool:
    lowered = (text or "").strip().lower()
    if answer == "not_enough_evidence":
        return lowered.startswith("unconfirmed") or lowered.startswith("there wasn't enough evidence")
    word = _lead(answer).lower()
    return lowered.startswith(word) and (len(lowered) == len(word) or not lowered[len(word)].isalpha())


def _tldr(answer: str, draft: str, caveats: list[str], correction: str) -> str:
    if answer == "not_enough_evidence":
        return "There wasn't enough evidence to settle this."
    if correction and answer in {"yes", "no", "partly"}:
        sentence = correction.removeprefix("Correction:").strip()
        lead = _lead(answer)
        if sentence and not _starts(sentence, answer):
            sentence = f"{lead}. {sentence[0].upper() + sentence[1:] if sentence else sentence}"
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        return sentence
    body = " ".join((draft or "").split())
    if body.lower().startswith("unconfirmed:"):
        body = body.split(":", 1)[-1].strip()
    if answer in {"yes", "no"} and not _starts(body, answer):
        body = f"{_lead(answer)}. {body}"
    if answer == "partly" and not _starts(body, "partly"):
        body = f"Partly. {body}"
    if caveats and not _CAVEAT.search(body):
        extra = caveats[0].rstrip(".")
        body = body.rstrip(".") + f", but {extra[0].lower() + extra[1:]}."
    words = body.split()
    if len(words) > 30:
        body = " ".join(words[:30]).rstrip(",;:") + "."
    if body and body[-1] not in ".!?":
        body += "."
    return body


def _summary(answer: str, rating: str, tldr: str, correction: str) -> str:
    if answer == "not_enough_evidence":
        return "Not enough evidence. There wasn't enough evidence to settle this."
    if correction:
        return f"{rating}. {correction if correction.endswith('.') else correction + '.'}"
    return f"{rating}. {tldr}"


def reconcile_answer(result: VerificationOut) -> None:
    """Question mode: one stance shared by the draft, the tldr, and the rating."""
    if result.synopsis is None:
        return
    if result.mode != "question" or result.draft_answer is None:
        result.synopsis.answer = {
            "Not enough evidence": "not_enough_evidence",
            "Mixed": "partly",
            "Inaccurate": "no",
            "Mostly inaccurate": "no",
        }.get(result.synopsis.rating, "yes" if result.synopsis.rating in {"Accurate", "Mostly accurate"} else "not_enough_evidence")
        return
    draft = result.draft_answer
    question = result.input_summary or ""
    stance = draft.stance or stance_of_text(draft.text) or parse_stance("", draft.text)
    facts = [claim for claim in result.claims if _checked(claim)]
    contradicted = [claim for claim in facts if claim.verdict == "Contradicted"]
    conflicting = [claim for claim in facts if claim.verdict == "Conflicting"]
    supported = [claim for claim in facts if claim.verdict == "Supported"]
    record = " ".join((source.quote or source.snippet or "") for claim in facts for source in claim.sources)
    criminal = bool(_YESNO.match(question.strip())) and bool(re.search(r"\bcriminal\b", question, re.I))
    guilt_in_record = bool(_GUILT.search(record))
    correction = ""
    if conflicting or (supported and contradicted):
        answer = "not_enough_evidence"
    elif contradicted:
        quote = next((source.quote for claim in contradicted for source in claim.sources if source.quote), "")
        correction = "Correction: the cited evidence says " + (quote or contradicted[0].text).strip().rstrip(".") + "."
        if _YESNO.match(question.strip()) and stance == "yes":
            answer = "no"
        elif _YESNO.match(question.strip()) and stance == "no":
            answer = "yes"
        else:
            answer = "partly"
    elif stance in {"yes", "no", "partly"} and (supported or (criminal and stance == "no" and not guilt_in_record)):
        answer = stance
    else:
        answer = "not_enough_evidence"
    caveats = [claim.text for claim in supported if _CAVEAT.search(claim.text)]
    if not caveats and answer in {"yes", "no"} and _CAVEAT.search(draft.text or ""):
        caveat = _CAVEAT.split(draft.text, maxsplit=1)
        if len(caveat) == 3 and caveat[2].strip():
            piece = re.sub(r"^(but|however|although|technically)\s+", "", caveat[2].strip(), flags=re.I)
            if piece:
                caveats = [piece]
    for item in result.synopsis.caveats:
        if item not in caveats:
            caveats.append(item)
    rating = _rating_for(answer, caveats if answer in {"yes", "no"} else [], supported)
    tldr = _tldr(answer, draft.text, caveats, correction)
    result.synopsis.answer = answer
    result.synopsis.rating = rating
    result.synopsis.tldr = tldr
    result.synopsis.summary = _summary(answer, rating, tldr, correction)
    kept = [item for item in caveats if item and item not in result.synopsis.caveats]
    result.synopsis.caveats.extend(kept[:2])
    _align_draft(draft, answer, correction)


def _align_draft(draft, answer: str, correction: str) -> None:
    text = (draft.text or "").strip()
    if correction and not text.lower().startswith("correction:"):
        text = f"{correction}\n\n{text}"
    elif answer == "not_enough_evidence" and not text.lower().startswith("unconfirmed"):
        text = f"Unconfirmed: {text}"
    elif answer in {"yes", "no", "partly"} and not _starts(text, answer):
        text = f"{_lead(answer)}. {text}"
    draft.text = text
    draft.stance = answer
    draft.correction = correction
