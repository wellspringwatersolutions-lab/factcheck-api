"""Claimify-style extraction. Opinions are kept and marked, not dropped."""

from __future__ import annotations

import re
from dataclasses import dataclass

from factcheck.categories import classify_category, is_hypothesis, is_math, normalize_claim
from factcheck.cleanup import clean_claim_text, prepare_answer
from factcheck.llm import LLMError, estimate_cost, loads_object
from factcheck.textutil import is_time_sensitive, split_sentences
from factcheck.types import ExtractedClaim, UsageMeter

_PRONOUN = re.compile(r"\b(He|She|They|It|This|That)\b(?!')")
_FILLER_ONLY = re.compile(r"^(of course|that said|well|anyway|look)[.!]?$", re.I)
_BAD_ANTECEDENT = {"Of", "It", "The", "A", "An", "That", "This", "And", "But", "It's", "That's"}
_ATTRIBUTION = re.compile(
    r"^(?P<who>[A-Z][^.]{1,60}?) (?:said|says|stated|wrote|claimed|announced)\b|"
    r"\baccording to (?P<who2>[^,.]{2,80})",
    re.I,
)
_OPINION = re.compile(
    r"\b(i think|i believe|in my opinion|arguably|should|shouldn't|probably|perhaps|"
    r"is a disaster|ridiculous|everyone knows)\b",
    re.I,
)


@dataclass
class Sentence:
    id: int
    text: str
    context: str


def build_sentences(text: str | None, claims: list[str] | None) -> list[Sentence]:
    if claims:
        context = " ".join((text or "").split())[:500]
        return [Sentence(i, claim.strip(), context) for i, claim in enumerate(claims) if claim and claim.strip()]
    parts = split_sentences(text or "")
    sentences = []
    for index, part in enumerate(parts):
        window = " ".join(parts[max(0, index - 1) : index + 2])
        sentences.append(Sentence(index, part, window))
    return sentences


def _attribution(text: str) -> str | None:
    match = _ATTRIBUTION.search(text)
    if not match:
        return None
    return (match.group("who") or match.group("who2") or "").strip() or None


def heuristic_label(text: str) -> str:
    if _FILLER_ONLY.match(text.strip()):
        return "drop"
    if text.endswith("?") or len(text.split()) < 4:
        return "drop"
    if is_math(text):
        return "fact"
    if _OPINION.search(text):
        return "opinion"
    return "fact"


def resolve_pronouns(text: str, context: str) -> tuple[str, bool]:
    """Keep the original words. Do not rewrite contractions or filler like 'Of course' or 'That said'."""
    if re.search(r"\bThat said\b", text) or re.search(r"\bOf course\b", text):
        return text, False
    match = _PRONOUN.search(text)
    if not match:
        return text, False
    antecedent = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", context or "")
    if not antecedent or antecedent.group(1) in _BAD_ANTECEDENT or antecedent.group(1) == match.group(1):
        return text, False
    if match.start() > 0 and text[match.start() - 1] in {"'", "’"}:
        return text, False
    return text, False


def heuristic_extract(text: str | None, claims: list[str] | None) -> list[ExtractedClaim]:
    prepared = prepare_answer(text) if text else None
    prepared_claims = [clean_claim_text(claim) for claim in claims] if claims else None
    found: list[ExtractedClaim] = []
    for sentence in build_sentences(prepared, prepared_claims):
        label = heuristic_label(sentence.text)
        if label == "drop":
            continue
        resolved, drop = resolve_pronouns(sentence.text, sentence.context)
        if drop:
            continue
        claim = ExtractedClaim(
            text=resolved,
            claim_type="opinion" if label == "opinion" else "fact",
            claim_category=classify_category(resolved, "opinion" if label == "opinion" else "fact"),
            time_sensitive=is_time_sensitive(resolved),
            attribution=_attribution(resolved),
            hypothesis=is_hypothesis(resolved),
        )
        found.append(normalize_claim(claim))
    return found


def parse_selection(payload: dict, count: int) -> dict[int, str]:
    labels: dict[int, str] = {}
    for item in payload.get("sentences") or []:
        try:
            index = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        label = item.get("label") or "drop"
        if label not in {"fact", "opinion", "drop"}:
            label = "drop"
        labels[index] = label
    return {index: labels.get(index, "drop") for index in range(count)}


def _faithful(original: str, rewritten: str) -> bool:
    if not rewritten or rewritten == original:
        return True
    if "Of's" in rewritten or re.search(r"\bIt said\b", rewritten):
        return False
    from factcheck.textutil import overlap

    return overlap(original, rewritten) >= 0.55


def parse_disambiguation(payload: dict, originals: dict[int, str] | None = None) -> list[dict]:
    rows = []
    for item in payload.get("sentences") or []:
        text = (item.get("text") or "").strip()
        index = int(item.get("id", -1))
        original = (originals or {}).get(index)
        if original and not _faithful(original, text):
            text = original
        rows.append(
            {
                "id": index,
                "text": text,
                "drop": bool(item.get("drop")),
                "claim_type": "opinion" if item.get("claim_type") == "opinion" else "fact",
                "attribution": item.get("attribution"),
            }
        )
    return rows


def parse_decomposition(payload: dict) -> list[ExtractedClaim]:
    claims = []
    for item in payload.get("claims") or []:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        claim_type = "opinion" if item.get("claim_type") == "opinion" else "fact"
        claim = ExtractedClaim(
            text=text,
            claim_type=claim_type,
            claim_category=item.get("claim_category") or classify_category(text, claim_type),
            time_sensitive=bool(item.get("time_sensitive")),
            scientific=bool(item.get("scientific")),
            attribution=item.get("attribution"),
            hypothesis=bool(item.get("hypothesis")) or is_hypothesis(text),
        )
        claims.append(normalize_claim(claim))
    return claims


_SELECTION = """STAGE: selection
You are the selection stage of a Claimify-style extractor.
Label each sentence fact, opinion, or drop.
Keep opinions and contested narratives; do not drop them.
Drop greetings, questions, and fragments.
fact means a verifiable claim, including an attributed speech act ("X said Y").
Return JSON {"sentences": [{"id": 0, "label": "fact|opinion|drop", "reason": "..."}]}"""

_DISAMBIG = """STAGE: disambiguation
Resolve a standalone pronoun only by copying an antecedent that appears in the context.
Never rewrite the original words. Keep "Of course", "It's", and "That said" exactly.
Do not turn "It's" into another word, and do not turn "That said" into "It said".
If you cannot resolve something without changing the wording, keep the original sentence.
Do not change fact into opinion or the reverse.
Preserve attribution.
Return JSON {"sentences": [{"id": 0, "text": "...", "drop": false, "claim_type": "fact|opinion", "attribution": null}]}"""

_DECOMPOSE = """STAGE: decomposition
Split the text into atomic claims that stand on their own.
Attributed claims must say who said it, not state the content as an unattributed fact.
Mark time_sensitive, scientific, hypothesis (prediction, novel conjecture, or business idea), and claim_category T1-T8.
T1 math, T2 scientific law, T3 well-supported theory, T4 medical or empirical, T5 historical, T6 expert consensus, T7 eyewitness, T8 opinion.
Return JSON {"claims": [{"text": "...", "claim_type": "fact", "claim_category": "T5", "time_sensitive": false, "scientific": false, "hypothesis": false, "attribution": null}]}"""


class LiveExtractor:
    def __init__(self, client, model: str, meter: UsageMeter, settings):
        self.client = client
        self.model = model
        self.meter = meter
        self.settings = settings

    def _ask(self, instruction: str, user: str) -> dict:
        result = self.client.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        self.meter.add_tokens(
            model=result.model,
            provider=result.provider,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost=estimate_cost(self.settings, result.model, result.input_tokens, result.output_tokens),
        )
        return loads_object(result)

    def extract(self, text: str | None, claims: list[str] | None) -> tuple[list[ExtractedClaim], list[str]]:
        sentences = build_sentences(prepare_answer(text) if text else None, [clean_claim_text(claim) for claim in claims] if claims else None)
        if not sentences:
            return [], []
        try:
            payload = "\n".join(f"[{item.id}] {item.text}\ncontext: {item.context}" for item in sentences)
            labels = parse_selection(self._ask(_SELECTION, payload), len(sentences))
            kept = [item for item in sentences if labels[item.id] != "drop"]
            if not kept:
                return [], []
            disambiguated = parse_disambiguation(
                self._ask(_DISAMBIG, "\n".join(f"[{item.id}] {item.text}\ncontext: {item.context}" for item in kept)),
                {item.id: item.text for item in kept},
            )
            by_id = {row["id"]: row for row in disambiguated}
            resolved_lines = []
            for item in kept:
                row = by_id.get(item.id)
                if row is None:
                    resolved_lines.append(f"[{item.id}] ({labels[item.id]}) {item.text}")
                    continue
                if row["drop"] or not row["text"]:
                    continue
                resolved_lines.append(f"[{item.id}] ({row['claim_type']}) {row['text']}")
            if not resolved_lines:
                return [], []
            claims_out = parse_decomposition(self._ask(_DECOMPOSE, "\n".join(resolved_lines)))
            if not claims_out:
                raise ValueError("decomposition returned no claims")
            return claims_out, []
        except (LLMError, ValueError, KeyError, TypeError) as exc:
            return heuristic_extract(text, claims), [f"extraction_fallback:{exc.__class__.__name__}"]


class HeuristicExtractor:
    def extract(self, text: str | None, claims: list[str] | None) -> tuple[list[ExtractedClaim], list[str]]:
        return heuristic_extract(text, claims), []
