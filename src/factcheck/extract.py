"""Claimify-style extraction. Opinions are kept and marked, not dropped."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from factcheck.categories import classify_category, is_hypothesis, is_math, normalize_claim
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

from factcheck.cleanup import chunk_paragraphs, clean_claim_text, prepare_answer
from factcheck.llm import LLMError, estimate_cost, loads_object
from factcheck.request_ctx import bind_context, charge_tokens
from factcheck.textutil import is_time_sensitive, split_sentences
from factcheck.types import ExtractedClaim, UsageMeter

logger = logging.getLogger("factcheck")
_EXTRACT_BUDGET = 15.0

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


_CLAUSE_VERB = re.compile(
    r"\b(is|are|was|were|be|been|has|have|had|do|does|did|will|can|may|must|shall)\b|\b[a-z]{4,}(?:ed|ing)\b",
    re.I,
)


def _is_clause(text: str) -> bool:
    words = text.split()
    return len(words) >= 5 and bool(_CLAUSE_VERB.search(text))


def split_compounds(sentence: str) -> list[str]:
    """Split on semicolons, and on 'and' only when both sides are clauses."""
    pieces: list[str] = []
    for chunk in re.split(r";\s*", sentence or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in re.split(r"\s+\band\b\s+", chunk, flags=re.I) if part.strip()]
        if len(parts) >= 2 and all(_is_clause(part) for part in parts):
            pieces.extend(parts)
        else:
            pieces.append(chunk)
    return pieces or ([sentence.strip()] if sentence and sentence.strip() else [])


def build_sentences(text: str | None, claims: list[str] | None) -> list[Sentence]:
    if claims:
        context = " ".join((text or "").split())[:500]
        return [Sentence(i, claim.strip(), context) for i, claim in enumerate(claims) if claim and claim.strip()]
    parts: list[str] = []
    for sentence in split_sentences(text or ""):
        parts.extend(split_compounds(sentence))
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
        span = ""
        if isinstance(item, str):
            text = item.strip()
            claim_type = "opinion" if _OPINION.search(text) else "fact"
            category = classify_category(text, claim_type)
            attribution = _attribution(text)
            time_sensitive = is_time_sensitive(text)
            scientific = False
            hypothesis = is_hypothesis(text)
        elif isinstance(item, dict):
            text = (item.get("text") or "").strip()
            span = (item.get("span") or "").strip()
            claim_type = "opinion" if item.get("claim_type") == "opinion" else "fact"
            category = item.get("claim_category") or classify_category(text, claim_type)
            attribution = item.get("attribution")
            time_sensitive = bool(item.get("time_sensitive"))
            scientific = bool(item.get("scientific"))
            hypothesis = bool(item.get("hypothesis")) or is_hypothesis(text)
        else:
            continue
        if not text:
            continue
        claim = ExtractedClaim(
            text=text,
            claim_type=claim_type,
            claim_category=category,
            time_sensitive=time_sensitive,
            scientific=scientific,
            attribution=attribution,
            hypothesis=hypothesis,
            span=span,
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

_DECOMPOSE = """Pick up to 6 checkable key points. Use the same points for the same text.
Prefer specific claims: named programs, numbers, dates, and official guidance. Drop vague wrap-up lines.
Skip the user's question, headings, and meta commentary such as "breaks down into".
Each point is one checkable claim. span is a verbatim sentence copied from the passage.
Attributed claims must say who said it.
Mark time_sensitive, scientific, hypothesis, and claim_category T1-T8.
T1 math, T2 scientific law, T3 well-supported theory, T4 medical or empirical, T5 historical, T6 expert consensus, T7 eyewitness, T8 opinion.
Return JSON {"points": [{"text": "...", "span": "...", "claim_type": "fact", "claim_category": "T5", "time_sensitive": false, "scientific": false, "hypothesis": false, "attribution": null}]}"""


def _claims_from_result(result) -> list[ExtractedClaim]:
    """Parse one extraction response. Repair trailing commas locally. Never call the model again."""
    import json

    from factcheck.textutil import parse_json_object

    raw = result.content or ""
    snippet = raw[:400]
    if '"points"' in raw and '"claims"' not in raw:
        raw = raw.replace('"points"', '"claims"', 1)
    if not raw.strip():
        logger.warning("extraction ValueError: empty json snippet=%r", snippet)
        raise ValueError("empty json")
    repaired = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        payload = parse_json_object(repaired)
    except (ValueError, TypeError) as exc:
        start = repaired.find("[")
        end = repaired.rfind("]")
        value = None
        if start >= 0 and end > start:
            try:
                value = json.loads(repaired[start : end + 1])
            except json.JSONDecodeError:
                value = None
        if not isinstance(value, list):
            logger.warning("extraction ValueError: %s snippet=%r", exc, snippet)
            raise
        payload = {"claims": value}
    return parse_decomposition(payload)


_CLAIM_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "claims",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "text": {"type": "string"},
                            "claim_type": {"type": "string", "enum": ["fact", "opinion"]},
                            "claim_category": {"type": "string"},
                            "time_sensitive": {"type": "boolean"},
                            "scientific": {"type": "boolean"},
                            "hypothesis": {"type": "boolean"},
                            "attribution": {"type": ["string", "null"]},
                        },
                        "required": ["text", "claim_type", "claim_category", "time_sensitive", "scientific", "hypothesis", "attribution"],
                    },
                }
            },
            "required": ["claims"],
        },
    },
}


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
            if not payload:
                raise ValueError("empty json")
            return payload
        except (ValueError, TypeError):
            retry = self.client.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": instruction + "\nReturn one JSON object only."},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
            charge_tokens(
            self.meter,
                model=retry.model,
                provider=retry.provider,
                input_tokens=retry.input_tokens,
                output_tokens=retry.output_tokens,
                cost=estimate_cost(self.settings, retry.model, retry.input_tokens, retry.output_tokens),
            )
            try:
                payload = loads_object(retry)
                if payload:
                    return payload
            except (ValueError, TypeError):
                payload = None
            repaired = re.sub(r",\s*([}\]])", r"\1", retry.content or "")
            try:
                payload = loads_object(type(retry)(repaired, retry.input_tokens, retry.output_tokens, retry.model, retry.provider, retry.family))
            except (ValueError, TypeError):
                payload = None
            if not payload:
                raise ValueError("unrepairable json")
            return payload

    def _one_chunk(self, chunk: str) -> list[ExtractedClaim]:
        result = self.client.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": _DECOMPOSE},
                {"role": "user", "content": chunk},
            ],
            temperature=0.0,
            max_tokens=700,
            timeout=15.0,
            response_format=_CLAIM_SCHEMA,
        )
        charge_tokens(
            self.meter,
            model=result.model,
            provider=result.provider,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost=estimate_cost(self.settings, result.model, result.input_tokens, result.output_tokens),
        )
        claims = _claims_from_result(result)
        if not claims:
            logger.warning("extraction ValueError: decomposition returned no claims snippet=%r", (result.content or "")[:400])
            raise ValueError("decomposition returned no claims")
        return claims

    def extract(self, text: str | None, claims: list[str] | None) -> tuple[list[ExtractedClaim], list[str]]:
        if claims:
            return heuristic_extract(text, claims), []
        prepared = prepare_answer(text) if text else ""
        chunks = chunk_paragraphs(prepared, size=2400)
        if not chunks:
            return [], []
        deadline = time.monotonic() + _EXTRACT_BUDGET
        found: list[ExtractedClaim] = []
        warnings: list[str] = []
        pool = ThreadPoolExecutor(max_workers=min(4, len(chunks)))
        futures = {pool.submit(bind_context(self._one_chunk, chunk)): chunk for chunk in chunks}
        pending = set(futures)
        try:
            try:
                for future in as_completed(futures, timeout=max(0.1, deadline - time.monotonic())):
                    pending.discard(future)
                    chunk = futures[future]
                    try:
                        found.extend(future.result())
                    except (LLMError, ValueError, KeyError, TypeError) as exc:
                        logger.warning("extraction_fallback %s: %s snippet=%r", exc.__class__.__name__, exc, chunk[:240])
                        found.extend(heuristic_extract(chunk, None))
                        warnings.append(f"extraction_fallback:{exc.__class__.__name__}")
            except TimeoutError:
                logger.warning("extraction exceeded %.0fs; using the sentence splitter", _EXTRACT_BUDGET)
                warnings.append("extraction_fallback:TimeoutError")
                for future, chunk in futures.items():
                    if future not in pending:
                        continue
                    if future.done():
                        try:
                            found.extend(future.result())
                            continue
                        except (LLMError, ValueError, KeyError, TypeError):
                            pass
                    found.extend(heuristic_extract(chunk, None))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        seen: set[str] = set()
        unique = []
        for claim in found:
            key = " ".join(claim.text.lower().split())
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(claim)
        return select_key_points(unique, 6), warnings


_SPECIFIC = re.compile(
    r"\d|\b(19|20)\d{2}\b|\b(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"guidance|official|program|ready\.gov|europa|act|order)\b",
    re.I,
)


def select_key_points(claims: list[ExtractedClaim], limit: int = 6) -> list[ExtractedClaim]:
    """Stable order. Specific factual claims come before vague ones."""
    from factcheck.cleanup import priority

    def rank(claim: ExtractedClaim):
        opinion = 1 if claim.claim_type == "opinion" else 0
        specific = 0 if _SPECIFIC.search(claim.text or "") else 1
        return (opinion, specific, priority(claim), claim.text.lower())

    ranked = sorted(claims, key=rank)
    chosen: list[ExtractedClaim] = []
    seen: set[str] = set()
    for claim in ranked:
        key = " ".join(claim.text.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        if not claim.span:
            claim.span = claim.text
        chosen.append(claim)
        if len(chosen) >= limit:
            break
    return chosen


class HeuristicExtractor:
    def extract(self, text: str | None, claims: list[str] | None) -> tuple[list[ExtractedClaim], list[str]]:
        return heuristic_extract(text, claims), []
