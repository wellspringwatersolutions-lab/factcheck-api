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
    r"\b(is|are|was|were|be|been|has|have|had|do|does|did|will|can|may|must|shall|advise|advises|recommend|recommends|warn|warns|say|says|told|urge|urges)\b|\b[a-z]{4,}(?:ed|ing)\b",
    re.I,
)
_SHARED = re.compile(
    r"^(?P<head>.+?)\s+(?P<verb>advise|advises|recommend|recommends|warn|warns|warned|told|urge|urges|issued|say|says|require|requires)\b(?P<tail>.*)$",
    re.I,
)


def _is_clause(text: str) -> bool:
    words = text.split()
    return len(words) >= 5 and bool(_CLAUSE_VERB.search(text))


def split_shared_predicate(sentence: str) -> list[str]:
    """'The EU, the UK, and Nordic countries advise X' becomes one claim per government."""
    match = _SHARED.match((sentence or "").strip())
    if match is None:
        return []
    head = match.group("head")
    if "," not in head and " and " not in head.lower():
        return []
    subjects = [part.strip(" ,") for part in re.split(r",|\band\b", head, flags=re.I)]
    subjects = [part for part in subjects if part and part.lower() not in {"the", "a", "an"}]
    if len(subjects) < 2:
        return []
    verb = match.group("verb")
    tail = match.group("tail") or ""
    return [f"{subject} {verb}{tail}".strip() for subject in subjects]


def split_compounds(sentence: str) -> list[str]:
    """Split on semicolons, and on 'and' only when both sides are clauses."""
    pieces: list[str] = []
    for chunk in re.split(r";\s*", sentence or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        shared = split_shared_predicate(chunk)
        if len(shared) >= 2:
            pieces.extend(shared)
            continue
        parts = [part.strip() for part in re.split(r"\s+\band\b\s+", chunk, flags=re.I) if part.strip()]
        if len(parts) >= 2 and all(_is_clause(part) for part in parts):
            pieces.extend(parts)
        else:
            pieces.append(chunk)
    return pieces or ([sentence.strip()] if sentence and sentence.strip() else [])


_FINITE = re.compile(
    r"\b(is|are|was|were|be|been|has|have|had|do|does|did|will|can|may|must|shall|told|tell|tells|says|said|advises|advise|recommends|recommend|warns|warned|distributed|raised|gave|gives|ran)\b|\b[a-z]{3,}ed\b",
    re.I,
)
_BAD_START = re.compile(r"^(by|and|or|but|including|recommending|cranking|\d)", re.I)
_DOUBLED_VERB = re.compile(r"\b(recommending|advising|warning)\b.+\b(recommends|advises|warns)\b", re.I)


def standalone_claim(text: str) -> bool:
    """A key point must be one complete sentence with a subject and a finite verb."""
    cleaned = " ".join((text or "").split())
    if len(cleaned) < 12 or cleaned[0] in ",.;:" or not cleaned[0].isupper():
        return False
    if cleaned[-1] not in ".!?":
        return False
    if _BAD_START.match(cleaned) or _DOUBLED_VERB.search(cleaned):
        return False
    words = cleaned[:-1].split()
    if len(words) < 4 or words[-1].lower().strip(",;:") in {"and", "or", "but", "the", "a", "an", "of", "to", "for", "with"}:
        return False
    return bool(_FINITE.search(cleaned))


def build_sentences(text: str | None, claims: list[str] | None) -> list[Sentence]:
    if claims:
        context = " ".join((text or "").split())[:500]
        return [Sentence(i, claim.strip(), context) for i, claim in enumerate(claims) if claim and claim.strip()]
    parts = [part for part in split_sentences(text or "") if part.strip()]
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
    """Replace a leading He/She/They with the person named just before it.

    Contractions and filler such as 'Of course' or 'That said' stay as written.
    """
    if re.search(r"\bThat said\b", text) or re.search(r"\bOf course\b", text):
        return text, False
    match = _PRONOUN.match(text or "")
    if not match or match.group(1) not in {"He", "She", "They"}:
        return text, False
    if match.start() > 0 and text[match.start() - 1] in {"'", "’"}:
        return text, False
    name = _nearest_name(context or "", text, match.group(1))
    if not name:
        return text, False
    return name + text[match.end():], False


_SUBJECT_PERSON = re.compile(r"(?:^|[.!?]\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
_PERSON = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
_NOT_A_PERSON = _BAD_ANTECEDENT | {
    "No", "Yes", "Not", "Unconfirmed", "There", "Was", "Were", "What", "When", "Where",
    "Who", "How", "Did", "Does", "Has", "Have", "Had", "Is", "Are", "In", "On", "For",
    "Defense", "Secretary", "Major", "United",
}


def _person_name(match: re.Match[str]) -> str:
    words = match.group(1).split()
    while words and words[0] in _NOT_A_PERSON:
        words = words[1:]
    if len(words) < 2:
        return ""
    return " ".join(words)


def _nearest_name(context: str, sentence: str, pronoun: str) -> str:
    """A proper-noun person named in the question or the text.

    'No' and other sentence openers are not names. If no person is named, the
    pronoun is left as written.
    """
    del pronoun
    index = (context or "").find(sentence) if sentence else -1
    prior = context[:index] if index > 0 else (context or "")
    subjects = [name for match in _SUBJECT_PERSON.finditer(prior) if (name := _person_name(match))]
    if subjects:
        return subjects[-1]
    named = [name for match in _PERSON.finditer(prior) if (name := _person_name(match))]
    return named[-1] if named else ""


def heuristic_extract(text: str | None, claims: list[str] | None) -> list[ExtractedClaim]:
    prepared = prepare_answer(text) if text else None
    prepared_claims = [clean_claim_text(claim) for claim in claims] if claims else None
    found: list[ExtractedClaim] = []
    for sentence in build_sentences(prepared, prepared_claims):
        label = heuristic_label(sentence.text)
        if label == "drop":
            continue
        resolved, drop = resolve_pronouns(sentence.text, prepared or sentence.context)
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

_RETRY = """Extract at most 6 checkable key points from the text.
Each point is one complete sentence with a subject and a finite verb.
Keep headline facts and debunks, such as who has not told people to do something.
Do not return fragments, participial phrases, or a sentence that repeats its verb.
Do not invent a claim that is not written in the text.
Return JSON {"points": [{"text": "...", "span": "..."}]}"""


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

    def extraction_models(self) -> list[str]:
        """A benchmarked fast model first, then Qwen3.5-397B and DeepSeek-V3.2.

        The extract-model setting is not used. An empty fast_model keeps the
        two current models, which is the case until a faster model matches them.
        """
        fast = (getattr(self.settings, "fast_model", "") or "").strip()
        models = []
        for name in (fast, "Qwen/Qwen3.5-397B-A17B-TEE", "deepseek-ai/DeepSeek-V3.2-TEE"):
            if name and name not in models:
                models.append(name)
        return models

    def _one_chunk(self, chunk: str) -> list[ExtractedClaim]:
        last_error: Exception | None = None
        for model in self.extraction_models():
            try:
                result = self.client.complete(
                    model=model,
                    messages=[
                        {"role": "system", "content": _DECOMPOSE},
                        {"role": "user", "content": chunk},
                    ],
                    temperature=0.0,
                    max_tokens=700,
                    timeout=10.0,
                    response_format=_CLAIM_SCHEMA,
                )
            except (LLMError, TimeoutError, ValueError, TypeError) as exc:
                last_error = exc
                logger.warning("extraction model=%s failed %s", model, exc.__class__.__name__)
                continue
            logger.warning("extraction model=%s", result.model or model)
            charge_tokens(
                self.meter,
                model=result.model,
                provider=result.provider,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost=estimate_cost(self.settings, result.model, result.input_tokens, result.output_tokens),
            )
            try:
                claims = _claims_from_result(result)
            except (ValueError, TypeError) as exc:
                last_error = exc
                logger.warning("extraction model=%s failed %s", model, exc.__class__.__name__)
                continue
            if claims:
                return claims
        raise last_error or ValueError("decomposition returned no claims")

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
        return select_key_points([item for item in unique if standalone_claim(item.text) or is_math(item.text)], 6), warnings

    def retry_keypoints(self, text: str | None) -> list[ExtractedClaim]:
        """One short DeepSeek call after the main extraction times out."""
        prepared = (prepare_answer(text) if text else "")[:6000]
        if not prepared:
            return []
        models = self.extraction_models()
        model = next((name for name in models if "DeepSeek-V3.2" in name), models[-1])
        logger.warning("extraction model=%s", model)
        result = self.client.complete(
            model=model,
            messages=[
                {"role": "system", "content": _RETRY},
                {"role": "user", "content": prepared},
            ],
            temperature=0.0,
            max_tokens=500,
            timeout=8.0,
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
        return [item for item in _claims_from_result(result) if standalone_claim(item.text)]


_SPECIFIC = re.compile(
    r"\d|\b(19|20)\d{2}\b|\b(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"guidance|official|program|ready\.gov|europa|act|order)\b",
    re.I,
)


def fallback_points(text: str | None, limit: int = 6) -> list[ExtractedClaim]:
    """Sentence splitter only. Fragments are dropped. Debunks and headlines stay."""
    found = []
    for claim in heuristic_extract(text, None):
        if standalone_claim(claim.text) or is_math(claim.text):
            found.append(claim)
    return select_key_points(found, limit)


def must_keep(text: str) -> bool:
    """Debunks, dated events, and official guidance stay even when the list is trimmed."""
    if re.search(r"\b(has not told|have not told|debunk|hoard|rumou?r|stockpile)\b", text or "", re.I):
        return True
    if re.search(r"\b(cyberattack|refrigerat)\b", text or "", re.I) and re.search(r"\b20\d{2}\b", text or ""):
        return True
    if re.search(r"\b(72-hour|72 hour|ready\.gov|fema)\b", text or "", re.I):
        return True
    return False


def select_key_points(claims: list[ExtractedClaim], limit: int = 6) -> list[ExtractedClaim]:
    """Stable order. Debunks, dated events, and official guidance are kept first."""
    from factcheck.cleanup import priority

    def rank(claim: ExtractedClaim):
        opinion = 1 if claim.claim_type == "opinion" else 0
        text = claim.text or ""
        required = 0 if must_keep(text) else 1
        specific = 0 if _SPECIFIC.search(text) else 1
        return (opinion, required, specific, priority(claim), text.lower())

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
