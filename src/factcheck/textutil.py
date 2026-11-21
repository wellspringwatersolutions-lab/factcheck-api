"""Small text helpers shared by extraction, retrieval, and judging."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from urllib.parse import urlparse

STOP = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "without",
    "from", "by", "at", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "these", "those", "their", "his", "her", "its", "into",
    "over", "after", "before", "about", "not", "no", "than", "then", "also",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")
_TOKEN = re.compile(r"[a-z0-9]+")
_NEGATION = re.compile(
    r"\b(not|no|never|n't|without|neither|nor|fails?|failed|unable|refute[sd]?|"
    r"false|incorrect|untrue|despite|n't)\b",
    re.I,
)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_DATE = re.compile(
    r"\b(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}|"
    r"\d{4}-\d{2}-\d{2}|\b(?:1[0-9]{3}|20[0-9]{2})\b)",
    re.I,
)
_QUOTE = re.compile(r"[\"“](.{8,180})[\"”]")
_URL = re.compile(r"https?://[^\s)>\"]+")
_CITE = re.compile(
    r"\b(?:according to|as reported by|as quoted by|cited by|in a study by)\s+([^,.]{2,80})",
    re.I,
)
_RECENT = re.compile(
    r"\b(current|currently|today|yesterday|this week|this month|this year|latest|"
    r"price|ceo|president|as of|breaking|right now)\b",
    re.I,
)


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if len(t) > 2 and t not in STOP}


def overlap(query: str, document: str) -> float:
    """Share of the query's content words that also appear in the document."""
    left = tokens(query)
    if not left:
        return 0.0
    right = tokens(document)
    if not right:
        return 0.0
    return len(left & right) / len(left)


def jaccard(a: str, b: str) -> float:
    left, right = tokens(a), tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def split_sentences(text: str) -> list[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT.split(cleaned)
    return [part.strip() for part in parts if part.strip()]


def has_negation(text: str) -> bool:
    return bool(_NEGATION.search(text or ""))


def has_number(text: str) -> bool:
    return bool(_NUMBER.search(text or ""))


def has_date(text: str) -> bool:
    return bool(_DATE.search(text or ""))


def quotes(text: str) -> list[str]:
    return [match.group(1).strip() for match in _QUOTE.finditer(text or "")]


def urls(text: str) -> list[str]:
    return _URL.findall(text or "")


def citation_name(text: str) -> str | None:
    match = _CITE.search(text or "")
    if not match:
        return None
    return match.group(1).strip()


def is_time_sensitive(text: str) -> bool:
    return bool(_RECENT.search(text or ""))


def is_fragile(text: str) -> bool:
    """Details where language models are weakest: numbers, dates, quotes, citations, recency."""
    blob = text or ""
    if has_number(blob) or has_date(blob) or quotes(blob) or urls(blob):
        return True
    if citation_name(blob):
        return True
    if is_time_sensitive(blob):
        return True
    if re.search(r"\b(exactly|precisely|doi:|isbn|page \d+|quoted|verbatim)\b", blob, re.I):
        return True
    return False


def hostname(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    return host.lower().removeprefix("www.")


def parse_json_object(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    iso = re.search(r"(20\d{2}|19\d{2})-(\d{2})-(\d{2})", text)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None
    month = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})",
        text,
        re.I,
    )
    if month:
        try:
            return date(int(month.group(3)), _MONTHS[month.group(1)[:3].lower()], int(month.group(2)))
        except ValueError:
            return None
    year = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if year and len(text) <= 12:
        return date(int(year.group(1)), 1, 1)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.date()
    except ValueError:
        return None


def iso_or_none(value: str | None) -> str | None:
    parsed = parse_date(value)
    if parsed:
        return parsed.isoformat()
    return None
