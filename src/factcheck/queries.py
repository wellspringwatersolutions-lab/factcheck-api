"""Short search queries and the news date window for one key point.

The model writes keyword queries. A deterministic fallback is used when it does not.
News is limited to a month or year named in the claim. It is never limited to the past week.
"""

from __future__ import annotations

import calendar
import logging
import re
from datetime import date, timedelta

from factcheck.llm import LLMError, estimate_cost, loads_object
from factcheck.request_ctx import charge_tokens

logger = logging.getLogger("factcheck")

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_STOP = {
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "and", "or", "but", "if",
    "as", "by", "from", "with", "into", "over", "under", "about", "after", "before",
    "during", "between", "through", "without", "within", "across", "per", "via",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "has", "have", "had",
    "this", "that", "these", "those", "it", "its", "their", "his", "her", "they", "them",
    "who", "what", "when", "where", "which", "how", "why",
    "not", "no", "nor", "than", "then", "also", "just", "more", "most", "very",
    "can", "may", "will", "would", "could", "should",
    "said", "says", "gave", "gives", "given", "reported", "reports", "according",
    "new", "other", "such", "there", "here", "out", "up",
}
_UNITS = {"day", "days", "hour", "hours", "percent", "year", "years", "month", "months", "people", "bases"}
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
_YEAR = re.compile(r"(?:19|20)\d{2}")
_QUERY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "search_queries",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
            "required": ["queries"],
        },
    },
}
_INSTRUCTION = """Write 2 or 3 web search queries for the claim.
Each query is 4 to 8 keywords, not a sentence, and does not start with In, The, or A.
Keep names, places, and the month and year when the claim states them.
Example: France supermarket refrigeration cyberattack October 2026
Return JSON {"queries": ["...", "..."]}"""


def _month_in_token(token: str) -> int | None:
    """'October' and 'mid-October' both name October. A bare year does not."""
    for piece in re.split(r"[-/]", token.lower()):
        value = _MONTHS.get(piece)
        if value is not None:
            return value
    return None


def news_window(text: str) -> dict | None:
    """Month named: that month, plus or minus about two weeks. Year only: that year.

    'mid-October 2026' is October, not the whole year. No date means no window.
    """
    tokens = _TOKEN.findall(text or "")
    month = None
    year = None
    for index, token in enumerate(tokens):
        value = _month_in_token(token)
        if value is None:
            continue
        month = value
        for neighbor in tokens[max(0, index - 3) : index + 4]:
            if _YEAR.fullmatch(neighbor):
                year = int(neighbor)
                break
        if year is not None:
            break
    if year is None:
        for token in tokens:
            if _YEAR.fullmatch(token):
                year = int(token)
                break
    if year is None:
        return None
    if month is None:
        return {"start_date": f"{year}-01-01", "end_date": f"{year}-12-31"}
    last = calendar.monthrange(year, month)[1]
    start = date(year, month, 1) - timedelta(days=14)
    end = date(year, month, last) + timedelta(days=14)
    return {"start_date": start.isoformat(), "end_date": end.isoformat()}


def _content_tokens(text: str) -> list[str]:
    tokens = _TOKEN.findall(text or "")
    kept: list[str] = []
    for index, token in enumerate(tokens):
        low = token.lower().strip("'’")
        if low in _STOP or low in _MONTHS:
            continue
        if re.fullmatch(r"\d{1,2}", token):
            nxt = tokens[index + 1].lower() if index + 1 < len(tokens) else ""
            if nxt not in _UNITS:
                continue
        if len(token) < 3 and not token.isupper() and not _YEAR.fullmatch(token):
            continue
        kept.append(token)
    return kept


def _date_words(text: str) -> list[str]:
    tokens = _TOKEN.findall(text or "")
    words: list[str] = []
    for index, token in enumerate(tokens):
        if token.lower() not in _MONTHS:
            continue
        words.append(token[:1].upper() + token[1:].lower() if token[:1].isalpha() else token)
        for neighbor in tokens[index + 1 : index + 3]:
            if _YEAR.fullmatch(neighbor):
                words.append(neighbor)
                return words
    for token in tokens:
        if _YEAR.fullmatch(token) and token not in words:
            words.append(token)
            break
    return words


def _join(words: list[str], limit: int = 8) -> str:
    seen: list[str] = []
    for word in words:
        if word.lower() in {item.lower() for item in seen}:
            continue
        seen.append(word)
        if len(seen) >= limit:
            break
    return " ".join(seen)


def heuristic_queries(text: str) -> list[str]:
    """Keyword queries in claim order. Drops 'In' and stray small numbers."""
    content = _content_tokens(text)
    dated = _date_words(text)
    primary = _join([*content, *dated])
    names = [
        token
        for token in _TOKEN.findall(text or "")
        if token[:1].isupper() and token.lower() not in _STOP and token.lower() not in _MONTHS and not token.isupper()
    ]
    # Keep short all-caps names such as EU or WHO, but not a single letter.
    caps = [token for token in _TOKEN.findall(text or "") if token.isupper() and len(token) >= 2]
    secondary = _join([*caps, *names, *dated])
    queries = []
    for query in (primary, secondary):
        if query and query not in queries and len(query.split()) >= 2:
            queries.append(query)
    if not queries and primary:
        queries.append(primary)
    return queries[:2]


def _accept(raw: list) -> list[str]:
    kept: list[str] = []
    for item in raw:
        query = " ".join(str(item or "").replace("?", " ").replace(".", " ").split())
        words = query.split()
        if not 2 <= len(words) <= 10:
            continue
        if words[0].lower() in {"in", "the", "a", "an"}:
            words = words[1:]
        query = " ".join(words)
        if len(query.split()) < 2 or query in kept:
            continue
        kept.append(query)
        if len(kept) >= 2:
            break
    return kept


class QueryWriter:
    """Qwen writes the queries. DeepSeek is the fallback. Both are Chutes models."""

    def __init__(self, client, settings, meter):
        self.client = client
        self.settings = settings
        self.meter = meter
        self.primary = settings.judge_b_model
        self.fallback = settings.judge_a_model

    def write(self, text: str) -> list[str]:
        """At most two keyword queries. A timeout becomes one deterministic query, not another model call."""
        try:
            payload = self._ask(self.primary, text)
        except TimeoutError as exc:
            logger.warning("query writer %s timed out: %s", self.primary, exc)
            return heuristic_queries(text)[:1]
        except (LLMError, ValueError, TypeError, KeyError) as exc:
            logger.warning("query writer %s failed: %s", self.primary, exc)
            payload = None
        accepted = _accept((payload or {}).get("queries") or []) if isinstance(payload, dict) else []
        if accepted:
            return accepted[:2]
        return heuristic_queries(text)[:1]

    def _ask(self, model: str, text: str) -> dict:
        result = self.client.complete(
            model=model,
            messages=[
                {"role": "system", "content": _INSTRUCTION},
                {"role": "user", "content": (text or "")[:1500]},
            ],
            temperature=0.0,
            max_tokens=180,
            timeout=4.0,
            response_format=_QUERY_SCHEMA,
        )
        charge_tokens(
            self.meter,
                model=result.model,
                provider=result.provider,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost=estimate_cost(self.settings, result.model, result.input_tokens, result.output_tokens),
            )
        payload = loads_object(result)
        if not isinstance(payload, dict):
            raise ValueError("query writer returned no object")
        return payload
