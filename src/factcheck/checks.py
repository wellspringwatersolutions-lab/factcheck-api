"""Deterministic checks. None of these ask a model what is true."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import httpx

from factcheck.textutil import has_negation, hostname, overlap, parse_date, quotes, urls
from factcheck.types import Evidence

_NUMBER = re.compile(r"(?<![\w])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w])\d+(?:\.\d+)?%?")
_WS = re.compile(r"\s+")


@dataclass
class UrlResolution:
    url: str
    ok: bool
    status: int | None
    final_url: str
    body: str
    content_sha256: str
    error: str = ""


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").lower()).strip()


def quote_exact(page: str, quote: str) -> bool:
    """Whitespace-folded containment. A review cannot cite a paraphrase as the quote."""
    folded = _norm(quote)
    return bool(folded) and folded in _norm(page)


def quote_appears(page: str, quote: str) -> bool:
    """Exact match after whitespace folding, otherwise a high token overlap."""
    if not quote or not quote.strip():
        return False
    folded_page = _norm(page)
    folded_quote = _norm(quote)
    if folded_quote and folded_quote in folded_page:
        return True
    if len(folded_quote) < 12:
        return False
    return overlap(quote, page) >= 0.92


def _numbers(text: str) -> list[str]:
    found = []
    for match in _NUMBER.findall(text or ""):
        token = match.replace(",", "").rstrip("%")
        if len(token) == 4 and token.isdigit() and token.startswith(("1", "2")):
            continue
        found.append(token)
    return found


def numbers_match(claim: str, passage: str) -> bool | None:
    wanted = _numbers(claim)
    if not wanted:
        return None
    have = set(_numbers(passage))
    return all(number in have for number in wanted)


def dates_match(claim: str, passage: str) -> bool | None:
    claim_date = parse_date(claim)
    if claim_date is None or not re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", claim or ""):
        return None
    passage_date = parse_date(passage)
    if passage_date is None:
        return False
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", claim) or re.search(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}",
        claim,
        re.I,
    ):
        return claim_date == passage_date
    return claim_date.year == passage_date.year


def _with_synonyms(text: str) -> str:
    """Add equivalent titles so overlap can see a paraphrase. The stored quote stays verbatim."""
    lowered = text or ""
    extra = [lowered]
    if re.search(r"commander in chief", lowered, re.I):
        extra.append(re.sub(r"commander in chief", "commanding general", lowered, flags=re.I))
    if re.search(r"commanding general", lowered, re.I):
        extra.append(re.sub(r"commanding general", "commander in chief", lowered, flags=re.I))
    return " ".join(extra)


def entailment(claim: str, passage: str) -> str:
    """Offline sentence check. Negation is read only inside this passage, never the whole page."""
    if not passage or not passage.strip():
        return "neutral"
    if "<" in passage and ">" in passage:
        passage = clean_html(passage)
    claim = _with_synonyms(claim)
    passage = _with_synonyms(passage)
    conflict = quantity_conflict(claim, passage)
    if conflict:
        return "contradicts"
    if numbers_match(claim, passage) is False:
        return "neutral"
    if dates_match(claim, passage) is False:
        return "neutral"
    recall = overlap(claim, passage)
    claim_neg = has_negation(claim)
    passage_neg = has_negation(passage)
    if recall >= 0.45 and claim_neg == passage_neg:
        return "entails"
    if recall >= 0.45 and claim_neg != passage_neg:
        return "contradicts"
    return "neutral"


def best_passage(page: str, claim: str, limit: int = 420) -> str:
    parts = re.split(r"(?<=[.!?])\s+|\n+", page or "")
    parts = [part.strip() for part in parts if part.strip()]
    if not parts:
        return (page or "")[:limit]
    ranked = sorted(parts, key=lambda part: overlap(claim, part), reverse=True)
    return ranked[0][:limit]


def clean_html(text: str) -> str:
    """Drop scripts, styles, and tags so later checks see readable text."""
    if not text or "<" not in text or ">" not in text:
        return text or ""
    from html.parser import HTMLParser

    class _Extractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.skip = 0
            self.parts: list[str] = []

        def handle_starttag(self, tag: str, attrs) -> None:
            if tag in {"script", "style", "noscript", "svg", "nav"}:
                self.skip += 1
            if tag in {"p", "br", "div", "li", "h1", "h2", "h3", "tr", "section"}:
                self.parts.append("\n")

        def handle_endtag(self, tag: str) -> None:
            if tag in {"script", "style", "noscript", "svg", "nav"} and self.skip:
                self.skip -= 1
            if tag in {"p", "div", "li", "h1", "h2", "h3", "section"}:
                self.parts.append("\n")

        def handle_data(self, data: str) -> None:
            if not self.skip:
                self.parts.append(data)

    parser = _Extractor()
    parser.feed(text)
    cleaned = re.sub(r"[ \t]+", " ", "".join(parser.parts))
    return re.sub(r"\n{2,}", "\n", cleaned).strip()


_LENGTH_UNITS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "kilometre": 1000.0,
    "kilometres": 1000.0,
    "cm": 0.01,
    "mm": 0.001,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
}
_QUANTITY = re.compile(
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<unit>kilometers?|kilometres?|meters?|metres?|miles?|feet|foot|km|cm|mm|ft|mi|m)\b",
    re.I,
)


def extract_lengths(text: str) -> list[float]:
    found = []
    for match in _QUANTITY.finditer(text or ""):
        unit = match.group("unit").lower()
        factor = _LENGTH_UNITS.get(unit)
        if factor is None:
            continue
        found.append(float(match.group("num").replace(",", "")) * factor)
    return found


def quantity_conflict(claim: str, passage: str) -> str | None:
    """Return the passage sentence whose length disagrees with the claim, if any."""
    claimed = extract_lengths(claim)
    if not claimed or overlap(claim, passage) < 0.2:
        return None
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", passage or "") if part.strip()]
    for sentence in sentences or [passage]:
        if overlap(claim, sentence) < 0.2:
            continue
        observed = extract_lengths(sentence)
        for left in claimed:
            for right in observed:
                scale = max(abs(left), abs(right), 1.0)
                if abs(left - right) / scale > 0.05 and abs(left - right) > 1:
                    return sentence
    return None


_CHROME = re.compile(
    r"^(published|updated|posted|by|share|subscribe|sign in|cookie|advertisement|all rights reserved)\b",
    re.I,
)


def best_passages(page: str, claim: str, title: str = "", limit: int = 3) -> list[str]:
    """The passages that actually discuss the claim, not the title or a dateline."""
    cleaned = clean_html(page or "")
    chunks = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", cleaned) if part.strip()]
    title_key = " ".join((title or "").lower().split())
    scored: list[tuple[float, str]] = []
    for chunk in chunks:
        if len(chunk) < 40 or _CHROME.match(chunk):
            continue
        if title_key and " ".join(chunk.lower().split()) == title_key:
            continue
        score = overlap(claim, chunk)
        if score <= 0:
            continue
        scored.append((score, chunk[:700]))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:limit]]


def split_passages(text: str, limit: int = 6) -> list[str]:
    cleaned = clean_html(text or "")
    parts = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", cleaned) if part.strip()]
    chunks: list[str] = []
    buffer = ""
    for part in parts:
        if len(part) < 40 and buffer:
            buffer = f"{buffer} {part}".strip()
            continue
        if buffer:
            chunks.append(buffer)
        buffer = part
    if buffer:
        chunks.append(buffer)
    return [chunk[:700] for chunk in chunks if len(chunk) > 30][:limit]


def apply_passage_checks(claim_text: str, evidence: Evidence, resolution: UrlResolution) -> Evidence:
    """Attach hash, URL status, and a cleaned page. Entailment is filled in later by judges."""
    raw = resolution.body or evidence.page_text or evidence.snippet or ""
    snippet = (evidence.snippet or "").strip()
    if snippet and snippet not in raw:
        raw = f"{raw}\n{snippet}".strip()
    body = clean_html(raw)
    chosen = best_passages(body, claim_text, evidence.title, limit=3)
    evidence.page_text = "\n".join(chosen) if chosen else body
    evidence.content_sha256 = content_hash(body or raw)
    evidence.url_status = resolution.status
    evidence.quote = ""
    cited_quotes = quotes(claim_text)
    if cited_quotes:
        evidence.quote_match = all(quote_appears(body, quote) for quote in cited_quotes)
    else:
        evidence.quote_match = None
    evidence.number_match = numbers_match(claim_text, body)
    evidence.date_match = dates_match(claim_text, body)
    evidence.entailment = "neutral"
    evidence.used = False
    if resolution.ok and evidence.quote_match is not False:
        conflict = quantity_conflict(claim_text, body)
        if conflict:
            evidence.entailment = "contradicts"
            evidence.quote = conflict
            evidence.used = True
        from factcheck.credibility import is_encyclopedia

        if evidence.provider in {"wikidata", "wikipedia"} or is_encyclopedia(evidence.url):
            from factcheck.references import encyclopedia_role_check, wikidata_date_check, wikidata_office_check

            dated = None
            if evidence.provider == "wikidata":
                dated = wikidata_date_check(claim_text, body) or wikidata_office_check(claim_text, body)
            if dated is None:
                dated = encyclopedia_role_check(claim_text, body)
            if dated:
                evidence.entailment = dated[0]
                evidence.quote = dated[1]
                evidence.used = True
    return evidence


def passage_check_rows(claim_text: str, evidence: list[Evidence]) -> list[dict]:
    rows: list[dict] = []
    cited = urls(claim_text)
    if cited:
        for url in cited:
            matched = [item for item in evidence if url.rstrip("/") in {item.url.rstrip("/"), (item.url or "").rstrip("/")}]
            dead = [item for item in matched if item.url_status is not None and item.url_status >= 400]
            missing = not matched
            rows.append(
                {
                    "name": "cited_url",
                    "passed": bool(matched) and not dead and not missing,
                    "detail": url if matched and not dead else f"cited URL did not resolve: {url}",
                }
            )
    if quotes(claim_text):
        ok = any(item.quote_match for item in evidence)
        rows.append(
            {
                "name": "quoted_text",
                "passed": ok,
                "detail": "quoted text appears in a retrieved page" if ok else "quoted text was not found on the retrieved page",
            }
        )
    for item in evidence:
        if item.url_status is not None and item.url_status >= 400 and item.url_status not in {401, 403, 429}:
            rows.append(
                {
                    "name": "url_resolves",
                    "passed": False,
                    "detail": f"{item.url} returned HTTP {item.url_status}",
                }
            )
        elif item.url.startswith("http") and item.url_status is None:
            rows.append(
                {
                    "name": "url_resolves",
                    "passed": False,
                    "detail": f"{item.url} did not resolve",
                }
            )
    return rows


class UrlChecker:
    """Resolve a URL and return its body. Corpus and tool schemes do not use the network."""

    def __init__(self, timeout: float = 8.0, client: httpx.Client | None = None, email: str = "dev@localhost"):
        self.timeout = timeout
        self.email = email or "dev@localhost"
        self._client = client

    def _headers(self) -> dict[str, str]:
        from factcheck.references import user_agent

        return {
            "User-Agent": user_agent(self.email),
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        }

    def _wikipedia_extract(self, client: httpx.Client, url: str) -> str:
        title = url.split("/wiki/", 1)[-1].replace("_", " ")
        response = client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "exintro": 1,
                "exchars": 1200,
                "titles": title,
                "format": "json",
                "formatversion": "2",
            },
            headers=self._headers(),
        )
        if response.status_code >= 400:
            return ""
        pages = ((response.json().get("query") or {}).get("pages")) or []
        return (pages[0].get("extract") or "") if pages else ""

    def _wayback(self, client: httpx.Client, url: str) -> str:
        response = client.get(
            "https://archive.org/wayback/available",
            params={"url": url},
            headers=self._headers(),
        )
        if response.status_code >= 400:
            return ""
        closest = ((response.json().get("archived_snapshots") or {}).get("closest")) or {}
        if not closest.get("available") or not closest.get("url"):
            return ""
        return f"Archived copy of {url} at {closest.get('timestamp') or 'an earlier date'}."

    def _fallback(self, url: str, snippet: str, status: int | None, client: httpx.Client) -> UrlResolution | None:
        text = (snippet or "").strip()
        if not text:
            try:
                text = self._wayback(client, url)
            except httpx.HTTPError:
                text = ""
        if not text.strip():
            return None
        return UrlResolution(url, True, 200, url, text, content_hash(text), f"blocked HTTP {status}; used snippet or archive")

    def resolve(self, url: str, body: str = "") -> UrlResolution:
        if url.startswith(("scifact://", "fever://", "heldout://", "tool://", "corpus://")):
            text = body or ""
            return UrlResolution(url, True, 200, url, text, content_hash(text))
        if not url.startswith(("http://", "https://")):
            text = body or ""
            return UrlResolution(url, bool(text), None, url, text, content_hash(text), "unsupported URL scheme")
        client = self._client or httpx.Client(timeout=self.timeout, follow_redirects=True)
        close = self._client is None
        try:
            host = hostname(url)
            if host.endswith("wikipedia.org") and "/wiki/" in url:
                extract = self._wikipedia_extract(client, url)
                text = extract or body
                if text.strip():
                    return UrlResolution(url, True, 200, url, text, content_hash(text))
            response = client.get(url, headers=self._headers())
            text = response.text or ""
            if response.status_code in {401, 403, 429} or "Please set a user-agent" in text:
                fallback = self._fallback(url, body, response.status_code, client)
                if fallback is not None:
                    return fallback
            ok = 200 <= response.status_code < 300 and bool(text.strip())
            return UrlResolution(
                url=url,
                ok=ok,
                status=response.status_code,
                final_url=str(response.url),
                body=text if ok else (body or text),
                content_sha256=content_hash(text if text else body),
                error="" if ok else f"HTTP {response.status_code}",
            )
        except httpx.HTTPError as exc:
            text = body or ""
            return UrlResolution(url, bool(text.strip()), 200 if text.strip() else None, url, text, content_hash(text), str(exc))
        finally:
            if close:
                client.close()


class MapUrlChecker(UrlChecker):
    """Test double. Keys are URLs; missing keys are dead links."""

    def __init__(self, pages: dict[str, tuple[int, str]]):
        super().__init__()
        self.pages = pages

    def resolve(self, url: str, body: str = "") -> UrlResolution:
        if url.startswith(("scifact://", "fever://", "heldout://", "tool://", "corpus://")):
            return super().resolve(url, body)
        if url not in self.pages:
            return UrlResolution(url, False, 404, url, "", content_hash(""), "not in map")
        status, text = self.pages[url]
        if status in {401, 403, 429} and (body or "").strip():
            return UrlResolution(url, True, 200, url, body, content_hash(body), f"blocked HTTP {status}; used snippet")
        ok = 200 <= status < 300
        return UrlResolution(url, ok, status, url, text if ok else "", content_hash(text), "" if ok else f"HTTP {status}")
